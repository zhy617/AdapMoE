import numpy

import sys

import torch
from torch.nn import functional as F
# HQQ (Half-Quadratic Quantization) 是一个模型量化库
from hqq.core.quantize import BaseQuantizeConfig

from transformers import AutoConfig, AutoTokenizer

# 从 src 目录导入自定义的模块
# OffloadConfig: 数据类，用于存储模型卸载（offloading）相关的配置
# QuantConfig: 数据类，用于存储量化相关的配置
# build_model: 构建模型的函数
from src.build_model import OffloadConfig, QuantConfig, build_model
# get_cache_size: 计算专家模型缓存大小的函数
from src.dp import get_cache_size

# TextStreamer 用于流式输出生成的文本，实现打字机效果
from transformers import TextStreamer
import time
import argparse
import math

def main():
    # --- 1. 配置模型和设备 ---
    
    # 设置要使用的模型在 Hugging Face Hub 上的路径
    path = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"
    model_name = path
    quantized_model_name = path
    state_path = path # 模型权重文件的路径

    # 从预训练模型加载配置
    config = AutoConfig.from_pretrained(quantized_model_name)

    # 设置运行设备，这里使用第一个 CUDA 设备 (GPU)
    device = torch.device("cuda:0")

    # --- 2. 配置专家模型卸载 (Offloading) 和缓存策略 ---

    # 从命令行参数获取专家数量
    main_size = args.size
    # 根据主缓存大小和是否启用自适应门控，计算每层的缓存策略
    cache_strategy = get_cache_size(main_size,args.adapgate)
    print(cache_strategy)

    num_experts = config.num_local_experts

    # 创建卸载配置对象
    offload_config = OffloadConfig(
        main_size=main_size,  # 保留在主设备（GPU）上的专家数量
        cache_strategy=cache_strategy, # 每层的缓存策略
        offload_size=config.num_hidden_layers * num_experts, # 卸载到次级设备（CPU/RAM）的专家总数
        buffer_size=6,  # 用于预加载的缓冲区大小
    )

    # --- 3. 配置模型量化 ---

    # 配置注意力（Attention）层的量化参数
    attn_config = BaseQuantizeConfig(
        nbits=4,          # 4位量化
        group_size=64,    # 分组大小
        quant_zero=True,  # 量化零点
        quant_scale=True, # 量化缩放因子
    )
    # 为缩放因子设置一个不同的分组大小
    attn_config["scale_quant_params"]["group_size"] = 256

    # 配置前馈网络（FFN）层的量化参数
    ffn_config = BaseQuantizeConfig(
        nbits=2,          # 2位量化
        group_size=16,
        quant_zero=True,
        quant_scale=True,
    )
    # 将注意力层和FFN层的量化配置组合在一起
    quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

    # --- 4. 构建模型 ---

    # 调用自定义的 build_model 函数来创建模型
    # 这个函数会应用上面定义的量化和卸载配置
    model = build_model(
        device=device,
        quant_config=quant_config,
        offload_config=offload_config,
        state_path=state_path,
    )

    # --- 5. 配置自适应门控 (Adaptive Gating) ---

    # 如果命令行中指定了 --adapgate 参数
    if args.adapgate:
        # 这是一组预先计算好的权重，用于调整每个MoE层的门控阈值
        weight = [46.69189453125, 17.303466796875, 13.0157470703125, 7.640838623046875, 4.169464111328125, 2.2296905517578125, 1.2559890747070312, 0.8444786071777344, 0.6837844848632812, 0.5602836608886719, 0.5125999450683594, 0.4780292510986328, 0.44536590576171875, 0.4355907440185547, 0.38361549377441406, 0.30994415283203125, 0.23305416107177734, 0.1760721206665039, 0.13840198516845703, 0.1137852668762207, 0.10472536087036133, 0.09542703628540039, 0.08624792098999023, 0.07712841033935547, 0.06937980651855469, 0.06109476089477539, 0.0502467155456543, 0.042557716369628906, 0.03349781036376953, 0.025272369384765625, 0.020682811737060547, 0.02294778823852539]
        # 遍历模型的每一层，并根据权重设置门控阈值
        for idx, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.threshold = math.sqrt(0.005/weight[idx])

    # --- 6. 初始化 Tokenizer 和推理循环 ---

    # 加载与模型匹配的 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 初始化文本流式输出器，用于实时显示生成结果
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 初始化 past_key_values (用于缓存键值对，加速生成) 和 sequence
    past_key_values = None
    sequence = None

    # 初始化序列长度、总时间和总 token 数的计数器
    seq_len = 0
    total_time = 0
    total_tokens = 0
    
    # 进入无限循环，开始交互式对话
    while True:
        print("User: ", end="")
        user_input = input()
        print("\n")

        # 将用户输入格式化为聊天模板
        user_entry = dict(role="user", content=user_input)
        input_ids = tokenizer.apply_chat_template([user_entry], return_tensors="pt").to(device)

        # 管理 attention_mask 和 past_key_values 以进行连续对话
        if past_key_values is None:
            # 如果是第一轮对话，创建一个新的 attention_mask
            attention_mask = torch.ones_like(input_ids)
        else:
            # 如果是后续对话，更新 attention_mask 以包含历史上下文
            seq_len = input_ids.size(1) + past_key_values[0][0][0].size(1)
            attention_mask = torch.ones([1, seq_len - 1], dtype=torch.int, device=device)

        print("Mixtral: ", end="")
        start_time = time.time()
        # 调用 model.generate 开始生成文本
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values, # 传入上一轮的 K-V 缓存
            streamer=streamer,              # 使用流式输出
            do_sample=True,                 # 启用采样
            top_k=1,                        # Top-k 采样参数
            max_new_tokens=128,             # 设置最大生成 token 数量
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,   # 返回一个字典作为输出
            output_hidden_states=True,      # 输出隐藏状态
        )
        end_time = time.time()
        print("\n")

        # 更新 sequence 和 past_key_values 以备下一轮使用
        sequence = result["sequences"]
        past_key_values = result["past_key_values"]

        # 累加总时间和生成的 token 数
        total_time += end_time - start_time
        total_tokens += sequence.size(1)

        # 计算并打印每个 token 的平均生成时间
        avg_time_per_token = (end_time - start_time) / 128
        print(f"Average time per token: {avg_time_per_token} seconds")

# --- 7. 脚本入口和命令行参数解析 ---

if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加 --adapgate 参数，这是一个开关（布尔）参数
    parser.add_argument('--adapgate', action='store_true', help='Enable adaptive gating for MoE layers')
    # 添加 --size 参数，用于指定主缓存大小，类型为整数，默认值为64
    parser.add_argument('--size', type=int, default=64, help='Number of experts to keep on the main device (GPU)')
    # 解析命令行传入的参数
    args = parser.parse_args()
    # 调用主函数
    main()


