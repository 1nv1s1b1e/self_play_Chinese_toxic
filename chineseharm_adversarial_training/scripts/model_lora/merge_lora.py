#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并LoRA到基础模型
用于vLLM推理前的准备
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    """
    合并LoRA权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA路径
        output_path: 输出路径
    """
    print("=" * 60)
    print("合并LoRA到基础模型")
    print("=" * 60)
    print(f"基础模型: {base_model_path}")
    print(f"LoRA: {lora_path}")
    print(f"输出: {output_path}")
    print()
    
    # 验证路径存在
    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
    if not os.path.isdir(lora_path):
        raise FileNotFoundError(f"LoRA路径不存在: {lora_path}")
    
    adapter_config = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        raise FileNotFoundError(
            f"LoRA目录缺少 adapter_config.json: {lora_path}\n"
            "请确认LoRA训练已正确完成。"
        )
    
    # 创建输出目录
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 加载基础模型（在CPU上，避免NPU显存占用）
    print("加载基础模型（CPU模式）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu",  # 使用CPU合并
        trust_remote_code=True
    )
    print("✓ 基础模型加载完成")
    
    # 加载LoRA
    print(f"\n加载LoRA: {lora_path}")
    try:
        model = PeftModel.from_pretrained(base_model, lora_path)
    except Exception as e:
        raise RuntimeError(f"LoRA加载失败: {e}\n请检查LoRA权重与基础模型是否匹配。")
    print("✓ LoRA加载完成")
    
    # 合并
    print("\n合并LoRA权重到基础模型...")
    merged_model = model.merge_and_unload()
    print("✓ 合并完成")
    
    # 保存
    print(f"\n保存合并后的模型到: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # 验证保存成功
    config_path = os.path.join(output_path, "config.json")
    if not os.path.isfile(config_path):
        raise RuntimeError(f"合并模型保存异常：{output_path} 中缺少 config.json")
    
    print("\n" + "=" * 60)
    print("✓ 合并完成！")
    print("=" * 60)
    print(f"\n合并后的模型已保存到: {output_path}")
    print("现在可以使用vLLM加载这个模型进行快速推理。")

def main():
    parser = argparse.ArgumentParser(description="合并LoRA到基础模型")
    parser.add_argument("--base_model", type=str,
                       default="/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--lora_path", type=str,
                       required=True,
                       help="LoRA路径")
    parser.add_argument("--output_path", type=str,
                       required=True,
                       help="输出路径")
    
    args = parser.parse_args()
    
    merge_lora(args.base_model, args.lora_path, args.output_path)

if __name__ == "__main__":
    main()
