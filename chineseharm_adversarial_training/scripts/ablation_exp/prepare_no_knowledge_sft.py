#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验: 无知识 SFT 数据准备
===============================

生成不含 RULES 知识的 Reviewer SFT 训练数据。
Prompt 中只保留任务指令和类别范围，不提供任何关键词映射规则。

用于对比:
  - 有知识 SFT (当前版本，含 ~2000字 RULES)
  - 无知识 SFT (本脚本生成，不含 RULES)

验证: RULES 知识是否是 SFT 效果好的关键因素

用法:
  python prepare_no_knowledge_sft.py \
    --split_dir /home/ma-user/work/test/split_data \
    --output_dir /home/ma-user/work/test/prepared_data/reviewer_sft_no_knowledge
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import Counter

import pandas as pd


SYSTEM_PROMPT = "You are a helpful assistant."


def create_prompt_no_knowledge(text: str) -> str:
    """
    无知识版 prompt: 只有任务指令 + 类别范围，不含 RULES
    """
    return (
        '作为有毒语言检测专家，请依据有毒语言检测标准，判断以下文本是否包含有毒内容。\n'
        '如果文本有毒，请指出有毒类别，类别范围限定为："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果文本无毒，请回答"无毒"。\n'
        '请直接输出一个类别：包括"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"或"无毒"。\n\n'
        f'文本: {text}'
    )


def build_reviewer_sft_data(df: pd.DataFrame, split_name: str, output_dir: Path):
    """构建无知识版 Reviewer SFT 数据"""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for _, row in df.iterrows():
        text = row.get('文本', '')
        label = row.get('标签', '')
        if not text or not label:
            continue

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": create_prompt_no_knowledge(text)},
                {"role": "assistant", "content": label}
            ],
            "category": label,
            "original_text": text
        }
        samples.append(sample)

    # 保存 JSONL (训练脚本读取格式)
    jsonl_path = output_dir / f"{split_name}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    # 保存 Parquet
    flat = [{
        "instruction": s["messages"][1]["content"],
        "output": s["messages"][2]["content"],
        "category": s["category"],
        "original_text": s["original_text"]
    } for s in samples]
    parquet_path = output_dir / f"{split_name}.parquet"
    pd.DataFrame(flat).to_parquet(parquet_path, index=False)

    cat_counts = Counter(s['category'] for s in samples)
    print(f"  [{split_name}]: {len(samples)} 条")
    for cat in sorted(cat_counts):
        print(f"    {cat:10s}: {cat_counts[cat]}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="生成无知识版 Reviewer SFT 数据")
    parser.add_argument("--split_dir", type=str,
                       default="/home/ma-user/work/test/split_data",
                       help="split_data 目录")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/prepared_data/reviewer_sft_no_knowledge",
                       help="输出目录")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("无知识版 Reviewer SFT 数据准备")
    print("=" * 60)
    print(f"  输入: {split_dir}")
    print(f"  输出: {output_dir}")
    print()

    # Prompt 对比展示
    demo_text = "示例文本"
    print("Prompt 对比:")
    print(f"  有知识版 prompt长度: ~{len(create_prompt_no_knowledge(demo_text)) + 2000} 字符 (含 ~2000字 RULES)")
    print(f"  无知识版 prompt长度: ~{len(create_prompt_no_knowledge(demo_text))} 字符 (无 RULES)")
    print()

    for split_name in ['train', 'val']:
        # 尝试 parquet 和 json 两种格式
        parquet_path = split_dir / f"{split_name}.parquet"
        json_path = split_dir / f"{split_name}.json"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                df = pd.DataFrame(json.load(f))
        else:
            print(f"⚠️  跳过 {split_name}: 文件不存在")
            continue

        build_reviewer_sft_data(df, split_name, output_dir)

    print()
    print("=" * 60)
    print("✓ 完成! 数据已保存到:", output_dir)
    print()
    print("后续步骤:")
    print("  1. 用此数据训练无知识版 LoRA:")
    print(f"     python train_reviewer_lora.py --data_path {output_dir}/train.jsonl \\")
    print(f"       --output_dir /home/ma-user/work/test/lora_models/reviewer_3B_no_knowledge")
    print("  2. 合并 LoRA 并评测")
    print("  3. 与有知识版对比准确率")
    print("=" * 60)


if __name__ == "__main__":
    main()
