#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从train.parquet准备Reviewer训练数据（ToxiCN有毒语言检测）
- 划分train/val/test
- 转换为标准JSON格式
"""

import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# ToxiCN有毒语言检测规则
RULES = """
性别歧视：对特定性别群体的贬低、攻击、刻板印象或隐含偏见。
种族歧视：基于种族进行侮辱、偏见表达或刻板评价。
地域偏见：针对特定地区人群的侮辱、刻板印象或地域歧视言论。
LGBTQ歧视：对LGBTQ群体的侮辱、否定或隐含歧视。
其他仇恨：不属于上述特定群体但具有明显恶意的攻击性言论。
"""

def create_prompt(text: str, label: str) -> dict:
    """创建Reviewer训练样本（ToxiCN有毒语言检测）"""
    prompt = (
        '作为中文有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '有毒类别："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果无毒请回答"无毒"。\n'
        '注意：隐含歧视（如反讽、暗喻、阴阳怪气）也属于有毒内容；转述仇恨事件但本身不有毒的内容应判为无毒。\n'
        '请直接输出一个类别名。\n\n'
        f'规则: {RULES}\n'
        f'文本: {text}'
    )
    
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label}
        ]
    }

def main():
    print("="*80)
    print("准备Reviewer训练数据")
    print("="*80)
    
    # 读取数据
    print("\n读取train.parquet...")
    df = pd.read_parquet("train.parquet")
    print(f"总数据量: {len(df)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 统计类别分布
    print("\n类别分布:")
    label_counts = df['标签'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:12s}: {count:5d} ({count/len(df)*100:.1f}%)")
    
    # 创建输出目录
    output_dir = Path("reviewer_training_data")
    output_dir.mkdir(exist_ok=True)
    print(f"\n创建目录: {output_dir}")
    
    # 划分数据集
    # train: 80%, val: 10%, test: 10%
    print("\n划分数据集 (80% train, 10% val, 10% test)...")
    
    train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['标签'])
    train, val = train_test_split(train_val, test_size=0.111,  # 0.111 * 0.9 ≈ 0.1
                                  random_state=42, stratify=train_val['标签'])
    
    print(f"Train: {len(train)} 样本")
    print(f"Val:   {len(val)} 样本")
    print(f"Test:  {len(test)} 样本")
    
    # 转换格式并保存
    splits = {
        "train": train,
        "val": val,
        "test": test
    }
    
    for split_name, split_df in splits.items():
        print(f"\n处理 {split_name} 集...")
        
        # 转换为训练格式
        samples = []
        for _, row in split_df.iterrows():
            sample = create_prompt(row['文本'], row['标签'])
            samples.append(sample)
        
        # 保存为JSONL（适合训练）
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"  保存到: {jsonl_path}")
        
        # 也保存一份JSON（适合评测）
        json_data = []
        for _, row in split_df.iterrows():
            json_data.append({
                "文本": row['文本'],
                "标签": row['标签']
            })
        
        json_path = output_dir / f"{split_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"  保存到: {json_path}")
    
    print("\n" + "="*80)
    print("✓ 数据准备完成！")
    print("="*80)
    print(f"\n训练数据位置: {output_dir}")
    print(f"  - train.jsonl: Reviewer LoRA训练")
    print(f"  - val.jsonl:   验证集")
    print(f"  - test.json:   最终评测")

if __name__ == "__main__":
    main()
