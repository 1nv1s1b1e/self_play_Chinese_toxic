#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载外部中文有害文本数据集: ToxiCN + Safety-Prompts
用于跨域混合对抗训练

数据集:
  1. ToxiCN (ACL 2023): 12K条知乎/贴吧有毒评论, 4层级细粒度标注
     - HuggingFace: JunyuLu/ToxiCN
  2. Safety-Prompts (清华CoAI): 100K条安全场景prompt+回复
     - HuggingFace: thu-coai/Safety-Prompts

输出:
  {output_dir}/toxicn_raw.parquet
  {output_dir}/safety_prompts_raw.parquet
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import json
from pathlib import Path
import pandas as pd


def download_toxicn(output_dir: Path):
    """下载 ToxiCN 数据集"""
    print("=" * 60)
    print("下载 ToxiCN (ACL 2023)")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        ds = load_dataset("JunyuLu/ToxiCN", split="train")
        df = ds.to_pandas()
        
        print(f"  原始样本数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        
        # 预览
        print("\n  字段分布:")
        if 'toxic' in df.columns:
            print(f"    toxic: {df['toxic'].value_counts().to_dict()}")
        if 'toxic_type' in df.columns:
            print(f"    toxic_type: {df['toxic_type'].value_counts().to_dict()}")
        if 'expression' in df.columns:
            print(f"    expression: {df['expression'].value_counts().to_dict()}")
        
        out_path = output_dir / "toxicn_raw.parquet"
        df.to_parquet(out_path, index=False)
        print(f"\n  ✓ 保存到: {out_path}")
        return df
        
    except Exception as e:
        print(f"  ⚠️ 自动下载失败: {e}")
        print("  请手动下载:")
        print("    pip install datasets")
        print("    或从 https://huggingface.co/datasets/JunyuLu/ToxiCN 下载")
        return None


def download_safety_prompts(output_dir: Path):
    """下载 Safety-Prompts 数据集"""
    print("\n" + "=" * 60)
    print("下载 Safety-Prompts (清华CoAI)")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        # Safety-Prompts 按类别分 field
        all_records = []
        
        # 典型安全场景
        typical_categories = [
            "Insult",               # 脏话侮辱
            "Unfairness and Discrimination",  # 偏见歧视
            "Crimes and Illegal Activities",  # 违法犯罪
            "Physical Harm",         # 身体伤害
            "Mental Health",         # 心理健康
            "Privacy and Property",  # 财产隐私
            "Ethics and Morality",   # 道德伦理
        ]
        
        category_cn_map = {
            "Insult": "脏话侮辱",
            "Unfairness and Discrimination": "偏见歧视",
            "Crimes and Illegal Activities": "违法犯罪",
            "Physical Harm": "身体伤害",
            "Mental Health": "心理健康",
            "Privacy and Property": "财产隐私",
            "Ethics and Morality": "道德伦理",
        }
        
        for cat in typical_categories:
            try:
                ds = load_dataset(
                    "thu-coai/Safety-Prompts",
                    data_files='typical_safety_scenarios.json',
                    field=cat,
                    split='train'
                )
                df_cat = ds.to_pandas()
                df_cat['category_en'] = cat
                df_cat['category_cn'] = category_cn_map[cat]
                df_cat['scenario'] = 'typical'
                all_records.append(df_cat)
                print(f"  {cat}: {len(df_cat)} 条")
            except Exception as e:
                print(f"  ⚠️ {cat} 加载失败: {e}")
        
        # 指令攻击场景
        attack_categories = [
            "Goal Hijacking",
            "Prompt Leaking",
            "Role Play Instruction",
            "Unsafe Instruction Topic",
            "Inquiry with Unsafe Opinion",
            "Reverse Exposure",
        ]
        
        attack_cn_map = {
            "Goal Hijacking": "目标劫持",
            "Prompt Leaking": "Prompt泄漏",
            "Role Play Instruction": "角色扮演指令",
            "Unsafe Instruction Topic": "不安全指令主题",
            "Inquiry with Unsafe Opinion": "不安全观点询问",
            "Reverse Exposure": "反面诱导",
        }
        
        for cat in attack_categories:
            try:
                ds = load_dataset(
                    "thu-coai/Safety-Prompts",
                    data_files='instruction_attack_scenarios.json',
                    field=cat,
                    split='train'
                )
                df_cat = ds.to_pandas()
                df_cat['category_en'] = cat
                df_cat['category_cn'] = attack_cn_map[cat]
                df_cat['scenario'] = 'attack'
                all_records.append(df_cat)
                print(f"  {cat}: {len(df_cat)} 条")
            except Exception as e:
                print(f"  ⚠️ {cat} 加载失败: {e}")
        
        if all_records:
            df = pd.concat(all_records, ignore_index=True)
            print(f"\n  总样本数: {len(df)}")
            
            out_path = output_dir / "safety_prompts_raw.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  ✓ 保存到: {out_path}")
            return df
        
    except Exception as e:
        print(f"  ⚠️ 自动下载失败: {e}")
        print("  请手动下载:")
        print("    pip install datasets")
        print("    或从 https://github.com/thu-coai/Safety-Prompts 下载")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载外部中文有害文本数据集")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/external_data",
                       help="输出目录")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     下载外部中文有害文本数据集                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    toxicn_df = download_toxicn(output_dir)
    safety_df = download_safety_prompts(output_dir)
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    if toxicn_df is not None:
        print(f"  ToxiCN:         {len(toxicn_df)} 条")
    if safety_df is not None:
        print(f"  Safety-Prompts: {len(safety_df)} 条")
    print(f"  输出目录: {output_dir}")


if __name__ == "__main__":
    main()
