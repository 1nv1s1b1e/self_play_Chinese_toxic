#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToxiCN 数据集下载与格式转换
=============================

将 ToxiCN (ACL 2023) 数据下载并转换为与原 ChineseHarm 完全相同的格式 {文本, 标签}，
使得下游 pipeline (split_dataset.py → prepare_all_data.py → train) 无需改动数据加载逻辑。

ToxiCN 原始标注:
  - toxic: 0/1 (是否有毒)
  - toxic_type: 0(non-toxic) / 1(general offensive) / 2(hate speech)
  - expression: 0(non-hate) / 1(explicit) / 2(implicit) / 3(reporting)
  - target: [LGBTQ, Region, Sexism, Racism, Others, Non-hate] (多标签)

转换后的 6 分类:
  1. 性别歧视  (Sexism)       - target[2]=1
  2. 种族歧视  (Racism)       - target[3]=1
  3. 地域偏见  (Regional Bias) - target[1]=1
  4. LGBTQ歧视 (Anti-LGBTQ)   - target[0]=1
  5. 其他仇恨  (Others)       - target[4]=1, 或 toxic=1 但无特定target
  6. 无毒      (Non-toxic)    - toxic=0

多标签冲突解决策略: 按优先级取第一个匹配的 target
  Sexism > Racism > Region > LGBTQ > Others

额外保留 expression 信息作为难度标注 (不参与分类,仅用于分析):
  - explicit: 显式有毒 (关键词明显)
  - implicit: 隐式有毒 (无关键词但有害) ← 对抗训练核心价值
  - reporting: 转述有毒内容 (本身不有害) ← 易误判

输出:
  origin_data/bench.json     - [{文本, 标签}, ...] 兼容下游pipeline
  origin_data/bench.parquet  - 完整信息 (含expression, source等)
  origin_data/toxicn_stats.json - 数据统计报告

使用方法:
  python download_toxicn.py
  python download_toxicn.py --output_dir ../../origin_data
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import json
import argparse
import ast
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np


# ============================================================
# ToxiCN → 6分类 映射
# ============================================================

# 目标群体索引 (ToxiCN target list)
TARGET_LGBTQ = 0
TARGET_REGION = 1
TARGET_SEXISM = 2
TARGET_RACISM = 3
TARGET_OTHERS = 4
TARGET_NONHATE = 5

# 统一类别名
CATEGORY_SEXISM = "性别歧视"
CATEGORY_RACISM = "种族歧视"
CATEGORY_REGION = "地域偏见"
CATEGORY_LGBTQ = "LGBTQ歧视"
CATEGORY_OTHERS = "其他仇恨"
CATEGORY_NONTOXIC = "无毒"

# 标签优先级 (多标签时取第一个匹配的)
TARGET_PRIORITY = [
    (TARGET_SEXISM, CATEGORY_SEXISM),
    (TARGET_RACISM, CATEGORY_RACISM),
    (TARGET_REGION, CATEGORY_REGION),
    (TARGET_LGBTQ, CATEGORY_LGBTQ),
    (TARGET_OTHERS, CATEGORY_OTHERS),
]

# expression 类型映射
EXPRESSION_MAP = {
    0: "non-hate",
    1: "explicit",
    2: "implicit",
    3: "reporting",
}


def parse_target(target_value) -> list:
    """
    解析 target 字段 (可能是 list, str, 或 numpy array)
    返回 [0/1, 0/1, 0/1, 0/1, 0/1, 0/1] 格式的列表
    """
    if isinstance(target_value, (list, np.ndarray)):
        return [int(x) for x in target_value]
    if isinstance(target_value, str):
        try:
            parsed = ast.literal_eval(target_value.strip())
            return [int(x) for x in parsed]
        except (ValueError, SyntaxError):
            return [0, 0, 0, 0, 0, 1]  # 默认为 non-hate
    return [0, 0, 0, 0, 0, 1]


def assign_category(toxic: int, target: list) -> str:
    """
    根据 toxic 和 target 字段分配统一类别
    
    规则:
    1. toxic=0 → 无毒
    2. toxic=1 → 按 target 优先级分配有害类别
    3. toxic=1 但 target 全为0 → 其他仇恨
    """
    if toxic == 0:
        return CATEGORY_NONTOXIC
    
    # 按优先级查找第一个匹配的target
    for target_idx, category_name in TARGET_PRIORITY:
        if target_idx < len(target) and target[target_idx] == 1:
            return category_name
    
    # 有毒但没有特定target → 其他仇恨
    return CATEGORY_OTHERS


def download_toxicn_hf() -> pd.DataFrame:
    """从 HuggingFace 下载 ToxiCN"""
    print("  方式1: 从 HuggingFace 下载...")
    try:
        from datasets import load_dataset
        ds = load_dataset("JunyuLu/ToxiCN")
        
        # 合并 train + test
        dfs = []
        for split_name in ds.keys():
            split_df = ds[split_name].to_pandas()
            split_df['_split'] = split_name
            dfs.append(split_df)
            print(f"    {split_name}: {len(split_df)} 条")
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"    合计: {len(df)} 条")
        return df
    except Exception as e:
        print(f"    HuggingFace 下载失败: {e}")
        return None


def download_toxicn_github() -> pd.DataFrame:
    """从 GitHub 下载 ToxiCN CSV"""
    print("  方式2: 从 GitHub 下载 CSV...")
    try:
        import requests
        url = "https://raw.githubusercontent.com/DUT-lujunyu/ToxiCN/main/ToxiCN_1.0.csv"
        resp = requests.get(url, timeout=60)
        resp.encoding = 'utf-8'
        
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}")
            return None
        
        # 保存临时文件再读取 (处理encoding)
        import io
        df = pd.read_csv(io.StringIO(resp.text))
        df['_split'] = 'all'
        print(f"    下载成功: {len(df)} 条")
        return df
    except Exception as e:
        print(f"    GitHub 下载失败: {e}")
        return None


def convert_toxicn(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 ToxiCN 原始数据转换为统一格式
    
    Returns:
        DataFrame with columns: [文本, 标签, expression, _split, ...]
    """
    print("\n转换数据格式...")
    
    # 确认文本列名
    text_col = 'content' if 'content' in df.columns else 'text'
    print(f"  文本列: '{text_col}'")
    
    records = []
    errors = 0
    
    for idx, row in df.iterrows():
        # 获取文本 (ToxiCN用'content'列, 兼容'text'列)
        text = str(row.get('content', row.get('text', ''))).strip()
        if not text or len(text) < 3:
            errors += 1
            continue
        
        # 获取标注
        toxic = int(row.get('toxic', 0))
        toxic_type = int(row.get('toxic_type', 0))
        expression = int(row.get('expression', 0))
        target = parse_target(row.get('target', [0, 0, 0, 0, 0, 1]))
        
        # 分配统一类别
        category = assign_category(toxic, target)
        
        records.append({
            "文本": text,
            "标签": category,
            "expression": EXPRESSION_MAP.get(expression, "unknown"),
            "toxic": toxic,
            "toxic_type": toxic_type,
            "_split": row.get('_split', 'unknown'),
        })
    
    result_df = pd.DataFrame(records)
    
    print(f"  有效样本: {len(result_df)} (跳过 {errors} 条异常)")
    
    # 去重
    before = len(result_df)
    result_df = result_df.drop_duplicates(subset=['文本'], keep='first').reset_index(drop=True)
    print(f"  去重: {before} → {len(result_df)}")
    
    return result_df


def print_stats(df: pd.DataFrame) -> dict:
    """打印并返回数据统计"""
    stats = {
        "total": len(df),
        "label_distribution": {},
        "expression_distribution": {},
        "cross_table": {},
    }
    
    print(f"\n{'='*50}")
    print(f"数据统计 (共 {len(df)} 条)")
    print(f"{'='*50}")
    
    # 类别分布
    print("\n标签分布:")
    vc = df['标签'].value_counts()
    for label, count in vc.items():
        pct = count / len(df) * 100
        print(f"  {label:10s}: {count:5d} ({pct:5.1f}%)")
        stats["label_distribution"][label] = count
    
    # 表达方式分布
    if 'expression' in df.columns:
        print("\n表达方式分布:")
        ec = df['expression'].value_counts()
        for expr, count in ec.items():
            pct = count / len(df) * 100
            print(f"  {expr:12s}: {count:5d} ({pct:5.1f}%)")
            stats["expression_distribution"][expr] = count
    
    # 交叉表: 标签 × 表达方式
    if 'expression' in df.columns:
        print("\n标签 × 表达方式 交叉分布:")
        toxic_df = df[df['标签'] != CATEGORY_NONTOXIC]
        if len(toxic_df) > 0:
            ct = pd.crosstab(toxic_df['标签'], toxic_df['expression'])
            print(ct.to_string())
            stats["cross_table"] = ct.to_dict()
    
    # 隐式有毒占比 (核心难度指标)
    implicit_count = len(df[df['expression'] == 'implicit'])
    reporting_count = len(df[df['expression'] == 'reporting'])
    hard_count = implicit_count + reporting_count
    hard_pct = hard_count / len(df) * 100
    
    print(f"\n关键难度指标:")
    print(f"  隐式仇恨 (implicit): {implicit_count} 条")
    print(f"  转述类 (reporting):   {reporting_count} 条")
    print(f"  → 高难度样本占比:     {hard_pct:.1f}% (隐式+转述)")
    
    stats["hard_samples"] = {
        "implicit": implicit_count,
        "reporting": reporting_count,
        "total_hard": hard_count,
        "hard_percentage": round(hard_pct, 1),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="下载并转换 ToxiCN 数据集为统一格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python download_toxicn.py
  python download_toxicn.py --output_dir ../../origin_data

输出格式与 ChineseHarm bench.json 完全一致:
  [{"文本": "...", "标签": "性别歧视"}, ...]

下游pipeline使用:
  1. python download_toxicn.py                    # 下载数据
  2. python split_dataset.py --json_path bench.json  # 划分数据集
  3. python prepare_all_data.py                   # 准备SFT+RL数据
  4. python train_reviewer_lora.py                # 训练Reviewer
  5. python train_challenger_lora.py              # 训练Challenger
        """
    )
    parser.add_argument("--output_dir", type=str,
                       default="../../origin_data",
                       help="输出目录 (默认: origin_data/)")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     ToxiCN 数据集下载与格式转换                          ║")
    print("║     ACL 2023 · 12K 中文有毒语言 · 含隐式仇恨            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # ========================================
    # Step 1: 下载数据
    # ========================================
    print("\n[Step 1] 下载 ToxiCN...")
    
    # df = download_toxicn_hf()
    # if df is None:
    df = download_toxicn_github()
    if df is None:
        print("\n✗ 下载失败! 请手动下载:")
        print("  pip install datasets")
        print("  python -c \"from datasets import load_dataset; ds=load_dataset('JunyuLu/ToxiCN'); print(ds)\"")
        print("  或从 https://github.com/DUT-lujunyu/ToxiCN 下载 ToxiCN_1.0.csv")
        return
    
    print(f"\n  原始列名: {list(df.columns)}")
    print(f"  原始行数: {len(df)}")
    
    # ========================================
    # Step 2: 格式转换
    # ========================================
    print(f"\n[Step 2] 格式转换 (ToxiCN → 统一6分类)...")
    converted_df = convert_toxicn(df)
    
    # ========================================
    # Step 3: 统计分析
    # ========================================
    print(f"\n[Step 3] 数据统计...")
    stats = print_stats(converted_df)
    
    # ========================================
    # Step 4: 保存
    # ========================================
    print(f"\n[Step 4] 保存数据...")
    
    # bench.json (兼容 split_dataset.py 的 JSON 输入)
    bench_records = converted_df[['文本', '标签']].to_dict('records')
    bench_json_path = output_dir / "bench.json"
    with open(bench_json_path, 'w', encoding='utf-8') as f:
        json.dump(bench_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {bench_json_path} ({len(bench_records)} 条)")
    
    # bench.parquet (完整信息,包含expression等)
    bench_parquet_path = output_dir / "bench.parquet"
    converted_df.to_parquet(bench_parquet_path, index=False)
    print(f"  ✓ {bench_parquet_path}")
    
    # 统计报告
    stats_path = output_dir / "toxicn_stats.json"
    stats["description"] = "ToxiCN → 6分类 转换报告"
    stats["source"] = "JunyuLu/ToxiCN (ACL 2023)"
    stats["categories"] = [CATEGORY_SEXISM, CATEGORY_RACISM, CATEGORY_REGION, 
                          CATEGORY_LGBTQ, CATEGORY_OTHERS, CATEGORY_NONTOXIC]
    stats["conversion_rules"] = {
        "multi_label_strategy": "按优先级取第一个匹配: Sexism > Racism > Region > LGBTQ > Others",
        "non_toxic_rule": "toxic=0 → 无毒",
        "fallback_rule": "toxic=1 但无特定target → 其他仇恨",
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {stats_path}")
    
    # ========================================
    # 完成
    # ========================================
    print(f"\n{'='*50}")
    print("✓ ToxiCN 数据准备完成!")
    print(f"{'='*50}")
    print(f"\n输出目录: {output_dir}")
    print(f"  bench.json    ← split_dataset.py 输入 (格式: {{文本, 标签}})")
    print(f"  bench.parquet ← 含完整expression等附加信息")
    print(f"  toxicn_stats.json ← 数据统计报告")
    print(f"\n下一步:")
    print(f"  cd scripts/create_data")
    print(f"  python split_dataset.py --json_path {bench_json_path}")
    print(f"  python prepare_all_data.py")


if __name__ == "__main__":
    main()
