#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仓库标准数据转换脚本
====================

将 ChineseHarm 仓库的 train.json / test.json 转换为管线统一格式，
输出到 split_data/ 目录，同时生成 .json 和 .parquet

仓库原始字段:
  content, topic, toxic, toxic_type, expression, target, platform, length

转换后字段:
  文本, 标签, toxic, toxic_type, toxic_type_label, expression, expression_label,
  target, platform, topic

映射规则:
  topic → 标签:
    toxic==0                  → "无毒"
    toxic==1, toxic_type==1   → "其他仇恨" (一般攻击性, 不针对特定群体)
    toxic==1, toxic_type==2   → {gender→性别歧视, race→种族歧视, region→地域偏见, lgbt→LGBTQ歧视}

用法:
  python convert_repo_data.py \\
      --train_json ../../train.json \\
      --test_json  ../../test.json \\
      --output_dir ../../split_data \\
      --val_ratio 0.15
"""

import json
import argparse
import random
import os
import sys
from pathlib import Path
from collections import Counter

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠ pandas 未安装, 仅输出 .json 格式 (跳过 .parquet)")


# ============================================================
# 字段映射
# ============================================================
TOPIC_TO_CATEGORY = {
    "gender": "性别歧视",
    "race": "种族歧视",
    "region": "地域偏见",
    "lgbt": "LGBTQ歧视",
}

TOXIC_TYPE_MAP = {0: "无毒", 1: "一般攻击性", 2: "仇恨言论"}
EXPRESSION_MAP = {0: "非仇恨", 1: "显式仇恨", 2: "隐式仇恨", 3: "转述"}


def convert_record(record: dict) -> dict:
    """
    将仓库原始记录转换为管线统一格式
    """
    toxic = record["toxic"]
    topic = record["topic"]
    toxic_type_int = record["toxic_type"]
    expression_int = record["expression"]

    # 确定类别标签
    if toxic == 0:
        category = "无毒"
    elif toxic_type_int == 1:
        # 一般攻击性 → 不针对特定群体
        category = "其他仇恨"
    else:
        category = TOPIC_TO_CATEGORY.get(topic, "其他仇恨")

    return {
        "文本": record["content"],
        "标签": category,
        "toxic": toxic,
        "toxic_type": toxic_type_int,
        "toxic_type_label": TOXIC_TYPE_MAP.get(toxic_type_int, "仇恨言论"),
        "expression": expression_int,
        "expression_label": EXPRESSION_MAP.get(expression_int, "显式仇恨"),
        "target": record.get("target", [0, 0, 0, 0, 0]),
        "platform": record.get("platform", ""),
        "topic": topic,
    }


def save_split(records: list, output_dir: Path, name: str):
    """保存一个 split (同时输出 .json 和 .parquet)"""
    # JSON
    json_path = output_dir / f"{name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  {name}.json: {len(records)} 条")

    # Parquet
    if pd is not None:
        parquet_path = output_dir / f"{name}.parquet"
        df = pd.DataFrame(records)
        # target 是 list, parquet 存为 string 避免问题
        if "target" in df.columns:
            df["target"] = df["target"].apply(json.dumps)
        df.to_parquet(parquet_path, index=False)
        print(f"  {name}.parquet: {len(df)} 条")


def print_distribution(records: list, name: str):
    """打印分布统计"""
    cats = Counter(r["标签"] for r in records)
    tt = Counter(r["toxic_type_label"] for r in records)
    ex = Counter(r["expression_label"] for r in records)
    print(f"\n  [{name}] 类别分布:")
    for c in sorted(cats.keys()):
        print(f"    {c:10s}: {cats[c]}")
    print(f"  [{name}] 毒性类型分布: {dict(tt)}")
    print(f"  [{name}] 表达方式分布: {dict(ex)}")


def main():
    parser = argparse.ArgumentParser(
        description="将 ChineseHarm 仓库 train.json/test.json 转换为管线统一格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train_json", type=str, default="../../../train.json",
        help="仓库 train.json 路径",
    )
    parser.add_argument(
        "--test_json", type=str, default="../../../test.json",
        help="仓库 test.json 路径",
    )
    parser.add_argument(
        "--output_dir", type=str, default="../../split_data",
        help="输出目录 (split_data/)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15,
        help="从 train 中划分出 val 的比例",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (用于 train/val 划分)",
    )
    args = parser.parse_args()

    # 解析相对路径 (相对于本脚本所在目录)
    script_dir = Path(__file__).parent
    train_path = Path(args.train_json)
    test_path = Path(args.test_json)
    output_dir = Path(args.output_dir)

    if not train_path.is_absolute():
        train_path = (script_dir / train_path).resolve()
    if not test_path.is_absolute():
        test_path = (script_dir / test_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()

    print("=" * 60)
    print("仓库标准数据转换")
    print("=" * 60)
    print(f"  train_json: {train_path}")
    print(f"  test_json:  {test_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  val_ratio:  {args.val_ratio}")
    print(f"  seed:       {args.seed}")

    # ---- 加载原始数据 ----
    print("\n[1/3] 加载原始数据...")
    with open(train_path, "r", encoding="utf-8") as f:
        raw_train = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        raw_test = json.load(f)
    print(f"  原始 train: {len(raw_train)} 条")
    print(f"  原始 test:  {len(raw_test)} 条")

    # ---- 转换字段 ----
    print("\n[2/3] 转换字段...")
    converted_train = [convert_record(r) for r in raw_train]
    converted_test = [convert_record(r) for r in raw_test]

    # ---- 划分 train / val ----
    random.seed(args.seed)
    random.shuffle(converted_train)
    val_size = int(len(converted_train) * args.val_ratio)
    val_records = converted_train[:val_size]
    train_records = converted_train[val_size:]

    print(f"  划分结果: train={len(train_records)}, val={val_size}, test={len(converted_test)}")

    # ---- 保存 ----
    print("\n[3/3] 保存到 split_data/...")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(train_records, output_dir, "train")
    save_split(val_records, output_dir, "val")
    save_split(converted_test, output_dir, "test")

    # ---- 统计 ----
    print_distribution(train_records, "train")
    print_distribution(converted_test, "test")

    # ---- 生成报告 ----
    report = {
        "description": "仓库标准数据转换报告",
        "source": {
            "train_json": str(train_path),
            "test_json": str(test_path),
        },
        "output": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(converted_test),
        },
        "field_mapping": {
            "content → 文本": "原始文本",
            "topic+toxic+toxic_type → 标签": "6类中文标签",
            "toxic_type → toxic_type_label": "无毒/一般攻击性/仇恨言论",
            "expression → expression_label": "非仇恨/显式仇恨/隐式仇恨/转述",
        },
        "category_mapping": {
            "toxic==0": "无毒",
            "toxic==1, toxic_type==1": "其他仇恨 (一般攻击性)",
            "toxic==1, toxic_type==2, topic=gender": "性别歧视",
            "toxic==1, toxic_type==2, topic=race": "种族歧视",
            "toxic==1, toxic_type==2, topic=region": "地域偏见",
            "toxic==1, toxic_type==2, topic=lgbt": "LGBTQ歧视",
        },
        "seed": args.seed,
        "val_ratio": args.val_ratio,
    }
    report_path = output_dir / "data_split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✓ 转换完成! 输出到 {output_dir}")
    print(f"  train: {len(train_records)} 条")
    print(f"  val:   {len(val_records)} 条")
    print(f"  test:  {len(converted_test)} 条")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
