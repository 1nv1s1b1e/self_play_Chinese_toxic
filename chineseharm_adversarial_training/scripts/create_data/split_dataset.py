#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集全局重新划分脚本
=====================

目标：将ToxiCN数据集做干净的 70/15/15 划分。

数据来源:
- train.parquet: 15,515行(含重复)，去重后9,507条唯一文本
- bench.json: 6,000条(5,992唯一)，100%包含在train.parquet中
- 原始数据集6个类别: 性别歧视/种族歧视/地域偏见/LGBTQ歧视/其他仇恨/无毒

问题:
- bench.json完全被train.parquet包含，存在严重数据泄露
- ToxiCN数据集各类别分布相对均衡

方案: 全局重新划分(70% train / 15% val / 15% test)
- 按类别分层抽样，确保各split中类别比例一致
- 完全消除数据泄露风险
- 各类别训练数据充足

输出文件(存放到 split_data/ 目录):
- train.parquet / train.json    训练集(~6,655条)
- val.parquet / val.json        验证集(~1,426条)  
- test.parquet / test.json      测试集(~1,426条)
- data_split_report.json        数据划分报告
"""

import json
import argparse
import sys
import os
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_deduplicate(data_path: str) -> pd.DataFrame:
    """加载数据并去重，支持 parquet/json 格式"""
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    
    print(f"加载 {data_path}")
    print(f"  原始行数: {len(df)}")
    print(f"  唯一文本: {df['文本'].nunique()}")
    
    # 去重：保留第一次出现
    df_dedup = df.drop_duplicates(subset=['文本'], keep='first').reset_index(drop=True)
    print(f"  去重后行数: {len(df_dedup)}")
    
    return df_dedup


def validate_no_overlap(train_df, val_df, test_df):
    """验证三个split之间无重合"""
    train_texts = set(train_df['文本'])
    val_texts = set(val_df['文本'])
    test_texts = set(test_df['文本'])
    
    tv = train_texts & val_texts
    tt = train_texts & test_texts
    vt = val_texts & test_texts
    
    assert len(tv) == 0, f"Train-Val overlap: {len(tv)}"
    assert len(tt) == 0, f"Train-Test overlap: {len(tt)}"
    assert len(vt) == 0, f"Val-Test overlap: {len(vt)}"
    
    print("✓ 三个split之间无数据重合")


def print_distribution(df, split_name):
    """打印类别分布"""
    vc = df['标签'].value_counts()
    print(f"\n  [{split_name}] ({len(df)} 条)")
    for label in sorted(vc.index):
        count = vc[label]
        print(f"    {label:10s}: {count:5d} ({count/len(df)*100:5.1f}%)")


def save_split(df, output_dir, split_name):
    """保存为parquet和json格式"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parquet (高效存储，训练用)
    parquet_path = output_dir / f"{split_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # JSON (人类可读，评测用)
    json_path = output_dir / f"{split_name}.json"
    records = df[['文本', '标签']].to_dict('records')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ {split_name}: {parquet_path} + {json_path}")


def generate_report(df_all, train_df, val_df, test_df, output_dir, bench_path=None):
    """生成数据划分报告"""
    report = {
        "description": "ToxiCN数据集全局重新划分报告",
        "method": "分层抽样 (stratified split): 70% train / 15% val / 15% test",
        "random_seed": 42,
        "total": {
            "unique_samples": len(df_all),
            "label_distribution": df_all['标签'].value_counts().to_dict()
        },
        "splits": {}
    }
    
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        report["splits"][name] = {
            "num_samples": len(split_df),
            "ratio": round(len(split_df) / len(df_all), 3),
            "label_distribution": split_df['标签'].value_counts().to_dict()
        }
    
    # 检查与原bench.json的重合情况
    if bench_path and os.path.exists(bench_path):
        with open(bench_path, 'r', encoding='utf-8') as f:
            bench_data = json.load(f)
        bench_texts = set(item['文本'] for item in bench_data)
        
        report["bench_overlap"] = {
            "bench_total": len(bench_data),
            "bench_unique": len(bench_texts),
            "in_train": len(bench_texts & set(train_df['文本'])),
            "in_val": len(bench_texts & set(val_df['文本'])),
            "in_test": len(bench_texts & set(test_df['文本'])),
            "note": "原始bench.json的数据被重新分配到三个split中，不再作为独立test set"
        }
    
    report_path = Path(output_dir) / "data_split_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 报告已保存: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="ToxiCN数据集全局重新划分 (70/15/15)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python split_dataset.py
  python split_dataset.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
  python split_dataset.py --output_dir ../../split_data
        """
    )
    parser.add_argument(
        "--parquet_path", type=str,
        default=None,
        help="数据文件路径 (parquet格式)"
    )
    parser.add_argument(
        "--json_path", type=str,
        default=None,
        help="数据文件路径 (json格式，与parquet_path二选一)"
    )
    parser.add_argument(
        "--bench_path", type=str,
        default="../../origin_data/bench.json",
        help="原始 bench.json 路径(用于报告中的重合分析)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="../../split_data",
        help="输出目录"
    )
    parser.add_argument("--train_ratio", type=float, default=0.70, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 验证比例
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, f"比例之和必须为1.0, 当前: {total_ratio}"
    
    print("=" * 70)
    print("ToxiCN 数据集全局重新划分")
    print(f"比例: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val / {args.test_ratio:.0%} test")
    print(f"随机种子: {args.seed}")
    print("=" * 70)
    
    # Step 1: 加载并去重
    print("\n[Step 1] 加载并去重数据...")
    data_path = args.json_path or args.parquet_path or "../../origin_data/bench.json"
    df_all = load_and_deduplicate(data_path)
    
    # Step 2: 分层划分
    print(f"\n[Step 2] 分层抽样划分...")
    
    # 先分出test
    test_size = args.test_ratio
    train_val_df, test_df = train_test_split(
        df_all,
        test_size=test_size,
        random_state=args.seed,
        stratify=df_all['标签']
    )
    
    # 再从train_val中分出val
    val_relative_ratio = args.val_ratio / (args.train_ratio + args.val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=args.seed,
        stratify=train_val_df['标签']
    )
    
    # 重置索引
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"  Train: {len(train_df)} 条 ({len(train_df)/len(df_all)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} 条 ({len(val_df)/len(df_all)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} 条 ({len(test_df)/len(df_all)*100:.1f}%)")
    
    # Step 3: 验证
    print(f"\n[Step 3] 验证数据完整性...")
    validate_no_overlap(train_df, val_df, test_df)
    assert len(train_df) + len(val_df) + len(test_df) == len(df_all), "数据量不匹配"
    print("✓ 数据量完整:", len(train_df), "+", len(val_df), "+", len(test_df), "=", len(df_all))
    
    # Step 4: 打印分布
    print(f"\n[Step 4] 各split类别分布:")
    print_distribution(df_all, "全部数据")
    print_distribution(train_df, "Train")
    print_distribution(val_df, "Val")
    print_distribution(test_df, "Test")
    
    # Step 5: 保存
    print(f"\n[Step 5] 保存文件到 {args.output_dir}...")
    save_split(train_df, args.output_dir, "train")
    save_split(val_df, args.output_dir, "val")
    save_split(test_df, args.output_dir, "test")
    
    # Step 6: 生成报告
    print(f"\n[Step 6] 生成报告...")
    report = generate_report(df_all, train_df, val_df, test_df, args.output_dir, args.bench_path)
    
    # 打印与bench.json的关系
    if "bench_overlap" in report:
        bo = report["bench_overlap"]
        print(f"\n  原始bench.json ({bo['bench_total']}条) 重新分配情况:")
        print(f"    → train: {bo['in_train']} 条")
        print(f"    → val:   {bo['in_val']} 条")
        print(f"    → test:  {bo['in_test']} 条")
    
    print("\n" + "=" * 70)
    print("✓ 数据划分完成！")
    print("=" * 70)
    print(f"\n输出目录: {args.output_dir}")
    print(f"  train.parquet / train.json  → 训练集 ({len(train_df)} 条)")
    print(f"  val.parquet   / val.json    → 验证集 ({len(val_df)} 条)")
    print(f"  test.parquet  / test.json   → 测试集 ({len(test_df)} 条)")
    print(f"  data_split_report.json      → 划分报告")


if __name__ == "__main__":
    main()
