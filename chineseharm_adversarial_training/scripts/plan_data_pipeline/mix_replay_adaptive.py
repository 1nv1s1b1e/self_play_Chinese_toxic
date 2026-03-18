#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 3b+3c: 自适应回放混合 + 多轮缓冲区
==========================================

替代 v1 的 mix_replay_data.py，改进两个方面:

3b. 自适应回放比例
--------------------
  v1: seed_ratio=2.0 固定
  Plan3: seed_ratio 随轮次递减:
    Round 1: 3.0 (多种子稳定冷启动)
    Round 2: 2.0 (v1 默认)
    Round 3: 1.0 (均衡)
    Round 4+: 0.5 (主要靠动态数据)

3c. 多轮缓冲区 (Replay Buffer)
-------------------------------
  v1: 每轮只混合 "本轮新数据 + 种子数据"
  Plan3: 混合 "本轮新数据 + 最近 K 轮的动态数据 + 种子数据"
  SSP 论文 Table 5: Periodic Reset (K=2-3) 比 Full Reuse / No Reuse 效果更好

用法:
  python mix_replay_adaptive.py \\
      --dynamic_data /path/to/reviewer_grpo_round3.parquet \\
      --seed_data /path/to/train_seed.parquet \\
      --output_data /path/to/reviewer_mixed_round3.parquet \\
      --round_idx 3 \\
      --buffer_dir /path/to/dynamic_data_dir \\
      --buffer_k 2
"""

import os
import argparse
import pandas as pd
import json
import glob


def parse_args():
    parser = argparse.ArgumentParser(description="自适应回放混合 (Plan 3)")
    parser.add_argument("--dynamic_data", required=True, type=str,
                        help="本轮动态生成的 Reviewer parquet 数据")
    parser.add_argument("--seed_data",    required=True, type=str,
                        help="原始种子数据 parquet/json")
    parser.add_argument("--output_data",  required=True, type=str,
                        help="混合后的输出 parquet 数据")
    parser.add_argument("--round_idx",    required=True, type=int,
                        help="当前轮次编号 (从 1 开始)")
    parser.add_argument("--buffer_dir",   default="",    type=str,
                        help="动态数据存放目录 (用于查找历史轮次数据)")
    parser.add_argument("--buffer_k",     default=2,     type=int,
                        help="回放缓冲区保留最近 K 轮数据 (默认 2)")
    parser.add_argument("--seed",         default=42,    type=int,
                        help="随机种子")
    return parser.parse_args()


def get_adaptive_seed_ratio(round_idx: int) -> float:
    """
    自适应种子数据混合比例。

    设计原则:
      - 早期轮次: 动态数据少、质量不稳定 → 多种子稳定
      - 后期轮次: 动态数据质量提升 → 降低种子依赖，强化对新模式适应
    """
    ratio_schedule = {
        1: 3.0,   # 冷启动，大量种子数据
        2: 2.0,   # v1 默认值
        3: 1.0,   # 均衡
    }
    return ratio_schedule.get(round_idx, 0.5)  # Round 4+: 0.5


def find_recent_dynamic_data(
    buffer_dir: str,
    current_round: int,
    buffer_k: int,
    role: str = "reviewer",
) -> list:
    """
    查找最近 K 轮的动态数据 parquet 文件。

    在 buffer_dir 下查找 round_{N}/reviewer_grpo_round{N}.parquet
    其中 N ∈ [current_round - buffer_k, current_round - 1]
    """
    history_files = []
    if not buffer_dir or not os.path.isdir(buffer_dir):
        return history_files

    for r in range(max(1, current_round - buffer_k), current_round):
        # 查找多种可能的路径格式
        candidates = [
            os.path.join(buffer_dir, f"round_{r}", f"{role}_grpo_round{r}.parquet"),
            os.path.join(buffer_dir, f"round_{r}", f"{role}_mixed_round{r}.parquet"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                history_files.append(path)
                break  # 同一轮找到一个即可

    return history_files


def convert_seed_to_grpo_format(df_seed: pd.DataFrame) -> pd.DataFrame:
    """
    将原始种子数据转为 Reviewer GRPO 格式。
    与 v1 mix_replay_data.py 逻辑完全一致。
    """
    # 导入 v1 的转换函数 (避免重复代码)
    import sys
    V1_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rl_train")
    sys.path.insert(0, V1_DIR)

    from mix_replay_data import convert_seed_to_grpo_format as v1_convert
    return v1_convert(df_seed)


def main():
    args = parse_args()
    round_idx = args.round_idx
    seed_ratio = get_adaptive_seed_ratio(round_idx)

    print(f"🔄 [Plan 3] 自适应回放混合 (Round {round_idx})...")
    print(f"   - 动态新数据  : {args.dynamic_data}")
    print(f"   - 历史种子数据: {args.seed_data}")
    print(f"   - [Plan 3b] 自适应 seed_ratio = {seed_ratio}  (v1 固定 2.0)")
    print(f"   - [Plan 3c] 回放缓冲区 K = {args.buffer_k} 轮")

    # 1. 读入本轮动态数据
    df_dynamic = pd.read_parquet(args.dynamic_data)
    dynamic_size = len(df_dynamic)
    print(f"   👉 本轮动态样本数: {dynamic_size}")

    if dynamic_size == 0:
        raise ValueError("动态数据为空！")

    # ────────────────────────────────────────────────
    # Plan 3c: 加载历史轮次缓冲区数据
    # ────────────────────────────────────────────────
    df_buffer_parts = [df_dynamic]
    total_dynamic = dynamic_size

    if args.buffer_dir:
        history_files = find_recent_dynamic_data(
            args.buffer_dir, round_idx, args.buffer_k
        )
        for hf in history_files:
            try:
                df_hist = pd.read_parquet(hf)
                df_buffer_parts.append(df_hist)
                total_dynamic += len(df_hist)
                print(f"   📦 缓冲区加入: {hf} ({len(df_hist)} 条)")
            except Exception as e:
                print(f"   ⚠️  跳过损坏的历史文件 {hf}: {e}")

    if len(df_buffer_parts) > 1:
        df_all_dynamic = pd.concat(df_buffer_parts, ignore_index=True)
        print(f"   👉 缓冲区合计动态样本: {total_dynamic} 条 (本轮 {dynamic_size} + 历史 {total_dynamic - dynamic_size})")
    else:
        df_all_dynamic = df_dynamic

    # ────────────────────────────────────────────────
    # Plan 3b: 自适应种子数据混合
    # ────────────────────────────────────────────────
    if args.seed_data.endswith('.parquet'):
        df_seed_raw = pd.read_parquet(args.seed_data)
    else:
        df_seed_raw = pd.read_json(args.seed_data)
    print(f"   👉 种子数据池总量: {len(df_seed_raw)}")

    # 种子采样量基于本轮动态数据量 (非缓冲区总量)
    target_seed_size = int(dynamic_size * seed_ratio)
    target_seed_size = min(target_seed_size, len(df_seed_raw))

    df_seed_sampled = df_seed_raw.sample(n=target_seed_size, random_state=args.seed)
    print(f"   👉 种子采样: {target_seed_size} 条 (ratio={seed_ratio}x)")

    df_seed_grpo = convert_seed_to_grpo_format(df_seed_sampled)

    # ────────────────────────────────────────────────
    # 合并 + 打乱
    # ────────────────────────────────────────────────
    df_mixed = pd.concat([df_all_dynamic, df_seed_grpo], ignore_index=True)
    df_mixed = df_mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    df_mixed.to_parquet(args.output_data, index=False)

    print(f"✅ [Plan 3] 混合完成！")
    print(f"   总数据量: {len(df_mixed)}")
    print(f"     本轮动态: {dynamic_size}")
    print(f"     历史缓冲: {total_dynamic - dynamic_size}")
    print(f"     种子采样: {target_seed_size}")
    print(f"   保存至: {args.output_data}")


if __name__ == "__main__":
    main()
