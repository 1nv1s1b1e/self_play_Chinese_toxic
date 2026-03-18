#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换: Reviewer GRPO (Parquet) → Reviewer SFT (JSONL/Parquet)
====================================================================
v2 改进: 兼容 rationale 增强格式。
当 Verifier 提供了 rationale 时，拼接到 assistant 输出中；
否则会留空（Self-Play 循环中，当前 Verifier 不生成 rationale，
模型已在初始 SFT 阶段学会生成分析，这里只需提供正确标签作为监督信号）。
"""

import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",  required=True, type=str)
    parser.add_argument("--output_data", required=True, type=str)
    return parser.parse_args()


def convert_to_sft(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 GRPO parquet (prompt + reward_model + extra_info) 转换为
    SFT 格式 (messages: system + user + assistant)。

    Strategy: High-confidence filtering — 非回放样本若 Verifier 未确认则跳过。
    """
    rows = []
    skipped_low_conf = 0

    for _, row in df.iterrows():
        try:
            prompt_msgs = row.get("prompt", [])
            rm_data = row.get("reward_model", {})
            extra = row.get("extra_info", {})

            if hasattr(prompt_msgs, 'tolist'):
                prompt_msgs = prompt_msgs.tolist()
            elif not isinstance(prompt_msgs, (list, tuple)):
                prompt_msgs = list(prompt_msgs) if prompt_msgs else []

            if prompt_msgs is None or len(prompt_msgs) == 0 or not isinstance(prompt_msgs, (list, tuple)):
                continue

            # High-confidence filter
            is_replay = False
            if isinstance(extra, dict):
                is_replay = extra.get("is_replay", False)
                verifier_confirmed = extra.get("verifier_confirmed")
                if not is_replay and verifier_confirmed is not None and not verifier_confirmed:
                    skipped_low_conf += 1
                    continue

            cat  = rm_data.get("category", "无毒")
            tt   = rm_data.get("toxic_type", "无毒")
            expr = rm_data.get("expression", "非仇恨")

            # 尝试获取 rationale（若 Verifier/上一轮生成了）
            rationale = ""
            if isinstance(extra, dict):
                rationale = extra.get("rationale", "")

            # 构建带分析的 assistant 内容
            parts = []
            if rationale:
                parts.append(f"分析：{rationale}")

            if cat in ["无毒", ""]:
                parts.append("判断：无害")
            else:
                parts.append("判断：有害")
                parts.append(f"类别：{cat}")
                if tt and tt != "无毒":
                    parts.append(f"毒性类型：{tt}")
                if expr and expr != "非仇恨":
                    parts.append(f"表达方式：{expr}")

            assistant_content = "\n".join(parts)

            new_msgs = list(prompt_msgs)
            new_msgs.append({"role": "assistant", "content": assistant_content})

            rows.append({"messages": new_msgs})

        except Exception as e:
            print(f"解析错误跳过: {e}")
            continue

    if skipped_low_conf > 0:
        print(f"   ⚠️ High-confidence filter: 跳过 {skipped_low_conf} 条低置信样本")

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    print(f"🔄 格式转换 (GRPO → SFT)...")
    print(f"   输入: {args.input_data}")

    if args.input_data.endswith(".parquet"):
        df_in = pd.read_parquet(args.input_data)
    else:
        df_in = pd.read_json(args.input_data, lines=True)

    print(f"   输入行数: {len(df_in)}")

    df_out = convert_to_sft(df_in)

    print(f"   转换成功: {len(df_out)} 行")

    out_dir = os.path.dirname(args.output_data)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_parquet(args.output_data, index=False)

    print(f"✅ SFT 数据: {args.output_data}")


if __name__ == "__main__":
    main()
