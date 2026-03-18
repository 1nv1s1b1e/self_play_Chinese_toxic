#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据格式转换脚本: Reviewer GRPO (Parquet) -> Reviewer SFT (JSONL/Parquet)
========================================================================
作用:
  由于 GRPO 的生成式强化学习不适合 Reviewer 判别器，我们将 `reviewer_mixed_roundX.parquet`
  或 `reviewer_grpo_roundX.parquet` 中的数据转换回标准多轮对话 SFT 格式。
"""

import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", required=True, type=str, help="输入的 GRPO parquet 或 jsonl 数据")
    parser.add_argument("--output_data", required=True, type=str, help="输出的用于 SFT 的 parquet 数据 (含 messages 字段)")
    return parser.parse_args()

def convert_to_sft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with 'prompt' (list of dict) and 'reward_model' (dict)
    Output: df with 'messages' (list of dict, sys & user & assistant)
    
    Strategy B: High-confidence filtering - skip samples where Verifier
    was uncertain or didn't confirm the target category.
    """
    rows = []
    skipped_low_conf = 0
    
    for _, row in df.iterrows():
        try:
            prompt_msgs = row.get("prompt", [])
            rm_data = row.get("reward_model", {})
            extra = row.get("extra_info", {})
            
            # Convert array types to list
            if isinstance(prompt_msgs, type(pd.array([]))) or type(prompt_msgs).__name__ == 'ndarray':
                prompt_msgs = list(prompt_msgs)
                
            if prompt_msgs is None or len(prompt_msgs) == 0 or not isinstance(prompt_msgs, (list, tuple)):
                continue
            
            # High-confidence filter: skip samples where Verifier didn't confirm
            is_replay = False
            if isinstance(extra, dict):
                is_replay = extra.get("is_replay", False)
                verifier_confirmed = extra.get("verifier_confirmed")
                # For non-replay (dynamic) data, require Verifier confirmation
                if not is_replay and verifier_confirmed is not None and not verifier_confirmed:
                    skipped_low_conf += 1
                    continue
                
            cat = rm_data.get("category", "\u65e0\u6bd2")
            tt  = rm_data.get("toxic_type", "\u65e0\u6bd2")
            expr = rm_data.get("expression", "\u975e\u4ec7\u6068")
            
            if cat in ["\u65e0\u6bd2", ""]:
                assistant_content = "\u5224\u65ad\uff1a\u65e0\u5bb3"
            else:
                assistant_content = (
                    f"\u5224\u65ad\uff1a\u6709\u5bb3\n"
                    f"\u7c7b\u522b\uff1a{cat}\n"
                    f"\u6bd2\u6027\u7c7b\u578b\uff1a{tt}\n"
                    f"\u8868\u8fbe\u65b9\u5f0f\uff1a{expr}"
                )
            
            new_msgs = list(prompt_msgs)
            new_msgs.append({"role": "assistant", "content": assistant_content})
            
            rows.append({
                "messages": new_msgs
            })
            
        except Exception as e:
            print(f"\u89e3\u6790\u9519\u8bef\u8df3\u8fc7: {e}")
            continue
    
    if skipped_low_conf > 0:
        print(f"   \u26a0\ufe0f High-confidence filter: skipped {skipped_low_conf} low-confidence samples")
            
    return pd.DataFrame(rows)

def main():
    args = parse_args()
    print(f"🔄 开始格式转换 (GRPO -> SFT)...")
    print(f"   - 输入数据: {args.input_data}")
    
    if args.input_data.endswith(".parquet"):
        df_in = pd.read_parquet(args.input_data)
    else:
        df_in = pd.read_json(args.input_data, lines=True)
        
    print(f"   => 输入有效载荷数: {len(df_in)}")
    
    df_out = convert_to_sft(df_in)
    
    print(f"   => 转换成功行数: {len(df_out)}")
    
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    df_out.to_parquet(args.output_data, index=False)
    
    print(f"✅ SFT 数据已准备: {args.output_data}")

if __name__ == "__main__":
    main()
