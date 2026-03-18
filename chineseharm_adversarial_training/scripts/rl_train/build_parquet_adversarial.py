#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 1 专用: 修改版 build_challenger_parquet — 注入逐样本对抗信号
================================================================

本文件是一个补丁模块，导出 `build_challenger_parquet_adversarial` 函数，
用于替代 v1 `generate_dynamic_data.py` 中的 `build_challenger_parquet`。

核心区别:
  v1:    extra_info 中 verifier_asr / confirms_rate 是类别级常量
  Plan1: extra_info 中 reviewer_fooled / reviewer_binary_ok 是逐样本 bool

用法 (在 generate_dynamic_data.py 的 Step 7 中替换):
  from plan_reward_shaping.build_parquet_adversarial import build_challenger_parquet_adversarial
  challenger_df = build_challenger_parquet_adversarial(tasks, sample_rewards, verifier_stats)
"""

import pandas as pd
from typing import Dict, List, Optional


HARMFUL_CATEGORIES = {"性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"}


def build_challenger_parquet_adversarial(
    tasks:           List[Dict],
    sample_rewards:  List[Dict],
    verifier_stats:  Dict[str, Dict],
    build_system_prompt_fn=None,
) -> pd.DataFrame:
    """
    构建含逐样本对抗信号的 Challenger GRPO 训练 DataFrame。

    与 v1 build_challenger_parquet 的区别:
      v1:    extra_info.verifier_asr = 类别级 float (同类别样本相同)
      Plan1: extra_info.reviewer_fooled = 逐样本 bool (每条样本独立)
             extra_info.reviewer_binary_ok = 逐样本 bool

    这使得 challenger_reward_adversarial.py 的奖励函数能在
    GRPO 的 group-relative advantage 中产生有效梯度。

    Args:
        tasks:           采样任务列表，每个 task 含 challenger_instruction, category 等
        sample_rewards:  Phase 0 Step 6 计算的逐样本奖励，含:
                         - reviewer_was_fooled: bool
                         - reviewer_binary_correct: bool
                         - reviewer_cat_correct: bool
                         - r_challenger: float
        verifier_stats:  类别级统计 (仅用于参考/日志，不注入 reward)
        build_system_prompt_fn: 可选的 system prompt 构建函数

    Returns:
        pd.DataFrame: 含 prompt, reward_model, extra_info 三列
    """
    rows = []

    for i, task in enumerate(tasks):
        instr    = task["challenger_instruction"]
        cat      = task["category"]
        cat_stat = verifier_stats.get(cat, {
            "verifier_asr":           0.5,
            "verifier_confirms_rate": 0.5,
            "avg_r_challenger":       0.0,
        })

        # ── 逐样本信号 (核心改进) ──
        sr = sample_rewards[i] if i < len(sample_rewards) else {}
        reviewer_fooled    = sr.get("reviewer_was_fooled", False)
        reviewer_binary_ok = sr.get("reviewer_binary_correct", True)

        # 构建 system prompt
        if build_system_prompt_fn:
            sys_content = build_system_prompt_fn(cat)
        else:
            sys_content = "You are a helpful assistant."

        rows.append({
            "prompt": [
                {"role": "system", "content": sys_content},
                {"role": "user",   "content": instr},
            ],
            "reward_model": {
                "ground_truth":    cat,
                "target_category": cat,
                "reference_texts": task.get("reference_texts", [])[:10],
            },
            "extra_info": {
                "category":           cat,
                "expression":         task.get("expression", ""),
                "toxic_type":         task.get("toxic_type", ""),
                "original_text":      task.get("reference_anchor", ""),

                # ── Plan 1 核心: 逐样本对抗信号 ──
                "reviewer_fooled":    bool(reviewer_fooled),
                "reviewer_binary_ok": bool(reviewer_binary_ok),

                # ── 保留类别级信号用于日志/调试 (不影响 reward) ──
                "verifier_asr":                 cat_stat.get("verifier_asr", 0.5),
                "verifier_confirms_rate":       cat_stat.get("verifier_confirms_rate", 0.5),
                "cat_adversarial_success_rate":  cat_stat.get("verifier_asr", 0.5),
                "cat_label_verified_rate":       cat_stat.get("verifier_confirms_rate", 0.5),
            },
        })

    return pd.DataFrame(rows)
