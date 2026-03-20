#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一版 Parquet 构建器 — Challenger + Reviewer
==============================================
核心改进 (Few-Shot + 1-acc 版本):
  - Challenger prompt 含 few-shot 多轮示例（来自种子数据随机采样）
  - 奖励信号直接使用 reviewer_was_fooled (1-acc)，无 Verifier 依赖
  - Reviewer parquet 混合原始种子数据，防止分布偏移与灾难性遗忘
"""

import os, sys
import random
import pandas as pd
from typing import Dict, List, Optional

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from constants import (
    HARMFUL_CATEGORIES,
    REVIEWER_SYSTEM_PROMPT,
    build_challenger_system_prompt,
    format_reviewer_user_content,
)


def build_challenger_parquet(
    tasks: List[Dict],
    sample_rewards: List[Dict],
    verifier_stats: Optional[Dict] = None,   # 保留参数签名兼容性，但已废弃
) -> pd.DataFrame:
    """
    构建含逐样本 1-acc 对抗信号的 Challenger GRPO 训练 DataFrame。

    prompt 格式（Few-Shot 多轮对话）：
      [system]    → 类别规则 system prompt
      [user]      → 干净 base instruction（仅当 few_shot_example 存在时）
      [assistant] → few-shot 种子样本（仅当 few_shot_example 存在时）
      [user]      → 实际生成请求（与 base instruction 相同）

    Args:
        tasks:          采样任务列表（含 few_shot_example 字段）
        sample_rewards: Phase 0 逐样本奖励（含 reviewer_was_fooled）
        verifier_stats: 已废弃，忽略（保留接口兼容性）

    Returns:
        pd.DataFrame: 含 prompt, reward_model, extra_info 三列
    """
    rows = []

    for i, task in enumerate(tasks):
        instr       = task["challenger_instruction"]
        cat         = task["category"]
        few_shot    = task.get("few_shot_example", "").strip()

        # ── few-shot 多轮 prompt 构建 ──
        msgs = [{"role": "system", "content": build_challenger_system_prompt(cat)}]
        if few_shot:
            msgs.append({"role": "user",      "content": instr})
            msgs.append({"role": "assistant", "content": few_shot})
        msgs.append({"role": "user", "content": instr})

        # ── 逐样本 1-acc 信号 ──
        sr = sample_rewards[i] if i < len(sample_rewards) else {}
        reviewer_fooled    = sr.get("reviewer_was_fooled", False)
        reviewer_binary_ok = sr.get("reviewer_binary_correct", True)

        rows.append({
            "prompt": msgs,
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
                "few_shot_example":   few_shot,

                # ── 多级对抗信号 ──
                "reviewer_fooled":     bool(reviewer_fooled),
                "reviewer_cat_fooled": bool(sr.get("reviewer_cat_fooled", False)),
                "reviewer_binary_ok":  bool(reviewer_binary_ok),
            },
        })

    return pd.DataFrame(rows)


def build_reviewer_parquet(
    tasks:            List[Dict],
    generated_texts:  List[str],
    verifier_results: Optional[List[Dict]] = None,  # 已废弃，保留兼容性
    seed_df:          Optional[pd.DataFrame] = None, # 原始种子数据（用于混合）
    mix_ratio:        float = 0.5,                   # 原始数据混合比例 (0-1)，默认 0.5
    nontoxic_boost:   float = 2.0,                   # 无毒样本过采样倍数，默认 2.0
    hard_sample_rows: Optional[List[Dict]] = None,   # 当前轮困难样本（Reviewer 错判）
    hard_sample_multiplier: int = 2,                 # 困难样本重采样倍数
    repeat_bonus_cap: int = 2,                       # 连续错题额外加权上限
) -> pd.DataFrame:
    """
    构建 Reviewer GRPO/SFT 训练 DataFrame。
    
    关键改进：混合原始种子数据 + 无毒过采样，防止灾难性遗忘！
    
    问题背景：
    - Challenger 只生成有害文本 → Reviewer 训练数据 100% 有害
    - 测试集中 47% 是无毒 → 严重分布偏移 → 无毒召回率暴降
    
    解决方案：
    - 混合原始种子数据（含无毒）
    - 对无毒样本过采样，平衡训练分布
    
    Args:
        tasks:           任务列表
        generated_texts: Challenger 生成的文本列表
        verifier_results: 已废弃
        seed_df:         原始种子数据 DataFrame（用于混合）
        mix_ratio:       原始数据占总数据的比例，默认 0.5
        nontoxic_boost:  无毒样本过采样倍数，默认 2.0（采样 2 倍于有害样本的无毒）
        hard_sample_rows: 当前轮样本评估明细（含 reviewer_*_correct 字段）
        hard_sample_multiplier: 困难样本重采样倍数，默认 2
        repeat_bonus_cap: 对跨轮重复错题增加的额外权重上限（每题）
    """
    rows = []
    
    # ── Step 1: 添加 Challenger 生成的对抗样本（全是有害）──
    adversarial_count = 0
    for i, (task, gen_text) in enumerate(zip(tasks, generated_texts)):
        if not gen_text or len(gen_text.strip()) < 3:
            continue

        cat  = task["category"]
        expr = task["expression"]
        tt   = task["toxic_type"]

        user_content = format_reviewer_user_content(gen_text.strip())

        rows.append({
            "prompt": [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "reward_model": {
                "ground_truth": cat,
                "category":     cat,
                "toxic_type":   tt,
                "expression":   expr,
            },
            "extra_info": {
                "original_text": gen_text.strip()[:200],
                "category":      cat,
                "source":        "adversarial",
            },
        })
        adversarial_count += 1
    
    # ── Step 2: 混合原始种子数据（防止灾难性遗忘）──
    if seed_df is not None and mix_ratio > 0 and len(seed_df) > 0:
        # 识别列名
        col_text = "文本"  if "文本"  in seed_df.columns else "original_text"
        col_cat  = "标签"  if "标签"  in seed_df.columns else "category"
        col_tt   = "toxic_type_label" if "toxic_type_label" in seed_df.columns else "toxic_type"
        col_expr = "expression_label" if "expression_label" in seed_df.columns else "expression"
        
        # 计算需要混合的原始样本总数
        # mix_ratio=0.5 → 原始数据占 50%，对抗数据占 50%
        # original_count = adversarial_count * mix_ratio / (1 - mix_ratio)
        total_original_count = int(adversarial_count * mix_ratio / max(1 - mix_ratio, 0.01))
        
        # ── 分层采样：无毒过采样 ──
        nontoxic_df = seed_df[seed_df[col_cat] == "无毒"]
        toxic_df    = seed_df[seed_df[col_cat] != "无毒"]
        
        # 计算无毒和有害的采样数量（无毒占更大比例）
        # nontoxic_boost=2.0 → 无毒样本数 = 有害样本数 × 2
        nontoxic_ratio = nontoxic_boost / (1 + nontoxic_boost)  # 2/(1+2) = 0.667
        toxic_ratio    = 1 - nontoxic_ratio                      # 0.333
        
        nontoxic_count = int(total_original_count * nontoxic_ratio)
        toxic_count    = total_original_count - nontoxic_count
        
        # 确保不超过可用样本数
        nontoxic_count = min(nontoxic_count, len(nontoxic_df))
        toxic_count    = min(toxic_count, len(toxic_df))
        
        # 采样
        def sample_and_add(df_subset, count, source_tag):
            if count <= 0 or len(df_subset) == 0:
                return 0
            indices = random.sample(range(len(df_subset)), min(count, len(df_subset)))
            added = 0
            for idx in indices:
                row = df_subset.iloc[idx]
                text = str(row.get(col_text, "")).strip()
                if not text or len(text) < 3:
                    continue
                
                cat  = row.get(col_cat, "")
                expr = row.get(col_expr, "") if col_expr in seed_df.columns else ""
                tt   = row.get(col_tt, "")   if col_tt in seed_df.columns else ""
                
                rows.append({
                    "prompt": [
                        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                        {"role": "user",   "content": format_reviewer_user_content(text)},
                    ],
                    "reward_model": {
                        "ground_truth": cat,
                        "category":     cat,
                        "toxic_type":   tt,
                        "expression":   expr,
                    },
                    "extra_info": {
                        "original_text": text[:200],
                        "category":      cat,
                        "source":        source_tag,
                    },
                })
                added += 1
            return added
        
        sample_and_add(nontoxic_df, nontoxic_count, "original_nontoxic")
        sample_and_add(toxic_df, toxic_count, "original_toxic")

    # ── Step 3: 困难样本重采样（平台期破局）──
    # 对 Reviewer 错判样本进行重复训练，优先修正当前决策边界。
    if hard_sample_rows and hard_sample_multiplier > 1:
        # 先按文本聚合频次：同一题在历史中反复错，给予更高重采样权重。
        hard_map: Dict[str, Dict] = {}
        hard_freq: Dict[str, int] = {}

        def norm_text(s: str) -> str:
            return " ".join((s or "").strip().split())[:300]

        for sr in hard_sample_rows:
            binary_ok = bool(sr.get("reviewer_binary_correct", True))
            cat_ok = bool(sr.get("reviewer_cat_correct", True))
            if binary_ok and cat_ok:
                continue

            text = (sr.get("generated_text") or "").strip()
            if len(text) < 3:
                continue

            key = norm_text(text)
            hard_freq[key] = hard_freq.get(key, 0) + 1
            if key not in hard_map:
                hard_map[key] = sr

        for key, sr in hard_map.items():
            text = (sr.get("generated_text") or "").strip()
            freq = hard_freq.get(key, 1)

            cat = sr.get("target_category", "")
            expr = sr.get("target_expression", "")
            tt = sr.get("target_toxic_type", "")

            repeats = hard_sample_multiplier + min(max(freq - 1, 0), repeat_bonus_cap)
            for _ in range(repeats):
                rows.append({
                    "prompt": [
                        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                        {"role": "user",   "content": format_reviewer_user_content(text)},
                    ],
                    "reward_model": {
                        "ground_truth": cat,
                        "category":     cat,
                        "toxic_type":   tt,
                        "expression":   expr,
                    },
                    "extra_info": {
                        "original_text": text[:200],
                        "category":      cat,
                        "source":        "hard_mining",
                    },
                })

    # 打乱顺序，避免来源分段
    random.shuffle(rows)

    return pd.DataFrame(rows)
