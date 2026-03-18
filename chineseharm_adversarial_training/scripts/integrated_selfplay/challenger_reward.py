#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger 奖励函数 — 多级对抗版 (binary fooling + category confusion)
======================================================================
在 1-acc 基础上增加 category confusion 信号：
  → reviewer binary 被骗        : adversarial = 1.0
  → reviewer binary 对, category 错 : adversarial = 0.6
  → reviewer 完全正确           : adversarial = 0.0

这让 Challenger 不仅学习生成"骗过二分类"的文本，
还学习生成"让 Reviewer 在有害子类别间混淆"的文本，
从而推动 Reviewer 提升 category 区分能力。

extra_info 字段由 generate_dynamic_data 在 Phase 0 注入:
  - reviewer_fooled: bool      (binary 被骗)
  - reviewer_cat_fooled: bool  (binary 对但 category 错)
"""

import os, sys
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from quality_gate import quality_gate


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    Challenger 多级对抗奖励函数。

    Returns:
        float ∈ [-1.0, 1.0]
    """
    if data_source != "toxicn_challenger":
        return 0.0

    generated = solution_str.strip() if solution_str else ""

    # ── 质量门控 ──
    gate = quality_gate(generated)
    if gate < 0.01:
        return -1.0

    # ── 提取对抗信号 ──
    if not extra_info or not isinstance(extra_info, dict):
        return gate * 2.0 - 1.0

    reviewer_fooled = extra_info.get("reviewer_fooled", None)
    if reviewer_fooled is None:
        reviewer_fooled = extra_info.get("reviewer_was_fooled", None)

    reviewer_cat_fooled = extra_info.get("reviewer_cat_fooled", None)

    if reviewer_fooled is None and reviewer_cat_fooled is None:
        return gate * 2.0 - 1.0

    # 多级对抗信号
    if reviewer_fooled:
        # binary 被骗 → 最高奖励
        adversarial_signal = 1.0
    elif reviewer_cat_fooled:
        # binary 对但 category 错 → 中等奖励
        adversarial_signal = 0.6
    else:
        # Reviewer 完全正确 → 不奖励
        adversarial_signal = 0.0

    raw = gate * adversarial_signal
    return raw * 2.0 - 1.0
