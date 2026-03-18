#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拒绝采样过滤器 (Rejection Sampler) — 统一版
=============================================
来自 Plan 3a: 在 Phase 0 生成 Challenger GRPO parquet 之前，过滤低质量样本。

两级过滤:
  Level 1 — 质量门控 (quality_gate):
    过滤 AI 拒绝、纯重复、太短、格式泄漏等垃圾文本。
    在 Verifier 评估之前执行，减少不必要的 API 调用。

  Level 2 — 标签一致性 (label_consistency):
    在 Verifier 评估之后执行。利用 Verifier 分类结果与 Challenger 目标
    类别比对，过滤 off-target 样本:
      - Challenger 目标为有害类别，Verifier 判定无毒 → off-target
      - Challenger 目标为无毒，Verifier 判定有害   → off-target (反向)
      - 可选: 有害类别不匹配 (如目标"性别歧视"但判定"其他仇恨")

用法:
  # Level 1: 质量过滤 (在 Verifier 之前)
  result = filter_low_quality_samples(tasks, generated_texts, threshold=0.3)

  # Level 2: 标签一致性过滤 (在 Verifier 之后)
  result = filter_label_inconsistent(
      tasks, generated_texts, verifier_results,
      reviewer_outputs=..., sample_rewards=...,
  )
"""

import os, sys
import logging

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
if _SELF_DIR not in sys.path:
    sys.path.insert(0, _SELF_DIR)

from quality_gate import quality_gate
from constants import HARMFUL_CATEGORIES

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════════
# Level 1: 质量门控过滤 (Verifier 之前)
# ══════════════════════════════════════════════════════════════════════════════════

def filter_low_quality_samples(
    tasks: list,
    generated_texts: list,
    threshold: float = 0.3,
    verifier_results: list = None,
    reviewer_outputs: list = None,
    sample_rewards: list = None,
) -> dict:
    """
    基于 quality_gate 过滤低质量 Challenger 生成文本。

    同步过滤所有相关列表 (tasks, texts, verifier results, etc.)，
    保证索引对齐。

    Args:
        tasks:            采样任务列表
        generated_texts:  Challenger 生成文本
        threshold:        质量门控阈值 (默认 0.3，低于此值过滤)
        verifier_results: 可选，Verifier 验证结果
        reviewer_outputs: 可选，Reviewer 输出
        sample_rewards:   可选，逐样本奖励

    Returns:
        dict: {
            "tasks": filtered_tasks,
            "generated_texts": filtered_texts,
            "verifier_results": filtered_vr,
            "reviewer_outputs": filtered_ro,
            "sample_rewards": filtered_sr,
            "n_original": int,
            "n_filtered": int,
            "n_kept": int,
            "rejection_rate": float,
        }
    """
    n_original = len(generated_texts)
    keep_indices = []

    for i, text in enumerate(generated_texts):
        gate = quality_gate(text.strip() if text else "")
        if gate >= threshold:
            keep_indices.append(i)

    n_kept = len(keep_indices)
    n_filtered = n_original - n_kept

    result = {
        "tasks":            [tasks[i] for i in keep_indices],
        "generated_texts":  [generated_texts[i] for i in keep_indices],
        "verifier_results": [verifier_results[i] for i in keep_indices] if verifier_results else None,
        "reviewer_outputs":  [reviewer_outputs[i] for i in keep_indices] if reviewer_outputs else None,
        "sample_rewards":   [sample_rewards[i] for i in keep_indices] if sample_rewards else None,
        "n_original":       n_original,
        "n_filtered":       n_filtered,
        "n_kept":           n_kept,
        "rejection_rate":   n_filtered / n_original if n_original > 0 else 0.0,
    }

    return result


# ══════════════════════════════════════════════════════════════════════════════════
# Level 2: 标签一致性过滤 (Verifier 之后)
# ══════════════════════════════════════════════════════════════════════════════════

def _check_label_consistency(
    target_category: str,
    verifier_result: dict,
    strict_category: bool = False,
) -> tuple:
    """
    检查 Verifier 分类与 Challenger 目标类别的一致性。

    规则:
      1. 目标有害 + Verifier 判无毒     → off-target (不一致)
      2. 目标无毒 + Verifier 判有害     → off-target (反向不一致)
      3. 目标有害 + Verifier 有害但类别不同 → 若 strict_category=True 则不一致
      4. Verifier 无法解析 (binary=None) → 不一致 (不可信样本)

    Args:
        target_category:  Challenger 的目标类别
        verifier_result:  Verifier 分类结果 dict (binary, category, ...)
        strict_category:  是否要求有害类别精确匹配

    Returns:
        (is_consistent: bool, reason: str)
    """
    v_binary   = verifier_result.get("binary")
    v_category = verifier_result.get("category")
    target_is_harmful = target_category in HARMFUL_CATEGORIES

    # Verifier 无法解析 → 不可信
    if v_binary is None:
        return False, "verifier_parse_fail"

    v_is_harmful = (v_binary == "有害")

    # Case 1: 目标有害但 Verifier 判无毒 (Challenger 生成的是无害文本，off-target)
    if target_is_harmful and not v_is_harmful:
        return False, "target_harmful_but_verifier_safe"

    # Case 2: 目标无毒但 Verifier 判有害 (生成了有毒文本，off-target)
    if not target_is_harmful and v_is_harmful:
        return False, "target_safe_but_verifier_harmful"

    # Case 3: 都是有害，但类别不同
    if target_is_harmful and v_is_harmful and strict_category:
        if v_category and v_category != target_category:
            return False, f"category_mismatch:{target_category}→{v_category}"

    return True, "consistent"


def filter_label_inconsistent(
    tasks: list,
    generated_texts: list,
    verifier_results: list,
    reviewer_outputs: list = None,
    sample_rewards: list = None,
    strict_category: bool = False,
) -> dict:
    """
    基于 Verifier 标签一致性过滤 off-target 样本。
    在 Verifier 评估完成后、构建 parquet 之前调用。

    过滤逻辑 (基于分类规则):
      - 有害目标 + Verifier 判无毒 → 生成文本偏离了目标类别规则，过滤
      - 无毒目标 + Verifier 判有害 → 生成了违规内容，过滤
      - Verifier 解析失败          → 标签不可信，过滤
      - strict_category=True 时: 有害大类一致但细类别不同也过滤

    Args:
        tasks:            采样任务列表 (含 "category" 字段)
        generated_texts:  Challenger 生成文本
        verifier_results: Verifier 分类结果列表
        reviewer_outputs: 可选，Reviewer 输出
        sample_rewards:   可选，逐样本奖励
        strict_category:  是否要求有害类别精确匹配 (默认 False)

    Returns:
        dict: 同 filter_low_quality_samples，额外包含:
            "n_off_target": int,
            "consistency_stats": dict  — 按过滤原因分类的统计
    """
    n_original = len(generated_texts)
    keep_indices = []
    consistency_stats = {}

    for i in range(n_original):
        target_cat = tasks[i].get("category", "")
        vr = verifier_results[i] if i < len(verifier_results) else {}

        is_consistent, reason = _check_label_consistency(
            target_category=target_cat,
            verifier_result=vr,
            strict_category=strict_category,
        )

        if is_consistent:
            keep_indices.append(i)
        else:
            consistency_stats[reason] = consistency_stats.get(reason, 0) + 1

    n_kept = len(keep_indices)
    n_off_target = n_original - n_kept

    result = {
        "tasks":            [tasks[i] for i in keep_indices],
        "generated_texts":  [generated_texts[i] for i in keep_indices],
        "verifier_results": [verifier_results[i] for i in keep_indices],
        "reviewer_outputs":  [reviewer_outputs[i] for i in keep_indices] if reviewer_outputs else None,
        "sample_rewards":   [sample_rewards[i] for i in keep_indices] if sample_rewards else None,
        "n_original":       n_original,
        "n_off_target":     n_off_target,
        "n_kept":           n_kept,
        "rejection_rate":   n_off_target / n_original if n_original > 0 else 0.0,
        "consistency_stats": consistency_stats,
    }

    return result
