#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer 奖励函数 — 多级 acc (binary + category 双层信号)
==========================================================
核心改进：从纯 binary (+1/-1) 升级为多级奖励，让 GRPO 拥有
category 级别的梯度信号，打破平台期。

奖励层次:
  +1.0  binary 正确 + category 完全匹配
  +0.4  binary 正确 + category 错误（有害文本判对了二分类，但细类别错）
  +0.6  binary 正确 + 无毒文本正确判为无害
  -0.5  binary 错误（有害判无害 / 无害判有害）
  -1.0  parse 失败（格式错误、空输出）

设计原理：
  旧版 binary-only 奖励下，GRPO 的 8 个 rollout 中只要 binary 对了
  就全部拿 +1.0 → 完全无法区分"binary对+category对"和"binary对+category错"
  → category 准确率没有梯度 → 永远无法提升。
  新版让 category-correct 比 category-wrong 多拿 0.6 分 → GRPO
  可以从 8 个 rollout 中优选 category 也对的那些 → 产生改进。
"""

import os, sys, json as _json
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SELF_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from constants import (
    HARMFUL_CATEGORIES,
    parse_classification_output,
)

try:
    from reward_functions.reward_logger import RewardLogger
except ImportError:
    from reward_logger import RewardLogger

_logger = RewardLogger("reviewer")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    Reviewer 多级奖励函数（binary + category 双层信号）。

    Returns:
        float ∈ [-1.0, 1.0]
    """
    if data_source != "toxicn_reviewer":
        return 0.0

    # ── 解析 Reviewer 输出 ──
    pred = parse_classification_output(solution_str)

    # ── 提取 ground truth ──
    if isinstance(ground_truth, dict):
        true_label = ground_truth.get("category", ground_truth.get("ground_truth", ""))
    elif isinstance(ground_truth, str):
        try:
            gt_dict = _json.loads(ground_truth)
            true_label = gt_dict.get("category", "") if isinstance(gt_dict, dict) else ground_truth
        except (ValueError, _json.JSONDecodeError):
            true_label = ground_truth
    else:
        true_label = str(ground_truth) if ground_truth else ""

    true_is_harmful = true_label in HARMFUL_CATEGORIES

    # ── 多级奖励计算 ──
    if pred["binary"] is None:
        # parse 失败 → 最严厉惩罚
        reward = -1.0
        bonus = 0.0
    else:
        reviewer_says_harmful = (pred["binary"] == "有害")
        binary_correct = (reviewer_says_harmful == true_is_harmful)

        if not binary_correct:
            # binary 错误 → 基础惩罚
            reward = -0.5
            bonus = 0.0
        elif not true_is_harmful:
            # 无毒文本正确判为无害 → 奖励（但低于 category-correct 有害）
            reward = 0.6
            bonus = 0.0
        else:
            # 有害文本，binary 正确 → 看 category 是否也对
            category_correct = (pred.get("category") == true_label)
            if category_correct:
                # binary + category 全对 → 最高奖励
                reward = 1.0
                bonus = 0.0
            else:
                # binary 对但 category 错 → 部分奖励
                # 这是关键：让 GRPO 能区分 category-correct 和 category-wrong
                reward = 0.4
                bonus = 0.0

    # 日志
    text_preview = ""
    if extra_info and isinstance(extra_info, dict):
        text_preview = extra_info.get("original_text", "")[:60]
    _logger.log_reviewer_sample(
        text_preview=text_preview, pred=pred,
        true_cat=true_label, true_is_harmful=true_is_harmful,
        base_score=reward, bonus=bonus, reward=reward,
    )
    return reward


def flush_batch_summary():
    """外部调用: 在一个 batch 的所有样本评分完成后打印汇总。"""
    _logger.log_batch_summary()
