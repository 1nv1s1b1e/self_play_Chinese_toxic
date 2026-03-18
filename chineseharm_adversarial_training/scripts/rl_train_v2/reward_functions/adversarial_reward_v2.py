#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗性奖励函数 v2 — 对齐 PDF 设计
====================================
对应 PDF 图4「对抗性奖励设计」:

  判断正确:  R_challenger = 0,  R_reviewer = 2
  判断错误:  R_challenger = 2,  R_reviewer = 0
  (深度思考的 +1 信号暂不实现)

辅助奖励 (Auxiliary Rewards):
  R_reviewer_aux = (|y*| - |y|) / |y*|
    y* = 该问题最短正确答案长度
    y  = 当前答案长度
    鼓励 Reviewer 用更短的答案做出正确判断 → 提升效率

设计原则:
  1. Challenger 奖励: 只要 Reviewer 判错就得高分 → 鼓励生成边界文本
  2. Reviewer 奖励: 判对得高分 + 长度辅助 → 鼓励准确且简洁
  3. 奖励值归一化到 [-1, 1]

适配:
  - 直接替换 reward_functions/challenger_reward_selfplay.py
  - 同时提供 reviewer reward 计算
  - 支持课程学习难度加权
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# ── 常量 ────────────────────────────────────────────────────────────────────────
HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
ALL_CATEGORIES     = HARMFUL_CATEGORIES + ["无毒"]

# PDF 中的离散奖励值
REWARD_CORRECT   = 2.0   # Reviewer 判对时的奖励
REWARD_WRONG     = 0.0   # Reviewer 判错时的奖励
REWARD_NORMALIZE = 2.0   # 归一化因子，使输出在 [-1, 1]


# ── 质量门控 (复用 v1) ──────────────────────────────────────────────────────────

def _ngram_set(text: str, n: int) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def quality_gate(generated_text: str, min_length: int = 10, max_length: int = 500) -> float:
    """
    质量门控: 过滤明显低质量的生成文本
    返回 [0, 1]，1=通过，0=严重低质

    检查:
      - 长度约束
      - 非重复性
      - 中文内容占比
    """
    text = generated_text.strip()

    # 长度检查
    if len(text) < min_length:
        return 0.0
    if len(text) > max_length:
        return max(0.5, 1.0 - (len(text) - max_length) / max_length)

    # 重复度检查 (字符级 3-gram 去重率)
    chars = list(text)
    if len(chars) >= 3:
        trigrams = [text[i:i+3] for i in range(len(text) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 0
        if unique_ratio < 0.3:  # 70%+ 重复
            return 0.1
    
    # 中文占比
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if len(text) > 0 and chinese_chars / len(text) < 0.2:
        return 0.3

    return 1.0


# ── Challenger 奖励 (PDF 对齐版) ─────────────────────────────────────────────────

def compute_challenger_reward(
    generated_text: str,
    reviewer_is_correct: bool,
    verifier_is_harmful: bool,
    topic_text: str = "",
    difficulty_weight: float = 1.0,
) -> float:
    """
    计算 Challenger 单条样本奖励

    PDF 设计 (简化版，去掉深度思考):
      Reviewer 判错 → R_challenger = 2 (归一化后 = 1.0)
      Reviewer 判对 → R_challenger = 0 (归一化后 = -1.0)

    额外:
      - quality_gate 门控，低质量生成得负分
      - topic_signal 惩罚离题
      - difficulty_weight 用于课程学习加权

    Returns:
        reward ∈ [-1, 1]
    """
    gate = quality_gate(generated_text)

    # 话题相关性 (若提供了原始文本)
    topic_signal = 1.0
    if topic_text:
        gen_ngrams = _ngram_set(generated_text, 3)
        ref_ngrams = _ngram_set(topic_text, 3)
        topic_signal = _jaccard(gen_ngrams, ref_ngrams)
        topic_signal = max(0.1, min(1.0, topic_signal * 2))  # 放缩到合理范围

    # PDF 核心奖励
    if reviewer_is_correct:
        # Reviewer 判对 → Challenger 失败
        base_reward = REWARD_WRONG
    else:
        # Reviewer 判错 → Challenger 成功
        base_reward = REWARD_CORRECT

    # 归一化到 [-1, 1]
    normalized = (base_reward / REWARD_NORMALIZE) * 2 - 1  # 0→-1, 2→1

    # 应用门控和话题相关性
    reward = gate * topic_signal * normalized

    # 课程学习加权
    reward = reward * difficulty_weight

    return max(-1.0, min(1.0, reward))


def compute_challenger_rewards_batch(
    generated_texts: List[str],
    reviewer_is_correct: List[bool],
    verifier_is_harmful: List[bool],
    topic_texts: List[str] = None,
    difficulty_weights: List[float] = None,
) -> List[float]:
    """批量计算 Challenger 奖励"""
    n = len(generated_texts)
    topic_texts = topic_texts or [""] * n
    difficulty_weights = difficulty_weights or [1.0] * n

    rewards = []
    for i in range(n):
        r = compute_challenger_reward(
            generated_text=generated_texts[i],
            reviewer_is_correct=reviewer_is_correct[i],
            verifier_is_harmful=verifier_is_harmful[i],
            topic_text=topic_texts[i],
            difficulty_weight=difficulty_weights[i],
        )
        rewards.append(r)

    return rewards


# ── Reviewer 奖励 (PDF 对齐版) ──────────────────────────────────────────────────

def compute_reviewer_reward(
    reviewer_output: str,
    gold_answer: dict,
    min_correct_length: int = None,
    difficulty_weight: float = 1.0,
    length_bonus_scale: float = 0.2,
) -> float:
    """
    计算 Reviewer 单条样本奖励

    PDF 设计:
      判断正确 → R_reviewer = 2 (归一化后 = 1.0)
      判断错误 → R_reviewer = 0 (归一化后 = -1.0)

    辅助奖励 (Auxiliary):
      R_aux = (|y*| - |y|) / |y*|
        y* = 该问题最短正确答案长度
        y  = 当前答案长度
      鼓励更短的正确答案

    Args:
        reviewer_output: Reviewer 原始输出文本
        gold_answer: Gold Answer dict, 包含 is_harmful, category 等
        min_correct_length: 该问题最短正确答案长度 (用于 aux reward)
        difficulty_weight: 课程学习难度加权
        length_bonus_scale: 长度辅助奖励权重

    Returns:
        reward ∈ [-1, 1]
    """
    # 解析 Reviewer 输出
    reviewer_says_harmful = ("有害" in reviewer_output and "无害" not in reviewer_output)

    # 提取类别
    reviewer_category = "无毒"
    for line in reviewer_output.strip().split("\n"):
        line = line.strip()
        if line.startswith("类别：") or line.startswith("类别:"):
            cat = line.split("：")[-1].split(":")[-1].strip()
            if cat in HARMFUL_CATEGORIES:
                reviewer_category = cat

    if not reviewer_says_harmful:
        reviewer_category = "无毒"

    # 判断正确性
    gold_is_harmful = gold_answer.get("is_harmful", False)
    gold_category = gold_answer.get("category", "无毒")

    # 多层次正确性判断
    binary_correct = reviewer_says_harmful == gold_is_harmful
    category_correct = reviewer_category == gold_category

    if binary_correct and (not gold_is_harmful or category_correct):
        # 判对(二分类正确 + 类别正确)
        base_reward = REWARD_CORRECT
    elif binary_correct and gold_is_harmful and not category_correct:
        # 二分类正确但类别错误 → 部分奖励
        base_reward = REWARD_CORRECT * 0.5
    else:
        # 判错
        base_reward = REWARD_WRONG

    # 归一化到 [-1, 1]
    normalized = (base_reward / REWARD_NORMALIZE) * 2 - 1

    # 辅助奖励: 长度压缩
    length_bonus = 0.0
    if base_reward > 0 and min_correct_length and min_correct_length > 0:
        current_length = len(reviewer_output.strip())
        if current_length > 0:
            length_bonus = (min_correct_length - current_length) / min_correct_length
            length_bonus = max(-0.5, min(0.5, length_bonus))  # 裁剪

    reward = normalized + length_bonus_scale * length_bonus

    # 课程学习加权
    reward = reward * difficulty_weight

    return max(-1.0, min(1.0, reward))


def compute_reviewer_rewards_batch(
    reviewer_outputs: List[str],
    gold_answers: List[dict],
    min_correct_lengths: List[int] = None,
    difficulty_weights: List[float] = None,
    length_bonus_scale: float = 0.2,
) -> List[float]:
    """批量计算 Reviewer 奖励"""
    n = len(reviewer_outputs)
    min_correct_lengths = min_correct_lengths or [None] * n
    difficulty_weights = difficulty_weights or [1.0] * n

    rewards = []
    for i in range(n):
        r = compute_reviewer_reward(
            reviewer_output=reviewer_outputs[i],
            gold_answer=gold_answers[i],
            min_correct_length=min_correct_lengths[i],
            difficulty_weight=difficulty_weights[i],
            length_bonus_scale=length_bonus_scale,
        )
        rewards.append(r)

    return rewards


# ── TRL GRPO 兼容接口 ───────────────────────────────────────────────────────────

def challenger_reward_fn(completions: list, **kwargs) -> list:
    """
    TRL GRPOTrainer 兼容的 Challenger 奖励函数

    expected kwargs:
        prompts: list of dicts (含 extra_info)
    """
    rewards = []
    extra_infos = kwargs.get("extra_infos", [{}] * len(completions))

    for completion, extra_info in zip(completions, extra_infos):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        reviewer_correct = extra_info.get("reviewer_is_correct", True)
        verifier_harmful = extra_info.get("verifier_is_harmful", False)
        topic = extra_info.get("topic_text", "")
        difficulty = extra_info.get("difficulty_weight", 1.0)

        reward = compute_challenger_reward(
            generated_text=text,
            reviewer_is_correct=reviewer_correct,
            verifier_is_harmful=verifier_harmful,
            topic_text=topic,
            difficulty_weight=difficulty,
        )
        rewards.append(reward)

    return rewards


def reviewer_reward_fn(completions: list, **kwargs) -> list:
    """
    TRL GRPOTrainer 兼容的 Reviewer 奖励函数
    """
    rewards = []
    gold_answers = kwargs.get("gold_answers", [{}] * len(completions))
    difficulty_weights = kwargs.get("difficulty_weights", [1.0] * len(completions))

    for completion, gold, diff_w in zip(completions, gold_answers, difficulty_weights):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)

        reward = compute_reviewer_reward(
            reviewer_output=text,
            gold_answer=gold,
            difficulty_weight=diff_w,
        )
        rewards.append(reward)

    return rewards


# ── 统计工具 ─────────────────────────────────────────────────────────────────────

def compute_reward_stats(
    challenger_rewards: List[float],
    reviewer_rewards: List[float],
    categories: List[str] = None,
) -> Dict[str, Any]:
    """计算奖励统计信息"""
    stats = {
        "challenger": {
            "mean": sum(challenger_rewards) / len(challenger_rewards) if challenger_rewards else 0,
            "min": min(challenger_rewards) if challenger_rewards else 0,
            "max": max(challenger_rewards) if challenger_rewards else 0,
            "positive_rate": sum(1 for r in challenger_rewards if r > 0) / len(challenger_rewards) if challenger_rewards else 0,
        },
        "reviewer": {
            "mean": sum(reviewer_rewards) / len(reviewer_rewards) if reviewer_rewards else 0,
            "min": min(reviewer_rewards) if reviewer_rewards else 0,
            "max": max(reviewer_rewards) if reviewer_rewards else 0,
            "positive_rate": sum(1 for r in reviewer_rewards if r > 0) / len(reviewer_rewards) if reviewer_rewards else 0,
        },
    }

    if categories:
        cat_stats = defaultdict(lambda: {"c_rewards": [], "r_rewards": []})
        for c_r, r_r, cat in zip(challenger_rewards, reviewer_rewards, categories):
            cat_stats[cat]["c_rewards"].append(c_r)
            cat_stats[cat]["r_rewards"].append(r_r)

        stats["per_category"] = {}
        for cat, data in cat_stats.items():
            stats["per_category"][cat] = {
                "challenger_mean": sum(data["c_rewards"]) / len(data["c_rewards"]),
                "reviewer_mean": sum(data["r_rewards"]) / len(data["r_rewards"]),
                "count": len(data["c_rewards"]),
            }

    return stats
