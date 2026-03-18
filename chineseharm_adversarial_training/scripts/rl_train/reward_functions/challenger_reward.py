#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger Agent 奖励函数 (v11: 真正对抗信号)

架构（基于 Stackelberg 博弈设计图）:
  Verifier 根据以下两个维度计算 R_challenger:
    ① label_verified:       生成文本是否真正属于目标类别
                            （冻结 reviewer_v0 在 Phase A-3 预计算）
    ② adversarial_success:  是否骗过了当前 Reviewer_N-1
                            （当前 Reviewer 在 Phase A-2 预计算）

信号来源 (从 extra_info 读取，由 Phase A 预计算注入):
  extra_info["label_verified"]      : bool / None
  extra_info["adversarial_success"] : bool / None

Reward 设计:
  退化文本 (gate < 0.1)             → -1.0
  标签漂移 (label_verified=False)   → -1.0
  标签正确 + 骗过 Reviewer          → gate × 2.0 - 1.0  ∈ (+0.0, +1.0)
  标签正确 + 未骗过 Reviewer        → -0.3
  无预计算信号 (fallback)           → gate × anti_copy × 2.0 - 1.0
"""

import os
import re
import math
from collections import Counter

try:
    from reward_functions.reward_logger import RewardLogger
except ImportError:
    from reward_logger import RewardLogger

# 模块级单例 Logger（每个工作进程共享一个）
_logger = RewardLogger("challenger")


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Gate: 质量门控                                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

def _length_gate(n: int) -> float:
    if n < 10:
        return 0.0
    elif n <= 15:
        return 0.3 + (n - 10) / 5 * 0.3
    elif n <= 30:
        return 0.6 + (n - 15) / 15 * 0.3
    elif n <= 300:
        return 1.0
    elif n <= 500:
        return 1.0 - (n - 300) / 200 * 0.3
    else:
        return max(0.3, 0.7 - (n - 500) / 500 * 0.4)


def _repetition_score(text: str) -> float:
    if len(text) < 10:
        return 0.0
    freq = Counter(text)
    if max(freq.values()) / len(text) > 0.3:
        return min(1.0, max(freq.values()) / len(text) * 2)
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    if bigrams:
        bg_max = max(Counter(bigrams).values()) / len(bigrams)
        if bg_max > 0.2:
            return min(1.0, bg_max * 3)
    if len(text) >= 4:
        fg = [text[i:i+4] for i in range(len(text) - 3)]
        fg_max = max(Counter(fg).values()) / len(fg)
        if fg_max > 0.15:
            return min(1.0, fg_max * 4)
    run, max_run = 1, 1
    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    if max_run > 5:
        return min(1.0, max_run / 10)
    return 0.0


def _repetition_gate(text: str) -> float:
    rep = _repetition_score(text)
    if rep < 0.1:
        return 1.0
    elif rep < 0.5:
        return 1.0 - (rep - 0.1) / 0.4 * 0.7
    else:
        return max(0.0, 0.3 - (rep - 0.5) / 0.5 * 0.3)


def _format_gate(text: str) -> float:
    patterns = [
        r"作为.{0,5}(AI|语言模型|助手)",
        r"我不能|我无法|抱歉|sorry|对不起",
        r"以下是|如下所示|示例如下",
        r"^类别[：:]|^标签[：:]|^表达方式",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return 0.0
    return 1.0


def _diversity_gate(text: str) -> float:
    if len(text) < 5:
        return 0.0
    ngrams = [text[i:i+2] for i in range(len(text) - 1)]
    if not ngrams:
        return 0.0
    d2 = len(set(ngrams)) / len(ngrams)
    if d2 >= 0.50:
        return 1.0
    elif d2 >= 0.10:
        return (d2 - 0.10) / 0.40
    else:
        return 0.0


def compute_quality_gate(text: str) -> float:
    """质量门控: 4 个子门控的乘积，返回 [0, 1]"""
    text = text.strip()
    if len(text) < 5:
        return 0.0
    return (_length_gate(len(text))
            * _repetition_gate(text)
            * _format_gate(text)
            * _diversity_gate(text))


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Anti-Copy Factor                                                ║
# ╚════════════════════════════════════════════════════════════════════╝

def _ngram_set(text: str, n: int) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def compute_anti_copy(generated: str, reference: str) -> float:
    """反抄袭因子: novel trigram ratio"""
    if not generated or not reference:
        return 1.0
    gen_trigrams = _ngram_set(generated, 3)
    ref_trigrams = _ngram_set(reference, 3)
    if len(gen_trigrams) == 0:
        return 1.0
    novel_ratio = len(gen_trigrams - ref_trigrams) / len(gen_trigrams)
    if novel_ratio < 0.3:
        return 0.3 + (novel_ratio / 0.3) * 0.7
    return 1.0


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Fallback: n-gram Topic Relevance（无预计算信号时使用）           ║
# ╚════════════════════════════════════════════════════════════════════╝


def _pseudo_tokens(text: str) -> list:
    # 按空白和常见标点切分，生成 bigram/trigram 伪词元
    # 用 re.sub 先将各类标点统一替换为空格再 split，避免复杂字符类
    punct_pattern = re.compile(
        "[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"  # ASCII punct
        "\u3000-\u303F"   # CJK 标点
        "\uFF00-\uFFEF"   # 全角符号
        "\u2000-\u206F"   # 常规标点
        "\u2014\u2026"    # 破折号 省略号
        "]+"
    )
    segs = punct_pattern.sub(" ", text).split()
    tokens = []
    for s in segs:
        for i in range(max(0, len(s) - 1)):
            tokens.append(s[i:i+2])
        for i in range(max(0, len(s) - 2)):
            tokens.append(s[i:i+3])
    return tokens

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def compute_topic_relevance(generated: str, reference: str) -> float:
    """Fallback: 多尺度 n-gram Jaccard 话题相关性"""
    if not generated or not reference:
        return 0.0
    bi   = _jaccard(_ngram_set(generated, 2), _ngram_set(reference, 2))
    tri  = _jaccard(_ngram_set(generated, 3), _ngram_set(reference, 3))
    quad = _jaccard(_ngram_set(generated, 4), _ngram_set(reference, 4))
    penta= _jaccard(_ngram_set(generated, 5), _ngram_set(reference, 5))
    gen_ptokens = set(_pseudo_tokens(generated))
    ref_ptokens = set(_pseudo_tokens(reference))
    ptok = _jaccard(gen_ptokens, ref_ptokens)
    raw_sim = bi * 0.10 + tri * 0.20 + quad * 0.25 + penta * 0.15 + ptok * 0.30
    topic_sim = 1.0 / (1.0 + math.exp(-(raw_sim - 0.06) * 25))
    # 反抄袭
    gen_trigrams = _ngram_set(generated, 3)
    ref_trigrams = _ngram_set(reference, 3)
    if len(gen_trigrams) == 0:
        return 0.0
    novel_ratio = len(gen_trigrams - ref_trigrams) / len(gen_trigrams)
    anti_copy = 1.0 if novel_ratio >= 0.3 else 0.3 + (novel_ratio / 0.3) * 0.7
    return min(1.0, topic_sim * anti_copy)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  主评分函数 (v11)                                                ║
# ╚════════════════════════════════════════════════════════════════════╝



# ╔════════════════════════════════════════════════════════════════════╗
# ║  主评分函数 (v11)                                                ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Challenger 奖励函数 (v11)
    # 日志控制: REWARD_DEBUG=0(默认,批次汇总) / 1(逐样本) / 2(+JSONL落盘)
    if data_source != "toxicn_challenger":
        return 0.0

    generated = solution_str.strip() if solution_str else ""

    reference = ""
    label_verified = None
    adversarial_success = None
    category = ""

    if extra_info and isinstance(extra_info, dict):
        reference           = extra_info.get("original_text", "")
        label_verified      = extra_info.get("label_verified")
        adversarial_success = extra_info.get("adversarial_success")
        category            = extra_info.get("category", "")

    if not reference and isinstance(ground_truth, str) and len(ground_truth) > 5:
        reference = ground_truth

    # Step 1: 质量门控
    gate = compute_quality_gate(generated)
    if gate < 0.1:
        _logger.log_challenger_sample(
            generated=generated, category=category, gate=gate,
            label_verified=label_verified, adv_success=adversarial_success,
            topic_sim=0.0, reward=-1.0, signal_source="gate_fail",
        )
        _logger.log_batch_summary()
        return -1.0

    # Step 2: Phase A 双信号 (label_verified + adversarial_success)
    if label_verified is not None and adversarial_success is not None:
        if not label_verified:
            reward = -1.0                  # 标签漂移
        elif adversarial_success:
            reward = gate * 2.0 - 1.0     # 骗过 Reviewer → 正奖励
        else:
            reward = -0.3                  # 标签正确但未骗过 → 小惩罚
        _logger.log_challenger_sample(
            generated=generated, category=category, gate=gate,
            label_verified=label_verified, adv_success=adversarial_success,
            topic_sim=0.0, reward=reward, signal_source="phase_a",
        )
        _logger.log_batch_summary()
        return reward

    # Step 2b: 只有 label_verified，无 adversarial_success（部分信号）
    if label_verified is not None:
        if not label_verified:
            reward = -1.0
        else:
            anti_copy = compute_anti_copy(generated, reference)
            reward = gate * anti_copy * 2.0 - 1.0
        _logger.log_challenger_sample(
            generated=generated, category=category, gate=gate,
            label_verified=label_verified, adv_success=None,
            topic_sim=0.0, reward=reward, signal_source="partial",
        )
        _logger.log_batch_summary()
        return reward

    # Step 3: Fallback — n-gram topic_relevance
    topic_sim = compute_topic_relevance(generated, reference)
    reward = gate * topic_sim * 2.0 - 1.0
    _logger.log_challenger_sample(
        generated=generated, category=category, gate=gate,
        label_verified=None, adv_success=None,
        topic_sim=topic_sim, reward=reward, signal_source="fallback",
    )
    _logger.log_batch_summary()
    return reward
