#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger 奖励函数 — 自博弈版 (selfplay v2)
=============================================

核心改进 (v2)
--------------
旧版 (v1): reward = gate × topic_signal × (1 + 0.25 × asr)
  问题: topic_signal (n-gram Jaccard) 占 ~80% 权重, 对 GRPO 同 prompt
        的多个候选几乎无区分度, 导致策略梯度接近零

新版 (v2): reward = gate × (0.65 × adversarial + 0.35 × topic)
  对抗信号成为主导, topic_signal 降为正则化防止语义漂移

信号来源优先级:
  1. sample_r_challenger  — 逐样本 Verifier 奖励 (最强)
  2. verifier_asr × verifier_confirms_rate — 类别级统计 (退化)
  3. topic_signal — n-gram 相关性 (仅作正则)

质量门控 (gate) 不变: 长度 × 重复 × 格式 × 多样性
"""

import re
import math
from collections import Counter

# ── 与 v7 共享的 Gate 实现 ──────────────────────────────────────────

def _ngram_set(text: str, n: int) -> set:
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _pseudo_tokens(text: str) -> list:
    segs = re.split(
        r"""[\s，。！？、；：""''…——《》（）,.!?;:()\[\]{}~@#$%^&*+=/<>\\|]""",
        text,
    )
    tokens = []
    for s in segs:
        s = s.strip()
        for i in range(max(0, len(s) - 1)):
            tokens.append(s[i:i+2])
        for i in range(max(0, len(s) - 2)):
            tokens.append(s[i:i+3])
    return tokens


def _longest_common_substr_len(s1: str, s2: str) -> int:
    if not s1 or not s2:
        return 0
    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    if len(shorter) < 20:
        return 0
    win = 20
    targets = {longer[i:i+win] for i in range(len(longer) - win + 1)}
    best = 0
    for i in range(len(shorter) - win + 1):
        if shorter[i:i+win] in targets:
            cur = win
            while i + cur < len(shorter) and shorter[i:i+cur+1] in longer:
                cur += 1
            best = max(best, cur)
    return best


def _compute_topic_signal(generated: str, reference: str) -> float:
    """
    话题相关性信号: 多尺度 n-gram Jaccard × 反抄袭系数
    完全复用 v7 的计算逻辑, 是奖励函数中唯一的区分信号
    """
    if not generated or not reference:
        return 0.0

    bi   = _jaccard(_ngram_set(generated, 2), _ngram_set(reference, 2))
    tri  = _jaccard(_ngram_set(generated, 3), _ngram_set(reference, 3))
    quad = _jaccard(_ngram_set(generated, 4), _ngram_set(reference, 4))
    penta= _jaccard(_ngram_set(generated, 5), _ngram_set(reference, 5))
    ptok = _jaccard(set(_pseudo_tokens(generated)), set(_pseudo_tokens(reference)))

    raw_sim = bi*0.10 + tri*0.20 + quad*0.25 + penta*0.15 + ptok*0.30
    topic_sim = 1.0 / (1.0 + math.exp(-(raw_sim - 0.06) * 25))

    # 反抄袭
    gen_tri = _ngram_set(generated, 3)
    ref_tri = _ngram_set(reference, 3)
    if not gen_tri:
        return 0.0
    novel = len(gen_tri - ref_tri) / len(gen_tri)
    anti_copy = (0.3 + (novel / 0.3) * 0.7) if novel < 0.3 else 1.0

    # 长子串抄袭惩罚
    lcs = _longest_common_substr_len(generated, reference)
    if lcs >= 20:
        substr_factor = max(0.2, 1.0 - (lcs - 20) / 30)
        anti_copy = min(anti_copy, substr_factor)

    return min(1.0, topic_sim * anti_copy)


# ── Gate 实现 (与 v7 完全一致) ──────────────────────────────────────

def _length_gate(n: int) -> float:
    if n < 10:    return 0.0
    if n <= 15:   return 0.3 + (n - 10) / 5 * 0.3
    if n <= 30:   return 0.6 + (n - 15) / 15 * 0.3
    if n <= 300:  return 1.0
    if n <= 500:  return 1.0 - (n - 300) / 200 * 0.3
    return max(0.3, 0.7 - (n - 500) / 500 * 0.4)


def _repetition_gate(text: str) -> float:
    if len(text) < 10:
        return 1.0
    freq = Counter(text)
    if max(freq.values()) / len(text) > 0.3:
        return max(0.0, 1.0 - max(freq.values()) / len(text))
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    if bigrams and max(Counter(bigrams).values()) / len(bigrams) > 0.2:
        rep = max(Counter(bigrams).values()) / len(bigrams)
        return max(0.0, 1.0 - rep * 2)
    run, max_run = 1, 1
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i-1] else 1
        max_run = max(max_run, run)
    if max_run > 5:
        return max(0.0, 1.0 - max_run / 10)
    return 1.0


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
    ng = [text[i:i+2] for i in range(len(text) - 1)]
    if not ng:
        return 0.0
    d2 = len(set(ng)) / len(ng)
    if d2 >= 0.50: return 1.0
    if d2 >= 0.10: return (d2 - 0.10) / 0.40
    return 0.0


def _quality_gate(text: str) -> float:
    """乘法质量门控: 任一维度退化 → 整体衰减"""
    text = text.strip()
    if len(text) < 5:
        return 0.0
    return (
        _length_gate(len(text))
        * _repetition_gate(text)
        * _format_gate(text)
        * _diversity_gate(text)
    )


# ── 对抗信号 ────────────────────────────────────────────────────────

_ADVERSARIAL_WEIGHT = 0.65   # 对抗信号权重 (主导)
_TOPIC_WEIGHT       = 0.35   # 话题相关性权重 (正则化)


def _adversarial_signal(extra_info: dict) -> float:
    """
    对抗信号: 衡量 Challenger 对当前 Reviewer 的欺骗效果。

    信号优先级:
      1. sample_r_challenger — 逐样本 Verifier 奖励 (最精确)
         已归一化至 [-1, 1]，直接使用
      2. verifier_asr × verifier_confirms_rate — 类别级统计 (退化)
         映射到 [-1, 1]
      3. 默认 0.0 (无信号时)

    返回值 ∈ [-1.0, 1.0]
    """
    if not extra_info or not isinstance(extra_info, dict):
        return 0.0

    # ── 最优: 样本级 Verifier 奖励 ────────────────────────────────────
    sample_r = extra_info.get("sample_r_challenger", None)
    if sample_r is not None:
        return max(-1.0, min(1.0, float(sample_r)))

    # ── 退化: 类别级 Verifier 统计 ────────────────────────────────────
    verifier_asr      = extra_info.get("verifier_asr", -1.0)
    verifier_confirms = extra_info.get("verifier_confirms_rate", -1.0)

    if verifier_asr >= 0 and verifier_confirms >= 0:
        # [0,1] × [0,1] → [0,1] → 映射到 [-1, 1]
        return float(verifier_asr) * float(verifier_confirms) * 2.0 - 1.0

    # ── 最终退化: 旧字段 ──────────────────────────────────────────────
    cat_asr = extra_info.get("cat_adversarial_success_rate", -1.0)
    cat_lvr = extra_info.get("cat_label_verified_rate",      -1.0)

    if cat_asr >= 0 and cat_lvr >= 0:
        return float(cat_asr) * float(cat_lvr) * 2.0 - 1.0

    return 0.0


# ── 主评分函数 ──────────────────────────────────────────────────────

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """
    Challenger 奖励函数 — 自博弈版 v2

    公式 (v2):
        gate       = quality_gate(generated)
        adv_signal = adversarial_signal(extra_info)    # ∈ [-1, 1]  主信号
        topic      = topic_signal(generated, reference) # ∈ [0, 1]  正则化
        combined   = 0.65 × adv_signal + 0.35 × (topic × 2 - 1)
        reward     = gate × clamp(combined, -1, 1)

    Args:
        data_source:   必须是 "toxicn_challenger"
        solution_str:  Challenger 生成的响应文本
        ground_truth:  参考文本 (种子样本)
        extra_info:    verl 传入的 extra_info 字典

    Returns:
        float ∈ [-1.0, 1.0]
    """
    if data_source != "toxicn_challenger":
        return 0.0

    generated = solution_str.strip() if solution_str else ""

    # 参考文本: 优先从 extra_info 取, 其次 ground_truth
    reference = ""
    if extra_info and isinstance(extra_info, dict):
        reference = extra_info.get("original_text", "")
    if not reference and isinstance(ground_truth, str) and len(ground_truth) > 5:
        reference = ground_truth

    gate = _quality_gate(generated)

    # 对抗信号 (主导): 来自 Verifier 的样本级/类别级对抗评估
    adv  = _adversarial_signal(extra_info)

    # 话题信号 (正则化): n-gram 相关性，防止语义漂移
    topic = _compute_topic_signal(generated, reference)
    topic_normalized = topic * 2.0 - 1.0   # [0,1] → [-1,1]

    # 加权组合
    combined = _ADVERSARIAL_WEIGHT * adv + _TOPIC_WEIGHT * topic_normalized
    combined = max(-1.0, min(1.0, combined))

    return gate * combined
