#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger Agent 的奖励函数 (v6: NLG 评估方法驱动, 零硬规则, 无 RAG 依赖)
用于 verl GRPO 训练中 Challenger 的 reward 计算

══════════════════════════════════════════════════════════════════
 v5.1 存在的问题及 v6 的改进
══════════════════════════════════════════════════════════════════

 问题 1: BM25 RAG 检索的是 SFT 训练数据 — 数据复用, 信号冗余
   Challenger 已在该数据上做过 SFT 微调, 再用同一份数据做 BM25 检索
   相当于: 鼓励模型复制训练样本 → 正反馈闭环 → 抑制生成多样性
   → v6: 完全移除 BM25 RAG, 类别忠实度由外部机制保证

 问题 2: "标点多样性"/"句长变化" 等指标缺乏说服力
   这些指标没有明确的 NLP 文献支撑, 且部分维度存在重叠
   → v6: 每个维度均有 NLG 评估文献依据

══════════════════════════════════════════════════════════════════
 三个评分维度 (每个有明确文献支撑)
══════════════════════════════════════════════════════════════════

 ① 话题相关性 (Topic Relevance) — 权重 0.40
    方法: 多尺度 character n-gram Jaccard 相似度 + 反抄袭
    原理: 参考文本来自目标类别 → 相似度 = 间接类别信号
          但过高相似度 = 抄袭训练样本 → 用 novel n-gram ratio 惩罚
    文献: Jaccard similarity (Jaccard, 1901)
          Character n-gram for text categorization (Cavnar & Trenkle, 1994)
          N-gram novelty in abstractive summarization (See et al., 2017)

 ② 文本自然度 (Naturalness) — 权重 0.30
    方法: Distinct-2 + 字符分布归一化熵
    原理: Distinct-n 是 NLG 评估的标准多样性指标
          Shannon 熵衡量字符分布的均匀性
    文献: "A Diversity-Promoting Objective Function" (Li et al., NAACL 2016)
          "A Mathematical Theory of Communication" (Shannon, 1948)

 ③ 结构合法性 (Structural Validity) — 权重 0.30
    方法: 长度评分 + 多尺度重复检测 + 格式合规检测
    原理: 结构层面的质量门控, 类比 SSP 的 format filtering
          SSP 检查 <question></question> 标签完整性和非空
          我们检查: 长度合理性 / 非重复 / 非拒绝
    文献: Search Self-Play (arXiv:2510.18821) Section 3.2

 类别忠实度保证 (不在 reward 内, 由外部机制负责):
   ① Phase A-3: 冻结 Reviewer 模型验证 (rejection sampling)
   ② Replay Buffer: 只收录已验证样本
   ③ Prompt: 明确包含类别 + 表达方式描述
   ④ GRPO: group-relative 在同 prompt 下放大质量差异

使用方式 (verl GRPO):
    custom_reward_function:
      path: /path/to/challenger_reward.py
      name: compute_score
"""

import re
import math
from collections import Counter


# ╔════════════════════════════════════════════════════════════════════╗
# ║  维度 ①: 话题相关性 (Topic Relevance)  — 权重 0.40              ║
# ║  文献: Jaccard (1901), Cavnar & Trenkle (1994), See et al. (2017)║
# ╚════════════════════════════════════════════════════════════════════╝

def _ngram_set(text: str, n: int) -> set:
    """提取 character n-gram 集合"""
    return {text[i:i+n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient"""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def _pseudo_tokens(text: str) -> list:
    """按标点分段后取 2/3-char 窗口, 近似词级别单元"""
    segs = re.split(
        r"""[\s，。！？、；：""''…——《》（）,.!?;:()\[\]{}~@#$%^&*+=/<>\\|]""",
        text
    )
    tokens = []
    for s in segs:
        s = s.strip()
        for i in range(max(0, len(s) - 1)):
            tokens.append(s[i:i+2])
        for i in range(max(0, len(s) - 2)):
            tokens.append(s[i:i+3])
    return tokens


def compute_topic_relevance(generated: str, reference: str) -> float:
    """
    话题相关性: 多尺度 n-gram Jaccard 相似度 + 反抄袭调节

    组成:
      A) 多尺度相似度 → sigmoid 归一化 → topic_sim ∈ [0, 1]
         高 = 与参考文本话题一致 (间接类别信号)
         低 = 偏离目标类别话题域

      B) Novel trigram ratio (See et al., 2017)
         = |gen_trigrams \\ ref_trigrams| / |gen_trigrams|
         衡量生成内容中有多少是"新的" (不在参考文本中出现)
         过低 (<0.3) = 近乎抄袭 → 惩罚
         适中 (0.3-0.9) = 话题相关但有新内容 → 不惩罚

      最终: topic_sim × anti_copy_factor

    Args:
        generated: 模型生成的文本
        reference: 来自同类别的参考文本

    Returns: 0.0-1.0
    """
    if not generated or not reference:
        return 0.0

    # ── A) 多尺度 n-gram Jaccard ──
    bi = _jaccard(_ngram_set(generated, 2), _ngram_set(reference, 2))
    tri = _jaccard(_ngram_set(generated, 3), _ngram_set(reference, 3))
    quad = _jaccard(_ngram_set(generated, 4), _ngram_set(reference, 4))
    penta = _jaccard(_ngram_set(generated, 5), _ngram_set(reference, 5))

    gen_ptokens = set(_pseudo_tokens(generated))
    ref_ptokens = set(_pseudo_tokens(reference))
    ptok = _jaccard(gen_ptokens, ref_ptokens)

    raw_sim = bi * 0.10 + tri * 0.20 + quad * 0.25 + penta * 0.15 + ptok * 0.30

    # Sigmoid: raw_sim ∈ [0, ~0.3] → [0, 1]
    # center=0.06 (典型同话题文本的中位相似度), steepness=25
    topic_sim = 1.0 / (1.0 + math.exp(-(raw_sim - 0.06) * 25))

    # ── B) Novel trigram ratio (反抄袭) ──
    gen_trigrams = _ngram_set(generated, 3)
    ref_trigrams = _ngram_set(reference, 3)

    if len(gen_trigrams) == 0:
        return 0.0

    novel_ratio = len(gen_trigrams - ref_trigrams) / len(gen_trigrams)

    # novel_ratio < 0.3: 70%+ 的 trigram 和参考文本重复 → 近乎抄袭
    # 惩罚: 从 0.3 线性递减到 0.3 倍 (novel_ratio=0 时)
    if novel_ratio < 0.3:
        anti_copy = 0.3 + (novel_ratio / 0.3) * 0.7
    else:
        anti_copy = 1.0

    return min(1.0, topic_sim * anti_copy)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  维度 ②: 文本自然度 (Naturalness)  — 权重 0.30                  ║
# ║  文献: Li et al. (NAACL 2016), Shannon (1948)                    ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_distinct_n(text: str, n: int = 2) -> float:
    """
    Distinct-n (Li et al., NAACL 2016)

    定义: |unique n-grams| / |total n-grams|
    含义: 生成文本中有多少比例的 n-gram 是唯一的
    高值 = 多样性好, 表达丰富; 低值 = 重复/公式化

    典型值:
      · 真实中文互联网文本: Distinct-2 ≈ 0.75-0.95
      · 退化重复文本:       Distinct-2 ≈ 0.10-0.40
      · 随机乱码:          Distinct-2 ≈ 0.95-1.00

    Args:
        text: 输入文本
        n: n-gram 阶数 (默认 bigram)

    Returns: 0.0-1.0
    """
    if len(text) < n + 1:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def compute_char_entropy(text: str) -> float:
    """
    归一化字符 Shannon 熵 (Shannon, 1948)

    公式: H(X) = -Σ p(c_i) · log₂(p(c_i))
    归一化: H_norm = H(X) / log₂(|unique_chars|) ∈ [0, 1]

    含义:
      · 1.0 = 字符分布完全均匀 (最大信息量)
      · 0.0 = 所有字符相同 (零信息量)

    典型值:
      · 真实中文文本: H_norm ≈ 0.85-0.98
      · 重复堆叠文本: H_norm ≈ 0.30-0.60

    Args:
        text: 输入文本

    Returns: 0.0-1.0
    """
    if len(text) < 5:
        return 0.0

    freq = Counter(text)
    n = len(text)
    n_unique = len(freq)

    if n_unique <= 1:
        return 0.0

    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values())
    max_entropy = math.log2(n_unique)

    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_naturalness(text: str) -> float:
    """
    文本自然度 (综合 Distinct-2 + 字符熵)

    Distinct-2: 局部多样性 (bigram 级别的重复程度)
    字符熵:     全局分布均匀性 (整体字符分布的信息量)

    两者互补:
      · Distinct-2 高 + 熵高 → 自然、多样
      · Distinct-2 低 + 熵低 → 退化、重复
      · Distinct-2 高 + 熵低 → 少量字符的多种组合 (罕见)

    权重: Distinct-2 (0.55) + 字符熵 (0.45)
    Distinct-2 权重稍高: 它对退化检测更敏感 (Li et al., 2016)

    Returns: 0.0-1.0
    """
    if len(text) < 10:
        return 0.0

    d2 = compute_distinct_n(text, n=2)
    ent = compute_char_entropy(text)

    return d2 * 0.55 + ent * 0.45


# ╔════════════════════════════════════════════════════════════════════╗
# ║  维度 ③: 结构合法性 (Structural Validity)  — 权重 0.30          ║
# ║  类比 SSP Section 3.2 的 format filtering                       ║
# ╚════════════════════════════════════════════════════════════════════╝

def _length_score(n: int) -> float:
    """
    长度评分: 分段线性, 15-300 字最佳

    曲线设计:
      < 15 字:  0.0     (过短, 无效)
      15-30:    0.2→0.7 (偏短但可接受)
      30-200:   0.7→1.0 (理想范围)
      200-400:  1.0→0.7 (偏长, 轻微惩罚)
      > 400:    0.7→0.3 (过长, 较大惩罚)

    Returns: 0.0-1.0
    """
    if n < 15:
        return 0.0
    elif n <= 30:
        return 0.2 + (n - 15) / 15 * 0.5
    elif n <= 200:
        return 0.7 + (n - 30) / 170 * 0.3
    elif n <= 400:
        return 1.0 - (n - 200) / 200 * 0.3
    else:
        return max(0.3, 0.7 - (n - 400) / 600 * 0.4)


def check_repetition(text: str) -> float:
    """
    多尺度重复检测

    四个粒度:
      1. 单字频率 > 30%: 某字符主导全文
      2. Bigram 最高频率 > 20%: 短模式循环
      3. 4-gram 最高频率 > 15%: 长模式循环
      4. 连续重复字符 > 5: 直接重复 (如 "啊啊啊啊啊啊")

    Returns: 0.0(无重复) ~ 1.0(严重重复)
    """
    if len(text) < 10:
        return 0.0

    # 1) 单字频率
    freq = Counter(text)
    if max(freq.values()) / len(text) > 0.3:
        return min(1.0, max(freq.values()) / len(text) * 2)

    # 2) bigram 重复
    bigrams = [text[i:i+2] for i in range(len(text) - 1)]
    if bigrams:
        bg_max = max(Counter(bigrams).values()) / len(bigrams)
        if bg_max > 0.2:
            return min(1.0, bg_max * 3)

    # 3) 4-gram 重复
    if len(text) >= 4:
        fg = [text[i:i+4] for i in range(len(text) - 3)]
        fg_max = max(Counter(fg).values()) / len(fg)
        if fg_max > 0.15:
            return min(1.0, fg_max * 4)

    # 4) 连续重复字符
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


def _format_compliance(text: str) -> float:
    """
    格式合规检测 (类比 SSP 的 format filtering)

    检测模型的"输出格式"是否正确, 非内容规则:
      · 模型应直接输出目标文本, 不应拒绝/解释/列格式
      · 类比 SSP: 检查 <question></question> 标签完整性

    被检测的异常:
      · AI 身份暴露 (拒绝回答)
      · 说明性前缀 (非直接输出)
      · Prompt 格式泄漏 (复述类别/标签)

    Returns: 1.0(合规) / 0.0(违规)
    """
    patterns = [
        r"作为.{0,5}(AI|语言模型|助手)",      # AI 身份暴露
        r"我不能|我无法|抱歉|sorry|对不起",    # 拒绝生成
        r"以下是|如下所示|示例如下",            # 说明性前缀
        r"^类别[：:]|^标签[：:]|^表达方式",    # Prompt 格式泄漏
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return 0.0
    return 1.0


def compute_validity(text: str) -> float:
    """
    结构合法性: 长度(0.40) + 重复(0.35) + 格式合规(0.25)

    设计:
      · 长度: 连续评分, 分段线性
      · 重复: 连续惩罚, 1 - repetition_score
      · 格式: 二值门控, 违规则该维度被压缩

    Returns: 0.0-1.0
    """
    text = text.strip()
    if len(text) < 10:
        return 0.0

    length = _length_score(len(text))
    rep = 1.0 - check_repetition(text)
    fmt = _format_compliance(text)

    # 格式违规是"硬错误": 直接将结果压缩到 0.2
    if fmt < 0.5:
        return (length * 0.40 + rep * 0.35 + 0.0) * 0.2

    return length * 0.40 + rep * 0.35 + fmt * 0.25


# ╔════════════════════════════════════════════════════════════════════╗
# ║  主评分函数                                                       ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Challenger 奖励函数 (v6: NLG 评估方法驱动)

    三个维度:
      话题相关性 (topic)      0.40  — n-gram Jaccard + 反抄袭
      文本自然度 (naturalness) 0.30  — Distinct-2 + 字符熵
      结构合法性 (validity)    0.30  — 长度 + 重复 + 格式合规

    类别忠实度由外部机制保证:
      ① Phase A-3: 冻结 Reviewer 验证 (rejection sampling)
      ② Replay Buffer: 已验证样本复用
      ③ Prompt: 类别 + 表达方式描述
      ④ GRPO: group-relative 组内放大质量差异

    Args:
        data_source: "toxicn_challenger"
        solution_str: 模型生成的文本
        ground_truth: 参考文本 (备用)
        extra_info: {category, expression, original_text, ...}

    Returns:
        float: [-1.0, 1.0]
    """
    if data_source != "toxicn_challenger":
        return 0.0

    generated = solution_str.strip() if solution_str else ""

    # 提取参考文本
    reference = ""
    if extra_info and isinstance(extra_info, dict):
        reference = extra_info.get("original_text", "")
    if not reference and isinstance(ground_truth, str) and len(ground_truth) > 5:
        reference = ground_truth

    # ── 维度 ①: 话题相关性 ──
    topic = compute_topic_relevance(generated, reference)

    # ── 维度 ②: 文本自然度 ──
    natural = compute_naturalness(generated)

    # ── 维度 ③: 结构合法性 ──
    valid = compute_validity(generated)

    # ── 加权总分 ──
    total = topic * 0.40 + natural * 0.30 + valid * 0.30

    # ── 严重重复保护 (SSP quality gate) ──
    if check_repetition(generated) > 0.5:
        total *= 0.2

    # ── 映射到 [-1.0, 1.0] ──
    return total * 2.0 - 1.0
