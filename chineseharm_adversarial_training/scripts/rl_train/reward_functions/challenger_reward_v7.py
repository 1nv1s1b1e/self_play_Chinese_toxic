#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger Agent 的奖励函数 (v7: Gate × Signal 架构)
用于 verl GRPO 训练中 Challenger 的 reward 计算

══════════════════════════════════════════════════════════════════
 v6 → v7 的核心变更
══════════════════════════════════════════════════════════════════

 v6 的问题诊断:
   naturalness (Distinct-2 + 字符熵) 在合法文本上方差 ≈ 0.0003
   validity (长度 + 重复 + 格式) 在合法文本上方差 ≈ 0.0008
   → 60% 的奖励预算 (权重 0.30 + 0.30) 贡献的区分度近零
   → GRPO group-relative advantage ≈ 0.40 × (topic_i - mean_topic) + noise
   → 信噪比仅 ~40%

 v7 设计原理 (Gate × Signal 架构):

   参考文献中的攻击者奖励设计模式:

   1. SSP (arXiv:2510.18821):
      R(τ) = valid? × (1 - solver_rate)
      → 二值门控 × 变化信号; valid 不占加性预算

   2. CRT (Hong et al., ICLR 2024):
      R(y) + G(x) gibberish_penalty
      → 核心信号 R(y) + 质量门控 G(x); G(x) 是约束不是信号

   3. Perez et al. (2022):
      R(y) - β·D_KL(π||π_ref)
      → 核心信号 R(y); KL 惩罚防止退化, 不参与信号区分

   共同模式: 质量保证以 "门控/约束" 形式存在, 不与核心信号竞争加性预算。

   v7 架构:
     reward = quality_gate × topic_signal

     quality_gate ∈ [0, 1]:  乘法门控, 合法文本 ≈ 1.0
     topic_signal ∈ [0, 1]:  唯一区分信号, 100% 区分预算

   GRPO advantage:
     A_i ≈ topic_signal_i - mean(topic_signal)
     信噪比从 v6 的 ~40% 提升到接近 100%

══════════════════════════════════════════════════════════════════
 架构约束说明 (为什么不用目标模型响应作为奖励)
══════════════════════════════════════════════════════════════════

 所有 RL 红队论文 (Perez, CRT, MART, SSP) 的攻击者奖励
 都依赖目标模型的响应 (R(y) = classifier(target_response))。

 我们的分阶段架构无法实现这一点:
   Phase B (Challenger GRPO) 中:
   - verl 的 reward function 是纯 Python, 不做模型推理
   - Reviewer 模型不可用于打分
   - compute_score 逐样本调用, 无法访问同组其他 rollouts

 因此, 对抗信号来自游戏动态, 不来自奖励函数:
   ① Phase A-3: 冻结 Reviewer 验证标签忠实度 (rejection sampling)
   ② Replay Buffer: 只收录已验证样本
   ③ 交替训练: Challenger↔Reviewer 的 Stackelberg 对抗
   ④ GRPO: group-relative 在同 prompt 下放大质量差异

══════════════════════════════════════════════════════════════════
 Signal: 话题相关性 (Topic Relevance)
══════════════════════════════════════════════════════════════════

 方法: 多尺度 character n-gram Jaccard 相似度 + 反抄袭
 原理: 参考文本来自目标类别 → 相似度 = 间接类别信号
       但过高相似度 = 抄袭训练样本 → 用 novel n-gram ratio 惩罚

 文献:
   Jaccard similarity (Jaccard, 1901)
   Character n-gram for text categorization (Cavnar & Trenkle, 1994)
   N-gram novelty in abstractive summarization (See et al., 2017)

══════════════════════════════════════════════════════════════════
 Gate: 质量门控 (Quality Gate)
══════════════════════════════════════════════════════════════════

 乘法门控, 合法文本 ≈ 1.0, 退化文本快速衰减:
   length_gate:     长度合理性
   repetition_gate: 非重复退化
   format_gate:     非拒绝/非格式泄漏
   diversity_gate:  最低多样性保证 (防止极端退化)

 类比:
   SSP Section 3.2 的 format filtering (检查标签完整性和非空)
   CRT Appendix A.5 的 gibberish penalty G(x)

使用方式 (verl GRPO):
    custom_reward_function:
      path: /path/to/challenger_reward.py
      name: compute_score
"""

import re
import math
from collections import Counter


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Signal: 话题相关性 (Topic Relevance) — 唯一区分信号             ║
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

    这是 v7 架构中唯一的区分信号 (Signal 部分)

    组成:
      A) 多尺度相似度 → sigmoid 归一化 → topic_sim ∈ [0, 1]
         高 = 与参考文本话题一致 (间接类别信号)
         低 = 偏离目标类别话题域

      B) Novel trigram ratio (See et al., 2017)
         = |gen_trigrams \\ ref_trigrams| / |gen_trigrams|
         过低 (<0.3) = 近乎抄袭 → 惩罚

      C) 长子串抄袭检测
         如果生成文本包含参考文本的连续长子串 (≥20 字符),
         视为直接复制, 施加重惩罚

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

    # 高阶 n-gram 权重更高: 它们捕获更具话题区分度的模式
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
    if novel_ratio < 0.3:
        anti_copy = 0.3 + (novel_ratio / 0.3) * 0.7
    else:
        anti_copy = 1.0

    # ── C) 长子串抄袭检测 ──
    max_common_len = _longest_common_substring_length(generated, reference)
    if max_common_len >= 20:
        # 包含 ≥20 字符的连续共有子串 → 重惩罚
        substr_penalty = max(0.2, 1.0 - (max_common_len - 20) / 30)
        anti_copy = min(anti_copy, substr_penalty)

    return min(1.0, topic_sim * anti_copy)


def _longest_common_substring_length(s1: str, s2: str) -> int:
    """
    计算两个字符串的最长公共子串长度
    使用固定窗口检测 + 扩展, 避免 O(n*m) 的 DP 开销
    """
    if not s1 or not s2:
        return 0

    shorter, longer = (s1, s2) if len(s1) <= len(s2) else (s2, s1)
    if len(shorter) < 20:
        return 0  # 不可能有 ≥20 的公共子串

    max_len = 0
    # 只检查 ≥20 长度的子串, 使用集合加速
    window = 20
    target_substrings = {longer[i:i+window] for i in range(len(longer) - window + 1)}

    for i in range(len(shorter) - window + 1):
        if shorter[i:i+window] in target_substrings:
            # 找到了长度为 window 的匹配, 尝试扩展
            cur_len = window
            while (i + cur_len < len(shorter) and
                   shorter[i:i+cur_len+1] in longer):
                cur_len += 1
            max_len = max(max_len, cur_len)

    return max_len


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Gate: 质量门控 (Quality Gate) — 乘法门控, 不占区分预算          ║
# ║  类比: SSP format filtering, CRT gibberish penalty               ║
# ╚════════════════════════════════════════════════════════════════════╝

def _length_gate(n: int) -> float:
    """
    长度门控: 合理范围内 → 1.0, 否则衰减

    设计理念 (Gate 思维, 不是 Score 思维):
      合法文本的长度几乎都在 15-400 范围 → gate = 1.0
      只有退化输出才会触发衰减 → gate < 1.0

    曲线:
      < 10 字:  0.0     (退化: 空输出)
      10-15:    0.3→0.6 (过短, 可能退化)
      15-30:    0.6→0.9 (偏短但可接受)
      30-300:   1.0     (理想范围, gate 全开)
      300-500:  1.0→0.7 (偏长, 轻微衰减)
      > 500:    0.7→0.3 (过长, 疑似退化)

    Returns: 0.0-1.0
    """
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
    """
    重复门控: 无重复 → 1.0, 重复严重 → 衰减到 0

    设计理念:
      rep_score < 0.1:  1.0   (正常文本, 绝大多数 rollouts)
      rep_score 0.1-0.5: 线性衰减 1.0 → 0.3
      rep_score > 0.5:  0.3 → 0.0  (严重退化)
    """
    rep = _repetition_score(text)
    if rep < 0.1:
        return 1.0
    elif rep < 0.5:
        return 1.0 - (rep - 0.1) / 0.4 * 0.7
    else:
        return max(0.0, 0.3 - (rep - 0.5) / 0.5 * 0.3)


def _format_gate(text: str) -> float:
    """
    格式合规门控 (类比 SSP 的 format filtering)

    检测模型的"输出格式"是否正确:
      · 模型应直接输出目标文本, 不应拒绝/解释/列格式
      · 类比 SSP: 检查 <question></question> 标签完整性
      · 类比 CRT: G(x) gibberish penalty

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


def _diversity_gate(text: str) -> float:
    """
    最低多样性门控: Distinct-2 过低表示极端退化

    设计理念 (Gate, 不是 Score):
      Distinct-2 ≥ 0.50:  gate = 1.0 (几乎所有合法文本)
      Distinct-2 < 0.50:  线性衰减 (退化文本)
      Distinct-2 < 0.10:  gate = 0.0 (严重退化)

    注意: 这是 gate 不是 signal
      合法中文文本的 Distinct-2 ≈ 0.75-0.95, 全部 → 1.0
      只有退化文本 (重复公式化输出) 才会低于 0.50
      因此这个 gate 不会在正常 rollouts 间产生区分度
      — 这正是 Gate × Signal 架构的设计目标
    """
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
    """
    质量门控: 4 个子门控的乘积

    乘法结构意味着任一维度退化都会压低整体 gate:
      正常文本: 1.0 × 1.0 × 1.0 × 1.0 = 1.0
      长度退化: 0.3 × 1.0 × 1.0 × 1.0 = 0.3
      重复退化: 1.0 × 0.2 × 1.0 × 1.0 = 0.2
      格式违规: 1.0 × 1.0 × 0.0 × 1.0 = 0.0
      多重退化: 0.3 × 0.2 × 1.0 × 1.0 = 0.06

    Returns: 0.0-1.0 (正常文本 ≈ 1.0)
    """
    text = text.strip()
    if len(text) < 5:
        return 0.0

    g_length = _length_gate(len(text))
    g_repetition = _repetition_gate(text)
    g_format = _format_gate(text)
    g_diversity = _diversity_gate(text)

    return g_length * g_repetition * g_format * g_diversity


# ╔════════════════════════════════════════════════════════════════════╗
# ║  主评分函数: Gate × Signal                                       ║
# ╚════════════════════════════════════════════════════════════════════╝

def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """
    Challenger 奖励函数 (v7: Gate × Signal 架构)

    核心公式:
      reward = quality_gate × topic_signal

    与 v6 的区别:
      v6: score = 0.40×topic + 0.30×natural + 0.30×valid  (加性)
      v7: score = gate × topic                             (乘法)

      v6 中, naturalness 和 validity 占 60% 但方差近零 → 信号浪费
      v7 中, gate 在合法文本上 ≈ 1.0 → 不干扰区分度
                topic_signal 获得 100% 的区分预算

    对抗信号来源:
      不在此函数中 — 来自游戏动态:
      ① Phase A-3: 冻结 Reviewer 验证 (rejection sampling)
      ② Replay Buffer: 已验证样本复用
      ③ 交替训练: 产生 Stackelberg 对抗动态
      ④ GRPO: group-relative 组内放大差异

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

    # ── 提取参考文本 ──
    reference = ""
    if extra_info and isinstance(extra_info, dict):
        reference = extra_info.get("original_text", "")
    if not reference and isinstance(ground_truth, str) and len(ground_truth) > 5:
        reference = ground_truth

    # ── Gate: 质量门控 ──
    gate = compute_quality_gate(generated)

    # ── Signal: 话题相关性 ──
    signal = compute_topic_relevance(generated, reference)

    # ── Gate × Signal ──
    total = gate * signal

    # ── 映射到 [-1.0, 1.0] ──
    return total * 2.0 - 1.0
