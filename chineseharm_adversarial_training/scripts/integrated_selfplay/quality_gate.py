#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
质量门控公共模块 (统一版)
=========================
将 v1 / Plan 1 / Plan 3 中重复出现 4 次的 quality_gate 代码抽取为单一模块，
所有 reward 函数和 rejection_sampler 统一从此处导入。
"""

import re
from collections import Counter


def length_gate(n: int) -> float:
    """文本长度门控：太短/太长均衰减。"""
    if n < 10:    return 0.0
    if n <= 15:   return 0.3 + (n - 10) / 5 * 0.3
    if n <= 30:   return 0.6 + (n - 15) / 15 * 0.3
    if n <= 300:  return 1.0
    if n <= 500:  return 1.0 - (n - 300) / 200 * 0.3
    return max(0.3, 0.7 - (n - 500) / 500 * 0.4)


def repetition_gate(text: str) -> float:
    """文本重复检测门控。"""
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


def format_gate(text: str) -> float:
    """格式泄漏检测门控：AI 拒绝/模板泄漏直接置零。"""
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


def diversity_gate(text: str) -> float:
    """词汇多样性门控。"""
    if len(text) < 5:
        return 0.0
    ng = [text[i:i+2] for i in range(len(text) - 1)]
    if not ng:
        return 0.0
    d2 = len(set(ng)) / len(ng)
    if d2 >= 0.50: return 1.0
    if d2 >= 0.10: return (d2 - 0.10) / 0.40
    return 0.0


def quality_gate(text: str) -> float:
    """
    乘法质量门控: 任一维度退化 → 整体衰减。
    
    这是所有 reward 函数和 rejection_sampler 的核心质量评估函数。
    """
    text = text.strip()
    if len(text) < 5:
        return 0.0
    return (
        length_gate(len(text))
        * repetition_gate(text)
        * format_gate(text)
        * diversity_gate(text)
    )
