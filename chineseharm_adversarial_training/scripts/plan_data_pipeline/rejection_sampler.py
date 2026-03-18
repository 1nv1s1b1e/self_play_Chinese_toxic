#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 3a: 拒绝采样过滤器 (Rejection Sampler)
=============================================

在 Phase 0 生成 Challenger GRPO parquet 之前，过滤低质量样本。

SSP 论文: "Filtering out invalid questions improves training stability."

v1 问题: 所有 Challenger 生成文本都进入 GRPO，包括:
  - AI 拒绝生成 ("作为AI助手，我不能...")
  - 纯重复 ("hahahaha...")
  - 太短 (<10 字)
  - 格式泄漏 ("类别：性别歧视\n...")

这些垃圾文本在 GRPO 训练中获得 reward ≈ -1 (被 quality_gate 拦截)，
但仍消耗 batch 容量和梯度计算。更重要的是，它们拉低了 group-level
baseline，导致其他正常样本的 advantage 被人为抬高，引入噪声梯度。

本模块的过滤阈值参考 challenger_reward_selfplay.py 的 _quality_gate。

用法:
  from rejection_sampler import filter_low_quality_samples
  tasks, texts = filter_low_quality_samples(tasks, generated_texts, threshold=0.3)
"""

import re
from collections import Counter


def _quality_gate(text: str) -> float:
    """与 v1 challenger_reward_selfplay.py 完全一致的质量门控"""
    text = text.strip()
    if len(text) < 5:
        return 0.0
    return (
        _length_gate(len(text))
        * _repetition_gate(text)
        * _format_gate(text)
        * _diversity_gate(text)
    )


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
        gate = _quality_gate(text.strip() if text else "")
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
