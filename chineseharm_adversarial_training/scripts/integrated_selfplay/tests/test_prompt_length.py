#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 3: Prompt 长度对比 — 验证精简版 prompt 更短
================================================
精简版 REVIEWER_USER_TEMPLATE 去除了内联 RULES (~500 tokens),
只保留格式指引. 此测试量化 prompt token 数量差异.

用法:
  python tests/test_prompt_length.py [--model_path /path/to/model]

无 model_path 时按字符数估算, 有 model_path 时用真实 tokenizer.
"""

import os, sys, argparse

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)

from constants import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    REVIEWER_USER_TEMPLATE_FULL,
    RULES,
)


def test_prompt_length(model_path=None):
    print("=" * 70)
    print("测试 3: Prompt 长度对比 (精简版 vs 完整版)")
    print("=" * 70)

    sample_text = "女司机就是马路杀手，这不是歧视，这是事实"

    short_prompt = REVIEWER_USER_TEMPLATE.format(text=sample_text)
    full_prompt = REVIEWER_USER_TEMPLATE_FULL.format(text=sample_text)

    print(f"\n   RULES 长度: {len(RULES)} 字符")
    print(f"\n   精简版 user prompt ({len(short_prompt)} chars):")
    print(f"   ---")
    for line in short_prompt.split('\n'):
        print(f"   {line}")
    print(f"   ---")

    print(f"\n   完整版 user prompt ({len(full_prompt)} chars):")
    print(f"   --- (前200字)")
    for line in full_prompt[:200].split('\n'):
        print(f"   {line}")
    print(f"   ... (共 {len(full_prompt)} 字符)")
    print(f"   ---")

    ratio = len(short_prompt) / len(full_prompt)
    print(f"\n   精简版 / 完整版 = {ratio:.1%}")
    print(f"   节省 {len(full_prompt) - len(short_prompt)} 字符 ({1-ratio:.0%})")

    if model_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            short_tokens = len(tok.encode(short_prompt))
            full_tokens = len(tok.encode(full_prompt))
            system_tokens = len(tok.encode(REVIEWER_SYSTEM_PROMPT))

            print(f"\n   Tokenizer 精确计数 ({model_path.split('/')[-1]}):")
            print(f"     system prompt: {system_tokens} tokens")
            print(f"     精简版 user:   {short_tokens} tokens")
            print(f"     完整版 user:   {full_tokens} tokens")
            print(f"     总 prompt (精简): {system_tokens + short_tokens} tokens")
            print(f"     总 prompt (完整): {system_tokens + full_tokens} tokens")
            print(f"     节省: {full_tokens - short_tokens} tokens ({(full_tokens-short_tokens)/full_tokens:.0%})")

            # 对于 3B 模型, max_length=1024, 看 prompt 占比
            max_len = 1024
            short_total = system_tokens + short_tokens
            full_total = system_tokens + full_tokens
            print(f"\n     在 max_length={max_len} 下:")
            print(f"     精简版占 {short_total/max_len:.0%} — 留给文本+生成空间: {max_len-short_total} tokens")
            print(f"     完整版占 {full_total/max_len:.0%} — 留给文本+生成空间: {max_len-full_total} tokens")

        except ImportError:
            print("\n   [跳过] 需要 transformers 库才能做精确 token 计数")
    else:
        # 粗略估算: 中文约 1.5 tokens/char
        est_ratio = 1.5
        short_est = int(len(short_prompt) * est_ratio)
        full_est = int(len(full_prompt) * est_ratio)
        print(f"\n   Token 估算 (1.5 tokens/char):")
        print(f"     精简版: ~{short_est} tokens")
        print(f"     完整版: ~{full_est} tokens")
        print(f"     节省: ~{full_est - short_est} tokens")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="", type=str)
    args = parser.parse_args()
    test_prompt_length(args.model_path if args.model_path else None)
