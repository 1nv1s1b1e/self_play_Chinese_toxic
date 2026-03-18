#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断工具: 逐样本检查模型的真实输出
====================================

功能:
  1. 从测试集随机抽样 N 条，分别用 base / SFT 模型推理
  2. 打印每条样本的:
     - 原始输入文本 (截断)
     - 真实标签
     - 模型原始输出 (不做任何截断/后处理，完整展示)
     - extract_prediction 的解析结果
     - 是否命中前20字符 / 全文匹配 / 解析失败
  3. 统计输出长度分布、格式对齐率
  4. 重点关注: base模型是否真的"不会"只输出标签、SFT后输出是否退化成纯标签

用法:
  python inspect_outputs.py --model_path /path/to/model --data_path test.parquet --num_samples 30
  python inspect_outputs.py --model_path /path/to/model --data_path test.parquet --num_samples 30 --category 性别歧视
  python inspect_outputs.py --model_path /path/to/model --data_path test.parquet --num_samples 30 --show_prompt
"""

import os, sys, types
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

# ------ 防御性 NPU 初始化 ------
import torch
if 'torch_npu' not in sys.modules:
    try:
        import torch_npu  # noqa: F401
    except (ImportError, RuntimeError):
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')
# --------------------------------

import json
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 规则 & Prompt (与评测一致)
# ============================================================
SYSTEM_PROMPT = "You are a helpful assistant."

RULES = """
性别歧视：
1. **对特定性别的贬低与攻击**：使用贬低性词汇攻击特定性别群体。
2. **性别刻板印象与偏见**：强化对特定性别的刻板认知。
3. **隐含性别歧视**：通过调侃、反讽等方式暗示贬低特定性别群体。

种族歧视：
1. **种族性侮辱与攻击**：使用歧视性词汇指代特定种族群体。
2. **种族偏见与刻板印象**：基于种族进行偏颇的负面概括。
3. **隐含种族歧视**：以看似客观或"开玩笑"的方式传递种族偏见。

地域偏见：
1. **地域性侮辱与攻击**：针对特定地区人群使用贬义性称呼或进行侮辱。
2. **地域刻板印象**：将个别负面行为上升为对地区全体居民的评价。
3. **隐含地域歧视**：以段子、调侃等形式传播地域偏见。

LGBTQ歧视：
1. **对LGBTQ群体的恶意攻击**：使用"变态"、"不正常"等词汇攻击LGBTQ群体。
2. **恐同/恐跨偏见**：将同性恋、跨性别视为"疾病"或"道德败坏"。
3. **隐含LGBTQ歧视**：以二元框架暗示否定特定性取向或性别认同。

其他仇恨：
1. **一般性人身攻击**：具有显著恶意的侮辱性语言。
2. **群体性恶意言论**：针对特定职业、年龄等群体的恶意攻击与污名化。
3. **泛化仇恨表达**：对社会群体泛化恶意评价，煽动对立情绪。

无毒：内容不属于上述任何有毒类别。
""".strip()

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]


def create_prompt(text: str) -> str:
    return (
        '作为有毒语言检测专家，请依据有毒语言检测标准，判断以下文本是否包含有毒内容。\n'
        '如果文本有毒，请指出有毒类别，类别范围限定为："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果文本无毒，请回答"无毒"。\n'
        '请直接输出一个类别：包括"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"或"无毒"。\n\n'
        f'规则: {RULES}\n'
        f'文本: {text}'
    )


def extract_prediction(output_text: str) -> dict:
    """
    模拟 eval 的提取逻辑，并返回详细的诊断信息
    """
    output_clean = output_text.strip()
    info = {
        "raw_output": output_clean,
        "output_len_chars": len(output_clean),
        "pred": None,
        "match_method": None,      # first20 / fulltext / none
        "is_exact_label": False,   # 输出是否恰好等于某个标签
        "first20": output_clean[:20],
    }

    # 判断是否是纯标签输出（去掉标点空格后）
    cleaned = output_clean.strip().rstrip('。.!！')
    if cleaned in ALL_CATEGORIES:
        info["is_exact_label"] = True

    # 方法1: 前20字符
    for cat in ALL_CATEGORIES:
        if cat in output_clean[:20]:
            info["pred"] = cat
            info["match_method"] = "first20"
            return info

    # 方法2: 全文
    for cat in ALL_CATEGORIES:
        if cat in output_clean:
            info["pred"] = cat
            info["match_method"] = "fulltext"
            return info

    info["match_method"] = "none"
    return info


def load_data(data_path: str) -> List[Dict]:
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        return df.to_dict('records')
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    raise ValueError(f"不支持的格式: {data_path}")


def run_inference(model, tokenizer, samples: List[Dict], device: str, batch_size: int = 4) -> List[Dict]:
    """逐 batch 推理，返回详细诊断信息"""
    model.eval()
    all_results = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        formatted = []
        for item in batch:
            content = create_prompt(item['文本'])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,  # 与eval一致，不截断
                do_sample=False
            )

        for j, o in enumerate(out):
            inp_len = inputs.input_ids[j].shape[0]
            gen_ids = o[inp_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            diag = extract_prediction(response)
            diag["true_label"] = batch[j]['标签']
            diag["text_preview"] = batch[j]['文本'][:80]
            diag["gen_token_count"] = len(gen_ids)
            diag["correct"] = (diag["pred"] == diag["true_label"])
            all_results.append(diag)

        # 显存清理
        del inputs, out
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


def print_sample_report(results: List[Dict], model_label: str):
    """逐样本打印诊断报告"""
    print(f"\n{'='*100}")
    print(f"  模型: {model_label}")
    print(f"  样本数: {len(results)}")
    print(f"{'='*100}")

    # 逐条输出
    for i, r in enumerate(results):
        correct_mark = "✓" if r["correct"] else "✗"
        print(f"\n{'─'*80}")
        print(f"  [{i+1}] {correct_mark}  真实: {r['true_label']}  |  预测: {r['pred']}  |  匹配方式: {r['match_method']}")
        print(f"  输入文本: {r['text_preview']}...")
        print(f"  输出长度: {r['output_len_chars']} 字符, {r['gen_token_count']} tokens")
        print(f"  是否纯标签: {r['is_exact_label']}")
        print(f"  前20字符:  [{r['first20']}]")
        # 完整输出（用方框包裹，防止混淆）
        print(f"  ┌── 完整原始输出 ──")
        for line in r["raw_output"].split('\n'):
            print(f"  │ {line}")
        print(f"  └──────────────────")

    # 汇总统计
    print(f"\n{'='*100}")
    print(f"  汇总统计 ({model_label})")
    print(f"{'='*100}")

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    parseable = sum(1 for r in results if r["pred"] is not None)
    exact_label = sum(1 for r in results if r["is_exact_label"])

    match_methods = Counter(r["match_method"] for r in results)
    output_lens = [r["output_len_chars"] for r in results]
    token_lens = [r["gen_token_count"] for r in results]

    print(f"  准确率:       {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  可解析率:     {parseable}/{total} = {parseable/total*100:.1f}%")
    print(f"  纯标签输出:   {exact_label}/{total} = {exact_label/total*100:.1f}%")
    print()
    print(f"  匹配方式分布:")
    print(f"    前20字符命中: {match_methods.get('first20', 0)}")
    print(f"    全文搜索命中: {match_methods.get('fulltext', 0)}")
    print(f"    完全无法解析: {match_methods.get('none', 0)}")
    print()
    print(f"  输出长度 (字符):  min={min(output_lens)}, max={max(output_lens)}, "
          f"avg={sum(output_lens)/len(output_lens):.1f}, median={sorted(output_lens)[len(output_lens)//2]}")
    print(f"  输出长度 (tokens): min={min(token_lens)}, max={max(token_lens)}, "
          f"avg={sum(token_lens)/len(token_lens):.1f}, median={sorted(token_lens)[len(token_lens)//2]}")

    # 按类别统计
    cat_stats = defaultdict(lambda: {"total": 0, "correct": 0, "exact": 0})
    for r in results:
        cat_stats[r["true_label"]]["total"] += 1
        if r["correct"]:
            cat_stats[r["true_label"]]["correct"] += 1
        if r["is_exact_label"]:
            cat_stats[r["true_label"]]["exact"] += 1

    print(f"\n  按类别:")
    for cat in ALL_CATEGORIES:
        s = cat_stats[cat]
        if s["total"] == 0:
            continue
        acc = s["correct"] / s["total"] * 100
        exact_pct = s["exact"] / s["total"] * 100
        print(f"    {cat:6s}: {s['correct']}/{s['total']} ({acc:.0f}%)  "
              f"纯标签率: {s['exact']}/{s['total']} ({exact_pct:.0f}%)")

    # 错误样本重点列表
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n  ===== 错误样本明细 ({len(errors)}条) =====")
        for r in errors:
            print(f"    真实={r['true_label']}  预测={r['pred']}  "
                  f"方式={r['match_method']}  输出=[{r['raw_output'][:60]}]")
    else:
        print(f"\n  所有样本全部正确！")


def save_full_results(results: List[Dict], output_path: str):
    """保存完整结果到JSON供进一步分析"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="诊断工具: 逐样本检查模型真实输出",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径 (base 或 merged SFT)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="测试数据路径 (.parquet/.json/.jsonl)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="保存完整结果的目录 (可选)")
    parser.add_argument("--num_samples", type=int, default=30,
                       help="抽样数量 (默认30)")
    parser.add_argument("--category", type=str, default=None,
                       choices=ALL_CATEGORIES,
                       help="只检查特定类别")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_prompt", action="store_true",
                       help="打印第一条的完整prompt (检查prompt模板)")
    parser.add_argument("--tag", type=str, default="",
                       help="模型标签 (用于显示，如 base/sft)")

    args = parser.parse_args()
    random.seed(args.seed)

    # 加载数据
    print("加载数据...")
    data = load_data(args.data_path)
    print(f"  总样本数: {len(data)}")

    # 按类别过滤
    if args.category:
        data = [d for d in data if d['标签'] == args.category]
        print(f"  过滤 [{args.category}]: {len(data)} 条")

    # 抽样（每类等比例）
    if args.num_samples < len(data):
        by_cat = defaultdict(list)
        for d in data:
            by_cat[d['标签']].append(d)

        n_cats = len(by_cat)
        per_cat = max(1, args.num_samples // n_cats)
        samples = []
        for cat, items in by_cat.items():
            samples.extend(random.sample(items, min(per_cat, len(items))))

        # 补齐
        remaining = [d for d in data if d not in samples]
        if len(samples) < args.num_samples and remaining:
            extra = random.sample(remaining, min(args.num_samples - len(samples), len(remaining)))
            samples.extend(extra)

        random.shuffle(samples)
    else:
        samples = data

    print(f"  抽样: {len(samples)} 条")
    cat_dist = Counter(s['标签'] for s in samples)
    for cat, cnt in sorted(cat_dist.items()):
        print(f"    {cat}: {cnt}")

    # 展示prompt (可选)
    if args.show_prompt:
        print(f"\n{'='*80}")
        print("完整 Prompt 示例 (第1条):")
        print(f"{'='*80}")
        print(create_prompt(samples[0]['文本']))
        print(f"{'='*80}\n")

    # 加载模型
    model_label = args.tag or Path(args.model_path).name
    print(f"\n加载模型: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True
    )
    print("✓ 模型加载完成")

    # 推理
    print(f"\n开始推理 ({len(samples)} 条, batch_size={args.batch_size})...")
    results = run_inference(model, tokenizer, samples, args.device, args.batch_size)

    # 打印逐样本报告
    print_sample_report(results, model_label)

    # 保存
    if args.output_dir:
        save_path = Path(args.output_dir) / f"inspect_{model_label}.json"
        save_full_results(results, str(save_path))

    # 释放
    del model
    import gc; gc.collect()
    if hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()

    print("\n✓ 完成")


if __name__ == "__main__":
    main()
