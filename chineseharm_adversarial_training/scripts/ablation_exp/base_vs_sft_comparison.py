#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 3: Base vs SFT 多维对比
==================================

系统性对比 base模型 和 SFT模型 在多个维度上的表现差异:

1. 格式对齐率:  模型输出是否可被正确解析为类别名 (而非返回长段解释)
2. 准确率/F1:   分类是否正确
3. 输出长度:    模型输出的平均token数 (SFT后应显著缩短)
4. 分类别增益:  哪些类别受益于SFT最多
5. 混淆矩阵:   详细的分类混淆分析

验证假设: SFT的主要贡献是格式对齐(只输出标签名)而非分类能力
"""

import os, sys, types
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

# ------ 防御性 NPU 初始化: 必须在其他任何库之前完成 ------
import torch
if 'torch_npu' not in sys.modules:
    try:
        import torch_npu  # noqa: F401
    except (ImportError, RuntimeError):
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')
# ------------------------------------------------------------------

import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
SYSTEM_PROMPT = "You are a helpful assistant."

# ToxiCN有毒语言分类规则
FULL_RULES = """
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
"""


def create_prompt(text: str) -> str:
    return (
        '作为中文有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '有毒类别："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果无毒请回答"无毒"。\n'
        '请直接输出一个类别名。\n\n'
        f'规则: {FULL_RULES}\n'
        f'文本: {text}'
    )


def extract_prediction(output_text: str) -> Optional[str]:
    output_clean = output_text.strip()
    for cat in ALL_CATEGORIES:
        if cat in output_clean[:30]:
            return cat
    for cat in ALL_CATEGORIES:
        if cat in output_clean:
            return cat
    return None


def load_eval_data(path: str) -> List[Dict]:
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        return [{"文本": row.get("original_text", row.get("文本", "")),
                 "标签": row.get("category", row.get("标签", ""))} for _, row in df.iterrows()]
    elif path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(l.strip()) for l in f]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def run_inference(
    model_path: str, data: List[Dict],
    mode: str = "npu", device: str = "npu:0",
    batch_size: int = 8, tp: int = 1,
) -> List[Dict]:
    """推理并返回含详细信息的结果"""
    import torch

    if mode == "vllm":
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16",
                   max_model_len=4096, tensor_parallel_size=tp)
        sp = SamplingParams(max_tokens=1024, temperature=0.0, top_p=1.0)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        formatted = []
        for item in data:
            msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":create_prompt(item['文本'])}]
            formatted.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        outputs = llm.generate(formatted, sp)

        results = []
        for i, out in enumerate(outputs):
            resp = out.outputs[0].text.strip()
            pred = extract_prediction(resp)
            results.append({
                "true_label": data[i]['标签'], "pred_label": pred,
                "response": resp, "response_length": len(resp),
                "response_tokens": len(tokenizer.encode(resp)),
            })

        del llm
        import gc; gc.collect()
        return results

    else:  # HF
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        model.eval()

        results = []
        for start in range(0, len(data), batch_size):
            batch = data[start:start+batch_size]
            formatted = []
            for item in batch:
                msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":create_prompt(item['文本'])}]
                formatted.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            for j, o in enumerate(out):
                inp_len = inputs.input_ids[j].shape[0]
                gen_ids = o[inp_len:]
                resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                results.append({
                    "true_label": data[start+j]['标签'], "pred_label": extract_prediction(resp),
                    "response": resp, "response_length": len(resp),
                    "response_tokens": len(gen_ids),
                })

        del model
        import gc; gc.collect()
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available(): torch.npu.empty_cache()
            elif torch.cuda.is_available(): torch.cuda.empty_cache()
        except: pass
        return results


def analyze_results(results: List[Dict], label: str) -> Dict:
    """多维度分析"""
    total = len(results)
    correct = sum(1 for r in results if r['pred_label'] == r['true_label'])
    failed = sum(1 for r in results if r['pred_label'] is None)
    parseable = total - failed

    # 格式对齐率
    format_alignment = parseable / total if total > 0 else 0

    # 准确率 (仅计算可解析的)
    accuracy_all = correct / total if total > 0 else 0
    accuracy_parseable = correct / parseable if parseable > 0 else 0

    # 输出长度统计
    resp_lengths = [r['response_length'] for r in results]
    resp_tokens = [r.get('response_tokens', 0) for r in results]

    # 分类别
    label_stats = defaultdict(lambda: {'TP':0,'FP':0,'FN':0})
    for r in results:
        tl, pl = r['true_label'], r['pred_label']
        if pl is None:
            label_stats[tl]['FN'] += 1
        elif pl == tl:
            label_stats[tl]['TP'] += 1
        else:
            label_stats[pl]['FP'] += 1
            label_stats[tl]['FN'] += 1

    cat_metrics = {}
    f1_list = []
    for cat in ALL_CATEGORIES:
        s = label_stats[cat]
        p = s['TP']/(s['TP']+s['FP']) if (s['TP']+s['FP'])>0 else 0
        r = s['TP']/(s['TP']+s['FN']) if (s['TP']+s['FN'])>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f1_list.append(f1)
        cat_metrics[cat] = {"precision":round(p,4),"recall":round(r,4),"f1":round(f1,4),"support":s['TP']+s['FN']}

    macro_f1 = sum(f1_list) / len(f1_list)

    # 混淆矩阵
    confusion = {true_cat: {pred_cat: 0 for pred_cat in ALL_CATEGORIES + ["解析失败"]}
                 for true_cat in ALL_CATEGORIES}
    for r in results:
        tl = r['true_label']
        pl = r['pred_label'] if r['pred_label'] else "解析失败"
        if tl in confusion and pl in confusion[tl]:
            confusion[tl][pl] += 1

    return {
        "label": label,
        "total": total,
        "correct": correct,
        "failed_extractions": failed,
        "format_alignment_rate": round(format_alignment, 4),
        "accuracy_all": round(accuracy_all, 4),
        "accuracy_parseable": round(accuracy_parseable, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_response_length_chars": round(np.mean(resp_lengths), 1),
        "median_response_length_chars": round(np.median(resp_lengths), 1),
        "avg_response_tokens": round(np.mean(resp_tokens), 1) if any(resp_tokens) else 0,
        "category_metrics": cat_metrics,
        "confusion_matrix": confusion,
    }


def print_comparison(base_a: Dict, sft_a: Dict):
    """打印对比表"""
    print(f"\n{'=' * 70}")
    print("Base vs SFT 多维对比")
    print(f"{'=' * 70}")

    print(f"\n{'指标':<28} {'Base':>12} {'SFT':>12} {'差值':>12}")
    print("-" * 66)

    rows = [
        ("格式对齐率", base_a['format_alignment_rate'], sft_a['format_alignment_rate']),
        ("准确率 (全部)", base_a['accuracy_all'], sft_a['accuracy_all']),
        ("准确率 (可解析)", base_a['accuracy_parseable'], sft_a['accuracy_parseable']),
        ("Macro-F1", base_a['macro_f1'], sft_a['macro_f1']),
        ("解析失败数", base_a['failed_extractions'], sft_a['failed_extractions']),
        ("平均输出长度(字符)", base_a['avg_response_length_chars'], sft_a['avg_response_length_chars']),
        ("中位输出长度(字符)", base_a['median_response_length_chars'], sft_a['median_response_length_chars']),
    ]

    for name, b, s in rows:
        diff = s - b
        if isinstance(b, float) and b < 2:
            print(f"{name:<28} {b:>12.2%} {s:>12.2%} {diff:>+12.2%}")
        else:
            print(f"{name:<28} {b:>12.1f} {s:>12.1f} {diff:>+12.1f}")

    # 分类别F1对比
    print(f"\n{'类别':<10} {'Base F1':>10} {'SFT F1':>10} {'增益':>10} {'Base Recall':>12} {'SFT Recall':>12}")
    print("-" * 68)
    for cat in ALL_CATEGORIES:
        bm = base_a['category_metrics'].get(cat, {})
        sm = sft_a['category_metrics'].get(cat, {})
        bf1 = bm.get('f1', 0)
        sf1 = sm.get('f1', 0)
        br = bm.get('recall', 0)
        sr = sm.get('recall', 0)
        print(f"{cat:<10} {bf1:>10.4f} {sf1:>10.4f} {sf1-bf1:>+10.4f} {br:>12.4f} {sr:>12.4f}")

    # 判断SFT主要贡献
    fmt_gain = sft_a['format_alignment_rate'] - base_a['format_alignment_rate']
    acc_parseable_gain = sft_a['accuracy_parseable'] - base_a['accuracy_parseable']

    print(f"\n{'=' * 70}")
    print("分析结论:")
    print(f"{'=' * 70}")
    print(f"  格式对齐率提升:           {fmt_gain:+.2%}")
    print(f"  可解析样本准确率提升:     {acc_parseable_gain:+.2%}")

    if fmt_gain > 0.1 and abs(acc_parseable_gain) < 0.05:
        print(f"\n  >> 结论: SFT的主要贡献是格式对齐（让模型只输出标签名），")
        print(f"     而非分类能力。Base模型在给定RULES后已具备足够的分类能力。")
    elif fmt_gain > 0.1 and acc_parseable_gain > 0.05:
        print(f"\n  >> 结论: SFT同时提升了格式对齐和分类能力。")
    else:
        print(f"\n  >> 结论: Base模型格式已较好，SFT主要提升分类准确率。")


def main():
    parser = argparse.ArgumentParser(
        description="Base vs SFT 多维对比分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python base_vs_sft_comparison.py \\
    --base_model /path/to/base \\
    --sft_model /path/to/sft_merged \\
    --mode npu
""")
    parser.add_argument("--base_model", type=str, required=True, help="Base模型路径")
    parser.add_argument("--sft_model", type=str, required=True, help="SFT(merged)模型路径")
    parser.add_argument("--data_path", type=str, default="/home/ma-user/work/test/split_data/test.parquet")
    parser.add_argument("--output_dir", type=str, default="/home/ma-user/work/test/ablation_results/base_vs_sft")
    parser.add_argument("--mode", type=str, choices=["vllm", "npu"], default="vllm")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("消融实验: Base vs SFT 多维对比")
    print("=" * 70)
    print(f"Base模型: {args.base_model}")
    print(f"SFT模型:  {args.sft_model}")
    print()

    data = load_eval_data(args.data_path)
    if args.num_samples:
        data = data[:args.num_samples]
    print(f"测试数据: {len(data)} 条\n")

    # Base评测
    print(f"\n{'━' * 50}")
    print("评测 Base 模型...")
    print(f"{'━' * 50}")
    t0 = time.time()
    base_raw = run_inference(args.base_model, data, args.mode, args.device, args.batch_size, args.tp)
    base_time = time.time() - t0
    base_analysis = analyze_results(base_raw, "base")
    base_analysis['elapsed_sec'] = round(base_time, 1)
    print(f"  完成: Acc={base_analysis['accuracy_all']:.2%} F1={base_analysis['macro_f1']:.4f}")

    # SFT评测
    print(f"\n{'━' * 50}")
    print("评测 SFT 模型...")
    print(f"{'━' * 50}")
    t0 = time.time()
    sft_raw = run_inference(args.sft_model, data, args.mode, args.device, args.batch_size, args.tp)
    sft_time = time.time() - t0
    sft_analysis = analyze_results(sft_raw, "sft")
    sft_analysis['elapsed_sec'] = round(sft_time, 1)
    print(f"  完成: Acc={sft_analysis['accuracy_all']:.2%} F1={sft_analysis['macro_f1']:.4f}")

    # 对比
    print_comparison(base_analysis, sft_analysis)

    # 保存
    base_name = Path(args.base_model).name
    sft_name = Path(args.sft_model).name

    with open(output_dir / f"base_vs_sft_{base_name}_vs_{sft_name}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "base_model": args.base_model,
            "sft_model": args.sft_model,
            "data_path": args.data_path,
            "num_samples": len(data),
            "base_analysis": base_analysis,
            "sft_analysis": sft_analysis,
            "base_predictions_sample": base_raw[:30],
            "sft_predictions_sample": sft_raw[:30],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {output_dir}/")


if __name__ == "__main__":
    main()
