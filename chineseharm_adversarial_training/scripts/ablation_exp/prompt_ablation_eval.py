#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 1: Prompt 变体消融
============================

对比同一个模型在不同 prompt 下的表现，验证 RULES 对结果的贡献程度。

Prompt 变体:
  A. full_rules    — 完整RULES（当前训练/评测使用的prompt，~2000字规则）
  B. brief_rules   — 精简规则（仅列出类别名+一句话描述，无具体关键词）
  C. no_rules      — 无规则（仅告知类别范围，不提供任何规则知识）
  D. zero_shot     — 零样本（最简prompt，只给文本让模型判断）

支持: vLLM 加速推理 / transformers fallback
"""

import os, sys, types
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

# ------ 防御性 NPU 初始化: 必须在其他任何库之前完成 ------
import torch
if 'torch_npu' not in sys.modules:
    try:
        import torch_npu  # noqa: F401
    except (ImportError, RuntimeError):
        # NPU 已经内置在 torch 中，放一个哑元 module 防止后续库重复 import
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')
# ------------------------------------------------------------------

import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

# ============================================================
# 完整规则（ToxiCN有毒语言检测规则）
# ============================================================
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

# ============================================================
# 精简规则（每类别一句话描述）
# ============================================================
BRIEF_RULES = """
性别歧视：使用贬低性词汇攻击特定性别群体，强化性别刻板印象或隐含性别歧视。
种族歧视：使用歧视性词汇指代特定种族群体，基于种族进行负面概括或隐含种族偏见。
地域偏见：针对特定地区人群使用贬义性称呼，传播地域刻板印象或隐含地域歧视。
LGBTQ歧视：使用恶意词汇攻击LGBTQ群体，传播恐同/恐跨偏见或隐含LGBTQ歧视。
其他仇恨：具有显著恶意的人身攻击、群体性恶意言论或泛化仇恨表达。
"""


# ============================================================
# 四种 Prompt 生成函数
# ============================================================
def create_prompt_full_rules(text: str) -> str:
    """A. 完整规则 prompt（与训练/原评测一致）"""
    return (
        '作为有毒语言检测专家，请依据有毒语言检测标准，判断以下文本是否包含有毒内容。\n'
        '如果文本有毒，请指出有毒类别，类别范围限定为："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果文本无毒，请回答"无毒"。\n'
        '请直接输出一个类别：包括"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"或"无毒"。\n\n'
        f'规则: {FULL_RULES}\n'
        f'文本: {text}'
    )


def create_prompt_brief_rules(text: str) -> str:
    """B. 精简规则 prompt（仅类别一句话描述）"""
    return (
        '作为有毒语言检测专家，请依据有毒语言检测标准，判断以下文本是否包含有毒内容。\n'
        '如果文本有毒，请指出有毒类别，类别范围限定为："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果文本无毒，请回答"无毒"。\n'
        '请直接输出一个类别：包括"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"或"无毒"。\n\n'
        f'各类别简要说明: {BRIEF_RULES}\n'
        f'文本: {text}'
    )


def create_prompt_no_rules(text: str) -> str:
    """C. 无规则 prompt（只告知类别范围）"""
    return (
        '作为有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '如果文本有毒，请从以下类别中选择一个：性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨。\n'
        '如果文本无毒，请回答"无毒"。\n'
        '请只输出类别名称，不要输出任何解释。\n\n'
        f'文本: {text}'
    )


def create_prompt_zero_shot(text: str) -> str:
    """D. 零样本 prompt（最简）"""
    return (
        '请判断以下文本属于哪个类别：性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨、无毒。\n'
        '只输出类别名称。\n\n'
        f'{text}'
    )


PROMPT_VARIANTS = {
    "full_rules": create_prompt_full_rules,
    "brief_rules": create_prompt_brief_rules,
    "no_rules": create_prompt_no_rules,
    "zero_shot": create_prompt_zero_shot,
}

PROMPT_DESCRIPTIONS = {
    "full_rules": "完整规则（ToxiCN详细规则）",
    "brief_rules": "精简规则（每类别一句话描述）",
    "no_rules": "无规则 (仅提供类别名称范围)",
    "zero_shot": "零样本 (最简prompt)",
}


# ============================================================
# 评测核心
# ============================================================
ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
SYSTEM_PROMPT = "You are a helpful assistant."


def extract_prediction(output_text: str) -> Optional[str]:
    """从模型输出中提取预测类别"""
    output_clean = output_text.strip()
    for category in ALL_CATEGORIES:
        if category in output_clean[:30]:
            return category
    for category in ALL_CATEGORIES:
        if category in output_clean:
            return category
    return None


def calculate_metrics(results: List[Dict]) -> Dict:
    """计算完整评测指标"""
    label_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    correct = 0
    total = 0
    failed = 0

    for item in results:
        true_label = item["true_label"]
        pred_label = item["pred_label"]
        total += 1
        if pred_label is None:
            failed += 1
            label_stats[true_label]['FN'] += 1
        elif pred_label == true_label:
            correct += 1
            label_stats[true_label]['TP'] += 1
        else:
            label_stats[pred_label]['FP'] += 1
            label_stats[true_label]['FN'] += 1

    accuracy = correct / total if total > 0 else 0

    category_metrics = {}
    f1_scores = []
    for cat in ALL_CATEGORIES:
        s = label_stats[cat]
        p = s['TP'] / (s['TP'] + s['FP']) if (s['TP'] + s['FP']) > 0 else 0
        r = s['TP'] / (s['TP'] + s['FN']) if (s['TP'] + s['FN']) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
        category_metrics[cat] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": s['TP'] + s['FN']
        }

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": total,
        "failed_extractions": failed,
        "category_metrics": category_metrics,
    }


def load_eval_data(data_path: str) -> List[Dict]:
    """加载评测数据"""
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        data = []
        for _, row in df.iterrows():
            data.append({
                "文本": row.get("original_text", row.get("文本", "")),
                "标签": row.get("category", row.get("标签", ""))
            })
        return data
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def run_vllm_eval(
    model_path: str, data: List[Dict],
    prompt_fn, tensor_parallel: int = 1
) -> List[Dict]:
    """使用 vLLM 推理"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=model_path, trust_remote_code=True,
        dtype="bfloat16", max_model_len=4096,
        tensor_parallel_size=tensor_parallel,
    )
    sampling_params = SamplingParams(max_tokens=64, temperature=0.0, top_p=1.0)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    formatted = []
    for item in data:
        content = prompt_fn(item['文本'])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]
        formatted.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    outputs = llm.generate(formatted, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        pred = extract_prediction(response)
        results.append({
            "true_label": data[i]['标签'],
            "pred_label": pred,
            "response": response,
        })

    del llm
    import gc; gc.collect()
    try:
        import torch
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except: pass

    return results


def run_hf_eval(
    model_path: str, data: List[Dict],
    prompt_fn, device: str = "npu:0", batch_size: int = 8
) -> List[Dict]:
    """transforms fallback"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True
    )
    model.eval()

    results = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        formatted = []
        for item in batch:
            content = prompt_fn(item['文本'])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)

        for j, o in enumerate(out):
            inp_len = inputs.input_ids[j].shape[0]
            response = tokenizer.decode(o[inp_len:], skip_special_tokens=True).strip()
            pred = extract_prediction(response)
            idx = start + j
            results.append({
                "true_label": data[idx]['标签'],
                "pred_label": pred,
                "response": response,
            })

    del model
    import gc; gc.collect()
    try:
        import torch as _torch
        if hasattr(_torch, 'npu') and _torch.npu.is_available():
            _torch.npu.empty_cache()
        elif _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except: pass

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Prompt消融实验 — 对比不同prompt变体下模型的分类性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有prompt变体 (vLLM)
  python prompt_ablation_eval.py --model_path /path/to/model --mode vllm

  # 只测指定变体
  python prompt_ablation_eval.py --model_path /path/to/model --variants full_rules no_rules

  # SFT模型 vs Base模型 在不同prompt下
  python prompt_ablation_eval.py --model_path /path/to/sft_model --mode npu --tag sft
  python prompt_ablation_eval.py --model_path /path/to/base_model --mode npu --tag base
""")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_path", type=str,
                       default="/home/ma-user/work/test/split_data/test.parquet")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/ablation_results/prompt_ablation")
    parser.add_argument("--mode", type=str, choices=["vllm", "npu"], default="vllm")
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--batch_size", type=int, default=8, help="HF batch size")
    parser.add_argument("--device", type=str, default="npu:0")
    parser.add_argument("--tag", type=str, default="", help="模型标签 (base/sft/rl等)")
    parser.add_argument("--variants", nargs='+',
                       choices=list(PROMPT_VARIANTS.keys()),
                       default=list(PROMPT_VARIANTS.keys()),
                       help="要测试的prompt变体")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="限制样本数(调试用)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.model_path).name
    tag = f"_{args.tag}" if args.tag else ""

    print("=" * 70)
    print("消融实验: Prompt 变体消融")
    print("=" * 70)
    print(f"模型:     {args.model_path}")
    print(f"数据:     {args.data_path}")
    print(f"推理模式: {args.mode}")
    print(f"变体:     {args.variants}")
    print()

    # 加载数据
    data = load_eval_data(args.data_path)
    if args.num_samples:
        data = data[:args.num_samples]
    print(f"测试数据: {len(data)} 条\n")

    all_results = {}

    for variant_name in args.variants:
        prompt_fn = PROMPT_VARIANTS[variant_name]
        desc = PROMPT_DESCRIPTIONS[variant_name]

        print(f"\n{'─' * 60}")
        print(f"[{variant_name}] {desc}")
        print(f"{'─' * 60}")

        t0 = time.time()
        if args.mode == "vllm":
            raw_results = run_vllm_eval(
                args.model_path, data, prompt_fn, args.tp
            )
        else:
            raw_results = run_hf_eval(
                args.model_path, data, prompt_fn, args.device, args.batch_size
            )
        elapsed = time.time() - t0

        metrics = calculate_metrics(raw_results)
        metrics["prompt_variant"] = variant_name
        metrics["prompt_description"] = desc
        metrics["elapsed_sec"] = round(elapsed, 1)

        all_results[variant_name] = metrics

        print(f"  准确率:   {metrics['accuracy']:.2%}")
        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  耗时:     {elapsed:.1f}s")
        print(f"  失败提取: {metrics['failed_extractions']}")

        # 分类别
        for cat in ALL_CATEGORIES:
            m = metrics['category_metrics'].get(cat, {})
            print(f"    {cat:<8} P={m.get('precision',0):.3f} R={m.get('recall',0):.3f} F1={m.get('f1',0):.3f} (n={m.get('support',0)})")

        # 保存单变体结果 (含详细预测)
        variant_file = output_dir / f"prompt_{variant_name}_{model_name}{tag}.json"
        with open(variant_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "predictions": raw_results[:50],  # 保存前50条详细结果供分析
            }, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 汇总对比
    # ============================================================
    print(f"\n\n{'=' * 70}")
    print("Prompt消融实验汇总")
    print(f"{'=' * 70}")
    print(f"模型: {model_name} {tag}")
    print(f"\n{'Prompt变体':<16} {'Accuracy':>10} {'Macro-F1':>10} {'Failed':>8} {'Time(s)':>8}")
    print("-" * 56)
    for name in args.variants:
        m = all_results[name]
        print(f"{name:<16} {m['accuracy']:>10.2%} {m['macro_f1']:>10.4f} {m['failed_extractions']:>8} {m['elapsed_sec']:>8.1f}")

    # 计算RULES贡献度
    if 'full_rules' in all_results and 'no_rules' in all_results:
        diff_acc = all_results['full_rules']['accuracy'] - all_results['no_rules']['accuracy']
        diff_f1 = all_results['full_rules']['macro_f1'] - all_results['no_rules']['macro_f1']
        print(f"\nRULES贡献度 (full_rules - no_rules):")
        print(f"  Accuracy差: {diff_acc:+.2%}")
        print(f"  Macro-F1差: {diff_f1:+.4f}")

    if 'full_rules' in all_results and 'zero_shot' in all_results:
        diff_acc = all_results['full_rules']['accuracy'] - all_results['zero_shot']['accuracy']
        diff_f1 = all_results['full_rules']['macro_f1'] - all_results['zero_shot']['macro_f1']
        print(f"\n总prompt贡献度 (full_rules - zero_shot):")
        print(f"  Accuracy差: {diff_acc:+.2%}")
        print(f"  Macro-F1差: {diff_f1:+.4f}")

    # 保存汇总
    summary_file = output_dir / f"prompt_ablation_summary_{model_name}{tag}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model_path,
            "tag": args.tag,
            "data_path": args.data_path,
            "num_samples": len(data),
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存: {summary_file}")


if __name__ == "__main__":
    main()
