#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估: 对比精简版 vs 完整版 prompt 在现有模型上的效果
===================================================
不需要重新训练! 直接用现有 SFT 模型分别跑两种模板,
看精简版是否掉点严重, 决定是否需要 re-SFT.

用法 (需要 GPU/NPU):
  python tests/eval_template_compare.py \
      --model_path /path/to/reviewer_3B \
      --test_data /path/to/split_data/test.json \
      --max_samples 200

输出: 两种模板的 accuracy / macro-F1 / per-category F1 对比
"""

import os, sys, json, argparse, time
from collections import Counter, defaultdict

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)

from constants import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    REVIEWER_USER_TEMPLATE_FULL,
    HARMFUL_CATEGORIES, ALL_CATEGORIES,
    parse_classification_output,
)


def compute_metrics(results):
    """计算 accuracy, macro-F1, per-category metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["predict_label"] == r["true_label"])
    binary_correct = sum(1 for r in results if r["predict_binary_correct"])

    # Per-category
    cat_tp = defaultdict(int)
    cat_fp = defaultdict(int)
    cat_fn = defaultdict(int)
    cat_total = defaultdict(int)

    for r in results:
        true_label = r["true_label"]
        pred_label = r["predict_label"]
        cat_total[true_label] += 1

        if pred_label == true_label:
            cat_tp[true_label] += 1
        else:
            cat_fn[true_label] += 1
            cat_fp[pred_label] += 1

    cat_metrics = {}
    f1_scores = []
    for cat in ALL_CATEGORIES:
        tp = cat_tp.get(cat, 0)
        fp = cat_fp.get(cat, 0)
        fn = cat_fn.get(cat, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        cat_metrics[cat] = {
            "precision": precision, "recall": recall, "f1": f1,
            "total": cat_total.get(cat, 0),
        }
        if cat_total.get(cat, 0) > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return {
        "accuracy": correct / total if total > 0 else 0,
        "binary_accuracy": binary_correct / total if total > 0 else 0,
        "macro_f1": macro_f1,
        "total": total,
        "correct": correct,
        "category_metrics": cat_metrics,
    }


def run_evaluation(model_path, test_data, template, max_samples=0, batch_size=8):
    """用指定模板运行评估."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        import torch_npu
    except ImportError:
        pass

    if template == "short":
        tmpl = REVIEWER_USER_TEMPLATE
    else:
        tmpl = REVIEWER_USER_TEMPLATE_FULL

    # 限制样本数
    data = test_data
    if max_samples > 0 and len(data) > max_samples:
        data = data[:max_samples]

    print(f"\n   模板: {template} ({len(tmpl.format(text='X'))} chars)")
    print(f"   样本数: {len(data)}")

    # 加载模型
    print(f"   加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # 构建 prompt
    results = []
    device = next(model.parameters()).device

    from tqdm import tqdm
    for start in tqdm(range(0, len(data), batch_size), desc=f"   [{template}] 评估"):
        batch = data[start:start + batch_size]

        raw_prompts = []
        for row in batch:
            text = row.get("文本", "").strip()[:500]
            user_content = tmpl.format(text=text)
            msgs = [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            try:
                raw = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                raw = f"{REVIEWER_SYSTEM_PROMPT}\n{user_content}\n"
            raw_prompts.append(raw)

        enc = tokenizer(
            raw_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=2048,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                max_new_tokens=80, do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, row in enumerate(batch):
            gen = tokenizer.decode(out[j][input_ids.shape[1]:], skip_special_tokens=True).strip()
            parsed = parse_classification_output(gen)
            true_label = row.get("标签", "无毒")
            true_is_harmful = true_label in HARMFUL_CATEGORIES

            pred_binary = parsed.get("binary")
            pred_is_harmful = (pred_binary == "有害")

            results.append({
                "true_label": true_label,
                "predict_label": parsed.get("category", "无毒") if pred_binary else "无毒",
                "predict_binary_correct": (pred_is_harmful == true_is_harmful) if pred_binary else False,
                "response": gen,
            })

    # 释放
    del model
    import gc; gc.collect()
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="模板对比评估")
    parser.add_argument("--model_path", required=True, type=str,
                        help="现有 Reviewer 模型路径")
    parser.add_argument("--test_data", default="", type=str,
                        help="测试集 (默认: split_data/test.json)")
    parser.add_argument("--max_samples", default=200, type=int,
                        help="最大评估样本数 (默认 200, 快速验证)")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--output_dir", default="", type=str,
                        help="结果输出目录 (默认: tests/)")
    args = parser.parse_args()

    # 查找测试数据
    test_path = args.test_data
    if not test_path:
        for candidate in [
            os.path.join(_PARENT, "..", "..", "split_data", "test.json"),
            os.path.join(_PARENT, "..", "..", "split_data", "val.json"),
        ]:
            if os.path.exists(candidate):
                test_path = candidate
                break

    if not test_path or not os.path.exists(test_path):
        print("[错误] 找不到测试数据, 请用 --test_data 指定")
        sys.exit(1)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试集: {test_path} ({len(test_data)} 条)")

    output_dir = args.output_dir or os.path.join(_SELF_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 两次评估
    print("\n" + "=" * 60)
    print("Step 1: 完整版模板 (与 SFT 训练时一致)")
    print("=" * 60)
    results_full = run_evaluation(
        args.model_path, test_data, "full",
        max_samples=args.max_samples, batch_size=args.batch_size
    )
    metrics_full = compute_metrics(results_full)

    print("\n" + "=" * 60)
    print("Step 2: 精简版模板 (去掉 RULES)")
    print("=" * 60)
    results_short = run_evaluation(
        args.model_path, test_data, "short",
        max_samples=args.max_samples, batch_size=args.batch_size
    )
    metrics_short = compute_metrics(results_short)

    # 对比输出
    print("\n" + "=" * 70)
    print("对比结果: 完整版 vs 精简版 (现有模型, 无 re-SFT)")
    print("=" * 70)

    print(f"\n{'指标':<20} {'完整版':>10} {'精简版':>10} {'差异':>10}")
    print("-" * 55)

    def _row(name, v1, v2, pct=True):
        diff = v2 - v1
        fmt = ".1%" if pct else ".4f"
        print(f"{name:<20} {v1:{fmt}:>10} {v2:{fmt}:>10} {diff:+{fmt}:>10}")

    _row("Overall Accuracy", metrics_full["accuracy"], metrics_short["accuracy"])
    _row("Binary Accuracy", metrics_full["binary_accuracy"], metrics_short["binary_accuracy"])
    _row("Macro F1", metrics_full["macro_f1"], metrics_short["macro_f1"], pct=False)

    print(f"\n{'类别':<12} {'完整版F1':>10} {'精简版F1':>10} {'差异':>10}")
    print("-" * 45)
    for cat in ALL_CATEGORIES:
        f1_full = metrics_full["category_metrics"].get(cat, {}).get("f1", 0)
        f1_short = metrics_short["category_metrics"].get(cat, {}).get("f1", 0)
        diff = f1_short - f1_full
        flag = " <---" if abs(diff) > 0.05 else ""
        print(f"{cat:<12} {f1_full:>10.3f} {f1_short:>10.3f} {diff:>+10.3f}{flag}")

    # 判定
    acc_drop = metrics_full["accuracy"] - metrics_short["accuracy"]
    f1_drop = metrics_full["macro_f1"] - metrics_short["macro_f1"]

    print(f"\n{'=' * 70}")
    if acc_drop < 0.02 and f1_drop < 0.02:
        print("结论: 精简版掉点很小 (<2%), 可以直接使用, 不需要 re-SFT!")
        print("建议: 直接跑 self-play, 精简版 prompt 既快又不影响效果.")
    elif acc_drop < 0.05 and f1_drop < 0.05:
        print("结论: 精简版有一定掉点 (2-5%), 建议做轻量 re-SFT.")
        print("建议: 用精简版 prompt 跑 1-2 epoch SFT 即可恢复.")
    else:
        print(f"结论: 精简版掉点较大 (acc={acc_drop:.1%}, f1={f1_drop:.4f}), 必须 re-SFT!")
        print("建议: 用精简版 prompt 重建 SFT 数据, 从头训练 3 epoch.")

    # 保存结果
    report = {
        "model_path": args.model_path,
        "test_data": test_path,
        "max_samples": args.max_samples,
        "full_template": metrics_full,
        "short_template": metrics_short,
        "accuracy_drop": acc_drop,
        "f1_drop": f1_drop,
    }
    report_path = os.path.join(output_dir, "template_compare_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {report_path}")


if __name__ == "__main__":
    main()
