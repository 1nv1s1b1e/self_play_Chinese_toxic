#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifier (API) 在测试集上的评测脚本
====================================
验证外部 API Verifier（如 qwen-plus）在标准测试集上的分类精度，
量化 Verifier 标签偏差对 Reviewer SFT 训练的影响。

评测维度:
  1. 二分类 (有害/无害): Accuracy, Precision, Recall, F1
  2. 类别级 (6类): Macro-F1, Weighted-F1, 逐类 P/R/F1
  3. 混淆矩阵: 完整 6×6 分类混淆
  4. 与各轮 Reviewer 的对比
  5. 多 Verifier 集成投票 (Ensemble) 效果对比

用法:
  # 单模型 API 评测
  VERIFIER_API_KEY=sk-xxx python eval_verifier_api.py \
      --test_data ../../origin_data/bench.json \
      --backend api \
      --api_model qwen-plus \
      --output_dir ../../eval_results

  # 三个 DashScope 模型分别评测，并输出横向对比
  VERIFIER_API_KEY=sk-xxx python eval_verifier_api.py \
      --test_data ../../origin_data/bench.json \
      --api_models qwen-plus qwen-max qwen-turbo \
      --output_dir ../../eval_results

  # 三模型投票集成评测
  VERIFIER_API_KEY=sk-xxx python eval_verifier_api.py \
      --test_data ../../origin_data/bench.json \
      --api_models qwen-plus qwen-max qwen-turbo \
      --ensemble \
      --output_dir ../../eval_results

  # 使用已有评测结果文件提取测试子集
  VERIFIER_API_KEY=sk-xxx python eval_verifier_api.py \
      --test_data ../../eval_vllm_npu_reviewer_3B_reviewer_round0_baseline.json \
      --backend api \
      --api_model qwen-plus \
      --output_dir ../../eval_results

  # 比较模式: Verifier vs 各轮 Reviewer
  python eval_verifier_api.py \
      --compare \
      --verifier_result ../../eval_results/eval_verifier_qwen-plus.json \
      --reviewer_results \
          ../../eval_vllm_npu_reviewer_3B_reviewer_round0_baseline.json \
          ../../eval_vllm_npu_reviewer_reviewer_round1.json \
          ../../eval_vllm_npu_reviewer_reviewer_round2.json \
          ../../eval_vllm_npu_reviewer_reviewer_round3.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

# ── 统一导入 constants ───────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import (
    HARMFUL_CATEGORIES, ALL_CATEGORIES,
    parse_classification_output,
    build_verifier_messages,
)

LABEL_ORDER = ALL_CATEGORIES  # ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]

MODEL_REGISTRY = {
    "qwen-plus": {"supports_thinking": False},
    "qwen-max": {"supports_thinking": False},
    "qwen-turbo": {"supports_thinking": False},
    "qwq-32b": {"supports_thinking": True},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_data(path: str) -> list:
    """
    加载测试集。支持:
      - bench.json: [{"文本": ..., "标签": ...}, ...]
      - eval_vllm_*.json: {"metrics": ..., "results": [{"文本": ..., "标签": ...}, ...]}
      - parquet / jsonl
    """
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        col_text = "文本" if "文本" in df.columns else "original_text"
        col_cat = "标签" if "标签" in df.columns else "category"
        return [{"文本": str(row[col_text]), "标签": str(row[col_cat])}
                for _, row in df.iterrows()]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # eval_vllm 格式
    if isinstance(data, dict) and "results" in data:
        return [{"文本": r["文本"], "标签": r["标签"]} for r in data["results"]]

    # bench.json 格式
    if isinstance(data, list) and len(data) > 0 and "文本" in data[0]:
        return data

    raise ValueError(f"无法识别测试集格式: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. API 调用
# ═══════════════════════════════════════════════════════════════════════════════

def call_api_single(
    client,
    model: str,
    text: str,
    max_retries: int = 3,
    enable_thinking: bool = False,
    thinking_budget: int = 0,
    use_few_shots: bool = True,
) -> str:
    """
    单条 API 调用，带重试。
    使用 Verifier 专用 prompt（强化 system + few-shot 示例）。
    当 enable_thinking=True 时使用流式接口。
    """
    # 使用 Verifier 专用的强化 prompt（与 Reviewer SFT 完全分离）
    messages = build_verifier_messages(text, use_few_shots=use_few_shots)

    # 构建 extra_body
    extra_body = {"enable_thinking": enable_thinking}
    if enable_thinking and thinking_budget > 0:
        extra_body["thinking_budget"] = thinking_budget

    for attempt in range(max_retries):
        try:
            if enable_thinking:
                # 思考模式：使用流式输出（多数思考模型仅支持流式）
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=256,
                    extra_body=extra_body,
                    stream=True,
                )
                answer_content = ""
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    # 只收集最终回复 content，忽略 reasoning_content
                    if hasattr(delta, "content") and delta.content:
                        answer_content += delta.content
                return answer_content.strip()
            else:
                # 非思考模式：直接非流式调用
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=128,
                    extra_body=extra_body,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"  [重试] 第{attempt+1}次失败: {e}, {wait}s 后重试")
                time.sleep(wait)
            else:
                print(f"  [失败] 已用尽重试: {e}")
                return ""


def batch_verify_api(
    texts: list,
    api_key: str,
    api_base: str,
    api_model: str,
    max_workers: int = 8,
    max_retries: int = 3,
    enable_thinking: bool = False,
    thinking_budget: int = 0,
    use_few_shots: bool = True,
    progress_position: int = 0,
) -> list:
    """并发调用 API 验证所有文本。"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base)

    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = {"binary": "无害", "category": "无毒",
                              "toxic_type": "无毒", "expression": "非仇恨",
                              "raw_output": ""}
                continue
            future = executor.submit(
                call_api_single, client, api_model, text,
                max_retries, enable_thinking, thinking_budget, use_few_shots,
            )
            future_to_idx[future] = i

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx),
                           desc=f"[{api_model}] 评测中", position=progress_position, leave=True):
            idx = future_to_idx[future]
            try:
                raw = future.result()
                parsed = parse_classification_output(raw)
                parsed["raw_output"] = raw
                results[idx] = parsed
            except Exception as e:
                print(f"  [错误] idx={idx}: {e}")
                results[idx] = {"binary": None, "category": None,
                                "toxic_type": None, "expression": None,
                                "raw_output": ""}

    for i in range(len(results)):
        if results[i] is None:
            results[i] = {"binary": None, "category": None,
                          "toxic_type": None, "expression": None,
                          "raw_output": ""}
    return results


def pick_best_vote(votes: Dict[str, float], priority_order: List[str]) -> Optional[str]:
    """按票数选择结果；票数相同时按 priority_order 兜底。"""
    if not votes:
        return None

    priority_rank = {name: idx for idx, name in enumerate(priority_order)}
    return max(
        votes.items(),
        key=lambda item: (item[1], -priority_rank.get(item[0], len(priority_order))),
    )[0]


def ensemble_predictions(
    model_predictions: Dict[str, List[dict]],
    model_weights: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """对多个 verifier 的结果执行多数投票 / 加权投票。"""
    if not model_predictions:
        return []

    model_names = list(model_predictions.keys())
    sample_count = len(next(iter(model_predictions.values())))
    ensemble_results = []

    for sample_idx in range(sample_count):
        binary_votes = defaultdict(float)
        category_votes = defaultdict(float)
        toxic_type_votes = defaultdict(float)
        expression_votes = defaultdict(float)

        for model_name in model_names:
            pred = model_predictions[model_name][sample_idx]
            weight = (model_weights or {}).get(model_name, 1.0)
            binary = pred.get("binary")
            if not binary:
                continue

            binary_votes[binary] += weight
            if binary == "有害":
                if pred.get("category"):
                    category_votes[pred["category"]] += weight
                if pred.get("toxic_type"):
                    toxic_type_votes[pred["toxic_type"]] += weight
                if pred.get("expression"):
                    expression_votes[pred["expression"]] += weight

        best_binary = pick_best_vote(binary_votes, ["有害", "无害"])
        if best_binary is None:
            ensemble_results.append({
                "binary": None,
                "category": None,
                "toxic_type": None,
                "expression": None,
                "raw_output": "ensemble(no_vote)",
            })
            continue

        if best_binary == "无害":
            ensemble_results.append({
                "binary": "无害",
                "category": "无毒",
                "toxic_type": "无毒",
                "expression": "非仇恨",
                "raw_output": f"ensemble({','.join(model_names)})",
            })
            continue

        ensemble_results.append({
            "binary": "有害",
            "category": pick_best_vote(category_votes, HARMFUL_CATEGORIES) or "其他仇恨",
            "toxic_type": pick_best_vote(toxic_type_votes, ["仇恨言论", "一般攻击性"]),
            "expression": pick_best_vote(expression_votes, ["隐式仇恨", "显式仇恨", "转述"]),
            "raw_output": f"ensemble({','.join(model_names)})",
        })

    return ensemble_results


def print_multi_model_comparison(report_dict: Dict[str, dict], title: str = "多 Verifier 横向对比"):
    """打印多个 verifier 及 ensemble 的横向对比表。"""
    if not report_dict:
        return

    names = list(report_dict.keys())
    col_width = max(14, max(len(name) for name in names) + 2)

    print(f"\n{'=' * (22 + col_width * len(names))}")
    print(f"  {title}")
    print(f"{'=' * (22 + col_width * len(names))}")

    header = f"  {'指标':<22}"
    for name in names:
        header += f" {name:^{col_width}}"
    print(header)
    print(f"  {'-' * (22 + col_width * len(names))}")

    def format_row(label: str, getter):
        row = f"  {label:<22}"
        for name in names:
            metrics = report_dict[name]["metrics"]
            row += f" {getter(metrics):^{col_width}.4f}"
        print(row)

    format_row("Overall Acc (%)", lambda m: m["overall_accuracy"])
    format_row("Macro F1", lambda m: m["macro_f1"])
    format_row("Weighted F1", lambda m: m["weighted_f1"])
    format_row("Binary Acc (%)", lambda m: m["binary_metrics"]["accuracy"])
    format_row("Binary Precision", lambda m: m["binary_metrics"]["precision"])
    format_row("Binary Recall", lambda m: m["binary_metrics"]["recall"])
    format_row("Binary F1", lambda m: m["binary_metrics"]["f1_score"])

    print("\n  ── 各类别 F1 ──")
    for cat in LABEL_ORDER:
        format_row(cat, lambda m, category=cat: m["category_metrics"][category]["f1_score"])

    print("\n  ── 无毒 Recall（关键误判指标） ──")
    format_row("无毒 Recall", lambda m: m["category_metrics"]["无毒"]["recall"])
    print(f"{'=' * (22 + col_width * len(names))}")


def build_result_payload(
    api_model: str,
    api_base: str,
    backend: str,
    enable_thinking: bool,
    thinking_budget: Optional[int],
    use_few_shots: bool,
    test_data: str,
    elapsed_seconds: float,
    report: dict,
    texts: List[str],
) -> dict:
    """构造保存到 JSON 的统一结果结构。"""
    for idx, detail in enumerate(report["results_detail"]):
        detail["文本"] = texts[idx]

    return {
        "api_model": api_model,
        "api_base": api_base,
        "backend": backend,
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget if enable_thinking else None,
        "use_few_shots": use_few_shots,
        "test_data": test_data,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "metrics": report["metrics"],
        "results": report["results_detail"],
    }


def evaluate_model_task(
    model_name: str,
    texts: List[str],
    true_labels: List[str],
    api_key: str,
    api_base: str,
    args,
    use_few_shots: bool,
    progress_position: int,
) -> dict:
    """执行单个模型的一整轮评测，便于模型间并发。"""
    enable_thinking = args.enable_thinking and MODEL_REGISTRY.get(model_name, {}).get("supports_thinking", False)
    start_time = time.time()
    predictions = batch_verify_api(
        texts=texts,
        api_key=api_key,
        api_base=api_base,
        api_model=model_name,
        max_workers=args.max_workers,
        enable_thinking=enable_thinking,
        thinking_budget=args.thinking_budget,
        use_few_shots=use_few_shots,
        progress_position=progress_position,
    )
    elapsed = time.time() - start_time
    report = compute_metrics(true_labels, predictions)
    payload = build_result_payload(
        api_model=model_name,
        api_base=api_base,
        backend=args.backend,
        enable_thinking=enable_thinking,
        thinking_budget=args.thinking_budget,
        use_few_shots=use_few_shots,
        test_data=args.test_data,
        elapsed_seconds=elapsed,
        report=report,
        texts=texts,
    )

    return {
        "model_name": model_name,
        "enable_thinking": enable_thinking,
        "elapsed": elapsed,
        "predictions": predictions,
        "report": report,
        "payload": payload,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(true_labels: list, predictions: list) -> dict:
    """
    计算完整评测指标。

    Args:
        true_labels: 真实标签列表 (ALL_CATEGORIES 中的值)
        predictions: 预测结果列表 (parse_classification_output 输出的 dict)

    Returns:
        包含 binary_metrics, category_metrics, confusion_matrix 等的完整报告
    """
    total = len(true_labels)
    correct = 0
    failed_extractions = 0

    # 二分类统计
    bin_tp = bin_tn = bin_fp = bin_fn = 0

    # 类别级统计
    label_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "sample_count": 0})

    # 混淆矩阵
    confusion = defaultdict(lambda: defaultdict(int))

    results_detail = []

    for i, (true_label, pred) in enumerate(zip(true_labels, predictions)):
        true_is_harmful = true_label in HARMFUL_CATEGORIES
        true_binary = "有害" if true_is_harmful else "无害"

        p_binary = pred.get("binary")
        p_category = pred.get("category")

        if p_binary is None:
            failed_extractions += 1
            # 将解析失败视为无害（保守策略）
            p_binary = "无害"
            p_category = "无毒"

        # 映射 predict_label
        if p_binary == "无害":
            predict_label = "无毒"
        else:
            predict_label = p_category if p_category and p_category != "无毒" else "其他仇恨"

        # 混淆矩阵
        confusion[true_label][predict_label] += 1

        # 类别级
        label_stats[true_label]["sample_count"] += 1
        if predict_label == true_label:
            label_stats[true_label]["TP"] += 1
            correct += 1
        else:
            label_stats[true_label]["FN"] += 1
            label_stats[predict_label]["FP"] += 1

        # 二分类
        pred_is_harmful = (p_binary == "有害")
        if true_is_harmful and pred_is_harmful:
            bin_tp += 1
        elif true_is_harmful and not pred_is_harmful:
            bin_fn += 1
        elif not true_is_harmful and pred_is_harmful:
            bin_fp += 1
        else:
            bin_tn += 1

        results_detail.append({
            "index": i,
            "文本": "",  # 稍后填充
            "标签": true_label,
            "predict_label": predict_label,
            "predict_binary": p_binary,
            "predict_toxic_type": pred.get("toxic_type"),
            "predict_expression": pred.get("expression"),
            "response": pred.get("raw_output", ""),
        })

    # 二分类指标
    bin_p = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0
    bin_r = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0
    bin_f1 = 2 * bin_p * bin_r / (bin_p + bin_r) if (bin_p + bin_r) > 0 else 0
    bin_acc = (bin_tp + bin_tn) / total * 100 if total > 0 else 0

    # 类别级指标
    category_metrics = {}
    f1_scores = []
    weighted_p_sum = weighted_r_sum = weighted_f1_sum = 0

    for cat in LABEL_ORDER:
        s = label_stats[cat]
        p = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0
        r = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        n = s["sample_count"]

        category_metrics[cat] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "TP": s["TP"],
            "FP": s["FP"],
            "FN": s["FN"],
            "sample_count": n,
        }
        f1_scores.append(f1)
        weighted_p_sum += p * n
        weighted_r_sum += r * n
        weighted_f1_sum += f1 * n

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0
    weighted_p = weighted_p_sum / total if total > 0 else 0
    weighted_r = weighted_r_sum / total if total > 0 else 0

    # 混淆矩阵
    confusion_matrix = {}
    for true_cat in LABEL_ORDER:
        row = {}
        for pred_cat in LABEL_ORDER:
            row[pred_cat] = confusion[true_cat][pred_cat]
        confusion_matrix[true_cat] = row

    return {
        "metrics": {
            "overall_accuracy": round(correct / total * 100, 4) if total > 0 else 0,
            "correct": correct,
            "total": total,
            "failed_extractions": failed_extractions,
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "weighted_precision": round(weighted_p, 4),
            "weighted_recall": round(weighted_r, 4),
            "binary_metrics": {
                "accuracy": round(bin_acc, 2),
                "precision": round(bin_p, 4),
                "recall": round(bin_r, 4),
                "f1_score": round(bin_f1, 4),
            },
            "category_metrics": category_metrics,
            "confusion_matrix": confusion_matrix,
        },
        "results_detail": results_detail,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 打印报告
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, title: str = "Verifier"):
    """打印格式化的评测报告。"""
    m = metrics["metrics"]
    bm = m["binary_metrics"]
    cm = m["category_metrics"]

    print(f"\n{'=' * 70}")
    print(f"  {title} 评测报告")
    print(f"{'=' * 70}")
    print(f"  总样本: {m['total']}  |  正确: {m['correct']}  "
          f"|  准确率: {m['overall_accuracy']:.2f}%")
    print(f"  解析失败: {m['failed_extractions']}")

    print(f"\n  ── 二分类 (有害/无害) ──")
    print(f"  Accuracy  : {bm['accuracy']:.2f}%")
    print(f"  Precision : {bm['precision']:.4f}")
    print(f"  Recall    : {bm['recall']:.4f}")
    print(f"  F1        : {bm['f1_score']:.4f}")

    print(f"\n  ── 类别级指标 ──")
    print(f"  {'类别':<12} {'Prec':<10} {'Rec':<10} {'F1':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'N':<6}")
    print(f"  {'-' * 68}")
    for cat in LABEL_ORDER:
        s = cm[cat]
        print(f"  {cat:<12} {s['precision']:<10.4f} {s['recall']:<10.4f} "
              f"{s['f1_score']:<10.4f} {s['TP']:<6} {s['FP']:<6} {s['FN']:<6} {s['sample_count']:<6}")

    print(f"\n  Macro-F1    : {m['macro_f1']:.4f}")
    print(f"  Weighted-F1 : {m['weighted_f1']:.4f}")

    # 混淆矩阵
    if "confusion_matrix" in m:
        print(f"\n  ── 混淆矩阵 (行=真实, 列=预测) ──")
        short = {"性别歧视": "性别", "种族歧视": "种族", "地域偏见": "地域",
                 "LGBTQ歧视": "LGBT", "其他仇恨": "其他", "无毒": "无毒"}
        header = "  " + f"{'':>8}" + "".join(f"{short.get(c, c):>8}" for c in LABEL_ORDER)
        print(header)
        for true_cat in LABEL_ORDER:
            row = m["confusion_matrix"][true_cat]
            vals = "".join(f"{row[c]:>8}" for c in LABEL_ORDER)
            print(f"  {short.get(true_cat, true_cat):>8}{vals}")

    print(f"{'=' * 70}")


def print_comparison(verifier_metrics: dict, reviewer_list: list):
    """打印 Verifier vs 多轮 Reviewer 对比表。"""
    print(f"\n{'=' * 90}")
    print(f"  Verifier vs Reviewer 各轮对比")
    print(f"{'=' * 90}")

    vm = verifier_metrics["metrics"]

    # 表头
    header = f"  {'指标':<20} {'Verifier':<12}"
    for name, _ in reviewer_list:
        header += f" {name:<14}"
    print(header)
    print(f"  {'-' * (20 + 12 + 14 * len(reviewer_list))}")

    # 主要指标
    rows = [
        ("Overall Acc (%)", "overall_accuracy"),
        ("Macro F1", "macro_f1"),
        ("Weighted F1", "weighted_f1"),
        ("Binary Acc (%)", ("binary_metrics", "accuracy")),
        ("Binary Precision", ("binary_metrics", "precision")),
        ("Binary Recall", ("binary_metrics", "recall")),
        ("Binary F1", ("binary_metrics", "f1_score")),
    ]

    for label, key in rows:
        if isinstance(key, tuple):
            v_val = vm[key[0]][key[1]]
        else:
            v_val = vm[key]
        line = f"  {label:<20} {v_val:<12.4f}"

        for _, rm in reviewer_list:
            if isinstance(key, tuple):
                r_val = rm[key[0]][key[1]]
            else:
                r_val = rm[key]
            line += f" {r_val:<14.4f}"
        print(line)

    # 类别 F1 对比
    print(f"\n  ── 类别 F1 对比 ──")
    header2 = f"  {'类别':<12} {'Verifier':<12}"
    for name, _ in reviewer_list:
        header2 += f" {name:<14}"
    print(header2)
    print(f"  {'-' * (12 + 12 + 14 * len(reviewer_list))}")

    for cat in LABEL_ORDER:
        v_f1 = vm["category_metrics"][cat]["f1_score"]
        line = f"  {cat:<12} {v_f1:<12.4f}"
        for _, rm in reviewer_list:
            r_f1 = rm["category_metrics"][cat]["f1_score"]
            line += f" {r_f1:<14.4f}"
        print(line)

    # 无毒 Recall 对比 (核心退化指标)
    print(f"\n  ── 无毒 Recall 对比 (核心退化指标) ──")
    v_notox_r = vm["category_metrics"]["无毒"]["recall"]
    line = f"  {'无毒 Recall':<20} {v_notox_r:<12.4f}"
    for name, rm in reviewer_list:
        r_notox_r = rm["category_metrics"]["无毒"]["recall"]
        line += f" {r_notox_r:<14.4f}"
    print(line)

    print(f"{'=' * 90}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 主流程
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verifier (API) 在测试集上的评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 评测模式
    parser.add_argument("--compare", action="store_true",
                        help="比较模式: Verifier 结果 vs 各轮 Reviewer 结果")

    # 评测参数
    parser.add_argument("--test_data", type=str,
                        help="测试集路径 (bench.json / eval_vllm_*.json / parquet)")
    parser.add_argument("--backend", type=str, default="api",
                        choices=["api", "async"],
                        help="API 后端: api(同步) / async(异步)")
    parser.add_argument("--api_model", type=str, default="qwen-plus",
                        help="API 模型名称")
    parser.add_argument("--api_models", nargs="+", type=str,
                        help="多个 API 模型名称，适合同一个 DashScope key 做横向评测")
    parser.add_argument("--api_base", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API Base URL")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="API 并发数")
    parser.add_argument("--max_model_workers", type=int, default=0,
                        help="模型级并发数；0 表示自动（默认最多同时跑 3 个模型）")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="最多评测多少条样本，0 表示全部")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="启用深度思考模式 (DashScope enable_thinking=True)")
    parser.add_argument("--thinking_budget", type=int, default=500,
                        help="思考模式最大 Token 数 (默认 500, 约250-500字; 仅 --enable_thinking 时生效)")
    parser.add_argument("--no_few_shots", action="store_true",
                        help="不使用 few-shot 示例 (默认开启 few-shot，显著提升边界案例准确率)")
    parser.add_argument("--ensemble", action="store_true",
                        help="对多个 verifier 执行投票集成")
    parser.add_argument("--ensemble_weights", nargs="*", default=[],
                        help="集成权重，格式: 模型名=权重，例如 qwen-max=1.5")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="输出目录")

    # 比较模式参数
    parser.add_argument("--verifier_result", type=str,
                        help="Verifier 评测结果 JSON 路径 (比较模式)")
    parser.add_argument("--reviewer_results", nargs="+", type=str,
                        help="各轮 Reviewer 评测结果 JSON 路径列表 (比较模式)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── 比较模式 ──
    if args.compare:
        if not args.verifier_result or not args.reviewer_results:
            print("错误: 比较模式需要 --verifier_result 和 --reviewer_results")
            sys.exit(1)

        with open(args.verifier_result, "r", encoding="utf-8") as f:
            v_data = json.load(f)

        reviewer_list = []
        for rp in args.reviewer_results:
            name = Path(rp).stem
            # 简化名称
            if "round0" in name or "baseline" in name:
                short_name = "Round0(Base)"
            elif "round1" in name:
                short_name = "Round1"
            elif "round2" in name:
                short_name = "Round2"
            elif "round3" in name:
                short_name = "Round3"
            else:
                short_name = name[-15:]
            with open(rp, "r", encoding="utf-8") as f:
                r_data = json.load(f)
            reviewer_list.append((short_name, r_data["metrics"]))

        print_report(v_data, title=f"Verifier ({v_data.get('api_model', 'API')})")
        print_comparison(v_data, reviewer_list)
        return

    # ── 评测模式 ──
    if not args.test_data:
        print("错误: 请指定 --test_data")
        sys.exit(1)

    api_key = os.environ.get("VERIFIER_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("错误: 请设置环境变量 VERIFIER_API_KEY 或 DASHSCOPE_API_KEY")
        sys.exit(1)

    api_base = args.api_base

    # 加载测试数据
    print(f"\n[1] 加载测试集: {args.test_data}")
    test_data = load_test_data(args.test_data)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
    print(f"    共 {len(test_data)} 条")

    # 类别分布
    dist = Counter(d["标签"] for d in test_data)
    for cat in LABEL_ORDER:
        print(f"    {cat}: {dist.get(cat, 0)}")

    texts = [d["文本"] for d in test_data]
    true_labels = [d["标签"] for d in test_data]

    # API 验证
    use_few_shots = not args.no_few_shots
    thinking_info = f", thinking_budget={args.thinking_budget}" if args.enable_thinking else ""
    few_shot_info = "" if use_few_shots else ", no_few_shots"

    model_names = args.api_models if args.api_models else [args.api_model]
    if args.ensemble and not args.api_models:
        model_names = ["qwen-plus", "qwen-max", "qwen-turbo"]

    os.makedirs(args.output_dir, exist_ok=True)
    all_reports = {}
    all_payloads = {}
    all_predictions = {}
    elapsed_map = {}
    total_eval_start = time.time()

    model_parallelism = args.max_model_workers if args.max_model_workers > 0 else min(len(model_names), 3)
    print(f"\n[2] 调用 API 进行验证 (模型数={len(model_names)}, 单模型并发={args.max_workers}, 模型级并发={model_parallelism}{thinking_info}{few_shot_info})...")
    if use_few_shots:
        print(f"    使用 Verifier 专用 system prompt + {7} 个 few-shot 示例 (7 shots)")
    else:
        print(f"    使用 Verifier 专用 system prompt，不使用 few-shot")

    for model_name in model_names:
        enable_thinking = args.enable_thinking and MODEL_REGISTRY.get(model_name, {}).get("supports_thinking", False)
        print(f"    -> 提交模型任务: {model_name} (thinking={'on' if enable_thinking else 'off'})")

    task_results = {}
    with ThreadPoolExecutor(max_workers=model_parallelism) as model_executor:
        future_to_model = {}
        for progress_position, model_name in enumerate(model_names):
            future = model_executor.submit(
                evaluate_model_task,
                model_name,
                texts,
                true_labels,
                api_key,
                api_base,
                args,
                use_few_shots,
                progress_position,
            )
            future_to_model[future] = model_name

        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            result = future.result()
            task_results[model_name] = result
            elapsed = result["elapsed"]
            elapsed_map[model_name] = round(elapsed, 1)
            all_predictions[model_name] = result["predictions"]
            all_reports[model_name] = result["report"]
            all_payloads[model_name] = result["payload"]

            print(f"\n    -> 模型完成: {model_name}，耗时 {elapsed:.1f}s ({len(texts)/elapsed:.1f} 条/秒)")

    for model_name in model_names:
        result = task_results[model_name]
        enable_thinking = result["enable_thinking"]
        report = result["report"]
        payload = result["payload"]

        print_report(report, title=f"Verifier ({model_name})")

        suffix_think = f"_think{args.thinking_budget}" if enable_thinking else ""
        suffix_shot = "" if use_few_shots else "_noshot"
        out_path = Path(args.output_dir) / f"eval_verifier_{model_name}{suffix_think}{suffix_shot}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"    单模型结果保存至: {out_path}")

    ensemble_out_path = None
    if args.ensemble and len(model_names) > 1:
        print(f"\n[3] 计算多 Verifier 集成投票结果...")
        ensemble_weights = {}
        for item in args.ensemble_weights:
            if "=" not in item:
                continue
            model_name, weight = item.split("=", 1)
            try:
                ensemble_weights[model_name.strip()] = float(weight)
            except ValueError:
                print(f"    [警告] 非法权重配置，已忽略: {item}")

        ensemble_predictions_list = ensemble_predictions(
            all_predictions,
            model_weights=ensemble_weights or None,
        )
        ensemble_report = compute_metrics(true_labels, ensemble_predictions_list)
        ensemble_payload = build_result_payload(
            api_model="ensemble",
            api_base=api_base,
            backend=args.backend,
            enable_thinking=False,
            thinking_budget=None,
            use_few_shots=use_few_shots,
            test_data=args.test_data,
            elapsed_seconds=time.time() - total_eval_start,
            report=ensemble_report,
            texts=texts,
        )
        ensemble_payload["ensemble_models"] = model_names
        ensemble_payload["ensemble_weights"] = ensemble_weights or None

        all_reports["Ensemble"] = ensemble_report
        all_payloads["Ensemble"] = ensemble_payload

        print_report(ensemble_report, title=f"Verifier Ensemble ({', '.join(model_names)})")
        ensemble_out_path = Path(args.output_dir) / "eval_verifier_ensemble.json"
        with open(ensemble_out_path, "w", encoding="utf-8") as f:
            json.dump(ensemble_payload, f, ensure_ascii=False, indent=2)
        print(f"    集成结果保存至: {ensemble_out_path}")

    if len(all_reports) > 1:
        print_multi_model_comparison(all_reports)

        summary_path = Path(args.output_dir) / "eval_verifier_multi_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_data": args.test_data,
                "api_base": api_base,
                "backend": args.backend,
                "models": model_names,
                "use_few_shots": use_few_shots,
                "wall_clock_seconds": round(time.time() - total_eval_start, 1),
                "elapsed_seconds": elapsed_map,
                "ensemble_enabled": args.ensemble and len(model_names) > 1,
                "ensemble_result": str(ensemble_out_path) if ensemble_out_path else None,
                "reports": all_payloads,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 汇总结果保存至: {summary_path}")

    primary_result_path = ensemble_out_path if ensemble_out_path else Path(args.output_dir) / f"eval_verifier_{model_names[0]}{'_think' + str(args.thinking_budget) if args.enable_thinking and MODEL_REGISTRY.get(model_names[0], {}).get('supports_thinking', False) else ''}{'' if use_few_shots else '_noshot'}.json"
    print(f"\n✅ 主结果文件: {primary_result_path}")

    # 提示比较命令
    print(f"\n💡 可使用以下命令进行 Verifier vs Reviewer 对比:")
    print(f"   python {Path(__file__).name} --compare \\")
    print(f"       --verifier_result {primary_result_path} \\")
    print(f"       --reviewer_results \\")
    print(f"           eval_vllm_npu_reviewer_3B_reviewer_round0_baseline.json \\")
    print(f"           eval_vllm_npu_reviewer_reviewer_round1.json \\")
    print(f"           eval_vllm_npu_reviewer_reviewer_round2.json \\")
    print(f"           eval_vllm_npu_reviewer_reviewer_round3.json")


if __name__ == "__main__":
    main()
