#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型 API 并行测试集评估
==========================
在同一测试集上并行调用多个 API 模型，逐一评估并横向对比性能。

与 eval_verifier_api.py 的区别:
  - 支持同时指定多个模型，一次跑完全部对比
  - 输出统一的横向对比表 (Accuracy / F1 / 各类别指标)
  - 支持多数投票 Ensemble 模式，作为额外一列对比
  - 复用 eval_verifier_api.py 中的 call_api_single / compute_metrics / print_report

模型配置（通过 --models 指定，格式: 模型名[:base_url[:key_env]]）:
  内置快捷名（需设置对应环境变量）:
    qwen-plus     → DASHSCOPE_API_KEY
    qwen-max      → DASHSCOPE_API_KEY
    qwen-turbo    → DASHSCOPE_API_KEY
    qwq-32b       → DASHSCOPE_API_KEY  (thinking 模式)
    deepseek-chat → DEEPSEEK_API_KEY
    deepseek-reasoner → DEEPSEEK_API_KEY  (thinking 模式)

用法示例:
  # 最简: 两个模型并行测
  python eval_multi_api.py \\
      --test_data ../../origin_data/bench.json \\
      --models qwen-plus qwen-max \\
      --output_dir ../../eval_results/multi_api

  # 指定 API key（也可用环境变量）
  python eval_multi_api.py \\
      --test_data ../../origin_data/bench.json \\
      --models qwen-plus deepseek-chat \\
      --api_keys DASHSCOPE_API_KEY=sk-aaa DEEPSEEK_API_KEY=sk-bbb \\
      --output_dir ../../eval_results/multi_api \\
      --max_workers 8

  # 加上 Ensemble 投票列
  python eval_multi_api.py \\
      --test_data ../../origin_data/bench.json \\
      --models qwen-plus qwen-max deepseek-chat \\
      --ensemble \\
      --output_dir ../../eval_results/multi_api

  # 与已有 Reviewer 结果对比（把本地 Reviewer 的结果 json 也带入对比表）
  python eval_multi_api.py \\
      --test_data ../../origin_data/bench.json \\
      --models qwen-plus deepseek-chat \\
      --compare_with \\
          ../../eval_vllm_npu_reviewer_3B_reviewer_round0_baseline.json \\
          ../../eval_vllm_npu_reviewer_reviewer_round1.json \\
      --output_dir ../../eval_results/multi_api

  # 只跑前 50 条做快速测试
  python eval_multi_api.py \\
      --test_data ../../origin_data/bench.json \\
      --models qwen-plus \\
      --max_samples 50
"""

import os
import sys
import json
import time
import argparse
import logging
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# ── 导入本包 constants ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from constants import (
    HARMFUL_CATEGORIES, ALL_CATEGORIES,
    build_verifier_messages,
    parse_classification_output,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_ORDER = ALL_CATEGORIES  # 6 类：5 有害 + 无毒


# ══════════════════════════════════════════════════════════════════════════════
# 1. 内置模型注册表
# ══════════════════════════════════════════════════════════════════════════════

# 模型名 → (默认 base_url, API key 环境变量名, 是否支持 thinking)
_MODEL_REGISTRY: Dict[str, Tuple[str, str, bool]] = {
    "qwen-plus":          ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", False),
    "qwen-max":           ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", False),
    "qwen-turbo":         ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", False),
    "qwq-32b":            ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", True),
    "deepseek-chat":      ("https://api.deepseek.com/v1",                        "DEEPSEEK_API_KEY",  False),
    "deepseek-reasoner":  ("https://api.deepseek.com/v1",                        "DEEPSEEK_API_KEY",  True),
    "gpt-4o-mini":        ("https://api.openai.com/v1",                          "OPENAI_API_KEY",    False),
    "gpt-4o":             ("https://api.openai.com/v1",                          "OPENAI_API_KEY",    False),
}


def resolve_model_config(model_spec: str, extra_keys: Dict[str, str]) -> Optional[Dict]:
    """
    解析模型配置字符串，格式：
      model_name                 (使用注册表默认值)
      model_name:base_url        (自定义 base_url，key 从注册表环境变量取)
      model_name:base_url:KEY_ENV (全自定义)
    """
    from openai import OpenAI

    parts = model_spec.split(":", 2)
    name = parts[0].strip()

    if name in _MODEL_REGISTRY:
        default_base, key_env, supports_thinking = _MODEL_REGISTRY[name]
    else:
        default_base = "https://api.openai.com/v1"
        key_env = "OPENAI_API_KEY"
        supports_thinking = False

    base_url = parts[1] if len(parts) > 1 else default_base
    if len(parts) > 2:
        key_env = parts[2]

    # 取 API key：优先 --api_keys 参数 > 环境变量
    api_key = extra_keys.get(key_env, "") or os.environ.get(key_env, "")
    if not api_key:
        logger.warning(f"模型 '{name}' 未找到 API key (env={key_env})，跳过")
        return None

    return {
        "name":              name,
        "base_url":          base_url,
        "supports_thinking": supports_thinking,
        "client":            OpenAI(api_key=api_key, base_url=base_url),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_test_data(path: str) -> List[Dict]:
    """
    加载测试集。支持多种格式，统一返回 [{"文本": ..., "标签": ...}, ...]。
    """
    import pandas as pd

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        col_text = "文本" if "文本" in df.columns else "original_text"
        col_cat  = "标签" if "标签" in df.columns else "category"
        return [{"文本": str(row[col_text]).strip(), "标签": str(row[col_cat])}
                for _, row in df.iterrows()]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # eval_vllm_*.json 格式
    if isinstance(data, dict) and "results" in data:
        items = data["results"]
        return [{"文本": r.get("文本", "").strip(), "标签": r.get("标签", "")} for r in items]

    # bench.json / array 格式
    if isinstance(data, list) and len(data) > 0:
        if "文本" in data[0]:
            return [{"文本": r["文本"].strip(), "标签": r.get("标签", r.get("label", ""))} for r in data]
        if "text" in data[0]:
            return [{"文本": r["text"].strip(), "标签": r.get("label", r.get("标签", ""))} for r in data]

    raise ValueError(f"无法识别测试集格式: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 单模型 API 调用（带重试）
# ══════════════════════════════════════════════════════════════════════════════

def call_one(
    client,
    model_name: str,
    text: str,
    supports_thinking: bool = False,
    max_retries: int = 3,
    use_few_shots: bool = True,
) -> Dict:
    """调用单个模型，返回 parse_classification_output 结果 + raw_output。"""
    messages = build_verifier_messages(text, use_few_shots=use_few_shots)

    for attempt in range(max_retries):
        try:
            if supports_thinking:
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                    extra_body={"enable_thinking": True, "thinking_budget": 300},
                    stream=True,
                )
                raw = ""
                for chunk in stream:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if getattr(delta, "content", None):
                            raw += delta.content
                raw = raw.strip()
            else:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=150,
                )
                raw = resp.choices[0].message.content.strip()

            parsed = parse_classification_output(raw)
            parsed["raw_output"] = raw
            return parsed

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.debug(f"[{model_name}] 失败: {e}")
                return {"binary": None, "category": None,
                        "toxic_type": None, "expression": None,
                        "raw_output": f"ERROR: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# 4. 单模型批量评估
# ══════════════════════════════════════════════════════════════════════════════

def run_one_model(
    model_cfg: Dict,
    texts: List[str],
    max_workers: int = 8,
    use_few_shots: bool = True,
) -> List[Dict]:
    """并发调用单个模型，返回与 texts 等长的预测列表。"""
    results = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_one,
                model_cfg["client"], model_cfg["name"], text,
                model_cfg["supports_thinking"], 3, use_few_shots,
            ): i
            for i, text in enumerate(texts) if text.strip()
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc=f"  [{model_cfg['name']}]", leave=True,
        ):
            idx = futures[future]
            results[idx] = future.result()

    # 空文本兜底
    for i in range(len(results)):
        if results[i] is None:
            results[i] = {"binary": "无害", "category": "无毒",
                          "toxic_type": "无毒", "expression": "非仇恨",
                          "raw_output": ""}
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. 指标计算（复用 eval_verifier_api.py 逻辑）
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(true_labels: List[str], predictions: List[Dict]) -> Dict:
    total = len(true_labels)
    correct = failed = 0
    bin_tp = bin_tn = bin_fp = bin_fn = 0
    label_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "sample_count": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for true_label, pred in zip(true_labels, predictions):
        true_is_harmful = true_label in HARMFUL_CATEGORIES

        p_binary   = pred.get("binary")
        p_category = pred.get("category")

        if p_binary is None:
            failed += 1
            p_binary   = "无害"
            p_category = "无毒"

        predict_label = ("无毒" if p_binary == "无害"
                         else (p_category if p_category and p_category != "无毒" else "其他仇恨"))

        confusion[true_label][predict_label] += 1
        label_stats[true_label]["sample_count"] += 1

        if predict_label == true_label:
            label_stats[true_label]["TP"] += 1
            correct += 1
        else:
            label_stats[true_label]["FN"] += 1
            label_stats[predict_label]["FP"] += 1

        pred_is_harmful = (p_binary == "有害")
        if   true_is_harmful and pred_is_harmful:  bin_tp += 1
        elif true_is_harmful:                       bin_fn += 1
        elif pred_is_harmful:                       bin_fp += 1
        else:                                       bin_tn += 1

    bin_p  = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
    bin_r  = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
    bin_f1 = 2 * bin_p * bin_r / (bin_p + bin_r) if (bin_p + bin_r) > 0 else 0.0
    bin_acc = (bin_tp + bin_tn) / total * 100 if total > 0 else 0.0

    cat_metrics = {}
    f1_list = []
    w_p = w_r = w_f1 = 0.0
    for cat in LABEL_ORDER:
        s = label_stats[cat]
        p  = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0.0
        r  = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        n  = s["sample_count"]
        cat_metrics[cat] = {
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "TP": s["TP"], "FP": s["FP"], "FN": s["FN"], "N": n,
        }
        f1_list.append(f1)
        w_p  += p  * n
        w_r  += r  * n
        w_f1 += f1 * n

    macro_f1    = sum(f1_list) / len(f1_list) if f1_list else 0.0
    weighted_f1 = w_f1 / total if total > 0 else 0.0
    weighted_p  = w_p  / total if total > 0 else 0.0
    weighted_r  = w_r  / total if total > 0 else 0.0

    confusion_out = {tc: dict(confusion[tc]) for tc in LABEL_ORDER}

    return {
        "overall_accuracy": round(correct / total * 100, 4) if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "failed_extractions": failed,
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(weighted_f1, 4),
        "weighted_precision": round(weighted_p, 4),
        "weighted_recall":  round(weighted_r, 4),
        "binary": {
            "accuracy":  round(bin_acc, 2),
            "precision": round(bin_p, 4),
            "recall":    round(bin_r, 4),
            "f1":        round(bin_f1, 4),
            "TP": bin_tp, "FP": bin_fp, "FN": bin_fn, "TN": bin_tn,
        },
        "per_category": cat_metrics,
        "confusion_matrix": confusion_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Ensemble（多数投票）
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_predictions(
    all_preds: Dict[str, List[Dict]],
    model_weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    对多模型预测结果进行多数投票。
    all_preds: {model_name: [pred1, pred2, ...]}
    """
    model_names = list(all_preds.keys())
    n = len(next(iter(all_preds.values())))
    ensembled = []

    for i in range(n):
        binary_votes:   Dict[str, float] = defaultdict(float)
        category_votes: Dict[str, float] = defaultdict(float)
        tt_votes:       Dict[str, float] = defaultdict(float)
        ex_votes:       Dict[str, float] = defaultdict(float)

        for name in model_names:
            pred = all_preds[name][i]
            w = (model_weights or {}).get(name, 1.0)
            if pred.get("binary"):
                binary_votes[pred["binary"]] += w
                if pred["binary"] == "有害":
                    if pred.get("category"):
                        category_votes[pred["category"]] += w
                    if pred.get("toxic_type"):
                        tt_votes[pred["toxic_type"]] += w
                    if pred.get("expression"):
                        ex_votes[pred["expression"]] += w

        if not binary_votes:
            ensembled.append({"binary": None, "category": None,
                              "toxic_type": None, "expression": None,
                              "raw_output": "no_vote"})
            continue

        ens_binary   = max(binary_votes,   key=binary_votes.get)
        ens_category = max(category_votes, key=category_votes.get) if category_votes else "无毒"
        ens_tt       = max(tt_votes,       key=tt_votes.get)       if tt_votes       else None
        ens_ex       = max(ex_votes,       key=ex_votes.get)       if ex_votes       else None

        ensembled.append({
            "binary":     ens_binary,
            "category":   ens_category if ens_binary == "有害" else "无毒",
            "toxic_type": ens_tt,
            "expression": ens_ex,
            "raw_output": f"ensemble({','.join(model_names)})",
        })

    return ensembled


# ══════════════════════════════════════════════════════════════════════════════
# 7. 报告打印
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, pct=False) -> str:
    if v is None:
        return "  N/A   "
    if pct:
        return f"{v:.2f}%"
    return f"{v:.4f}"


def print_single_report(m: Dict, title: str):
    """打印单模型完整报告（与 eval_verifier_api 格式一致）。"""
    bm = m["binary"]
    cm = m["per_category"]
    short = {"性别歧视": "性别", "种族歧视": "种族", "地域偏见": "地域",
             "LGBTQ歧视": "LGBT", "其他仇恨": "其他", "无毒": "无毒"}

    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")
    print(f"  总计: {m['total']}  |  正确: {m['correct']}  "
          f"|  整体准确率: {m['overall_accuracy']:.2f}%  |  解析失败: {m['failed_extractions']}")
    print(f"  Macro-F1: {m['macro_f1']:.4f}   Weighted-F1: {m['weighted_f1']:.4f}")

    print(f"\n  ── 二分类 ──")
    print(f"  Acc={bm['accuracy']:.2f}%  P={bm['precision']:.4f}  "
          f"R={bm['recall']:.4f}  F1={bm['f1']:.4f}  "
          f"(TP={bm['TP']} FP={bm['FP']} FN={bm['FN']} TN={bm['TN']})")

    print(f"\n  ── 类别级指标 ──")
    print(f"  {'类别':<10} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'N':>5}")
    print(f"  {'-' * 60}")
    for cat in LABEL_ORDER:
        s = cm[cat]
        print(f"  {cat:<10} {s['precision']:>8.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} "
              f"{s['TP']:>5} {s['FP']:>5} {s['FN']:>5} {s['N']:>5}")

    print(f"\n  ── 混淆矩阵 (行=真实标签, 列=预测标签) ──")
    header = f"  {'':>8}" + "".join(f"{short.get(c, c):>7}" for c in LABEL_ORDER)
    print(header)
    conf = m["confusion_matrix"]
    for tc in LABEL_ORDER:
        row_vals = "".join(f"{conf.get(tc, {}).get(pc, 0):>7}" for pc in LABEL_ORDER)
        print(f"  {short.get(tc, tc):>8}{row_vals}")
    print(f"{'=' * 68}")


def print_comparison_table(results_dict: Dict[str, Dict], title: str = "API 模型横向对比"):
    """
    输出横向对比表。
    results_dict: {model_name_or_label: metrics_dict}
    """
    names = list(results_dict.keys())
    col_w = max(14, max(len(n) for n in names) + 2)

    sep = "=" * (22 + col_w * len(names))
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

    # 表头
    hdr = f"  {'指标':<22}"
    for n in names:
        hdr += f" {n:^{col_w}}"
    print(hdr)
    print(f"  {'-' * (22 + col_w * len(names))}")

    def _row(label, getter):
        line = f"  {label:<22}"
        for n in names:
            try:
                v = getter(results_dict[n])
                line += f" {v:^{col_w}.4f}"
            except Exception:
                line += f" {'N/A':^{col_w}}"
        return line

    # 主要指标
    print(_row("Overall Acc (%)",    lambda m: m["overall_accuracy"]))
    print(_row("Macro-F1",           lambda m: m["macro_f1"]))
    print(_row("Weighted-F1",        lambda m: m["weighted_f1"]))
    print(_row("Binary Acc (%)",     lambda m: m["binary"]["accuracy"]))
    print(_row("Binary Precision",   lambda m: m["binary"]["precision"]))
    print(_row("Binary Recall",      lambda m: m["binary"]["recall"]))
    print(_row("Binary F1",          lambda m: m["binary"]["f1"]))

    print(f"\n  ── 各类别 F1 ──")
    for cat in LABEL_ORDER:
        print(_row(cat, lambda m, c=cat: m["per_category"][c]["f1"]))

    print(f"\n  ── 无毒 Recall（误判率指标）──")
    print(_row("无毒 Recall",     lambda m: m["per_category"]["无毒"]["recall"]))
    print(_row("无毒 Precision",  lambda m: m["per_category"]["无毒"]["precision"]))

    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 加载已有 Reviewer 结果
# ══════════════════════════════════════════════════════════════════════════════

def load_reviewer_result(path: str) -> Tuple[str, Dict]:
    """
    加载 eval_vllm_*.json 格式的已有 Reviewer 评测结果，
    转换为与 compute_metrics 输出相同的结构。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = data.get("metrics", data)

    # 原始格式：binary_metrics / category_metrics → 统一为 binary / per_category
    def _get(d, *keys):
        for k in keys:
            if k in d:
                d = d[k]
            else:
                return None
        return d

    binary = m.get("binary_metrics") or m.get("binary") or {}
    cat_raw = m.get("category_metrics") or m.get("per_category") or {}

    per_category = {}
    for cat in LABEL_ORDER:
        src = cat_raw.get(cat, {})
        per_category[cat] = {
            "precision": src.get("precision", 0.0),
            "recall":    src.get("recall",    0.0),
            "f1":        src.get("f1_score", src.get("f1", 0.0)),
            "TP": src.get("TP", 0), "FP": src.get("FP", 0),
            "FN": src.get("FN", 0), "N":  src.get("sample_count", src.get("N", 0)),
        }

    out = {
        "overall_accuracy":   m.get("overall_accuracy", 0.0),
        "correct":            m.get("correct", 0),
        "total":              m.get("total", 0),
        "failed_extractions": m.get("failed_extractions", 0),
        "macro_f1":           m.get("macro_f1", 0.0),
        "weighted_f1":        m.get("weighted_f1", 0.0),
        "weighted_precision": m.get("weighted_precision", 0.0),
        "weighted_recall":    m.get("weighted_recall", 0.0),
        "binary": {
            "accuracy":  binary.get("accuracy", 0.0),
            "precision": binary.get("precision", 0.0),
            "recall":    binary.get("recall", 0.0),
            "f1":        binary.get("f1_score", binary.get("f1", 0.0)),
            "TP": 0, "FP": 0, "FN": 0, "TN": 0,
        },
        "per_category": per_category,
        "confusion_matrix": {},
    }

    # 自动提取文件名作为标签
    stem = Path(path).stem
    if "round0" in stem or "baseline" in stem:
        label = "Reviewer-R0"
    elif "round1" in stem:
        label = "Reviewer-R1"
    elif "round2" in stem:
        label = "Reviewer-R2"
    elif "round3" in stem:
        label = "Reviewer-R3"
    else:
        label = stem[-15:]

    return label, out


# ══════════════════════════════════════════════════════════════════════════════
# 9. 主流程
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="多模型 API 并行测试集评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--test_data",   required=True, help="测试集路径")
    p.add_argument("--models",      nargs="+",     required=True,
                   help="模型列表，格式: 模型名 或 模型名:base_url 或 模型名:base_url:KEY_ENV")
    p.add_argument("--api_keys",    nargs="*",     default=[],
                   help="API key 覆盖，格式: ENV_VAR=value (如 DASHSCOPE_API_KEY=sk-xxx)")
    p.add_argument("--output_dir",  default="./eval_multi_api_results",
                   help="结果输出目录")
    p.add_argument("--max_workers", type=int, default=8,
                   help="每个模型的并发请求数")
    p.add_argument("--max_samples", type=int, default=0,
                   help="最大测试条数 (0=全部，>0 用于快速调试)")
    p.add_argument("--ensemble",    action="store_true",
                   help="额外计算多数投票 Ensemble 列")
    p.add_argument("--ensemble_weights", nargs="*", default=[],
                   help="Ensemble 权重，格式: 模型名=权重 (如 qwen-max=2.0 qwen-plus=1.0)")
    p.add_argument("--compare_with", nargs="*", default=[],
                   help="额外对比的已有 Reviewer 结果 JSON 路径")
    p.add_argument("--no_few_shots", action="store_true",
                   help="禁用 few-shot 示例")
    p.add_argument("--save_per_model", action="store_true",
                   help="为每个模型单独保存完整预测结果 JSON")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  多模型 API 并行测试集评估")
    logger.info("=" * 60)

    # ── 解析 api_keys 覆盖 ──
    extra_keys: Dict[str, str] = {}
    for kv in (args.api_keys or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra_keys[k.strip()] = v.strip()
            os.environ[k.strip()] = v.strip()  # 同时写入环境变量

    # ── 初始化模型客户端 ──
    logger.info("\n[1] 初始化模型...")
    model_configs = []
    for spec in args.models:
        cfg = resolve_model_config(spec, extra_keys)
        if cfg:
            model_configs.append(cfg)
            logger.info(f"  ✓ {cfg['name']}  base={cfg['base_url'][:40]}...")
    if not model_configs:
        logger.error("没有可用模型，退出")
        sys.exit(1)

    # ── 加载测试集 ──
    logger.info(f"\n[2] 加载测试集: {args.test_data}")
    test_data = load_test_data(args.test_data)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
        logger.info(f"  (截断至 {len(test_data)} 条)")
    logger.info(f"  共 {len(test_data)} 条")

    dist = Counter(d["标签"] for d in test_data)
    for cat in LABEL_ORDER:
        logger.info(f"  {cat}: {dist.get(cat, 0)}")

    texts       = [d["文本"] for d in test_data]
    true_labels = [d["标签"] for d in test_data]
    use_few_shots = not args.no_few_shots

    # ── 逐模型评估 ──
    all_preds:   Dict[str, List[Dict]] = {}
    all_metrics: Dict[str, Dict]       = {}
    timings:     Dict[str, float]      = {}

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for cfg in model_configs:
        name = cfg["name"]
        logger.info(f"\n[3] 评估模型: {name}")
        t0 = time.time()
        preds = run_one_model(cfg, texts, args.max_workers, use_few_shots)
        elapsed = time.time() - t0
        timings[name] = elapsed
        logger.info(f"  完成  耗时={elapsed:.1f}s  速度={len(texts)/elapsed:.1f}条/s")

        all_preds[name] = preds
        all_metrics[name] = compute_metrics(true_labels, preds)

        # 单模型完整报告
        print_single_report(all_metrics[name], title=f"模型: {name}")

        # 可选保存每模型结果
        if args.save_per_model:
            out_path = Path(args.output_dir) / f"eval_{name.replace('/', '_')}.json"
            result_rows = []
            for i, (pred, text) in enumerate(zip(preds, texts)):
                result_rows.append({
                    "index": i,
                    "文本": text,
                    "标签": true_labels[i],
                    "predict_binary":   pred.get("binary"),
                    "predict_category": pred.get("category"),
                    "predict_toxic_type": pred.get("toxic_type"),
                    "predict_expression": pred.get("expression"),
                    "raw_output": pred.get("raw_output", ""),
                })
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model": name,
                    "use_few_shots": use_few_shots,
                    "elapsed_s": round(elapsed, 1),
                    "metrics": all_metrics[name],
                    "results": result_rows,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"  结果保存: {out_path}")

    # ── Ensemble ──
    if args.ensemble and len(model_configs) > 1:
        logger.info("\n[4] 计算 Ensemble 投票...")
        ens_weights = None
        if args.ensemble_weights:
            ens_weights = {}
            for ew in args.ensemble_weights:
                if "=" in ew:
                    mn, wv = ew.split("=", 1)
                    ens_weights[mn.strip()] = float(wv)
        ens_preds = ensemble_predictions(all_preds, ens_weights)
        all_metrics["[Ensemble]"] = compute_metrics(true_labels, ens_preds)
        print_single_report(all_metrics["[Ensemble]"], title="Ensemble 多数投票")

    # ── 加载已有 Reviewer 结果 ──
    if args.compare_with:
        logger.info("\n[5] 加载已有 Reviewer 结果...")
        for path in args.compare_with:
            try:
                label, rm = load_reviewer_result(path)
                all_metrics[label] = rm
                logger.info(f"  ✓ {label}  从 {path} 加载")
            except Exception as e:
                logger.warning(f"  加载失败 {path}: {e}")

    # ── 横向对比表 ──
    print_comparison_table(all_metrics, title="API 模型 vs Reviewer 横向对比")

    # ── 保存汇总 ──
    summary_path = Path(args.output_dir) / "summary.json"
    summary = {
        "test_data":     args.test_data,
        "models_tested": [c["name"] for c in model_configs],
        "use_few_shots": use_few_shots,
        "timings_s":     timings,
        "metrics":       all_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 汇总结果保存至: {summary_path}")


if __name__ == "__main__":
    main()
