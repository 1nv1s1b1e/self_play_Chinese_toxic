#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型 API 协同评估 (Ensemble Labeler)
=======================================
用多个大模型对数据集进行联合标注/评估，通过投票机制提高标注准确率。

主要用途:
  1. eval_dataset  — 对原始 ToxiCN 训练/测试集用多模型验证标注质量
  2. eval_challenger — 评估 Challenger 生成文本是否对上目标标签 (off-target 检测)
  3. label_only    — 对无标签数据纯标注，输出高置信度结果

投票机制:
  - 每个模型独立调用，得到 {binary, category} 预测
  - 对 binary (有害/无害) 进行多数票
  - 对 category 进行加权票（权重可配置）
  - confidence = 模型间一致率 (0~1)
  - conflict_flag = True 表示模型间严重分歧，建议人工复审

支持模型配置（通过 --models 或配置文件）:
  qwen-plus      (DashScope)
  qwen-max       (DashScope)
  deepseek-v3    (DeepSeek)
  deepseek-r1    (DeepSeek, 带 thinking)
  gpt-4o-mini    (OpenAI)
  ...

用法:
  # 验证 challenger 生成数据是否对上标签
  python multi_model_eval.py \\
    --mode eval_challenger \\
    --input ../../challenger_gen_3B.jsonl \\
    --output ../../eval_results/multi_model_challenger.json \\
    --models qwen-plus qwen-max deepseek-v3 \\
    --api_key_qwen sk-xxx \\
    --api_key_deepseek sk-yyy \\
    --workers 8

  # 验证数据集标注质量
  python multi_model_eval.py \\
    --mode eval_dataset \\
    --input ../../origin_data/bench.json \\
    --output ../../eval_results/multi_model_dataset.json \\
    --models qwen-plus deepseek-v3 \\
    --api_key_qwen sk-xxx \\
    --api_key_deepseek sk-yyy
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ── 导入本包 constants ──
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


# ═══════════════════════════════════════════════════════════════════════════════
# 模型配置注册表
# ═══════════════════════════════════════════════════════════════════════════════

# 模型名 → (api_base, 需要的 api_key 参数名, 是否支持 thinking)
MODEL_REGISTRY: Dict[str, Tuple[str, str, bool]] = {
    "qwen-plus":         ("https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_qwen",     False),
    "qwen-max":          ("https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_qwen",     False),
    "qwen-turbo":        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_qwen",     False),
    "qwq-32b":           ("https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_qwen",     True),
    "deepseek-chat":     ("https://api.deepseek.com/v1",                        "api_key_deepseek", False),
    "deepseek-reasoner": ("https://api.deepseek.com/v1",                        "api_key_deepseek", True),
    "gpt-4o-mini":       ("https://api.openai.com/v1",                          "api_key_openai",   False),
    "gpt-4o":            ("https://api.openai.com/v1",                          "api_key_openai",   False),
}


def build_model_clients(model_names: List[str], args: argparse.Namespace):
    """
    根据模型名和 args 中的 API key，构建 (model_name, client, supports_thinking) 列表。
    """
    from openai import OpenAI

    configs = []
    for name in model_names:
        if name not in MODEL_REGISTRY:
            logger.warning(f"未知模型 '{name}'，跳过。已知: {list(MODEL_REGISTRY)}")
            continue
        api_base, key_attr, supports_thinking = MODEL_REGISTRY[name]
        api_key = getattr(args, key_attr, None) or os.environ.get(key_attr.upper(), "")
        if not api_key:
            logger.warning(f"模型 '{name}' 缺少 API key ({key_attr})，跳过")
            continue
        client = OpenAI(api_key=api_key, base_url=api_base)
        configs.append({
            "name":              name,
            "client":            client,
            "supports_thinking": supports_thinking,
        })
        logger.info(f"  ✓ 已注册模型: {name}  (base={api_base[:30]}...)")

    if not configs:
        raise ValueError("没有可用的模型，请检查 --models 和 API key 设置")
    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# 单模型 API 调用
# ═══════════════════════════════════════════════════════════════════════════════

def call_single_model(
    client,
    model_name: str,
    text: str,
    supports_thinking: bool = False,
    max_retries: int = 3,
    use_few_shots: bool = True,
) -> Dict:
    """
    对单条文本调用一个模型，返回解析后的分类结果。
    """
    messages = build_verifier_messages(text, use_few_shots=use_few_shots)

    for attempt in range(max_retries):
        try:
            if supports_thinking:
                # 流式思考模式
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=512,
                    extra_body={"enable_thinking": True, "thinking_budget": 200},
                    stream=True,
                )
                content = ""
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        content += delta.content
                raw = content.strip()
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
            parsed["model"] = model_name
            return parsed

        except Exception as e:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                logger.warning(f"[{model_name}] 调用失败: {e}")
                return {
                    "binary": None, "category": None,
                    "toxic_type": None, "expression": None,
                    "raw_output": f"ERROR: {e}", "model": model_name,
                }


# ═══════════════════════════════════════════════════════════════════════════════
# 多模型协同：对单条文本调用所有模型
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_single(
    model_configs: List[Dict],
    text: str,
    model_weights: Optional[Dict[str, float]] = None,
    use_few_shots: bool = True,
) -> Dict:
    """
    调用所有模型，汇总投票结果。

    Returns:
        {
            "ensemble_binary":    "有害"/"无害",   # 投票结果
            "ensemble_category":  "性别歧视",...,  # 投票结果
            "binary_confidence":  0.0~1.0,          # 模型一致率
            "category_confidence":0.0~1.0,
            "conflict_flag":      bool,             # 严重分歧 (< 50% 一致)
            "vote_detail": {
                "binary": {"有害": 2, "无害": 1},
                "category": {"性别歧视": 2, "地域偏见": 1},
            },
            "model_results": [...]                  # 各模型原始结果
        }
    """
    model_results = []

    # 并发调用所有模型（线程池）
    with ThreadPoolExecutor(max_workers=len(model_configs)) as executor:
        futures = {
            executor.submit(
                call_single_model,
                cfg["client"], cfg["name"], text,
                cfg["supports_thinking"], 3, use_few_shots,
            ): cfg["name"]
            for cfg in model_configs
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                model_results.append(result)

    if not model_results:
        return {
            "ensemble_binary": None, "ensemble_category": None,
            "binary_confidence": 0.0, "category_confidence": 0.0,
            "conflict_flag": True, "vote_detail": {}, "model_results": [],
        }

    # ── binary 投票 (带权重) ──
    binary_votes: Dict[str, float] = defaultdict(float)
    for r in model_results:
        if r.get("binary"):
            w = (model_weights or {}).get(r["model"], 1.0)
            binary_votes[r["binary"]] += w

    total_binary_weight = sum(binary_votes.values())
    if total_binary_weight > 0:
        ensemble_binary = max(binary_votes, key=binary_votes.get)
        binary_conf = binary_votes[ensemble_binary] / total_binary_weight
    else:
        ensemble_binary = None
        binary_conf = 0.0

    # ── category 投票 (只在有害票中) ──
    cat_votes: Dict[str, float] = defaultdict(float)
    for r in model_results:
        if r.get("binary") == "有害" and r.get("category"):
            w = (model_weights or {}).get(r["model"], 1.0)
            cat_votes[r["category"]] += w

    total_cat_weight = sum(cat_votes.values())
    if total_cat_weight > 0:
        ensemble_category = max(cat_votes, key=cat_votes.get)
        cat_conf = cat_votes[ensemble_category] / total_cat_weight
    else:
        ensemble_category = "无毒" if ensemble_binary == "无害" else None
        cat_conf = binary_conf if ensemble_binary == "无害" else 0.0

    # ── toxic_type / expression 投票 ──
    tt_votes: Dict[str, float] = defaultdict(float)
    ex_votes: Dict[str, float] = defaultdict(float)
    for r in model_results:
        if r.get("binary") == "有害":
            w = (model_weights or {}).get(r["model"], 1.0)
            if r.get("toxic_type"):
                tt_votes[r["toxic_type"]] += w
            if r.get("expression"):
                ex_votes[r["expression"]] += w

    ensemble_tt = max(tt_votes, key=tt_votes.get) if tt_votes else None
    ensemble_ex = max(ex_votes, key=ex_votes.get) if ex_votes else None

    # ── conflict flag ──
    conflict_flag = binary_conf < 0.5

    return {
        "ensemble_binary":    ensemble_binary,
        "ensemble_category":  ensemble_category,
        "ensemble_toxic_type": ensemble_tt,
        "ensemble_expression": ensemble_ex,
        "binary_confidence":  round(binary_conf, 4),
        "category_confidence": round(cat_conf, 4),
        "conflict_flag":      conflict_flag,
        "vote_detail": {
            "binary":   dict(binary_votes),
            "category": dict(cat_votes),
        },
        "model_results": model_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_input_data(path: str, mode: str) -> List[Dict]:
    """
    统一数据加载。返回 list of dict，每条至少包含 'text'，
    eval 模式还有 'label'，challenger 模式还有 'category'（目标标签）。
    """
    import pandas as pd

    path = str(path)
    logger.info(f"加载数据: {path}")

    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        col_text = "文本" if "文本" in df.columns else "original_text"
        col_cat  = "标签" if "标签" in df.columns else "category"
        items = []
        for _, row in df.iterrows():
            item = {"text": str(row.get(col_text, "")).strip()}
            if col_cat in df.columns:
                item["label"] = str(row[col_cat])
                item["category"] = str(row[col_cat])
            items.append(item)
        return items

    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # challenger_gen_3B.jsonl 格式: {category, prompt, generation}
                if "generation" in obj:
                    items.append({
                        "text":     obj["generation"].strip(),
                        "category": obj.get("category", ""),
                        "prompt":   obj.get("prompt", ""),
                        "sample_id": obj.get("sample_id", -1),
                    })
                # train.jsonl messages 格式
                elif "messages" in obj:
                    assistant_content = ""
                    user_text = ""
                    for m in obj["messages"]:
                        if m.get("role") == "assistant":
                            assistant_content = m.get("content", "")
                        if m.get("role") == "user":
                            # 从 prompt 末尾提取文本
                            c = m.get("content", "")
                            idx = c.rfind("文本:")
                            if idx < 0:
                                idx = c.rfind("文本：")
                            user_text = c[idx + 3:].strip() if idx >= 0 else c[-200:]
                    items.append({
                        "text":     user_text or obj.get("original_text", ""),
                        "label":    obj.get("category", ""),
                        "category": obj.get("category", ""),
                    })
                else:
                    items.append({
                        "text":     obj.get("text", obj.get("文本", obj.get("generation", ""))).strip(),
                        "label":    obj.get("label", obj.get("标签", obj.get("category", ""))),
                        "category": obj.get("category", obj.get("label", "")),
                    })
        return items

    # JSON
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        data = data["results"]

    items = []
    for obj in data:
        items.append({
            "text":     obj.get("文本", obj.get("text", obj.get("generation", ""))).strip(),
            "label":    obj.get("标签", obj.get("label", obj.get("category", ""))),
            "category": obj.get("category", obj.get("标签", "")),
        })
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# 评估指标计算
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(records: List[Dict], mode: str) -> Dict:
    """计算整体评估指标。"""
    total = len(records)
    if total == 0:
        return {}

    # ── 基本统计 ──
    valid = [r for r in records if r["ensemble_result"]["ensemble_binary"] is not None]
    parse_fail_rate = 1.0 - len(valid) / total
    conflict_rate   = sum(1 for r in records if r["ensemble_result"]["conflict_flag"]) / total
    avg_binary_conf = sum(r["ensemble_result"]["binary_confidence"] for r in records) / total
    avg_cat_conf    = sum(r["ensemble_result"]["category_confidence"] for r in records) / total

    stats = {
        "total": total,
        "parse_fail_rate":   round(parse_fail_rate, 4),
        "conflict_rate":     round(conflict_rate, 4),
        "avg_binary_confidence":   round(avg_binary_conf, 4),
        "avg_category_confidence": round(avg_cat_conf, 4),
        "high_confidence_count": sum(
            1 for r in records if r["ensemble_result"]["binary_confidence"] >= 0.8
        ),
    }

    # ── eval_dataset / eval_challenger: 与真实标签对比 ──
    if mode in ("eval_dataset", "eval_challenger"):
        label_key = "label" if mode == "eval_dataset" else "category"

        binary_correct  = 0
        binary_total    = 0
        cat_correct     = 0
        cat_total       = 0
        off_target_count = 0  # challenger 特有: 生成文本与目标类别不符

        binary_tp = binary_fp = binary_fn = binary_tn = 0
        # category confusion: true_cat → {pred_cat: count}
        cat_confusion: Dict[str, Dict] = defaultdict(lambda: defaultdict(int))

        for r in records:
            true_label = r.get(label_key, "")
            ens = r["ensemble_result"]
            pred_binary  = ens.get("ensemble_binary")
            pred_cat     = ens.get("ensemble_category")

            if not true_label or pred_binary is None:
                continue

            true_is_harmful = true_label in HARMFUL_CATEGORIES
            pred_is_harmful = pred_binary == "有害"

            # binary 指标
            if true_is_harmful and pred_is_harmful:   binary_tp += 1
            elif not true_is_harmful and pred_is_harmful: binary_fp += 1
            elif true_is_harmful and not pred_is_harmful: binary_fn += 1
            else:                                          binary_tn += 1

            binary_total += 1
            if true_is_harmful == pred_is_harmful:
                binary_correct += 1

            # category 指标
            true_cat = true_label if true_is_harmful else "无毒"
            if pred_cat:
                cat_confusion[true_cat][pred_cat] += 1
                cat_total += 1
                if pred_cat == true_cat:
                    cat_correct += 1

            # off-target (Challenger 评估专用)
            if mode == "eval_challenger" and true_is_harmful:
                if not pred_is_harmful or (pred_cat and pred_cat != true_label):
                    off_target_count += 1

        binary_acc = binary_correct / binary_total if binary_total > 0 else 0.0
        cat_acc    = cat_correct    / cat_total    if cat_total    > 0 else 0.0
        precision  = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0.0
        recall     = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0.0
        f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        stats.update({
            "binary_accuracy":  round(binary_acc, 4),
            "category_accuracy": round(cat_acc, 4),
            "binary_precision": round(precision, 4),
            "binary_recall":    round(recall, 4),
            "binary_f1":        round(f1, 4),
            "binary_tp": binary_tp, "binary_fp": binary_fp,
            "binary_fn": binary_fn, "binary_tn": binary_tn,
            "category_confusion": {k: dict(v) for k, v in cat_confusion.items()},
        })

        if mode == "eval_challenger":
            harmful_total = sum(1 for r in records if r.get("category", "") in HARMFUL_CATEGORIES)
            stats["off_target_count"] = off_target_count
            stats["off_target_rate"]  = round(off_target_count / harmful_total, 4) if harmful_total > 0 else 0.0
            stats["on_target_rate"]   = round(1 - off_target_count / harmful_total, 4) if harmful_total > 0 else 0.0

        # 逐类别 F1
        per_cat_stats = {}
        for cat in ALL_CATEGORIES:
            c_tp = cat_confusion[cat].get(cat, 0)
            c_fp = sum(cat_confusion[other].get(cat, 0) for other in ALL_CATEGORIES if other != cat)
            c_fn = sum(cat_confusion[cat].get(other, 0) for other in ALL_CATEGORIES if other != cat)
            c_p  = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
            c_r  = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
            c_f1 = 2 * c_p * c_r / (c_p + c_r) if (c_p + c_r) > 0 else 0.0
            per_cat_stats[cat] = {"precision": round(c_p, 4), "recall": round(c_r, 4), "f1": round(c_f1, 4)}
        stats["per_category"] = per_cat_stats

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# 主批量评估流程
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch_eval(
    items: List[Dict],
    model_configs: List[Dict],
    mode: str,
    workers: int = 8,
    model_weights: Optional[Dict[str, float]] = None,
    min_confidence: float = 0.0,
    use_few_shots: bool = True,
    resume_path: Optional[str] = None,
) -> List[Dict]:
    """
    批量对所有文本进行多模型评估。

    workers: 文本级并发数（模型间调用在 ensemble_single 内部已并发）
    """
    # ── 断点续跑 ──
    done_indices = set()
    resume_records = []
    if resume_path and Path(resume_path).exists():
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_records = json.load(f)
        done_indices = {r["_idx"] for r in resume_records if "_idx" in r}
        logger.info(f"断点续跑: 已完成 {len(done_indices)} 条")

    results = list(resume_records)  # 已完成的结果先放进去

    pending = [(i, item) for i, item in enumerate(items) if i not in done_indices]
    logger.info(f"待处理: {len(pending)} 条 (共 {len(items)} 条)")

    def process_one(idx_item):
        idx, item = idx_item
        text = item.get("text", "").strip()
        if not text:
            return None
        ens = ensemble_single(model_configs, text, model_weights, use_few_shots)
        record = {
            "_idx":     idx,
            "text":     text,
            "label":    item.get("label", ""),
            "category": item.get("category", ""),
            "sample_id": item.get("sample_id", idx),
            "ensemble_result": ens,
        }
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, x): x[0] for x in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="多模型评估"):
            rec = future.result()
            if rec is not None:
                results.append(rec)

    # 按原始顺序排序
    results.sort(key=lambda r: r.get("_idx", 9999))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 结果打印与保存
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(stats: Dict, mode: str, model_names: List[str]):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  多模型协同评估结果 ({mode})")
    print(f"  模型: {', '.join(model_names)}")
    print(sep)
    print(f"  总计样本:        {stats['total']}")
    print(f"  解析失败率:      {stats['parse_fail_rate']*100:.1f}%")
    print(f"  模型分歧率:      {stats['conflict_rate']*100:.1f}%  (越低越可信)")
    print(f"  平均 Binary 置信度: {stats['avg_binary_confidence']:.3f}")
    print(f"  平均 Category 置信度: {stats['avg_category_confidence']:.3f}")
    print(f"  高置信度 (≥0.8) 样本: {stats['high_confidence_count']}")

    if mode in ("eval_dataset", "eval_challenger"):
        print(f"\n  ── 与真实标签对比 ──")
        print(f"  Binary Accuracy:  {stats['binary_accuracy']*100:.2f}%")
        print(f"  Category Accuracy:{stats['category_accuracy']*100:.2f}%")
        print(f"  "
              f"P={stats['binary_precision']:.3f}  "
              f"R={stats['binary_recall']:.3f}  "
              f"F1={stats['binary_f1']:.3f}")

    if mode == "eval_challenger":
        print(f"\n  ── Challenger 对标率 ──")
        print(f"  Off-target 数量: {stats.get('off_target_count', 0)}")
        print(f"  Off-target 率:   {stats.get('off_target_rate', 0)*100:.1f}%")
        print(f"  On-target  率:   {stats.get('on_target_rate', 0)*100:.1f}%  ← 越高越好")

    if "per_category" in stats:
        print(f"\n  ── 逐类别 F1 ──")
        for cat, m in stats["per_category"].items():
            print(f"  {cat:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

    print(sep)


def save_results(records: List[Dict], stats: Dict, output_path: str, model_names: List[str], mode: str):
    out = {
        "mode":        mode,
        "models":      model_names,
        "total":       len(records),
        "metrics":     stats,
        "results":     [],
    }

    # 构建简洁的每条结果
    for r in records:
        ens = r["ensemble_result"]
        row = {
            "text":             r["text"][:200],
            "true_label":       r.get("label") or r.get("category", ""),
            "ensemble_binary":  ens.get("ensemble_binary"),
            "ensemble_category": ens.get("ensemble_category"),
            "ensemble_toxic_type": ens.get("ensemble_toxic_type"),
            "ensemble_expression": ens.get("ensemble_expression"),
            "binary_confidence": ens.get("binary_confidence"),
            "category_confidence": ens.get("category_confidence"),
            "conflict_flag":    ens.get("conflict_flag"),
            "vote_detail":      ens.get("vote_detail"),
            "per_model_output": [
                {"model": m["model"], "binary": m["binary"],
                 "category": m["category"], "raw": m["raw_output"][:80]}
                for m in ens.get("model_results", [])
            ],
        }
        out["results"].append(row)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已保存: {output_path}")

    # 额外保存高分歧 / 低置信度样本，便于人工复审
    flagged = [r for r in out["results"] if r["conflict_flag"] or r["binary_confidence"] < 0.6]
    if flagged:
        flag_path = output_path.replace(".json", "_flagged.json")
        with open(flag_path, "w", encoding="utf-8") as f:
            json.dump({"total_flagged": len(flagged), "items": flagged},
                      f, ensure_ascii=False, indent=2)
        logger.info(f"分歧样本 ({len(flagged)} 条) → {flag_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="多模型协同评估工具")

    parser.add_argument("--mode", default="eval_challenger",
                        choices=["eval_dataset", "eval_challenger", "label_only"],
                        help="评估模式")
    parser.add_argument("--input",  required=True,  help="输入文件 (.json/.jsonl/.parquet)")
    parser.add_argument("--output", required=True,  help="输出 JSON 文件")
    parser.add_argument("--resume", default="",     help="断点续跑缓存文件 (可选)")

    # 模型选择
    parser.add_argument("--models", nargs="+",
                        default=["qwen-plus", "deepseek-chat"],
                        help=f"使用的模型列表，可选: {list(MODEL_REGISTRY)}")
    parser.add_argument("--model_weights", nargs="*", default=[],
                        help="各模型权重，格式: model=weight (如 qwen-max=2.0 qwen-plus=1.0)")

    # API Keys（也可通过环境变量 API_KEY_QWEN / API_KEY_DEEPSEEK / API_KEY_OPENAI）
    parser.add_argument("--api_key_qwen",     default="", help="DashScope API key")
    parser.add_argument("--api_key_deepseek", default="", help="DeepSeek API key")
    parser.add_argument("--api_key_openai",   default="", help="OpenAI API key")

    # 运行参数
    parser.add_argument("--workers",      type=int,   default=6,
                        help="文本级并发数 (建议 ≤8，避免超限)")
    parser.add_argument("--max_samples",  type=int,   default=0,
                        help="最大处理条数 (0=全部，调试用)")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="最低置信度阈值，低于此值标记为 conflict")
    parser.add_argument("--no_few_shots", action="store_true",
                        help="禁用 few-shot 示例 (加快速度但精度下降)")
    parser.add_argument("--dry_run",      action="store_true",
                        help="只显示前 3 条结果，不保存")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"  多模型协同评估  mode={args.mode}")
    logger.info("=" * 60)

    # ── 1. 构建模型客户端 ──
    logger.info("\n[1] 初始化模型...")

    # 从环境变量补充 key
    if not args.api_key_qwen:
        args.api_key_qwen = os.environ.get("API_KEY_QWEN", os.environ.get("DASHSCOPE_API_KEY", ""))
    if not args.api_key_deepseek:
        args.api_key_deepseek = os.environ.get("API_KEY_DEEPSEEK", "")
    if not args.api_key_openai:
        args.api_key_openai = os.environ.get("OPENAI_API_KEY", "")

    model_configs = build_model_clients(args.models, args)

    model_weights = None
    if args.model_weights:
        model_weights = {}
        for mw in args.model_weights:
            parts = mw.split("=")
            if len(parts) == 2:
                model_weights[parts[0]] = float(parts[1])

    # ── 2. 加载数据 ──
    logger.info(f"\n[2] 加载数据: {args.input}")
    items = load_input_data(args.input, args.mode)
    logger.info(f"   共 {len(items)} 条")

    if args.max_samples > 0:
        items = items[:args.max_samples]
        logger.info(f"   (截断至 {len(items)} 条)")

    if args.dry_run:
        items = items[:3]
        logger.info("   [dry_run] 只处理前 3 条")

    # ── 3. 批量评估 ──
    logger.info(f"\n[3] 开始多模型评估 (workers={args.workers})...")
    records = run_batch_eval(
        items        = items,
        model_configs = model_configs,
        mode         = args.mode,
        workers      = args.workers,
        model_weights = model_weights,
        min_confidence = args.min_confidence,
        use_few_shots = not args.no_few_shots,
        resume_path  = args.resume or None,
    )

    # ── 4. 计算指标 ──
    logger.info("\n[4] 计算评估指标...")
    stats = compute_metrics(records, args.mode)

    # ── 5. 打印 & 保存 ──
    print_summary(stats, args.mode, [c["name"] for c in model_configs])

    if not args.dry_run:
        save_results(records, stats, args.output, [c["name"] for c in model_configs], args.mode)

    logger.info("\n✅ 完成")


if __name__ == "__main__":
    main()
