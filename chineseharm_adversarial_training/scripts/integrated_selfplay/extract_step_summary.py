#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取每步 Self-Play 的完整数据使用情况。

从 datagen_history/, eval_history/, metrics.jsonl 中汇总每步信息:
- Challenger/Reviewer 模型来源
- 训练数据构成（对抗/种子/困难样本）
- 评估指标
- ASR / fooling 统计

用法:
    python scripts/integrated_selfplay/extract_step_summary.py
    python scripts/integrated_selfplay/extract_step_summary.py --output summary.csv
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path


def find_selfplay_dir(base_dir: str) -> str:
    candidates = glob.glob(os.path.join(base_dir, "selfplay_integrated", "*npu"))
    return max(candidates, key=os.path.getmtime) if candidates else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfplay_dir", default="")
    parser.add_argument("--base_dir", default="")
    parser.add_argument("--output", default="", help="输出 CSV 路径")
    args = parser.parse_args()

    if not args.base_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_dir = os.path.dirname(os.path.dirname(script_dir))
    if not args.selfplay_dir:
        args.selfplay_dir = find_selfplay_dir(args.base_dir)
        if not args.selfplay_dir:
            print("未找到 selfplay 目录")
            sys.exit(1)

    sp = args.selfplay_dir
    datagen_dir = os.path.join(sp, "datagen_history")
    eval_dir = os.path.join(sp, "eval_history")
    metrics_path = os.path.join(sp, "metrics.jsonl")

    # 1. 加载 metrics.jsonl
    metrics_map = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    metrics_map[e.get("step")] = e
                except:
                    pass

    # 2. 加载 datagen_stats
    datagen_map = {}
    if os.path.isdir(datagen_dir):
        for f in glob.glob(os.path.join(datagen_dir, "datagen_stats_step*.json")):
            step = int(os.path.basename(f).replace("datagen_stats_step", "").replace(".json", ""))
            try:
                with open(f) as fh:
                    datagen_map[step] = json.load(fh)
            except:
                pass

    # 3. 加载 eval_history
    eval_map = {}
    if os.path.isdir(eval_dir):
        for f in glob.glob(os.path.join(eval_dir, "eval_step*.json")):
            step = int(os.path.basename(f).replace("eval_step", "").replace(".json", ""))
            try:
                with open(f) as fh:
                    d = json.load(fh)
                eval_map[step] = d.get("metrics", {})
            except:
                pass

    # 4. 汇总
    all_steps = sorted(set(list(metrics_map.keys()) + list(datagen_map.keys()) + list(eval_map.keys())))
    if not all_steps:
        print("没有找到任何 step 数据")
        return

    rows = []
    for step in all_steps:
        m = metrics_map.get(step, {})
        d = datagen_map.get(step, {})
        e = eval_map.get(step, {})

        row = {
            "step": step,
            # 模型来源
            "challenger_model": m.get("challenger", d.get("challenger_model", "")),
            "reviewer_model": m.get("reviewer", d.get("reviewer_model", "")),
            # datagen 统计
            "total_generated": d.get("total_generated", ""),
            "total_fooled": d.get("total_fooled", ""),
            "asr": d.get("overall_asr_1acc", m.get("asr", "")),
            "reviewer_acc_datagen": d.get("overall_reviewer_acc", ""),
            # Reviewer GRPO 数据构成
            "r_grpo_size": d.get("reviewer_grpo_size", ""),
            "r_adversarial": d.get("reviewer_adversarial_count", ""),
            "r_orig_nontoxic": d.get("reviewer_orig_nontoxic_count", ""),
            "r_orig_toxic": d.get("reviewer_orig_toxic_count", ""),
            "r_hard_count": d.get("reviewer_hard_count", ""),
            "r_current_hard": d.get("reviewer_current_hard_count", ""),
            "r_history_hard": d.get("reviewer_history_hard_count", ""),
            "r_mix_ratio": d.get("reviewer_mix_ratio", ""),
            # Challenger GRPO 数据
            "c_grpo_size": d.get("challenger_grpo_size", ""),
            # 评估指标
            "eval_acc": e.get("overall_accuracy", ""),
            "eval_macro_f1": e.get("macro_f1", ""),
            "best_acc": m.get("best_acc", ""),
        }

        # 每类别统计
        cats = d.get("stats_by_category", {})
        for cat in ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]:
            cs = cats.get(cat, {})
            row[f"{cat}_asr"] = cs.get("adversarial_success_rate", "")
            row[f"{cat}_bin_acc"] = cs.get("reviewer_binary_accuracy", "")
            row[f"{cat}_cat_acc"] = cs.get("reviewer_category_accuracy", "")

        rows.append(row)

    # 5. 打印
    print(f"\n{'='*100}")
    print(f"  Self-Play 每步数据汇总 ({len(rows)} 步)")
    print(f"  目录: {sp}")
    print(f"{'='*100}\n")

    for row in rows:
        step = row["step"]
        print(f"── Step {step} ──")
        c_model = os.path.basename(row["challenger_model"]) if row["challenger_model"] else "?"
        r_model = os.path.basename(row["reviewer_model"]) if row["reviewer_model"] else "?"
        print(f"  模型: C={c_model}  R={r_model}")

        if row["total_generated"]:
            print(f"  Datagen: 生成{row['total_generated']}条, fooled={row['total_fooled']}, ASR={row['asr']}")

        if row["c_grpo_size"]:
            print(f"  Challenger GRPO: {row['c_grpo_size']}条 prompt")

        if row["r_grpo_size"]:
            print(f"  Reviewer GRPO: {row['r_grpo_size']}条 (对抗{row['r_adversarial']} + 无毒{row['r_orig_nontoxic']} + 有害{row['r_orig_toxic']} + 困难{row['r_hard_count']})")

        if row["r_current_hard"] or row["r_history_hard"]:
            print(f"    困难样本: 当前{row['r_current_hard']} + 历史{row['r_history_hard']}")

        if row["eval_acc"]:
            print(f"  评估: acc={row['eval_acc']:.2f}%, F1={row['eval_macro_f1']:.4f}, best={row['best_acc']}")

        # 类别统计
        cats_with_data = [c for c in ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
                          if row.get(f"{c}_asr")]
        if cats_with_data:
            print(f"  类别: {'类别':10s} ASR    BinAcc CatAcc")
            for cat in cats_with_data:
                asr = row.get(f"{cat}_asr", "")
                ba = row.get(f"{cat}_bin_acc", "")
                ca = row.get(f"{cat}_cat_acc", "")
                asr_s = f"{asr:.3f}" if isinstance(asr, (int, float)) else str(asr)
                ba_s = f"{ba:.3f}" if isinstance(ba, (int, float)) else str(ba)
                ca_s = f"{ca:.3f}" if isinstance(ca, (int, float)) else str(ca)
                print(f"         {cat:10s} {asr_s:6s} {ba_s:6s} {ca_s:6s}")
        print()

    # 6. 保存 CSV
    if args.output or True:
        csv_path = args.output or os.path.join(sp, "monitor_snapshots", "step_summary.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        keys = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                vals = [str(row.get(k, "")) for k in keys]
                f.write(",".join(vals) + "\n")
        print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
