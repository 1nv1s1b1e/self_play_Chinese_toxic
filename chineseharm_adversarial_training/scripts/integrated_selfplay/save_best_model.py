#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 eval_history/ 中找到历史最优模型并保存到 best_saved/

用法:
    python scripts/integrated_selfplay/save_best_model.py
    python scripts/integrated_selfplay/save_best_model.py --selfplay_dir selfplay_integrated/3B_4npu
    python scripts/integrated_selfplay/save_best_model.py --metric macro_f1  # 按 F1 选 best
"""

import argparse
import json
import os
import sys
import glob
import shutil


def find_selfplay_dir(base_dir: str) -> str:
    candidates = glob.glob(os.path.join(base_dir, "selfplay_integrated", "*npu"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    return ""


def load_all_eval_results(eval_history_dir: str) -> list:
    """加载所有 eval_history/eval_step*.json，提取指标"""
    results = []
    for f in sorted(glob.glob(os.path.join(eval_history_dir, "eval_step*.json"))):
        step_str = os.path.basename(f).replace("eval_step", "").replace(".json", "")
        try:
            step = int(step_str)
        except ValueError:
            continue

        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        metrics = data.get("metrics", {})
        results.append({
            "step": step,
            "file": f,
            "acc": metrics.get("overall_accuracy", 0.0),
            "macro_f1": metrics.get("macro_f1", 0.0),
            "category_metrics": metrics.get("category_metrics", {}),
        })

    return sorted(results, key=lambda x: x["step"])


def find_step_model_dir(selfplay_dir: str, step: int):
    """找到 step 对应的模型目录"""
    # 优先查 step_N/reviewer 和 step_N/challenger
    step_dir = os.path.join(selfplay_dir, f"step_{step}")
    challenger = os.path.join(step_dir, "challenger")
    reviewer = os.path.join(step_dir, "reviewer")

    # 也检查 best/ 目录
    best_challenger = os.path.join(selfplay_dir, "best", "challenger")
    best_reviewer = os.path.join(selfplay_dir, "best", "reviewer")

    # 检查 latest/ 的记录
    latest_paths = os.path.join(selfplay_dir, "latest", "latest_paths.json")

    found = {"challenger": None, "reviewer": None}

    # 1. 直接查 step 目录
    if os.path.isdir(challenger):
        found["challenger"] = challenger
    if os.path.isdir(reviewer):
        found["reviewer"] = reviewer

    # 2. 如果 step 目录已被清理，查 best/
    if found["challenger"] is None and os.path.isdir(best_challenger):
        best_step_file = os.path.join(selfplay_dir, "best", "best_step.txt")
        if os.path.exists(best_step_file):
            with open(best_step_file) as f:
                best_step = f.read().strip()
            if best_step == str(step):
                found["challenger"] = best_challenger
                found["reviewer"] = best_reviewer

    # 3. 查 progress.json 中记录的路径
    progress_file = os.path.join(selfplay_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            prog = json.load(f)
        if prog.get("last_completed_step") == step:
            c_path = prog.get("current_challenger", "")
            r_path = prog.get("current_reviewer", "")
            if found["challenger"] is None and os.path.isdir(c_path):
                found["challenger"] = c_path
            if found["reviewer"] is None and os.path.isdir(r_path):
                found["reviewer"] = r_path

    return found


def main():
    parser = argparse.ArgumentParser(description="找到历史最优模型并保存")
    parser.add_argument("--selfplay_dir", type=str, default="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="",
                        help="保存目录 (默认: selfplay_dir/best_saved/)")
    parser.add_argument("--metric", type=str, default="acc",
                        choices=["acc", "macro_f1"],
                        help="选择 best 的指标 (默认: acc)")
    args = parser.parse_args()

    if not args.base_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_dir = os.path.dirname(os.path.dirname(script_dir))

    if not args.selfplay_dir:
        args.selfplay_dir = find_selfplay_dir(args.base_dir)
        if not args.selfplay_dir:
            print("未找到 selfplay 目录")
            sys.exit(1)

    eval_history = os.path.join(args.selfplay_dir, "eval_history")
    if not os.path.exists(eval_history):
        print(f"eval_history 目录不存在: {eval_history}")
        sys.exit(1)

    # 1. 加载所有评估结果
    results = load_all_eval_results(eval_history)
    if not results:
        print("没有找到评估结果")
        sys.exit(1)

    # 2. 打印所有结果
    print(f"\n{'Step':>5} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 30)
    for r in results:
        print(f"{r['step']:>5} {r['acc']:>9.2f}% {r['macro_f1']:>10.4f}")

    # 3. 找 best
    if args.metric == "acc":
        best = max(results, key=lambda x: x["acc"])
    else:
        best = max(results, key=lambda x: x["macro_f1"])

    print(f"\nBest by {args.metric}: Step {best['step']} "
          f"(acc={best['acc']:.2f}%, macro_f1={best['macro_f1']:.4f})")

    # 4. 找到对应模型
    models = find_step_model_dir(args.selfplay_dir, best["step"])

    if not models["challenger"] and not models["reviewer"]:
        print(f"\n⚠️ Step {best['step']} 的模型目录已被清理")
        # 尝试用 best/ 目录
        best_dir = os.path.join(args.selfplay_dir, "best")
        if os.path.isdir(os.path.join(best_dir, "challenger")):
            models["challenger"] = os.path.join(best_dir, "challenger")
            models["reviewer"] = os.path.join(best_dir, "reviewer")
            print(f"   使用 best/ 目录中的模型")
        else:
            print("   无法找到模型，退出")
            sys.exit(1)

    # 5. 复制到输出目录（默认就是 best/）
    if not args.output_dir:
        args.output_dir = os.path.join(args.selfplay_dir, "best")

    os.makedirs(args.output_dir, exist_ok=True)

    for role in ["challenger", "reviewer"]:
        src = models.get(role)
        if src and os.path.isdir(src):
            dst = os.path.join(args.output_dir, role)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            print(f"\n  复制 {role}: {src} → {dst}")
            shutil.copytree(src, dst)
        else:
            print(f"\n  ⚠️ {role} 模型未找到")

    # 6. 保存 best info
    info = {
        "step": best["step"],
        "metric": args.metric,
        "acc": best["acc"],
        "macro_f1": best["macro_f1"],
        "category_metrics": best["category_metrics"],
        "source_challenger": models.get("challenger", ""),
        "source_reviewer": models.get("reviewer", ""),
    }
    info_path = os.path.join(args.output_dir, "best_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Best 模型已保存到: {args.output_dir}/")
    print(f"   best_info.json: Step {best['step']}, acc={best['acc']:.2f}%, F1={best['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
