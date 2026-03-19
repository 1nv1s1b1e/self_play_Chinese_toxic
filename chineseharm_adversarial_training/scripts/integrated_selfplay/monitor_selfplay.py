#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Play 训练监控脚本
======================
在独立进程中运行，实时监控 metrics.jsonl 和 selfplay 目录变化。

用法:
    python scripts/integrated_selfplay/monitor_selfplay.py
    python scripts/integrated_selfplay/monitor_selfplay.py --selfplay_dir selfplay_integrated/3B_4npu
    python scripts/integrated_selfplay/monitor_selfplay.py --watch  # 持续监控模式
"""

import argparse
import json
import os
import sys
import time
import glob
from pathlib import Path
from datetime import datetime


def find_selfplay_dir(base_dir: str) -> str:
    """自动查找 selfplay_integrated 目录"""
    candidates = glob.glob(os.path.join(base_dir, "selfplay_integrated", "*npu"))
    if candidates:
        # 取最新修改的
        return max(candidates, key=os.path.getmtime)
    return ""


def load_metrics(metrics_path: str) -> list:
    """加载 metrics.jsonl"""
    if not os.path.exists(metrics_path):
        return []
    entries = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def print_metrics_table(entries: list):
    """打印 metrics 表格"""
    if not entries:
        print("  暂无 metrics 数据")
        return

    print(f"\n{'Step':>5} {'ASR':>8} {'Rev Acc':>10} {'Macro-F1':>10} {'Best Acc':>10} {'Time':>20}")
    print("-" * 70)
    for e in entries:
        step = e.get("step", "?")
        asr = e.get("asr")
        acc = e.get("reviewer_acc")
        f1 = e.get("reviewer_macro_f1")
        best = e.get("best_acc")
        ts = e.get("timestamp", "")
        if ts:
            ts = ts.split("T")[-1].split(".")[0]  # HH:MM:SS

        asr_s = f"{asr:.3f}" if asr is not None else "-"
        acc_s = f"{acc:.2f}%" if acc is not None else "-"
        f1_s = f"{f1:.4f}" if f1 is not None else "-"
        best_s = f"{best:.2f}%" if best is not None else "-"

        print(f"{step:>5} {asr_s:>8} {acc_s:>10} {f1_s:>10} {best_s:>10} {ts:>20}")


def print_step_data_summary(data_dir: str):
    """汇总每步的 datagen 数据"""
    if not os.path.exists(data_dir):
        return

    steps = sorted(glob.glob(os.path.join(data_dir, "step_*")))
    if not steps:
        return

    print(f"\n{'Step':>5} {'C-parquet':>12} {'R-parquet':>12} {'SampleReward':>14} {'Fooled':>8}")
    print("-" * 58)

    for step_path in steps:
        step_name = os.path.basename(step_path)
        step_num = step_name.replace("step_", "")

        c_pq = glob.glob(os.path.join(step_path, "challenger_grpo_*.parquet"))
        r_pq = glob.glob(os.path.join(step_path, "reviewer_grpo_*.parquet"))
        sr_pq = glob.glob(os.path.join(step_path, "sample_rewards_*.parquet"))

        c_rows = "-"
        r_rows = "-"
        sr_rows = "-"
        fooled = "-"

        try:
            import pandas as pd
            if c_pq:
                c_rows = str(len(pd.read_parquet(c_pq[0])))
            if r_pq:
                r_rows = str(len(pd.read_parquet(r_pq[0])))
            if sr_pq:
                df = pd.read_parquet(sr_pq[0])
                sr_rows = str(len(df))
                if "reviewer_was_fooled" in df.columns:
                    n_fooled = df["reviewer_was_fooled"].sum()
                    fooled = f"{n_fooled}/{len(df)}"
        except ImportError:
            if c_pq:
                c_rows = "exists"
            if r_pq:
                r_rows = "exists"

        print(f"{step_num:>5} {c_rows:>12} {r_rows:>12} {sr_rows:>14} {fooled:>8}")


def print_best_info(selfplay_dir: str):
    """显示 best 模型信息"""
    best_step_file = os.path.join(selfplay_dir, "best", "best_step.txt")
    if os.path.exists(best_step_file):
        with open(best_step_file) as f:
            best_step = f.read().strip()
        print(f"\n  Best 模型来自 Step {best_step}")
    else:
        print("\n  暂无 best 模型")


def print_progress(selfplay_dir: str):
    """显示训练进度"""
    progress_file = os.path.join(selfplay_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            p = json.load(f)
        step = p.get("last_completed_step", 0)
        total = p.get("total_steps", "?")
        phase = p.get("last_completed_phase", "?")
        ts = p.get("timestamp", "")
        print(f"  进度: Step {step}/{total} (phase: {phase})")
        print(f"  更新: {ts}")
    else:
        print("  暂无进度文件")


def collect_step_data(data_dir: str) -> list:
    """收集每步的 datagen 数据统计"""
    if not os.path.exists(data_dir):
        return []

    rows = []
    steps = sorted(glob.glob(os.path.join(data_dir, "step_*")))
    for step_path in steps:
        step_num = int(os.path.basename(step_path).replace("step_", ""))
        info = {"step": step_num, "c_rows": 0, "r_rows": 0, "sr_rows": 0,
                "fooled": 0, "cat_fooled": 0, "asr": 0.0}
        try:
            import pandas as pd
            for pq in glob.glob(os.path.join(step_path, "challenger_grpo_*.parquet")):
                info["c_rows"] = len(pd.read_parquet(pq))
            for pq in glob.glob(os.path.join(step_path, "reviewer_grpo_*.parquet")):
                info["r_rows"] = len(pd.read_parquet(pq))
            for pq in glob.glob(os.path.join(step_path, "sample_rewards_*.parquet")):
                df = pd.read_parquet(pq)
                info["sr_rows"] = len(df)
                if "reviewer_was_fooled" in df.columns:
                    info["fooled"] = int(df["reviewer_was_fooled"].sum())
                if "reviewer_cat_fooled" in df.columns:
                    info["cat_fooled"] = int(df["reviewer_cat_fooled"].sum())
                if len(df) > 0:
                    info["asr"] = round(info["fooled"] / len(df), 4)
        except ImportError:
            pass
        rows.append(info)
    return rows


def rescue_eval_jsons(selfplay_dir: str):
    """扫描所有 step_*/eval_gate/*.json，拷贝到 eval_history/ 避免被 cleanup 删除"""
    eval_history = os.path.join(selfplay_dir, "eval_history")
    os.makedirs(eval_history, exist_ok=True)

    import shutil
    saved = 0
    for eval_json in sorted(glob.glob(os.path.join(selfplay_dir, "step_*/eval_gate/*.json"))):
        # 提取 step 编号
        parts = eval_json.split(os.sep)
        step_part = [p for p in parts if p.startswith("step_")]
        if not step_part:
            continue
        step_num = step_part[0].replace("step_", "")
        dst = os.path.join(eval_history, f"eval_step{step_num}.json")
        if not os.path.exists(dst):
            shutil.copy2(eval_json, dst)
            saved += 1

    # 也检查 step_0（baseline 评估，在 eval_init/ 里）
    for eval_json in glob.glob(os.path.join(selfplay_dir, "eval_init/*.json")):
        dst = os.path.join(eval_history, "eval_step0.json")
        if not os.path.exists(dst):
            shutil.copy2(eval_json, dst)
            saved += 1

    return saved


def save_snapshot(selfplay_dir: str, data_dir: str, entries: list, step_data: list):
    """保存完整监控快照到 selfplay_dir/monitor_snapshots/"""
    save_dir = os.path.join(selfplay_dir, "monitor_snapshots")
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存 metrics CSV（覆盖，始终是最新完整版）
    csv_path = os.path.join(save_dir, "metrics_history.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("step,asr,reviewer_acc,reviewer_macro_f1,best_acc,timestamp\n")
        for e in entries:
            row = [
                str(e.get("step", "")),
                str(e.get("asr", "")),
                str(e.get("reviewer_acc", "")),
                str(e.get("reviewer_macro_f1", "")),
                str(e.get("best_acc", "")),
                str(e.get("timestamp", "")),
            ]
            f.write(",".join(row) + "\n")

    # 2. 保存每步数据统计 CSV
    data_csv = os.path.join(save_dir, "step_data_history.csv")
    with open(data_csv, "w", encoding="utf-8") as f:
        f.write("step,c_rows,r_rows,sr_rows,fooled,cat_fooled,asr\n")
        for d in step_data:
            row = [str(d.get(k, "")) for k in
                   ["step", "c_rows", "r_rows", "sr_rows", "fooled", "cat_fooled", "asr"]]
            f.write(",".join(row) + "\n")

    # 3. 保存完整 JSON 快照（带时间戳）
    snapshot = {
        "snapshot_time": datetime.now().isoformat(),
        "metrics": entries,
        "step_data": step_data,
    }
    # 最新快照（覆盖）
    json_path = os.path.join(save_dir, "latest_snapshot.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    return csv_path


def monitor_once(selfplay_dir: str, data_dir: str):
    """单次输出全部监控信息并保存快照"""
    metrics_path = os.path.join(selfplay_dir, "metrics.jsonl")

    print("=" * 70)
    print(f"  Self-Play 监控  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print(f"  目录: {selfplay_dir}")
    print("=" * 70)

    print_progress(selfplay_dir)
    print_best_info(selfplay_dir)

    print("\n── Metrics (每步评估) ──")
    entries = load_metrics(metrics_path)
    print_metrics_table(entries)

    print("\n── 每步数据统计 ──")
    step_data = collect_step_data(data_dir)
    print_step_data_summary(data_dir)

    # 抢救 eval JSON（在被 cleanup 删除前拷贝到 eval_history/）
    rescued = rescue_eval_jsons(selfplay_dir)
    eval_history = os.path.join(selfplay_dir, "eval_history")
    total_evals = len(glob.glob(os.path.join(eval_history, "eval_step*.json")))
    if rescued > 0:
        print(f"\n  已抢救 {rescued} 个 eval JSON → eval_history/ (共 {total_evals} 个)")
    else:
        print(f"\n  eval_history/: {total_evals} 个评估结果已保存")

    # 保存快照
    if entries or step_data:
        csv_path = save_snapshot(selfplay_dir, data_dir, entries, step_data)
        print(f"  快照已保存: {os.path.dirname(csv_path)}/")

    print()


def main():
    parser = argparse.ArgumentParser(description="Self-Play 训练监控")
    parser.add_argument("--selfplay_dir", type=str, default="",
                        help="selfplay_integrated/3B_4npu 路径")
    parser.add_argument("--data_dir", type=str, default="",
                        help="selfplay_integrated_data/3B 路径")
    parser.add_argument("--base_dir", type=str, default="",
                        help="chineseharm_adversarial_training 根目录")
    parser.add_argument("--watch", action="store_true",
                        help="持续监控模式 (每 60s 刷新)")
    parser.add_argument("--interval", type=int, default=60,
                        help="监控间隔秒数 (默认 60)")
    args = parser.parse_args()

    # 自动推断路径
    if not args.base_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_dir = os.path.dirname(os.path.dirname(script_dir))

    if not args.selfplay_dir:
        args.selfplay_dir = find_selfplay_dir(args.base_dir)
        if not args.selfplay_dir:
            print(f"未找到 selfplay 目录，请指定 --selfplay_dir")
            sys.exit(1)

    if not args.data_dir:
        # 从 selfplay_dir 推断: selfplay_integrated/3B_4npu → selfplay_integrated_data/3B
        dirname = os.path.basename(args.selfplay_dir)  # e.g. "3B_4npu"
        model_size = dirname.split("_")[0]  # "3B"
        args.data_dir = os.path.join(args.base_dir, "selfplay_integrated_data", model_size)

    if args.watch:
        print(f"持续监控模式 (每 {args.interval}s 刷新, Ctrl+C 退出)\n")
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                monitor_once(args.selfplay_dir, args.data_dir)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控结束")
    else:
        monitor_once(args.selfplay_dir, args.data_dir)


if __name__ == "__main__":
    main()
