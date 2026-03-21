#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 metrics.jsonl 中是否发生过回滚（模型路径指向 best/ 目录）。

用法:
    python check_rollback.py <metrics.jsonl路径>
    python check_rollback.py /path/to/selfplay_integrated/3B_4npu/metrics.jsonl
"""

import json
import sys


def check_rollback(metrics_path: str):
    entries = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("metrics.jsonl 为空")
        return

    print(f"共 {len(entries)} 步记录\n")

    rollback_steps = []
    for e in entries:
        step = e.get("step", "?")
        challenger = e.get("challenger", "")
        reviewer = e.get("reviewer", "")
        acc = e.get("reviewer_acc")
        best_acc = e.get("best_acc")

        is_rollback = "/best/" in challenger or "/best/" in reviewer

        if is_rollback:
            rollback_steps.append(e)
            print(f"  ⚠️  Step {step}: 回滚! acc={acc}, best={best_acc}")
            print(f"       challenger={challenger}")
            print(f"       reviewer={reviewer}")

    print(f"\n{'='*60}")
    print(f"总步数: {len(entries)}")
    print(f"回滚步数: {len(rollback_steps)}")
    if rollback_steps:
        steps = [e["step"] for e in rollback_steps]
        print(f"回滚发生在: {steps}")

        # 检测连续回滚
        consecutive = 1
        max_consecutive = 1
        for i in range(1, len(steps)):
            if steps[i] == steps[i - 1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        print(f"最大连续回滚: {max_consecutive} 步")
    else:
        print("未发生过回滚")

    # acc 趋势摘要
    print(f"\n{'='*60}")
    print("Reviewer acc 趋势:")
    for e in entries:
        step = e.get("step", "?")
        acc = e.get("reviewer_acc")
        best = e.get("best_acc")
        rolled = " ← 回滚" if ("/best/" in e.get("challenger", "") or "/best/" in e.get("reviewer", "")) else ""
        acc_str = f"{acc:.4f}" if acc else "  N/A "
        print(f"  Step {step:3d}: acc={acc_str}  best={best:.4f}{rolled}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <metrics.jsonl路径>")
        sys.exit(1)
    check_rollback(sys.argv[1])
