#!/usr/bin/env python3
"""找出历史最佳 acc 并写入 best_acc.txt"""
import json, glob, os

base = "selfplay_integrated/3B_4npu"

# 自动查找
if not os.path.exists(base):
    candidates = glob.glob("selfplay_integrated/*npu")
    if candidates:
        base = max(candidates, key=os.path.getmtime)
    else:
        print("未找到 selfplay_integrated 目录")
        exit(1)

print(f"目录: {base}")

best_acc = 0
best_step = 0
best_source = ""

# 1. 从 metrics.jsonl
mf = os.path.join(base, "metrics.jsonl")
if os.path.exists(mf):
    print(f"\n=== metrics.jsonl ===")
    for line in open(mf):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        step = d.get("step", "?")
        acc = d.get("reviewer_acc")
        bacc = d.get("best_acc")
        print(f"  step {step}: reviewer_acc={acc}, best_acc={bacc}")
        for v in [acc, bacc]:
            if v and float(v) > best_acc:
                best_acc = float(v)
                best_step = step
                best_source = "metrics.jsonl"

# 2. 从 eval_history
eval_dir = os.path.join(base, "eval_history")
if os.path.exists(eval_dir):
    print(f"\n=== eval_history ===")
    for f in sorted(glob.glob(os.path.join(eval_dir, "*.json"))):
        try:
            m = json.load(open(f)).get("metrics", {})
            acc = m.get("overall_accuracy", 0)
            print(f"  {os.path.basename(f)}: acc={acc}")
            if float(acc) > best_acc:
                best_acc = float(acc)
                best_source = os.path.basename(f)
        except Exception:
            pass

# 3. 从 datagen_history
dg_dir = os.path.join(base, "datagen_history")
if os.path.exists(dg_dir):
    print(f"\n=== datagen_history ===")
    for f in sorted(glob.glob(os.path.join(dg_dir, "datagen_stats_*.json"))):
        try:
            d = json.load(open(f))
            asr = d.get("overall_asr_1acc", "?")
            rev_acc = d.get("overall_reviewer_acc", "?")
            print(f"  {os.path.basename(f)}: ASR={asr}, rev_acc={rev_acc}")
        except Exception:
            pass

# 4. best_step.txt
bs = os.path.join(base, "best", "best_step.txt")
if os.path.exists(bs):
    print(f"\n当前 best_step.txt: step {open(bs).read().strip()}")

# 结果
print(f"\n{'='*50}")
print(f"历史最佳: acc={best_acc}, step={best_step} (来源: {best_source})")

# 写入
if best_acc > 0:
    out = os.path.join(base, "best", "best_acc.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(str(best_acc))
    print(f"已写入: {out}")
else:
    print("未找到有效的 acc 数据")
