#!/usr/bin/env python3
"""修复 progress.json，将指定 step 设为从头重跑"""
import json, glob, os, sys

base = "selfplay_integrated/3B_4npu"

if not os.path.exists(base):
    candidates = glob.glob("selfplay_integrated/*npu")
    if candidates:
        base = max(candidates, key=os.path.getmtime)
    else:
        print("未找到 selfplay_integrated 目录")
        exit(1)

progress_file = os.path.join(base, "progress.json")

if not os.path.exists(progress_file):
    print(f"progress.json 不存在: {progress_file}")
    exit(1)

with open(progress_file, "r") as f:
    d = json.load(f)

print(f"当前 progress.json:")
print(f"  last_completed_step:  {d.get('last_completed_step')}")
print(f"  last_completed_phase: {d.get('last_completed_phase')}")
print(f"  current_challenger:   {d.get('current_challenger', '')[-60:]}")
print(f"  current_reviewer:     {d.get('current_reviewer', '')[-60:]}")

# 回退到上一步完成
step = d.get("last_completed_step", 1)
rollback_to = step - 1 if d.get("last_completed_phase") != "done" else step

if len(sys.argv) > 1:
    rollback_to = int(sys.argv[1])

print(f"\n将回退到 step {rollback_to} done（step {rollback_to + 1} 从头重跑）")

d["last_completed_step"] = rollback_to
d["last_completed_phase"] = "done"

with open(progress_file, "w") as f:
    json.dump(d, f, indent=2)

print(f"已写入: {progress_file}")
