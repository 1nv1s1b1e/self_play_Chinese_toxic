#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 SFT 数据: 用精简版 prompt 重建 Reviewer SFT 训练集
=======================================================
从原始 split_data 重建, 支持两种模板:
  --template short : 精简版 (去掉 RULES, ~200 chars)
  --template full  : 完整版 (含 RULES, ~1300 chars, 与旧 SFT 一致)

可以用 --max_samples 限制数量做小规模快速测试.

用法:
  # 精简版全量
  python tests/prepare_sft_data.py --template short --output_dir /path/to/output

  # 精简版小规模 (500条, 用于快速验证)
  python tests/prepare_sft_data.py --template short --max_samples 500 --output_dir /tmp/mini_sft

  # 完整版 (对照组)
  python tests/prepare_sft_data.py --template full --max_samples 500 --output_dir /tmp/mini_sft_full
"""

import os, sys, json, argparse, random
from collections import Counter

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)

from constants import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    REVIEWER_USER_TEMPLATE_FULL,
    HARMFUL_CATEGORIES,
)


def build_assistant_response(row):
    """根据标签构建标准 assistant 回复."""
    label = row.get("标签", "无毒")
    if label == "无毒" or label not in HARMFUL_CATEGORIES:
        return "判断：无害"

    toxic_type_map = {0: "无毒", 1: "一般攻击性", 2: "仇恨言论"}
    expr_map = {0: "非仇恨", 1: "显式仇恨", 2: "隐式仇恨", 3: "转述"}

    tt_val = row.get("toxic_type", 0)
    expr_val = row.get("expression", 0)

    if isinstance(tt_val, str):
        toxic_type = tt_val
    else:
        toxic_type = toxic_type_map.get(int(tt_val), "仇恨言论")

    if isinstance(expr_val, str):
        expression = expr_val
    else:
        expression = expr_map.get(int(expr_val), "显式仇恨")

    if toxic_type == "无毒":
        toxic_type = "仇恨言论"
    if expression == "非仇恨":
        expression = "显式仇恨"

    return (
        f"判断：有害\n"
        f"类别：{label}\n"
        f"毒性类型：{toxic_type}\n"
        f"表达方式：{expression}"
    )


def build_sft_sample(row, template="short"):
    """构建一条 SFT 训练样本."""
    text = row.get("文本", "").strip()
    if not text:
        return None

    if template == "short":
        user_content = REVIEWER_USER_TEMPLATE.format(text=text[:500])
    else:
        user_content = REVIEWER_USER_TEMPLATE_FULL.format(text=text[:500])

    assistant_content = build_assistant_response(row)

    return {
        "messages": [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="准备 SFT 数据")
    parser.add_argument("--train_data", default="", type=str,
                        help="训练集路径 (默认: split_data/train.json)")
    parser.add_argument("--val_data", default="", type=str,
                        help="验证集路径 (默认: split_data/val.json)")
    parser.add_argument("--template", default="short", choices=["short", "full"],
                        help="Prompt 模板: short(精简版) / full(完整版)")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="输出目录")
    parser.add_argument("--max_samples", default=0, type=int,
                        help="最大样本数 (0=全量, >0=小规模测试)")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 查找数据
    base_dirs = [
        os.path.join(_PARENT, "..", "..", "split_data"),
        os.path.join(_PARENT, "..", "..", "..", "split_data"),
    ]
    train_path = args.train_data
    val_path = args.val_data
    for bd in base_dirs:
        if not train_path and os.path.exists(os.path.join(bd, "train.json")):
            train_path = os.path.join(bd, "train.json")
        if not val_path and os.path.exists(os.path.join(bd, "val.json")):
            val_path = os.path.join(bd, "val.json")

    if not train_path or not os.path.exists(train_path):
        print(f"[错误] 找不到训练数据, 请用 --train_data 指定")
        sys.exit(1)

    print(f"模板: {args.template}")
    print(f"训练集: {train_path}")
    print(f"验证集: {val_path or '(无)'}")
    print(f"最大样本: {args.max_samples or '全量'}")
    print(f"输出: {args.output_dir}")

    # 加载
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"原始训练集: {len(train_data)} 条")

    # 限制数量 (按原始比例分层采样, 保持与 benchmark 一致的分布)
    if args.max_samples > 0 and len(train_data) > args.max_samples:
        by_label = {}
        for row in train_data:
            label = row.get("标签", "无毒")
            by_label.setdefault(label, []).append(row)

        # 按原始比例采样, 不均匀分配
        total_orig = len(train_data)
        sampled = []
        for label, rows in by_label.items():
            # 该类别在原始数据中的占比 × 目标总数
            n = max(1, round(len(rows) / total_orig * args.max_samples))
            n = min(n, len(rows))
            sampled.extend(random.sample(rows, n))
        random.shuffle(sampled)
        train_data = sampled[:args.max_samples]
        print(f"采样后: {len(train_data)} 条 (按原始比例分层采样)")

    # 构建 SFT 数据
    train_samples = []
    for row in train_data:
        sample = build_sft_sample(row, template=args.template)
        if sample:
            train_samples.append(sample)

    # 保存
    train_out = os.path.join(args.output_dir, "train.jsonl")
    with open(train_out, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"训练集: {train_out} ({len(train_samples)} 条)")

    # 验证集
    if val_path and os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        if args.max_samples > 0:
            val_data = val_data[:min(300, len(val_data))]

        val_samples = []
        for row in val_data:
            sample = build_sft_sample(row, template=args.template)
            if sample:
                val_samples.append(sample)

        val_out = os.path.join(args.output_dir, "val.jsonl")
        with open(val_out, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"验证集: {val_out} ({len(val_samples)} 条)")

    # 统计
    label_counter = Counter()
    for row in train_data:
        label_counter[row.get("标签", "?")] += 1
    print(f"\n类别分布:")
    for label, count in label_counter.most_common():
        print(f"  {label}: {count}")

    # 检查 prompt 长度
    if train_samples:
        user_lens = [len(s["messages"][1]["content"]) for s in train_samples[:100]]
        avg_len = sum(user_lens) / len(user_lens)
        print(f"\nUser prompt 平均长度: {avg_len:.0f} chars")

    print(f"\n完成! SFT 数据保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
