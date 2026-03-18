#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 1: 奖励方差对比 — 证明多级奖励比 binary 奖励有更大方差
==========================================================
核心假设: 旧版 binary 奖励 (+1/-1) 下, 大部分 rollout 拿相同分数,
GRPO 无法区分好坏; 新版多级奖励下, 不同 rollout 拿不同分数,
GRPO 有梯度信号.

用法:
  python tests/test_reward_variance.py

无需 GPU, 纯离线模拟.
"""

import os, sys, json, statistics
from collections import Counter

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)
sys.path.insert(0, os.path.join(_PARENT, "reward_functions"))

from constants import HARMFUL_CATEGORIES, parse_classification_output


# ── 旧版 binary-only 奖励 ──
def old_binary_reward(solution_str, ground_truth):
    pred = parse_classification_output(solution_str)
    true_label = ground_truth.get("category", "")
    true_is_harmful = true_label in HARMFUL_CATEGORIES
    if pred["binary"] is None:
        return -1.0
    reviewer_says_harmful = (pred["binary"] == "有害")
    binary_correct = (reviewer_says_harmful == true_is_harmful)
    return 1.0 if binary_correct else -1.0


# ── 新版多级奖励 ──
from reward_functions.reviewer_reward import compute_score as new_multilevel_reward


def simulate_grpo_rollouts():
    """
    模拟一个有害文本 prompt 的 8 个 GRPO rollout.

    现实场景: Reviewer 对 "性别歧视" 文本生成 8 个不同输出:
    - 一些输出正确类别
    - 一些输出错误类别 (但 binary 对)
    - 一些 binary 就判错了
    """
    ground_truth = {"category": "性别歧视"}

    # 模拟真实的 8 个 rollout (温度=0.7 下可能的输出)
    rollout_outputs = [
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # binary+cat 对
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：显式仇恨",  # binary+cat 对
        "判断：有害\n类别：种族歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # binary对, cat错
        "判断：有害\n类别：其他仇恨\n毒性类型：一般攻击性\n表达方式：显式仇恨",  # binary对, cat错
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：转述",      # binary+cat 对
        "判断：有害\n类别：地域偏见\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # binary对, cat错
        "判断：无害",                                                         # binary 错
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # binary+cat 对
    ]

    return ground_truth, rollout_outputs


def simulate_nontoxic_rollouts():
    """模拟无毒文本的 8 个 rollout."""
    ground_truth = {"category": "无毒"}
    rollout_outputs = [
        "判断：无害",
        "判断：无害",
        "判断：无害",
        "判断：无害",
        "判断：无害",
        "判断：有害\n类别：性别歧视",  # FP
        "判断：无害",
        "判断：无害",
    ]
    return ground_truth, rollout_outputs


def test_reward_variance():
    print("=" * 70)
    print("测试 1: 奖励方差对比 (旧版 binary vs 新版多级)")
    print("=" * 70)

    scenarios = [
        ("有害文本(性别歧视)", simulate_grpo_rollouts),
        ("无毒文本", simulate_nontoxic_rollouts),
    ]

    for name, gen_fn in scenarios:
        gt, outputs = gen_fn()
        print(f"\n── 场景: {name} (ground_truth={gt['category']}) ──")
        print(f"   8 个 rollout 输出:")

        old_rewards = []
        new_rewards = []

        for i, output in enumerate(outputs):
            old_r = old_binary_reward(output, gt)
            new_r = new_multilevel_reward("toxicn_reviewer", output, gt)
            old_rewards.append(old_r)
            new_rewards.append(new_r)

            parsed = parse_classification_output(output)
            cat = parsed.get("category", "?")
            print(f"     [{i}] {output[:40]:40s} -> old={old_r:+.1f}  new={new_r:+.1f}  (cat={cat})")

        # 统计
        old_var = statistics.variance(old_rewards) if len(old_rewards) > 1 else 0
        new_var = statistics.variance(new_rewards) if len(new_rewards) > 1 else 0
        old_unique = len(set(old_rewards))
        new_unique = len(set(new_rewards))

        print(f"\n   旧版 binary:  rewards={[f'{r:+.1f}' for r in old_rewards]}")
        print(f"                 unique={old_unique}  variance={old_var:.4f}  mean={statistics.mean(old_rewards):+.3f}")
        print(f"   新版多级:     rewards={[f'{r:+.1f}' for r in new_rewards]}")
        print(f"                 unique={new_unique}  variance={new_var:.4f}  mean={statistics.mean(new_rewards):+.3f}")

        # 判定
        if new_var > old_var:
            improvement = new_var / old_var if old_var > 0 else float('inf')
            print(f"   >>> 新版方差是旧版的 {improvement:.1f}x — GRPO 梯度信号更丰富")
        elif new_var == old_var:
            print(f"   >>> 方差相同 — 此场景无差异")
        else:
            print(f"   >>> 警告: 新版方差更小")

    print()


def test_category_discrimination_signal():
    """验证: 新版奖励能区分 category-correct 和 category-wrong, 旧版不能."""
    print("=" * 70)
    print("测试 2: Category 区分信号")
    print("=" * 70)

    gt = {"category": "性别歧视"}

    pairs = [
        ("cat正确", "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨"),
        ("cat错误", "判断：有害\n类别：种族歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨"),
    ]

    print(f"\n   Ground truth: {gt['category']}")
    for label, output in pairs:
        old_r = old_binary_reward(output, gt)
        new_r = new_multilevel_reward("toxicn_reviewer", output, gt)
        print(f"   {label}: old={old_r:+.1f}  new={new_r:+.1f}")

    old_diff = abs(old_binary_reward(pairs[0][1], gt) - old_binary_reward(pairs[1][1], gt))
    new_diff = abs(new_multilevel_reward("toxicn_reviewer", pairs[0][1], gt) -
                   new_multilevel_reward("toxicn_reviewer", pairs[1][1], gt))

    print(f"\n   旧版 cat正确 vs cat错误 差异: {old_diff:.2f}")
    print(f"   新版 cat正确 vs cat错误 差异: {new_diff:.2f}")

    if old_diff == 0:
        print("   >>> 旧版完全无法区分 cat 正确/错误 — 这就是平台期的根因!")
    if new_diff > 0:
        print(f"   >>> 新版差异 {new_diff:.1f} — GRPO 可以优选 category 正确的 rollout")
    print()


def test_reward_distribution_on_real_eval():
    """用真实评估数据模拟奖励分布."""
    print("=" * 70)
    print("测试 3: 真实评估数据上的奖励分布对比")
    print("=" * 70)

    eval_files = [
        os.path.join(_PARENT, "eval_result", "eval_vllm_npu_merged_baseline_baseline_1000.json"),
        os.path.join(_PARENT, "..", "..", "eval_vllm_npu_reviewer_3B_reviewer_round0_baseline.json"),
    ]

    eval_file = None
    for f in eval_files:
        if os.path.exists(f):
            eval_file = f
            break

    if eval_file is None:
        print("   [跳过] 未找到评估数据文件")
        return

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    old_rewards = []
    new_rewards = []

    for r in results:
        true_label = r.get("标签", "")
        response = r.get("response", "")
        gt = {"category": true_label}

        old_r = old_binary_reward(response, gt)
        new_r = new_multilevel_reward("toxicn_reviewer", response, gt)
        old_rewards.append(old_r)
        new_rewards.append(new_r)

    print(f"\n   评估数据: {eval_file.split('/')[-1]} ({len(results)} 条)")

    # 分布统计
    print(f"\n   旧版 binary 奖励分布:")
    old_counter = Counter(old_rewards)
    for val, cnt in sorted(old_counter.items()):
        pct = 100 * cnt / len(old_rewards)
        bar = "#" * int(pct / 2)
        print(f"     {val:+.1f}: {cnt:5d} ({pct:5.1f}%) {bar}")

    print(f"\n   新版多级奖励分布:")
    new_counter = Counter(new_rewards)
    for val, cnt in sorted(new_counter.items()):
        pct = 100 * cnt / len(new_rewards)
        bar = "#" * int(pct / 2)
        print(f"     {val:+.1f}: {cnt:5d} ({pct:5.1f}%) {bar}")

    old_var = statistics.variance(old_rewards)
    new_var = statistics.variance(new_rewards)

    print(f"\n   旧版方差: {old_var:.4f}  unique值: {len(set(old_rewards))}")
    print(f"   新版方差: {new_var:.4f}  unique值: {len(set(new_rewards))}")

    if new_var > old_var:
        print(f"   >>> 新版方差提升 {new_var/old_var:.2f}x — GRPO 获得更丰富的梯度信号")
    print()


if __name__ == "__main__":
    test_reward_variance()
    test_category_discrimination_signal()
    test_reward_distribution_on_real_eval()
    print("=" * 70)
    print("所有离线测试完成")
    print("=" * 70)
