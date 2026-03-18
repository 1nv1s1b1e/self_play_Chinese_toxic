#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 4: Mini GRPO 模拟 — 证明多级奖励下 GRPO 能优选 category-correct
=====================================================================
不需要真实训练, 用模拟的 GRPO 选择过程展示:
  旧版: GRPO 无法区分 cat-correct 和 cat-wrong rollout
  新版: GRPO 优选 cat-correct rollout, 提升 category 准确率

GRPO 核心机制:
  对每个 prompt 的 K 个 rollout, 计算 advantage = reward - mean(rewards)
  advantage > 0 的 rollout 被强化, < 0 的被抑制

用法:
  python tests/test_mini_grpo_simulation.py
"""

import os, sys, random, statistics
from collections import Counter

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)
sys.path.insert(0, os.path.join(_PARENT, "reward_functions"))

from constants import HARMFUL_CATEGORIES, parse_classification_output
from reward_functions.reviewer_reward import compute_score as new_reward


def old_reward(output, gt):
    pred = parse_classification_output(output)
    true_is_harmful = gt["category"] in HARMFUL_CATEGORIES
    if pred["binary"] is None:
        return -1.0
    return 1.0 if (pred["binary"] == "有害") == true_is_harmful else -1.0


def simulate_grpo_selection(reward_fn, gt, rollouts, label=""):
    """
    模拟 GRPO 的 advantage-based 选择过程.

    GRPO: advantage_i = reward_i - mean(all_rewards)
    被强化的 rollout = advantage > 0 的那些
    """
    rewards = [reward_fn(o, gt) for o in rollouts]
    mean_r = statistics.mean(rewards)
    advantages = [r - mean_r for r in rewards]

    reinforced = []  # advantage > 0 的 rollout
    suppressed = []  # advantage < 0 的 rollout

    for i, (output, adv) in enumerate(zip(rollouts, advantages)):
        parsed = parse_classification_output(output)
        cat = parsed.get("category", "?")
        cat_correct = (cat == gt["category"])

        entry = {
            "idx": i,
            "category": cat,
            "cat_correct": cat_correct,
            "reward": rewards[i],
            "advantage": adv,
        }
        if adv > 0.001:  # 微小阈值避免浮点误差
            reinforced.append(entry)
        elif adv < -0.001:
            suppressed.append(entry)

    # 统计被强化/抑制的 rollout 中 category-correct 的比例
    reinforced_cat_correct = sum(1 for e in reinforced if e["cat_correct"])
    suppressed_cat_correct = sum(1 for e in suppressed if e["cat_correct"])

    return {
        "label": label,
        "rewards": rewards,
        "mean_reward": mean_r,
        "reinforced": reinforced,
        "suppressed": suppressed,
        "reinforced_count": len(reinforced),
        "suppressed_count": len(suppressed),
        "reinforced_cat_correct": reinforced_cat_correct,
        "suppressed_cat_correct": suppressed_cat_correct,
        "reinforced_cat_rate": reinforced_cat_correct / max(len(reinforced), 1),
        "suppressed_cat_rate": suppressed_cat_correct / max(len(suppressed), 1),
    }


def main():
    print("=" * 70)
    print("测试 4: Mini GRPO 模拟 (advantage-based 选择)")
    print("=" * 70)

    # 场景: 性别歧视文本, 8个rollout, 不同category输出
    gt = {"category": "性别歧视"}
    rollouts = [
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # correct
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：显式仇恨",  # correct
        "判断：有害\n类别：种族歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # cat wrong
        "判断：有害\n类别：其他仇恨\n毒性类型：一般攻击性\n表达方式：显式仇恨",  # cat wrong
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：转述",      # correct
        "判断：有害\n类别：地域偏见\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # cat wrong
        "判断：无害",                                                         # binary wrong
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨",  # correct
    ]

    print(f"\n场景: ground_truth={gt['category']}, 8个rollout:")
    for i, o in enumerate(rollouts):
        p = parse_classification_output(o)
        mark = "v" if p.get("category") == gt["category"] else "x"
        print(f"  [{i}] [{mark}] {o[:50]}")

    # 旧版
    print("\n" + "-" * 50)
    old_result = simulate_grpo_selection(
        lambda o, g: old_reward(o, g), gt, rollouts, "旧版 binary"
    )
    print(f"\n旧版 binary 奖励:")
    print(f"  rewards = {[f'{r:+.1f}' for r in old_result['rewards']]}")
    print(f"  mean = {old_result['mean_reward']:+.3f}")
    print(f"  被强化 (adv>0): {old_result['reinforced_count']} 个")
    for e in old_result["reinforced"]:
        print(f"    [{e['idx']}] cat={e['category']:6s} cat_ok={'Y' if e['cat_correct'] else 'N'} "
              f"reward={e['reward']:+.1f} adv={e['advantage']:+.3f}")
    print(f"  被抑制 (adv<0): {old_result['suppressed_count']} 个")
    for e in old_result["suppressed"]:
        print(f"    [{e['idx']}] cat={e['category']:6s} cat_ok={'Y' if e['cat_correct'] else 'N'} "
              f"reward={e['reward']:+.1f} adv={e['advantage']:+.3f}")

    print(f"\n  被强化 rollout 中 category-correct 占比: "
          f"{old_result['reinforced_cat_correct']}/{old_result['reinforced_count']} "
          f"= {old_result['reinforced_cat_rate']:.0%}")

    # 新版
    print("\n" + "-" * 50)
    new_result = simulate_grpo_selection(
        lambda o, g: new_reward("toxicn_reviewer", o, g), gt, rollouts, "新版多级"
    )
    print(f"\n新版多级奖励:")
    print(f"  rewards = {[f'{r:+.1f}' for r in new_result['rewards']]}")
    print(f"  mean = {new_result['mean_reward']:+.3f}")
    print(f"  被强化 (adv>0): {new_result['reinforced_count']} 个")
    for e in new_result["reinforced"]:
        print(f"    [{e['idx']}] cat={e['category']:6s} cat_ok={'Y' if e['cat_correct'] else 'N'} "
              f"reward={e['reward']:+.1f} adv={e['advantage']:+.3f}")
    print(f"  被抑制 (adv<0): {new_result['suppressed_count']} 个")
    for e in new_result["suppressed"]:
        print(f"    [{e['idx']}] cat={e['category']:6s} cat_ok={'Y' if e['cat_correct'] else 'N'} "
              f"reward={e['reward']:+.1f} adv={e['advantage']:+.3f}")

    print(f"\n  被强化 rollout 中 category-correct 占比: "
          f"{new_result['reinforced_cat_correct']}/{new_result['reinforced_count']} "
          f"= {new_result['reinforced_cat_rate']:.0%}")

    # 对比
    print("\n" + "=" * 70)
    print("对比总结:")
    print("=" * 70)
    print(f"  旧版: 被强化{old_result['reinforced_count']}个中 category-correct "
          f"= {old_result['reinforced_cat_rate']:.0%}")
    print(f"  新版: 被强化{new_result['reinforced_count']}个中 category-correct "
          f"= {new_result['reinforced_cat_rate']:.0%}")

    if new_result["reinforced_cat_rate"] > old_result["reinforced_cat_rate"]:
        print("\n  >>> 新版 GRPO 优选了更多 category-correct 的 rollout!")
        print("  >>> 这意味着模型会学习输出正确的 category, 而不仅仅是正确的 binary.")
    elif old_result["reinforced_cat_rate"] == new_result["reinforced_cat_rate"]:
        print("\n  >>> 两版选中的 cat-correct 比例相同 (可能此场景差异不大)")
    else:
        print("\n  >>> 警告: 旧版反而更好 (需要检查)")

    # 额外: 模拟多个不同场景取平均
    print("\n" + "=" * 70)
    print("扩展: 100 个随机场景的平均效果")
    print("=" * 70)
    random.seed(42)

    old_rates = []
    new_rates = []

    for _ in range(100):
        cat = random.choice(HARMFUL_CATEGORIES)
        gt_sim = {"category": cat}

        sim_rollouts = []
        for _ in range(8):
            r = random.random()
            if r < 0.4:  # 40% cat correct
                sim_rollouts.append(f"判断：有害\n类别：{cat}")
            elif r < 0.75:  # 35% cat wrong
                alt = random.choice([c for c in HARMFUL_CATEGORIES if c != cat])
                sim_rollouts.append(f"判断：有害\n类别：{alt}")
            elif r < 0.85:  # 10% binary wrong
                sim_rollouts.append("判断：无害")
            else:  # 15% 其他仇恨
                sim_rollouts.append("判断：有害\n类别：其他仇恨")

        old_res = simulate_grpo_selection(
            lambda o, g: old_reward(o, g), gt_sim, sim_rollouts
        )
        new_res = simulate_grpo_selection(
            lambda o, g: new_reward("toxicn_reviewer", o, g), gt_sim, sim_rollouts
        )

        old_rates.append(old_res["reinforced_cat_rate"])
        new_rates.append(new_res["reinforced_cat_rate"])

    print(f"  旧版 avg reinforced cat-correct rate: {statistics.mean(old_rates):.1%}")
    print(f"  新版 avg reinforced cat-correct rate: {statistics.mean(new_rates):.1%}")
    improvement = statistics.mean(new_rates) - statistics.mean(old_rates)
    print(f"  提升: {improvement:+.1%}")
    if improvement > 0:
        print(f"\n  >>> 新版在 100 个场景中平均多选了 {improvement:.0%} 的 cat-correct rollout")
        print(f"  >>> 这直接转化为 GRPO 对 category 准确率的优化信号")
    print()


if __name__ == "__main__":
    main()
