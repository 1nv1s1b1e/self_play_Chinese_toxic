#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版 adversarial_trl_grpo — 使用逐样本真对抗奖励信号
======================================================

与 v1 adversarial_trl_grpo.py 的唯一区别:
  Challenger reward 函数替换为 challenger_reward_adversarial.py (逐样本信号)

v1 的缺陷:
  adversarial_bonus = 0.25 × verifier_asr(类别级常量)
  → 同类别样本 bonus 相同 → GRPO group-relative advantage 消去 → 无梯度

修复:
  R = quality_gate × (reviewer_fooled ? 1.0 : 0.0)
  → 逐样本信号 → GRPO advantage 中 fooled/not-fooled 产生真实梯度差异

本文件供所有 Plan 的 Phase A 阶段使用 (替代 adversarial_trl_grpo.py)。

用法:
  python -m torch.distributed.run --nproc_per_node=4 \\
      scripts/rl_train/adversarial_trl_grpo_fixed.py \\
      --role challenger \\
      --model_path /path/to/model \\
      --dataset_path /path/to/challenger_grpo_round1.parquet \\
      --output_dir /path/to/output \\
      --max_steps 50 \\
      --deepspeed scripts/rl_train/ds_zero2.json
"""

import os
import sys

# ── 路径设置: 本文件位于 scripts/rl_train/ ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "reward_functions"))

# ── 加载逐样本对抗奖励函数 (同目录) ──
from challenger_reward_adversarial import compute_score as challenger_adversarial_reward

# ── 复用 v1 的 reviewer reward ──
from reviewer_reward import compute_score as reviewer_reward_func

# ── 复用 v1 的 main() 逻辑，仅替换 reward 包装器 ──
import adversarial_trl_grpo as v1_grpo

# ── Monkey-patch Challenger reward ──
_original_get_challenger_reward = v1_grpo.get_challenger_reward


def get_challenger_reward_fixed(use_selfplay: bool):
    """
    修复版 Challenger reward 包装器。
    始终使用真对抗奖励 (逐样本 reviewer_fooled)，忽略 use_selfplay 参数。
    """
    def reward_fn(prompts, completions, **kwargs):
        scores = []
        reward_models = kwargs.get("reward_model", [])
        extra_infos   = kwargs.get("extra_info",   [])

        for i in range(len(prompts)):
            solution_str = completions[i][0]["content"] if completions[i] else ""

            gt = ""
            extra = {}
            if i < len(reward_models):
                rm = reward_models[i]
                if isinstance(rm, dict):
                    gt = rm
                    extra = dict(extra_infos[i]) if i < len(extra_infos) and isinstance(extra_infos[i], dict) else {}
                elif isinstance(rm, str):
                    gt = rm

            if i < len(extra_infos) and isinstance(extra_infos[i], dict) and not extra:
                extra = extra_infos[i]

            try:
                score = challenger_adversarial_reward(
                    data_source="toxicn_challenger",
                    solution_str=solution_str,
                    ground_truth=gt,
                    extra_info=extra,
                )
            except Exception as ex:
                print(f"[Reward Fix Warning] adversarial reward error at index {i}: {ex}")
                score = 0.0

            scores.append(float(score))

        return scores

    return reward_fn


# ── 替换 v1 的 reward 获取函数 ──
v1_grpo.get_challenger_reward = get_challenger_reward_fixed


def _get_reward_wrapper_fixed(role: str, use_selfplay: bool = True):
    """统一入口: Challenger 用修复版 reward, Reviewer 用 v1 reward。"""
    if role == "challenger":
        return get_challenger_reward_fixed(use_selfplay=use_selfplay)
    elif role == "reviewer":
        return v1_grpo.get_reviewer_reward()
    else:
        raise ValueError(f"未知 role: {role}")


v1_grpo.get_reward_wrapper = _get_reward_wrapper_fixed


if __name__ == "__main__":
    print("[Reward Fix] 使用逐样本真对抗奖励信号 (reviewer_fooled)")
    v1_grpo.main()
