#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 1 专用: 修改版 adversarial_trl_grpo.py — 使用真对抗奖励
==============================================================

唯一区别: Challenger reward 函数替换为 challenger_reward_adversarial.py

其他所有逻辑 (数据加载、GRPO 训练、Reviewer reward) 完全复用 v1。

用法:
  python -m torch.distributed.run --nproc_per_node=4 \\
      scripts/plan_reward_shaping/adversarial_trl_grpo_plan1.py \\
      --role challenger \\
      --model_path /path/to/model \\
      --dataset_path /path/to/challenger_grpo_round1.parquet \\
      --output_dir /path/to/output \\
      --max_steps 50 \\
      --deepspeed scripts/rl_train/ds_zero2.json
"""

import os
import sys

# ── 添加 v1 rl_train 到 sys.path，复用所有公共逻辑 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_TRAIN_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "rl_train")
sys.path.insert(0, RL_TRAIN_DIR)
sys.path.insert(0, os.path.join(RL_TRAIN_DIR, "reward_functions"))

# ── 加载 Plan 1 的新 reward 函数 ──
sys.path.insert(0, SCRIPT_DIR)
from challenger_reward_adversarial import compute_score as challenger_adversarial_reward

# ── 复用 v1 的 reviewer reward 和公共工具 ──
from reviewer_reward import compute_score as reviewer_reward_func

# ── 复用 v1 的 main() 逻辑，仅替换 reward 包装器 ──
# 我们直接修改 adversarial_trl_grpo 中的 reward 映射
import adversarial_trl_grpo as v1_grpo

# ── Monkey-patch Challenger reward ──
_original_get_challenger_reward = v1_grpo.get_challenger_reward

def get_challenger_reward_plan1(use_selfplay: bool):
    """
    Plan 1 版 Challenger reward 包装器。
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
                    gt = rm  # 传完整 dict 给 reward function
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
                print(f"[Plan1 Reward Warning] adversarial reward error at index {i}: {ex}")
                score = 0.0

            scores.append(float(score))

        return scores

    return reward_fn

# ── 替换 v1 的 reward 获取函数 ──
v1_grpo.get_challenger_reward = get_challenger_reward_plan1

def _get_reward_wrapper_plan1(role: str, use_selfplay: bool = True):
    """统一入口: Challenger 用 Plan1 reward, Reviewer 用 v1 reward。"""
    if role == "challenger":
        return get_challenger_reward_plan1(use_selfplay=use_selfplay)
    elif role == "reviewer":
        return v1_grpo.get_reviewer_reward()
    else:
        raise ValueError(f"未知 role: {role}")

v1_grpo.get_reward_wrapper = _get_reward_wrapper_plan1


if __name__ == "__main__":
    print("[Plan 1] 使用真对抗奖励信号 (逐样本 reviewer_fooled)")
    v1_grpo.main()
