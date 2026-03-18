# Plan 2: REINFORCE 替代 GRPO (Challenger 侧)

## 核心问题

v1 的 Challenger GRPO 每个 prompt 生成 n=4 个 rollout:
- 75% 的推理时间用于产生对比组 (comparison group)
- 4 个 rollout 的方差往往很小 (同一 prompt → 相似输出)
- 有效 training step 受限: 同样的时间只能跑 MAX_STEPS=50 步

## 改进方案

SSP 论文 (arxiv 2510.18821) Table 3 的关键发现:
- **RF(n=1) for Proposer + GRPO for Solver** 是效率/效果最优组合
- Proposer(Challenger) 只需 n=1，因为 reward 已有足够信号 (fooled vs not-fooled)
- n=1 时 TRL GRPOTrainer 退化为 REINFORCE: advantage = reward - baseline

## 实现细节

1. `C_NUM_GEN=1` (从 4 降到 1)
2. 推理量降 75%，同样时间可跑 `MAX_STEPS=200` (~4倍)
3. baseline 使用 EMA of recent rewards (TRL 内置)
4. 其他所有逻辑不变

## 修改的文件

| 文件 | 作用 |
|------|------|
| `run_selfplay_plan2.sh` | 修改版主循环: C_NUM_GEN=1, MAX_STEPS=200 |
| `README.md` | 本文件 |

## 如何使用

```bash
bash scripts/plan_reinforce/run_selfplay_plan2.sh
```

## 可与方案 1 组合

方案 2 与方案 1 完全正交:
- 方案 1 改 reward 函数 → 改的是"优化什么"
- 方案 2 改 n=1 → 改的是"怎么优化"

组合使用:
```bash
# 在 Plan 1 shell 中设置 C_NUM_GEN=1 MAX_STEPS=200
C_NUM_GEN=1 MAX_STEPS=200 bash scripts/plan_reward_shaping/run_selfplay_plan1.sh
```

## 预期效果

- 训练效率提升 ~4x (每轮 Phase A 推理时间降 75%)
- 同样的总训练时间可以跑更多步/更多轮
- 效果与 GRPO(n=4) 相当 (SSP Table 3 验证)
