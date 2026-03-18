# Plan 1: 真对抗奖励信号 (Reward Shaping)

## 核心问题

v1 的 `challenger_reward_selfplay.py` 中，`adversarial_bonus` 是**类别级常量**：
```
bonus = 0.25 × verifier_asr(类别) × verifier_confirms_rate(类别)
```

同一类别的所有样本得到相同的 bonus → GRPO 计算 group-relative advantage 时：
```
advantage_i = reward_i - mean(reward_group)
```
由于 bonus 是常量，它在减法中被消去 → **对抗信号不产生任何梯度**。

Challenger 实际上只在优化 `topic_signal`（n-gram 相似度），而不是"骗过 Reviewer"。

## 改进方案

借鉴 SSP 论文 (arxiv 2510.18821): $R_{proposer} = 1 - \bar{r}_{solve}$

将对抗信号从类别级改为**逐样本级**：

$$R_{challenger} = quality\_gate \times (1 - \mathbb{1}(\text{reviewer\_correct}))$$

- Reviewer 判对了 → reward = 0（没骗到）
- Reviewer 判错了 → reward = quality_gate（成功欺骗，且文本质量好）

## 修改的文件

| 文件 | 作用 |
|------|------|
| `challenger_reward_adversarial.py` | 新奖励函数: 逐样本对抗信号 |
| `generate_dynamic_data_rs.py` | 修改版 Phase 0: 在 challenger parquet 中注入逐样本 `reviewer_fooled` |
| `run_selfplay_plan1.sh` | 修改版主循环: 引用新 reward 和 data gen |

## 如何使用

```bash
# 替代 v1 的 run_selfplay_trl_npu.sh
bash scripts/plan_reward_shaping/run_selfplay_plan1.sh
```

## 预期效果

- Challenger 将学会生成"真正能骗过 Reviewer 的文本"，而非"形似参考的文本"
- GRPO 的 advantage 将在 fooled/not-fooled 样本间产生分化，gradient 有效
- 预期 ASR 在训练初期会快速上升，然后随 Reviewer SFT 适应后趋于对抗平衡
