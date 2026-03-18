# Plan 3: 数据管线优化 (Data Pipeline)

## 包含三项独立子改进

### 3a. 拒绝采样 (Rejection Sampling)

**问题**: v1 Phase 0 中 Challenger 生成的所有文本都进入 GRPO parquet，包括质量极差的垃圾文本（AI 拒绝、纯重复、太短等）。SSP 论文指出："过滤无效输出能显著提升训练稳定性"。

**改进**: 在 Phase 0 Step 7 之前，用 `quality_gate` 过滤，只保留 gate > 0.3 的样本。

### 3b. 自适应回放比例

**问题**: v1 `mix_replay_data.py` 的 `seed_ratio=2.0` 全程固定。早期轮次动态数据少、质量不稳定时需要多种子数据稳定训练；后期轮次动态数据质量提升，应降低种子依赖以强化对新对抗模式的适应。

**改进**: 回放比例随轮次递减：
```
Round 1: seed_ratio=3.0  (保守，多种子稳定)
Round 2: seed_ratio=2.0  (v1 默认)
Round 3: seed_ratio=1.0  (均衡)
Round 4+: seed_ratio=0.5 (激进，主要用新数据)
```

### 3c. 回放缓冲区周期性重置

**问题**: v1 每轮只混合种子 + 本轮新数据，不利用之前轮次的动态数据。SSP 论文 Table 5: "Periodic Reset (每 10 step 重置)" 比 "Full Reuse" 高 +4.3 分。

**改进**: 维护最近 K 轮的动态数据缓冲区 (默认 K=2)，和种子数据一起混合。

## 修改的文件

| 文件 | 作用 |
|------|------|
| `rejection_sampler.py` | 拒绝采样过滤器 |
| `mix_replay_adaptive.py` | 自适应回放比例 + 多轮缓冲区 |
| `run_selfplay_plan3.sh` | 修改版主循环 |
| `README.md` | 本文件 |

## 如何使用

```bash
bash scripts/plan_data_pipeline/run_selfplay_plan3.sh
```

## 可与方案 1/2 组合

完全正交: 方案 1/2 改 reward/优化器，方案 3 改数据管线。
