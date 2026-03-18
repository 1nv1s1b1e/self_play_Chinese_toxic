# Integrated Self-Play: 统一对抗博弈训练系统

> 整合 Plan 1 (逐样本对抗奖励) + Plan 3 (数据管线优化) + Plan 4 (API Verifier) 为统一可运行的代码库。

## 架构概览

```
Stackelberg Self-Play Loop (每轮):
┌─────────────────────────────────────────────────────────────┐
│  Phase 0 — 动态数据生成                                       │
│    Challenger 生成有毒文本                                     │
│    [Plan 3a] 拒绝采样过滤低质量输出                             │
│    Reviewer 进行审核                                          │
│    [Plan 4]  Verifier 判定 (local / api / async)              │
│    [Plan 1]  注入 per-sample 对抗信号到 parquet                │
├─────────────────────────────────────────────────────────────┤
│  Phase A — Challenger GRPO 强化学习                            │
│    [Plan 1] 逐样本对抗奖励: quality_gate × adversarial_signal  │
├─────────────────────────────────────────────────────────────┤
│  Phase B — Reviewer SFT 监督微调                               │
│    [Plan 3b] 自适应回放比例 (随轮次递减)                        │
│    [Plan 3c] 多轮缓冲区 (最近 K 轮数据)                        │
│    LoRA 微调 → 合并                                           │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
integrated_selfplay/
├── __init__.py                     # 包标记
├── constants.py                    # 统一常量: RULES, 分类, prompt 模板, 解析函数
├── quality_gate.py                 # 共享质量门控 (消除 4× 重复)
├── rejection_sampler.py            # [Plan 3a] 拒绝采样过滤器
├── mix_replay_adaptive.py          # [Plan 3b+3c] 自适应回放 + 多轮缓冲区
├── verifier.py                     # 统一 Verifier: local + api + async (Plan 4)
├── challenger_reward.py            # [Plan 1] 逐样本对抗奖励
├── build_parquet.py                # 统一 parquet 构建 (注入 per-sample 信号)
├── generate_dynamic_data.py        # Phase 0: 统一动态数据生成
├── adversarial_trl_grpo.py         # Phase A: GRPO 训练 (直接集成 Plan 1 奖励)
├── convert_grpo_to_sft.py          # GRPO → SFT 格式转换
├── eval_selfplay_results.py        # 3 模式评估 (challenger ASR / reviewer F1 / 汇总曲线)
├── ds_zero2.json                   # DeepSpeed ZeRO-2 配置
├── run_selfplay.sh                 # 主循环脚本 (helper 函数化)
├── run_selfplay_npu.sh             # NPU 入口别名 → run_selfplay.sh
├── run_05_trl_grpo.sh              # 多模式入口 (selfplay / single / eval)
├── check_env_npu.sh                # NPU 环境预检脚本
├── configs/
│   ├── challenger_grpo_npu.yaml    # verl Challenger GRPO 配置
│   └── reviewer_grpo_npu.yaml      # verl Reviewer GRPO 配置 (备用)
├── reward_functions/
│   ├── __init__.py
│   ├── reviewer_reward.py          # Reviewer 多维度奖励函数
│   ├── reward_logger.py            # 3 级结构化奖励日志
│   └── llm_judge.py                # LLM-as-a-Judge 标签忠实度验证
├── tests/
│   ├── test_03_reward_fn.py        # 奖励函数单元测试
│   └── test_qwen_api.py            # Qwen API 连通性 + 分类测试
└── README.md                       # 本文档
```

## 各 Plan 改进点

### Plan 1: 逐样本对抗奖励信号
- **问题**: v1 使用整批统计 (fooling_rate) 作为所有样本的奖励，信号噪声大
- **改进**: 每个 Challenger 样本携带独立的 `reviewer_fooled` 信号
- **实现**: `challenger_reward.py` + `build_parquet.py` 注入 `extra_info`
- **公式**: `reward = quality_gate(text) × adversarial_signal`，映射到 [-1, 1]

### Plan 3: 数据管线优化
- **3a 拒绝采样**: `rejection_sampler.py` — 过滤 quality_gate < 阈值的 Challenger 输出
- **3b 自适应回放**: `mix_replay_adaptive.py` — seed_ratio 随轮次递减 {1:3.0, 2:2.0, 3:1.0, 4+:0.5}
- **3c 多轮缓冲区**: 加载最近 K 轮的动态数据，防止遗忘早期分布

### Plan 4: API Verifier
- **问题**: 本地冻结 7B 模型判断能力受限
- **改进**: 支持 72B+ 大模型 API (DashScope / OpenAI 兼容) 作为 Verifier
- **实现**: `verifier.py` 统一 3 种后端 (`local` / `api` / `async`)
- **降级**: API 不可用时自动回退本地模型

## 快速开始

### 环境预检

```bash
bash scripts/integrated_selfplay/check_env_npu.sh
```

### 基本运行 (本地 Verifier)

```bash
bash scripts/integrated_selfplay/run_selfplay.sh
# 或通过别名:
bash scripts/integrated_selfplay/run_selfplay_npu.sh
```

### 多模式入口

```bash
# 完整自对弈循环
MODE=selfplay bash scripts/integrated_selfplay/run_05_trl_grpo.sh

# 单次 GRPO (调试用)
MODE=single ROLE=challenger bash scripts/integrated_selfplay/run_05_trl_grpo.sh

# 仅评估
MODE=eval bash scripts/integrated_selfplay/run_05_trl_grpo.sh
```

### 使用 API Verifier

```bash
VERIFIER_API_KEY=sk-xxx \
VERIFIER_BACKEND=async \
bash scripts/integrated_selfplay/run_selfplay.sh
```

### 自定义参数

```bash
MODEL_SIZE=7B \
N_GPUS=8 \
SELFPLAY_ROUNDS=10 \
BUFFER_K=3 \
ENABLE_REJECTION=1 \
VERIFIER_BACKEND=api \
VERIFIER_API_KEY=sk-xxx \
VERIFIER_API_MODEL=qwen-max \
bash scripts/integrated_selfplay/run_selfplay.sh
```

### 评估

```bash
# 评估 Challenger ASR (攻击成功率)
python eval_selfplay_results.py \
    --mode challenger \
    --model_path /path/to/challenger \
    --reviewer_model /path/to/reviewer

# 评估 Reviewer F1 / Accuracy (支持 parquet / json / jsonl)
python eval_selfplay_results.py \
    --mode reviewer \
    --model_path /path/to/reviewer \
    --test_data /path/to/test.parquet

# 汇总所有轮次曲线
python eval_selfplay_results.py \
    --mode summary \
    --stats_dir /path/to/selfplay_dynamic_data/3B
```

## 环境变量参考

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_SIZE` | `3B` | 模型尺寸 (3B / 7B / 14B) |
| `N_GPUS` | `4` | NPU 卡数 |
| `SELFPLAY_ROUNDS` | `5` | 自对弈轮次 |
| `MAX_STEPS` | `50` | 每阶段 GRPO 训练步数 |
| `SAMPLES_PER_CAT` | `256` | Phase 0 每类别采样数 |
| `BUFFER_K` | `2` | [Plan 3c] 缓冲区保留最近 K 轮 |
| `ENABLE_REJECTION` | `1` | [Plan 3a] 是否启用拒绝采样 |
| `VERIFIER_BACKEND` | `local` | [Plan 4] Verifier 后端: local / api / async |
| `VERIFIER_API_KEY` | - | API 密钥 (api/async 必需) |
| `VERIFIER_API_BASE` | DashScope | API 端点 |
| `VERIFIER_API_MODEL` | `qwen-plus` | API 模型名称 |
| `REWARD_DEBUG` | `0` | 奖励日志级别 (0=关 / 1=摘要 / 2=详细) |
| `RESUME` | `1` | 是否断点续训 |
| `LLM_JUDGE_API_KEY` | - | LLM Judge API 密钥 (标签验证) |
| `LLM_JUDGE_MODEL` | `qwen-plus` | LLM Judge 使用的模型 |

## 相对于 rl_train (v1) 的改进

| 改进项 | rl_train (v1) | 集成版 |
|---|---|---|
| 常量定义 | 分散在 5+ 文件中重复 | `constants.py` 统一管理 |
| quality_gate | 复制 4 份 | `quality_gate.py` 提取为共享模块 |
| 奖励信号 | 整批 fooling_rate | 逐样本 per-sample signal |
| Challenger GRPO | monkey-patch (`_fixed.py`) | 直接 import，无需 monkey-patch |
| Verifier | 3 套独立实现 | 统一 factory + 3 后端 |
| 回放策略 | 固定 seed_ratio=2.0 | 自适应递减 + 多轮 buffer |
| Challenger 过滤 | 无 | 拒绝采样 (quality_gate 阈值) |
| 主循环结构 | 内联所有阶段 | helper 函数 (`run_phase0_datagen`, `run_grpo_phase`) |
| 轮次元数据 | 无 | `--selfplay_round` 参数注入 metadata |
| 标签验证 | 无 | `llm_judge.py` LLM-as-a-Judge |
| 标签一致性 | 无 | `rejection_sampler.py` Step 4.5 过滤 |
| 评估脚本 | 独立零散 | `eval_selfplay_results.py` 3 模式集成 |
| 入口脚本 | 单一入口 | `run_05_trl_grpo.sh` 多模式 + 别名 |
| 环境检查 | 基础检查 | `check_env_npu.sh` 7 项全面检查 |
| 单元测试 | 基础测试 | `tests/` 奖励函数 + API 全覆盖 |
| verl 配置 | YAML 文件 | `configs/` 目录规范管理 |

## 硬件要求

- **NPU**: 昇腾 910B × 4+ 卡
- **框架**: PyTorch + torch_npu + HCCL
- **训练**: TRL GRPOTrainer + DeepSpeed ZeRO-2
- **环境**: conda `ssp_train`
