# 对抗博弈 RL 训练 — 昇腾 910B 单机多卡

## 概述

本模块实现了基于 **verl GRPO** 的 Challenger–Reviewer 对抗博弈强化学习训练，针对**华为昇腾 910B NPU 单机多卡**环境优化。

```
┌─────────────────────────────────────────────────────────────┐
│  Challenger  →  GRPO 强化学习  →  生成对抗性有害文本         │
│  Reviewer    →  GRPO 强化学习  →  检测/分类有害文本           │
│  自对弈循环  →  Stackelberg 博弈  →  双方能力螺旋上升         │
└─────────────────────────────────────────────────────────────┘
```

## 环境要求

| 软件 | 版本 |
|------|------|
| Python | >= 3.10 |
| CANN | == 8.3.RC1 |
| torch | == 2.7.1 |
| torch_npu | == 2.7.1 |
| vllm | == 0.11.0 (empty device) |
| vllm-ascend | == 0.11.0rc1 |
| verl | latest |
| triton-ascend | == 3.2.0rc4 |

**昇腾暂不支持**: `flash_attn`、`liger-kernel`（已在脚本中自动使用 XFORMERS 替代）

## 快速安装

```bash
# 1. 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 2. 安装 torch_npu
pip install torchvision==0.22.1
pip install triton-ascend==3.2.0rc4

# 3. 安装 vllm (昇腾空设备模式)
git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git
cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..

# 4. 安装 vllm-ascend
git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -v -e . && cd ..

# 5. 安装 verl
git clone --depth 1 https://github.com/volcengine/verl.git
cd verl && pip install -r requirements-npu.txt && pip install -v -e . && cd ..

# 6. 检查环境
bash scripts/rl_train/check_env_npu.sh
```

## 文件结构

```
scripts/rl_train/
├── check_env_npu.sh              # 环境检查脚本
├── run_grpo_challenger_npu.sh    # Challenger GRPO 训练主脚本
├── run_grpo_reviewer_npu.sh      # Reviewer GRPO 训练主脚本
├── run_selfplay_npu.sh           # 自对弈循环主脚本 ★ 主入口
├── configs/
│   ├── challenger_grpo_npu.yaml  # verl YAML 配置 (Challenger)
│   └── reviewer_grpo_npu.yaml    # verl YAML 配置 (Reviewer)
└── reward_functions/
    ├── challenger_reward_v7.py   # Challenger 奖励函数 (Gate×Signal)
    └── reviewer_reward.py        # Reviewer 奖励函数 (binary+category)

scripts/run_pipeline/
├── run_07_rl_grpo_challenger.sh  # Pipeline 入口: Step 7
├── run_08_rl_grpo_reviewer.sh    # Pipeline 入口: Step 8
└── run_09_rl_selfplay.sh         # Pipeline 入口: Step 9 ★ 推荐
```

## 使用方式

### 方式一：完整自对弈训练（推荐）

```bash
cd scripts/run_pipeline

# 双卡，3B 模型，3 轮自对弈
N_GPUS=2 MODEL_SIZE=3B SELFPLAY_ROUNDS=3 bash run_09_rl_selfplay.sh

# 四卡，1.5B 模型，5 轮自对弈，每 Phase 2 epoch
N_GPUS=4 MODEL_SIZE=1.5B SELFPLAY_ROUNDS=5 EPOCHS_PER_PHASE=2 \
    bash run_09_rl_selfplay.sh

# 自定义 BASE_DIR
BASE_DIR=/your/data/path N_GPUS=2 bash run_09_rl_selfplay.sh
```

### 方式二：单独训练某个角色

```bash
# 仅训练 Challenger
N_GPUS=2 MODEL_SIZE=3B RL_EPOCHS=3 bash run_07_rl_grpo_challenger.sh

# 仅训练 Reviewer
N_GPUS=2 MODEL_SIZE=3B RL_EPOCHS=3 bash run_08_rl_grpo_reviewer.sh
```

### 方式三：直接调用核心脚本

```bash
BASE_DIR=/home/ma-user/work/test \
MODEL_SIZE=3B \
N_GPUS=2 \
RL_EPOCHS=3 \
bash scripts/rl_train/run_grpo_challenger_npu.sh
```

## 完整 Pipeline

```
run_01_download.sh          →  下载 Qwen2.5 基座模型
run_02_prepare_data.sh      →  准备 SFT 训练数据
run_03_lora_sft.sh          →  LoRA SFT 训练 (Challenger + Reviewer)
run_04_merge_lora.sh        →  合并 LoRA 权重
run_05_evaluate.sh          →  SFT 效果评估
run_06_prepare_rl_data.sh   →  准备 RL parquet 数据
run_07_rl_grpo_challenger.sh →  Challenger GRPO RL 训练  ← 新增
run_08_rl_grpo_reviewer.sh  →  Reviewer GRPO RL 训练    ← 新增
run_09_rl_selfplay.sh       →  自对弈循环训练 ★          ← 新增
```

## 关键技术参数

### 昇腾 910B 分布式关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `HCCL_WHITELIST_DISABLE` | 1 | 关闭白名单，允许任意节点通信 |
| `HCCL_CONNECT_TIMEOUT` | 7200 | 大模型加载需较长时间 |
| `VLLM_ATTENTION_BACKEND` | XFORMERS | 替代不支持的 flash_attn |
| `trainer.device` | npu | 激活昇腾 NPU 支持 |
| `rollout.name` | vllm-ascend | 昇腾专用推理后端 |
| `actor.strategy` | fsdp | 全切片数据并行 |
| 通信后端 | hccl | 昇腾集合通信 |

### 模型适配参数

| 模型尺寸 | PPO mini-batch | micro-batch/GPU | Tensor 并行 |
|---------|---------------|-----------------|------------|
| 0.5B | 32 | 8 | 1 |
| 1.5B | 16 | 4 | 1 |
| 3B | 16 | 2 | 2 |

### GRPO 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rollout.n` | 8 | 每 prompt 采样 8 个候选 (group size) |
| `kl_loss_coef` | 0.001 | KL 散度正则化系数 |
| `entropy_coeff` | 0.001 | 熵正则化系数 |
| `kl_loss_type` | low_var_kl | 低方差 KL 估计 |

## 奖励函数

### Challenger (`challenger_reward_v7.py`)
- 架构：**Gate × Signal**（乘法门控）
- Signal：话题相关性（multi-scale character n-gram Jaccard）
- Gate：质量门控（长度 × 重复 × 格式 × 多样性）
- 范围：`[-1.0, 1.0]`

### Reviewer (`reviewer_reward.py`)
- 基础分（权重 0.7）：二值分类 + 类别识别
- 附加分（权重 0.3）：toxic_type + expression
- 误检惩罚：-0.5；漏检惩罚：-1.0

## 注意事项

1. **端口冲突**：Challenger Phase 使用 29600，Reviewer Phase 使用 29601，避免串行训练时端口残留
2. **显存管理**：3B 模型建议 `GPU_MEM_UTIL=0.6`，显存不足时开启 `PARAM_OFFLOAD=True`
3. **checkpoint 格式**：verl 保存在 `{output_dir}/actor/global_step_N/`，自对弈脚本自动追踪最新 checkpoint
4. **Bus Error**：昇腾 ModelArts 偶发 Bus error 属已知问题，重新运行通常可解决

## 参考

- [verl Ascend 快速入门](https://ascend.github.io/docs/sources/_generated/sources/verl/ascend_quick_start.html)
- [LLaMA-Factory 昇腾分布式训练](https://ascend.github.io/docs/sources/_generated/sources/LLaMA-Factory/source/advanced/distributed.html)
- [昇腾单机多卡训练指南](https://blog.csdn.net/verse_armour/article/details/143567672)
