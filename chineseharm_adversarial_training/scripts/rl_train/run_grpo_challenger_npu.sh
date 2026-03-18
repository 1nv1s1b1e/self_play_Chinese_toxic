#!/bin/bash
# =============================================================================
# Challenger GRPO RL 训练 — 昇腾 910B 单机多卡
# =============================================================================
# 算法 : GRPO (Group Relative Policy Optimization)
# 框架 : verl  (https://github.com/volcengine/verl)
# 硬件 : 华为昇腾 910B NPU，单机 N 卡 (默认 2 卡，支持 4/8 卡)
# 通信 : HCCL
# 策略 : FSDP (Fully Sharded Data Parallel)
# 推理 : vllm-ascend
#
# 用法:
#   N_GPUS=2 bash run_grpo_challenger_npu.sh               # 双卡
#   N_GPUS=4 bash run_grpo_challenger_npu.sh               # 四卡
#   N_GPUS=2 MODEL_SIZE=3B bash run_grpo_challenger_npu.sh # 指定模型尺寸
# =============================================================================
set -e

# ─────────────────────────────────────────────────────────────────
# 0. 初始化昇腾 CANN 运行环境
# ─────────────────────────────────────────────────────────────────
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    echo "✓ 已加载 ascend-toolkit 环境"
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
    echo "✓ 已加载 nnal/atb 环境"
fi

# ─────────────────────────────────────────────────────────────────
# 1. 昇腾 NPU 分布式通信环境变量
#    参考: 华为 ModelArts Notebook 昇腾 910B 单机多卡官方指南
# ─────────────────────────────────────────────────────────────────
export HCCL_WHITELIST_DISABLE=1          # 关闭白名单,允许任意节点通信
export HCCL_CONNECT_TIMEOUT=7200         # 连接超时 (秒), 大模型加载需较长时间
export HCCL_EXEC_TIMEOUT=7200            # 执行超时 (秒)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29600}

# ── 关键: 防止 Ray 在启动 Worker 时覆盖 Ascend 设备可见性 ──────
# 缺少此项会导致 HCCL hcclCommInitRootInfoConfig 在非0 Rank 失败 (error code 5)
export RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1

# ── 与 LoRA 脚本保持一致: 禁止 torch 在 Ray Worker 中自动加载 NPU backend ──────
# 自动加载就会与 HCCL 多进程初始化产生竞争，导致 error code 5 (两个 rank 同时失败)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ── Ascend RT 设备可见性 (按实际 N_GPUS 动态生成, 非固定8卡) ───
ASCEND_DEV_IDS=$(python3 -c "print(','.join(str(i) for i in range(${N_GPUS:-2})))" 2>/dev/null || echo "0,1")
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$ASCEND_DEV_IDS}

# ── HCCL bootstrap 网络接口 ─────────────────────────────────────
_HCCL_IF=$(python3 -c "
import socket, subprocess, sys
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8',80))
    lip = s.getsockname()[0]; s.close()
    r = subprocess.check_output(['ip','-4','-o','addr','show'],stderr=subprocess.DEVNULL,text=True)
    for ln in r.splitlines():
        if lip in ln: print(ln.split()[1]); sys.exit(0)
except Exception: pass
try:
    r = subprocess.check_output(['ip','-4','-o','addr','show'],stderr=subprocess.DEVNULL,text=True)
    for ln in r.splitlines():
        p = ln.split()
        if len(p)>=2 and p[1]!='lo': print(p[1]); sys.exit(0)
except Exception: pass
print('eth0')
" 2>/dev/null || echo "eth0")
[ -z "$_HCCL_IF" ] && _HCCL_IF="eth0"
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$_HCCL_IF}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$_HCCL_IF}

# ── HCCL 通信缓冲区 (防止大模型参数广播时缓冲不足) ──────────────
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}

# ── vllm-ascend: 禁用 flash_attn (昇腾 910B 暂不支持) ──────────
export VLLM_ATTENTION_BACKEND=XFORMERS

# ── 昇腾任务队列 & 内存配置 ─────────────────────────────────────
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

# ─────────────────────────────────────────────────────────────────
# 2. 目录与参数配置
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"

# 模型尺寸: 0.5B / 1.5B / 3B
MODEL_SIZE="${MODEL_SIZE:-3B}"
# NPU 卡数 (单机多卡)
N_GPUS="${N_GPUS:-2}"
# GRPO 训练轮次
RL_EPOCHS="${RL_EPOCHS:-3}"
# 每轮全局 batch size (跨所有卡)
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
# 每卡 rollout micro-batch
ROLLOUT_MICRO_BATCH="${ROLLOUT_MICRO_BATCH:-4}"
# 每轮采样数 (GRPO group size)
ROLLOUT_N="${ROLLOUT_N:-8}"
# Prompt 最大长度
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
# Response 最大长度
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
# vllm GPU 显存占比
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
# KL 系数
KL_COEF="${KL_COEF:-0.001}"
# 学习率
LR="${LR:-5e-7}"
# 是否开启参数卸载 (显存不足时开启)
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPT_OFFLOAD="${OPT_OFFLOAD:-False}"
# 实验名
EXP_NAME="${EXP_NAME:-challenger_grpo_${MODEL_SIZE}_${N_GPUS}npu}"

# ─────────────────────────────────────────────────────────────────
# 3. 路径配置
# ─────────────────────────────────────────────────────────────────
MERGED_MODEL_DIR="$BASE_DIR/merged_models_toxicn/challenger_${MODEL_SIZE}"
RL_DATA_DIR="$BASE_DIR/prepared_data/rl"
REWARD_FUNC_PATH="$SCRIPT_DIR/reward_functions/challenger_reward_v7.py"
OUTPUT_DIR="$BASE_DIR/rl_outputs/challenger_${MODEL_SIZE}_grpo"
LOG_DIR="$BASE_DIR/logs/rl_challenger_${MODEL_SIZE}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────
# 4. 预检查
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Challenger GRPO RL 训练 — 昇腾 910B 单机多卡"
echo "============================================================"
echo "  模型尺寸  : $MODEL_SIZE"
echo "  模型路径  : $MERGED_MODEL_DIR"
echo "  NPU 卡数  : $N_GPUS"
echo "  RL 轮次   : $RL_EPOCHS"
echo "  奖励函数  : $REWARD_FUNC_PATH"
echo "  数据目录  : $RL_DATA_DIR"
echo "  输出目录  : $OUTPUT_DIR"
echo "============================================================"

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "❌ 合并模型不存在: $MERGED_MODEL_DIR"
    echo "   请先运行 run_04_merge_lora.sh 完成 LoRA 权重合并"
    exit 1
fi
if [ ! -f "$RL_DATA_DIR/challenger_grpo_train.parquet" ]; then
    echo "❌ Challenger RL 数据不存在: $RL_DATA_DIR/challenger_grpo_train.parquet"
    echo "   请先运行 run_06_prepare_rl_data.sh"
    exit 1
fi
if [ ! -f "$REWARD_FUNC_PATH" ]; then
    echo "❌ 奖励函数不存在: $REWARD_FUNC_PATH"
    exit 1
fi

# 检查 verl 是否安装
python3 -c "import verl" 2>/dev/null || {
    echo "❌ verl 未安装，请参考 verl Ascend 安装指南:"
    echo "   https://ascend.github.io/docs/sources/_generated/sources/verl/ascend_quick_start.html"
    exit 1
}

# ─────────────────────────────────────────────────────────────────
# 5. 根据显存 & 卡数自动调整 micro-batch
# ─────────────────────────────────────────────────────────────────
# 0.5B: 显存较小, micro-batch 可稍大
# 3B:   显存较大, micro-batch 需要缩小
case "$MODEL_SIZE" in
    "0.5B")
        PPO_MINI_BATCH=${PPO_MINI_BATCH:-32}
        PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-8}
        LOG_PROB_MICRO=${LOG_PROB_MICRO:-16}
        TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
        ;;
    "1.5B")
        PPO_MINI_BATCH=${PPO_MINI_BATCH:-16}
        PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
        LOG_PROB_MICRO=${LOG_PROB_MICRO:-8}
        TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
        ;;
    "3B")
        PPO_MINI_BATCH=${PPO_MINI_BATCH:-16}
        PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-2}
        LOG_PROB_MICRO=${LOG_PROB_MICRO:-4}
        TENSOR_PARALLEL=${TENSOR_PARALLEL:-2}
        ;;
    *)
        PPO_MINI_BATCH=${PPO_MINI_BATCH:-16}
        PPO_MICRO_BATCH=${PPO_MICRO_BATCH:-4}
        LOG_PROB_MICRO=${LOG_PROB_MICRO:-8}
        TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
        ;;
esac

# tensor_model_parallel_size 不能超过 N_GPUS
if [ "$TENSOR_PARALLEL" -gt "$N_GPUS" ]; then
    TENSOR_PARALLEL=$N_GPUS
fi

echo "  Tensor 并行: $TENSOR_PARALLEL"
echo "  PPO mini-batch: $PPO_MINI_BATCH"
echo "  PPO micro-batch/GPU: $PPO_MICRO_BATCH"
echo ""

# ─────────────────────────────────────────────────────────────────
# 6. 启动 verl GRPO 训练
#    参考昇腾 verl Quickstart 标准命令格式
#    文档: https://ascend.github.io/docs/sources/_generated/sources/verl/ascend_quick_start.html
# ─────────────────────────────────────────────────────────────────
LOG_FILE="$LOG_DIR/grpo_challenger_$(date +%Y%m%d_%H%M%S).log"
echo "📋 训练日志: $LOG_FILE"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$RL_DATA_DIR/challenger_grpo_train.parquet" \
    data.val_files="$RL_DATA_DIR/challenger_grpo_val.parquet" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="$MERGED_MODEL_DIR" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPT_OFFLOAD \
    \
    actor_rollout_ref.rollout.name=vllm-ascend \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    custom_reward_function.path="$REWARD_FUNC_PATH" \
    custom_reward_function.name=compute_score \
    \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    \
    trainer.device=npu \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='chineseharm_adversarial' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=$RL_EPOCHS \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$OUTPUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Challenger GRPO 训练完成"
echo "   输出目录: $OUTPUT_DIR"
echo "   日志文件: $LOG_FILE"
