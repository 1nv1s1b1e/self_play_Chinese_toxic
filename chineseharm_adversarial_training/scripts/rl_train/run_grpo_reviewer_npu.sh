#!/bin/bash
# =============================================================================
# Reviewer GRPO RL 训练 — 昇腾 910B 单机多卡
# =============================================================================
# 算法 : GRPO (Group Relative Policy Optimization)
# 框架 : verl  (https://github.com/volcengine/verl)
# 硬件 : 华为昇腾 910B NPU，单机 N 卡 (默认 2 卡，支持 4/8 卡)
# 通信 : HCCL
# 策略 : FSDP
# 推理 : vllm-ascend
#
# Reviewer 任务: 对文本进行有害性检测 + 毒性类别分类
# 奖励函数: reviewer_reward.py (精准率 + 召回率)
#
# 用法:
#   N_GPUS=2 bash run_grpo_reviewer_npu.sh
#   N_GPUS=4 MODEL_SIZE=3B bash run_grpo_reviewer_npu.sh
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
# ─────────────────────────────────────────────────────────────────
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29601}    # 与 Challenger 错开端口
# ── 与 LoRA 脚本保持一致: 禁止 torch 在 Ray Worker 中自动加载 NPU backend ──────
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# ── 关键: 防止 Ray 在启动 Worker 时覆盖 Ascend 设备可见性 ──────
export RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1

# ── Ascend RT 设备可见性 (按实际 N_GPUS 动态生成) ───────────────
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

export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export VLLM_ATTENTION_BACKEND=XFORMERS
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

# ─────────────────────────────────────────────────────────────────
# 2. 目录与参数配置
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"

MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-2}"
RL_EPOCHS="${RL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
KL_COEF="${KL_COEF:-0.001}"
LR="${LR:-5e-7}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-False}"
OPT_OFFLOAD="${OPT_OFFLOAD:-False}"
EXP_NAME="${EXP_NAME:-reviewer_grpo_${MODEL_SIZE}_${N_GPUS}npu}"

# ─────────────────────────────────────────────────────────────────
# 3. 路径配置
# ─────────────────────────────────────────────────────────────────
MERGED_MODEL_DIR="$BASE_DIR/merged_models_toxicn/reviewer_${MODEL_SIZE}"
RL_DATA_DIR="$BASE_DIR/prepared_data/rl"
REWARD_FUNC_PATH="$SCRIPT_DIR/reward_functions/reviewer_reward.py"
OUTPUT_DIR="$BASE_DIR/rl_outputs/reviewer_${MODEL_SIZE}_grpo"
LOG_DIR="$BASE_DIR/logs/rl_reviewer_${MODEL_SIZE}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─────────────────────────────────────────────────────────────────
# 4. 预检查
# ─────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Reviewer GRPO RL 训练 — 昇腾 910B 单机多卡"
echo "============================================================"
echo "  模型尺寸  : $MODEL_SIZE"
echo "  模型路径  : $MERGED_MODEL_DIR"
echo "  NPU 卡数  : $N_GPUS"
echo "  RL 轮次   : $RL_EPOCHS"
echo "  奖励函数  : $REWARD_FUNC_PATH"
echo "  输出目录  : $OUTPUT_DIR"
echo "============================================================"

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "❌ 合并模型不存在: $MERGED_MODEL_DIR"
    echo "   请先运行 run_04_merge_lora.sh"
    exit 1
fi
if [ ! -f "$RL_DATA_DIR/reviewer_grpo_train.parquet" ]; then
    echo "❌ Reviewer RL 数据不存在: $RL_DATA_DIR/reviewer_grpo_train.parquet"
    exit 1
fi
if [ ! -f "$REWARD_FUNC_PATH" ]; then
    echo "❌ 奖励函数不存在: $REWARD_FUNC_PATH"
    exit 1
fi

python3 -c "import verl" 2>/dev/null || {
    echo "❌ verl 未安装"; exit 1
}

# ─────────────────────────────────────────────────────────────────
# 5. 根据模型大小自动调整 micro-batch
# ─────────────────────────────────────────────────────────────────
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

if [ "$TENSOR_PARALLEL" -gt "$N_GPUS" ]; then
    TENSOR_PARALLEL=$N_GPUS
fi

echo "  Tensor 并行: $TENSOR_PARALLEL"
echo "  PPO mini-batch: $PPO_MINI_BATCH"
echo ""

# ─────────────────────────────────────────────────────────────────
# 6. 启动 verl GRPO 训练 (Reviewer)
# ─────────────────────────────────────────────────────────────────
LOG_FILE="$LOG_DIR/grpo_reviewer_$(date +%Y%m%d_%H%M%S).log"
echo "📋 训练日志: $LOG_FILE"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$RL_DATA_DIR/reviewer_grpo_train.parquet" \
    data.val_files="$RL_DATA_DIR/reviewer_grpo_val.parquet" \
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
echo "✅ Reviewer GRPO 训练完成"
echo "   输出目录: $OUTPUT_DIR"
echo "   日志文件: $LOG_FILE"
