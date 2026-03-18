#!/bin/bash
# =============================================================================
# Step 8: Reviewer GRPO RL 训练 (昇腾 910B 单机多卡)
# =============================================================================
# 前置: run_04_merge_lora.sh, run_06_prepare_rl_data.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-2}"
RL_EPOCHS="${RL_EPOCHS:-3}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_dir)   BASE_DIR="$2";   shift 2 ;;
        --model_size) MODEL_SIZE="$2"; shift 2 ;;
        --n_gpus)     N_GPUS="$2";     shift 2 ;;
        --epochs)     RL_EPOCHS="$2";  shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Step 8: Reviewer GRPO RL 训练"
echo "============================================================"
echo "  模型大小: $MODEL_SIZE"
echo "  NPU 卡数: $N_GPUS"
echo "  RL 轮次 : $RL_EPOCHS"
echo ""

BASE_DIR="$BASE_DIR" \
MODEL_SIZE="$MODEL_SIZE" \
N_GPUS="$N_GPUS" \
RL_EPOCHS="$RL_EPOCHS" \
bash ../rl_train/run_grpo_reviewer_npu.sh

echo ""
echo "✓ Step 8 完成: Reviewer GRPO RL 训练"
