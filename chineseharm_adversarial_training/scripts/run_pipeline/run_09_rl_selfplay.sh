#!/bin/bash
# =============================================================================
# Step 9: 对抗博弈自对弈 RL 训练 (昇腾 910B 单机多卡)
# =============================================================================
# 完整自对弈管线: Challenger ↔ Reviewer 交替 GRPO 训练
#
# 前置: run_04_merge_lora.sh, run_06_prepare_rl_data.sh
#
# 用法:
#   bash run_09_rl_selfplay.sh
#   N_GPUS=4 MODEL_SIZE=3B SELFPLAY_ROUNDS=5 bash run_09_rl_selfplay.sh
#   bash run_09_rl_selfplay.sh --model_size 1.5B --n_gpus 4 --rounds 3
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-2}"
SELFPLAY_ROUNDS="${SELFPLAY_ROUNDS:-3}"
EPOCHS_PER_PHASE="${EPOCHS_PER_PHASE:-2}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --base_dir)    BASE_DIR="$2";         shift 2 ;;
        --model_size)  MODEL_SIZE="$2";       shift 2 ;;
        --n_gpus)      N_GPUS="$2";           shift 2 ;;
        --rounds)      SELFPLAY_ROUNDS="$2";  shift 2 ;;
        --epochs)      EPOCHS_PER_PHASE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "Step 9: 对抗博弈自对弈 RL 训练"
echo "============================================================"
echo "  模型大小    : $MODEL_SIZE"
echo "  NPU 卡数    : $N_GPUS"
echo "  自对弈轮次  : $SELFPLAY_ROUNDS"
echo "  每Phase轮次 : $EPOCHS_PER_PHASE"
echo ""

BASE_DIR="$BASE_DIR" \
MODEL_SIZE="$MODEL_SIZE" \
N_GPUS="$N_GPUS" \
SELFPLAY_ROUNDS="$SELFPLAY_ROUNDS" \
EPOCHS_PER_PHASE="$EPOCHS_PER_PHASE" \
bash ../rl_train/run_selfplay_npu.sh

echo ""
echo "✓ Step 9 完成: 对抗博弈自对弈 RL 训练"
