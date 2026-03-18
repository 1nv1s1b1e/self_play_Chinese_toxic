#!/bin/bash
# =============================================================================
# Step 3: LoRA SFT训练 - Challenger + Reviewer
# 对 0.5B, 1.5B, 3B 三个尺寸分别训练
# 支持多NPU并行训练 (通过 N_DEVICES 环境变量控制)
# =============================================================================
set -e

# 注意: 不需要设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0
#       LoRA训练脚本已在 Python 内部自行处理

# HCCL 双卡通信配置 (升腾 910B)
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 初始化昇腾 CANN 环境 (如存在)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="/home/ma-user/work/test"
MODELS_BASE="$BASE_DIR/models_base/Qwen"
LORA_DIR="$BASE_DIR/lora_models_toxicn"
DATA_DIR="$BASE_DIR/prepared_data"

# =============================================================================
# 全局固定超参数
# =============================================================================
BATCH_SIZE=4           # 每卡 batch size
NUM_EPOCHS=3           # 训练轮数
DEVICE="npu:0"

# 目标有效 batch size（保持梯度更新次数一致的基准）
# 有效batch = BATCH_SIZE × GRAD_ACCUM × N_DEVICES = TARGET_EFFECTIVE_BATCH
TARGET_EFFECTIVE_BATCH=32

# NPU数量: 四卡昇腾910B
N_DEVICES=${N_DEVICES:-4}

# 根据卡数动态计算 grad_accum，使有效 batch 始终 = TARGET_EFFECTIVE_BATCH
# 梯度更新次数 = ceil(8160 / TARGET_EFFECTIVE_BATCH) × NUM_EPOCHS = 765 次（与基准一致）
GRAD_ACCUM=$(( TARGET_EFFECTIVE_BATCH / BATCH_SIZE / N_DEVICES ))
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

# =============================================================================
# 各模型尺寸专属参数说明
# rank/alpha/LR 按模型尺寸独立设置，不受卡数影响
# 原则：rank/hidden_dim ≈ 0.01-0.02；alpha = 2×rank；LR 随模型增大而保守
#
#  尺寸  hidden_dim  rank  alpha    LR
#  0.5B     896       16    32    3e-4
#  1.5B    1536       16    32    3e-4
#  3B      2048       32    64    2e-4
#  7B      3584       32    64    2e-4
#  14B     5120       64   128    1e-4
# =============================================================================

# 是否强制重新训练 (设为1时忽略已完成的模型)
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

# 仅重训Reviewer (设为1时跳过Challenger并强制训练Reviewer)
RETRAIN_REVIEWER_ONLY=${RETRAIN_REVIEWER_ONLY:-0}

# 模型尺寸列表（可按需修改）
MODEL_SIZES=("0.5B" "1.5B" "3B" "7B" "14B")

# 根据NPU数量选择启动方式
# 用 python -m torch.distributed.run 而非 torchrun，保证使用同一Python解释器
if [ "$N_DEVICES" -ge 2 ]; then
    LAUNCH_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$N_DEVICES"
    echo "🚀 多卡模式: 使用 $N_DEVICES 个 NPU (DDP + HCCL)"
else
    LAUNCH_CMD="python"
    echo "🔹 单卡模式: $DEVICE"
fi

if [ "$RETRAIN_REVIEWER_ONLY" -eq 1 ]; then
    echo "⚠ 仅重训Reviewer：Challenger将被跳过"
fi

echo "============================================================"
echo "Step 3: LoRA SFT 训练"
echo "============================================================"
echo "模型尺寸: ${MODEL_SIZES[*]}"
echo "NPU数量: $N_DEVICES | 有效batch: $TARGET_EFFECTIVE_BATCH | grad_accum: $GRAD_ACCUM | epochs: $NUM_EPOCHS"
echo ""

for SIZE in "${MODEL_SIZES[@]}"; do
    MODEL_PATH="$MODELS_BASE/Qwen2.5-${SIZE}-Instruct"

    # ------------------------------------------------------------------
    # 按模型尺寸选择专属 LoRA 参数
    # ------------------------------------------------------------------
    case "$SIZE" in
        "0.5B"|"1.5B")
            LORA_RANK=16; LORA_ALPHA=32; LR=3e-4 ;;
        "3B"|"7B")
            LORA_RANK=32; LORA_ALPHA=64; LR=2e-4 ;;
        "14B")
            LORA_RANK=64; LORA_ALPHA=128; LR=1e-4 ;;
        *)
            echo "⚠️  未知尺寸 $SIZE，使用默认参数 rank=32 alpha=64 lr=2e-4"
            LORA_RANK=32; LORA_ALPHA=64; LR=2e-4 ;;
    esac

    echo ">>> [$SIZE] rank=$LORA_RANK  alpha=$LORA_ALPHA  lr=$LR  grad_accum=$GRAD_ACCUM  有效batch=$(( BATCH_SIZE * GRAD_ACCUM * N_DEVICES ))"
    echo ""

    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  模型不存在: $MODEL_PATH, 跳过..."
        continue
    fi
    
    echo "============================================================"
    echo "[${SIZE}] Challenger LoRA训练"
    echo "============================================================"
    
    if [ "$RETRAIN_REVIEWER_ONLY" -eq 1 ]; then
        echo "⏭️  跳过Challenger ${SIZE} (RETRAIN_REVIEWER_ONLY=1)"
    else
        CHALLENGER_OUT="$LORA_DIR/challenger_${SIZE}"
        if [ "$FORCE_RETRAIN" -eq 0 ] && [ -f "$CHALLENGER_OUT/adapter_config.json" ]; then
            echo "✅ Challenger ${SIZE} 已完成训练，跳过 (设置 FORCE_RETRAIN=1 可强制重训)"
        else
            $LAUNCH_CMD ../model_lora/train_challenger_lora.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_DIR/challenger_sft/train.jsonl" \
        --val_data_path "$DATA_DIR/challenger_sft/val.jsonl" \
        --output_dir "$LORA_DIR/challenger_${SIZE}" \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --max_length 1024 \
        --seed 42 \
        --device "$DEVICE" \
        --n_devices $N_DEVICES
            
            echo ""
            echo "✓ Challenger ${SIZE} LoRA训练完成!"
        fi
    fi
    echo ""
    
    echo "============================================================"
    echo "[${SIZE}] Reviewer LoRA训练"
    echo "============================================================"
    
    REVIEWER_OUT="$LORA_DIR/reviewer_${SIZE}"
    if [ "$RETRAIN_REVIEWER_ONLY" -eq 1 ]; then
        FORCE_RETRAIN=1
    fi
    if [ "$FORCE_RETRAIN" -eq 0 ] && [ -f "$REVIEWER_OUT/adapter_config.json" ]; then
        echo "✅ Reviewer ${SIZE} 已完成训练，跳过 (设置 FORCE_RETRAIN=1 可强制重训)"
    else
        $LAUNCH_CMD ../model_lora/train_reviewer_lora.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_DIR/reviewer_sft/train.jsonl" \
        --val_data_path "$DATA_DIR/reviewer_sft/val.jsonl" \
        --output_dir "$LORA_DIR/reviewer_${SIZE}" \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --max_length 2048 \
        --seed 42 \
        --device "$DEVICE" \
        --n_devices $N_DEVICES
        
        echo ""
        echo "✓ Reviewer ${SIZE} LoRA训练完成!"
    fi
    echo ""
done

echo "============================================================"
echo "✓ 全部LoRA SFT训练完成!"
echo "============================================================"
echo ""
echo "LoRA模型保存在:"
ls -la "$LORA_DIR/" 2>/dev/null || echo "目录不存在"
