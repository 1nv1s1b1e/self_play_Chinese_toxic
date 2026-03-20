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

# 训练超参数
BATCH_SIZE=4
GRAD_ACCUM=4
NUM_EPOCHS=3
LR=2e-4
LORA_RANK=32
LORA_ALPHA=64
DEVICE="npu:0"

# NPU数量: 双卡升腾910B
N_DEVICES=${N_DEVICES:-2}

# 是否强制重新训练 (设为1时忽略已完成的模型)
FORCE_RETRAIN=${FORCE_RETRAIN:-0}

# 仅重训Reviewer (设为1时跳过Challenger并强制训练Reviewer)
RETRAIN_REVIEWER_ONLY=${RETRAIN_REVIEWER_ONLY:-0}

# 模型尺寸列表（可修改）
MODEL_SIZES=("0.5B" "1.5B" "3B")

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
echo "训练配置: batch=$BATCH_SIZE, grad_accum=$GRAD_ACCUM, epochs=$NUM_EPOCHS"
echo "LoRA配置: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "NPU数量: $N_DEVICES"
echo ""

for SIZE in "${MODEL_SIZES[@]}"; do
    MODEL_PATH="$MODELS_BASE/Qwen2.5-${SIZE}-Instruct"
    
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
