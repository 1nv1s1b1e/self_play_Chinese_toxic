#!/bin/bash
# =============================================================================
# Step 4: 合并LoRA到基础模型
# 为后续vLLM推理做准备
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODELS_BASE="$BASE_DIR/models_base/Qwen"
LORA_DIR="$BASE_DIR/lora_models_toxicn"
MERGED_DIR="$BASE_DIR/merged_models_toxicn"

MODEL_SIZES=("0.5B" "1.5B" "3B")
AGENTS=("challenger" "reviewer")

echo "============================================================"
echo "Step 4: 合并LoRA权重到基础模型"
echo "============================================================"
echo ""

for SIZE in "${MODEL_SIZES[@]}"; do
    MODEL_PATH="$MODELS_BASE/Qwen2.5-${SIZE}-Instruct"
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  基础模型不存在: $MODEL_PATH, 跳过..."
        continue
    fi
    
    for AGENT in "${AGENTS[@]}"; do
        LORA_PATH="$LORA_DIR/${AGENT}_${SIZE}"
        OUTPUT_PATH="$MERGED_DIR/${AGENT}_${SIZE}"
        
        if [ ! -d "$LORA_PATH" ]; then
            echo "⚠️  LoRA不存在: $LORA_PATH, 跳过..."
            continue
        fi
        
        echo "------------------------------------------------------------"
        echo "[${SIZE}] 合并 ${AGENT} LoRA"
        echo "  基础模型: $MODEL_PATH"
        echo "  LoRA:    $LORA_PATH"
        echo "  输出:    $OUTPUT_PATH"
        echo "------------------------------------------------------------"
        
        python ../model_lora/merge_lora.py \
            --base_model "$MODEL_PATH" \
            --lora_path "$LORA_PATH" \
            --output_path "$OUTPUT_PATH"
        
        echo ""
        echo "✓ ${AGENT} ${SIZE} 合并完成!"
        echo ""
    done
done

echo "============================================================"
echo "✓ 全部LoRA合并完成!"
echo "============================================================"
echo ""
echo "合并模型保存在:"
ls -la "$MERGED_DIR/" 2>/dev/null || echo "目录不存在"
