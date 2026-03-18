#!/bin/bash
# =============================================================================
# Step 5: 评测 - 在test集上评测 base模型 和 LoRA(merged)模型
# 支持vLLM评测 和 NPU批量评测两种模式
# 新增: Challenger生成质量评测
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="/home/ma-user/work/test"
MODELS_BASE="$BASE_DIR/models_base/Qwen"
MERGED_DIR="$BASE_DIR/merged_models_toxicn"
TEST_DATA="$BASE_DIR/prepared_data/rl/test_eval.json"
EVAL_DIR="$BASE_DIR/eval_results"

# 评测配置
EVAL_MODE="vllm"          # vllm 或 npu
TENSOR_PARALLEL=1          # vLLM并行数 (vllm模式)
NUM_NPUS=1                 # NPU数量 (npu模式)
BATCH_SIZE=8               # 批量大小 (npu模式)

# Challenger 生成评测配置
CHALLENGER_EVAL=${CHALLENGER_EVAL:-1}   # 是否评测Challenger生成质量 (1=是)
CHALLENGER_SAMPLES=3                     # 每(类别,表达方式)组合生成样本数
CHALLENGER_DEVICE="npu:0"

MODEL_SIZES=("0.5B" "1.5B" "3B")

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)       EVAL_MODE="$2";       shift 2 ;;
        --tp)         TENSOR_PARALLEL="$2"; shift 2 ;;
        --num_npus)   NUM_NPUS="$2";        shift 2 ;;
        --batch)      BATCH_SIZE="$2";      shift 2 ;;
        --no_challenger) CHALLENGER_EVAL=0; shift ;;
        --test_data)  TEST_DATA="$2";       shift 2 ;;
        *)            echo "未知参数: $1"; exit 1 ;;
    esac
done

mkdir -p "$EVAL_DIR"

echo "============================================================"
echo "Step 5: 模型评测"
echo "============================================================"
echo "评测模式:   $EVAL_MODE"
echo "测试数据:   $TEST_DATA"
echo "结果输出:   $EVAL_DIR"
echo "模型尺寸:   ${MODEL_SIZES[*]}"
echo "Challenger: $([ $CHALLENGER_EVAL -eq 1 ] && echo '评测' || echo '跳过')"
echo ""

run_reviewer_eval() {
    local model_path=$1
    local tag=$2
    
    echo "------------------------------------------------------------"
    echo "[Reviewer] 评测: $tag"
    echo "  模型: $model_path"
    echo "------------------------------------------------------------"
    
    if [ "$EVAL_MODE" = "vllm" ]; then
        python ../model_eval/baseline_eval_vllm.py \
            --model_path "$model_path" \
            --data_path "$TEST_DATA" \
            --output_dir "$EVAL_DIR" \
            --tensor_parallel_size $TENSOR_PARALLEL \
            --tag "$tag"
    else
        python ../model_eval/batch_eval_npu.py \
            --model_path "$model_path" \
            --data_path "$TEST_DATA" \
            --output_dir "$EVAL_DIR" \
            --num_npus $NUM_NPUS \
            --batch_size $BATCH_SIZE \
            --tag "$tag"
    fi
    
    echo ""
    echo "✓ $tag 评测完成!"
    echo ""
}

run_challenger_eval() {
    local model_path=$1
    local tag=$2
    
    echo "------------------------------------------------------------"
    echo "[Challenger] 生成质量评测: $tag"
    echo "  模型: $model_path"
    echo "------------------------------------------------------------"
    
    python ../model_eval/eval_challenger_generation.py \
        --model_path "$model_path" \
        --output_file "$EVAL_DIR/challenger_gen_${tag}.jsonl" \
        --num_samples $CHALLENGER_SAMPLES \
        --device "$CHALLENGER_DEVICE"
    
    echo ""
    echo "✓ Challenger $tag 生成评测完成!"
    echo ""
}

# =============================================================================
# 5.1 评测基础模型 (Base Reviewer)
# =============================================================================
echo ""
echo "======================== 基础模型评测 ========================"
echo ""

for SIZE in "${MODEL_SIZES[@]}"; do
    MODEL_PATH="$MODELS_BASE/Qwen2.5-${SIZE}-Instruct"
    
    if [ ! -d "$MODEL_PATH" ]; then
        echo "⚠️  模型不存在: $MODEL_PATH, 跳过..."
        continue
    fi
    
    run_reviewer_eval "$MODEL_PATH" "base_${SIZE}"
done

# =============================================================================
# 5.2 评测LoRA合并模型 (Reviewer - 分类评测)
# =============================================================================
echo ""
echo "====================== Reviewer LoRA评测 ======================"
echo ""

for SIZE in "${MODEL_SIZES[@]}"; do
    MERGED_PATH="$MERGED_DIR/reviewer_${SIZE}"
    
    if [ ! -d "$MERGED_PATH" ]; then
        echo "⚠️  合并模型不存在: $MERGED_PATH, 跳过..."
        continue
    fi
    
    run_reviewer_eval "$MERGED_PATH" "reviewer_lora_${SIZE}"
done

# =============================================================================
# 5.3 评测Challenger生成质量 (可选)
# =============================================================================
if [ "$CHALLENGER_EVAL" -eq 1 ]; then
    echo ""
    echo "==================== Challenger 生成质量评测 ===================="
    echo ""
    
    for SIZE in "${MODEL_SIZES[@]}"; do
        MERGED_PATH="$MERGED_DIR/challenger_${SIZE}"
        
        if [ ! -d "$MERGED_PATH" ]; then
            echo "⚠️  合并模型不存在: $MERGED_PATH, 跳过..."
            continue
        fi
        
        run_challenger_eval "$MERGED_PATH" "$SIZE"
    done
fi

# =============================================================================
# 汇总结果
# =============================================================================
echo ""
echo "============================================================"
echo "✓ 全部评测完成!"
echo "============================================================"
echo ""
echo "评测结果文件:"
ls -la "$EVAL_DIR/"*.json "$EVAL_DIR/"*.jsonl 2>/dev/null || echo "  无结果文件"
echo ""

# 汇总评测结果 (如果有多个结果)
RESULT_COUNT=$(find "$EVAL_DIR" -name "eval_*.json" 2>/dev/null | wc -l)
if [ "$RESULT_COUNT" -gt 0 ]; then
    echo "汇总评测结果..."
    python ../model_eval/summarize_results.py \
        --eval_dir "$EVAL_DIR" \
        --export_csv "$EVAL_DIR/sft_comparison.csv" 2>/dev/null || true
fi
