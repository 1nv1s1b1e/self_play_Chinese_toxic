#!/bin/bash
# =============================================================================
# 消融实验一键运行脚本
# 
# 实验内容:
#   1. Prompt消融: 4种prompt变体 × 2个模型(base + sft)
#   2. Epoch消融:  训练→保存各epoch→逐个评测
#   3. Base vs SFT多维对比: 格式对齐率/准确率/输出长度/混淆矩阵
#   4. 结果分析与可视化
#
# 使用方式:
#   bash run_ablation.sh                    # 全部实验
#   bash run_ablation.sh --only prompt      # 只跑prompt消融
#   bash run_ablation.sh --only epoch       # 只跑epoch消融
#   bash run_ablation.sh --only compare     # 只跑base vs sft
#   bash run_ablation.sh --only analyze     # 只跑结果分析
#   bash run_ablation.sh --model_size 1.5B  # 指定模型规模
# =============================================================================
set -e

# 关键: 必须在Python进程启动前设置，防止PyTorch autoload与torch_npu双重注册NPU后端
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="/home/ma-user/work/test"
MODELS_BASE="$BASE_DIR/models_base/Qwen"
MERGED_DIR="$BASE_DIR/merged_models"
SPLIT_DIR="$BASE_DIR/split_data"
RESULTS_DIR="$BASE_DIR/ablation_results"
SFT_DATA="$BASE_DIR/prepared_data/reviewer_sft/train.jsonl"

# 默认配置
MODEL_SIZE="3B"
EVAL_MODE="npu"     # npu 或 vllm
TP_SIZE=1
BATCH_SIZE=8
DEVICE="npu:0"
ONLY_EXP=""         # prompt / epoch / compare / analyze / 空=全部

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)  MODEL_SIZE="$2";  shift 2 ;;
        --mode)        EVAL_MODE="$2";   shift 2 ;;
        --tp)          TP_SIZE="$2";     shift 2 ;;
        --batch)       BATCH_SIZE="$2";  shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --only)        ONLY_EXP="$2";    shift 2 ;;
        *)             echo "未知参数: $1"; exit 1 ;;
    esac
done

BASE_MODEL="$MODELS_BASE/Qwen2.5-${MODEL_SIZE}-Instruct"
SFT_MODEL="$MERGED_DIR/reviewer_${MODEL_SIZE}"
TEST_DATA="$SPLIT_DIR/test.parquet"

should_run() {
    [ -z "$ONLY_EXP" ] || [ "$ONLY_EXP" = "$1" ]
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            消融实验一键运行                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "模型规模:    $MODEL_SIZE"
echo "Base模型:    $BASE_MODEL"
echo "SFT模型:     $SFT_MODEL"
echo "评测模式:    $EVAL_MODE"
echo "测试数据:    $TEST_DATA"
echo "结果目录:    $RESULTS_DIR"
echo ""

# 前置检查
if [ ! -d "$BASE_MODEL" ]; then
    echo "⚠️  Base模型不存在: $BASE_MODEL"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ============================================================
# 实验1: Prompt消融
# ============================================================
if should_run "prompt"; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "实验1: Prompt消融"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 1a. Base模型 × 4种prompt
    echo ""
    echo ">>> Base模型 prompt消融..."
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python prompt_ablation_eval.py \
        --model_path "$BASE_MODEL" \
        --data_path "$TEST_DATA" \
        --output_dir "$RESULTS_DIR/prompt_ablation" \
        --mode "$EVAL_MODE" \
        --tp "$TP_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --tag "base"
    
    # 1b. SFT模型 × 4种prompt
    if [ -d "$SFT_MODEL" ]; then
        echo ""
        echo ">>> SFT模型 prompt消融..."
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python prompt_ablation_eval.py \
            --model_path "$SFT_MODEL" \
            --data_path "$TEST_DATA" \
            --output_dir "$RESULTS_DIR/prompt_ablation" \
            --mode "$EVAL_MODE" \
            --tp "$TP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --tag "sft"
    else
        echo "⚠️  SFT模型不存在: $SFT_MODEL, 跳过SFT prompt消融"
    fi
    
    echo ""
    echo "✓ Prompt消融完成"
fi

# ============================================================
# 实验2: Epoch消融
# ============================================================
if should_run "epoch"; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "实验2: Epoch消融"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    EPOCH_CKPT_DIR="$RESULTS_DIR/epoch_ablation/checkpoints"
    
    # 2a. 训练 (每epoch保存checkpoint)
    echo ""
    echo ">>> 训练 (每epoch保存checkpoint)..."
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python epoch_ablation.py train \
        --model_path "$BASE_MODEL" \
        --data_path "$SFT_DATA" \
        --output_dir "$EPOCH_CKPT_DIR" \
        --max_epochs 3 \
        --device "$DEVICE"
    
    # 2b. 评测各epoch
    echo ""
    echo ">>> 评测各epoch..."
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python epoch_ablation.py eval \
        --model_path "$BASE_MODEL" \
        --ckpt_dir "$EPOCH_CKPT_DIR" \
        --data_path "$TEST_DATA" \
        --output_dir "$RESULTS_DIR/epoch_ablation" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE"
    
    echo ""
    echo "✓ Epoch消融完成"
fi

# ============================================================
# 实验3: Base vs SFT 多维对比
# ============================================================
if should_run "compare"; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "实验3: Base vs SFT 多维对比"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ -d "$SFT_MODEL" ]; then
        TORCH_DEVICE_BACKEND_AUTOLOAD=0 python base_vs_sft_comparison.py \
            --base_model "$BASE_MODEL" \
            --sft_model "$SFT_MODEL" \
            --data_path "$TEST_DATA" \
            --output_dir "$RESULTS_DIR/base_vs_sft" \
            --mode "$EVAL_MODE" \
            --tp "$TP_SIZE" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE"
        
        echo ""
        echo "✓ Base vs SFT对比完成"
    else
        echo "⚠️  SFT模型不存在: $SFT_MODEL, 跳过对比实验"
    fi
fi

# ============================================================
# 结果分析
# ============================================================
if should_run "analyze"; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "结果分析与可视化"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python analyze_ablation_results.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$RESULTS_DIR/analysis"
    
    echo ""
    echo "✓ 分析完成"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                消融实验全部完成!                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "结果目录: $RESULTS_DIR/"
echo "  prompt_ablation/  — Prompt变体消融结果"
echo "  epoch_ablation/   — Epoch消融结果"
echo "  base_vs_sft/      — Base vs SFT对比结果"
echo "  analysis/         — 综合分析与可视化"
echo ""
echo "诊断工具 (逐样本检查模型真实输出):"
echo "  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python inspect_outputs.py \\"
echo "    --model_path $BASE_MODEL --data_path $TEST_DATA --tag base --num_samples 30"
echo "  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python inspect_outputs.py \\"
echo "    --model_path $SFT_MODEL --data_path $TEST_DATA --tag sft --num_samples 30"
