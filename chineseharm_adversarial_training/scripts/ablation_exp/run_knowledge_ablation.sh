#!/bin/bash
# =============================================================================
# 消融实验: 有知识 SFT vs 无知识 SFT
#
# 对比:
#   A. 有知识 SFT (当前版本，prompt含 ~2000字 RULES 关键词映射)
#   B. 无知识 SFT (本实验，prompt不含 RULES，只有任务指令)
#
# 流程:
#   1. 生成无知识版 SFT 数据
#   2. 训练无知识版 LoRA
#   3. 合并 LoRA
#   4. 评测 (有知识prompt + 无知识prompt 两种)
#   5. 与有知识版结果对比
#
# 用法:
#   bash run_knowledge_ablation.sh                          # 完整流程
#   bash run_knowledge_ablation.sh --skip_train             # 跳过训练 (已训练过)
#   bash run_knowledge_ablation.sh --model_size 1.5B        # 指定模型
# =============================================================================
set -e


export TORCH_DEVICE_BACKEND_AUTOLOAD=0

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
SPLIT_DIR="$BASE_DIR/split_data"

MODEL_SIZE="3B"
DEVICE="npu:0"
BATCH_SIZE=4
GRAD_ACCUM=4
NUM_EPOCHS=3
LR=2e-4
LORA_RANK=32
LORA_ALPHA=64
SKIP_TRAIN=false
N_DEVICES=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)  MODEL_SIZE="$2";  shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --n_devices)   N_DEVICES="$2";   shift 2 ;;
        --skip_train)  SKIP_TRAIN=true;  shift ;;
        *)             echo "未知参数: $1"; exit 1 ;;
    esac
done

BASE_MODEL="$MODELS_BASE/Qwen2.5-${MODEL_SIZE}-Instruct"

# 无知识版路径
NK_DATA_DIR="$BASE_DIR/prepared_data/reviewer_sft_no_knowledge"
NK_LORA_DIR="$BASE_DIR/lora_models/reviewer_${MODEL_SIZE}_no_knowledge"
NK_MERGED_DIR="$BASE_DIR/merged_models/reviewer_${MODEL_SIZE}_no_knowledge"

# 有知识版路径 (已有)
K_MERGED_DIR="$BASE_DIR/merged_models/reviewer_${MODEL_SIZE}"

TEST_DATA="$SPLIT_DIR/test.parquet"
RESULTS_DIR="$BASE_DIR/ablation_results/knowledge_ablation"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        消融实验: 有知识 SFT vs 无知识 SFT                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "模型:         $MODEL_SIZE"
echo "Base模型:     $BASE_MODEL"
echo "有知识SFT:    $K_MERGED_DIR"
echo "无知识SFT:    $NK_MERGED_DIR"
echo ""

if [ ! -d "$BASE_MODEL" ]; then
    echo "⚠️  Base模型不存在: $BASE_MODEL"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# 根据NPU数量选择启动方式
# 用 python -m torch.distributed.run 而非 torchrun，保证使用同一Python解释器
if [ "$N_DEVICES" -ge 2 ]; then
    TRAIN_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$N_DEVICES"
    echo "🚀 多卡训练模式: $N_DEVICES NPUs (DDP + HCCL)"
else
    TRAIN_CMD="python"
    echo "🔹 单卡训练模式: $DEVICE"
fi

# ============================================================
# Step 1: 生成无知识版 SFT 数据
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: 生成无知识版 SFT 数据"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python prepare_no_knowledge_sft.py \
    --split_dir "$SPLIT_DIR" \
    --output_dir "$NK_DATA_DIR"

echo "✓ 数据准备完成"

# ============================================================
# Step 2: 训练无知识版 LoRA
# ============================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 2: 训练无知识版 LoRA"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    $TRAIN_CMD ../model_lora/train_reviewer_lora.py \
        --model_path "$BASE_MODEL" \
        --data_path "$NK_DATA_DIR/train.jsonl" \
        --output_dir "$NK_LORA_DIR" \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --device "$DEVICE" \
        --n_devices $N_DEVICES

    echo "✓ 训练完成"

    # ============================================================
    # Step 3: 合并 LoRA
    # ============================================================
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 3: 合并 LoRA"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python ../model_lora/merge_lora.py \
        --base_model "$BASE_MODEL" \
        --lora_path "$NK_LORA_DIR" \
        --output_path "$NK_MERGED_DIR"

    echo "✓ 合并完成"
else
    echo ""
    echo "[跳过训练和合并] 使用已有模型: $NK_MERGED_DIR"
fi

# ============================================================
# Step 4: 评测对比
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: 评测对比"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 4a. 用 inspect_outputs.py 检查无知识版模型输出
echo ""
echo ">>> 无知识版 SFT 模型输出检查..."
if [ -d "$NK_MERGED_DIR" ]; then
    python inspect_outputs.py \
        --model_path "$NK_MERGED_DIR" \
        --data_path "$TEST_DATA" \
        --tag "sft_no_knowledge" \
        --num_samples 50 \
        --device "$DEVICE" \
        --output_dir "$RESULTS_DIR"
else
    echo "⚠️  无知识版模型不存在: $NK_MERGED_DIR"
fi

# 4b. 用 prompt_ablation 同时评测有知识版和无知识版
echo ""
echo ">>> Prompt 消融评测 (无知识版模型)..."
if [ -d "$NK_MERGED_DIR" ]; then
    python prompt_ablation_eval.py \
        --model_path "$NK_MERGED_DIR" \
        --data_path "$TEST_DATA" \
        --output_dir "$RESULTS_DIR" \
        --mode npu \
        --batch_size 8 \
        --device "$DEVICE" \
        --tag "sft_no_knowledge"
fi

echo ""
echo ">>> Prompt 消融评测 (有知识版模型)..."
if [ -d "$K_MERGED_DIR" ]; then
    python prompt_ablation_eval.py \
        --model_path "$K_MERGED_DIR" \
        --data_path "$TEST_DATA" \
        --output_dir "$RESULTS_DIR" \
        --mode npu \
        --batch_size 8 \
        --device "$DEVICE" \
        --tag "sft_with_knowledge"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║             知识消融实验完成!                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "结果目录: $RESULTS_DIR/"
echo ""
echo "对比维度:"
echo "  1. 有知识训练 + 有知识推理  vs  无知识训练 + 有知识推理"
echo "     → 训练时RULES是否必要?"
echo "  2. 有知识训练 + 无知识推理  vs  无知识训练 + 无知识推理"
echo "     → 纯语义分类能力对比"
echo "  3. 无知识训练 + 有知识推理  vs  无知识训练 + 无知识推理"
echo "     → 推理时RULES对无知识模型的增益"
