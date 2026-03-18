#!/bin/bash
# =============================================================================
# 全流程一键运行脚本 (SFT → Merge → Baseline评估 → SFT评估 → Self-Play)
# =============================================================================
# 用法:
#   cd /home/ma-user/work/test/chineseharm_adversarial_training
#   bash scripts/run_pipeline/run_full_pipeline.sh
#
# 环境变量 (可覆盖):
#   N_DEVICES=4        NPU 卡数 (SFT阶段)
#   N_GPUS=4           NPU 卡数 (Self-Play阶段)
#   MODEL_SIZE=3B      模型尺寸
#   SKIP_SFT=0         跳过SFT (已训练完毕时设为1)
#   SKIP_MERGE=0       跳过合并
#   SKIP_EVAL=0        跳过评估
#   SKIP_SELFPLAY=0    跳过Self-Play
# =============================================================================
set -e

# ─────────────────────────────────────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="/home/ma-user/work/test"
MODEL_SIZE="${MODEL_SIZE:-3B}"
N_DEVICES="${N_DEVICES:-4}"
N_GPUS="${N_GPUS:-4}"

SKIP_SFT="${SKIP_SFT:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_SELFPLAY="${SKIP_SELFPLAY:-0}"

MODELS_BASE="$BASE_DIR/models_base/Qwen"
MODEL_PATH="$MODELS_BASE/Qwen2.5-${MODEL_SIZE}-Instruct"
LORA_DIR="$BASE_DIR/lora_models_toxicn"
MERGED_DIR="$BASE_DIR/merged_models_toxicn"
PREPARED_DATA="$BASE_DIR/prepared_data"
EVAL_DIR="$BASE_DIR/eval_results"
LOG_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$EVAL_DIR" "$LOG_DIR"

# 昇腾环境初始化
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ChineseHarm 全流程 Pipeline                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  模型: Qwen2.5-${MODEL_SIZE}-Instruct"
echo "  NPU:  ${N_DEVICES} 卡"
echo "  时间: ${TIMESTAMP}"
echo ""

# 检查基础模型
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 基础模型不存在: $MODEL_PATH"
    echo "   请先运行: bash run_01_download.sh"
    exit 1
fi

# 检查数据
if [ ! -f "$PREPARED_DATA/challenger_sft/train.jsonl" ]; then
    echo "❌ SFT 数据未部署，先运行 setup_corrected_data.sh"
    exit 1
fi

echo "✓ 前置检查通过"
echo ""

# =============================================================================
# Step 1: LoRA SFT 微调
# =============================================================================
if [ "$SKIP_SFT" -eq 0 ]; then
    echo "============================================================"
    echo "  Step 1/5: LoRA SFT 训练 (${MODEL_SIZE})"
    echo "============================================================"

    SFT_LOG="$LOG_DIR/sft_${MODEL_SIZE}_${TIMESTAMP}.log"

    # ── 超参配置 ──
    BATCH_SIZE=4
    NUM_EPOCHS=3
    TARGET_EFFECTIVE_BATCH=32
    GRAD_ACCUM=$(( TARGET_EFFECTIVE_BATCH / BATCH_SIZE / N_DEVICES ))
    [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

    case "$MODEL_SIZE" in
        "0.5B"|"1.5B") LORA_RANK=16; LORA_ALPHA=32; LR=3e-4 ;;
        "3B"|"7B")     LORA_RANK=32; LORA_ALPHA=64; LR=2e-4 ;;
        "14B")         LORA_RANK=64; LORA_ALPHA=128; LR=1e-4 ;;
        *)             LORA_RANK=32; LORA_ALPHA=64; LR=2e-4 ;;
    esac

    if [ "$N_DEVICES" -ge 2 ]; then
        LAUNCH_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$N_DEVICES"
    else
        LAUNCH_CMD="python"
    fi

    echo "  配置: rank=$LORA_RANK alpha=$LORA_ALPHA lr=$LR grad_accum=$GRAD_ACCUM"
    echo "  日志: $SFT_LOG"
    echo ""

    # ── Challenger SFT ──
    echo "[${MODEL_SIZE}] Challenger LoRA 训练..."
    CHALLENGER_OUT="$LORA_DIR/challenger_${MODEL_SIZE}"
    mkdir -p "$CHALLENGER_OUT"

    $LAUNCH_CMD ../model_lora/train_challenger_lora.py \
        --model_path "$MODEL_PATH" \
        --data_path "$PREPARED_DATA/challenger_sft/train.jsonl" \
        --val_data_path "$PREPARED_DATA/challenger_sft/val.jsonl" \
        --output_dir "$CHALLENGER_OUT" \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --max_length 1024 \
        --seed 42 \
        --device "npu:0" \
        --n_devices $N_DEVICES \
        2>&1 | tee -a "$SFT_LOG"

    echo "✓ Challenger SFT 完成"
    echo ""

    # ── Reviewer SFT ──
    echo "[${MODEL_SIZE}] Reviewer LoRA 训练..."
    REVIEWER_OUT="$LORA_DIR/reviewer_${MODEL_SIZE}"
    mkdir -p "$REVIEWER_OUT"

    $LAUNCH_CMD ../model_lora/train_reviewer_lora.py \
        --model_path "$MODEL_PATH" \
        --data_path "$PREPARED_DATA/reviewer_sft/train.jsonl" \
        --val_data_path "$PREPARED_DATA/reviewer_sft/val.jsonl" \
        --output_dir "$REVIEWER_OUT" \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $LR \
        --max_length 2048 \
        --seed 42 \
        --device "npu:0" \
        --n_devices $N_DEVICES \
        2>&1 | tee -a "$SFT_LOG"

    echo "✓ Reviewer SFT 完成"
    echo ""
else
    echo "⏭  跳过 Step 1 (SKIP_SFT=1)"
fi

# =============================================================================
# Step 2: 合并 LoRA 到基础模型
# =============================================================================
if [ "$SKIP_MERGE" -eq 0 ]; then
    echo "============================================================"
    echo "  Step 2/5: 合并 LoRA 权重 (${MODEL_SIZE})"
    echo "============================================================"

    MERGE_LOG="$LOG_DIR/merge_${MODEL_SIZE}_${TIMESTAMP}.log"

    for ROLE in challenger reviewer; do
        LORA_PATH="$LORA_DIR/${ROLE}_${MODEL_SIZE}"
        OUTPUT_PATH="$MERGED_DIR/${ROLE}_${MODEL_SIZE}"

        if [ ! -d "$LORA_PATH" ] || [ ! -f "$LORA_PATH/adapter_config.json" ]; then
            echo "⚠️  LoRA 不存在: $LORA_PATH，跳过"
            continue
        fi

        echo "  合并 ${ROLE}_${MODEL_SIZE}..."
        python ../model_lora/merge_lora.py \
            --base_model "$MODEL_PATH" \
            --lora_path "$LORA_PATH" \
            --output_path "$OUTPUT_PATH" \
            2>&1 | tee -a "$MERGE_LOG"

        echo "  ✓ ${ROLE}_${MODEL_SIZE} 合并完成 → $OUTPUT_PATH"
        echo ""
    done
else
    echo "⏭  跳过 Step 2 (SKIP_MERGE=1)"
fi

# =============================================================================
# Step 3: Baseline 评估 (原始 Qwen2.5 base 模型)
# =============================================================================
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo "============================================================"
    echo "  Step 3/5: 模型评估"
    echo "============================================================"

    EVAL_LOG="$LOG_DIR/eval_${MODEL_SIZE}_${TIMESTAMP}.log"
    TEST_DATA="$PREPARED_DATA/rl/test_eval.json"

    if [ ! -f "$TEST_DATA" ]; then
        echo "⚠️  评估数据不存在: $TEST_DATA，使用 split_data/test.json"
        TEST_DATA="$BASE_DIR/split_data/test.json"
    fi

    echo "  评估数据: $TEST_DATA"
    echo "  日志: $EVAL_LOG"
    echo ""

    # ── 3a. Baseline: 原始 Qwen2.5 base (未微调) ──
    echo "────── [Baseline] Qwen2.5-${MODEL_SIZE}-Instruct (未微调) ──────"
    python ../model_eval/batch_eval_npu.py \
        --model_path "$MODEL_PATH" \
        --data_path "$TEST_DATA" \
        --output_dir "$EVAL_DIR" \
        --num_npus 1 \
        --batch_size 8 \
        --tag "baseline_${MODEL_SIZE}" \
        2>&1 | tee -a "$EVAL_LOG"

    echo ""

    # ── 3b. Reviewer SFT 模型 ──
    REVIEWER_MERGED="$MERGED_DIR/reviewer_${MODEL_SIZE}"
    if [ -d "$REVIEWER_MERGED" ]; then
        echo "────── [Reviewer SFT] reviewer_${MODEL_SIZE} ──────"
        python ../model_eval/batch_eval_npu.py \
            --model_path "$REVIEWER_MERGED" \
            --data_path "$TEST_DATA" \
            --output_dir "$EVAL_DIR" \
            --num_npus 1 \
            --batch_size 8 \
            --tag "reviewer_sft_${MODEL_SIZE}" \
            2>&1 | tee -a "$EVAL_LOG"
    else
        echo "⚠️  Reviewer 合并模型不存在: $REVIEWER_MERGED，跳过"
    fi

    echo ""
    echo "✓ 评估完成，结果在 $EVAL_DIR/"
    echo ""
else
    echo "⏭  跳过 Step 3 (SKIP_EVAL=1)"
fi

# =============================================================================
# Step 4: 汇总评估对比
# =============================================================================
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo "============================================================"
    echo "  Step 4/5: 评估结果对比"
    echo "============================================================"

    python3 - "$EVAL_DIR" "$MODEL_SIZE" << 'PYEOF'
import json, sys, os, glob

eval_dir = sys.argv[1]
model_size = sys.argv[2]

results = []
for f in sorted(glob.glob(os.path.join(eval_dir, f"eval_*{model_size}*.json"))):
    try:
        with open(f) as fh:
            d = json.load(fh)
        m = d.get("metrics", d)
        tag = os.path.basename(f).replace("eval_", "").replace(".json", "")
        acc = m.get("overall_accuracy", m.get("accuracy", "?"))
        macro_f1 = m.get("macro_f1", "?")
        results.append((tag, acc, macro_f1))
    except Exception:
        pass

if results:
    print(f"\n{'模型':<45} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 68)
    for tag, acc, f1 in results:
        acc_str = f"{acc:.2f}%" if isinstance(acc, (int, float)) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else str(f1)
        print(f"  {tag:<43} {acc_str:>10} {f1_str:>10}")
    print()
else:
    print("  暂无评估结果文件")
PYEOF

fi

# =============================================================================
# Step 5: Self-Play 对抗训练
# =============================================================================
if [ "$SKIP_SELFPLAY" -eq 0 ]; then
    echo "============================================================"
    echo "  Step 5/5: Self-Play 对抗训练 (${MODEL_SIZE}, ${N_GPUS} NPU)"
    echo "============================================================"
    echo ""

    # 检查合并模型
    if [ ! -d "$MERGED_DIR/challenger_${MODEL_SIZE}" ] || [ ! -d "$MERGED_DIR/reviewer_${MODEL_SIZE}" ]; then
        echo "❌ 合并模型不存在，无法启动 Self-Play"
        echo "   需要: $MERGED_DIR/challenger_${MODEL_SIZE}"
        echo "         $MERGED_DIR/reviewer_${MODEL_SIZE}"
        exit 1
    fi

    N_GPUS=$N_GPUS \
    MODEL_SIZE=$MODEL_SIZE \
    TOTAL_STEPS="${TOTAL_STEPS:-50}" \
    CHECK_INTERVAL="${CHECK_INTERVAL:-10}" \
    RESUME=1 \
    bash ../integrated_selfplay/run_selfplay.sh

    echo ""
    echo "✓ Self-Play 训练完成"
else
    echo "⏭  跳过 Step 5 (SKIP_SELFPLAY=1)"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全流程完成                                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "产出物:"
echo "  LoRA 模型:   $LORA_DIR/{challenger,reviewer}_${MODEL_SIZE}/"
echo "  合并模型:    $MERGED_DIR/{challenger,reviewer}_${MODEL_SIZE}/"
echo "  评估结果:    $EVAL_DIR/"
echo "  Self-Play:   $BASE_DIR/selfplay_integrated/${MODEL_SIZE}_${N_GPUS}npu/"
echo "  日志:        $LOG_DIR/"
echo ""
echo "日志文件:"
ls -t "$LOG_DIR"/*.log 2>/dev/null | head -5
