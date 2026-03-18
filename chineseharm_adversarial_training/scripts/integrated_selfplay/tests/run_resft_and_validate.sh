#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  精简版 Prompt Re-SFT + 多级奖励验证 — 一键流水线
# ═══════════════════════════════════════════════════════════════════════
#
#  流程:
#    Step 1: 用精简版 prompt 重建 SFT 数据 (从 split_data)
#    Step 2: 在 Qwen2.5-3B-Instruct 基座上 LoRA SFT
#    Step 3: Merge LoRA → 完整模型
#    Step 4: 评估新模型 (精简版 prompt)
#    Step 5: Temperature 多样性验证
#    Step 6: 3 步 mini self-play (新奖励函数 + 新模型)
#
#  用法:
#    # 小规模快速验证 (1000 条 SFT, ~30分钟):
#    bash tests/run_resft_and_validate.sh
#
#    # 全量 SFT (8000+ 条, ~2小时):
#    SFT_SAMPLES=0 bash tests/run_resft_and_validate.sh
#
#  环境变量:
#    BASE_DIR        - 工作根目录 (默认 /home/ma-user/work/test)
#    BASE_MODEL      - Qwen 基座模型路径
#    SFT_SAMPLES     - SFT 样本数 (默认 1000, 0=全量)
#    SFT_EPOCHS      - SFT 训练轮数 (默认 3)
#    N_GPUS          - NPU 数量 (默认 4)
#    SKIP_SFT        - 跳过 SFT, 直接用已有模型 (设为已 merge 的模型路径)
#
# ═══════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$(dirname "$PARENT_DIR")"

cd "$PARENT_DIR"

# ── 环境配置 ──
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
BASE_MODEL="${BASE_MODEL:-${BASE_DIR}/models_base/Qwen/Qwen2.5-3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/corrected_data/train.json}"
VAL_DATA="${VAL_DATA:-${BASE_DIR}/corrected_data/val.json}"
TEST_DATA="${TEST_DATA:-${BASE_DIR}/corrected_data/test.json}"
SEED_DATA="${SEED_DATA:-${BASE_DIR}/corrected_data/train.json}"
SFT_TRAIN="${SFT_TRAIN:-${BASE_DIR}/corrected_data/sft_train.jsonl}"
SFT_VAL="${SFT_VAL:-${BASE_DIR}/corrected_data/sft_val.jsonl}"
OLD_REVIEWER="${OLD_REVIEWER:-${BASE_DIR}/merged_models_toxicn/reviewer_3B}"
CHALLENGER_MODEL="${CHALLENGER_MODEL:-${BASE_DIR}/merged_models_toxicn/challenger_3B}"

SFT_SAMPLES="${SFT_SAMPLES:-1000}"   # 小规模测试用 1000 条
SFT_EPOCHS="${SFT_EPOCHS:-3}"
N_GPUS="${N_GPUS:-4}"
SKIP_SFT="${SKIP_SFT:-}"            # 设为模型路径则跳过 SFT

# 输出目录
RUN_TAG="resft_short_$(date +%m%d_%H%M)"
OUTPUT_DIR="${BASE_DIR}/validation_${RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

# ── HCCL 环境 ──
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29700

[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh

echo ""
echo "=================================================================="
echo "  精简版 Prompt Re-SFT + 多级奖励验证"
echo "=================================================================="
echo "  基座模型    : $BASE_MODEL"
echo "  训练集      : $TRAIN_DATA"
echo "  SFT 样本数  : ${SFT_SAMPLES:-全量}"
echo "  SFT Epochs  : $SFT_EPOCHS"
echo "  NPU 数量    : $N_GPUS"
echo "  输出目录    : $OUTPUT_DIR"
echo "=================================================================="
echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 0: 离线逻辑测试 (确认奖励函数正确)
# ═════════════════════════════════════════════════════════════════════
echo "── Step 0: 离线逻辑测试 ──"
python3 tests/test_reward_variance.py > "$OUTPUT_DIR/step0_reward_test.log" 2>&1
python3 tests/test_mini_grpo_simulation.py > "$OUTPUT_DIR/step0_grpo_sim.log" 2>&1
echo "  离线测试通过 (详见 step0_*.log)"
echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 1: 生成精简版 SFT 数据
# ═════════════════════════════════════════════════════════════════════
SFT_DATA_DIR="$OUTPUT_DIR/sft_data_short"

echo "── Step 1: 准备 SFT 数据 ──"
if [ -f "$SFT_TRAIN" ] && [ "$SFT_SAMPLES" -eq 0 ] 2>/dev/null; then
    # 全量: 直接使用预生成的 corrected_data/sft_*.jsonl
    cp "$SFT_TRAIN" "$SFT_DATA_DIR/train.jsonl"
    cp "$SFT_VAL" "$SFT_DATA_DIR/val.jsonl" 2>/dev/null
    echo "  使用预生成的 SFT 数据: $SFT_TRAIN"
    wc -l "$SFT_DATA_DIR/train.jsonl"
else
    # 小规模采样
    SAMPLE_ARG=""
    if [ "$SFT_SAMPLES" -gt 0 ] 2>/dev/null; then
        SAMPLE_ARG="--max_samples $SFT_SAMPLES"
    fi
    python3 tests/prepare_sft_data.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --template short \
        $SAMPLE_ARG \
        --output_dir "$SFT_DATA_DIR"
fi
echo "  SFT 数据就绪" 2>&1 | tee "$OUTPUT_DIR/step1_prepare_data.log"

echo ""

# ═════════════════════════════════════════════════════════════════════
# Step 2: LoRA SFT 训练
# ═════════════════════════════════════════════════════════════════════
LORA_DIR="$OUTPUT_DIR/lora_reviewer_short"
MERGED_DIR="$OUTPUT_DIR/merged_reviewer_short"

if [ -n "$SKIP_SFT" ] && [ -d "$SKIP_SFT" ]; then
    echo "── Step 2: 跳过 SFT (使用已有模型: $SKIP_SFT) ──"
    MERGED_DIR="$SKIP_SFT"
else
    echo "── Step 2: LoRA SFT 训练 (精简版 prompt) ──"
    echo "  基座: $BASE_MODEL"
    echo "  数据: $SFT_DATA_DIR/train.jsonl"
    echo "  Epochs: $SFT_EPOCHS"
    echo ""

    # 根据卡数调整 grad_accum 保持 effective batch=32
    BATCH_SIZE=4
    GRAD_ACCUM=$(( 32 / BATCH_SIZE / N_GPUS ))
    [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

    if [ "$N_GPUS" -ge 2 ]; then
        LAUNCH_CMD="python3 -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS --master_port=$MASTER_PORT"
    else
        LAUNCH_CMD="python3"
    fi

    VAL_ARG=""
    if [ -f "$SFT_DATA_DIR/val.jsonl" ]; then
        VAL_ARG="--val_data_path $SFT_DATA_DIR/val.jsonl"
    fi

    $LAUNCH_CMD "$SCRIPTS_DIR/model_lora/train_reviewer_lora.py" \
        --model_path "$BASE_MODEL" \
        --data_path "$SFT_DATA_DIR/train.jsonl" \
        $VAL_ARG \
        --output_dir "$LORA_DIR" \
        --lora_rank 32 \
        --lora_alpha 64 \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $SFT_EPOCHS \
        --learning_rate 2e-4 \
        --max_length 1024 \
        --seed 42 \
        --device "npu:0" \
        --n_devices $N_GPUS \
        2>&1 | tee "$OUTPUT_DIR/step2_sft_training.log"

    echo ""
    echo "  LoRA 保存: $LORA_DIR"
    echo ""

    # ── Step 3: Merge LoRA ──
    echo "── Step 3: Merge LoRA → 完整模型 ──"
    python3 "$SCRIPTS_DIR/model_lora/merge_lora.py" \
        --base_model "$BASE_MODEL" \
        --lora_path "$LORA_DIR" \
        --output_path "$MERGED_DIR" \
        2>&1 | tee "$OUTPUT_DIR/step3_merge.log"

    echo "  Merged 模型: $MERGED_DIR"
    echo ""
fi

# ═════════════════════════════════════════════════════════════════════
# Step 4: 评估 — 新模型(精简版) vs 旧模型(完整版)
# ═════════════════════════════════════════════════════════════════════
echo "── Step 4: 评估对比 ──"
echo "  新模型(精简版 prompt): $MERGED_DIR"
echo "  旧模型(完整版 prompt): $OLD_REVIEWER"
echo ""

# 4a: 新模型 + 精简版 prompt
echo "  [4a] 新模型 + 精简版 prompt..."
python3 tests/eval_template_compare.py \
    --model_path "$MERGED_DIR" \
    --test_data "$TEST_DATA" \
    --max_samples 500 \
    --output_dir "$OUTPUT_DIR/step4_eval_new" \
    2>&1 | tee "$OUTPUT_DIR/step4a_eval_new.log"

# 4b: 旧模型 + 完整版 prompt (对照基线)
if [ -d "$OLD_REVIEWER" ]; then
    echo ""
    echo "  [4b] 旧模型 + 完整版 prompt (基线)..."
    python3 tests/eval_template_compare.py \
        --model_path "$OLD_REVIEWER" \
        --test_data "$TEST_DATA" \
        --max_samples 500 \
        --output_dir "$OUTPUT_DIR/step4_eval_old" \
        2>&1 | tee "$OUTPUT_DIR/step4b_eval_old.log"
fi

# ═════════════════════════════════════════════════════════════════════
# Step 5: Temperature 多样性验证 (用新模型)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 5: Temperature 多样性测试 ──"
python3 tests/test_temperature_diversity.py \
    --model_path "$MERGED_DIR" \
    --num_samples 8 \
    --num_gen 8 \
    2>&1 | tee "$OUTPUT_DIR/step5_temperature.log"

# ═════════════════════════════════════════════════════════════════════
# Step 6: Mini Self-Play (3 步)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "── Step 6: Mini Self-Play (3 步) ──"
echo "  Challenger: $CHALLENGER_MODEL"
echo "  Reviewer:   $MERGED_DIR"
echo ""

# 设置环境变量
export BASE_DIR
export MODEL_SIZE=3B
export N_GPUS
export TOTAL_STEPS=3
export SAMPLES_PER_CAT=10
export NONTOXIC_SAMPLES=15
export CHECK_INTERVAL=1
export RESUME=0
export REVIEWER_MIX_RATIO=0.3
export REVIEWER_NONTOXIC_BOOST=1.5

# 覆盖 selfplay 的模型路径
SELFPLAY_OVERRIDE_DIR="$OUTPUT_DIR/step6_selfplay"
mkdir -p "$SELFPLAY_OVERRIDE_DIR"

# 创建临时脚本覆盖模型路径
cat > "$SELFPLAY_OVERRIDE_DIR/run_mini.sh" << 'SELFPLAY_EOF'
#!/bin/bash
# Mini self-play wrapper — 覆盖模型路径后调用主脚本
export CHALLENGER_INIT="$1"
export REVIEWER_INIT="$2"
shift 2
exec bash "$@"
SELFPLAY_EOF
chmod +x "$SELFPLAY_OVERRIDE_DIR/run_mini.sh"

# 因为 run_selfplay.sh 内部用 CHALLENGER_INIT/REVIEWER_INIT,
# 但它们是在脚本内部设置的, 我们需要直接 export 覆盖
export SELFPLAY_DIR="${SELFPLAY_OVERRIDE_DIR}/sp_run"
export LOG_DIR="${SELFPLAY_OVERRIDE_DIR}/logs"
export DATA_DIR="${SELFPLAY_OVERRIDE_DIR}/data"
mkdir -p "$SELFPLAY_DIR" "$LOG_DIR" "$DATA_DIR"

# 直接执行 self-play (用 env 变量覆盖)
CHALLENGER_INIT="$CHALLENGER_MODEL" \
REVIEWER_INIT="$MERGED_DIR" \
SELFPLAY_DIR="$SELFPLAY_OVERRIDE_DIR/sp_run" \
LOG_DIR="$SELFPLAY_OVERRIDE_DIR/logs" \
DATA_DIR="$SELFPLAY_OVERRIDE_DIR/data" \
bash run_selfplay.sh 2>&1 | tee "$OUTPUT_DIR/step6_selfplay.log"

# ═════════════════════════════════════════════════════════════════════
# 汇总
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "=================================================================="
echo "  全部完成!"
echo "=================================================================="
echo ""
echo "  输出目录: $OUTPUT_DIR/"
echo ""
echo "  Step 0: 离线逻辑测试    -> step0_*.log"
echo "  Step 1: SFT 数据准备    -> step1_prepare_data.log"
echo "  Step 2: LoRA SFT 训练   -> step2_sft_training.log"
echo "  Step 3: Merge LoRA      -> step3_merge.log"
echo "  Step 4: 评估对比        -> step4a_eval_new.log, step4b_eval_old.log"
echo "  Step 5: Temperature     -> step5_temperature.log"
echo "  Step 6: Mini Self-Play  -> step6_selfplay.log"
echo ""
echo "  关键检查点:"
echo "  1. Step 4: 新模型(精简prompt) 是否接近旧模型(完整prompt) 的效果?"
echo "     grep 'Overall Accuracy' $OUTPUT_DIR/step4a_eval_new.log"
echo "  2. Step 5: temp=0.7 是否比 temp=0.3 产生更多不同类别输出?"
echo "     grep 'unique_cat' $OUTPUT_DIR/step5_temperature.log"
echo "  3. Step 6: 3步 self-play 后 accuracy 是否有上升趋势?"
echo "     grep '评估' $OUTPUT_DIR/step6_selfplay.log"
echo ""
echo "  如果结果满意, 运行全量 self-play:"
echo "    REVIEWER_INIT=$MERGED_DIR TOTAL_STEPS=30 bash run_selfplay.sh"
echo ""
