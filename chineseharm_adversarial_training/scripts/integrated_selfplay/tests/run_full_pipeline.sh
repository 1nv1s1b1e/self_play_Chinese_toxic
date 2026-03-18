#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  全流程: Reviewer SFT + Challenger SFT → 评估 → Self-Play
#  数据: corrected_data (正确标签, 无 rules)
#  模型: 全部从 Qwen2.5-3B-Instruct 基座开始, 不使用任何旧模型
# ═══════════════════════════════════════════════════════════════════════
#
#  用法:
#    cd scripts/integrated_selfplay
#    bash tests/run_full_pipeline.sh
#
#  环境变量:
#    BASE_DIR          工作根目录
#    BASE_MODEL        Qwen 基座模型
#    N_GPUS            NPU 数量 (默认 4)
#    SFT_EPOCHS        SFT 轮数 (默认 3)
#    SELFPLAY_STEPS    self-play 步数 (默认 5)
#    SKIP_TO_EVAL      已 merge 的 Reviewer 路径, 跳到评估
#    SKIP_TO_SELFPLAY  已 merge 的 Reviewer+Challenger, 跳到 self-play
#
# ═══════════════════════════════════════════════════════════════════════
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$(dirname "$PARENT_DIR")"

cd "$PARENT_DIR"

# ── 配置 ──
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
BASE_MODEL="${BASE_MODEL:-${BASE_DIR}/models_base/Qwen/Qwen2.5-3B-Instruct}"
N_GPUS="${N_GPUS:-4}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SELFPLAY_STEPS="${SELFPLAY_STEPS:-5}"
SKIP_TO_EVAL="${SKIP_TO_EVAL:-}"
SKIP_TO_SELFPLAY="${SKIP_TO_SELFPLAY:-}"

CORRECTED="${BASE_DIR}/corrected_data"

TAG="run_$(date +%m%d_%H%M)"
OUT="${BASE_DIR}/${TAG}"
mkdir -p "$OUT"

# ── 昇腾环境 ──
set +u
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1

# 训练启动器
BATCH_SIZE=4
GRAD_ACCUM=$(( 32 / BATCH_SIZE / N_GPUS ))
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
if [ "$N_GPUS" -ge 2 ]; then
    LAUNCH_REVIEWER="python3 -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS --master_port=29700"
    LAUNCH_CHALLENGER="python3 -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS --master_port=29701"
else
    LAUNCH_REVIEWER="python3"
    LAUNCH_CHALLENGER="python3"
fi

echo ""
echo "=================================================================="
echo "  Reviewer SFT + Challenger SFT → 评估 → Self-Play"
echo "=================================================================="
echo "  BASE_MODEL     : $BASE_MODEL"
echo "  N_GPUS         : $N_GPUS"
echo "  SFT_EPOCHS     : $SFT_EPOCHS"
echo "  SELFPLAY_STEPS : $SELFPLAY_STEPS"
echo "  OUTPUT         : $OUT"
echo "=================================================================="

# ═════════════════════════════════════════════════════════════════════
# Phase 1: 数据检查
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 1: 数据检查 ──"
for f in train.json val.json test.json sft_train.jsonl sft_val.jsonl challenger_sft_train.jsonl challenger_sft_val.jsonl multi_label_map.json; do
    if [ ! -f "$CORRECTED/$f" ]; then
        echo "  [缺失] $CORRECTED/$f"
        exit 1
    fi
done
echo "  corrected_data/ OK (含 Reviewer SFT + Challenger SFT + 多标签映射)"
cp "$CORRECTED/multi_label_map.json" "$PARENT_DIR/multi_label_map.json"

# ═════════════════════════════════════════════════════════════════════
# Phase 2: Reviewer SFT
# ═════════════════════════════════════════════════════════════════════
REVIEWER_MERGED="$OUT/merged_reviewer"
CHALLENGER_MERGED="$OUT/merged_challenger"

if [ -n "$SKIP_TO_SELFPLAY" ]; then
    echo ""
    echo "── Phase 2/3/4: 跳过 (SKIP_TO_SELFPLAY) ──"
    REVIEWER_MERGED="${SKIP_TO_SELFPLAY%,*}"
    CHALLENGER_MERGED="${SKIP_TO_SELFPLAY#*,}"
elif [ -n "$SKIP_TO_EVAL" ]; then
    echo ""
    echo "── Phase 2/3: 跳过 SFT (SKIP_TO_EVAL=$SKIP_TO_EVAL) ──"
    REVIEWER_MERGED="$SKIP_TO_EVAL"
    # Challenger 还是需要跑
else

echo ""
echo "── Phase 2: Reviewer LoRA SFT ──"
echo "  数据: $(wc -l < "$CORRECTED/sft_train.jsonl") 条 (正确标签, 无 rules)"

$LAUNCH_REVIEWER "$SCRIPTS_DIR/model_lora/train_reviewer_lora.py" \
    --model_path "$BASE_MODEL" \
    --data_path "$CORRECTED/sft_train.jsonl" \
    --val_data_path "$CORRECTED/sft_val.jsonl" \
    --output_dir "$OUT/lora_reviewer" \
    --lora_rank 32 --lora_alpha 64 \
    --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $SFT_EPOCHS --learning_rate 2e-4 --max_length 1024 \
    --seed 42 --device "npu:0" --n_devices $N_GPUS \
    2>&1 | tee "$OUT/phase2_reviewer_sft.log"

echo "  Merge Reviewer LoRA..."
python3 "$SCRIPTS_DIR/model_lora/merge_lora.py" \
    --base_model "$BASE_MODEL" \
    --lora_path "$OUT/lora_reviewer" \
    --output_path "$REVIEWER_MERGED" \
    2>&1 | tee -a "$OUT/phase2_reviewer_sft.log"
echo "  Reviewer: $REVIEWER_MERGED"

# ═════════════════════════════════════════════════════════════════════
# Phase 3: Challenger SFT
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: Challenger LoRA SFT ──"
echo "  数据: $(wc -l < "$CORRECTED/challenger_sft_train.jsonl") 条 (正确标签)"

$LAUNCH_CHALLENGER "$SCRIPTS_DIR/model_lora/train_challenger_lora.py" \
    --model_path "$BASE_MODEL" \
    --data_path "$CORRECTED/challenger_sft_train.jsonl" \
    --val_data_path "$CORRECTED/challenger_sft_val.jsonl" \
    --output_dir "$OUT/lora_challenger" \
    --lora_rank 32 --lora_alpha 64 \
    --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $SFT_EPOCHS --learning_rate 2e-4 --max_length 512 \
    --seed 42 --device "npu:0" --n_devices $N_GPUS \
    2>&1 | tee "$OUT/phase3_challenger_sft.log"

echo "  Merge Challenger LoRA..."
python3 "$SCRIPTS_DIR/model_lora/merge_lora.py" \
    --base_model "$BASE_MODEL" \
    --lora_path "$OUT/lora_challenger" \
    --output_path "$CHALLENGER_MERGED" \
    2>&1 | tee -a "$OUT/phase3_challenger_sft.log"
echo "  Challenger: $CHALLENGER_MERGED"

fi  # end SKIP

# ═════════════════════════════════════════════════════════════════════
# Phase 4: Reviewer 评估 (SFT 基线)
# ═════════════════════════════════════════════════════════════════════
if [ -z "$SKIP_TO_SELFPLAY" ]; then
echo ""
echo "── Phase 4: Reviewer 评估 (正确标签测试集) ──"
echo "  模型: $REVIEWER_MERGED"
echo "  测试: $CORRECTED/test.json"

EVAL_SCRIPT="$SCRIPTS_DIR/model_eval/batch_eval_npu_vllm.py"
if [ -f "$EVAL_SCRIPT" ]; then
    mkdir -p "$OUT/eval"
    set +u
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    python3 "$EVAL_SCRIPT" \
        --data_path "$CORRECTED/test.json" \
        --model_path "$REVIEWER_MERGED" \
        --output_dir "$OUT/eval" \
        --num_npus "$N_GPUS" \
        --tag "sft_baseline" \
        --batch_size 128 \
        2>&1 | tee "$OUT/phase4_eval.log"
    set -u
else
    echo "  [fallback] eval_template_compare.py"
    python3 tests/eval_template_compare.py \
        --model_path "$REVIEWER_MERGED" \
        --test_data "$CORRECTED/test.json" \
        --max_samples 0 \
        --output_dir "$OUT/eval" \
        2>&1 | tee "$OUT/phase4_eval.log"
fi
echo ""
echo "  SFT 基线已记录. self-play 在此基础上提升."
fi  # end SKIP_TO_SELFPLAY

# ═════════════════════════════════════════════════════════════════════
# Phase 5: Self-Play
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 5: Self-Play ($SELFPLAY_STEPS 步) ──"
echo "  Challenger: $CHALLENGER_MERGED"
echo "  Reviewer:   $REVIEWER_MERGED"

SP_DIR="$OUT/selfplay"
mkdir -p "$SP_DIR/run" "$SP_DIR/logs" "$SP_DIR/data"

set +u
CHALLENGER_INIT="$CHALLENGER_MERGED" \
REVIEWER_INIT="$REVIEWER_MERGED" \
SEED_DATA="$CORRECTED/train.json" \
REVIEWER_EVAL_DATA="$CORRECTED/val.json" \
SELFPLAY_DIR="$SP_DIR/run" \
LOG_DIR="$SP_DIR/logs" \
DATA_DIR="$SP_DIR/data" \
MODEL_SIZE=3B \
N_GPUS=$N_GPUS \
TOTAL_STEPS=$SELFPLAY_STEPS \
SAMPLES_PER_CAT=20 \
NONTOXIC_SAMPLES=17 \
CHECK_INTERVAL=1 \
RESUME=0 \
REVIEWER_MIX_RATIO=0.3 \
REVIEWER_NONTOXIC_BOOST=1.0 \
bash run_selfplay.sh 2>&1 | tee "$OUT/phase5_selfplay.log"

# ═════════════════════════════════════════════════════════════════════
# 汇总
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "=================================================================="
echo "  完成!"
echo "=================================================================="
echo ""
echo "  输出: $OUT/"
echo "  Reviewer:   $REVIEWER_MERGED"
echo "  Challenger: $CHALLENGER_MERGED"
echo ""
echo "  Phase 2: Reviewer SFT   -> phase2_reviewer_sft.log"
echo "  Phase 3: Challenger SFT -> phase3_challenger_sft.log"
echo "  Phase 4: 评估 (SFT基线)  -> phase4_eval.log"
echo "  Phase 5: Self-Play      -> phase5_selfplay.log"
echo ""
echo "  查看 SFT 基线: cat $OUT/phase4_eval.log | tail -30"
echo "  查看 self-play: grep '评估\|acc\|f1' $OUT/phase5_selfplay.log"
echo ""
echo "  跳过 SFT 直接跑更多 self-play:"
echo "    SKIP_TO_SELFPLAY=$REVIEWER_MERGED,$CHALLENGER_MERGED \\"
echo "    SELFPLAY_STEPS=30 bash tests/run_full_pipeline.sh"
echo ""
