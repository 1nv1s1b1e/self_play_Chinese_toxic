#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# 完整验证流水线 — 在 NPU 服务器上运行
# ═══════════════════════════════════════════════════════════════════════
#
# 验证步骤:
#   Phase 1: 离线逻辑测试 (无需 GPU, 验证奖励函数正确性)
#   Phase 2: 模板对比测试 (需要模型, 决定是否 re-SFT)
#   Phase 3: Temperature 多样性测试 (需要模型, 验证 0.7 的效果)
#   Phase 4: [条件] 精简版 SFT 训练 (如 Phase 2 显示掉点>2%)
#   Phase 5: 小规模 self-play 对比 (3步, 验证整体流程)
#
# 用法:
#   # 基本 (会自动跳过不需要的步骤):
#   bash tests/run_full_validation.sh
#
#   # 指定模型路径:
#   REVIEWER_MODEL=/path/to/reviewer_3B \
#   BASE_MODEL=/path/to/Qwen2.5-3B-Instruct \
#   bash tests/run_full_validation.sh
#
#   # 只跑离线测试 (无 GPU):
#   SKIP_GPU=1 bash tests/run_full_validation.sh
#
# ═══════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$PARENT_DIR")")"

cd "$PARENT_DIR"

# ── 路径配置 (根据你的环境修改) ──
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
REVIEWER_MODEL="${REVIEWER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_3B}"
BASE_MODEL="${BASE_MODEL:-${BASE_DIR}/models_base/Qwen/Qwen2.5-3B-Instruct}"
CHALLENGER_MODEL="${CHALLENGER_MODEL:-${BASE_DIR}/merged_models_toxicn/challenger_3B}"
TEST_DATA="${TEST_DATA:-${BASE_DIR}/split_data/test.json}"
TRAIN_DATA="${TRAIN_DATA:-${BASE_DIR}/split_data/train.json}"
VAL_DATA="${VAL_DATA:-${BASE_DIR}/split_data/val.json}"
SEED_DATA="${SEED_DATA:-${BASE_DIR}/prepared_data/rl/train_seed.parquet}"

SKIP_GPU="${SKIP_GPU:-0}"
N_GPUS="${N_GPUS:-4}"

# 输出目录
VALIDATION_DIR="${BASE_DIR}/validation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$VALIDATION_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  多级奖励改进 — 完整验证流水线                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Reviewer 模型 : $REVIEWER_MODEL"
echo "  Base 模型     : $BASE_MODEL"
echo "  测试集        : $TEST_DATA"
echo "  输出目录      : $VALIDATION_DIR"
echo ""

# ═════════════════════════════════════════════════════════════════════
# Phase 1: 离线逻辑测试
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 1: 离线逻辑测试 (奖励函数 + GRPO 模拟)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 tests/test_reward_variance.py 2>&1 | tee "$VALIDATION_DIR/phase1_reward_variance.log"
python3 tests/test_mini_grpo_simulation.py 2>&1 | tee "$VALIDATION_DIR/phase1_grpo_simulation.log"
python3 tests/test_prompt_length.py 2>&1 | tee "$VALIDATION_DIR/phase1_prompt_length.log"

echo ""
echo "Phase 1 完成! 离线测试通过."
echo ""

if [ "$SKIP_GPU" = "1" ]; then
    echo "SKIP_GPU=1, 跳过 GPU 测试."
    echo "离线测试结果: $VALIDATION_DIR/"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════
# Phase 2: 模板对比 (精简版 vs 完整版, 用现有模型)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 2: 模板对比测试 (决定是否需要 re-SFT)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 tests/eval_template_compare.py \
    --model_path "$REVIEWER_MODEL" \
    --test_data "$TEST_DATA" \
    --max_samples 500 \
    --output_dir "$VALIDATION_DIR/phase2" \
    2>&1 | tee "$VALIDATION_DIR/phase2_template_compare.log"

# 读取掉点情况
ACC_DROP=$(python3 -c "
import json
r = json.load(open('$VALIDATION_DIR/phase2/template_compare_results.json'))
print(f'{r[\"accuracy_drop\"]:.4f}')
" 2>/dev/null || echo "0.0")

echo ""
echo "精简版模板 accuracy 掉点: $ACC_DROP"

NEED_RESFT=0
IS_SIGNIFICANT=$(python3 -c "print('1' if float('$ACC_DROP') > 0.02 else '0')")
if [ "$IS_SIGNIFICANT" = "1" ]; then
    echo "掉点 > 2%, 需要 re-SFT!"
    NEED_RESFT=1
else
    echo "掉点 <= 2%, 不需要 re-SFT, 可直接使用精简版!"
fi

# ═════════════════════════════════════════════════════════════════════
# Phase 3: Temperature 多样性测试
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 3: Temperature 多样性测试 (0.3 vs 0.5 vs 0.7 vs 1.0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 tests/test_temperature_diversity.py \
    --model_path "$REVIEWER_MODEL" \
    --num_samples 10 \
    --num_gen 8 \
    2>&1 | tee "$VALIDATION_DIR/phase3_temperature.log"

# ═════════════════════════════════════════════════════════════════════
# Phase 4: [条件] 精简版 re-SFT
# ═════════════════════════════════════════════════════════════════════
if [ "$NEED_RESFT" = "1" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 4: 精简版 re-SFT (小规模, 1000条, 2epoch)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    SFT_DATA_DIR="$VALIDATION_DIR/phase4_sft_data"

    # 准备精简版 SFT 数据
    python3 tests/prepare_sft_data.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --template short \
        --max_samples 1000 \
        --output_dir "$SFT_DATA_DIR"

    # SFT 训练 (LoRA, 在已有 Reviewer 基础上微调)
    SFT_OUTPUT_DIR="$VALIDATION_DIR/phase4_reviewer_short"

    echo ""
    echo "开始 LoRA SFT (精简版 prompt, 1000条, 2 epoch)..."
    python3 -m torch.distributed.run \
        --nproc_per_node=$N_GPUS \
        --master_port=29700 \
        ../../model_lora/train_reviewer_lora.py \
        --model_path "$REVIEWER_MODEL" \
        --data_path "$SFT_DATA_DIR/train.jsonl" \
        --val_data_path "$SFT_DATA_DIR/val.jsonl" \
        --output_dir "$SFT_OUTPUT_DIR/lora" \
        --lora_rank 32 \
        --lora_alpha 64 \
        --batch_size 4 \
        --gradient_accumulation_steps 2 \
        --num_epochs 2 \
        --learning_rate 2e-4 \
        --max_length 1024 \
        --seed 42 \
        --device npu:0 \
        --n_devices $N_GPUS \
        2>&1 | tee "$VALIDATION_DIR/phase4_sft_training.log"

    echo "Phase 4 SFT 完成!"
    echo "  LoRA 权重: $SFT_OUTPUT_DIR/lora"
    echo ""
    echo "  注意: 如需 merge, 请运行:"
    echo "  python merge_lora.py --base $BASE_MODEL --lora $SFT_OUTPUT_DIR/lora --output $SFT_OUTPUT_DIR/merged"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Phase 4: 跳过 (精简版掉点 <= 2%, 不需要 re-SFT)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# ═════════════════════════════════════════════════════════════════════
# Phase 5: 小规模 self-play (3步)
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Phase 5: 小规模 self-play (3步, 验证完整流程)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 决定用哪个 Reviewer
if [ "$NEED_RESFT" = "1" ] && [ -d "$SFT_OUTPUT_DIR/lora" ]; then
    SELFPLAY_REVIEWER="$SFT_OUTPUT_DIR/merged"
    if [ ! -d "$SELFPLAY_REVIEWER" ]; then
        echo "  注意: re-SFT 的 merged 模型不存在, 使用原始模型"
        SELFPLAY_REVIEWER="$REVIEWER_MODEL"
    fi
else
    SELFPLAY_REVIEWER="$REVIEWER_MODEL"
fi

echo "  Challenger: $CHALLENGER_MODEL"
echo "  Reviewer:   $SELFPLAY_REVIEWER"
echo ""

# 导出变量给 run_selfplay.sh
export BASE_DIR
export MODEL_SIZE=3B
export N_GPUS
export TOTAL_STEPS=3
export SAMPLES_PER_CAT=10
export NONTOXIC_SAMPLES=15
export CHECK_INTERVAL=1
export RESUME=0

# 覆盖模型路径 (如果 run_selfplay.sh 支持的话)
export CHALLENGER_INIT="$CHALLENGER_MODEL"
export REVIEWER_INIT="$SELFPLAY_REVIEWER"

bash run_selfplay.sh 2>&1 | tee "$VALIDATION_DIR/phase5_selfplay.log"

# ═════════════════════════════════════════════════════════════════════
# 汇总
# ═════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  验证流水线完成!                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  结果目录: $VALIDATION_DIR/"
echo ""
echo "  Phase 1: 离线逻辑测试     -> phase1_*.log"
echo "  Phase 2: 模板对比         -> phase2_template_compare.log"
echo "  Phase 3: Temperature 测试 -> phase3_temperature.log"
if [ "$NEED_RESFT" = "1" ]; then
echo "  Phase 4: 精简版 re-SFT    -> phase4_sft_training.log"
fi
echo "  Phase 5: 小规模 self-play -> phase5_selfplay.log"
echo ""
echo "  下一步:"
echo "  1. 检查 Phase 2 结果, 确认模板选择"
echo "  2. 检查 Phase 3 结果, 确认 temperature=0.7 的多样性"
echo "  3. 检查 Phase 5 结果, 确认 self-play 是否有提升趋势"
echo "  4. 如果都 OK, 运行全量 self-play:"
echo "     TOTAL_STEPS=30 bash run_selfplay.sh"
echo ""
