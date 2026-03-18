#!/bin/bash
set -e

# ==============================================================================
# 自对弈多轮模型评测脚本 (昇腾 NPU)
# 依次评测 Challenger 的生成攻击成功率 和 Reviewer 的分类准确率
# ==============================================================================

BASE_DIR="/home/ma-user/work/test"
MODEL_SIZE="0.5B"
N_GPUS=2
ROUNDS=3

# 基础路径
SELFPLAY_DIR="${BASE_DIR}/selfplay_outputs_trl/${MODEL_SIZE}_${N_GPUS}npu"
EVAL_OUT_DIR="${BASE_DIR}/eval_results/selfplay_0.5B"
VERIFIER_MODEL="${BASE_DIR}/merged_models_toxicn/reviewer_3B" # 用作评估 Challenger 攻击成功率的标准
TEST_DATA="${BASE_DIR}/split_data/test.parquet"

mkdir -p "${EVAL_OUT_DIR}"

echo "=========================================================="
echo " 开始评测自对弈模型，共 ${ROUNDS} 轮"
echo "=========================================================="

for ROUND in $(seq 1 ${ROUNDS}); do
    echo "──────────────────────────────────────────────────────────"
    echo " 🚀 正在评测 Round ${ROUND} ..."
    echo "──────────────────────────────────────────────────────────"

    ROUND_DIR="${SELFPLAY_DIR}/round_${ROUND}"
    CHALLENGER_MODEL="${ROUND_DIR}/challenger"
    REVIEWER_MODEL="${ROUND_DIR}/reviewer"

    # --------------------------------------------------------
    # 1. 评测 Challenger
    # 生成对抗文本，然后用 Verifier_3B 评估其 ASR（攻击成功率）
    # --------------------------------------------------------
    if [ -d "${CHALLENGER_MODEL}" ]; then
        echo "[1] 评测 Challenger 生成能力 (Round ${ROUND})"
        GEN_FILE="${EVAL_OUT_DIR}/challenger_r${ROUND}_gen.jsonl"
        
        # 1.1 生成文本
        echo "   -> 生成对抗样本..."
        python scripts/model_eval/eval_challenger_generation.py \
            --model_path "${CHALLENGER_MODEL}" \
            --output_file "${GEN_FILE}" \
            --num_samples 20 \
            --max_new_tokens 128 \
            --device "npu:0"
    else
        echo "⚠️ 找不到 Challenger 模型: ${CHALLENGER_MODEL}，跳过"
    fi

    # --------------------------------------------------------
    # 2. 评测 Reviewer
    # 在测试集 (test.parquet) 上评测分类准确率、F1 等指标
    # --------------------------------------------------------
    if [ -d "${REVIEWER_MODEL}" ]; then
        echo "[2] 评测 Reviewer 分类能力 (Round ${ROUND})"
        echo "   -> 使用 test.parquet 测试分类准确率..."
        python scripts/model_eval/batch_eval_npu.py \
            --data_path "${TEST_DATA}" \
            --model_path "${REVIEWER_MODEL}" \
            --output_dir "${EVAL_OUT_DIR}" \
            --batch_size 16 \
            --num_npus 2 \
            --tag "eval_reviewer_r${ROUND}"
    else
        echo "⚠️ 找不到 Reviewer 模型: ${REVIEWER_MODEL}，跳过"
    fi
done

echo "=========================================================="
echo " 🎉 所有评测完成！结果保存在: ${EVAL_OUT_DIR}"
echo "=========================================================="
