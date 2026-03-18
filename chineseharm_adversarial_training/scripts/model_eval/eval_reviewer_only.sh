#!/bin/bash
set -e

# ==============================================================================
# 仅评测 Reviewer 分类准确率 (昇腾 NPU)
# 依次评测每轮训练出的 Reviewer 模型在 test.parquet 上的表现
# ==============================================================================

BASE_DIR="/home/ma-user/work/test"
MODEL_SIZE=${1:-"3B"}  # 可通过参数传入 0.5B / 1.5B 等
N_GPUS=4
ROUNDS=5

# 基础路径
SELFPLAY_DIR="${BASE_DIR}/selfplay_outputs_sft_reviewer/${MODEL_SIZE}_${N_GPUS}npu"
EVAL_OUT_DIR="${BASE_DIR}/eval_results/selfplay_${MODEL_SIZE}_reviewer_only_4npu"
TEST_DATA="${BASE_DIR}/split_data/test.parquet"

mkdir -p "${EVAL_OUT_DIR}"

echo "=========================================================="
echo " 🛡️ 开始评测 Reviewer 分类能力，共 ${ROUNDS} 轮 (模型: ${MODEL_SIZE})"
echo "=========================================================="

for ROUND in $(seq 1 ${ROUNDS}); do
    echo "──────────────────────────────────────────────────────────"
    echo " 🚀 正在评测 Round ${ROUND} Reviewer ..."
    echo "──────────────────────────────────────────────────────────"

    ROUND_DIR="${SELFPLAY_DIR}/round_${ROUND}"
    REVIEWER_BASE="${ROUND_DIR}/reviewer"      # 该轮合并后的基座模型
    REVIEWER_LORA="${ROUND_DIR}/reviewer_lora" # 该轮在基座上的LoRA adapter

    if [ -d "${REVIEWER_LORA}" ]; then
        echo "   -> 基座模型: ${REVIEWER_BASE}"
        echo "   -> LoRA adapter: ${REVIEWER_LORA}"
        echo "   -> 评测数据集: ${TEST_DATA}"
        
        python batch_eval_npu_vllm.py \
            --data_path "${TEST_DATA}" \
            --model_path "${REVIEWER_BASE}" \
            --lora_path "${REVIEWER_LORA}" \
            --output_dir "${EVAL_OUT_DIR}" \
            --num_npus ${N_GPUS} \
            --tag "eval_reviewer_r${ROUND}" \
            --enforce_eager
    else
        echo "⚠️ 找不到 Reviewer LoRA: ${REVIEWER_LORA}，跳过"
    fi
done

echo "=========================================================="
echo " 🎉 所有 Reviewer 评测完成！结果保存在: ${EVAL_OUT_DIR}"
echo "=========================================================="
