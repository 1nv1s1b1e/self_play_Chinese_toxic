#!/bin/bash
set -e

# ==============================================================================
# 初始 Reviewer 模型效果评测 (昇腾 NPU)
# 评测 rl_train 使用的初始 Reviewer 模型 (如 0.5B, 1.5B)
# ==============================================================================

BASE_DIR="/home/ma-user/work/test"
MODEL_SIZE=${1:-"0.5B"}  # 可通过参数传入 0.5B / 1.5B 等
N_GPUS=2

# 初始 Reviewer 模型路径 (对应 run_selfplay_trl_npu.sh 中的 REVIEWER_INIT)
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"

EVAL_OUT_DIR="${BASE_DIR}/eval_results/initial_models_reviewer"
TEST_DATA="${BASE_DIR}/split_data/test.parquet"

mkdir -p "${EVAL_OUT_DIR}"

echo "=========================================================="
echo " 🛡️ 开始评测初始 Reviewer 分类能力 (模型: ${MODEL_SIZE})"
echo "=========================================================="

if [ -d "${REVIEWER_INIT}" ]; then
    echo "   -> 模型路径: ${REVIEWER_INIT}"
    echo "   -> 评测数据集: ${TEST_DATA}"
    
    python scripts/model_eval/batch_eval_npu.py \
        --data_path "${TEST_DATA}" \
        --model_path "${REVIEWER_INIT}" \
        --output_dir "${EVAL_OUT_DIR}" \
        --batch_size 16 \
        --num_npus ${N_GPUS} \
        --tag "eval_reviewer_initial_${MODEL_SIZE}"
else
    echo "⚠️ 找不到初始 Reviewer 模型: ${REVIEWER_INIT}"
    echo "请检查模型合并 (merge_peft) 是否已成功运行或路径是否正确。"
fi

echo "=========================================================="
echo " 🎉 初始 Reviewer 评测完成！结果保存在: ${EVAL_OUT_DIR}"
echo "=========================================================="
