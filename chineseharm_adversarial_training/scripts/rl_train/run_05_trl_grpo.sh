#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  对抗博弈 RL 训练入口脚本 — TRL + DeepSpeed (昇腾 910B)                        ║
# ║                                                                              ║
# ║  支持两种模式:                                                                   ║
# ║    MODE=single    : 单角色单次 GRPO 训练 (调试/验证)                             ║
# ║    MODE=selfplay  : 完整对抗博弈自对弈循环 (默认推荐)                              ║
# ║                                                                              ║
# ║  快速用法:                                                                      ║
# ║    bash run_05_trl_grpo.sh                           # 默认: selfplay 模式     ║
# ║    MODE=single ROLE=challenger bash run_05_trl_grpo.sh  # 单次 challenger     ║
# ║    MODE=single ROLE=reviewer   bash run_05_trl_grpo.sh  # 单次 reviewer       ║
# ║    N_GPUS=4 MODEL_SIZE=3B SELFPLAY_ROUNDS=5 bash run_05_trl_grpo.sh           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# 1. Python 解释器 (ssp_train 环境绝对路径，不依赖 conda activate)
# ─────────────────────────────────────────────────────────────────────────────────
PYTHON_EXEC="${PYTHON_EXEC:-/home/ma-user/.conda/envs/ssp_train/bin/python}"
echo "Python 解释器: $PYTHON_EXEC"
$PYTHON_EXEC --version

# ─────────────────────────────────────────────────────────────────────────────────
# 2. 昇腾 CANN 运行环境
# ─────────────────────────────────────────────────────────────────────────────────
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && echo "✓ ascend-toolkit"
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh       && echo "✓ nnal/atb"

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export TASK_QUEUE_ENABLE=2

# ─────────────────────────────────────────────────────────────────────────────────
# 3. 通用参数
# ─────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
MODEL_SIZE="${MODEL_SIZE:-0.5B}"
N_GPUS="${N_GPUS:-2}"

# 训练模式: selfplay (完整对抗循环) 或 single (单次 GRPO)
MODE="${MODE:-selfplay}"
# 单次模式下的角色 (challenger 或 reviewer)
ROLE="${ROLE:-challenger}"
SELFPLAY_SCRIPT="${SELFPLAY_SCRIPT:-run_selfplay_trl_npu.sh}"

echo "==========================================================="
echo "  训练模式: ${MODE}  |  模型尺寸: ${MODEL_SIZE}  |  N_GPUS: ${N_GPUS}"
echo "==========================================================="

if [ "${MODE}" != "selfplay" ] && [ "${MODE}" != "single" ] && [ "${MODE}" != "eval" ]; then
    echo "❌ 不支持的 MODE=${MODE}，仅支持: selfplay / single / eval"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 4. 分支: selfplay 模式 → 转发给专用脚本
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${MODE}" = "selfplay" ]; then
    echo "→ 启动完整对抗博弈自对弈循环..."
    # 传递 VERIFIER_MODEL 环境变量给子脚本
    # 默认规则: 固定使用 merged_models_toxicn/reviewer_3B (LoRA 合并后)
    export VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_3B}"
    # Verifier 后端: local (本地模型) 或 api (Qwen DashScope API)
    export VERIFIER_BACKEND="${VERIFIER_BACKEND:-local}"
    export VERIFIER_API_MODEL="${VERIFIER_API_MODEL:-qwen-plus}"
    echo "  Verifier [冻结]: ${VERIFIER_MODEL}"
    echo "  Verifier 后端  : ${VERIFIER_BACKEND}"
    export BASE_DIR MODEL_SIZE N_GPUS PYTHON_EXEC VERIFIER_MODEL VERIFIER_BACKEND VERIFIER_API_MODEL
    if [ ! -f "${SCRIPT_DIR}/${SELFPLAY_SCRIPT}" ]; then
        echo "❌ 找不到 selfplay 脚本: ${SCRIPT_DIR}/${SELFPLAY_SCRIPT}"
        exit 1
    fi
    bash "${SCRIPT_DIR}/${SELFPLAY_SCRIPT}"
    exit $?
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 评估汇总模式
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${MODE}" = "eval" ]; then
    SELFPLAY_STATS_DIR="${SELFPLAY_STATS_DIR:-${BASE_DIR}/selfplay_dynamic_data/${MODEL_SIZE}}"
    echo "评估目录: ${SELFPLAY_STATS_DIR}"

    if [ ! -d "${SELFPLAY_STATS_DIR}" ]; then
        echo "❌ 目录不存在: ${SELFPLAY_STATS_DIR}"
        exit 1
    fi

    STATS_FILES=$(find "${SELFPLAY_STATS_DIR}" -name "selfplay_stats_round*.json" | sort)
    if [ -z "${STATS_FILES}" ]; then
        echo "❌ 未找到评估文件 selfplay_stats_round*.json"
        exit 1
    fi

    echo ""
    echo "按轮评估汇总:"
    $PYTHON_EXEC - "${SELFPLAY_STATS_DIR}" << 'PY'
import json
import os
import glob
import sys

stats_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(stats_dir, "**", "selfplay_stats_round*.json"), recursive=True))
print("round | overall_asr | reviewer_bin_acc | reviewer_cat_acc | challenger_data | reviewer_data")
print("----- | ----------- | ---------------- | ---------------- | --------------- | -------------")
for p in files:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    round_idx = d.get("round")
    asr = d.get("overall_verifier_asr", 0.0)
    overall = d.get("overall_metrics", {})
    bin_acc = overall.get("reviewer_binary_acc", 0.0)
    cat_acc = overall.get("reviewer_category_acc", 0.0)
    csz = d.get("challenger_grpo_size", 0)
    rsz = d.get("reviewer_grpo_size", 0)
    print(f"{round_idx:>5} | {asr:>11.4f} | {bin_acc:>16.4f} | {cat_acc:>16.4f} | {csz:>15} | {rsz:>13}")
PY
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 6. 单次 GRPO 训练模式
# ─────────────────────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/merged_models_toxicn/${ROLE}_${MODEL_SIZE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/grpo_single_output/${ROLE}_${MODEL_SIZE}}"
SEED_DATA="${SEED_DATA:-${BASE_DIR}/prepared_data/rl/train_seed.parquet}"
MAX_STEPS="${MAX_STEPS:-100}"
MASTER_PORT="${MASTER_PORT:-29600}"
SINGLE_USE_SELFPLAY="${SINGLE_USE_SELFPLAY:-0}"

echo ""
echo "单次 GRPO 训练参数:"
echo "  角色      : ${ROLE}"
echo "  模型路径  : ${MODEL_PATH}"
echo "  数据路径  : ${SEED_DATA}"
echo "  输出目录  : ${OUTPUT_DIR}"
echo "  最大步数  : ${MAX_STEPS}"
echo "  Selfplay  : ${SINGLE_USE_SELFPLAY}"

mkdir -p "${OUTPUT_DIR}"

SELFPLAY_ARG="--no_selfplay"
if [ "${SINGLE_USE_SELFPLAY}" = "1" ]; then
    SELFPLAY_ARG="--use_selfplay"
fi

$PYTHON_EXEC -m torch.distributed.run \
    --nproc_per_node="${N_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/adversarial_trl_grpo.py" \
    --role                   "${ROLE}" \
    --model_path             "${MODEL_PATH}" \
    --dataset_path           "${SEED_DATA}" \
    --output_dir             "${OUTPUT_DIR}" \
    --max_steps              "${MAX_STEPS}" \
    --save_steps             "${MAX_STEPS}" \
    ${SELFPLAY_ARG} \
    --deepspeed "${SCRIPT_DIR}/ds_zero2.json"

echo "✅ 单次 GRPO 训练完成！输出: ${OUTPUT_DIR}"
