#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  对抗博弈 RL 训练入口脚本 — TRL + DeepSpeed (昇腾 910B) (集成版)              ║
# ║                                                                              ║
# ║  支持三种模式:                                                                 ║
# ║    MODE=selfplay : 完整对抗博弈自对弈循环 (默认推荐)                             ║
# ║    MODE=single   : 单角色单次 GRPO 训练 (调试/验证)                             ║
# ║    MODE=eval     : 汇总评估已有的 selfplay_stats                                ║
# ║                                                                              ║
# ║  快速用法:                                                                      ║
# ║    bash run_05_trl_grpo.sh                           # 默认: selfplay          ║
# ║    MODE=single ROLE=challenger bash run_05_trl_grpo.sh  # 单次 challenger      ║
# ║    MODE=eval bash run_05_trl_grpo.sh                    # 评估汇总             ║
# ║    N_GPUS=4 MODEL_SIZE=3B SELFPLAY_ROUNDS=5 bash run_05_trl_grpo.sh           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# 1. Python 解释器
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

# 训练模式: selfplay (完整对抗循环) | single (单次 GRPO) | eval (评估汇总)
MODE="${MODE:-selfplay}"
# 单次模式下的角色 (challenger 或 reviewer)
ROLE="${ROLE:-challenger}"

echo "==========================================================="
echo "  训练模式: ${MODE}  |  模型尺寸: ${MODEL_SIZE}  |  N_GPUS: ${N_GPUS}"
echo "==========================================================="

if [ "${MODE}" != "selfplay" ] && [ "${MODE}" != "single" ] && [ "${MODE}" != "eval" ]; then
    echo "❌ 不支持的 MODE=${MODE}，仅支持: selfplay / single / eval"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 4. 分支: selfplay 模式 → 转发给主脚本
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${MODE}" = "selfplay" ]; then
    echo "→ 启动完整对抗博弈自对弈循环 (集成版 Plan 1+3+4)..."
    export VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_3B}"
    # Verifier 后端: local (本地模型) / api (Qwen DashScope API) / async (异步 API)
    export VERIFIER_BACKEND="${VERIFIER_BACKEND:-local}"
    export VERIFIER_API_KEY="${VERIFIER_API_KEY:-${DASHSCOPE_API_KEY:-}}"
    export VERIFIER_API_MODEL="${VERIFIER_API_MODEL:-qwen-plus}"
    echo "  Verifier [冻结]: ${VERIFIER_MODEL}"
    echo "  Verifier 后端  : ${VERIFIER_BACKEND}"
    export BASE_DIR MODEL_SIZE N_GPUS PYTHON_EXEC VERIFIER_MODEL VERIFIER_BACKEND VERIFIER_API_KEY VERIFIER_API_MODEL
    if [ ! -f "${SCRIPT_DIR}/run_selfplay.sh" ]; then
        echo "❌ 找不到 selfplay 脚本: ${SCRIPT_DIR}/run_selfplay.sh"
        exit 1
    fi
    bash "${SCRIPT_DIR}/run_selfplay.sh"
    exit $?
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 评估汇总模式
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${MODE}" = "eval" ]; then
    SELFPLAY_STATS_DIR="${SELFPLAY_STATS_DIR:-${BASE_DIR}/selfplay_integrated_data/${MODEL_SIZE}}"
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
    $PYTHON_EXEC "${SCRIPT_DIR}/eval_selfplay_results.py" \
        --mode summary \
        --stats_dir "${SELFPLAY_STATS_DIR}" \
        --output_dir "${BASE_DIR}/eval_results"
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

echo ""
echo "单次 GRPO 训练参数:"
echo "  角色      : ${ROLE}"
echo "  模型路径  : ${MODEL_PATH}"
echo "  数据路径  : ${SEED_DATA}"
echo "  输出目录  : ${OUTPUT_DIR}"
echo "  最大步数  : ${MAX_STEPS}"

mkdir -p "${OUTPUT_DIR}"

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
    --selfplay_round         0 \
    --deepspeed "${SCRIPT_DIR}/ds_zero2.json"

echo "✅ 单次 GRPO 训练完成！输出: ${OUTPUT_DIR}"
