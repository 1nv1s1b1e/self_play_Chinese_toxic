#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Plan 4: API Verifier 自对弈训练                                              ║
# ║                                                                              ║
# ║  核心改动: 用 72B+ 大模型 API 替换本地冻结 7B Verifier                           ║
# ║    - Phase 0 使用 plan_verifier_api/generate_dynamic_data_plan4.py             ║
# ║    - Phase A / Phase B 完全复用 v1 rl_train/ 脚本                               ║
# ║    - 独立输出目录: selfplay_plan4_verifier_api/                                 ║
# ║                                                                              ║
# ║  所需环境变量 (API 配置):                                                       ║
# ║    VERIFIER_API_KEY   - API 密钥                                               ║
# ║    VERIFIER_API_BASE  - API 端点 (默认: https://dashscope.aliyuncs.com/...)     ║
# ║    VERIFIER_API_MODEL - 模型名 (默认: qwen-plus)                                ║
# ║                                                                              ║
# ║  用法:                                                                         ║
# ║    VERIFIER_API_KEY=sk-xxx bash run_selfplay_plan4.sh                          ║
# ║    VERIFIER_API_KEY=sk-xxx MODEL_SIZE=7B bash run_selfplay_plan4.sh            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# 1. 昇腾 CANN 环境初始化
# ─────────────────────────────────────────────────────────────────────────────────
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && echo "✓ ascend-toolkit 已加载"
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && echo "✓ nnal/atb 已加载"

# ─────────────────────────────────────────────────────────────────────────────────
# 2. Python 解释器
# ─────────────────────────────────────────────────────────────────────────────────
PYTHON_EXEC="${PYTHON_EXEC:-/home/ma-user/.conda/envs/ssp_train/bin/python}"
echo "Python: $PYTHON_EXEC  ($($PYTHON_EXEC --version 2>&1))"

# ─────────────────────────────────────────────────────────────────────────────────
# 3. 昇腾 NPU 环境变量
# ─────────────────────────────────────────────────────────────────────────────────
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export TASK_QUEUE_ENABLE=1

# ─────────────────────────────────────────────────────────────────────────────────
# 3.1 API Verifier 配置检查
# ─────────────────────────────────────────────────────────────────────────────────
if [ -z "${VERIFIER_API_KEY}" ]; then
    echo ""
    echo "⚠️  VERIFIER_API_KEY 未设置！"
    echo "   运行方式: VERIFIER_API_KEY=sk-xxx bash $0"
    echo "   如果不设置，Phase 0 将自动降级使用本地 7B Verifier"
    echo ""
fi

export VERIFIER_API_KEY="${VERIFIER_API_KEY:-}"
export VERIFIER_API_BASE="${VERIFIER_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VERIFIER_API_MODEL="${VERIFIER_API_MODEL:-qwen-plus}"

echo "API Verifier:"
echo "  API Base : ${VERIFIER_API_BASE}"
echo "  Model    : ${VERIFIER_API_MODEL}"
echo "  Key Set  : $([ -n "${VERIFIER_API_KEY}" ] && echo 'YES' || echo 'NO (will fallback to local)')"

# ─────────────────────────────────────────────────────────────────────────────────
# 4. 训练超参数（可通过环境变量覆盖）
# ─────────────────────────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V1_DIR="${SCRIPT_DIR}/../rl_train"

MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-4}"
SELFPLAY_ROUNDS="${SELFPLAY_ROUNDS:-5}"
MAX_STEPS="${MAX_STEPS:-50}"
SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"

# Challenger GRPO 超参 (与 v1 相同)
C_LR="${C_LR:-5e-7}"
C_PER_DEVICE_BS="${C_PER_DEVICE_BS:-2}"
C_NUM_GEN="${C_NUM_GEN:-4}"
C_MAX_COMP_LEN="${C_MAX_COMP_LEN:-256}"
C_GRAD_ACCUM="${C_GRAD_ACCUM:-4}"

# Reviewer SFT 超参 (与 v1 Phase B 相同)
R_LORA_RANK="${R_LORA_RANK:-32}"

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 路径配置 — 独立输出目录
# ─────────────────────────────────────────────────────────────────────────────────
SEED_DATA="${BASE_DIR}/prepared_data/rl/train_seed.parquet"
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_${MODEL_SIZE}"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"

# [Plan 4] 本地 fallback Verifier（当 API 不可用时自动降级）
VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_7B}"

# 独立输出目录，不与 v1 混合
SELFPLAY_DIR="${BASE_DIR}/selfplay_plan4_verifier_api/${MODEL_SIZE}_${N_GPUS}npu"
LOG_DIR="${BASE_DIR}/logs/selfplay_plan4_${MODEL_SIZE}"
DATA_DIR="${BASE_DIR}/selfplay_plan4_data/${MODEL_SIZE}"

mkdir -p "${SELFPLAY_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# ─────────────────────────────────────────────────────────────────────────────────
# 5.1 断点续训恢复 (复用 v1 逻辑)
# ─────────────────────────────────────────────────────────────────────────────────
RESUME_FROM_ROUND=0
RESUME_SKIP_PHASE=""
PROGRESS_FILE="${SELFPLAY_DIR}/progress.json"

if [ "${RESUME}" = "1" ] && [ -f "${PROGRESS_FILE}" ]; then
    echo ""
    echo "── 🔄 检测到断点续训文件: ${PROGRESS_FILE} ──"
    RESUME_FROM_ROUND=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f:
    d = json.load(f)
print(d.get('last_completed_round', 0))
")
    RESUME_PHASE=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f:
    d = json.load(f)
print(d.get('last_completed_phase', 'done'))
")
    SAVED_CHALLENGER=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f:
    d = json.load(f)
print(d.get('current_challenger', ''))
")
    SAVED_REVIEWER=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f:
    d = json.load(f)
print(d.get('current_reviewer', ''))
")

    if [ -n "${SAVED_CHALLENGER}" ] && [ -d "${SAVED_CHALLENGER}" ]; then
        CURRENT_CHALLENGER="${SAVED_CHALLENGER}"
    fi
    if [ -n "${SAVED_REVIEWER}" ] && [ -d "${SAVED_REVIEWER}" ]; then
        CURRENT_REVIEWER="${SAVED_REVIEWER}"
    fi

    if [ "${RESUME_PHASE}" = "done" ]; then
        echo "  上次完整完成到第 ${RESUME_FROM_ROUND} 轮，将从第 $((RESUME_FROM_ROUND + 1)) 轮开始"
    else
        echo "  第 ${RESUME_FROM_ROUND} 轮在 ${RESUME_PHASE} 阶段后中断"
        RESUME_FROM_ROUND=$((RESUME_FROM_ROUND - 1))
        RESUME_SKIP_PHASE="${RESUME_PHASE}"
    fi
else
    if [ "${RESUME}" = "0" ] && [ -f "${PROGRESS_FILE}" ]; then
        echo "── ⚠️ RESUME=0，忽略已有断点文件，从头开始训练 ──"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 6. 打印训练参数
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   Plan 4: API Verifier 对抗博弈自对弈  (昇腾 910B)                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  模型尺寸      : ${MODEL_SIZE}"
echo "  NPU 卡数      : ${N_GPUS}"
echo "  自对弈轮次    : ${SELFPLAY_ROUNDS}"
echo "  每阶段步数    : ${MAX_STEPS}"
echo "  API Verifier  : ${VERIFIER_API_MODEL}"
echo "  输出目录      : ${SELFPLAY_DIR}"
echo "────────────────────────────────────────────────────────────────────"
echo "  初始 Challenger: ${CHALLENGER_INIT}"
echo "  初始 Reviewer  : ${REVIEWER_INIT}"
echo "  种子数据       : ${SEED_DATA}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────────
# 7. 初始化
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${RESUME_FROM_ROUND}" -eq 0 ]; then
    CURRENT_CHALLENGER="${CHALLENGER_INIT}"
    CURRENT_REVIEWER="${REVIEWER_INIT}"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase 0 — 动态数据生成 (用 Plan 4 的 API Verifier 版本)
# ─────────────────────────────────────────────────────────────────────────────────
run_phase0_datagen_api() {
    local ROUND="$1"
    local CHALLENGER_PATH="$2"
    local REVIEWER_PATH="$3"
    local ROUND_DATA_DIR="${DATA_DIR}/round_${ROUND}"
    local LOG_FILE="${LOG_DIR}/round${ROUND}_phase0_api_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "${ROUND_DATA_DIR}"

    echo ""
    echo "  ▶ [$(date +%H:%M:%S)] Phase 0 — API Verifier 数据生成 (Round ${ROUND})"
    echo "     Challenger: ${CHALLENGER_PATH}"
    echo "     Reviewer  : ${REVIEWER_PATH}"
    echo "     Verifier  : API (${VERIFIER_API_MODEL})"
    echo "     Fallback  : ${VERIFIER_MODEL}"
    echo "     日志      : ${LOG_FILE}"

    # ★ Plan 4 核心: 使用 generate_dynamic_data_plan4.py
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${SCRIPT_DIR}/generate_dynamic_data_plan4.py" \
        --challenger_model "${CHALLENGER_PATH}" \
        --reviewer_model   "${REVIEWER_PATH}" \
        --verifier_model   "${VERIFIER_MODEL}" \
        --seed_data        "${SEED_DATA}" \
        --output_dir       "${ROUND_DATA_DIR}" \
        --round_idx        "${ROUND}" \
        --samples_per_cat  "${SAMPLES_PER_CAT}" \
        --batch_size       "${GEN_BATCH_SIZE}" \
        --num_npus         "${N_GPUS}" \
    2>&1 | tee "${LOG_FILE}"

    CHALLENGER_GRPO_DATA=$(grep "^CHALLENGER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)
    REVIEWER_GRPO_DATA=$(grep "^REVIEWER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)

    if [ -z "${CHALLENGER_GRPO_DATA}" ]; then
        CHALLENGER_GRPO_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
    fi
    if [ -z "${REVIEWER_GRPO_DATA}" ]; then
        REVIEWER_GRPO_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"
    fi

    echo "  ✓ [$(date +%H:%M:%S)] Phase 0 完成"
    echo "     Challenger 数据: ${CHALLENGER_GRPO_DATA}"
    echo "     Reviewer   数据: ${REVIEWER_GRPO_DATA}"

    PHASE0_CHALLENGER_DATA="${CHALLENGER_GRPO_DATA}"
    PHASE0_REVIEWER_DATA="${REVIEWER_GRPO_DATA}"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase A — Challenger GRPO (完全复用 v1)
# ─────────────────────────────────────────────────────────────────────────────────
run_grpo_challenger() {
    local MODEL_PATH="$1"
    local DATASET_PATH="$2"
    local OUTPUT_PATH="$3"
    local LOG_FILE="$4"

    echo ""
    echo "  ▶▶ [$(date +%H:%M:%S)] Phase A: Challenger GRPO [修复版] 启动"
    echo "     模型 : ${MODEL_PATH}"
    echo "     数据 : ${DATASET_PATH}"

    $PYTHON_EXEC -m torch.distributed.run \
        --nproc_per_node="${N_GPUS}" \
        --master_port="29600" \
        "${V1_DIR}/adversarial_trl_grpo_fixed.py" \
        --role                   "challenger" \
        --model_path             "${MODEL_PATH}" \
        --dataset_path           "${DATASET_PATH}" \
        --output_dir             "${OUTPUT_PATH}" \
        --max_steps              "${MAX_STEPS}" \
        --save_steps             "${MAX_STEPS}" \
        --learning_rate          "${C_LR}" \
        --per_device_batch_size  "${C_PER_DEVICE_BS}" \
        --num_generations        "${C_NUM_GEN}" \
        --max_completion_length  "${C_MAX_COMP_LEN}" \
        --grad_accum             "${C_GRAD_ACCUM}" \
        --selfplay_round         "${CURRENT_ROUND:-0}" \
        --use_selfplay \
        --deepspeed "${V1_DIR}/ds_zero2.json" \
        2>&1 | tee "${LOG_FILE}"

    if [ -f "${OUTPUT_PATH}/training_done.txt" ]; then
        echo "  ✓ [$(date +%H:%M:%S)] Phase A 完成"
    else
        echo "  ⚠️  Challenger GRPO 可能未正常完成"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────────
# 8. 自对弈主循环
# ─────────────────────────────────────────────────────────────────────────────────
for ROUND in $(seq 1 "${SELFPLAY_ROUNDS}"); do
    CURRENT_ROUND="${ROUND}"

    if [ "${ROUND}" -le "${RESUME_FROM_ROUND}" ]; then
        echo "  ⏭️  跳过已完成的第 ${ROUND} 轮"
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🎮 [Plan 4] 自对弈第 ${ROUND} / ${SELFPLAY_ROUNDS} 轮  (API Verifier)                 ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo "  当前 Challenger: ${CURRENT_CHALLENGER}"
    echo "  当前 Reviewer  : ${CURRENT_REVIEWER}"

    ROUND_DIR="${SELFPLAY_DIR}/round_${ROUND}"
    mkdir -p "${ROUND_DIR}/challenger" "${ROUND_DIR}/reviewer"

    # ════════════════════════════════════════════════════════════
    # Phase 0 — API Verifier 动态数据生成 (Plan 4 核心)
    # ════════════════════════════════════════════════════════════
    if [ -n "${RESUME_SKIP_PHASE}" ]; then
        echo "  ⏭️  断点恢复: 该轮 ${RESUME_SKIP_PHASE} 已完成，跳过"
    fi

    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phase0" ]; then
        echo ""
        echo "── Phase 0: API Verifier 数据生成 (Round ${ROUND}) ──"

        run_phase0_datagen_api \
            "${ROUND}" \
            "${CURRENT_CHALLENGER}" \
            "${CURRENT_REVIEWER}"

        cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "phase0",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "plan": "plan4_verifier_api",
  "api_model": "${VERIFIER_API_MODEL}",
  "timestamp": "$(date -Iseconds)"
}
EOF
    fi

    # 解析本轮数据路径
    ROUND_DATA_DIR="${DATA_DIR}/round_${ROUND}"
    DYNAMIC_CHALLENGER_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
    DYNAMIC_REVIEWER_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"
    if [ -n "${PHASE0_CHALLENGER_DATA}" ] && [ -f "${PHASE0_CHALLENGER_DATA}" ]; then
        DYNAMIC_CHALLENGER_DATA="${PHASE0_CHALLENGER_DATA}"
    fi
    if [ -n "${PHASE0_REVIEWER_DATA}" ] && [ -f "${PHASE0_REVIEWER_DATA}" ]; then
        DYNAMIC_REVIEWER_DATA="${PHASE0_REVIEWER_DATA}"
    fi

    # 安全检查
    if [ ! -f "${DYNAMIC_CHALLENGER_DATA}" ]; then
        echo "  ❌ Challenger 数据不存在，回退种子数据"
        DYNAMIC_CHALLENGER_DATA="${SEED_DATA}"
    fi
    if [ ! -f "${DYNAMIC_REVIEWER_DATA}" ]; then
        echo "  ❌ Reviewer 数据不存在，回退种子数据"
        DYNAMIC_REVIEWER_DATA="${SEED_DATA}"
    fi

    # ════════════════════════════════════════════════════════════
    # Phase A — Challenger GRPO (完全复用 v1)
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseA" ]; then
        echo ""
        echo "── Phase A: Challenger GRPO (Round ${ROUND}) ──"
        CHALLENGER_OUT="${ROUND_DIR}/challenger"
        CHALLENGER_LOG="${LOG_DIR}/round${ROUND}_challenger_$(date +%Y%m%d_%H%M%S).log"

        run_grpo_challenger \
            "${CURRENT_CHALLENGER}" \
            "${DYNAMIC_CHALLENGER_DATA}" \
            "${CHALLENGER_OUT}" \
            "${CHALLENGER_LOG}"

        CURRENT_CHALLENGER="${CHALLENGER_OUT}"

        cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "phaseA",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "plan": "plan4_verifier_api",
  "timestamp": "$(date -Iseconds)"
}
EOF
    else
        CHALLENGER_OUT="${ROUND_DIR}/challenger"
        if [ -d "${CHALLENGER_OUT}" ]; then
            CURRENT_CHALLENGER="${CHALLENGER_OUT}"
        fi
    fi

    # ════════════════════════════════════════════════════════════
    # Phase B — Reviewer SFT (完全复用 v1)
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseB" ]; then
        echo ""
        echo "── Phase B: Reviewer SFT (Round ${ROUND}) ──"
        REVIEWER_MIXED_DATA="${ROUND_DIR}/reviewer_mixed_round${ROUND}.parquet"
        REVIEWER_SFT_DATA="${ROUND_DIR}/reviewer_sft_round${ROUND}.parquet"
        REVIEWER_LORA_OUT="${ROUND_DIR}/reviewer_lora"
        REVIEWER_OUT="${ROUND_DIR}/reviewer"
        REVIEWER_LOG="${LOG_DIR}/round${ROUND}_reviewer_$(date +%Y%m%d_%H%M%S).log"

        # [1] 经验回放数据混合 (v1 脚本)
        echo "   -> 混合历史经验数据..."
        $PYTHON_EXEC "${V1_DIR}/mix_replay_data.py" \
            --dynamic_data "${DYNAMIC_REVIEWER_DATA}" \
            --seed_data "${SEED_DATA}" \
            --output_data "${REVIEWER_MIXED_DATA}" \
            --seed_ratio 2.0

        # [2] GRPO → SFT 格式转换 (v1 脚本)
        echo "   -> 转换数据格式为 SFT..."
        $PYTHON_EXEC "${V1_DIR}/convert_grpo_to_sft.py" \
            --input_data "${REVIEWER_MIXED_DATA}" \
            --output_data "${REVIEWER_SFT_DATA}"

        # [3] Reviewer LoRA SFT (v1 脚本)
        echo "   -> 启动 Reviewer LoRA SFT..."
        if [ "$N_GPUS" -ge 2 ]; then
            LAUNCH_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS"
        else
            LAUNCH_CMD="python"
        fi

        $LAUNCH_CMD "${V1_DIR}/../model_lora/train_reviewer_lora.py" \
            --model_path "${CURRENT_REVIEWER}" \
            --data_path "${REVIEWER_SFT_DATA}" \
            --output_dir "${REVIEWER_LORA_OUT}" \
            --lora_rank "${R_LORA_RANK}" \
            --lora_alpha "${R_LORA_RANK}" \
            --batch_size 4 \
            --gradient_accumulation_steps 4 \
            --num_epochs 1 \
            --learning_rate 5e-5 \
            --max_length 2048 \
            --n_devices $N_GPUS 2>&1 | tee "${REVIEWER_LOG}"

        # [4] 合并 LoRA (v1 脚本)
        echo "   -> 合并 LoRA..."
        $PYTHON_EXEC "${V1_DIR}/../model_lora/merge_lora.py" \
            --base_model "${CURRENT_REVIEWER}" \
            --lora_path "${REVIEWER_LORA_OUT}" \
            --output_path "${REVIEWER_OUT}"

        CURRENT_REVIEWER="${REVIEWER_OUT}"
    else
        REVIEWER_OUT="${ROUND_DIR}/reviewer"
        if [ -d "${REVIEWER_OUT}" ]; then
            CURRENT_REVIEWER="${REVIEWER_OUT}"
        fi
    fi

    # ════════════════════════════════════════════════════════════
    # 轮次总结
    # ════════════════════════════════════════════════════════════
    RESUME_SKIP_PHASE=""

    echo ""
    echo "── Round ${ROUND} 完成 ──"
    echo "   更新后 Challenger: ${CURRENT_CHALLENGER}"
    echo "   更新后 Reviewer  : ${CURRENT_REVIEWER}"

    cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "done",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "plan": "plan4_verifier_api",
  "api_model": "${VERIFIER_API_MODEL}",
  "timestamp": "$(date -Iseconds)"
}
EOF

done

# ─────────────────────────────────────────────────────────────────────────────────
# 9. 训练结束汇总
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Plan 4: API Verifier 自对弈训练全部完成！                           ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  最终 Challenger: ${CURRENT_CHALLENGER}"
echo "  最终 Reviewer  : ${CURRENT_REVIEWER}"
echo "  API Verifier   : ${VERIFIER_API_MODEL}"
echo "  日志目录       : ${LOG_DIR}"

FINAL_SUMMARY="${SELFPLAY_DIR}/final_models_plan4.txt"
cat > "${FINAL_SUMMARY}" << EOF
# Plan 4: API Verifier 对抗博弈自对弈最终模型路径
# 完成时间: $(date)
# 模型尺寸: ${MODEL_SIZE}
# NPU 卡数: ${N_GPUS}
# 训练轮次: ${SELFPLAY_ROUNDS}
# API 模型: ${VERIFIER_API_MODEL}
challenger_final: ${CURRENT_CHALLENGER}
reviewer_final  : ${CURRENT_REVIEWER}
EOF

echo "最终模型路径: ${FINAL_SUMMARY}"
