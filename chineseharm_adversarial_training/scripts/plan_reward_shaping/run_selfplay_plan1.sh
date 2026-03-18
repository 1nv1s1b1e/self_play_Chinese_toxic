#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  基线对照: v1 + 奖励信号修复 (Reward Fix Baseline)                               ║
# ║                                                                              ║
# ║  本方案是其他 3 个 Plan 的对照基线。                                            ║
# ║  唯一改动: Phase 0 + Phase A 使用修复版奖励函数 (逐样本对抗信号)           ║
# ║  其他超参和数据管线与 v1 完全相同。                                           ║
# ║                                                                              ║
# ║  用法:                                                                        ║
# ║    bash scripts/plan_reward_shaping/run_selfplay_plan1.sh                     ║
# ║    MODEL_SIZE=3B N_GPUS=4 bash scripts/plan_reward_shaping/run_selfplay_plan1.sh ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# 1. 环境初始化 (复用 v1)
# ─────────────────────────────────────────────────────────────────────────────────
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && echo "✓ ascend-toolkit 已加载"
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && echo "✓ nnal/atb 已加载"

PYTHON_EXEC="${PYTHON_EXEC:-/home/ma-user/.conda/envs/ssp_train/bin/python}"
echo "Python: $PYTHON_EXEC  ($($PYTHON_EXEC --version 2>&1))"

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export TASK_QUEUE_ENABLE=1

# ─────────────────────────────────────────────────────────────────────────────────
# 2. 路径设置
# ─────────────────────────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
PLAN1_DIR="$(cd "$(dirname "$0")" && pwd)"
V1_SCRIPT_DIR="$(dirname "$PLAN1_DIR")/rl_train"

MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-4}"
SELFPLAY_ROUNDS="${SELFPLAY_ROUNDS:-5}"
MAX_STEPS="${MAX_STEPS:-50}"
SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"

# Challenger GRPO 超参
C_LR="${C_LR:-5e-7}"
C_PER_DEVICE_BS="${C_PER_DEVICE_BS:-2}"
C_NUM_GEN="${C_NUM_GEN:-4}"
C_MAX_COMP_LEN="${C_MAX_COMP_LEN:-256}"
C_GRAD_ACCUM="${C_GRAD_ACCUM:-4}"

# Reviewer SFT 超参
R_LR="${R_LR:-5e-5}"

# 路径
SEED_DATA="${BASE_DIR}/prepared_data/rl/train_seed.parquet"
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_${MODEL_SIZE}"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"
VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_7B}"

# 输出 (与 v1 隔离，独立目录)
SELFPLAY_DIR="${BASE_DIR}/selfplay_plan1_reward_shaping/${MODEL_SIZE}_${N_GPUS}npu"
LOG_DIR="${BASE_DIR}/logs/plan1_reward_shaping_${MODEL_SIZE}"
DATA_DIR="${BASE_DIR}/selfplay_dynamic_data_plan1/${MODEL_SIZE}"

mkdir -p "${SELFPLAY_DIR}" "${LOG_DIR}" "${DATA_DIR}"

CURRENT_CHALLENGER="${CHALLENGER_INIT}"
CURRENT_REVIEWER="${REVIEWER_INIT}"

# ─────────────────────────────────────────────────────────────────────────────────
# 3. 断点续训
# ─────────────────────────────────────────────────────────────────────────────────
RESUME="${RESUME:-1}"
RESUME_FROM_ROUND=0
RESUME_SKIP_PHASE=""
PROGRESS_FILE="${SELFPLAY_DIR}/progress.json"

if [ "${RESUME}" = "1" ] && [ -f "${PROGRESS_FILE}" ]; then
    echo "── 🔄 检测到断点续训文件: ${PROGRESS_FILE} ──"
    RESUME_FROM_ROUND=$($PYTHON_EXEC -c "import json; print(json.load(open('${PROGRESS_FILE}')).get('last_completed_round', 0))")
    RESUME_PHASE=$($PYTHON_EXEC -c "import json; print(json.load(open('${PROGRESS_FILE}')).get('last_completed_phase', 'done'))")
    SAVED_CHALLENGER=$($PYTHON_EXEC -c "import json; print(json.load(open('${PROGRESS_FILE}')).get('current_challenger', ''))")
    SAVED_REVIEWER=$($PYTHON_EXEC -c "import json; print(json.load(open('${PROGRESS_FILE}')).get('current_reviewer', ''))")

    [ -n "${SAVED_CHALLENGER}" ] && [ -d "${SAVED_CHALLENGER}" ] && CURRENT_CHALLENGER="${SAVED_CHALLENGER}"
    [ -n "${SAVED_REVIEWER}" ] && [ -d "${SAVED_REVIEWER}" ] && CURRENT_REVIEWER="${SAVED_REVIEWER}"

    if [ "${RESUME_PHASE}" = "done" ]; then
        echo "  上次完整完成到第 ${RESUME_FROM_ROUND} 轮"
    else
        RESUME_FROM_ROUND=$((RESUME_FROM_ROUND - 1))
        RESUME_SKIP_PHASE="${RESUME_PHASE}"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 4. 打印参数
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   Plan 1: 真对抗奖励信号 (Reward Shaping)                             ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  模型尺寸      : ${MODEL_SIZE}"
echo "  NPU 卡数      : ${N_GPUS}"
echo "  自对弈轮次    : ${SELFPLAY_ROUNDS}"
echo "  每阶段步数    : ${MAX_STEPS}"
echo "  输出目录      : ${SELFPLAY_DIR}"
echo "  [Plan 1 改动] Challenger reward = quality_gate × (1 - reviewer_correct)"
echo "  [Plan 1 改动] Phase 0 注入逐样本 reviewer_fooled 信号"
echo ""

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: 保存进度
# ─────────────────────────────────────────────────────────────────────────────────
save_progress() {
    local ROUND="$1" PHASE="$2"
    cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "${PHASE}",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "plan": "plan1_reward_shaping",
  "timestamp": "$(date -Iseconds)"
}
EOF
}

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 主循环
# ─────────────────────────────────────────────────────────────────────────────────
for ROUND in $(seq 1 "${SELFPLAY_ROUNDS}"); do
    if [ "${ROUND}" -le "${RESUME_FROM_ROUND}" ]; then
        echo "  ⏭️  跳过已完成的第 ${ROUND} 轮"
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🎮 Plan 1 · 第 ${ROUND} / ${SELFPLAY_ROUNDS} 轮                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"

    ROUND_DIR="${SELFPLAY_DIR}/round_${ROUND}"
    ROUND_DATA_DIR="${DATA_DIR}/round_${ROUND}"
    mkdir -p "${ROUND_DIR}/challenger" "${ROUND_DIR}/reviewer" "${ROUND_DATA_DIR}"

    # ════════════════════════════════════════════════════════════
    # Phase 0 — 动态对抗数据生成 (使用修复版)
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phase0" ]; then
        echo "── Phase 0: 动态对抗数据生成 [修复版] (Round ${ROUND}) ──"
        LOG_FILE="${LOG_DIR}/round${ROUND}_phase0_$(date +%Y%m%d_%H%M%S).log"

        ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
        $PYTHON_EXEC "${V1_SCRIPT_DIR}/generate_dynamic_data_fixed.py" \
            --challenger_model "${CURRENT_CHALLENGER}" \
            --reviewer_model   "${CURRENT_REVIEWER}" \
            --verifier_model   "${VERIFIER_MODEL}" \
            --seed_data        "${SEED_DATA}" \
            --output_dir       "${ROUND_DATA_DIR}" \
            --round_idx        "${ROUND}" \
            --samples_per_cat  "${SAMPLES_PER_CAT}" \
            --batch_size       "${GEN_BATCH_SIZE}" \
            --num_npus         "${N_GPUS}" \
        2>&1 | tee "${LOG_FILE}"

        # 提取数据路径
        DYNAMIC_CHALLENGER_DATA=$(grep "^CHALLENGER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2- || echo "")
        DYNAMIC_REVIEWER_DATA=$(grep "^REVIEWER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2- || echo "")
        [ -z "${DYNAMIC_CHALLENGER_DATA}" ] && DYNAMIC_CHALLENGER_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
        [ -z "${DYNAMIC_REVIEWER_DATA}" ] && DYNAMIC_REVIEWER_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"

        save_progress "${ROUND}" "phase0"
    else
        DYNAMIC_CHALLENGER_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
        DYNAMIC_REVIEWER_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"
    fi

    # 安全检查
    [ ! -f "${DYNAMIC_CHALLENGER_DATA}" ] && DYNAMIC_CHALLENGER_DATA="${SEED_DATA}"
    [ ! -f "${DYNAMIC_REVIEWER_DATA}" ] && DYNAMIC_REVIEWER_DATA="${SEED_DATA}"

    # ════════════════════════════════════════════════════════════
    # Phase A — Challenger GRPO (使用修复版奖励信号)
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseA" ]; then
        echo "── Phase A: Challenger GRPO [修复版] (Round ${ROUND}) ──"
        CHALLENGER_OUT="${ROUND_DIR}/challenger"
        CHALLENGER_LOG="${LOG_DIR}/round${ROUND}_challenger_$(date +%Y%m%d_%H%M%S).log"

        $PYTHON_EXEC -m torch.distributed.run \
            --nproc_per_node="${N_GPUS}" \
            --master_port="29600" \
            "${V1_SCRIPT_DIR}/adversarial_trl_grpo_fixed.py" \
            --role                   "challenger" \
            --model_path             "${CURRENT_CHALLENGER}" \
            --dataset_path           "${DYNAMIC_CHALLENGER_DATA}" \
            --output_dir             "${CHALLENGER_OUT}" \
            --max_steps              "${MAX_STEPS}" \
            --save_steps             "${MAX_STEPS}" \
            --learning_rate          "${C_LR}" \
            --per_device_batch_size  "${C_PER_DEVICE_BS}" \
            --num_generations        "${C_NUM_GEN}" \
            --max_completion_length  "${C_MAX_COMP_LEN}" \
            --grad_accum             "${C_GRAD_ACCUM}" \
            --use_selfplay \
            --deepspeed "${V1_SCRIPT_DIR}/ds_zero2.json" \
            2>&1 | tee "${CHALLENGER_LOG}"

        CURRENT_CHALLENGER="${CHALLENGER_OUT}"
        save_progress "${ROUND}" "phaseA"
    fi

    # ════════════════════════════════════════════════════════════
    # Phase B — Reviewer SFT (与 v1 完全相同，复用 v1 脚本)
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseB" ]; then
        echo "── Phase B: Reviewer SFT (Round ${ROUND}) ──"
        REVIEWER_MIXED_DATA="${ROUND_DIR}/reviewer_mixed_round${ROUND}.parquet"
        REVIEWER_SFT_DATA="${ROUND_DIR}/reviewer_sft_round${ROUND}.parquet"
        REVIEWER_LORA_OUT="${ROUND_DIR}/reviewer_lora"
        REVIEWER_OUT="${ROUND_DIR}/reviewer"
        REVIEWER_LOG="${LOG_DIR}/round${ROUND}_reviewer_$(date +%Y%m%d_%H%M%S).log"

        # [1] 经验回放混合 (复用 v1)
        $PYTHON_EXEC "${V1_SCRIPT_DIR}/mix_replay_data.py" \
            --dynamic_data "${DYNAMIC_REVIEWER_DATA}" \
            --seed_data "${SEED_DATA}" \
            --output_data "${REVIEWER_MIXED_DATA}" \
            --seed_ratio 2.0

        # [2] GRPO → SFT 格式转换 (复用 v1)
        $PYTHON_EXEC "${V1_SCRIPT_DIR}/convert_grpo_to_sft.py" \
            --input_data "${REVIEWER_MIXED_DATA}" \
            --output_data "${REVIEWER_SFT_DATA}"

        # [3] Reviewer LoRA SFT (复用 v1)
        if [ "$N_GPUS" -ge 2 ]; then
            LAUNCH_CMD="$PYTHON_EXEC -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS"
        else
            LAUNCH_CMD="$PYTHON_EXEC"
        fi

        $LAUNCH_CMD "${V1_SCRIPT_DIR}/../model_lora/train_reviewer_lora.py" \
            --model_path "${CURRENT_REVIEWER}" \
            --data_path "${REVIEWER_SFT_DATA}" \
            --output_dir "${REVIEWER_LORA_OUT}" \
            --lora_rank 32 --lora_alpha 32 \
            --batch_size 4 --gradient_accumulation_steps 4 \
            --num_epochs 1 --learning_rate "${R_LR}" \
            --max_length 2048 --n_devices $N_GPUS \
            2>&1 | tee "${REVIEWER_LOG}"

        # [4] LoRA 合并
        $PYTHON_EXEC "${V1_SCRIPT_DIR}/../model_lora/merge_lora.py" \
            --base_model "${CURRENT_REVIEWER}" \
            --lora_path "${REVIEWER_LORA_OUT}" \
            --output_path "${REVIEWER_OUT}"

        CURRENT_REVIEWER="${REVIEWER_OUT}"
    fi

    RESUME_SKIP_PHASE=""
    save_progress "${ROUND}" "done"
    echo "── Plan 1 · Round ${ROUND} 完成 ──"
done

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Plan 1 (真对抗奖励) 训练完成！                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  最终 Challenger: ${CURRENT_CHALLENGER}"
echo "  最终 Reviewer  : ${CURRENT_REVIEWER}"
