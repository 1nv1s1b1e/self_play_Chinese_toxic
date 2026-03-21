#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Self-Play 对抗训练 (昇腾 910B)                                              ║
# ║                                                                              ║
# ║  每个 Step 的流程:                                                            ║
# ║    1. Challenger 生成 ~64 条对抗文本                                           ║
# ║    2. Reviewer 评估，计算 1-acc 奖励                                           ║
# ║    3. Challenger GRPO 更新 (奖励 = 骗过 Reviewer)                              ║
# ║    4. Reviewer   GRPO 更新 (奖励 = 正确分类)                                   ║
# ║                                                                              ║
# ║  用法:                                                                        ║
# ║    bash run_selfplay.sh                                                       ║
# ║    TOTAL_STEPS=100 bash run_selfplay.sh                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────────
# 1. 昇腾 CANN 环境初始化
# ─────────────────────────────────────────────────────────────────────────────────
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && echo "✓ ascend-toolkit 已加载"
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && echo "✓ nnal/atb 已加载"

# ─────────────────────────────────────────────────────────────────────────────────
# 2. Python 解释器 & NPU 环境变量
# ─────────────────────────────────────────────────────────────────────────────────
PYTHON_EXEC="${PYTHON_EXEC:-/home/ma-user/.conda/envs/ssp_train/bin/python}"
echo "Python: $PYTHON_EXEC  ($($PYTHON_EXEC --version 2>&1))"

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export TASK_QUEUE_ENABLE=1

# ─────────────────────────────────────────────────────────────────────────────────
# 3. 训练超参数
# ─────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"              # 自对弈总步数
SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-20}"      # 每类有毒生成样本数 (5有毒类×20=100条)
NONTOXIC_SAMPLES="${NONTOXIC_SAMPLES:-89}"    # 无毒样本数, 使有毒:无毒≈53:47 匹配 benchmark
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"

CHECK_INTERVAL="${CHECK_INTERVAL:-1}"         # 每 N 步做一次评估检查 (默认每步都评估)

# GRPO 训练: 每个 step 训 1 个 epoch（TRL 自动根据数据量计算步数）
GRPO_EPOCHS="${GRPO_EPOCHS:-1}"             # 1 epoch/步：避免过拟合当前步数据，靠多步迭代提升

# Challenger GRPO 超参
C_LR="${C_LR:-5e-7}"
C_PER_DEVICE_BS="${C_PER_DEVICE_BS:-2}"
C_NUM_GEN="${C_NUM_GEN:-4}"
C_MAX_COMP_LEN="${C_MAX_COMP_LEN:-128}"
C_GRAD_ACCUM="${C_GRAD_ACCUM:-4}"

# Reviewer GRPO 超参
R_LR="${R_LR:-5e-7}"
R_PER_DEVICE_BS="${R_PER_DEVICE_BS:-4}"
R_NUM_GEN="${R_NUM_GEN:-4}"
R_MAX_COMP_LEN="${R_MAX_COMP_LEN:-64}"
R_GRAD_ACCUM="${R_GRAD_ACCUM:-4}"

# 在线 Reviewer 推理 batch size (Challenger GRPO 用)
REVIEWER_BATCH_SIZE="${REVIEWER_BATCH_SIZE:-8}"

# Reviewer 数据混合（混合种子数据防止遗忘）
# mix_ratio=0.3: 对抗数据 70% + 原始种子 30%
# nontoxic_boost=1.0: 不再人为过采样, 保持 benchmark 原始分布 (~47% 无毒)
REVIEWER_MIX_RATIO="${REVIEWER_MIX_RATIO:-0.3}"
REVIEWER_NONTOXIC_BOOST="${REVIEWER_NONTOXIC_BOOST:-1.0}"

# 评估数据（每 CHECK_INTERVAL 步评估一次 Reviewer）
REVIEWER_EVAL_DATA="${REVIEWER_EVAL_DATA:-${BASE_DIR}/split_data/test.json}"

# Verifier 后端
VERIFIER_BACKEND="${VERIFIER_BACKEND:-local}"
export VERIFIER_API_KEY="${VERIFIER_API_KEY:-}"
export VERIFIER_API_BASE="${VERIFIER_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export VERIFIER_API_MODEL="${VERIFIER_API_MODEL:-qwen-plus}"

if [ "${VERIFIER_BACKEND}" != "local" ] && [ -z "${VERIFIER_API_KEY}" ]; then
    echo "⚠️  VERIFIER_BACKEND=${VERIFIER_BACKEND} 但 VERIFIER_API_KEY 未设置，降级为 local"
    VERIFIER_BACKEND="local"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 4. 路径配置
# ─────────────────────────────────────────────────────────────────────────────────
SEED_DATA="${BASE_DIR}/prepared_data/rl/train_seed.parquet"
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_${MODEL_SIZE}"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"
VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_7B}"

SELFPLAY_DIR="${BASE_DIR}/selfplay_integrated/${MODEL_SIZE}_${N_GPUS}npu"
LOG_DIR="${BASE_DIR}/logs/selfplay_integrated_${MODEL_SIZE}"
DATA_DIR="${BASE_DIR}/selfplay_integrated_data/${MODEL_SIZE}"
LATEST_DIR="${SELFPLAY_DIR}/latest"
BEST_DIR="${SELFPLAY_DIR}/best"  # 评估最优的模型

EVAL_HISTORY_DIR="${SELFPLAY_DIR}/eval_history"
mkdir -p "${SELFPLAY_DIR}" "${LOG_DIR}" "${DATA_DIR}" "${LATEST_DIR}" "${BEST_DIR}" "${EVAL_HISTORY_DIR}"

# 全局指标日志（每步追加一行 JSON，便于绘图分析）
METRICS_LOG="${SELFPLAY_DIR}/metrics.jsonl"

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 断点续训
# ─────────────────────────────────────────────────────────────────────────────────
RESUME_FROM=0
RESUME_SKIP_PHASE=""
PROGRESS_FILE="${SELFPLAY_DIR}/progress.json"

if [ "${RESUME}" = "1" ] && [ -f "${PROGRESS_FILE}" ]; then
    echo ""
    echo "── 🔄 检测到断点续训: ${PROGRESS_FILE} ──"
    RESUME_FROM=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f: d = json.load(f)
print(d.get('last_completed_step', 0))
")
    RESUME_PHASE=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f: d = json.load(f)
print(d.get('last_completed_phase', 'done'))
")
    SAVED_C=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f: d = json.load(f)
print(d.get('current_challenger', ''))
")
    SAVED_R=$($PYTHON_EXEC -c "
import json
with open('${PROGRESS_FILE}') as f: d = json.load(f)
print(d.get('current_reviewer', ''))
")
    [ -n "${SAVED_C}" ] && [ -d "${SAVED_C}" ] && CURRENT_CHALLENGER="${SAVED_C}"
    [ -n "${SAVED_R}" ] && [ -d "${SAVED_R}" ] && CURRENT_REVIEWER="${SAVED_R}"

    if [ "${RESUME_PHASE}" = "done" ]; then
        echo "  上次完成到第 ${RESUME_FROM} 步，从第 $((RESUME_FROM + 1)) 步开始"
    else
        echo "  第 ${RESUME_FROM} 步在 ${RESUME_PHASE} 阶段中断"
        RESUME_FROM=$((RESUME_FROM - 1))
        RESUME_SKIP_PHASE="${RESUME_PHASE}"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 6. 打印参数
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Self-Play 对抗训练 · 昇腾 910B                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  模型       : ${MODEL_SIZE} (${N_GPUS} NPUs)"
echo "  总步数     : ${TOTAL_STEPS} (每 ${CHECK_INTERVAL} 步检查)"
echo "  每步样本   : $((SAMPLES_PER_CAT * 5 + NONTOXIC_SAMPLES)) 条 (有毒${SAMPLES_PER_CAT}/类×5 + 无毒${NONTOXIC_SAMPLES})"
echo "  GRPO       : ${GRPO_EPOCHS} epoch, Top-K=${C_NUM_GEN}"
echo "  Challenger : ${CHALLENGER_INIT}"
echo "  Reviewer   : ${REVIEWER_INIT}"
echo "  种子数据   : ${SEED_DATA}"
echo "  输出目录   : ${SELFPLAY_DIR}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────────
# 7. 初始化模型路径
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${RESUME_FROM}" -eq 0 ]; then
    CURRENT_CHALLENGER="${CHALLENGER_INIT}"
    CURRENT_REVIEWER="${REVIEWER_INIT}"
fi
# 安全网
[ -z "${CURRENT_CHALLENGER}" ] || [ ! -d "${CURRENT_CHALLENGER}" ] && CURRENT_CHALLENGER="${CHALLENGER_INIT}"
[ -z "${CURRENT_REVIEWER}" ]   || [ ! -d "${CURRENT_REVIEWER}" ]   && CURRENT_REVIEWER="${REVIEWER_INIT}"

# ═════════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════════

save_progress() {
    local STEP="$1" PHASE="$2"
    cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_step": ${STEP},
  "last_completed_phase": "${PHASE}",
  "total_steps": ${TOTAL_STEPS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "timestamp": "$(date -Iseconds)"
}
EOF
}

update_latest() {
    local STEP="$1"
    cat > "${LATEST_DIR}/latest_paths.json" << EOF
{
  "step": ${STEP},
  "timestamp": "$(date -Iseconds)",
  "challenger": "${CURRENT_CHALLENGER}",
  "reviewer": "${CURRENT_REVIEWER}"
}
EOF
    echo "${STEP}" > "${LATEST_DIR}/latest_step.txt"
    echo "${CURRENT_CHALLENGER}" > "${LATEST_DIR}/challenger_latest.txt"
    echo "${CURRENT_REVIEWER}" > "${LATEST_DIR}/reviewer_latest.txt"
}

# 清理旧步骤（保留最近 3 步 + best，历史数据 sample_rewards 保留供错题回放）
cleanup_prev_step() {
    local CURRENT="$1"
    local KEEP=3
    local OLD=$((CURRENT - KEEP))
    [ "${OLD}" -lt 1 ] && return
    # 删除旧模型 checkpoint（占空间大），但保留 DATA_DIR 中的 sample_rewards parquet
    rm -rf "${SELFPLAY_DIR}/step_${OLD}" 2>/dev/null || true
}

# 保存最优模型
save_best_model() {
    echo "   🏆 保存当前模型为 Best"
    rm -rf "${BEST_DIR}/challenger" "${BEST_DIR}/reviewer" 2>/dev/null || true
    cp -r "${CURRENT_CHALLENGER}" "${BEST_DIR}/challenger"
    cp -r "${CURRENT_REVIEWER}"   "${BEST_DIR}/reviewer"
    echo "${CURRENT_STEP}" > "${BEST_DIR}/best_step.txt"
}

# 数据生成: Challenger 生成 → Reviewer 评估 → 计算 1-acc 奖励 → 构建 GRPO parquet
run_datagen() {
    local STEP="$1"
    local CHALLENGER_PATH="$2"
    local REVIEWER_PATH="$3"
    local STEP_DATA_DIR="${DATA_DIR}/step_${STEP}"
    local LOG_FILE="${LOG_DIR}/step${STEP}_datagen_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "${STEP_DATA_DIR}"

    echo ""
    echo "  ▶ [$(date +%H:%M:%S)] 数据生成 (Step ${STEP})"
    echo "     Challenger : ${CHALLENGER_PATH}"
    echo "     Reviewer   : ${REVIEWER_PATH}"

    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${SCRIPT_DIR}/generate_dynamic_data.py" \
        --challenger_model "${CHALLENGER_PATH}" \
        --reviewer_model   "${REVIEWER_PATH}" \
        --verifier_model   "${VERIFIER_MODEL}" \
        --seed_data        "${SEED_DATA}" \
        --output_dir       "${STEP_DATA_DIR}" \
        --round_idx        "${STEP}" \
        --samples_per_cat  "${SAMPLES_PER_CAT}" \
        --nontoxic_samples "${NONTOXIC_SAMPLES}" \
        --batch_size       "${GEN_BATCH_SIZE}" \
        --num_npus         "${N_GPUS}" \
        --verifier_backend "${VERIFIER_BACKEND}" \
        --verifier_api_model "${VERIFIER_API_MODEL}" \
        --reviewer_mix_ratio    "${REVIEWER_MIX_RATIO}" \
        --reviewer_nontoxic_boost "${REVIEWER_NONTOXIC_BOOST}" \
        --reviewer_hard_boost 2 \
        --history_hard_dir "${DATA_DIR}" \
        --no_rejection_sampling \
    2>&1 | tee "${LOG_FILE}"

    local C_DATA=$(grep "^CHALLENGER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2- || echo "")
    local R_DATA=$(grep "^REVIEWER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2- || echo "")
    [ -z "${C_DATA}" ] && C_DATA="${STEP_DATA_DIR}/challenger_grpo_round${STEP}.parquet"
    [ -z "${R_DATA}" ] && R_DATA="${STEP_DATA_DIR}/reviewer_grpo_round${STEP}.parquet"

    echo "  ✓ [$(date +%H:%M:%S)] 数据生成完成"

    # 从日志中提取本轮 ASR / fooling_rate 等指标
    STEP_ASR=$(grep -oP 'fooling_rate[=: ]+\K[0-9.]+' "${LOG_FILE}" | tail -1 || echo "")
    STEP_GEN_COUNT=$(grep -oP '生成.*?(\d+).*?条' "${LOG_FILE}" | grep -oP '\d+' | tail -1 || echo "")

    DATAGEN_CHALLENGER_DATA="${C_DATA}"
    DATAGEN_REVIEWER_DATA="${R_DATA}"
}

# GRPO 训练
run_grpo() {
    local ROLE="$1"
    local MODEL_PATH="$2"
    local DATASET_PATH="$3"
    local OUTPUT_PATH="$4"
    local LOG_FILE="$5"
    local MASTER_PORT="${6:-29600}"
    local REVIEWER_PATH="${7:-}"

    if [ "${ROLE}" = "challenger" ]; then
        local LR="${C_LR}"
        local PER_BS="${C_PER_DEVICE_BS}"
        local NUM_GEN="${C_NUM_GEN}"
        local MAX_COMP="${C_MAX_COMP_LEN}"
        local GRAD_ACC="${C_GRAD_ACCUM}"
    else
        local LR="${R_LR}"
        local PER_BS="${R_PER_DEVICE_BS}"
        local NUM_GEN="${R_NUM_GEN}"
        local MAX_COMP="${R_MAX_COMP_LEN}"
        local GRAD_ACC="${R_GRAD_ACCUM}"
    fi

    echo "  ▶▶ [$(date +%H:%M:%S)] ${ROLE} GRPO (${GRPO_EPOCHS} epoch, lr=${LR}, bs=${PER_BS}, gen=${NUM_GEN}, grad_acc=${GRAD_ACC})"

    $PYTHON_EXEC -m torch.distributed.run \
        --nproc_per_node="${N_GPUS}" \
        --master_port="${MASTER_PORT}" \
        "${SCRIPT_DIR}/adversarial_trl_grpo.py" \
        --role                   "${ROLE}" \
        --model_path             "${MODEL_PATH}" \
        --dataset_path           "${DATASET_PATH}" \
        --output_dir             "${OUTPUT_PATH}" \
        --max_steps              0 \
        --num_epochs             "${GRPO_EPOCHS}" \
        --learning_rate          "${LR}" \
        --per_device_batch_size  "${PER_BS}" \
        --num_generations        "${NUM_GEN}" \
        --max_completion_length  "${MAX_COMP}" \
        --grad_accum             "${GRAD_ACC}" \
        --selfplay_step          "${CURRENT_STEP:-0}" \
        $([ "${ROLE}" = "challenger" ] && [ -n "${REVIEWER_PATH}" ] && echo "--reviewer_model_path ${REVIEWER_PATH} --reviewer_batch_size ${REVIEWER_BATCH_SIZE}") \
        --deepspeed "${SCRIPT_DIR}/ds_zero2.json" \
        2>&1 | tee "${LOG_FILE}"

    if [ ! -f "${OUTPUT_PATH}/training_done.txt" ]; then
        echo "  ⚠️  ${ROLE} GRPO 可能未完成，检查: ${LOG_FILE}"
    else
        echo "  ✓ [$(date +%H:%M:%S)] ${ROLE} GRPO 完成"
    fi
}

# Reviewer 门控评估
evaluate_reviewer_metrics() {
    local MODEL_PATH="$1"
    local STEP="$2"
    local EVAL_OUT_DIR="${SELFPLAY_DIR}/step_${STEP}/eval_gate"
    mkdir -p "${EVAL_OUT_DIR}"

    # 评估用单卡 + enforce_eager 确保结果确定性
    ASCEND_RT_VISIBLE_DEVICES=0 \
    $PYTHON_EXEC "${SCRIPT_DIR}/../model_eval/batch_eval_npu_vllm.py" \
        --data_path "${REVIEWER_EVAL_DATA}" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${EVAL_OUT_DIR}" \
        --num_npus 1 \
        --tag "step${STEP}_gate" \
        --batch_size 128 \
        --enforce_eager \
        >/dev/null 2>&1

    local MODEL_NAME=$(basename "${MODEL_PATH}")
    local EVAL_JSON="${EVAL_OUT_DIR}/eval_vllm_npu_${MODEL_NAME}_step${STEP}_gate.json"
    [ ! -f "${EVAL_JSON}" ] && { echo "EVAL_FAILED"; return; }

    # 持久保存 eval JSON（含完整推理结果），不会被 cleanup 删除
    cp "${EVAL_JSON}" "${EVAL_HISTORY_DIR}/eval_step${STEP}.json" 2>/dev/null || true

    $PYTHON_EXEC - <<PY
import json
with open(r"${EVAL_JSON}", "r", encoding="utf-8") as f:
    d = json.load(f)
m = d.get("metrics", {})
acc = float(m.get("overall_accuracy", 0.0))
macro_f1 = float(m.get("macro_f1", 0.0))
nt_rec = float(m.get("category_metrics", {}).get("无毒", {}).get("recall", 0.0)) * 100.0
print(f"{acc:.6f},{macro_f1:.6f},{nt_rec:.6f}")
PY
}

# Reviewer 评估基线 — 从历史 eval_history 中恢复真正的最高 acc，避免覆盖 best
BEST_REVIEWER_ACC="0.0"

# 先从历史评估记录中找到真正的最高 acc
if [ -d "${EVAL_HISTORY_DIR}" ]; then
    HIST_BEST=$($PYTHON_EXEC - "${EVAL_HISTORY_DIR}" <<'HIST_PY'
import json, glob, sys, os
best_acc = 0.0
for f in glob.glob(os.path.join(sys.argv[1], "eval_step*.json")):
    try:
        with open(f) as fh:
            m = json.load(fh).get("metrics", {})
        acc = float(m.get("overall_accuracy", 0.0))
        if acc > best_acc:
            best_acc = acc
    except Exception:
        pass
print(f"{best_acc:.6f}")
HIST_PY
    )
    if [ -n "${HIST_BEST}" ] && [ "${HIST_BEST}" != "0.000000" ]; then
        BEST_REVIEWER_ACC="${HIST_BEST}"
        echo "  ▶ 从历史评估恢复 best acc: ${BEST_REVIEWER_ACC}"
    fi
fi

# 如果没有历史记录（首次运行），才做初始评估并保存为 best
if [ "${BEST_REVIEWER_ACC}" = "0.0" ] || [ "${BEST_REVIEWER_ACC}" = "0.000000" ]; then
    if [ -f "${REVIEWER_EVAL_DATA}" ]; then
        echo "  ▶ 首次运行，初始化 Reviewer 评估基线..."
        mkdir -p "${SELFPLAY_DIR}/eval_init"
        INIT_METRICS=$(evaluate_reviewer_metrics "${CURRENT_REVIEWER}" "0")
        if [ "${INIT_METRICS}" != "EVAL_FAILED" ]; then
            BEST_REVIEWER_ACC=$(echo "${INIT_METRICS}" | cut -d, -f1)
            INIT_MACRO=$(echo "${INIT_METRICS}" | cut -d, -f2)
            INIT_NT=$(echo "${INIT_METRICS}" | cut -d, -f3)
            echo "     基线: acc=${BEST_REVIEWER_ACC}, macro_f1=${INIT_MACRO}, nt_rec=${INIT_NT}%"
            save_best_model
        else
            echo "     ⚠️ 基线评估失败，跳过"
        fi
    else
        echo "  ⚠️ 找不到评估集: ${REVIEWER_EVAL_DATA}，跳过评估"
    fi
else
    echo "  ▶ best/ 模型已存在，不覆盖 (历史最优 acc=${BEST_REVIEWER_ACC})"
fi

# ═════════════════════════════════════════════════════════════════════════════════
# 主循环: 每个 Step = 生成数据 → Challenger GRPO → Reviewer GRPO
# ═════════════════════════════════════════════════════════════════════════════════

for STEP in $(seq 1 "${TOTAL_STEPS}"); do
    CURRENT_STEP="${STEP}"

    if [ "${STEP}" -le "${RESUME_FROM}" ]; then
        echo "  ⏭️  跳过已完成的第 ${STEP} 步"
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🎮 Self-Play Step ${STEP} / ${TOTAL_STEPS}                                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo "  Challenger: ${CURRENT_CHALLENGER}"
    echo "  Reviewer  : ${CURRENT_REVIEWER}"

    STEP_DIR="${SELFPLAY_DIR}/step_${STEP}"
    STEP_DATA_DIR="${DATA_DIR}/step_${STEP}"
    mkdir -p "${STEP_DIR}/challenger" "${STEP_DIR}/reviewer" "${STEP_DATA_DIR}"

    if [ -n "${RESUME_SKIP_PHASE}" ]; then
        echo "  ⏭️  断点恢复: ${RESUME_SKIP_PHASE} 已完成，跳过"
    fi

    # ── 1. 数据生成: Challenger生成 → Reviewer评估 → 计算 1-acc ──
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "datagen" ]; then
        run_datagen "${STEP}" "${CURRENT_CHALLENGER}" "${CURRENT_REVIEWER}"
        save_progress "${STEP}" "datagen"
    fi

    # 解析数据路径
    C_DATA="${STEP_DATA_DIR}/challenger_grpo_round${STEP}.parquet"
    R_DATA="${STEP_DATA_DIR}/reviewer_grpo_round${STEP}.parquet"
    [ -n "${DATAGEN_CHALLENGER_DATA}" ] && [ -f "${DATAGEN_CHALLENGER_DATA}" ] && C_DATA="${DATAGEN_CHALLENGER_DATA}"
    [ -n "${DATAGEN_REVIEWER_DATA}" ]   && [ -f "${DATAGEN_REVIEWER_DATA}" ]   && R_DATA="${DATAGEN_REVIEWER_DATA}"
    [ ! -f "${C_DATA}" ] && { echo "  ⚠️ Challenger 数据缺失，回退种子数据"; C_DATA="${SEED_DATA}"; }
    [ ! -f "${R_DATA}" ] && { echo "  ⚠️ Reviewer 数据缺失，回退种子数据";   R_DATA="${SEED_DATA}"; }

    # ── 2. Challenger GRPO 更新 ──
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "challenger" ]; then
        echo ""
        echo "── Challenger GRPO (Step ${STEP}) ──"
        # LoRA 训练下显存充足，启用在线 Reviewer 实时评估每条新 completion
        # 这样 GRPO 的 K 条 completion 得到不同 reward → 有效的 advantage 信号
        run_grpo \
            "challenger" \
            "${CURRENT_CHALLENGER}" \
            "${C_DATA}" \
            "${STEP_DIR}/challenger" \
            "${LOG_DIR}/step${STEP}_challenger_$(date +%Y%m%d_%H%M%S).log" \
            "29600" \
            "${CURRENT_REVIEWER}"

        CURRENT_CHALLENGER="${STEP_DIR}/challenger"
        save_progress "${STEP}" "challenger"
    else
        [ -d "${STEP_DIR}/challenger" ] && CURRENT_CHALLENGER="${STEP_DIR}/challenger"
    fi

    # ── 3. Reviewer GRPO 更新 ──
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "reviewer" ]; then
        echo ""
        echo "── Reviewer GRPO (Step ${STEP}) ──"
        run_grpo \
            "reviewer" \
            "${CURRENT_REVIEWER}" \
            "${R_DATA}" \
            "${STEP_DIR}/reviewer" \
            "${LOG_DIR}/step${STEP}_reviewer_$(date +%Y%m%d_%H%M%S).log" \
            "29601"

        CURRENT_REVIEWER="${STEP_DIR}/reviewer"

        # ── 每 CHECK_INTERVAL 步评估一次（仅记录指标，不回滚）──
        if [ $((STEP % CHECK_INTERVAL)) -eq 0 ] || [ "${STEP}" -eq "${TOTAL_STEPS}" ]; then
            echo "   📊 Step ${STEP}: 评估 Reviewer"
            METRICS=$(evaluate_reviewer_metrics "${CURRENT_REVIEWER}" "${STEP}")
            if [ "${METRICS}" != "EVAL_FAILED" ]; then
                CAND_ACC=$(echo "${METRICS}" | cut -d, -f1)
                CAND_MACRO=$(echo "${METRICS}" | cut -d, -f2)
                CAND_NT=$(echo "${METRICS}" | cut -d, -f3)
                echo "   📊 结果: acc=${CAND_ACC}, macro_f1=${CAND_MACRO}, nt_recall=${CAND_NT}%"

                # 仅记录最优，不回滚 — 始终用本步模型继续训练
                IS_BEST=$($PYTHON_EXEC -c "print('1' if float('${CAND_ACC}') > float('${BEST_REVIEWER_ACC}') else '0')")
                if [ "${IS_BEST}" = "1" ]; then
                    BEST_REVIEWER_ACC="${CAND_ACC}"
                    BEST_REVIEWER_MACRO_F1="${CAND_MACRO}"
                    BEST_REVIEWER_NT_REC="${CAND_NT}"
                    save_best_model
                else
                    echo "   📊 未超过 best (${BEST_REVIEWER_ACC})，继续使用本步模型"
                fi
            else
                echo "   ⚠️ 评估失败，跳过"
            fi
        fi

        save_progress "${STEP}" "reviewer"
    else
        [ -d "${STEP_DIR}/reviewer" ] && CURRENT_REVIEWER="${STEP_DIR}/reviewer"
    fi

    # ── Step 完成 ──
    RESUME_SKIP_PHASE=""
    save_progress "${STEP}" "done"
    update_latest "${STEP}"
    cleanup_prev_step "${STEP}"

    # ── 追加本步指标到 metrics.jsonl ──
    _M_STEP="${STEP}" \
    _M_ASR="${STEP_ASR:-}" \
    _M_GEN="${STEP_GEN_COUNT:-}" \
    _M_ACC="${CAND_ACC:-}" \
    _M_F1="${CAND_MACRO:-}" \
    _M_BEST="${BEST_REVIEWER_ACC}" \
    _M_C="${CURRENT_CHALLENGER}" \
    _M_R="${CURRENT_REVIEWER}" \
    _M_LOG="${METRICS_LOG}" \
    $PYTHON_EXEC << 'METRICS_PY'
import json, datetime, os

def safe_float(v):
    try:
        return float(v) if v else None
    except (ValueError, TypeError):
        return None

def safe_int(v):
    try:
        return int(v) if v else None
    except (ValueError, TypeError):
        return None

entry = {
    "step": safe_int(os.environ.get("_M_STEP")),
    "timestamp": datetime.datetime.now().isoformat(),
    "asr": safe_float(os.environ.get("_M_ASR")),
    "gen_count": safe_int(os.environ.get("_M_GEN")),
    "reviewer_acc": safe_float(os.environ.get("_M_ACC")),
    "reviewer_macro_f1": safe_float(os.environ.get("_M_F1")),
    "best_acc": safe_float(os.environ.get("_M_BEST")),
    "challenger": os.environ.get("_M_C", ""),
    "reviewer": os.environ.get("_M_R", ""),
}
with open(os.environ["_M_LOG"], "a") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
METRICS_PY

    echo ""
    echo "── Step ${STEP} 完成 ──"
    echo "   Challenger: ${CURRENT_CHALLENGER}"
    echo "   Reviewer  : ${CURRENT_REVIEWER}"
done

# ─────────────────────────────────────────────────────────────────────────────────
# 训练完成
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ Self-Play 训练完成！                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  最终 Challenger: ${CURRENT_CHALLENGER}"
echo "  最终 Reviewer  : ${CURRENT_REVIEWER}"
echo "  日志目录       : ${LOG_DIR}"

cat > "${SELFPLAY_DIR}/final_models.txt" << EOF
# Self-Play 最终模型路径
# 完成时间: $(date)
# 总步数: ${TOTAL_STEPS}
challenger_final: ${CURRENT_CHALLENGER}
reviewer_final  : ${CURRENT_REVIEWER}
EOF

echo "最终模型路径: ${SELFPLAY_DIR}/final_models.txt"
