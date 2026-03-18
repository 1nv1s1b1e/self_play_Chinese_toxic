#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  v2 对抗博弈自对弈 RL 训练主循环 — Stackelberg (昇腾 910B)                       ║
# ║                                                                              ║
# ║  架构改进 (相比 v1):                                                            ║
# ║    ① 外部 API Verifier 替代冻结本地模型 → 更准确的验证信号                          ║
# ║    ② 迭代答案收敛 → 更可靠的 Gold Answer                                         ║
# ║    ③ PDF 对齐奖励设计: R_c=2(欺骗)/R_r=2(正确) + quality_gate                    ║
# ║    ④ 课程学习: 渐进式难度过滤 + 动态 KL 权重 + 熵正则                               ║
# ║    ⑤ 多源集成验证: Qwen+DeepSeek 投票降低标签噪声                                  ║
# ║                                                                              ║
# ║  三阶段流程 (同 v1):                                                             ║
# ║    Phase 0: v2 动态数据生成 (API Verifier + 迭代收敛 + 课程打分)                    ║
# ║    Phase A: Challenger GRPO (v2 奖励 + 课程 KL 权重)                              ║
# ║    Phase B: Reviewer SFT + LoRA Merge (v2 数据 + 经验回放)                        ║
# ║                                                                              ║
# ║  用法:                                                                         ║
# ║    bash run_selfplay_v2_npu.sh                                                 ║
# ║    MODEL_SIZE=7B N_GPUS=4 SELFPLAY_ROUNDS=5 bash run_selfplay_v2_npu.sh        ║
# ║    API_PROVIDERS="qwen deepseek" bash run_selfplay_v2_npu.sh                   ║
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
# 4. 训练超参数（可通过环境变量覆盖）
# ─────────────────────────────────────────────────────────────────────────────────
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
V1_SCRIPT_DIR="${SCRIPT_DIR}/../rl_train"   # v1 目录（复用 adversarial_trl_grpo.py 等）

MODEL_SIZE="${MODEL_SIZE:-3B}"           # 0.5B / 1.5B / 3B / 7B / 14B
N_GPUS="${N_GPUS:-4}"                    # 每阶段使用的 NPU 卡数
SELFPLAY_ROUNDS="${SELFPLAY_ROUNDS:-5}"  # 自对弈总轮次
MAX_STEPS="${MAX_STEPS:-50}"             # 每个 GRPO 阶段的训练步数
SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-256}" # Phase 0 每类别生成样本数
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"    # Phase 0 推理 batch size
RESUME="${RESUME:-1}"                    # 1=自动断点续训, 0=强制从头

# ── v2 新增参数 ────────────────────────────────────────────────────
API_PROVIDERS="${API_PROVIDERS:-qwen}"                     # 空格分隔: "qwen deepseek openai"
USE_API_VERIFIER="${USE_API_VERIFIER:-1}"                  # 1=API Verifier, 0=本地模型
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"                      # 迭代收敛最大轮次
CONVERGENCE_K="${CONVERGENCE_K:-3}"                        # 连续 K 次答案一致视为收敛

# Challenger GRPO 超参
C_LR="${C_LR:-5e-7}"
C_PER_DEVICE_BS="${C_PER_DEVICE_BS:-2}"
C_NUM_GEN="${C_NUM_GEN:-4}"
C_MAX_COMP_LEN="${C_MAX_COMP_LEN:-256}"
C_GRAD_ACCUM="${C_GRAD_ACCUM:-4}"

# Reviewer GRPO 超参 (Phase B 仍用 SFT，但保留 GRPO 参数以备切换)
R_LR="${R_LR:-5e-7}"
R_PER_DEVICE_BS="${R_PER_DEVICE_BS:-4}"
R_NUM_GEN="${R_NUM_GEN:-4}"
R_MAX_COMP_LEN="${R_MAX_COMP_LEN:-80}"
R_GRAD_ACCUM="${R_GRAD_ACCUM:-2}"

# Reviewer SFT 超参 (Phase B 实际使用)
R_SFT_LR="${R_SFT_LR:-5e-5}"
R_SFT_EPOCHS="${R_SFT_EPOCHS:-1}"
R_SFT_BATCH="${R_SFT_BATCH:-4}"
R_SFT_GRAD_ACCUM="${R_SFT_GRAD_ACCUM:-4}"

# 根据模型大小自动选择 LoRA 参数 (同 run_03_lora_sft.sh)
case "${MODEL_SIZE}" in
    0.5B|1.5B)
        LORA_RANK="${LORA_RANK:-16}"
        LORA_ALPHA="${LORA_ALPHA:-32}"
        ;;
    3B|7B)
        LORA_RANK="${LORA_RANK:-32}"
        LORA_ALPHA="${LORA_ALPHA:-64}"
        ;;
    14B)
        LORA_RANK="${LORA_RANK:-64}"
        LORA_ALPHA="${LORA_ALPHA:-128}"
        ;;
    *)
        LORA_RANK="${LORA_RANK:-32}"
        LORA_ALPHA="${LORA_ALPHA:-64}"
        ;;
esac

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 路径配置
# ─────────────────────────────────────────────────────────────────────────────────
SEED_DATA="${BASE_DIR}/prepared_data/rl/train_seed.parquet"
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_${MODEL_SIZE}"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"

# v2: Verifier 不再需要本地冻结模型（由 API 接管）
# 但保留 fallback 路径以防 API 不可用
VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_7B}"

SELFPLAY_DIR="${BASE_DIR}/selfplay_outputs_v2/${MODEL_SIZE}_${N_GPUS}npu"
LOG_DIR="${BASE_DIR}/logs/selfplay_v2_${MODEL_SIZE}"
DATA_DIR="${BASE_DIR}/selfplay_dynamic_data_v2/${MODEL_SIZE}"

mkdir -p "${SELFPLAY_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# ─────────────────────────────────────────────────────────────────────────────────
# 5.1 API Key 检查 (v2 新增)
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${USE_API_VERIFIER}" = "1" ]; then
    echo ""
    echo "── 🔑 API Key 检查 ──"
    for provider in ${API_PROVIDERS}; do
        case "${provider}" in
            qwen)
                if [ -z "${DASHSCOPE_API_KEY}" ]; then
                    echo "  ⚠️  DASHSCOPE_API_KEY 未设置 (qwen provider 需要)"
                fi
                ;;
            deepseek)
                if [ -z "${DEEPSEEK_API_KEY}" ]; then
                    echo "  ⚠️  DEEPSEEK_API_KEY 未设置 (deepseek provider 需要)"
                fi
                ;;
            openai)
                if [ -z "${OPENAI_API_KEY}" ]; then
                    echo "  ⚠️  OPENAI_API_KEY 未设置 (openai provider 需要)"
                fi
                ;;
        esac
    done
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 5.2 断点续训恢复
# ─────────────────────────────────────────────────────────────────────────────────
RESUME_FROM_ROUND=0
RESUME_SKIP_PHASE=""
PROGRESS_FILE="${SELFPLAY_DIR}/progress_v2.json"

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
        echo "  将从第 ${RESUME_FROM_ROUND} 轮的下一阶段继续"
        RESUME_FROM_ROUND=$((RESUME_FROM_ROUND - 1))
        RESUME_SKIP_PHASE="${RESUME_PHASE}"
    fi

    echo "  恢复 Challenger: ${CURRENT_CHALLENGER}"
    echo "  恢复 Reviewer  : ${CURRENT_REVIEWER}"
else
    if [ "${RESUME}" = "0" ] && [ -f "${PROGRESS_FILE}" ]; then
        echo ""
        echo "── ⚠️ RESUME=0，忽略已有断点文件，从头开始训练 ──"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 6. 打印训练参数
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   v2 对抗博弈自对弈 RL — Stackelberg (昇腾 910B)                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  模型尺寸      : ${MODEL_SIZE}"
echo "  NPU 卡数      : ${N_GPUS}"
echo "  自对弈轮次    : ${SELFPLAY_ROUNDS}"
echo "  每阶段步数    : ${MAX_STEPS}"
echo "  每类别样本数  : ${SAMPLES_PER_CAT}"
echo "  输出目录      : ${SELFPLAY_DIR}"
echo "────────────────────────────────────────────────────────────────────"
echo "  初始 Challenger : ${CHALLENGER_INIT}"
echo "  初始 Reviewer   : ${REVIEWER_INIT}"
echo "  种子数据        : ${SEED_DATA}"
echo "──────────────────── v2 新增参数 ──────────────────────────────────"
echo "  API Verifier    : $([ ${USE_API_VERIFIER} = 1 ] && echo '开启' || echo '关闭')"
echo "  API Providers   : ${API_PROVIDERS}"
echo "  迭代收敛轮次    : ${MAX_ITERATIONS} (连续 ${CONVERGENCE_K} 次一致)"
echo "  LoRA Rank/Alpha : ${LORA_RANK} / ${LORA_ALPHA}"
echo "  课程学习        : 渐进式难度过滤 + 动态 KL 权重"
echo ""

# ─────────────────────────────────────────────────────────────────────────────────
# 7. 初始化模型路径
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${RESUME_FROM_ROUND}" -eq 0 ]; then
    CURRENT_CHALLENGER="${CHALLENGER_INIT}"
    CURRENT_REVIEWER="${REVIEWER_INIT}"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: save_progress
# ─────────────────────────────────────────────────────────────────────────────────
save_progress() {
    local ROUND_NUM="$1"
    local PHASE_NAME="$2"
    cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND_NUM},
  "last_completed_phase": "${PHASE_NAME}",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "model_size": "${MODEL_SIZE}",
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "api_providers": "${API_PROVIDERS}",
  "use_api_verifier": ${USE_API_VERIFIER},
  "version": "v2",
  "timestamp": "$(date -Iseconds)"
}
EOF
}

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase 0 — v2 动态数据生成
# ─────────────────────────────────────────────────────────────────────────────────
run_phase0_datagen_v2() {
    local ROUND="$1"
    local CHALLENGER_PATH="$2"
    local REVIEWER_PATH="$3"
    local ROUND_DATA_DIR="${DATA_DIR}/round_${ROUND}"
    local LOG_FILE="${LOG_DIR}/round${ROUND}_phase0_datagen_v2_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "${ROUND_DATA_DIR}"

    echo ""
    echo "  ▶ [$(date +%H:%M:%S)] Phase 0 — v2 动态数据生成 (Round ${ROUND})"
    echo "     Challenger : ${CHALLENGER_PATH}"
    echo "     Reviewer   : ${REVIEWER_PATH}"
    echo "     API Providers: ${API_PROVIDERS}"
    echo "     输出目录   : ${ROUND_DATA_DIR}"
    echo "     日志       : ${LOG_FILE}"

    # 构建 API 参数
    local API_FLAGS=""
    if [ "${USE_API_VERIFIER}" = "1" ]; then
        API_FLAGS="--use_api_verifier --api_providers ${API_PROVIDERS}"
    else
        API_FLAGS="--no_api_verifier"
    fi

    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${SCRIPT_DIR}/generate_dynamic_data_v2.py" \
        --challenger_model  "${CHALLENGER_PATH}" \
        --reviewer_model    "${REVIEWER_PATH}" \
        --seed_data         "${SEED_DATA}" \
        --output_dir        "${ROUND_DATA_DIR}" \
        --round_idx         "${ROUND}" \
        --samples_per_cat   "${SAMPLES_PER_CAT}" \
        --batch_size        "${GEN_BATCH_SIZE}" \
        --num_npus          "${N_GPUS}" \
        --max_iterations    "${MAX_ITERATIONS}" \
        --convergence_k     "${CONVERGENCE_K}" \
        --total_rounds      "${SELFPLAY_ROUNDS}" \
        ${API_FLAGS} \
    2>&1 | tee "${LOG_FILE}"

    # 提取输出路径
    CHALLENGER_GRPO_DATA=$(grep "^CHALLENGER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)
    REVIEWER_GRPO_DATA=$(grep "^REVIEWER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)
    SELFPLAY_STATS=$(grep "^SELFPLAY_STATS=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)

    if [ -z "${CHALLENGER_GRPO_DATA}" ]; then
        CHALLENGER_GRPO_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
    fi
    if [ -z "${REVIEWER_GRPO_DATA}" ]; then
        REVIEWER_GRPO_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"
    fi

    echo "  ✓ [$(date +%H:%M:%S)] Phase 0 完成"
    echo "     Challenger 数据 : ${CHALLENGER_GRPO_DATA}"
    echo "     Reviewer   数据 : ${REVIEWER_GRPO_DATA}"
    if [ -n "${SELFPLAY_STATS}" ]; then
        echo "     统计文件        : ${SELFPLAY_STATS}"
        # 打印关键统计
        $PYTHON_EXEC -c "
import json
with open('${SELFPLAY_STATS}') as f:
    s = json.load(f)
print(f'     总生成={s[\"total_generated\"]}, 纳入训练={s[\"included_in_training\"]}, '
      f'收敛率={s[\"converged_count\"]}/{s[\"total_generated\"]}, '
      f'平均难度={s[\"avg_difficulty\"]:.3f}, 平均置信={s[\"avg_confidence\"]:.3f}')
" 2>/dev/null || true
    fi

    PHASE0_CHALLENGER_DATA="${CHALLENGER_GRPO_DATA}"
    PHASE0_REVIEWER_DATA="${REVIEWER_GRPO_DATA}"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase A — Challenger GRPO (v2 奖励)
# ─────────────────────────────────────────────────────────────────────────────────
run_phase_a_challenger_grpo() {
    local ROUND="$1"
    local MODEL_PATH="$2"
    local DATASET_PATH="$3"
    local OUTPUT_PATH="$4"
    local LOG_FILE="${LOG_DIR}/round${ROUND}_challenger_v2_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "  ▶▶ [$(date +%H:%M:%S)] Phase A — Challenger GRPO (Round ${ROUND})"
    echo "     模型  : ${MODEL_PATH}"
    echo "     数据  : ${DATASET_PATH}"
    echo "     输出  : ${OUTPUT_PATH}"
    echo "     日志  : ${LOG_FILE}"

    $PYTHON_EXEC -m torch.distributed.run \
        --nproc_per_node="${N_GPUS}" \
        --master_port="29600" \
        "${V1_SCRIPT_DIR}/adversarial_trl_grpo.py" \
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
        --selfplay_round         "${ROUND}" \
        --use_selfplay \
        --deepspeed "${V1_SCRIPT_DIR}/ds_zero2.json" \
        2>&1 | tee "${LOG_FILE}"

    if [ ! -f "${OUTPUT_PATH}/training_done.txt" ]; then
        echo "  ⚠️  [警告] Challenger GRPO 训练可能未正常完成，请检查日志: ${LOG_FILE}"
    else
        echo "  ✓ [$(date +%H:%M:%S)] Phase A Challenger GRPO 完成"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase B — Reviewer SFT + LoRA Merge (含经验回放)
# ─────────────────────────────────────────────────────────────────────────────────
run_phase_b_reviewer_sft() {
    local ROUND="$1"
    local MODEL_PATH="$2"
    local DYNAMIC_DATA="$3"
    local ROUND_DIR="$4"
    local LOG_FILE="${LOG_DIR}/round${ROUND}_reviewer_sft_v2_$(date +%Y%m%d_%H%M%S).log"

    REVIEWER_MIXED_DATA="${ROUND_DIR}/reviewer_mixed_round${ROUND}.parquet"
    REVIEWER_SFT_DATA="${ROUND_DIR}/reviewer_sft_round${ROUND}.parquet"
    REVIEWER_LORA_OUT="${ROUND_DIR}/reviewer_lora"
    REVIEWER_OUT="${ROUND_DIR}/reviewer"

    echo ""
    echo "  ▶▶ [$(date +%H:%M:%S)] Phase B — Reviewer SFT (Round ${ROUND})"
    echo "     模型  : ${MODEL_PATH}"
    echo "     数据  : ${DYNAMIC_DATA}"
    echo "     输出  : ${REVIEWER_OUT}"
    echo "     日志  : ${LOG_FILE}"

    # [1] 经验回放数据混合 (1:2 动态:种子)
    echo "   -> [1/4] 混合历史经验数据 (防止灾难性遗忘)..."
    $PYTHON_EXEC "${V1_SCRIPT_DIR}/mix_replay_data.py" \
        --dynamic_data "${DYNAMIC_DATA}" \
        --seed_data "${SEED_DATA}" \
        --output_data "${REVIEWER_MIXED_DATA}" \
        --seed_ratio 2.0

    # [2] 数据格式转换: GRPO → SFT
    echo "   -> [2/4] 转换数据格式为 SFT JSONL 结构..."
    $PYTHON_EXEC "${V1_SCRIPT_DIR}/convert_grpo_to_sft.py" \
        --input_data "${REVIEWER_MIXED_DATA}" \
        --output_data "${REVIEWER_SFT_DATA}"

    # [3] 运行 LoRA SFT
    echo "   -> [3/4] 启动 Reviewer LoRA SFT 训练..."
    if [ "$N_GPUS" -ge 2 ]; then
        LAUNCH_CMD="$PYTHON_EXEC -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS"
    else
        LAUNCH_CMD="$PYTHON_EXEC"
    fi

    $LAUNCH_CMD "${V1_SCRIPT_DIR}/../model_lora/train_reviewer_lora.py" \
        --model_path "${MODEL_PATH}" \
        --data_path "${REVIEWER_SFT_DATA}" \
        --output_dir "${REVIEWER_LORA_OUT}" \
        --lora_rank "${LORA_RANK}" \
        --lora_alpha "${LORA_ALPHA}" \
        --batch_size "${R_SFT_BATCH}" \
        --gradient_accumulation_steps "${R_SFT_GRAD_ACCUM}" \
        --num_epochs "${R_SFT_EPOCHS}" \
        --learning_rate "${R_SFT_LR}" \
        --max_length 2048 \
        --n_devices "${N_GPUS}" 2>&1 | tee "${LOG_FILE}"

    # [4] 合并 LoRA → 全量模型
    echo "   -> [4/4] 合并 LoRA 权重..."
    $PYTHON_EXEC "${V1_SCRIPT_DIR}/../model_lora/merge_lora.py" \
        --base_model "${MODEL_PATH}" \
        --lora_path "${REVIEWER_LORA_OUT}" \
        --output_path "${REVIEWER_OUT}"

    if [ -d "${REVIEWER_OUT}" ]; then
        echo "  ✓ [$(date +%H:%M:%S)] Phase B Reviewer SFT 完成"
    else
        echo "  ⚠️  [警告] Reviewer SFT 输出目录不存在: ${REVIEWER_OUT}"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────────
# 8. 自对弈主循环
# ─────────────────────────────────────────────────────────────────────────────────
TOTAL_START_TIME=$(date +%s)

for ROUND in $(seq 1 "${SELFPLAY_ROUNDS}"); do
    CURRENT_ROUND="${ROUND}"
    ROUND_START_TIME=$(date +%s)

    # ── 断点: 跳过已完成轮次 ──
    if [ "${ROUND}" -le "${RESUME_FROM_ROUND}" ]; then
        echo "  ⏭️  跳过已完成的第 ${ROUND} 轮"
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🎮 v2 自对弈第 ${ROUND} / ${SELFPLAY_ROUNDS} 轮                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo "  当前 Challenger : ${CURRENT_CHALLENGER}"
    echo "  当前 Reviewer   : ${CURRENT_REVIEWER}"

    ROUND_DIR="${SELFPLAY_DIR}/round_${ROUND}"
    mkdir -p "${ROUND_DIR}/challenger" "${ROUND_DIR}/reviewer"

    # ════════════════════════════════════════════════════════════
    # Phase 0 — v2 动态数据生成
    # API Verifier + 迭代收敛 + 课程学习打分
    # ════════════════════════════════════════════════════════════
    if [ -n "${RESUME_SKIP_PHASE}" ]; then
        echo "  ⏭️  断点恢复: 该轮 ${RESUME_SKIP_PHASE} 已完成"
    fi

    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phase0" ]; then
        echo ""
        echo "── Phase 0: v2 动态数据生成 (Round ${ROUND}) ──"

        run_phase0_datagen_v2 \
            "${ROUND}" \
            "${CURRENT_CHALLENGER}" \
            "${CURRENT_REVIEWER}"

        save_progress "${ROUND}" "phase0"
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

    # 安全回退
    if [ ! -f "${DYNAMIC_CHALLENGER_DATA}" ]; then
        echo "  ❌ Challenger 数据缺失，回退使用种子数据"
        DYNAMIC_CHALLENGER_DATA="${SEED_DATA}"
    fi
    if [ ! -f "${DYNAMIC_REVIEWER_DATA}" ]; then
        echo "  ❌ Reviewer 数据缺失，回退使用种子数据"
        DYNAMIC_REVIEWER_DATA="${SEED_DATA}"
    fi

    # ════════════════════════════════════════════════════════════
    # Phase A — Challenger GRPO
    # 固定 Reviewer，优化 Challenger 生成更隐蔽的对抗文本
    # v2 改进: reward_model 中携带 difficulty_weight + kl_weight
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseA" ]; then
        echo ""
        echo "── Phase A: Challenger GRPO (Round ${ROUND}) ──"

        CHALLENGER_OUT="${ROUND_DIR}/challenger"
        run_phase_a_challenger_grpo \
            "${ROUND}" \
            "${CURRENT_CHALLENGER}" \
            "${DYNAMIC_CHALLENGER_DATA}" \
            "${CHALLENGER_OUT}"

        CURRENT_CHALLENGER="${CHALLENGER_OUT}"
        save_progress "${ROUND}" "phaseA"
    else
        CHALLENGER_OUT="${ROUND_DIR}/challenger"
        if [ -d "${CHALLENGER_OUT}" ]; then
            CURRENT_CHALLENGER="${CHALLENGER_OUT}"
        fi
    fi

    # ════════════════════════════════════════════════════════════
    # Phase B — Reviewer SFT + LoRA Merge
    # 使用 Challenger 生成的对抗文本 + 经验回放池进行 SFT
    # v2 改进: 数据含课程学习难度标注 + 新 Gold Answer
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseB" ]; then
        echo ""
        echo "── Phase B: Reviewer SFT (Round ${ROUND}) ──"

        run_phase_b_reviewer_sft \
            "${ROUND}" \
            "${CURRENT_REVIEWER}" \
            "${DYNAMIC_REVIEWER_DATA}" \
            "${ROUND_DIR}"

        REVIEWER_OUT="${ROUND_DIR}/reviewer"
        if [ -d "${REVIEWER_OUT}" ]; then
            CURRENT_REVIEWER="${REVIEWER_OUT}"
        fi
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
    ROUND_END_TIME=$(date +%s)
    ROUND_ELAPSED=$(( ROUND_END_TIME - ROUND_START_TIME ))
    ROUND_MINUTES=$(( ROUND_ELAPSED / 60 ))

    echo ""
    echo "── Round ${ROUND} 完成 (耗时 ${ROUND_MINUTES} 分钟) ──"
    echo "   更新后 Challenger : ${CURRENT_CHALLENGER}"
    echo "   更新后 Reviewer   : ${CURRENT_REVIEWER}"

    save_progress "${ROUND}" "done"

    # 估算剩余时间
    REMAINING=$(( SELFPLAY_ROUNDS - ROUND ))
    if [ "${REMAINING}" -gt 0 ]; then
        EST_REMAINING=$(( ROUND_ELAPSED * REMAINING / 60 ))
        echo "   预计剩余 ${REMAINING} 轮 ≈ ${EST_REMAINING} 分钟"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────────
# 9. 训练结束汇总
# ─────────────────────────────────────────────────────────────────────────────────
TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END_TIME - TOTAL_START_TIME) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ v2 对抗博弈自对弈训练全部完成！                                       ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  总耗时          : ${TOTAL_ELAPSED} 分钟"
echo "  最终 Challenger : ${CURRENT_CHALLENGER}"
echo "  最终 Reviewer   : ${CURRENT_REVIEWER}"
echo "  日志目录        : ${LOG_DIR}"
echo "  动态数据目录    : ${DATA_DIR}"

# 保存最终模型路径
FINAL_SUMMARY="${SELFPLAY_DIR}/final_models_v2.txt"
cat > "${FINAL_SUMMARY}" << EOF
# v2 对抗博弈自对弈最终模型路径
# 完成时间: $(date)
# 模型尺寸: ${MODEL_SIZE}
# NPU 卡数: ${N_GPUS}
# 训练轮次: ${SELFPLAY_ROUNDS}
# 每轮步数: ${MAX_STEPS}
# API Providers: ${API_PROVIDERS}
# v2 特性: API Verifier + 迭代收敛 + 课程学习 + PDF对齐奖励
challenger_final: ${CURRENT_CHALLENGER}
reviewer_final  : ${CURRENT_REVIEWER}
EOF

echo ""
echo "最终模型路径已记录: ${FINAL_SUMMARY}"

# ─────────────────────────────────────────────────────────────────────────────────
# 10. 可选: 生成训练曲线摘要
# ─────────────────────────────────────────────────────────────────────────────────
SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_training.py"
if [ -f "${SUMMARY_SCRIPT}" ]; then
    echo ""
    echo "── 生成训练摘要 ──"
    $PYTHON_EXEC "${SUMMARY_SCRIPT}" \
        --data_dir "${DATA_DIR}" \
        --output "${SELFPLAY_DIR}/training_summary_v2.json" \
        2>/dev/null || echo "  (训练摘要生成跳过)"
fi
