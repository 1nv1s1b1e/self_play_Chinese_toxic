#!/bin/bash
set -e

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  对抗博弈自对弈 RL 训练主循环 — TRL + DeepSpeed (昇腾 910B)                    ║
# ║                                                                              ║
# ║  架构: Stackelberg 迭代对抗训练框架 (见 README_NPU.md)                         ║
# ║  每轮三个阶段:                                                                  ║
# ║    Phase 0: 动态数据生成                                                        ║
# ║      - Challenger 模型推理生成对抗文本                                           ║
# ║      - Reviewer   模型推理评估生成文本                                           ║
# ║      - 计算各类别对抗成功率 (ASR) 并保存 parquet                                  ║
# ║    Phase A: Challenger GRPO                                                    ║
# ║      - 使用动态生成数据训练 Challenger                                            ║
# ║      - 奖励 = quality_gate × (topic_signal + ASR_bonus)                        ║
# ║    Phase B: Reviewer GRPO                                                      ║
# ║      - 使用 Challenger 生成文本训练 Reviewer                                     ║
# ║      - 奖励 = 正确分类 (binary + category + toxic_type + expression)             ║
# ║                                                                              ║
# ║  用法:                                                                         ║
# ║    bash run_selfplay_trl_npu.sh                    # 2卡 0.5B 3轮              ║
# ║    N_GPUS=4 MODEL_SIZE=1.5B bash run_selfplay_trl_npu.sh                       ║
# ║    SELFPLAY_ROUNDS=5 MAX_STEPS=100 bash run_selfplay_trl_npu.sh                ║
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

MODEL_SIZE="${MODEL_SIZE:-3B}"          # 0.5B / 1.5B / 3B / 7B
N_GPUS="${N_GPUS:-4}"                    # 每阶段使用的 NPU 卡数
SELFPLAY_ROUNDS="${SELFPLAY_ROUNDS:-5}"  # 自对弈总轮次
MAX_STEPS="${MAX_STEPS:-50}"            # 每个 GRPO 阶段的训练步数 (降低防止过拟合)
SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-256}" # Phase 0 每类别生成样本数
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"    # Phase 0 推理 batch size
RESUME="${RESUME:-1}"                    # 1=自动断点续训, 0=强制从头开始

# Verifier 后端配置 (local=本地模型 / api=Qwen DashScope API)
VERIFIER_BACKEND="${VERIFIER_BACKEND:-local}"     # "local" 或 "api"
VERIFIER_API_MODEL="${VERIFIER_API_MODEL:-qwen-plus}"  # API 模型名称

# Challenger GRPO 超参
C_LR="${C_LR:-5e-7}"
C_PER_DEVICE_BS="${C_PER_DEVICE_BS:-2}"
C_NUM_GEN="${C_NUM_GEN:-4}"
C_MAX_COMP_LEN="${C_MAX_COMP_LEN:-256}"
C_GRAD_ACCUM="${C_GRAD_ACCUM:-4}"

# Reviewer GRPO 超参
R_LR="${R_LR:-5e-7}"
R_PER_DEVICE_BS="${R_PER_DEVICE_BS:-4}"
R_NUM_GEN="${R_NUM_GEN:-4}"
R_MAX_COMP_LEN="${R_MAX_COMP_LEN:-80}"
R_GRAD_ACCUM="${R_GRAD_ACCUM:-2}"

# ─────────────────────────────────────────────────────────────────────────────────
# 5. 路径配置
# ─────────────────────────────────────────────────────────────────────────────────
SEED_DATA="${BASE_DIR}/prepared_data/rl/train_seed.parquet"
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_${MODEL_SIZE}"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"

# Verifier: 固定冻结的 LoRA Reviewer 7B，整个自对弈过程中不参与训练
# 用作 ground-truth oracle 计算 R_challenger / R_reviewer
# 使用 7B 模型提升标签质量（比 3B 准确率更高）
VERIFIER_MODEL="${VERIFIER_MODEL:-${BASE_DIR}/merged_models_toxicn/reviewer_7B}"

SELFPLAY_DIR="${BASE_DIR}/selfplay_outputs_sft_reviewer/${MODEL_SIZE}_${N_GPUS}npu"
LOG_DIR="${BASE_DIR}/logs/selfplay_trl_${MODEL_SIZE}"
DATA_DIR="${BASE_DIR}/selfplay_dynamic_data/${MODEL_SIZE}"

mkdir -p "${SELFPLAY_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# ─────────────────────────────────────────────────────────────────────────────────
# 5.1 断点续训恢复
# ─────────────────────────────────────────────────────────────────────────────────
RESUME_FROM_ROUND=0
RESUME_SKIP_PHASE=""   # 用于轮内阶段跳过: "phase0", "phaseA", "phaseB", "done"
PROGRESS_FILE="${SELFPLAY_DIR}/progress.json"

if [ "${RESUME}" = "1" ] && [ -f "${PROGRESS_FILE}" ]; then
    echo ""
    echo "── 🔄 检测到断点续训文件: ${PROGRESS_FILE} ──"
    # 使用 python 解析 JSON（比 jq 更通用）
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

    # 恢复模型路径
    if [ -n "${SAVED_CHALLENGER}" ] && [ -d "${SAVED_CHALLENGER}" ]; then
        CURRENT_CHALLENGER="${SAVED_CHALLENGER}"
    fi
    if [ -n "${SAVED_REVIEWER}" ] && [ -d "${SAVED_REVIEWER}" ]; then
        CURRENT_REVIEWER="${SAVED_REVIEWER}"
    fi

    # 判断恢复策略
    if [ "${RESUME_PHASE}" = "done" ]; then
        # 该轮已完全完成，从下一轮开始
        echo "  上次完整完成到第 ${RESUME_FROM_ROUND} 轮，将从第 $((RESUME_FROM_ROUND + 1)) 轮开始"
    else
        # 该轮中间中断，需要继续该轮的剩余阶段
        echo "  第 ${RESUME_FROM_ROUND} 轮在 ${RESUME_PHASE} 阶段后中断"
        echo "  将从第 ${RESUME_FROM_ROUND} 轮的下一阶段继续"
        RESUME_FROM_ROUND=$((RESUME_FROM_ROUND - 1))  # 减 1，因为主循环会 skip <= RESUME_FROM_ROUND
        RESUME_SKIP_PHASE="${RESUME_PHASE}"            # 记录已完成的阶段用于轮内跳过
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
echo "║   对抗博弈自对弈 RL  —  TRL + DeepSpeed (昇腾 910B)                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  模型尺寸      : ${MODEL_SIZE}"
echo "  NPU 卡数      : ${N_GPUS}"
echo "  自对弈轮次    : ${SELFPLAY_ROUNDS}"
echo "  每阶段步数    : ${MAX_STEPS}"
echo "  每类别样本数  : ${SAMPLES_PER_CAT}"
echo "  DeepSpeed配置 : ${SCRIPT_DIR}/ds_zero2.json"
echo "  输出目录      : ${SELFPLAY_DIR}"
echo "────────────────────────────────────────────────────────────────────"
echo "  初始 Challenger: ${CHALLENGER_INIT}"
echo "  初始 Reviewer  : ${REVIEWER_INIT}"
echo "  Verifier [冻结]: ${VERIFIER_MODEL}"
echo "  Verifier 后端  : ${VERIFIER_BACKEND} (API模型: ${VERIFIER_API_MODEL})"
echo "  种子数据       : ${SEED_DATA}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────────
# 7. 初始化: 模型路径（若已恢复断点则使用恢复的路径）
# ─────────────────────────────────────────────────────────────────────────────────
if [ "${RESUME_FROM_ROUND}" -eq 0 ]; then
    CURRENT_CHALLENGER="${CHALLENGER_INIT}"
    CURRENT_REVIEWER="${REVIEWER_INIT}"
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase 0 — 动态数据生成
# ─────────────────────────────────────────────────────────────────────────────────
run_phase0_datagen() {
    local ROUND="$1"
    local CHALLENGER_PATH="$2"
    local REVIEWER_PATH="$3"
    local ROUND_DATA_DIR="${DATA_DIR}/round_${ROUND}"
    local LOG_FILE="${LOG_DIR}/round${ROUND}_phase0_datagen_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "${ROUND_DATA_DIR}"

    echo ""
    echo "  ▶ [$(date +%H:%M:%S)] Phase 0 — 动态数据生成 (Round ${ROUND})"
    echo "     Challenger: ${CHALLENGER_PATH}"
    echo "     Reviewer  : ${REVIEWER_PATH}"
    echo "     输出目录  : ${ROUND_DATA_DIR}"
    echo "     日志      : ${LOG_FILE}"

    # 运行数据生成脚本 (单进程，自动选取可用 NPU)
    # 取消 DATAGEN_OUTPUT=$(...)，直接运行并利用 tee 实时输出到控制台与文件
    # Use all NPUs for parallel inference
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${SCRIPT_DIR}/generate_dynamic_data.py" \
        --challenger_model "${CHALLENGER_PATH}" \
        --reviewer_model   "${REVIEWER_PATH}" \
        --verifier_model   "${VERIFIER_MODEL}" \
        --seed_data        "${SEED_DATA}" \
        --output_dir       "${ROUND_DATA_DIR}" \
        --round_idx        "${ROUND}" \
        --samples_per_cat  "${SAMPLES_PER_CAT}" \
        --batch_size       "${GEN_BATCH_SIZE}" \
        --num_npus         "${N_GPUS}" \
        --verifier_backend "${VERIFIER_BACKEND}" \
        --verifier_api_model "${VERIFIER_API_MODEL}" \
    2>&1 | tee "${LOG_FILE}"

    # 从脚本末尾输出中提取 parquet 路径
    CHALLENGER_GRPO_DATA=$(grep "^CHALLENGER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)
    REVIEWER_GRPO_DATA=$(grep "^REVIEWER_GRPO_DATA=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)
    SELFPLAY_STATS=$(grep "^SELFPLAY_STATS=" "${LOG_FILE}" | tail -1 | cut -d= -f2-)

    # 如果提取失败，使用默认路径
    if [ -z "${CHALLENGER_GRPO_DATA}" ]; then
        CHALLENGER_GRPO_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
    fi
    if [ -z "${REVIEWER_GRPO_DATA}" ]; then
        REVIEWER_GRPO_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"
    fi

    echo "  ✓ [$(date +%H:%M:%S)] Phase 0 完成"
    echo "     Challenger 数据: ${CHALLENGER_GRPO_DATA}"
    echo "     Reviewer   数据: ${REVIEWER_GRPO_DATA}"

    # 通过全局变量传递路径给调用方
    PHASE0_CHALLENGER_DATA="${CHALLENGER_GRPO_DATA}"
    PHASE0_REVIEWER_DATA="${REVIEWER_GRPO_DATA}"
}

# ─────────────────────────────────────────────────────────────────────────────────
# 工具函数: Phase A/B — GRPO 训练
# ─────────────────────────────────────────────────────────────────────────────────
run_grpo_phase() {
    local ROLE="$1"
    local MODEL_PATH="$2"
    local DATASET_PATH="$3"
    local OUTPUT_PATH="$4"
    local LOG_FILE="$5"
    local MASTER_PORT="$6"

    # 按角色选择超参
    if [ "${ROLE}" = "challenger" ]; then
        local LR="${C_LR}"
        local PER_BS="${C_PER_DEVICE_BS}"
        local NUM_GEN="${C_NUM_GEN}"
        local MAX_COMP="${C_MAX_COMP_LEN}"
        local GRAD_ACC="${C_GRAD_ACCUM}"
        local SELFPLAY_FLAG="--use_selfplay"    # Challenger 始终使用 selfplay 奖励
    else
        local LR="${R_LR}"
        local PER_BS="${R_PER_DEVICE_BS}"
        local NUM_GEN="${R_NUM_GEN}"
        local MAX_COMP="${R_MAX_COMP_LEN}"
        local GRAD_ACC="${R_GRAD_ACCUM}"
        local SELFPLAY_FLAG=""
    fi

    echo ""
    echo "  ▶▶ [$(date +%H:%M:%S)] Phase ${ROLE} GRPO 启动"
    echo "     模型     : ${MODEL_PATH}"
    echo "     数据     : ${DATASET_PATH}"
    echo "     输出     : ${OUTPUT_PATH}"
    echo "     日志     : ${LOG_FILE}"

    $PYTHON_EXEC -m torch.distributed.run \
        --nproc_per_node="${N_GPUS}" \
        --master_port="${MASTER_PORT}" \
        "${SCRIPT_DIR}/adversarial_trl_grpo.py" \
        --role                   "${ROLE}" \
        --model_path             "${MODEL_PATH}" \
        --dataset_path           "${DATASET_PATH}" \
        --output_dir             "${OUTPUT_PATH}" \
        --max_steps              "${MAX_STEPS}" \
        --save_steps             "${MAX_STEPS}" \
        --learning_rate          "${LR}" \
        --per_device_batch_size  "${PER_BS}" \
        --num_generations        "${NUM_GEN}" \
        --max_completion_length  "${MAX_COMP}" \
        --grad_accum             "${GRAD_ACC}" \
        --selfplay_round         "${CURRENT_ROUND:-0}" \
        ${SELFPLAY_FLAG} \
        --deepspeed "${SCRIPT_DIR}/ds_zero2.json" \
        2>&1 | tee "${LOG_FILE}"

    # 检查训练完成标记文件
    if [ ! -f "${OUTPUT_PATH}/training_done.txt" ]; then
        echo "  ⚠️  [警告] ${ROLE} GRPO 训练可能未正常完成，请检查日志: ${LOG_FILE}"
    else
        echo "  ✓ [$(date +%H:%M:%S)] Phase ${ROLE} GRPO 完成"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────────
# 8. 自对弈主循环
# ─────────────────────────────────────────────────────────────────────────────────
for ROUND in $(seq 1 "${SELFPLAY_ROUNDS}"); do
    CURRENT_ROUND="${ROUND}"

    # ── 断点续训: 跳过已完成的轮次 ──
    if [ "${ROUND}" -le "${RESUME_FROM_ROUND}" ]; then
        echo "  ⏭️  跳过已完成的第 ${ROUND} 轮"
        continue
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  🎮 自对弈第 ${ROUND} / ${SELFPLAY_ROUNDS} 轮                                         ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo "  当前 Challenger: ${CURRENT_CHALLENGER}"
    echo "  当前 Reviewer  : ${CURRENT_REVIEWER}"

    ROUND_DIR="${SELFPLAY_DIR}/round_${ROUND}"
    mkdir -p "${ROUND_DIR}/challenger" "${ROUND_DIR}/reviewer"

    # ════════════════════════════════════════════════════════════
    # Phase 0 — 动态对抗数据生成
    # 用当前两个模型生成本轮的 GRPO 训练数据
    # ════════════════════════════════════════════════════════════
    # ── 断点续训: 判断是否跳过 Phase 0 ──
    if [ -n "${RESUME_SKIP_PHASE}" ]; then
        # 有阶段跳过标记，说明该轮中间中断，需检查已完成的阶段
        echo "  ⏭️  断点恢复: 该轮 ${RESUME_SKIP_PHASE} 已完成，跳过"
    fi

    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phase0" ]; then
        echo ""
        echo "── Phase 0: 动态对抗数据生成 (Round ${ROUND}) ──"

        run_phase0_datagen \
            "${ROUND}" \
            "${CURRENT_CHALLENGER}" \
            "${CURRENT_REVIEWER}"

        # 保存 Phase 0 完成进度
        cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "phase0",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "timestamp": "$(date -Iseconds)"
}
EOF
    fi

    # 解析本轮数据路径（无论是否跳过 Phase 0，都需要设置数据路径）
    ROUND_DATA_DIR="${DATA_DIR}/round_${ROUND}"
    DYNAMIC_CHALLENGER_DATA="${ROUND_DATA_DIR}/challenger_grpo_round${ROUND}.parquet"
    DYNAMIC_REVIEWER_DATA="${ROUND_DATA_DIR}/reviewer_grpo_round${ROUND}.parquet"
    # 如果 Phase 0 刚执行完，使用其输出路径
    if [ -n "${PHASE0_CHALLENGER_DATA}" ] && [ -f "${PHASE0_CHALLENGER_DATA}" ]; then
        DYNAMIC_CHALLENGER_DATA="${PHASE0_CHALLENGER_DATA}"
    fi
    if [ -n "${PHASE0_REVIEWER_DATA}" ] && [ -f "${PHASE0_REVIEWER_DATA}" ]; then
        DYNAMIC_REVIEWER_DATA="${PHASE0_REVIEWER_DATA}"
    fi

    # 安全检查: 数据文件是否存在
    if [ ! -f "${DYNAMIC_CHALLENGER_DATA}" ]; then
        echo "  ❌ Challenger GRPO 数据文件不存在: ${DYNAMIC_CHALLENGER_DATA}"
        echo "     回退使用种子数据: ${SEED_DATA}"
        DYNAMIC_CHALLENGER_DATA="${SEED_DATA}"
    fi
    if [ ! -f "${DYNAMIC_REVIEWER_DATA}" ]; then
        echo "  ❌ Reviewer GRPO 数据文件不存在: ${DYNAMIC_REVIEWER_DATA}"
        echo "     回退使用种子数据: ${SEED_DATA}"
        DYNAMIC_REVIEWER_DATA="${SEED_DATA}"
    fi

    # ════════════════════════════════════════════════════════════
    # Phase A — Challenger GRPO
    # 固定 Reviewer，优化 Challenger 生成更难以检测的对抗文本
    # 奖励: quality_gate × (topic_signal + ASR_bonus)
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseA" ]; then
        echo ""
        echo "── Phase A: Challenger GRPO (Round ${ROUND}) ──"
        CHALLENGER_OUT="${ROUND_DIR}/challenger"
        CHALLENGER_LOG="${LOG_DIR}/round${ROUND}_challenger_$(date +%Y%m%d_%H%M%S).log"

        run_grpo_phase \
            "challenger" \
            "${CURRENT_CHALLENGER}" \
            "${DYNAMIC_CHALLENGER_DATA}" \
            "${CHALLENGER_OUT}" \
            "${CHALLENGER_LOG}" \
            "29600"

        CURRENT_CHALLENGER="${CHALLENGER_OUT}"

        # 保存 Phase A 完成进度
        cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "phaseA",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
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
    # Phase B — Reviewer SFT (取代极不稳定的分类器 GRPO)
    # 使用 Challenger 生成的对抗文本和经验回放池，通过 SFT 微调分类能力
    # ════════════════════════════════════════════════════════════
    if [ -z "${RESUME_SKIP_PHASE}" ] || [ "${RESUME_SKIP_PHASE}" \< "phaseB" ]; then
        echo ""
        echo "── Phase B: Reviewer SFT (Round ${ROUND}) ──"
        REVIEWER_MIXED_DATA="${ROUND_DIR}/reviewer_mixed_round${ROUND}.parquet"
        REVIEWER_SFT_DATA="${ROUND_DIR}/reviewer_sft_round${ROUND}.parquet"
        REVIEWER_LORA_OUT="${ROUND_DIR}/reviewer_lora"
        REVIEWER_OUT="${ROUND_DIR}/reviewer"
        REVIEWER_LOG="${LOG_DIR}/round${ROUND}_reviewer_$(date +%Y%m%d_%H%M%S).log"

        # [1] 经验回放数据混合 (1:2)
        echo "   -> 混合历史经验数据 (防止灾难性遗忘)..."
        $PYTHON_EXEC "${SCRIPT_DIR}/mix_replay_data.py" \
            --dynamic_data "${DYNAMIC_REVIEWER_DATA}" \
            --seed_data "${SEED_DATA}" \
            --output_data "${REVIEWER_MIXED_DATA}" \
            --seed_ratio 2.0

        # [2] 数据格式转换: GRPO (Prompt+RM) => 标准多轮对话 SFT 格式
        echo "   -> 转换数据格式为 SFT JSONL 结构..."
        $PYTHON_EXEC "${SCRIPT_DIR}/convert_grpo_to_sft.py" \
            --input_data "${REVIEWER_MIXED_DATA}" \
            --output_data "${REVIEWER_SFT_DATA}"

        # [3] 运行 SFT (复用现有的 LoRA SFT 脚本)
        echo "   -> 启动 Reviewer LoRA SFT 训练..."
        if [ "$N_GPUS" -ge 2 ]; then
            LAUNCH_CMD="python -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS"
        else
            LAUNCH_CMD="python"
        fi
        
        $LAUNCH_CMD "${SCRIPT_DIR}/../model_lora/train_reviewer_lora.py" \
            --model_path "${CURRENT_REVIEWER}" \
            --data_path "${REVIEWER_SFT_DATA}" \
            --output_dir "${REVIEWER_LORA_OUT}" \
            --lora_rank 32 \
            --lora_alpha 32 \
            --batch_size 4 \
            --gradient_accumulation_steps 4 \
            --num_epochs 1 \
            --learning_rate 5e-5 \
            --max_length 2048 \
            --n_devices $N_GPUS 2>&1 | tee "${REVIEWER_LOG}"
            
        # [4] 合并权重到大模型基座
        echo "   -> 将 LoRA 合并为全量因果模型基座..."
        $PYTHON_EXEC "${SCRIPT_DIR}/../model_lora/merge_lora.py" \
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
    echo ""
    # 清除轮内跳过标记（仅在恢复轮生效一次）
    RESUME_SKIP_PHASE=""

    echo "── Round ${ROUND} 完成 ──"
    echo "   更新后 Challenger: ${CURRENT_CHALLENGER}"
    echo "   更新后 Reviewer  : ${CURRENT_REVIEWER}"

    # 保存本轮完整完成进度
    cat > "${PROGRESS_FILE}" << EOF
{
  "last_completed_round": ${ROUND},
  "last_completed_phase": "done",
  "total_rounds": ${SELFPLAY_ROUNDS},
  "current_challenger": "${CURRENT_CHALLENGER}",
  "current_reviewer": "${CURRENT_REVIEWER}",
  "timestamp": "$(date -Iseconds)"
}
EOF

done

# ─────────────────────────────────────────────────────────────────────────────────
# 9. 训练结束汇总
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ 对抗博弈自对弈 RL (TRL) 训练全部完成！                               ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo "  最终 Challenger: ${CURRENT_CHALLENGER}"
echo "  最终 Reviewer  : ${CURRENT_REVIEWER}"
echo "  日志目录       : ${LOG_DIR}"
echo "  动态数据目录   : ${DATA_DIR}"

# 保存最终模型路径
FINAL_SUMMARY="${SELFPLAY_DIR}/final_models_trl.txt"
cat > "${FINAL_SUMMARY}" << EOF
# TRL 对抗博弈自对弈最终模型路径
# 完成时间: $(date)
# 模型尺寸: ${MODEL_SIZE}
# NPU 卡数: ${N_GPUS}
# 训练轮次: ${SELFPLAY_ROUNDS}
# 每轮步数: ${MAX_STEPS}
challenger_final: ${CURRENT_CHALLENGER}
reviewer_final  : ${CURRENT_REVIEWER}
EOF

echo ""
echo "最终模型路径已记录: ${FINAL_SUMMARY}"
