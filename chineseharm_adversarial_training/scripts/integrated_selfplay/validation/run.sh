#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  一键执行脚本 — 直接在远程服务器运行
# ═══════════════════════════════════════════════════════════════════════════════
#
#  用法:
#    bash run.sh step1          # 应用工程修复
#    bash run.sh step2          # CoT pilot (1000条, ~2元)
#    bash run.sh step3          # CoT 全量重构 (9600条, ~19元)
#    bash run.sh step4          # CoT SFT 训练
#    bash run.sh step5_eval     # 评估 CoT 模型
#    bash run.sh step6          # 运行 Self-Play (含所有修复)
#    bash run.sh plan           # 查看完整计划
#
# ═══════════════════════════════════════════════════════════════════════════════
set -e

# ─────────────────────────────────────────────────────────────────────────────
#  ★★★ 配置区 — 请根据实际环境修改 ★★★
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR="/home/ma-user/work/test"
PYTHON_EXEC="/home/ma-user/.conda/envs/ssp_train/bin/python"
DASHSCOPE_API_KEY=""              # ← 填入你的 API Key
API_MODEL="qwen-plus"
MODEL_SIZE="3B"
N_GPUS=4

# ─────────────────────────────────────────────────────────────────────────────
#  路径 (一般不需要修改)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELFPLAY_DIR="$(dirname "$SCRIPT_DIR")"           # integrated_selfplay/
SCRIPTS_DIR="$(dirname "$SELFPLAY_DIR")"           # scripts/

TRAIN_JSON="${BASE_DIR}/split_data/train.json"
TEST_JSON="${BASE_DIR}/split_data/test.json"
VAL_JSON="${BASE_DIR}/split_data/val.json"

COT_DIR="${BASE_DIR}/cot_data"
COT_PILOT_DIR="${COT_DIR}/pilot"

BASE_MODEL="${BASE_DIR}/models_base/Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
REVIEWER_SFT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"
REVIEWER_COT_LORA="${BASE_DIR}/cot_models/reviewer_cot_${MODEL_SIZE}"
REVIEWER_COT_MERGED="${BASE_DIR}/cot_models/reviewer_cot_${MODEL_SIZE}_merged"

SELFPLAY_SH="${SELFPLAY_DIR}/run_selfplay.sh"
GRPO_PY="${SELFPLAY_DIR}/adversarial_trl_grpo.py"

# 昇腾环境
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

# ─────────────────────────────────────────────────────────────────────────────
#  查找 train.json (兼容不同目录结构)
# ─────────────────────────────────────────────────────────────────────────────
find_train_json() {
    # Prefer split_data/train.json (has 文本/标签 format)
    for p in \
        "${BASE_DIR}/split_data/train.json" \
        "${BASE_DIR}/chineseharm_adversarial_training/split_data/train.json" \
        "${BASE_DIR}/train.json" \
        "${BASE_DIR}/chineseharm_adversarial_training/train.json"; do
        if [ -f "$p" ]; then
            TRAIN_JSON="$p"
            return
        fi
    done
    echo "ERROR: train.json not found"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1: 工程修复
# ═══════════════════════════════════════════════════════════════════════════════
do_step1() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 1: 应用工程修复                                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    if [ ! -f "${SELFPLAY_SH}" ]; then
        echo "❌ ${SELFPLAY_SH} 不存在"; exit 1
    fi

    cp "${SELFPLAY_SH}" "${SELFPLAY_SH}.bak"
    echo "  备份: ${SELFPLAY_SH}.bak"

    # 1) REVIEWER_MIX_RATIO: 0 → 0.5
    sed -i 's/REVIEWER_MIX_RATIO="${REVIEWER_MIX_RATIO:-0}"/REVIEWER_MIX_RATIO="${REVIEWER_MIX_RATIO:-0.5}"/' "${SELFPLAY_SH}"
    echo "  ✓ REVIEWER_MIX_RATIO: 0 → 0.5  (防止灾难性遗忘)"

    # 2) SAMPLES_PER_CAT: 9 → 64
    sed -i 's/SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-9}"/SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-64}"/' "${SELFPLAY_SH}"
    echo "  ✓ SAMPLES_PER_CAT: 9 → 64  (充足 RL 数据)"

    # 3) NONTOXIC_SAMPLES: 20 → 80
    sed -i 's/NONTOXIC_SAMPLES="${NONTOXIC_SAMPLES:-20}"/NONTOXIC_SAMPLES="${NONTOXIC_SAMPLES:-80}"/' "${SELFPLAY_SH}"
    echo "  ✓ NONTOXIC_SAMPLES: 20 → 80  (平衡分布)"

    # 4) C_NUM_GEN: 8 → 1 (REINFORCE)
    sed -i 's/C_NUM_GEN="${C_NUM_GEN:-8}"/C_NUM_GEN="${C_NUM_GEN:-1}"/' "${SELFPLAY_SH}"
    echo "  ✓ C_NUM_GEN: 8 → 1  (REINFORCE 替代 GRPO)"

    # 5) C_GRAD_ACCUM: 4 → 8
    sed -i 's/C_GRAD_ACCUM="${C_GRAD_ACCUM:-4}"/C_GRAD_ACCUM="${C_GRAD_ACCUM:-8}"/' "${SELFPLAY_SH}"
    echo "  ✓ C_GRAD_ACCUM: 4 → 8  (补偿 n=1)"

    # 6) REVIEWER_NONTOXIC_BOOST: 1 → 2
    sed -i 's/REVIEWER_NONTOXIC_BOOST="${REVIEWER_NONTOXIC_BOOST:-1}"/REVIEWER_NONTOXIC_BOOST="${REVIEWER_NONTOXIC_BOOST:-2}"/' "${SELFPLAY_SH}"
    echo "  ✓ REVIEWER_NONTOXIC_BOOST: 1 → 2"

    # 7) Quality gate penalty: -1.0 → 0.0
    if [ -f "${GRPO_PY}" ]; then
        cp "${GRPO_PY}" "${GRPO_PY}.bak"
        sed -i 's/rewards = \[-1\.0\] \* len(completion_texts)/rewards = [0.0] * len(completion_texts)/' "${GRPO_PY}"
        echo "  ✓ Quality gate penalty: -1.0 → 0.0  (防止 Challenger 崩溃)"
    fi

    echo ""
    echo "  ✅ Step 1 完成。所有修复已应用。"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2: CoT Pilot (1000条)
# ═══════════════════════════════════════════════════════════════════════════════
do_step2() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 2: CoT Pilot (1000条, 约2元)                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    if [ -z "${DASHSCOPE_API_KEY}" ]; then
        echo "❌ 请在脚本顶部填入 DASHSCOPE_API_KEY"; exit 1
    fi

    find_train_json
    echo "  训练数据: ${TRAIN_JSON}"

    mkdir -p "${COT_PILOT_DIR}"

    $PYTHON_EXEC "${SCRIPT_DIR}/cot_generate.py" \
        --mode pilot \
        --train_data "${TRAIN_JSON}" \
        --output_dir "${COT_PILOT_DIR}" \
        --api_key "${DASHSCOPE_API_KEY}" \
        --api_model "${API_MODEL}" \
        --num_pilot 1000 \
        --workers 4

    echo ""
    echo "  ✅ Step 2 完成。"
    echo "  查看结果: ${COT_PILOT_DIR}/cot_pilot_results.json"
    echo "  SFT 数据: ${COT_PILOT_DIR}/cot_pilot_sft.jsonl"
    echo ""
    echo "  如果 Binary 匹配率 > 85%, 请运行: bash run.sh step3"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3: CoT 全量重构
# ═══════════════════════════════════════════════════════════════════════════════
do_step3() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 3: CoT 全量重构 (9600条, 约19元)                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    if [ -z "${DASHSCOPE_API_KEY}" ]; then
        echo "❌ 请在脚本顶部填入 DASHSCOPE_API_KEY"; exit 1
    fi

    find_train_json
    mkdir -p "${COT_DIR}"

    $PYTHON_EXEC "${SCRIPT_DIR}/cot_generate.py" \
        --mode full \
        --train_data "${TRAIN_JSON}" \
        --output_dir "${COT_DIR}" \
        --api_key "${DASHSCOPE_API_KEY}" \
        --api_model "${API_MODEL}" \
        --workers 4

    echo ""
    echo "  ✅ Step 3 完成。"
    echo "  SFT 数据: ${COT_DIR}/cot_full_sft.jsonl"
    echo ""
    echo "  下一步: bash run.sh step4"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4: CoT SFT 训练
# ═══════════════════════════════════════════════════════════════════════════════
do_step4() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 4: CoT SFT 训练 (约2-4小时)                           ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # 优先使用全量数据, 否则用 pilot 数据
    SFT_DATA="${COT_DIR}/cot_full_sft.jsonl"
    if [ ! -f "${SFT_DATA}" ]; then
        SFT_DATA="${COT_PILOT_DIR}/cot_pilot_sft.jsonl"
    fi

    if [ ! -f "${SFT_DATA}" ]; then
        echo "❌ 找不到 SFT 数据，请先运行 step2 或 step3"; exit 1
    fi

    if [ ! -d "${BASE_MODEL}" ]; then
        echo "❌ 基础模型不存在: ${BASE_MODEL}"; exit 1
    fi

    # 查找 LoRA 训练脚本
    LORA_TRAINER=""
    for p in \
        "${SCRIPTS_DIR}/model_lora/train_reviewer_lora.py" \
        "${SELFPLAY_DIR}/../model_lora/train_reviewer_lora.py"; do
        if [ -f "$p" ]; then LORA_TRAINER="$p"; break; fi
    done

    if [ -z "${LORA_TRAINER}" ]; then
        echo "❌ 找不到 train_reviewer_lora.py"; exit 1
    fi

    SFT_COUNT=$(wc -l < "${SFT_DATA}")
    echo "  SFT 数据: ${SFT_DATA} (${SFT_COUNT} 条)"
    echo "  基础模型: ${BASE_MODEL}"
    echo "  输出目录: ${REVIEWER_COT_LORA}"
    echo "  设备: ${N_GPUS} NPUs"

    mkdir -p "${REVIEWER_COT_LORA}"

    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=1200
    export HCCL_WHITELIST_DISABLE=1

    GRAD_ACCUM=$(( 32 / 4 / N_GPUS ))
    [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

    $PYTHON_EXEC -m torch.distributed.run \
        --standalone --nproc_per_node=$N_GPUS \
        "${LORA_TRAINER}" \
        --model_path "${BASE_MODEL}" \
        --data_path "${SFT_DATA}" \
        --output_dir "${REVIEWER_COT_LORA}" \
        --lora_rank 32 \
        --lora_alpha 64 \
        --batch_size 4 \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs 3 \
        --learning_rate 2e-4 \
        --max_length 2048 \
        --seed 42 \
        --device "npu:0" \
        --n_devices $N_GPUS

    echo ""
    echo "  ✅ Step 4 完成。LoRA 模型: ${REVIEWER_COT_LORA}"
    echo ""
    echo "  下一步: 合并 LoRA 并评估:"
    echo "    bash run.sh step5_eval"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4b: Baseline SFT (same 1000 samples, NO CoT) — for ablation
# ═══════════════════════════════════════════════════════════════════════════════
REVIEWER_BASELINE_LORA="${BASE_DIR}/cot_models/reviewer_baseline_${MODEL_SIZE}"
REVIEWER_BASELINE_MERGED="${BASE_DIR}/cot_models/reviewer_baseline_${MODEL_SIZE}_merged"

do_step4_baseline() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 4b: Baseline SFT (同样1000条, 无CoT) — 消融对比        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    PILOT_RESULTS="${COT_PILOT_DIR}/cot_pilot_results.json"
    if [ ! -f "${PILOT_RESULTS}" ]; then
        echo "ERROR: ${PILOT_RESULTS} not found. Run step2 first."; exit 1
    fi

    BASELINE_SFT="${COT_PILOT_DIR}/baseline_sft.jsonl"

    # Build baseline SFT data (same 1000 texts, direct labels, no CoT)
    echo "  [1/2] 生成 baseline SFT 数据 (无 CoT)..."
    $PYTHON_EXEC "${SCRIPT_DIR}/build_baseline_sft.py" \
        --pilot_results "${PILOT_RESULTS}" \
        --output "${BASELINE_SFT}"

    # Train
    echo "  [2/2] 训练 baseline 模型..."

    if [ ! -d "${BASE_MODEL}" ]; then
        echo "ERROR: ${BASE_MODEL} not found"; exit 1
    fi

    LORA_TRAINER=""
    for p in \
        "${SCRIPTS_DIR}/model_lora/train_reviewer_lora.py" \
        "${SELFPLAY_DIR}/../model_lora/train_reviewer_lora.py"; do
        if [ -f "$p" ]; then LORA_TRAINER="$p"; break; fi
    done
    if [ -z "${LORA_TRAINER}" ]; then
        echo "ERROR: train_reviewer_lora.py not found"; exit 1
    fi

    mkdir -p "${REVIEWER_BASELINE_LORA}"

    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=1200
    export HCCL_WHITELIST_DISABLE=1

    GRAD_ACCUM=$(( 32 / 4 / N_GPUS ))
    [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

    $PYTHON_EXEC -m torch.distributed.run \
        --standalone --nproc_per_node=$N_GPUS \
        "${LORA_TRAINER}" \
        --model_path "${BASE_MODEL}" \
        --data_path "${BASELINE_SFT}" \
        --output_dir "${REVIEWER_BASELINE_LORA}" \
        --lora_rank 32 \
        --lora_alpha 64 \
        --batch_size 4 \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs 3 \
        --learning_rate 2e-4 \
        --max_length 2048 \
        --seed 42 \
        --device "npu:0" \
        --n_devices $N_GPUS

    echo ""
    echo "  ✅ Step 4b 完成。Baseline LoRA: ${REVIEWER_BASELINE_LORA}"
    echo "  下一步: bash run.sh step5_baseline"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5b: 合并 + 评估 Baseline (无 CoT)
# ═══════════════════════════════════════════════════════════════════════════════
do_step5_baseline() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 5b: 合并 + 评估 Baseline 模型 (无 CoT)                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    if [ ! -d "${REVIEWER_BASELINE_LORA}" ]; then
        echo "ERROR: ${REVIEWER_BASELINE_LORA} not found. Run step4_baseline first."; exit 1
    fi

    MERGE_SCRIPT=""
    for p in \
        "${SCRIPTS_DIR}/model_lora/merge_lora.py" \
        "${SCRIPTS_DIR}/model_lora/merge_lora_weights.py" \
        "${SELFPLAY_DIR}/../model_lora/merge_lora.py"; do
        if [ -f "$p" ]; then MERGE_SCRIPT="$p"; break; fi
    done

    if [ -z "${MERGE_SCRIPT}" ]; then
        echo "  手动合并..."
        $PYTHON_EXEC - <<PYEOF
import torch, os
try:
    import torch_npu
except: pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
print("  Loading base model...")
base = AutoModelForCausalLM.from_pretrained("${BASE_MODEL}", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
print("  Loading LoRA...")
model = PeftModel.from_pretrained(base, "${REVIEWER_BASELINE_LORA}")
print("  Merging...")
merged = model.merge_and_unload()
os.makedirs("${REVIEWER_BASELINE_MERGED}", exist_ok=True)
merged.save_pretrained("${REVIEWER_BASELINE_MERGED}")
AutoTokenizer.from_pretrained("${BASE_MODEL}", trust_remote_code=True).save_pretrained("${REVIEWER_BASELINE_MERGED}")
print("  Done")
PYEOF
    else
        $PYTHON_EXEC "${MERGE_SCRIPT}" \
            --base_model "${BASE_MODEL}" \
            --lora_path "${REVIEWER_BASELINE_LORA}" \
            --output_path "${REVIEWER_BASELINE_MERGED}"
    fi

    EVAL_SCRIPT=""
    for p in \
        "${SCRIPTS_DIR}/model_eval/batch_eval_npu_vllm.py" \
        "${SELFPLAY_DIR}/../model_eval/batch_eval_npu_vllm.py"; do
        if [ -f "$p" ]; then EVAL_SCRIPT="$p"; break; fi
    done

    if [ -z "${EVAL_SCRIPT}" ]; then
        echo "  WARNING: eval script not found, evaluate manually"; return
    fi

    EVAL_OUT="${BASE_DIR}/eval_baseline_${MODEL_SIZE}"
    mkdir -p "${EVAL_OUT}"

    echo "  评估 Baseline 模型..."
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${EVAL_SCRIPT}" \
        --data_path "${TEST_JSON}" \
        --model_path "${REVIEWER_BASELINE_MERGED}" \
        --output_dir "${EVAL_OUT}" \
        --num_npus "${N_GPUS}" \
        --tag "baseline_1000" \
        --batch_size 128

    echo ""
    echo "  ✅ Step 5b 完成。"
    echo "  Baseline 评估结果: ${EVAL_OUT}/"
    echo ""
    echo "  比较:"
    echo "    原始 SFT:       78.18%"
    echo "    Baseline 1000:  → 查看 ${EVAL_OUT}/"
    echo "    CoT 1000:       → bash run.sh step5_eval"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5: 合并 LoRA + 评估 (CoT)
# ═══════════════════════════════════════════════════════════════════════════════
do_step5_eval() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 5: 合并 LoRA + 评估 CoT 模型                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    if [ ! -d "${REVIEWER_COT_LORA}" ]; then
        echo "❌ LoRA 模型不存在: ${REVIEWER_COT_LORA}"
        echo "   请先运行 step4"; exit 1
    fi

    # ── 合并 LoRA ──
    MERGE_SCRIPT=""
    for p in \
        "${SCRIPTS_DIR}/model_lora/merge_lora.py" \
        "${SCRIPTS_DIR}/model_lora/merge_lora_weights.py" \
        "${SELFPLAY_DIR}/../model_lora/merge_lora.py"; do
        if [ -f "$p" ]; then MERGE_SCRIPT="$p"; break; fi
    done

    if [ -z "${MERGE_SCRIPT}" ]; then
        echo "  ⚠️ 找不到 merge 脚本，尝试手动合并..."
        # 手动合并 (fallback)
        $PYTHON_EXEC - <<PYEOF
import torch, os, sys
try:
    import torch_npu
except: pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("  加载基础模型...")
base = AutoModelForCausalLM.from_pretrained("${BASE_MODEL}", torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
print("  加载 LoRA...")
model = PeftModel.from_pretrained(base, "${REVIEWER_COT_LORA}")
print("  合并权重...")
merged = model.merge_and_unload()
print("  保存到 ${REVIEWER_COT_MERGED}")
os.makedirs("${REVIEWER_COT_MERGED}", exist_ok=True)
merged.save_pretrained("${REVIEWER_COT_MERGED}")
tok = AutoTokenizer.from_pretrained("${BASE_MODEL}", trust_remote_code=True)
tok.save_pretrained("${REVIEWER_COT_MERGED}")
print("  ✓ 合并完成")
PYEOF
    else
        echo "  合并 LoRA: ${MERGE_SCRIPT}"
        $PYTHON_EXEC "${MERGE_SCRIPT}" \
            --base_model "${BASE_MODEL}" \
            --lora_path "${REVIEWER_COT_LORA}" \
            --output_path "${REVIEWER_COT_MERGED}"
    fi

    echo ""
    echo "  合并模型: ${REVIEWER_COT_MERGED}"

    # ── 评估 ──
    EVAL_SCRIPT=""
    for p in \
        "${SCRIPTS_DIR}/model_eval/batch_eval_npu_vllm.py" \
        "${SELFPLAY_DIR}/../model_eval/batch_eval_npu_vllm.py"; do
        if [ -f "$p" ]; then EVAL_SCRIPT="$p"; break; fi
    done

    if [ -z "${EVAL_SCRIPT}" ]; then
        echo "  ⚠️ 找不到评估脚本，请手动评估"; return
    fi

    if [ ! -f "${TEST_JSON}" ]; then
        echo "  ⚠️ 找不到测试数据 ${TEST_JSON}，请手动评估"; return
    fi

    EVAL_OUT="${BASE_DIR}/eval_cot_${MODEL_SIZE}"
    mkdir -p "${EVAL_OUT}"

    echo "  评估 CoT 模型..."
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${EVAL_SCRIPT}" \
        --data_path "${TEST_JSON}" \
        --model_path "${REVIEWER_COT_MERGED}" \
        --output_dir "${EVAL_OUT}" \
        --num_npus "${N_GPUS}" \
        --tag "cot_baseline" \
        --batch_size 128

    echo ""
    echo "  ✅ Step 5 完成。"
    echo "  评估结果: ${EVAL_OUT}/"
    echo ""
    echo "  请与原始 SFT baseline (78.18%) 比较。"
    echo "  如果 CoT 准确率 > 80%, 请运行: bash run.sh step6"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6: Self-Play (含所有修复)
# ═══════════════════════════════════════════════════════════════════════════════
do_step6() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Step 6: Self-Play 对抗训练 (含所有修复)                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # 检查工程修复是否已应用
    if ! grep -q 'REVIEWER_MIX_RATIO="${REVIEWER_MIX_RATIO:-0.5}"' "${SELFPLAY_SH}" 2>/dev/null; then
        echo "  ⚠️ 工程修复未应用，先执行 step1..."
        do_step1
    fi

    # 选择 Reviewer 初始模型 (优先 CoT)
    if [ -d "${REVIEWER_COT_MERGED}" ]; then
        INIT_REVIEWER="${REVIEWER_COT_MERGED}"
        echo "  Reviewer 初始化: CoT 模型 (${REVIEWER_COT_MERGED})"
    else
        INIT_REVIEWER="${REVIEWER_SFT}"
        echo "  Reviewer 初始化: 原始 SFT 模型 (${REVIEWER_SFT})"
    fi

    # 修改 run_selfplay.sh 中的 REVIEWER_INIT
    sed -i "s|REVIEWER_INIT=.*|REVIEWER_INIT=\"${INIT_REVIEWER}\"|" "${SELFPLAY_SH}" 2>/dev/null || true

    echo ""
    echo "  启动 Self-Play (50 步)..."
    echo ""

    cd "${SELFPLAY_DIR}"
    BASE_DIR="${BASE_DIR}" \
    MODEL_SIZE="${MODEL_SIZE}" \
    N_GPUS="${N_GPUS}" \
    TOTAL_STEPS=50 \
    PYTHON_EXEC="${PYTHON_EXEC}" \
    bash run_selfplay.sh
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Plan
# ═══════════════════════════════════════════════════════════════════════════════
do_plan() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                     验证计划总览                              ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║                                                            ║"
    echo "║  bash run.sh step1       工程修复 (免费, 1分钟)              ║"
    echo "║    → 修复 MIX_RATIO, penalty, 数据量, REINFORCE             ║"
    echo "║                                                            ║"
    echo "║  bash run.sh step2       CoT pilot 1000条 (~2元, 15分钟)    ║"
    echo "║    → 验证 API CoT 质量, 匹配率>85%则继续                     ║"
    echo "║                                                            ║"
    echo "║  bash run.sh step3       CoT 全量重构 (~19元, 30分钟)        ║"
    echo "║    → 为全部 9600 条训练数据生成 CoT                           ║"
    echo "║                                                            ║"
    echo "║  bash run.sh step4       CoT SFT 训练 (2-4小时)             ║"
    echo "║    → 训练新 Reviewer, 预期 78%→83-86%                       ║"
    echo "║                                                            ║"
    echo "║  bash run.sh step5_eval  合并 LoRA + 评估                    ║"
    echo "║    → 与 78% baseline 比较                                   ║"
    echo "║                                                            ║"
    echo "║  bash run.sh step6       Self-Play 对抗训练                  ║"
    echo "║    → 在 CoT baseline 上运行自对弈, 预期再提升 2-4%            ║"
    echo "║                                                            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  总 API 费用: 约 21 元"
    echo ""
    echo "  ★ 开始前请在脚本顶部填写 DASHSCOPE_API_KEY"
    echo ""
    echo "  决策流程:"
    echo "    step1 → step2 → 匹配率>85%? → step3 → step4 → step5_eval"
    echo "    → 准确率>80%? → step6"
}

# ═══════════════════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════════════════
case "${1:-plan}" in
    plan)           do_plan ;;
    step1)          do_step1 ;;
    step2)          do_step2 ;;
    step3)          do_step3 ;;
    step4)          do_step4 ;;
    step4_baseline) do_step4_baseline ;;
    step5_eval)     do_step5_eval ;;
    step5_baseline) do_step5_baseline ;;
    step6)          do_step6 ;;
    *)
        echo "Usage: bash run.sh [plan|step1|step2|step3|step4|step4_baseline|step5_eval|step5_baseline|step6]"
        exit 1 ;;
esac
