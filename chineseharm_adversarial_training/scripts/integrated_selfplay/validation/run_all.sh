#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════════════════════
#  ALL-IN-ONE: CoT vs Baseline 消融实验
# ═══════════════════════════════════════════════════════════════════════════════
#  一个脚本完成全部流程:
#    1. 从训练集分层采样 1000 条
#    2. 调用 API 生成 CoT (基于给定标签生成推理)
#    3. 用同样 1000 条构建 baseline SFT 数据 (无 CoT)
#    4. 分别训练两个模型: CoT版 vs Baseline版
#    5. 分别合并 LoRA + 评估
#    6. 输出对比结果
#
#  用法:
#    vi run_all.sh   # 修改下面的配置
#    bash run_all.sh
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
#  ★★★ 配置区 ★★★
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR="/home/ma-user/work/test"
PYTHON_EXEC="/home/ma-user/.conda/envs/ssp_train/bin/python"
DASHSCOPE_API_KEY=""              # ← 填入你的 API Key
API_MODEL="qwen-plus"
MODEL_SIZE="3B"
N_GPUS=4
NUM_SAMPLES=1000

# ─────────────────────────────────────────────────────────────────────────────
#  路径
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SELFPLAY_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$(dirname "$SELFPLAY_DIR")"

# 找 train.json
TRAIN_JSON=""
for p in \
    "${BASE_DIR}/split_data/train.json" \
    "${BASE_DIR}/chineseharm_adversarial_training/split_data/train.json" \
    "${BASE_DIR}/train.json"; do
    if [ -f "$p" ]; then TRAIN_JSON="$p"; break; fi
done
if [ -z "$TRAIN_JSON" ]; then echo "ERROR: train.json not found"; exit 1; fi

# 找 test.json
TEST_JSON=""
for p in \
    "${BASE_DIR}/split_data/test.json" \
    "${BASE_DIR}/chineseharm_adversarial_training/split_data/test.json" \
    "${BASE_DIR}/test.json"; do
    if [ -f "$p" ]; then TEST_JSON="$p"; break; fi
done
if [ -z "$TEST_JSON" ]; then echo "ERROR: test.json not found"; exit 1; fi

BASE_MODEL="${BASE_DIR}/models_base/Qwen/Qwen2.5-${MODEL_SIZE}-Instruct"
COT_DIR="${BASE_DIR}/cot_experiment"

COT_LORA="${COT_DIR}/lora_cot"
COT_MERGED="${COT_DIR}/merged_cot"
BASELINE_LORA="${COT_DIR}/lora_baseline"
BASELINE_MERGED="${COT_DIR}/merged_baseline"

# 找 LoRA 训练脚本
LORA_TRAINER=""
for p in \
    "${SCRIPTS_DIR}/model_lora/train_reviewer_lora.py" \
    "${SELFPLAY_DIR}/../model_lora/train_reviewer_lora.py"; do
    if [ -f "$p" ]; then LORA_TRAINER="$p"; break; fi
done
if [ -z "$LORA_TRAINER" ]; then echo "ERROR: train_reviewer_lora.py not found"; exit 1; fi

# 找 merge 脚本
MERGE_SCRIPT=""
for p in \
    "${SCRIPTS_DIR}/model_lora/merge_lora.py" \
    "${SELFPLAY_DIR}/../model_lora/merge_lora.py"; do
    if [ -f "$p" ]; then MERGE_SCRIPT="$p"; break; fi
done

# 找 eval 脚本
EVAL_SCRIPT=""
for p in \
    "${SCRIPTS_DIR}/model_eval/batch_eval_npu_vllm.py" \
    "${SELFPLAY_DIR}/../model_eval/batch_eval_npu_vllm.py"; do
    if [ -f "$p" ]; then EVAL_SCRIPT="$p"; break; fi
done

# 昇腾环境
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1

GRAD_ACCUM=$(( 32 / 4 / N_GPUS ))
[ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1

# ─────────────────────────────────────────────────────────────────────────────
#  检查
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CoT vs Baseline 消融实验                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  训练数据:    ${TRAIN_JSON}"
echo "  测试数据:    ${TEST_JSON}"
echo "  基础模型:    ${BASE_MODEL}"
echo "  样本数:      ${NUM_SAMPLES}"
echo "  API Model:   ${API_MODEL}"
echo "  NPUs:        ${N_GPUS}"
echo "  输出目录:    ${COT_DIR}"
echo ""

if [ -z "${DASHSCOPE_API_KEY}" ]; then
    echo "ERROR: DASHSCOPE_API_KEY is empty. Edit this script line 19."
    exit 1
fi
if [ ! -d "${BASE_MODEL}" ]; then
    echo "ERROR: Base model not found: ${BASE_MODEL}"
    exit 1
fi

mkdir -p "${COT_DIR}"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: 生成 CoT 数据 (调用 API, ~2元)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [1/6] 生成 CoT 数据 (${NUM_SAMPLES} 条, API=${API_MODEL})"
echo "══════════════════════════════════════════════════════════════"

$PYTHON_EXEC "${SCRIPT_DIR}/cot_generate.py" \
    --mode pilot \
    --train_data "${TRAIN_JSON}" \
    --output_dir "${COT_DIR}" \
    --api_key "${DASHSCOPE_API_KEY}" \
    --api_model "${API_MODEL}" \
    --num_pilot "${NUM_SAMPLES}" \
    --workers 4

COT_SFT="${COT_DIR}/cot_pilot_sft.jsonl"
COT_RESULTS="${COT_DIR}/cot_pilot_results.json"

if [ ! -f "${COT_SFT}" ]; then
    echo "ERROR: CoT SFT data not generated"; exit 1
fi
echo "  CoT SFT: ${COT_SFT} ($(wc -l < "${COT_SFT}") rows)"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: 生成 Baseline SFT 数据 (同样1000条, 无CoT)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [2/6] 生成 Baseline SFT 数据 (同样 ${NUM_SAMPLES} 条, 无 CoT)"
echo "══════════════════════════════════════════════════════════════"

BASELINE_SFT="${COT_DIR}/baseline_sft.jsonl"

$PYTHON_EXEC "${SCRIPT_DIR}/build_baseline_sft.py" \
    --pilot_results "${COT_RESULTS}" \
    --output "${BASELINE_SFT}"

echo "  Baseline SFT: ${BASELINE_SFT} ($(wc -l < "${BASELINE_SFT}") rows)"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: 训练 Baseline 模型 (无 CoT)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [3/6] 训练 Baseline 模型 (无 CoT)"
echo "══════════════════════════════════════════════════════════════"

mkdir -p "${BASELINE_LORA}"

$PYTHON_EXEC -m torch.distributed.run \
    --standalone --nproc_per_node=$N_GPUS \
    "${LORA_TRAINER}" \
    --model_path "${BASE_MODEL}" \
    --data_path "${BASELINE_SFT}" \
    --output_dir "${BASELINE_LORA}" \
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

echo "  Baseline LoRA: ${BASELINE_LORA}"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: 训练 CoT 模型
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [4/6] 训练 CoT 模型"
echo "══════════════════════════════════════════════════════════════"

mkdir -p "${COT_LORA}"

$PYTHON_EXEC -m torch.distributed.run \
    --standalone --nproc_per_node=$N_GPUS \
    "${LORA_TRAINER}" \
    --model_path "${BASE_MODEL}" \
    --data_path "${COT_SFT}" \
    --output_dir "${COT_LORA}" \
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

echo "  CoT LoRA: ${COT_LORA}"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: 合并 LoRA (两个模型)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [5/6] 合并 LoRA 权重"
echo "══════════════════════════════════════════════════════════════"

merge_model() {
    local LORA_PATH="$1"
    local OUTPUT_PATH="$2"
    local NAME="$3"

    echo "  合并 ${NAME}..."
    if [ -n "${MERGE_SCRIPT}" ]; then
        $PYTHON_EXEC "${MERGE_SCRIPT}" \
            --base_model "${BASE_MODEL}" \
            --lora_path "${LORA_PATH}" \
            --output_path "${OUTPUT_PATH}"
    else
        $PYTHON_EXEC -c "
import torch, os
try:
    import torch_npu
except: pass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained('${BASE_MODEL}', torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
model = PeftModel.from_pretrained(base, '${LORA_PATH}')
merged = model.merge_and_unload()
os.makedirs('${OUTPUT_PATH}', exist_ok=True)
merged.save_pretrained('${OUTPUT_PATH}')
AutoTokenizer.from_pretrained('${BASE_MODEL}', trust_remote_code=True).save_pretrained('${OUTPUT_PATH}')
print('  Done: ${OUTPUT_PATH}')
"
    fi
}

merge_model "${BASELINE_LORA}" "${BASELINE_MERGED}" "Baseline"
merge_model "${COT_LORA}" "${COT_MERGED}" "CoT"

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: 评估两个模型
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  [6/6] 评估两个模型"
echo "══════════════════════════════════════════════════════════════"

if [ -z "${EVAL_SCRIPT}" ]; then
    echo "  WARNING: eval script not found. Please evaluate manually:"
    echo "    Baseline: ${BASELINE_MERGED}"
    echo "    CoT:      ${COT_MERGED}"
else
    EVAL_BASELINE_DIR="${COT_DIR}/eval_baseline"
    EVAL_COT_DIR="${COT_DIR}/eval_cot"
    mkdir -p "${EVAL_BASELINE_DIR}" "${EVAL_COT_DIR}"

    echo "  [6a] 评估 Baseline 模型..."
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${EVAL_SCRIPT}" \
        --data_path "${TEST_JSON}" \
        --model_path "${BASELINE_MERGED}" \
        --output_dir "${EVAL_BASELINE_DIR}" \
        --num_npus "${N_GPUS}" \
        --tag "baseline_${NUM_SAMPLES}" \
        --batch_size 128

    echo ""
    echo "  [6b] 评估 CoT 模型..."
    ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1))) \
    $PYTHON_EXEC "${EVAL_SCRIPT}" \
        --data_path "${TEST_JSON}" \
        --model_path "${COT_MERGED}" \
        --output_dir "${EVAL_COT_DIR}" \
        --num_npus "${N_GPUS}" \
        --tag "cot_${NUM_SAMPLES}" \
        --batch_size 128

    # 提取结果对比
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  实验结果对比"
    echo "══════════════════════════════════════════════════════════════"

    $PYTHON_EXEC - <<PYEOF
import json, glob, os

def load_eval(eval_dir):
    files = glob.glob(os.path.join(eval_dir, "eval_*.json"))
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)

baseline = load_eval("${EVAL_BASELINE_DIR}")
cot = load_eval("${EVAL_COT_DIR}")

cats = ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨', '无毒']

print(f"{'':20s} {'Baseline':>12s} {'CoT':>12s} {'Delta':>12s}")
print("=" * 60)

if baseline and cot:
    bm = baseline['metrics']
    cm = cot['metrics']

    # Overall
    ba = bm.get('overall_accuracy', 0)
    ca = cm.get('overall_accuracy', 0)
    print(f"{'Overall Accuracy':20s} {ba:12.2f} {ca:12.2f} {ca-ba:+12.2f}")

    bf1 = bm.get('macro_f1', 0)
    cf1 = cm.get('macro_f1', 0)
    print(f"{'Macro F1':20s} {bf1:12.4f} {cf1:12.4f} {cf1-bf1:+12.4f}")

    bb = bm.get('binary_metrics', {})
    cb = cm.get('binary_metrics', {})
    for k in ['accuracy', 'precision', 'recall', 'f1_score']:
        bv = bb.get(k, 0)
        cv = cb.get(k, 0)
        if bv < 1: bv *= 100
        if cv < 1: cv *= 100
        print(f"{'Binary '+k:20s} {bv:12.2f} {cv:12.2f} {cv-bv:+12.2f}")

    print()
    print(f"{'Category':14s} {'Baseline F1':>12s} {'CoT F1':>12s} {'Delta':>12s}")
    print("-" * 55)
    for cat in cats:
        bf = bm.get('category_metrics', {}).get(cat, {}).get('f1_score', 0)
        cf = cm.get('category_metrics', {}).get(cat, {}).get('f1_score', 0)
        print(f"{cat:14s} {bf:12.4f} {cf:12.4f} {cf-bf:+12.4f}")
else:
    if not baseline: print("  WARNING: Baseline eval not found")
    if not cot: print("  WARNING: CoT eval not found")

print()
print("  Original full-data SFT baseline: 78.18%")
PYEOF
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ALL DONE!"
echo "══════════════════════════════════════════════════════════════"
echo "  输出目录:     ${COT_DIR}"
echo "  Baseline 模型: ${BASELINE_MERGED}"
echo "  CoT 模型:      ${COT_MERGED}"
echo "  Baseline 评估: ${COT_DIR}/eval_baseline/"
echo "  CoT 评估:      ${COT_DIR}/eval_cot/"
echo "══════════════════════════════════════════════════════════════"
