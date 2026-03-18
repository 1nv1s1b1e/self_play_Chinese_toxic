#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  华为云 ModelArts 一键启动脚本  —  集成版对抗博弈自对弈训练                    ║
# ║                                                                              ║
# ║  环境: ModelArts Notebook (昇腾 910B, 4卡)                                    ║
# ║  Verifier: 使用 Qwen DashScope API (节省 NPU 显存)                            ║
# ║                                                                              ║
# ║  前置条件:                                                                     ║
# ║    1. 已运行 Step 1-6 (下载模型 + 数据准备 + LoRA SFT + 合并)                   ║
# ║    2. /home/ma-user/work/test/ 下有:                                           ║
# ║       - merged_models_toxicn/challenger_3B                                     ║
# ║       - merged_models_toxicn/reviewer_3B                                       ║
# ║       - prepared_data/rl/train_seed.parquet                                    ║
# ║    3. 已获取阿里云 DashScope API Key                                            ║
# ║                                                                              ║
# ║  用法:                                                                         ║
# ║    # 方式一: 一行搞定                                                           ║
# ║    DASHSCOPE_API_KEY="sk-xxx" bash launch_modelarts.sh                         ║
# ║                                                                              ║
# ║    # 方式二: 自定义参数                                                          ║
# ║    DASHSCOPE_API_KEY="sk-xxx" MODEL_SIZE=1.5B SELFPLAY_ROUNDS=3 \             ║
# ║        bash launch_modelarts.sh                                                ║
# ║                                                                              ║
# ║    # 方式三: 只做环境检查，不训练                                                  ║
# ║    DASHSCOPE_API_KEY="sk-xxx" CHECK_ONLY=1 bash launch_modelarts.sh            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
set -e

# ─────────────────────────────────────────────────────────────────────────────────
# 0. 颜色输出
# ─────────────────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[⚠]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; }

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║   ModelArts 集成版对抗博弈自对弈训练 — 一键启动                          ║"
echo "║   (integrated_selfplay: Plan 1 + Plan 3 + Plan 4)                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

# ─────────────────────────────────────────────────────────────────────────────────
# 1. 参数配置
# ─────────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_SIZE="${MODEL_SIZE:-3B}"
N_GPUS="${N_GPUS:-4}"
SELFPLAY_ROUNDS="${SELFPLAY_ROUNDS:-3}"
MAX_STEPS="${MAX_STEPS:-50}"
SAMPLES_PER_CAT="${SAMPLES_PER_CAT:-128}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# Python 解释器
PYTHON_EXEC="${PYTHON_EXEC:-/home/ma-user/.conda/envs/ssp_train/bin/python}"
if [ ! -f "${PYTHON_EXEC}" ]; then
    PYTHON_EXEC="$(which python3 2>/dev/null || which python 2>/dev/null)"
    warn "ssp_train 环境不存在，使用系统 Python: ${PYTHON_EXEC}"
fi

# ── DashScope API Key ──
# 集成版 verifier.py 使用 VERIFIER_API_KEY 环境变量，同时兼容 DASHSCOPE_API_KEY
if [ -z "${DASHSCOPE_API_KEY}" ] && [ -z "${VERIFIER_API_KEY}" ]; then
    fail "请设置 DASHSCOPE_API_KEY 环境变量！"
    echo "  用法: DASHSCOPE_API_KEY=\"sk-xxx\" bash launch_modelarts.sh"
    echo "  获取: https://dashscope.console.aliyun.com/apiKey"
    exit 1
fi
# 统一: VERIFIER_API_KEY 优先，否则用 DASHSCOPE_API_KEY
export VERIFIER_API_KEY="${VERIFIER_API_KEY:-${DASHSCOPE_API_KEY}}"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-${VERIFIER_API_KEY}}"

# Verifier 配置: 使用 Qwen API 作为 Verifier（节省 NPU 显存）
VERIFIER_BACKEND="${VERIFIER_BACKEND:-api}"
VERIFIER_API_MODEL="${VERIFIER_API_MODEL:-qwen-plus}"
# 当 backend=api 时，VERIFIER_MODEL 仅作日志显示，不加载本地模型
VERIFIER_MODEL="${VERIFIER_MODEL:-api:${VERIFIER_API_MODEL}}"

echo ""
info "训练配置:"
echo "  ├─ 模型尺寸       : ${MODEL_SIZE}"
echo "  ├─ NPU 卡数       : ${N_GPUS}"
echo "  ├─ 自对弈轮次     : ${SELFPLAY_ROUNDS}"
echo "  ├─ 每阶段步数     : ${MAX_STEPS}"
echo "  ├─ 每类别样本数   : ${SAMPLES_PER_CAT}"
echo "  ├─ Verifier 后端  : ${VERIFIER_BACKEND} (${VERIFIER_API_MODEL})"
echo "  ├─ 基础目录       : ${BASE_DIR}"
echo "  └─ Python         : ${PYTHON_EXEC}"

# ─────────────────────────────────────────────────────────────────────────────────
# 2. 环境检查
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
info "═══════ 环境检查 ═══════"
ERRORS=0

# 2.1 CANN 环境
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

if $PYTHON_EXEC -c "import torch_npu; import torch; assert torch.npu.device_count() >= 1" 2>/dev/null; then
    NPU_COUNT=$($PYTHON_EXEC -c "import torch_npu; import torch; print(torch.npu.device_count())" 2>/dev/null)
    ok "NPU 设备: ${NPU_COUNT} 张卡可用"
else
    fail "NPU 设备不可用 (需要 torch_npu)"
    ERRORS=$((ERRORS+1))
fi

# 2.2 核心 Python 库检查
for pkg in "torch" "transformers" "trl" "peft" "datasets" "pandas" "openai"; do
    if $PYTHON_EXEC -c "import ${pkg}" 2>/dev/null; then
        ok "${pkg}"
    else
        if [ "${pkg}" = "openai" ]; then
            warn "${pkg} 未安装 (Verifier API 需要)，正在安装..."
            $PYTHON_EXEC -m pip install openai -q && ok "openai 已安装" || { fail "openai 安装失败"; ERRORS=$((ERRORS+1)); }
        else
            fail "${pkg} 未安装"
            ERRORS=$((ERRORS+1))
        fi
    fi
done

# 2.3 模型文件检查
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_${MODEL_SIZE}"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_${MODEL_SIZE}"
SEED_DATA="${BASE_DIR}/prepared_data/rl/train_seed.parquet"

for path_label in "Challenger模型:${CHALLENGER_INIT}" "Reviewer模型:${REVIEWER_INIT}"; do
    label="${path_label%%:*}"; path="${path_label##*:}"
    if [ -d "${path}" ]; then
        ok "${label}: ${path}"
    else
        fail "${label} 不存在: ${path}"
        ERRORS=$((ERRORS+1))
    fi
done

if [ -f "${SEED_DATA}" ]; then
    ok "种子数据: ${SEED_DATA}"
else
    fail "种子数据不存在: ${SEED_DATA}"
    ERRORS=$((ERRORS+1))
fi

# 2.4 DashScope API 连通性检查
info "测试 DashScope API 连接..."
API_TEST=$($PYTHON_EXEC -c "
from openai import OpenAI
client = OpenAI(
    api_key='${VERIFIER_API_KEY}',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)
try:
    resp = client.chat.completions.create(
        model='${VERIFIER_API_MODEL}',
        messages=[{'role':'user','content':'回复OK两个字母'}],
        max_tokens=5, temperature=0.0
    )
    print('OK:' + resp.choices[0].message.content.strip()[:20])
except Exception as e:
    print('FAIL:' + str(e)[:100])
" 2>/dev/null)

if [[ "${API_TEST}" == OK:* ]]; then
    ok "DashScope API (${VERIFIER_API_MODEL}) 连接正常: ${API_TEST#OK:}"
else
    fail "DashScope API 连接失败: ${API_TEST#FAIL:}"
    echo "  请检查: 1) API Key 是否正确  2) 网络是否通畅  3) 模型名称是否正确"
    ERRORS=$((ERRORS+1))
fi

# 2.5 LoRA SFT 脚本检查 (Phase B 需要)
LORA_TRAIN="${SCRIPT_DIR}/../model_lora/train_reviewer_lora.py"
LORA_MERGE="${SCRIPT_DIR}/../model_lora/merge_lora.py"
if [ -f "${LORA_TRAIN}" ] && [ -f "${LORA_MERGE}" ]; then
    ok "LoRA SFT 脚本 (Phase B)"
else
    fail "LoRA SFT 脚本缺失 (需要 ../model_lora/ 下 train_reviewer_lora.py + merge_lora.py)"
    ERRORS=$((ERRORS+1))
fi

# 2.6 DeepSpeed 配置检查
DS_CONFIG="${SCRIPT_DIR}/ds_zero2.json"
if [ -f "${DS_CONFIG}" ]; then
    ok "DeepSpeed 配置: ${DS_CONFIG}"
else
    fail "DeepSpeed 配置缺失: ${DS_CONFIG}"
    ERRORS=$((ERRORS+1))
fi

# 2.7 集成版关键文件检查
REQUIRED_FILES=(
    "constants.py" "quality_gate.py" "rejection_sampler.py"
    "challenger_reward.py" "build_parquet.py" "verifier.py"
    "generate_dynamic_data.py" "adversarial_trl_grpo.py"
    "convert_grpo_to_sft.py" "mix_replay_adaptive.py"
    "run_selfplay.sh"
    "reward_functions/reviewer_reward.py"
    "reward_functions/reward_logger.py"
    "reward_functions/llm_judge.py"
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/${f}" ]; then
        fail "缺少: ${f}"
        MISSING=$((MISSING+1))
    fi
done
if [ ${MISSING} -eq 0 ]; then
    ok "集成版文件完整 (${#REQUIRED_FILES[@]} 个关键文件)"
else
    ERRORS=$((ERRORS+MISSING))
fi

echo ""
if [ ${ERRORS} -gt 0 ]; then
    fail "环境检查发现 ${ERRORS} 个问题，请先修复后再运行"
    exit 1
else
    ok "环境检查全部通过！"
fi

if [ "${CHECK_ONLY}" = "1" ]; then
    info "CHECK_ONLY=1，仅检查环境，不启动训练"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────────
# 3. 设置 NPU 环境变量
# ─────────────────────────────────────────────────────────────────────────────────
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
export TASK_QUEUE_ENABLE=1

# ─────────────────────────────────────────────────────────────────────────────────
# 4. 启动训练
# ─────────────────────────────────────────────────────────────────────────────────
echo ""
info "═══════ 启动集成版对抗博弈自对弈训练 ═══════"
echo ""

export BASE_DIR
export MODEL_SIZE
export N_GPUS
export PYTHON_EXEC
export SELFPLAY_ROUNDS
export MAX_STEPS
export SAMPLES_PER_CAT
export VERIFIER_BACKEND
export VERIFIER_API_KEY
export VERIFIER_API_MODEL
export VERIFIER_MODEL

# 调用 run_selfplay.sh (集成版核心训练入口)
exec bash "${SCRIPT_DIR}/run_selfplay.sh"
