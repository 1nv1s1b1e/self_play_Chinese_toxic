#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 昇腾 910B NPU 环境检查脚本 (集成版)
# ═══════════════════════════════════════════════════════════════════════════
# 在启动训练之前运行，检查：
#   1. CANN Toolkit 是否安装
#   2. Python 解释器和 torch_npu 是否可用
#   3. NPU 设备数量
#   4. verl / vllm / trl 等依赖
#   5. 分布式通信
#   6. RL 训练所需数据文件
#
# 用法:
#   bash check_env_npu.sh
#   PYTHON_EXEC=/path/to/python bash check_env_npu.sh

set -u

PYTHON_EXEC="${PYTHON_EXEC:-/home/ma-user/.conda/envs/ssp_train/bin/python}"

PASS=0
FAIL=0
WARN=0

check() {
    local name="$1" condition="$2"
    if [ "$condition" = "0" ]; then
        echo "  ✅ $name"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $name"
        FAIL=$((FAIL + 1))
    fi
}

warn() {
    local name="$1"
    echo "  ⚠️  $name"
    WARN=$((WARN + 1))
}

echo "═══════════════════════════════════════════════════════════"
echo "  昇腾 910B NPU 环境检查 (集成版)"
echo "═══════════════════════════════════════════════════════════"

# ── 1. CANN Toolkit ──────────────────────────────────────────
echo ""
echo "【1】CANN Toolkit"
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
    check "ascend-toolkit 环境脚本存在" 0
else
    check "ascend-toolkit 环境脚本存在" 1
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null
    check "nnal/atb 环境脚本存在" 0
else
    warn "nnal/atb 环境脚本不存在 (非必须)"
fi

# ── 2. Python + torch_npu ─────────────────────────────────────
echo ""
echo "【2】Python 解释器 & torch_npu"
if [ -x "${PYTHON_EXEC}" ]; then
    PY_VER=$("${PYTHON_EXEC}" --version 2>&1)
    check "Python 可执行: ${PY_VER}" 0
else
    check "Python 可执行: ${PYTHON_EXEC}" 1
fi

"${PYTHON_EXEC}" -c "import torch; print(f'  torch {torch.__version__}')" 2>/dev/null
check "torch 可导入" $?

"${PYTHON_EXEC}" -c "import torch_npu; print(f'  torch_npu {torch_npu.__version__}')" 2>/dev/null
check "torch_npu 可导入" $?

# ── 3. NPU 设备 ──────────────────────────────────────────────
echo ""
echo "【3】NPU 设备"
NPU_COUNT=$("${PYTHON_EXEC}" -c "
import torch, torch_npu
print(torch.npu.device_count())
" 2>/dev/null || echo "0")
check "NPU 设备数量: ${NPU_COUNT}" $([ "${NPU_COUNT}" -ge 1 ] && echo 0 || echo 1)

# ── 4. 训练依赖 ──────────────────────────────────────────────
echo ""
echo "【4】训练依赖包"
for pkg in "trl" "transformers" "datasets" "peft" "deepspeed" "pandas" "tqdm"; do
    "${PYTHON_EXEC}" -c "import ${pkg}; print(f'  ${pkg} {${pkg}.__version__}')" 2>/dev/null
    check "${pkg}" $?
done

# openai (可选，API Verifier 需要)
"${PYTHON_EXEC}" -c "import openai; print(f'  openai {openai.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    warn "openai 未安装 (API Verifier 需要: pip install openai)"
else
    check "openai (API Verifier)" 0
fi

# ── 5. 分布式通信 ─────────────────────────────────────────────
echo ""
echo "【5】分布式通信"
"${PYTHON_EXEC}" -c "
import torch
import torch_npu
print(f'  HCCL 可用: {torch.distributed.is_hccl_available() if hasattr(torch.distributed, \"is_hccl_available\") else \"N/A\"}')
print(f'  NCCL 可用: {torch.distributed.is_nccl_available()}')
" 2>/dev/null || warn "分布式通信检测失败"

# ── 6. 环境变量 ──────────────────────────────────────────────
echo ""
echo "【6】关键环境变量"
for var in HCCL_WHITELIST_DISABLE HCCL_CONNECT_TIMEOUT TORCH_DEVICE_BACKEND_AUTOLOAD; do
    val=$(eval echo \${"$var":-""} 2>/dev/null || echo "")
    if [ -n "$val" ]; then
        echo "  ✅ ${var}=${val}"
    else
        echo "  ⚠️  ${var} 未设置 (训练脚本会自动设置)"
    fi
done

# ── 7. 集成版数据文件检查 ──────────────────────────────────────
echo ""
echo "【7】集成版文件完整性"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQUIRED_FILES=(
    "constants.py"
    "quality_gate.py"
    "rejection_sampler.py"
    "build_parquet.py"
    "verifier.py"
    "generate_dynamic_data.py"
    "adversarial_trl_grpo.py"
    "convert_grpo_to_sft.py"
    "mix_replay_adaptive.py"
    "challenger_reward.py"
    "eval_selfplay_results.py"
    "run_selfplay.sh"
    "ds_zero2.json"
    "reward_functions/reviewer_reward.py"
    "reward_functions/reward_logger.py"
    "reward_functions/llm_judge.py"
    "reward_functions/__init__.py"
)
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "${SCRIPT_DIR}/${f}" ]; then
        check "${f}" 0
    else
        check "${f}" 1
    fi
done

# ── 汇总 ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  结果: ✅ ${PASS} 通过 / ❌ ${FAIL} 失败 / ⚠️  ${WARN} 警告"
echo "═══════════════════════════════════════════════════════════"

if [ "${FAIL}" -gt 0 ]; then
    echo ""
    echo "  ❌ 存在必要项检查失败，请修复后再启动训练"
    exit 1
else
    echo ""
    echo "  🎉 环境检查通过！可以启动训练"
    echo "  建议下一步:"
    echo "    bash ${SCRIPT_DIR}/run_selfplay.sh"
fi
