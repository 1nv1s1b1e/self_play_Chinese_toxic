#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# 运行所有验证测试
# ═══════════════════════════════════════════════════════════════════════
#
# 离线测试 (本地即可运行, 不需要 GPU):
#   bash tests/run_all_tests.sh
#
# 真实模型测试 (需要在 NPU 服务器上):
#   bash tests/run_all_tests.sh --model_path /path/to/reviewer_3B
#
# ═══════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"

MODEL_PATH="${1:-}"
if [ "$1" = "--model_path" ]; then
    MODEL_PATH="$2"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  多级奖励改进验证测试套件                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 测试 1: 奖励方差对比 (离线) ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "运行测试 1: 奖励方差对比 (旧版 binary vs 新版多级)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 tests/test_reward_variance.py
echo ""

# ── 测试 2: Temperature 多样性 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "运行测试 2: Temperature 多样性验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "$MODEL_PATH" ]; then
    echo "   使用真实模型: $MODEL_PATH"
    python3 tests/test_temperature_diversity.py --model_path "$MODEL_PATH" --num_samples 5 --num_gen 8
else
    echo "   使用模拟模式 (无 GPU)"
    python3 tests/test_temperature_diversity.py --mock
fi
echo ""

# ── 测试 3: Prompt 长度对比 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "运行测试 3: Prompt 长度对比 (精简版 vs 完整版)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "$MODEL_PATH" ]; then
    python3 tests/test_prompt_length.py --model_path "$MODEL_PATH"
else
    python3 tests/test_prompt_length.py
fi
echo ""

# ── 测试 4: Mini GRPO 模拟 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "运行测试 4: Mini GRPO 模拟 (advantage-based 选择)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 tests/test_mini_grpo_simulation.py
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  所有测试完成!                                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "下一步:"
echo "  1. 如果离线测试通过, 在 NPU 服务器上运行真实模型测试:"
echo "     bash tests/run_all_tests.sh --model_path /path/to/reviewer_3B"
echo ""
echo "  2. 真实模型测试通过后, 运行一轮小规模 self-play 对比:"
echo "     TOTAL_STEPS=3 SAMPLES_PER_CAT=10 bash run_selfplay.sh"
echo ""
