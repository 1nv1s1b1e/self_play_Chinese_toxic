#!/bin/bash
# =============================================================================
# Step 6: 准备RL GRPO训练数据
# 将 split_data (富字段格式) 转换为 verl GRPO 所需的 parquet 格式
# 包含 Challenger + Reviewer 的 prompt messages (带system prompt)
#
# 前置: run_02_prepare_data.sh 已生成 split_data/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="/home/ma-user/work/test"
SPLIT_DIR="$BASE_DIR/split_data"
RL_DATA_DIR="$BASE_DIR/prepared_data/rl"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --split_dir)   SPLIT_DIR="$2";   shift 2 ;;
        --output_dir)  RL_DATA_DIR="$2"; shift 2 ;;
        *)             echo "未知参数: $1"; exit 1 ;;
    esac
done

# 检查输入
if [ ! -f "$SPLIT_DIR/train.parquet" ]; then
    echo "❌ 数据不存在: $SPLIT_DIR/train.parquet"
    echo "   请先运行 run_02_prepare_data.sh"
    exit 1
fi

echo "============================================================"
echo "Step 6: 准备RL GRPO训练数据"
echo "============================================================"
echo "输入目录: $SPLIT_DIR"
echo "输出目录: $RL_DATA_DIR"
echo ""

python ../create_data/prepare_rl_grpo_data.py \
    --split_dir "$SPLIT_DIR" \
    --output_dir "$RL_DATA_DIR"

echo ""
echo "✓ Step 6 完成: RL GRPO数据已准备"
echo "  $RL_DATA_DIR/"
echo ""
echo "文件列表:"
ls -lh "$RL_DATA_DIR"/*.parquet "$RL_DATA_DIR"/*.json 2>/dev/null || echo "(文件列表不可用)"
