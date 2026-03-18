#!/bin/bash
# =============================================================================
# Step 2: 数据处理 - 仓库数据转换 + 准备SFT/RL数据
#
# 数据流:
#   train.json + test.json → convert_repo_data.py → split_data/ (富字段)
#   split_data/ → prepare_all_data.py → prepared_data/{challenger_sft, reviewer_sft, rl}
#
# 前置: 将仓库的 train.json / test.json 放入 ORIGIN_DATA 目录
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="/home/ma-user/work/test"
ORIGIN_DATA="$BASE_DIR/origin_data"
SPLIT_DATA="$BASE_DIR/split_data"
PREPARED_DATA="$BASE_DIR/prepared_data"

# 仓库原始数据 (ChineseHarm repo 的 train.json / test.json)
REPO_TRAIN_JSON="${REPO_TRAIN_JSON:-$ORIGIN_DATA/train.json}"
REPO_TEST_JSON="${REPO_TEST_JSON:-$ORIGIN_DATA/test.json}"

echo "============================================================"
echo "Step 2: 数据处理"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────
# 2.1 转换仓库数据 → 富字段 split_data/
# ─────────────────────────────────────────────────────────────────
echo ""
echo "[2.1] 转换仓库原始数据..."
echo "  输入: $REPO_TRAIN_JSON (train) + $REPO_TEST_JSON (test)"
echo "  输出: $SPLIT_DATA/{train,val,test}.{parquet,json}"
echo ""

# 检查输入文件
for f in "$REPO_TRAIN_JSON" "$REPO_TEST_JSON"; do
    if [ ! -f "$f" ]; then
        echo "❌ 文件不存在: $f"
        echo "   请将仓库的 train.json 和 test.json 放到 $ORIGIN_DATA/ 下"
        echo "   或设置环境变量: REPO_TRAIN_JSON=/path/to/train.json REPO_TEST_JSON=/path/to/test.json"
        exit 1
    fi
done

python ../create_data/convert_repo_data.py \
    --train_json "$REPO_TRAIN_JSON" \
    --test_json "$REPO_TEST_JSON" \
    --output_dir "$SPLIT_DATA" \
    --val_ratio 0.15 \
    --seed 42

echo ""
echo "✓ 数据转换完成!"
echo ""

# ─────────────────────────────────────────────────────────────────
# 2.2 准备SFT和RL数据
# ─────────────────────────────────────────────────────────────────
echo "[2.2] 准备SFT和RL数据..."
echo "  输入: $SPLIT_DATA/"
echo "  输出: $PREPARED_DATA/{challenger_sft,reviewer_sft,rl}/"
echo ""

python ../create_data/prepare_all_data.py \
    --split_dir "$SPLIT_DATA" \
    --output_dir "$PREPARED_DATA"

echo ""
echo "============================================================"
echo "✓ 数据处理全部完成!"
echo "============================================================"
echo ""
echo "数据目录结构:"
echo "  $SPLIT_DATA/"
find "$SPLIT_DATA" -name "*.parquet" -o -name "*.json" 2>/dev/null | sort | while read f; do echo "    $(basename $f)"; done
echo ""
echo "  $PREPARED_DATA/"
find "$PREPARED_DATA" -type f 2>/dev/null | sort | while read f; do
    echo "    ${f#$PREPARED_DATA/}"
done
