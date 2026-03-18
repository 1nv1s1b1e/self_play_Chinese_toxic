#!/bin/bash
# =============================================================================
# Step 1: 下载Qwen2.5模型
# 使用modelscope下载 0.5B, 1.5B, 3B 三个尺寸的Instruct模型
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================"
echo "Step 1: 下载 Qwen2.5 Instruct 模型"
echo "============================================================"

python ../download_models/download_qwen.py \
    --models 0.5B 1.5B 3B \
    --output_dir "$BASE_DIR/models_base" \
    --source modelscope

echo ""
echo "✓ 模型下载完成!"
echo "模型保存在: $BASE_DIR/models_base/Qwen/"
echo ""
echo "验证已下载模型:"
ls -la $BASE_DIR/models_base/Qwen/ 2>/dev/null || echo "目录不存在"
