#!/bin/bash
set -e

# 兼容入口：该脚本仅作为 TRL 自对弈启动别名。
# 按项目规范，已禁用旧版 verl GRPO 路径。

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_selfplay_trl_npu.sh"

if [ "${SELFPLAY_ENGINE:-trl}" != "trl" ]; then
    echo "❌ 已禁用非 TRL 引擎。请使用 SELFPLAY_ENGINE=trl（或不设置该变量）。"
    exit 1
fi

if [ ! -f "${TARGET_SCRIPT}" ]; then
    echo "❌ 找不到 TRL 自对弈脚本: ${TARGET_SCRIPT}"
    exit 1
fi

echo "→ 使用 TRL + DeepSpeed 自对弈链路: ${TARGET_SCRIPT}"
exec bash "${TARGET_SCRIPT}"
