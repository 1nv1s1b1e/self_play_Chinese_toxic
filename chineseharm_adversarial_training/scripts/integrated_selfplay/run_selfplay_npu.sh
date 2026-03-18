#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# 对抗博弈 RL 训练入口 (别名脚本，集成版)
# ═══════════════════════════════════════════════════════════════════════════
# 直接转发给 run_selfplay.sh (TRL + DeepSpeed 方案)
# 用法: bash run_selfplay_npu.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/run_selfplay.sh" "$@"
