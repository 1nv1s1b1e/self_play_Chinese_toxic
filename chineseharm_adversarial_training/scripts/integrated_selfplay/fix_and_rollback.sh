#!/bin/bash
# =============================================================================
# 回滚到 step 22，从 best_saved/ 重建 step_22 目录
# 用法:
#   cd /home/ma-user/work/test/chineseharm_adversarial_training
#   bash scripts/integrated_selfplay/fix_and_rollback.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SP="${BASE_DIR}/selfplay_integrated/3B_4npu"
DATA="${BASE_DIR}/selfplay_integrated_data/3B"

ROLLBACK_TO=22

echo "============================================================"
echo "  回滚到 Step ${ROLLBACK_TO} (从 best_saved/ 重建)"
echo "============================================================"
echo "  selfplay_dir: ${SP}"
echo "  data_dir:     ${DATA}"
echo ""

# 检查目录存在
if [ ! -d "${SP}" ]; then
    echo "❌ selfplay 目录不存在: ${SP}"
    exit 1
fi

# 检查 best_saved 存在
BEST_SAVED="${SP}/best_saved"
if [ ! -d "${BEST_SAVED}/challenger" ] || [ ! -d "${BEST_SAVED}/reviewer" ]; then
    echo "❌ best_saved/ 中缺少 challenger 或 reviewer 模型"
    echo "   ${BEST_SAVED}/challenger: $([ -d ${BEST_SAVED}/challenger ] && echo '存在' || echo '不存在')"
    echo "   ${BEST_SAVED}/reviewer:   $([ -d ${BEST_SAVED}/reviewer ] && echo '存在' || echo '不存在')"
    exit 1
fi
echo "✓ best_saved/ 模型确认存在"

# ── 1. 删除 step (ROLLBACK_TO+1)+ 的模型 ──
echo ""
echo "[1/7] 删除 step $((ROLLBACK_TO+1))+ 的模型..."
deleted=0
for i in $(seq $((ROLLBACK_TO+1)) 100); do
    if [ -d "${SP}/step_${i}" ]; then
        rm -rf "${SP}/step_${i}"
        deleted=$((deleted+1))
    fi
done
echo "  已删除 ${deleted} 个 step 目录"

# ── 2. 删除 step (ROLLBACK_TO+1)+ 的 datagen 数据 ──
echo "[2/7] 删除 step $((ROLLBACK_TO+1))+ 的 datagen 数据..."
deleted=0
for i in $(seq $((ROLLBACK_TO+1)) 100); do
    if [ -d "${DATA}/step_${i}" ]; then
        rm -rf "${DATA}/step_${i}"
        deleted=$((deleted+1))
    fi
done
echo "  已删除 ${deleted} 个 data 目录"

# ── 3. 清理 eval_history 和 datagen_history ──
echo "[3/7] 清理 history 中 step $((ROLLBACK_TO+1))+ 的文件..."
deleted=0
for i in $(seq $((ROLLBACK_TO+1)) 100); do
    for f in \
        "${SP}/eval_history/eval_step${i}.json" \
        "${SP}/datagen_history/datagen_stats_step${i}.json" \
        "${SP}/datagen_history/sample_rewards_step${i}.parquet" \
        "${SP}/datagen_history/challenger_trace_step${i}.parquet"; do
        if [ -f "$f" ]; then
            rm -f "$f"
            deleted=$((deleted+1))
        fi
    done
done
echo "  已删除 ${deleted} 个 history 文件"

# ── 4. 清理 metrics.jsonl ──
echo "[4/7] 清理 metrics.jsonl (保留 step 1-${ROLLBACK_TO})..."
METRICS="${SP}/metrics.jsonl"
if [ -f "${METRICS}" ]; then
    ROLLBACK_TO=${ROLLBACK_TO} METRICS_PATH="${METRICS}" python3 << 'PYEOF'
import json, os
rollback = int(os.environ["ROLLBACK_TO"])
mpath = os.environ["METRICS_PATH"]
with open(mpath) as f:
    lines = f.readlines()
kept = []
for l in lines:
    l = l.strip()
    if not l:
        continue
    try:
        if json.loads(l).get("step", 999) <= rollback:
            kept.append(l + "\n")
    except:
        pass
with open(mpath, "w") as f:
    f.writelines(kept)
print(f"  保留 {len(kept)} 条 (step 1-{rollback})")
PYEOF
else
    echo "  metrics.jsonl 不存在，跳过"
fi

# ── 5. 从 best_saved/ 重建 step_22 目录 ──
echo "[5/7] 从 best_saved/ 重建 step_${ROLLBACK_TO}/ ..."
STEP_DIR="${SP}/step_${ROLLBACK_TO}"
rm -rf "${STEP_DIR}" 2>/dev/null || true
mkdir -p "${STEP_DIR}"

cp -r "${BEST_SAVED}/challenger" "${STEP_DIR}/challenger"
cp -r "${BEST_SAVED}/reviewer"   "${STEP_DIR}/reviewer"
echo "  ✓ ${STEP_DIR}/challenger"
echo "  ✓ ${STEP_DIR}/reviewer"

# ── 5b. 补全 config.json（DeepSpeed 保存时可能缺失，vLLM 必需）──
CHALLENGER_INIT="${BASE_DIR}/merged_models_toxicn/challenger_3B"
REVIEWER_INIT="${BASE_DIR}/merged_models_toxicn/reviewer_3B"

for role_dir in "${STEP_DIR}/challenger" "${STEP_DIR}/reviewer"; do
    if [ "${role_dir}" = "${STEP_DIR}/challenger" ]; then
        INIT_DIR="${CHALLENGER_INIT}"
    else
        INIT_DIR="${REVIEWER_INIT}"
    fi
    for fname in config.json generation_config.json tokenizer_config.json tokenizer.json special_tokens_map.json; do
        if [ ! -f "${role_dir}/${fname}" ] && [ -f "${INIT_DIR}/${fname}" ]; then
            cp "${INIT_DIR}/${fname}" "${role_dir}/${fname}"
            echo "  ✓ 补全 ${role_dir##*/}/${fname}"
        fi
    done
done

# ── 6. 复制 best_saved/ → best/ ──
echo "[6/7] 复制 best_saved/ → best/ ..."
BEST="${SP}/best"
rm -rf "${BEST}/challenger" "${BEST}/reviewer" 2>/dev/null || true
mkdir -p "${BEST}"

cp -r "${STEP_DIR}/challenger" "${BEST}/challenger"
cp -r "${STEP_DIR}/reviewer"   "${BEST}/reviewer"
echo "${ROLLBACK_TO}" > "${BEST}/best_step.txt"

if [ -f "${BEST_SAVED}/best_info.json" ]; then
    cp "${BEST_SAVED}/best_info.json" "${BEST}/best_info.json"
fi
echo "  ✓ best/ 已同步"

# ── 7. 回退 progress.json + latest/ ──
echo "[7/7] 更新 progress.json 和 latest/ ..."
PROGRESS="${SP}/progress.json"
STEP_C="${STEP_DIR}/challenger"
STEP_R="${STEP_DIR}/reviewer"

cat > "${PROGRESS}" << PEOF
{
  "last_completed_step": ${ROLLBACK_TO},
  "last_completed_phase": "done",
  "total_steps": 50,
  "current_challenger": "${STEP_C}",
  "current_reviewer": "${STEP_R}",
  "timestamp": "$(date -Iseconds)"
}
PEOF
echo "  ✓ progress.json → step ${ROLLBACK_TO}"

LATEST="${SP}/latest"
mkdir -p "${LATEST}"
cat > "${LATEST}/latest_paths.json" << LEOF
{
  "step": ${ROLLBACK_TO},
  "timestamp": "$(date -Iseconds)",
  "challenger": "${STEP_C}",
  "reviewer": "${STEP_R}"
}
LEOF
echo "${ROLLBACK_TO}" > "${LATEST}/latest_step.txt"
echo "${STEP_C}" > "${LATEST}/challenger_latest.txt"
echo "${STEP_R}" > "${LATEST}/reviewer_latest.txt"
echo "  ✓ latest/ 已更新"

# ── 确认 ──
echo ""
echo "============================================================"
echo "  回滚完成 → Step ${ROLLBACK_TO}"
echo "============================================================"
echo "  step_${ROLLBACK_TO}/challenger: $(ls "${STEP_DIR}/challenger/config.json" 2>/dev/null && echo '✓' || echo '⚠️ 无 config.json')"
echo "  step_${ROLLBACK_TO}/reviewer:   $(ls "${STEP_DIR}/reviewer/config.json" 2>/dev/null && echo '✓' || echo '⚠️ 无 config.json')"
echo "  best/:          $(ls "${BEST}/" 2>/dev/null | tr '\n' ' ')"
echo "  progress:       step $(python3 -c "import json; print(json.load(open('${PROGRESS}'))['last_completed_step'])")"
if [ -f "${METRICS}" ]; then
    echo "  metrics.jsonl:  $(wc -l < "${METRICS}") 条"
fi
echo "  eval_history:   $(ls "${SP}/eval_history/" 2>/dev/null | wc -l) 个文件"
echo ""
echo "下一步: RESUME=1 bash scripts/integrated_selfplay/run_selfplay.sh"
