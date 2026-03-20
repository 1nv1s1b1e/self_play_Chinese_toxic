#!/bin/bash
# =============================================================================
# 回滚到 step 23，清理 step 24+ 数据，复制 best_saved → best
# 用法:
#   cd /home/ma-user/work/test/chineseharm_adversarial_training
#   bash scripts/integrated_selfplay/fix_and_rollback.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SP="${BASE_DIR}/selfplay_integrated/3B_4npu"
DATA="${BASE_DIR}/selfplay_integrated_data/3B"

ROLLBACK_TO=23

echo "============================================================"
echo "  回滚到 Step ${ROLLBACK_TO}"
echo "============================================================"
echo "  selfplay_dir: ${SP}"
echo "  data_dir:     ${DATA}"
echo ""

# 检查目录存在
if [ ! -d "${SP}" ]; then
    echo "❌ selfplay 目录不存在: ${SP}"
    exit 1
fi

# ── 1. 删除 step 24+ 的模型 ──
echo "[1/6] 删除 step $((ROLLBACK_TO+1))+ 的模型..."
deleted=0
for i in $(seq $((ROLLBACK_TO+1)) 100); do
    if [ -d "${SP}/step_${i}" ]; then
        rm -rf "${SP}/step_${i}"
        deleted=$((deleted+1))
    fi
done
echo "  已删除 ${deleted} 个 step 目录"

# ── 2. 删除 step 24+ 的 datagen 数据 ──
echo "[2/6] 删除 step $((ROLLBACK_TO+1))+ 的 datagen 数据..."
deleted=0
for i in $(seq $((ROLLBACK_TO+1)) 100); do
    if [ -d "${DATA}/step_${i}" ]; then
        rm -rf "${DATA}/step_${i}"
        deleted=$((deleted+1))
    fi
done
echo "  已删除 ${deleted} 个 data 目录"

# ── 3. 清理 eval_history 和 datagen_history ──
echo "[3/6] 清理 history 中 step $((ROLLBACK_TO+1))+ 的文件..."
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
echo "[4/6] 清理 metrics.jsonl (保留 step 1-${ROLLBACK_TO})..."
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

# ── 5. 回退 progress.json ──
echo "[5/6] 回退 progress.json 到 step ${ROLLBACK_TO}..."
PROGRESS="${SP}/progress.json"
if [ -f "${PROGRESS}" ]; then
    ROLLBACK_TO=${ROLLBACK_TO} PROGRESS_PATH="${PROGRESS}" python3 << 'PYEOF'
import json, os
rollback = int(os.environ["ROLLBACK_TO"])
ppath = os.environ["PROGRESS_PATH"]
with open(ppath) as f:
    p = json.load(f)
p["last_completed_step"] = rollback
p["last_completed_phase"] = "done"
with open(ppath, "w") as f:
    json.dump(p, f, indent=2)
print(f"  已回退: last_completed_step={rollback}")
PYEOF
else
    echo "  progress.json 不存在，跳过"
fi

# ── 6. 复制 best_saved → best ──
echo "[6/6] 复制 best_saved/ → best/..."
BEST_SAVED="${SP}/best_saved"
BEST="${SP}/best"

if [ -d "${BEST_SAVED}/challenger" ] && [ -d "${BEST_SAVED}/reviewer" ]; then
    rm -rf "${BEST}/challenger" "${BEST}/reviewer"
    cp -r "${BEST_SAVED}/challenger" "${BEST}/challenger"
    cp -r "${BEST_SAVED}/reviewer" "${BEST}/reviewer"
    # 复制 best_info.json 中的 step 到 best_step.txt
    if [ -f "${BEST_SAVED}/best_info.json" ]; then
        cp "${BEST_SAVED}/best_info.json" "${BEST}/best_info.json"
        python3 -c "
import json
with open('${BEST_SAVED}/best_info.json') as f:
    info = json.load(f)
step = info.get('step', '?')
acc = info.get('acc', '?')
with open('${BEST}/best_step.txt', 'w') as f:
    f.write(str(step))
print(f'  best 模型来自 step {step} (acc={acc}%)')
"
    fi
    echo "  ✓ best_saved/ 已复制到 best/"
else
    echo "  ⚠️ best_saved/ 中没有模型，尝试保留现有 best/"
    if [ -d "${BEST}/challenger" ]; then
        echo "  现有 best/ 保留不动"
    else
        echo "  ❌ 没有可用的 best 模型"
    fi
fi

# ── 确认 ──
echo ""
echo "============================================================"
echo "  回滚完成"
echo "============================================================"
if [ -f "${PROGRESS}" ]; then
    echo "  progress: step $(python3 -c "import json; print(json.load(open('${PROGRESS}'))['last_completed_step'])")"
fi
if [ -f "${METRICS}" ]; then
    echo "  metrics:  $(wc -l < "${METRICS}") 条"
fi
echo "  best/:    $(ls "${BEST}/" 2>/dev/null | tr '\n' ' ')"
echo "  eval_history: $(ls "${SP}/eval_history/" 2>/dev/null | wc -l) 个文件"
echo ""
echo "下一步: RESUME=1 MODEL_SIZE=3B N_GPUS=4 bash scripts/integrated_selfplay/run_selfplay.sh"
