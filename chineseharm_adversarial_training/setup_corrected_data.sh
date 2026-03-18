#!/bin/bash
# =============================================================================
# 修正数据一键部署脚本
# =============================================================================
# 功能:
#   1. 将 corrected_data/ 中的 SFT 数据部署到脚本期望的路径
#   2. 生成 split_data/ 下的 parquet 文件（RL数据准备需要）
#   3. 重新生成 prepared_data/rl/ 下的所有 parquet（修正旧标签）
#   4. 生成 self-play 所需的 train_seed.parquet
#   5. 部署评估数据到正确位置
#
# 用法:
#   cd /home/ma-user/work/test/chineseharm_adversarial_training
#   bash setup_corrected_data.sh
#
# 前置: Python 环境中已安装 pandas, pyarrow
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CORRECTED_DIR="$SCRIPT_DIR/corrected_data"
PREPARED_DIR="$SCRIPT_DIR/prepared_data"
PREPARED_V2_DIR="$SCRIPT_DIR/prepared_data_v2"
SPLIT_DIR="$SCRIPT_DIR/split_data"

echo "============================================================"
echo "  修正数据一键部署"
echo "============================================================"
echo "  修正数据目录: $CORRECTED_DIR"
echo "  目标目录:     $PREPARED_DIR, $SPLIT_DIR"
echo ""

# 检查修正数据是否存在
for f in train.json test.json val.json \
         challenger_sft_train.jsonl challenger_sft_val.jsonl \
         sft_train.jsonl sft_val.jsonl multi_label_map.json; do
    if [ ! -f "$CORRECTED_DIR/$f" ]; then
        echo "❌ 缺少文件: $CORRECTED_DIR/$f"
        exit 1
    fi
done
echo "✓ 修正数据文件完整性检查通过"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 1: 部署 SFT 数据
# ─────────────────────────────────────────────────────────────────
echo "[1/5] 部署 SFT 数据到 prepared_data/"

mkdir -p "$PREPARED_DIR/challenger_sft"
mkdir -p "$PREPARED_DIR/reviewer_sft"

cp "$CORRECTED_DIR/challenger_sft_train.jsonl" "$PREPARED_DIR/challenger_sft/train.jsonl"
cp "$CORRECTED_DIR/challenger_sft_val.jsonl"   "$PREPARED_DIR/challenger_sft/val.jsonl"
cp "$CORRECTED_DIR/sft_train.jsonl"            "$PREPARED_DIR/reviewer_sft/train.jsonl"
cp "$CORRECTED_DIR/sft_val.jsonl"              "$PREPARED_DIR/reviewer_sft/val.jsonl"

echo "  challenger_sft/train.jsonl  <- challenger_sft_train.jsonl"
echo "  challenger_sft/val.jsonl    <- challenger_sft_val.jsonl"
echo "  reviewer_sft/train.jsonl    <- sft_train.jsonl"
echo "  reviewer_sft/val.jsonl      <- sft_val.jsonl"
echo "✓ SFT 数据部署完成"
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 2: 部署 split_data 并生成 parquet
# ─────────────────────────────────────────────────────────────────
echo "[2/5] 部署 split_data/ 并生成 parquet"

mkdir -p "$SPLIT_DIR"

cp "$CORRECTED_DIR/train.json" "$SPLIT_DIR/train.json"
cp "$CORRECTED_DIR/val.json"   "$SPLIT_DIR/val.json"
cp "$CORRECTED_DIR/test.json"  "$SPLIT_DIR/test.json"

python3 - <<'PYEOF'
import json
import pandas as pd
import sys
import os

split_dir = os.environ.get("SPLIT_DIR", "split_data")
corrected_dir = os.environ.get("CORRECTED_DIR", "corrected_data")

# 类别 → 默认 toxic_type / expression 映射 (与 constants.py CAT_DEFAULTS 一致)
CAT_DEFAULTS = {
    "性别歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "种族歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "地域偏见": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "LGBTQ歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "其他仇恨":  {"toxic_type": "一般攻击性", "expression": "非仇恨"},
    "无毒":      {"toxic_type": "无毒", "expression": "非仇恨"},
}

for split in ["train", "val", "test"]:
    json_path = os.path.join(split_dir, f"{split}.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 补充 toxic_type_label 和 expression_label (RL 数据准备需要)
    df["toxic_type_label"] = df["标签"].map(lambda c: CAT_DEFAULTS.get(c, {}).get("toxic_type", ""))
    df["expression_label"] = df["标签"].map(lambda c: CAT_DEFAULTS.get(c, {}).get("expression", ""))

    parquet_path = os.path.join(split_dir, f"{split}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"  {split}.parquet: {len(df)} 条 | 列: {list(df.columns)}")

print("✓ split_data parquet 生成完成")
PYEOF

echo ""

# ─────────────────────────────────────────────────────────────────
# Step 3: 重新生成 RL 数据 (parquet + JSON)
# ─────────────────────────────────────────────────────────────────
echo "[3/5] 重新生成 prepared_data/rl/ (修正标签版)"

mkdir -p "$PREPARED_DIR/rl"

SPLIT_DIR="$SPLIT_DIR" CORRECTED_DIR="$CORRECTED_DIR" python3 - <<'PYEOF'
import json
import os
import pandas as pd

split_dir = os.environ.get("SPLIT_DIR", "split_data")
corrected_dir = os.environ.get("CORRECTED_DIR", "corrected_data")
rl_dir = os.path.join(os.path.dirname(corrected_dir), "prepared_data", "rl")

# ── 加载修正数据 (已带 toxic_type_label / expression_label) ──
train_df = pd.read_parquet(os.path.join(split_dir, "train.parquet"))
val_df   = pd.read_parquet(os.path.join(split_dir, "val.parquet"))
test_df  = pd.read_parquet(os.path.join(split_dir, "test.parquet"))

# ── 生成 train_seed / val_eval / test_eval ──
# 格式: 简单表格 [{文本, 标签, all_labels, toxic_type_label, expression_label}, ...]
# all_labels: 多标签列表，评估时模型预测命中其中任一即算正确
# generate_dynamic_data.py 的 build_sampling_tasks() 直接读这些列
for split, df, name in [
    ("train", train_df, "train_seed"),
    ("val",   val_df,   "val_eval"),
    ("test",  test_df,  "test_eval"),
]:
    keep_cols = ["文本", "标签", "toxic_type_label", "expression_label"]
    if "all_labels" in df.columns:
        keep_cols.insert(2, "all_labels")
    out_df = df[keep_cols].copy()

    # JSON (用于评估脚本 batch_eval_npu_vllm.py 等)
    json_path = os.path.join(rl_dir, f"{name}.json")
    out_df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    # Parquet (用于 self-play seed_data)
    parquet_path = os.path.join(rl_dir, f"{name}.parquet")
    out_df.to_parquet(parquet_path, index=False)

    print(f"  {name}: {len(out_df)} 条 → .json + .parquet  列={list(out_df.columns)}")

print("✓ RL 数据重新生成完成 (使用修正标签)")
PYEOF

echo ""

# ─────────────────────────────────────────────────────────────────
# Step 4: 同步更新 prepared_data_v2/rl/ (如果存在)
# ─────────────────────────────────────────────────────────────────
echo "[4/5] 同步 prepared_data_v2/rl/"

if [ -d "$PREPARED_V2_DIR/rl" ]; then
    cp "$PREPARED_DIR/rl/train_seed.json"    "$PREPARED_V2_DIR/rl/train_seed.json"
    cp "$PREPARED_DIR/rl/train_seed.parquet"  "$PREPARED_V2_DIR/rl/train_seed.parquet"
    cp "$PREPARED_DIR/rl/val_eval.json"       "$PREPARED_V2_DIR/rl/val_eval.json"
    cp "$PREPARED_DIR/rl/val_eval.parquet"    "$PREPARED_V2_DIR/rl/val_eval.parquet"
    cp "$PREPARED_DIR/rl/test_eval.json"      "$PREPARED_V2_DIR/rl/test_eval.json"
    cp "$PREPARED_DIR/rl/test_eval.parquet"   "$PREPARED_V2_DIR/rl/test_eval.parquet"
    echo "  ✓ prepared_data_v2/rl/ 已同步"
else
    echo "  ⏭  prepared_data_v2/rl/ 不存在，跳过"
fi
echo ""

# ─────────────────────────────────────────────────────────────────
# Step 5: 验证
# ─────────────────────────────────────────────────────────────────
echo "[5/5] 验证数据完整性"
echo ""

ERRORS=0

check_file() {
    local path="$1"
    local desc="$2"
    if [ -f "$path" ]; then
        local size=$(du -h "$path" | cut -f1)
        echo "  ✓ $desc ($size)"
    else
        echo "  ❌ $desc 缺失: $path"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "SFT 数据:"
check_file "$PREPARED_DIR/challenger_sft/train.jsonl" "Challenger SFT train"
check_file "$PREPARED_DIR/challenger_sft/val.jsonl"   "Challenger SFT val"
check_file "$PREPARED_DIR/reviewer_sft/train.jsonl"   "Reviewer SFT train"
check_file "$PREPARED_DIR/reviewer_sft/val.jsonl"     "Reviewer SFT val"
echo ""

echo "Split 数据:"
check_file "$SPLIT_DIR/train.json"    "train.json"
check_file "$SPLIT_DIR/train.parquet" "train.parquet"
check_file "$SPLIT_DIR/val.json"      "val.json (self-play 评估)"
check_file "$SPLIT_DIR/val.parquet"   "val.parquet"
check_file "$SPLIT_DIR/test.json"     "test.json"
check_file "$SPLIT_DIR/test.parquet"  "test.parquet"
echo ""

echo "RL 数据:"
check_file "$PREPARED_DIR/rl/train_seed.json"    "train_seed.json"
check_file "$PREPARED_DIR/rl/train_seed.parquet"  "train_seed.parquet (self-play 种子)"
check_file "$PREPARED_DIR/rl/val_eval.json"       "val_eval.json"
check_file "$PREPARED_DIR/rl/val_eval.parquet"    "val_eval.parquet"
check_file "$PREPARED_DIR/rl/test_eval.json"      "test_eval.json"
check_file "$PREPARED_DIR/rl/test_eval.parquet"   "test_eval.parquet"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo "❌ 发现 $ERRORS 个错误，请检查上述输出"
    exit 1
fi

echo "============================================================"
echo "✓ 全部数据部署完成！"
echo "============================================================"
echo ""
echo "后续步骤:"
echo "  1. bash scripts/run_pipeline/run_01_download.sh    # 下载模型"
echo "  2. bash scripts/run_pipeline/run_03_lora_sft.sh   # LoRA 微调"
echo "  3. bash scripts/run_pipeline/run_04_merge_lora.sh  # 合并 LoRA"
echo "  4. bash scripts/run_pipeline/run_05_evaluate.sh    # 评估"
echo "  5. bash scripts/integrated_selfplay/run_selfplay.sh  # Self-Play"
echo ""
echo "注意: Step 2 (prepare_data) 和 Step 6 (prepare_rl_data) 已由本脚本完成，无需再运行。"
