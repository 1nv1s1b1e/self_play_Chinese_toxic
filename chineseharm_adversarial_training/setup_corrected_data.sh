#!/bin/bash
# =============================================================================
# 修正数据一键部署脚本
# =============================================================================
# 将 corrected_data/ 中的数据部署到 pipeline 脚本期望的路径:
#   $BASE_DIR/prepared_data/   (SFT + RL 数据)
#   $BASE_DIR/split_data/      (评估 + RL 数据)
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

# BASE_DIR = 仓库根目录 (chineseharm_adversarial_training 的上一级)
# 所有 pipeline 脚本用 $BASE_DIR/prepared_data, $BASE_DIR/split_data 等
BASE_DIR="$SCRIPT_DIR"

CORRECTED_DIR="$SCRIPT_DIR/corrected_data"
PREPARED_DIR="$BASE_DIR/prepared_data"
SPLIT_DIR="$BASE_DIR/split_data"

echo "============================================================"
echo "  修正数据一键部署"
echo "============================================================"
echo "  数据源:   $CORRECTED_DIR"
echo "  BASE_DIR: $BASE_DIR"
echo "  部署到:   $PREPARED_DIR, $SPLIT_DIR"
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
echo "[1/4] 部署 SFT 数据到 $PREPARED_DIR/"

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
echo "[2/4] 部署 split_data/ 并生成 parquet"

mkdir -p "$SPLIT_DIR"

cp "$CORRECTED_DIR/train.json" "$SPLIT_DIR/train.json"
cp "$CORRECTED_DIR/val.json"   "$SPLIT_DIR/val.json"
cp "$CORRECTED_DIR/test.json"  "$SPLIT_DIR/test.json"

SPLIT_DIR="$SPLIT_DIR" python3 - <<'PYEOF'
import json, pandas as pd, os

split_dir = os.environ["SPLIT_DIR"]

CAT_DEFAULTS = {
    "性别歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "种族歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "地域偏见": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "LGBTQ歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "其他仇恨":  {"toxic_type": "一般攻击性", "expression": "非仇恨"},
    "无毒":      {"toxic_type": "无毒", "expression": "非仇恨"},
}

for split in ["train", "val", "test"]:
    with open(os.path.join(split_dir, f"{split}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["toxic_type_label"] = df["标签"].map(lambda c: CAT_DEFAULTS.get(c, {}).get("toxic_type", ""))
    df["expression_label"] = df["标签"].map(lambda c: CAT_DEFAULTS.get(c, {}).get("expression", ""))
    df.to_parquet(os.path.join(split_dir, f"{split}.parquet"), index=False)
    print(f"  {split}.parquet: {len(df)} 条")

print("✓ split_data parquet 生成完成")
PYEOF

echo ""

# ─────────────────────────────────────────────────────────────────
# Step 3: 重新生成 RL 数据 (JSON + parquet)
# ─────────────────────────────────────────────────────────────────
echo "[3/4] 重新生成 $PREPARED_DIR/rl/"

mkdir -p "$PREPARED_DIR/rl"

SPLIT_DIR="$SPLIT_DIR" RL_DIR="$PREPARED_DIR/rl" python3 - <<'PYEOF'
import json, pandas as pd, os

split_dir = os.environ["SPLIT_DIR"]
rl_dir = os.environ["RL_DIR"]

for split, name in [("train", "train_seed"), ("val", "val_eval"), ("test", "test_eval")]:
    df = pd.read_parquet(os.path.join(split_dir, f"{split}.parquet"))
    keep = ["文本", "标签"]
    if "all_labels" in df.columns:
        keep.append("all_labels")
    keep += ["toxic_type_label", "expression_label"]
    out = df[keep].copy()
    out.to_json(os.path.join(rl_dir, f"{name}.json"), orient="records", force_ascii=False, indent=2)
    out.to_parquet(os.path.join(rl_dir, f"{name}.parquet"), index=False)
    print(f"  {name}: {len(out)} 条 → .json + .parquet")

print("✓ RL 数据生成完成")
PYEOF

echo ""

# ─────────────────────────────────────────────────────────────────
# Step 4: 验证
# ─────────────────────────────────────────────────────────────────
echo "[4/4] 验证数据完整性"
echo ""

ERRORS=0
check_file() {
    if [ -f "$1" ]; then
        echo "  ✓ $2 ($(du -h "$1" | cut -f1))"
    else
        echo "  ❌ $2 缺失: $1"
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
check_file "$SPLIT_DIR/train.parquet" "train.parquet"
check_file "$SPLIT_DIR/val.json"      "val.json (self-play 评估)"
check_file "$SPLIT_DIR/test.parquet"  "test.parquet"
echo ""

echo "RL 数据:"
check_file "$PREPARED_DIR/rl/train_seed.parquet" "train_seed.parquet (self-play 种子)"
check_file "$PREPARED_DIR/rl/test_eval.json"     "test_eval.json (评估)"
check_file "$PREPARED_DIR/rl/test_eval.parquet"  "test_eval.parquet"
echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo "❌ 发现 $ERRORS 个错误"
    exit 1
fi

echo "============================================================"
echo "✓ 全部数据部署完成！"
echo "============================================================"
echo ""
echo "后续步骤:"
echo "  MODEL_SIZE=3B bash scripts/run_pipeline/run_full_pipeline.sh"
