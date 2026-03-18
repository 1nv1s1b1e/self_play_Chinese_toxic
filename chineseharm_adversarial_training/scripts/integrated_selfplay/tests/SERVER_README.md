# 远程服务器操作步骤

## 前置条件
- 昇腾 910B NPU 服务器 (4 卡)
- Qwen2.5-3B-Instruct 基座模型

## Step 0: 上传文件

将本地修改的文件上传到服务器。需要上传的文件:

```bash
# 数据文件 (最关键)
corrected_data/train.json
corrected_data/val.json
corrected_data/test.json
corrected_data/sft_train.jsonl
corrected_data/sft_val.jsonl
corrected_data/multi_label_map.json

# 代码文件 (整个 integrated_selfplay 目录)
scripts/integrated_selfplay/
```

或者直接 rsync 整个项目:
```bash
rsync -avz --exclude='*.bak_wrong_labels' \
    chineseharm_adversarial_training/ \
    user@server:/home/ma-user/work/test/
```

## Step 1: 在服务器上替换数据

```bash
cd /home/ma-user/work/test

# 替换 split_data (评估脚本读这个目录)
cp corrected_data/train.json split_data/train.json
cp corrected_data/val.json split_data/val.json
cp corrected_data/test.json split_data/test.json

# 替换 prepared_data (self-play 种子数据)
# 注意: parquet 文件需要在服务器上重建
python3 -c "
import pandas as pd, json
for name in ['train_seed', 'test_eval', 'val_eval']:
    src = f'prepared_data/rl/{name}.json'
    dst = f'prepared_data/rl/{name}.parquet'
    df = pd.read_json(src)
    df.to_parquet(dst, index=False)
    print(f'  {dst}: {len(df)} rows')
"
```

## Step 2: 全量 re-SFT (精简版 prompt, 正确标签)

```bash
cd /home/ma-user/work/test/scripts/integrated_selfplay

# 全量 SFT (8167 条, 3 epoch, ~1小时)
SFT_SAMPLES=0 \
BASE_DIR=/home/ma-user/work/test \
BASE_MODEL=/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct \
N_GPUS=4 \
bash tests/run_resft_and_validate.sh
```

这会自动执行:
1. SFT 数据准备 (用预生成的 sft_train.jsonl)
2. LoRA SFT 训练 (3B, 8167条, 3 epoch)
3. Merge LoRA → 完整模型
4. 评估新模型
5. Temperature 多样性测试
6. 3 步 mini self-play

## Step 3: 检查结果

```bash
# 查看评估结果
cat validation_*/step4a_eval_new.log | grep -A 20 "对比结果"

# 查看 temperature 测试
cat validation_*/step5_temperature.log | grep "unique_cat"

# 查看 self-play 趋势
cat validation_*/step6_selfplay.log | grep "评估"
```

## Step 4: 全量 self-play (如果验证通过)

```bash
# 用新模型跑 30 步 self-play
REVIEWER_INIT=/path/to/validation_*/merged_reviewer_short \
SEED_DATA=/home/ma-user/work/test/corrected_data/train.json \
TOTAL_STEPS=30 \
bash run_selfplay.sh
```
