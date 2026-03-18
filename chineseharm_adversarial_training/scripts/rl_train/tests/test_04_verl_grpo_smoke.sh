#!/bin/bash
# =============================================================================
# 测试 04: verl GRPO 最小化冒烟测试 (昇腾 910B 单机多卡)
# =============================================================================
# 使用最小配置 (0.5B 模型, 极小 batch) 验证完整 GRPO 训练流程:
#   数据加载 → rollout → 奖励计算 → PPO 更新 → checkpoint 保存
#
# 预期运行时间: 5~15 分钟 (取决于 NPU 速度)
# 前置: test_01 和 test_03 通过
#
# 用法:
#   N_GPUS=2 bash test_04_verl_grpo_smoke.sh
#
#   # 指定本地已有模型 (推荐，最快)
#   MODEL_PATH=/home/ma-user/work/test/merged_models_toxicn/challenger_0.5B \
#     N_GPUS=2 bash test_04_verl_grpo_smoke.sh
#
#   # 或指定基础模型 (run_01 下载后可用)
#   MODEL_PATH=/home/ma-user/work/test/models_base/Qwen/Qwen2.5-0.5B-Instruct \
#     N_GPUS=2 bash test_04_verl_grpo_smoke.sh
#
#   # 不指定则自动按优先级查找:
#   #   ① merged_models_toxicn/challenger_0.5B  (run_04 输出)
#   #   ② models_base/Qwen/Qwen2.5-0.5B-Instruct (run_01 下载)
#   #   ③ ModelScope 在线下载 → models_base/Qwen/
#   #   ④ hf-mirror (https://hf-mirror.com)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RL_SCRIPTS="$(dirname "$SCRIPT_DIR")"

# ── 昇腾环境 ────────────────────────────────────────────────────
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh

export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29700       # 测试专用端口，避免与正式训练冲突

# ── 与 LoRA 脚本保持一致: 禁止 torch 在 Ray Worker 中自动加载 NPU backend ──────
# 自动加载就会与 HCCL 多进程初始化产生竞争，導致 hcclCommInitRootInfoConfig error code 5
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# ── 关键: 防止 Ray Worker 覆盖 Ascend 设备可见性 ────────────────
export RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1

# ── HCCL bootstrap 网络接口 ─────────────────────────────────────
# awk 在部分环境返回空字符且 exit 0，改用 Python 多策略探测
_HCCL_IF=$(python3 -c "
import socket, subprocess, sys
# 策略 1: 连接外网获取默认出口 IP对应的网卡
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    local_ip = s.getsockname()[0]
    s.close()
    import subprocess
    r = subprocess.check_output(['ip', '-4', 'addr'], stderr=subprocess.DEVNULL, text=True)
    for line in r.splitlines():
        if local_ip in line:
            # 找到该 IP 对应的接口名
            import re
            iface = subprocess.check_output(
                ['ip', '-4', 'addr', 'show'], stderr=subprocess.DEVNULL, text=True)
            for blk in iface.split('\\n\\n'):
                if local_ip in blk:
                    m = re.search(r'^\\d+: (\\S+):', blk, re.MULTILINE)
                    if m: print(m.group(1)); sys.exit(0)
except Exception: pass
# 策略 2: 取第一个非 lo 的 IPv4 网卡
try:
    r = subprocess.check_output(['ip', '-4', '-o', 'addr', 'show'], stderr=subprocess.DEVNULL, text=True)
    for line in r.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] != 'lo':
            print(parts[1]); sys.exit(0)
except Exception: pass
# 策略 3: hostname -I 取第一个 IP 再反查网卡
try:
    ip = subprocess.check_output(['hostname', '-I'], stderr=subprocess.DEVNULL, text=True).split()[0]
    r = subprocess.check_output(['ip', '-4', 'addr'], stderr=subprocess.DEVNULL, text=True)
    import re
    for blk in re.split(r'^\\d+:', r, flags=re.MULTILINE):
        if ip in blk:
            m = re.search(r' (\\S+)\\s*$', blk.splitlines()[0] if blk.splitlines() else '')
            if m: print(m.group(1).rstrip(':')); sys.exit(0)
except Exception: pass
# 策略 4: eth0 存在则用 eth0
try:
    subprocess.check_call(['ip', 'link', 'show', 'eth0'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('eth0'); sys.exit(0)
except Exception: pass
print('eth0')
" 2>/dev/null || echo "eth0")
# 防御性备用: 如果上面返回空字符串则回落 eth0
[ -z "$_HCCL_IF" ] && _HCCL_IF="eth0"
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-$_HCCL_IF}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$_HCCL_IF}

export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export VLLM_ATTENTION_BACKEND=XFORMERS
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

# ── 参数 ─────────────────────────────────────────────────────────
N_GPUS="${N_GPUS:-2}"
# 与 run_pipeline 各脚本保持一致的项目根路径
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
# 优先用环境变量指定的本地绝对路径；若未指定则按优先级自动查找
MODEL_PATH="${MODEL_PATH:-}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/verl_smoke_test}"
# 下载缓存与基础模型目录对齐 ($BASE_DIR/models_base/Qwen)，避免重复下载
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$BASE_DIR/models_base/Qwen}"
REWARD_FUNC="$RL_SCRIPTS/reward_functions/challenger_reward_v7.py"

# N_GPUS 确定后计算 ASCEND_RT_VISIBLE_DEVICES (与 run_selfplay_npu.sh 逻辑一致)
export ASCEND_RT_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($N_GPUS)))" 2>/dev/null || echo "0,1")

# 国内镜像：先尝试 hf-mirror，再尝试 ModelScope
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$MODEL_CACHE_DIR}"

mkdir -p "$SMOKE_DIR/data" "$SMOKE_DIR/output" "$MODEL_CACHE_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  测试 04: verl GRPO 最小化冒烟测试"
echo "  NPU 卡数: $N_GPUS"
echo "  BASE_DIR: $BASE_DIR"
echo "  输出目录: $SMOKE_DIR"
echo "  HF 镜像 : $HF_ENDPOINT"
echo "  ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
echo "  RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES: $RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES"
echo "═══════════════════════════════════════════════════════════"

# ── Step 0.5: HCCL 专项环境预检 ─────────────────────────────────
# 专门验证导致 hcclCommInitRootInfoConfig error code 5 的各项条件
echo ""
echo "── Step 0.5: HCCL/Ray/Ascend 环境预检"
python3 - <<'PYEOF_CHECK'
import sys, os, subprocess

PASS, FAIL = 0, 0
def ok(msg):  global PASS; PASS+=1; print(f"  ✅ {msg}")
def err(msg): global FAIL; FAIL+=1; print(f"  ❌ {msg}")
def warn(msg): print(f"  ⚠  {msg}")

# 1. torch_npu 可 import
try:
    import torch, torch_npu
    ok(f"torch_npu {torch_npu.__version__} 可导入")
except ImportError as e:
    err(f"torch_npu 未安装: {e}"); sys.exit(1)

# 2. NPU 数量充足
npu_count = torch.npu.device_count()
n_gpus = int(os.environ.get("N_GPUS", 2))
if npu_count >= n_gpus:
    ok(f"NPU 数量 {npu_count} >= 请求卡数 {n_gpus}")
else:
    err(f"NPU 数量 {npu_count} < 请求卡数 {n_gpus}，无法启动 FSDP")
    sys.exit(1)

# 3. ASCEND_RT_VISIBLE_DEVICES 正确 (必须含 N_GPUS 张卡)
arvd = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
expected = ",".join(str(i) for i in range(n_gpus))
if arvd == expected:
    ok(f"ASCEND_RT_VISIBLE_DEVICES={arvd} (与 N_GPUS={n_gpus} 匹配)")
else:
    err(f"ASCEND_RT_VISIBLE_DEVICES='{arvd}' 应为 '{expected}'")
    err("  → 这会导致 HCCL hcclCommInitRootInfoConfig error code 5")
    FAIL += 1

# 4. RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1 (核心修复)
val = os.environ.get("RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES", "")
if val == "1":
    ok("RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1 (防止 Ray 覆盖设备可见性)")
else:
    err(f"RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES='{val}'，应为 '1'")
    err("  → 缺少此项是 hcclCommInitRootInfoConfig error code 5 的主因")
    FAIL += 1

# 4b. TORCH_DEVICE_BACKEND_AUTOLOAD=0 (与 LoRA 脚本保持一致)
val2 = os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD", "")
if val2 == "0":
    ok("TORCH_DEVICE_BACKEND_AUTOLOAD=0 (防止 Ray Worker 自动加载 NPU backend)")
else:
    err(f"TORCH_DEVICE_BACKEND_AUTOLOAD='{val2}'，应为 '0'")
    err("  → Ray Worker 自动加载 NPU backend 会与 HCCL 多进程 init 冲突——这是最新次错误的直接原因")
    FAIL += 1

# 5. HCCL_SOCKET_IFNAME 有效
ifname = os.environ.get("HCCL_SOCKET_IFNAME", "")
if ifname:
    # 验证网卡确实存在
    try:
        import subprocess
        subprocess.check_call(["ip", "link", "show", ifname],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok(f"HCCL_SOCKET_IFNAME={ifname} (网卡存在)")
    except Exception:
        # 网卡名存在但无法验证，键入 warn 而非错误——内网环境可能无 ip 命令
        warn(f"HCCL_SOCKET_IFNAME={ifname} (已设置，但无法验证网卡是否存在，可继续)")
        PASS += 1  # 已设置就计为通过
else:
    err("HCCL_SOCKET_IFNAME 未设置，HCCL bootstrap 可能在 Ray 沙箱内失败")
    err("  → 手动设置: export HCCL_SOCKET_IFNAME=$(ip -4 route get 1 | awk '{print $5}')")
    FAIL += 1

# 6. verl 可 import
try:
    import verl
    ok(f"verl 可导入 (路径: {verl.__file__})")
except ImportError as e:
    err(f"verl 未安装: {e}"); FAIL += 1

# 7. vllm-ascend 可 import
try:
    import vllm
    ok(f"vllm {vllm.__version__} 可导入")
except ImportError as e:
    err(f"vllm 未安装: {e}"); FAIL += 1

# 8. ray 可 import
try:
    import ray
    ok(f"ray {ray.__version__} 可导入")
except ImportError as e:
    err(f"ray 未安装: {e}"); FAIL += 1

# 9. CANN toolkit 已加载 (检查 libascendcl.so 或 ASCEND_TOOLKIT_HOME)
ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
if ascend_home and os.path.isdir(ascend_home):
    ok(f"CANN 已加载 ASCEND_TOOLKIT_HOME={ascend_home}")
else:
    warn("ASCEND_TOOLKIT_HOME 未设置，请确认 set_env.sh 已 source")

print(f"")
if FAIL == 0:
    print(f"  ✅ 预检通过 ({PASS} 项)，可以启动 verl GRPO")
else:
    print(f"  ❌ 预检失败 {FAIL} 项 / 通过 {PASS} 项")
    print(f"  hcclCommInitRootInfoConfig error code 5 的解决条件:")
    print(f"    ① RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1  ← 阻止 Ray 覆盖设备")
    print(f"    ② ASCEND_RT_VISIBLE_DEVICES=0,1,...  ← 正确的设备 ID 列表")
    print(f"    ③ HCCL_SOCKET_IFNAME=<网卡名>  ← HCCL bootstrap 网络接口")
    print(f"    ④ 无残留 Ray 进程  ← 运行前执行 ray stop --force")
    sys.exit(1)
PYEOF_CHECK
echo "  ✅ Step 0.5 完成"

# ── Step 0: 解析/下载模型 ────────────────────────────────────────
echo ""
echo "── Step 0: 解析模型路径"

if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
    # 用户已指定本地路径
    echo "  ✅ 使用本地模型: $MODEL_PATH"
else
    # 按优先级自动查找，与 run_pipeline 路径约定一致：
    #   ① SFT 合并模型   $BASE_DIR/merged_models_toxicn/challenger_0.5B  (run_04 输出)
    #   ② 基础模型        $BASE_DIR/models_base/Qwen/Qwen2.5-0.5B-Instruct (run_01 下载)
    #   ③ ModelScope 下载 → 存入 $BASE_DIR/models_base/Qwen/
    #   ④ hf-mirror 下载
    MERGED_PATH="$BASE_DIR/merged_models_toxicn/challenger_0.5B"
    BASE_MODEL_PATH="$BASE_DIR/models_base/Qwen/Qwen2.5-0.5B-Instruct"

    if [ -d "$MERGED_PATH" ]; then
        MODEL_PATH="$MERGED_PATH"
        echo "  ✅ 使用 SFT 合并模型 (run_04 输出): $MODEL_PATH"
    elif [ -d "$BASE_MODEL_PATH" ]; then
        MODEL_PATH="$BASE_MODEL_PATH"
        echo "  ✅ 使用基础模型 (run_01 下载): $MODEL_PATH"
    else
        echo "  本地未找到模型，尝试从 ModelScope 下载 Qwen2.5-0.5B-Instruct ..."
        echo "  (下载至 $MODEL_CACHE_DIR，与 run_01 缓存目录一致)"
        MODEL_PATH=$(python3 - <<'PYEOF_DL'
import sys, os

cache_dir = os.environ.get("MODELSCOPE_CACHE",
            os.environ.get("BASE_DIR", "/home/ma-user/work/test") + "/models_base/Qwen")
model_id  = "Qwen/Qwen2.5-0.5B-Instruct"

# ── 方案 A: ModelScope SDK ────────────────────────────────────
try:
    from modelscope import snapshot_download
    local_path = snapshot_download(model_id, cache_dir=cache_dir)
    print(local_path)
    sys.exit(0)
except Exception as e:
    print(f"  [warn] ModelScope 下载失败: {e}", file=sys.stderr)

# ── 方案 B: huggingface_hub (走 hf-mirror) ──────────────────
try:
    from huggingface_hub import snapshot_download as hf_dl
    local_path = hf_dl(repo_id=model_id, cache_dir=cache_dir, local_files_only=False)
    print(local_path)
    sys.exit(0)
except Exception as e:
    print(f"  [warn] hf-mirror 下载失败: {e}", file=sys.stderr)

base = os.environ.get("BASE_DIR", "/home/ma-user/work/test")
print("", file=sys.stderr)
print("  ❌ 无法自动下载，请选择以下任一方式:", file=sys.stderr)
print(f"     # 方式 1: 先完成前置步骤 (run_01 下载基础模型)", file=sys.stderr)
print(f"     bash run_01_download.sh", file=sys.stderr)
print(f"     # 方式 2: 先完成 SFT 训练 (run_03 + run_04)", file=sys.stderr)
print(f"     bash run_03_lora_sft.sh && bash run_04_merge_lora.sh", file=sys.stderr)
print(f"     # 方式 3: 手动指定本地路径", file=sys.stderr)
print(f"     MODEL_PATH={base}/models_base/Qwen/Qwen2.5-0.5B-Instruct bash test_04_verl_grpo_smoke.sh", file=sys.stderr)
sys.exit(1)
PYEOF_DL
)
        [ $? -ne 0 ] && exit 1
        echo "  ✅ 模型已下载至: $MODEL_PATH"
    fi
fi

echo "  使用模型: $MODEL_PATH"

# ── Step 1: 生成极小测试 parquet 数据 ───────────────────────────
echo ""
echo "── Step 1: 生成最小测试数据"
python3 - <<'PYEOF'
import pandas as pd
import os

output_dir = os.environ.get("SMOKE_DIR", "/tmp/verl_smoke_test") + "/data"

# 必须与 prepare_rl_grpo_data.py 输出格式完全一致:
# verl 需要字段: data_source, prompt(list[dict]), ability,
#                reward_model(dict: style+ground_truth), extra_info(dict)
categories = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
records = []
for i in range(20):
    cat = categories[i % len(categories)]
    ref_text = f"这是关于{cat}的参考测试文本，供Challenger学习主题相关性。"
    records.append({
        "data_source": "toxicn_challenger",
        "prompt": [
            {"role": "system", "content": "你是一个文本生成助手。根据给定类别生成对应文本，只输出文本本身。"},
            {"role": "user",   "content": f"请生成一条关于'{cat}'主题的网络文本，50字以内。"},
        ],
        "ability": "safety",
        "reward_model": {
            "style": "rule",
            "ground_truth": ref_text,   # verl 传给 compute_score 的 ground_truth
        },
        "extra_info": {
            "split": "train",
            "index": i,
            "category": cat,
            "original_text": ref_text,
            "expression": "显式仇恨",
            "toxic_type": "仇恨言论",
        },
    })

df = pd.DataFrame(records)
train_path = os.path.join(output_dir, "smoke_train.parquet")
val_path   = os.path.join(output_dir, "smoke_val.parquet")
df[:16].to_parquet(train_path, index=False)
df[16:].to_parquet(val_path,   index=False)
print(f"  ✅ 生成训练数据: {train_path} ({len(df[:16])} 条)")
print(f"  ✅ 生成验证数据: {val_path} ({len(df[16:])} 条)")
PYEOF

echo "  ✅ Step 1 完成"

# ── Step 2: 验证 parquet 格式 ────────────────────────────────────
echo ""
echo "── Step 2: 验证数据格式"
python3 - <<PYEOF2
import pandas as pd, json, sys
df = pd.read_parquet("$SMOKE_DIR/data/smoke_train.parquet")
print(f"  行数: {len(df)}, 列: {list(df.columns)}")
# verl 真实需要的列名 (与 prepare_rl_grpo_data.py 输出对齐)
required = ["data_source", "prompt", "reward_model", "extra_info"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"  ❌ 缺少列: {missing}")
    sys.exit(1)
# 验证 reward_model 结构
rm0 = df['reward_model'].iloc[0]
if isinstance(rm0, dict) and 'ground_truth' in rm0:
    print(f"  ✅ reward_model.ground_truth 存在")
else:
    print(f"  ❌ reward_model 格式错误: {rm0}")
    sys.exit(1)
print(f"  ✅ 数据格式正确 (与 prepare_rl_grpo_data.py 输出一致)")
print(f"  示例 prompt[0]: {df['prompt'].iloc[0][0]}")
PYEOF2

# ── Step 3: 验证奖励函数在 verl 场景下的调用 ─────────────────────
echo ""
echo "── Step 3: 在线验证奖励函数调用 (verl 真实调用方式)"
python3 - <<PYEOF3
import sys, os
sys.path.insert(0, "$RL_SCRIPTS/reward_functions")
from challenger_reward_v7 import compute_score
import pandas as pd

df = pd.read_parquet("$SMOKE_DIR/data/smoke_train.parquet")
errors = []
for i, row in df.iterrows():
    try:
        # verl 实际传给 compute_score 的参数:
        # ground_truth = reward_model["ground_truth"]
        # extra_info   = extra_info 列
        rm = row["reward_model"]
        ground_truth = rm["ground_truth"] if isinstance(rm, dict) else str(rm)
        extra = row["extra_info"] if isinstance(row["extra_info"], dict) else {}

        score = compute_score(
            data_source=row["data_source"],
            solution_str="这是一段测试生成的文本，内容与参考文本相关。",
            ground_truth=ground_truth,
            extra_info=extra,
        )
        assert -1.0 <= score <= 1.0, f"score={score} 超出范围"
    except Exception as e:
        errors.append(f"row {i}: {e}")

if errors:
    print(f"  ❌ {len(errors)} 条失败: {errors[:3]}")
    sys.exit(1)
print(f"  ✅ {len(df)} 条数据奖励函数调用正常 (reward_model.ground_truth 路径)")
PYEOF3

# ── Step 4: 启动最小 verl GRPO 训练 (1 epoch) ───────────────────
echo ""
echo "── Step 4: 启动 verl GRPO 训练 (1 epoch 冒烟)"
echo "   模型: $MODEL_PATH"
echo "   批量: train_batch_size=16, rollout.n=4"
echo "   ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"
echo "   预计: 5~15 分钟"
echo ""

# 清理残留 Ray 进程 (上次失败后可能遗留，会导致 HCCL 端口冲突)
echo "  清理残留 Ray 进程 ..."
ray stop --force 2>/dev/null || true
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$SMOKE_DIR/data/smoke_train.parquet" \
    data.val_files="$SMOKE_DIR/data/smoke_val.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm-ascend \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    custom_reward_function.path="$REWARD_FUNC" \
    custom_reward_function.name=compute_score \
    \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.device=npu \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='smoke_test' \
    trainer.experiment_name='grpo_challenger_smoke' \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$SMOKE_DIR/output" \
    2>&1 | tee "$SMOKE_DIR/smoke_step4.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "═══════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ 冒烟测试通过！verl GRPO 在昇腾 910B 上运行正常"
    echo ""
    echo "  checkpoint 保存在: $SMOKE_DIR/output/actor/"
    echo ""
    echo "  现在可以运行完整训练:"
    echo "     N_GPUS=$N_GPUS MODEL_SIZE=3B bash ../run_pipeline/run_09_rl_selfplay.sh"
else
    echo "  ❌ 冒烟测试失败 (exit=$EXIT_CODE)"
    echo ""
    # 自动分析日志中的已知错误
    LOG="$SMOKE_DIR/smoke_step4.log"
    if grep -q "hcclCommInitRootInfoConfig" "$LOG" 2>/dev/null; then
        echo "  ► 诊断: 检测到 hcclCommInitRootInfoConfig error code 5"
        echo "    原因: Ray Worker 启动时 Ascend 设备可见性被覆盖"
        echo "    验证: 以下三项必须全部满足 ─"
        echo "      ① RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES=1  当前: ${RAY_EXPERIMENTAL_NOSET_ASCEND_VISIBLE_DEVICES}"
        echo "      ② ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES (应为 0,1,...共 $N_GPUS 张)"
        echo "      ③ 无残留 Ray 进程: ray stop --force 后重试"
    elif grep -q "Bus error\|Segmentation fault" "$LOG" 2>/dev/null; then
        echo "  ► 诊断: Bus error / Segfault (昇腾偶发，直接重试通常可解决)"
        echo "    ray stop --force && N_GPUS=$N_GPUS bash test_04_verl_grpo_smoke.sh"
    elif grep -q "out of memory\|OOM" "$LOG" 2>/dev/null; then
        echo "  ► 诊断: NPU 显存不足"
        echo "    尝试: 减小 gpu_memory_utilization 或开启 param_offload"
    elif grep -q "Connection refused\|address already in use" "$LOG" 2>/dev/null; then
        echo "  ► 诊断: 端口 $MASTER_PORT 被占用"
        echo "    尝试: MASTER_PORT=29701 bash test_04_verl_grpo_smoke.sh"
    else
        echo "  排查步骤:"
        echo "    1. 查看完整日志: cat $SMOKE_DIR/smoke_step4.log | grep -E 'ERROR|error|Error' | tail -30"
        echo "    2. 检查 CANN 环境: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        echo "    3. 检查 vllm-ascend 版本: python3 -c 'import vllm; print(vllm.__version__)'"
    fi
fi
echo "═══════════════════════════════════════════════════════════"
exit $EXIT_CODE
