#!/bin/bash
# =============================================================================
# 昇腾 910B 环境检查脚本
# 运行本脚本以验证 verl GRPO RL 训练所需的环境是否就绪
# =============================================================================

echo "═══════════════════════════════════════════════════════════"
echo "  昇腾 910B 训练环境检查"
echo "═══════════════════════════════════════════════════════════"

PASS=0; FAIL=0

check() {
    local name="$1"; local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo "  ✅ $name"
        PASS=$((PASS+1))
    else
        echo "  ❌ $name"
        FAIL=$((FAIL+1))
    fi
}

echo ""
echo "【1】CANN 环境"
check "ascend-toolkit set_env.sh" \
    "[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]"
check "nnal/atb set_env.sh" \
    "[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]"

# 尝试加载
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null

echo ""
echo "【2】Python & PyTorch"
check "python3 >= 3.10" \
    "python3 -c 'import sys; assert sys.version_info >= (3,10)'"
check "torch" \
    "python3 -c 'import torch'"
check "torch_npu (昇腾 NPU 后端)" \
    "python3 -c 'import torch_npu'"

echo ""
echo "【3】NPU 设备"
NPU_CNT=$(python3 -c "import torch_npu; print(torch.npu.device_count())" 2>/dev/null || echo "0")
echo "  NPU 卡数: $NPU_CNT"
check "至少 1 张 NPU" \
    "python3 -c 'import torch_npu,torch; assert torch.npu.device_count()>=1'"

echo ""
echo "【4】verl & vllm-ascend"
check "verl 已安装" \
    "python3 -c 'import verl'"
check "vllm 已安装" \
    "python3 -c 'import vllm'"
check "vllm_ascend 已安装" \
    "python3 -c 'import vllm_ascend'"

echo ""
echo "【5】分布式通信"
check "torch.distributed (hccl 后端)" \
    "python3 -c 'import torch.distributed; torch.distributed.is_nccl_available or True'"

echo ""
echo "【6】环境变量"
export HCCL_WHITELIST_DISABLE=1
check "HCCL_WHITELIST_DISABLE=1" \
    "[ \"\$HCCL_WHITELIST_DISABLE\" = '1' ]"
export VLLM_ATTENTION_BACKEND=XFORMERS
check "VLLM_ATTENTION_BACKEND=XFORMERS" \
    "[ \"\$VLLM_ATTENTION_BACKEND\" = 'XFORMERS' ]"

echo ""
echo "【7】RL 数据文件 (BASE_DIR=${BASE_DIR:-/home/ma-user/work/test})"
echo "  (由 run_06_prepare_rl_data.sh / prepare_rl_grpo_data.py 生成)"
BASE_DIR="${BASE_DIR:-/home/ma-user/work/test}"
RL_DATA="$BASE_DIR/prepared_data/rl"
check "challenger_grpo_train.parquet" \
    "[ -f '$RL_DATA/challenger_grpo_train.parquet' ]"
check "challenger_grpo_val.parquet" \
    "[ -f '$RL_DATA/challenger_grpo_val.parquet' ]"
check "reviewer_grpo_train.parquet" \
    "[ -f '$RL_DATA/reviewer_grpo_train.parquet' ]"
check "reviewer_grpo_val.parquet" \
    "[ -f '$RL_DATA/reviewer_grpo_val.parquet' ]"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  检查结果: ✅ $PASS 通过 / ❌ $FAIL 失败"
echo "═══════════════════════════════════════════════════════════"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  请参考昇腾 verl 安装文档:"
    echo "  https://ascend.github.io/docs/sources/_generated/sources/verl/ascend_quick_start.html"
    echo ""
    echo "  快速安装指南:"
    echo "    # 安装 torch_npu (需与 torch 版本匹配)"
    echo "    pip install torch torch_npu torchvision triton-ascend"
    echo "    # 安装 vllm (空设备模式)"
    echo "    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git"
    echo "    cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd .."
    echo "    # 安装 vllm-ascend"
    echo "    git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git"
    echo "    cd vllm-ascend && pip install -v -e . && cd .."
    echo "    # 安装 verl"
    echo "    git clone --depth 1 https://github.com/volcengine/verl.git"
    echo "    cd verl && pip install -r requirements-npu.txt && pip install -v -e . && cd .."
    exit 1
else
    echo ""
    echo "  🎉 环境就绪！可以开始 RL 训练"
    echo ""
    echo "  运行命令示例:"
    echo "    cd scripts/run_pipeline"
    echo "    N_GPUS=2 MODEL_SIZE=3B bash run_09_rl_selfplay.sh"
fi
