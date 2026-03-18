#!/bin/bash
# =============================================================================
# 测试 02: 昇腾 HCCL 单机多卡通信测试
# 验证 DDP + HCCL 后端在 NPU 多卡下的 AllReduce / Broadcast / AllGather
#
# 用法:
#   bash test_02_hccl_ddp.sh           # 自动检测卡数
#   N_GPUS=2 bash test_02_hccl_ddp.sh  # 指定卡数
# =============================================================================

# 加载昇腾 CANN 环境
[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ] && \
    source /usr/local/Ascend/nnal/atb/set_env.sh

# 昇腾 HCCL 必需环境变量
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# 自动探测 NPU 卡数
N_GPUS="${N_GPUS:-$(python3 -c 'import torch_npu,torch; print(torch.npu.device_count())' 2>/dev/null || echo 2)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══════════════════════════════════════════════════════════"
echo "  测试 02: 昇腾 HCCL 单机多卡通信"
echo "  NPU 卡数: $N_GPUS"
echo "  通信后端: HCCL  (昇腾替代 NCCL)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# 使用 torchrun 启动, 它会注入 LOCAL_RANK / RANK / WORLD_SIZE
# 等同于官方昇腾文档推荐的多进程启动方式
python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$N_GPUS \
    "$SCRIPT_DIR/test_02_hccl_ddp_worker.py"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ HCCL 多卡通信测试通过 (world_size=$N_GPUS)"
    echo "  → 可以继续运行 test_03_reward_fn.py"
else
    echo "  ❌ HCCL 测试失败 (exit=$EXIT_CODE)"
    echo ""
    echo "  常见原因:"
    echo "    1. CANN 环境未加载: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "    2. 端口被占用: 修改 MASTER_PORT"
    echo "    3. Bus error: 重新运行通常可解决 (已知昇腾偶发问题)"
fi
exit $EXIT_CODE
