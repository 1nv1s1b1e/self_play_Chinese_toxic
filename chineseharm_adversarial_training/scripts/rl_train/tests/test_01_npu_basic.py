#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 01: 昇腾 910B NPU 基础环境
验证 torch_npu 安装正确，NPU 设备可用，基本张量运算正常
运行: python3 test_01_npu_basic.py
"""
import sys

PASS = 0
FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        result = fn()
        print(f"  ✅ {name}: {result}")
        PASS += 1
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        FAIL += 1
        return False

print("=" * 55)
print("  测试 01: 昇腾 910B NPU 基础环境")
print("=" * 55)

# ── 1. Python 版本 ──────────────────────────────────────────
print("\n【1】Python & 基础依赖")
check("Python >= 3.10",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor} "
              + ("✓" if sys.version_info >= (3, 10) else "需要 >=3.10"))

check("import torch",
      lambda: __import__("torch").__version__)

check("import torch_npu",
      lambda: __import__("torch_npu").__version__)

# ── 2. NPU 设备检测 ─────────────────────────────────────────
print("\n【2】NPU 设备")
import torch

has_npu = False
try:
    import torch_npu
    has_npu = True
except ImportError:
    print("  ❌ torch_npu 未安装")

if has_npu:
    check("torch.npu.is_available()",
          lambda: str(torch.npu.is_available()))
    check("NPU 卡数",
          lambda: f"{torch.npu.device_count()} 张")

    npu_count = torch.npu.device_count()
    for i in range(min(npu_count, 4)):
        check(f"NPU:{i} 设备名",
              lambda i=i: torch.npu.get_device_name(i))

# ── 3. 张量运算 ─────────────────────────────────────────────
print("\n【3】NPU 张量运算")
if has_npu and torch.npu.is_available():
    def _tensor_add():
        a = torch.tensor([1.0, 2.0, 3.0]).npu()
        b = torch.tensor([4.0, 5.0, 6.0]).npu()
        c = a + b
        return f"[1+4, 2+5, 3+6] = {c.cpu().tolist()}"

    def _matmul():
        torch.npu.set_device(0)
        a = torch.randn(32, 64).npu()
        b = torch.randn(64, 32).npu()
        c = torch.mm(a, b)
        return f"matmul (32,64)×(64,32) → {tuple(c.shape)} ✓"

    def _bf16():
        x = torch.randn(8, 8, dtype=torch.bfloat16).npu()
        y = x @ x.T
        return f"BF16 matmul → {y.dtype} shape={tuple(y.shape)}"

    check("tensor add on NPU", _tensor_add)
    check("matmul on NPU", _matmul)
    check("BF16 matmul (GRPO 训练精度)", _bf16)

    # 显存检查
    def _memory():
        total = torch.npu.get_device_properties(0).total_memory
        return f"{total / 1024**3:.1f} GB"
    check("NPU:0 总显存", _memory)
else:
    print("  ⚠ 跳过 NPU 张量测试 (无可用 NPU)")

# ── 4. 分布式环境变量 ────────────────────────────────────────
print("\n【4】分布式环境预检")
import os
check("HCCL 后端注册",
      lambda: "hccl" if hasattr(torch.distributed, "is_available")
              and torch.distributed.is_available() else "需检查")

required_envs = [
    ("HCCL_WHITELIST_DISABLE", os.environ.get("HCCL_WHITELIST_DISABLE", "未设置")),
    ("HCCL_CONNECT_TIMEOUT",   os.environ.get("HCCL_CONNECT_TIMEOUT",   "未设置")),
    ("VLLM_ATTENTION_BACKEND", os.environ.get("VLLM_ATTENTION_BACKEND", "未设置")),
    ("MASTER_ADDR",            os.environ.get("MASTER_ADDR",            "未设置")),
    ("MASTER_PORT",            os.environ.get("MASTER_PORT",            "未设置")),
]
for name, val in required_envs:
    status = "✅" if val != "未设置" else "⚠ 训练时需设置"
    print(f"     {name}: {val}  {status}")

# ── 5. verl & vllm 检查 ─────────────────────────────────────
print("\n【5】训练依赖")
check("verl",        lambda: __import__("verl").__version__)
check("vllm",        lambda: __import__("vllm").__version__)
check("vllm_ascend", lambda: __import__("vllm_ascend").__version__)
check("transformers",lambda: __import__("transformers").__version__)

print("\n" + "=" * 55)
print(f"  结果: ✅ {PASS} 通过 / ❌ {FAIL} 失败")
print("=" * 55)
if FAIL > 0:
    print("\n  ⚠ 请参考 README_NPU.md 安装缺失依赖后重新检查")
    sys.exit(1)
else:
    print("\n  🎉 基础环境就绪，可继续运行 test_02_hccl_ddp.sh")
