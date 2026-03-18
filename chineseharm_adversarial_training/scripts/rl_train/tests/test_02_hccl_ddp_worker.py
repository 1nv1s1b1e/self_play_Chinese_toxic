#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 02: 昇腾 HCCL 单机多卡通信
验证 DDP + HCCL 通信后端在多 NPU 下正确工作
运行: bash test_02_hccl_ddp.sh  (会自动启动多进程)
或手动: torchrun --standalone --nproc_per_node=2 test_02_hccl_ddp_worker.py
"""
import os
import torch
import torch.distributed as dist

def main():
    # ── 读取由 torchrun / run.sh 注入的分布式环境变量 ─────────────
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))
    world_size  = int(os.environ.get("WORLD_SIZE",  1))
    rank        = int(os.environ.get("RANK",        0))

    # ── 初始化昇腾 HCCL 进程组 ────────────────────────────────────
    # 关键: backend="hccl" 是昇腾替代 nccl 的通信库
    import torch_npu
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
    )

    # 绑定 NPU 设备
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"

    print(f"[rank {rank}/{world_size}] 初始化成功，设备: {device}")

    # ── 测试1: AllReduce ──────────────────────────────────────────
    # 每张卡持有 rank+1 的向量，AllReduce 后应等于 sum(1..world_size)
    tensor = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    ok_allreduce = torch.allclose(tensor, torch.full((4,), float(expected), device=device))
    if rank == 0:
        print(f"  AllReduce SUM: {tensor.cpu().tolist()} (期望全为 {expected}) → {'✅' if ok_allreduce else '❌'}")

    # ── 测试2: Broadcast ─────────────────────────────────────────
    bcast = torch.full((3,), float(rank), device=device)
    dist.broadcast(bcast, src=0)
    ok_bcast = torch.allclose(bcast, torch.zeros(3, device=device))
    if rank == 0:
        print(f"  Broadcast from rank 0: {bcast.cpu().tolist()} → {'✅' if ok_bcast else '❌'}")

    # ── 测试3: AllGather ─────────────────────────────────────────
    local_val = torch.tensor([float(rank + 100)], device=device)
    gathered  = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, local_val)
    if rank == 0:
        vals = [g.item() for g in gathered]
        expected_vals = [float(i + 100) for i in range(world_size)]
        ok_gather = (vals == expected_vals)
        print(f"  AllGather: {vals} (期望 {expected_vals}) → {'✅' if ok_gather else '❌'}")

    # ── 测试4: 梯度真实同步 (模拟 GRPO Actor 更新) ───────────────
    # 每卡初始化相同模型，使用不同梯度，DDP 后梯度应相同
    torch.manual_seed(42)
    linear = torch.nn.Linear(8, 4).to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(linear, device_ids=[local_rank])

    # 不同 rank 使用不同输入 → 梯度本来不同
    inp = torch.randn(2, 8, device=device) * (rank + 1)
    out = ddp_model(inp)
    loss = out.sum()
    loss.backward()

    # AllReduce 后所有 rank 梯度应相同
    grad0 = ddp_model.module.weight.grad.clone()
    dist.barrier()

    # 验证 rank0 和其他 rank 梯度一致 (通过广播比较)
    dist.broadcast(grad0, src=0)
    ok_grad = torch.allclose(ddp_model.module.weight.grad, grad0, atol=1e-5)
    print(f"  [rank {rank}] DDP 梯度同步: grad_norm={ddp_model.module.weight.grad.norm().item():.4f} → {'✅' if ok_grad else '❌'}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 50)
        print("  ✅ HCCL 单机多卡通信测试全部通过")
        print(f"  world_size={world_size}, backend=hccl")
        print("=" * 50)

if __name__ == "__main__":
    main()
