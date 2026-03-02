#!/usr/bin/env python
"""Test DistributedDataParallel with 2 processes on CPU.

Usage:
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=2 RANK=0 python test_ddp_cpu.py &
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=2 RANK=1 python test_ddp_cpu.py
"""

import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _set_pg_env(master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)


def _ensure_pg_destroyed():
    if dist.is_initialized():
        dist.destroy_process_group()


def test_ddp_cpu():
    """Test basic DDP functionality on CPU."""
    _ensure_pg_destroyed()
    port = _free_port()
    _set_pg_env(port)
    dist.init_process_group('hccl', rank=0, world_size=1, init_method=f'tcp://127.0.0.1:{port}')
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"[Rank {rank}] Initialized, world_size={world_size}")

        # Create model on CPU
        model = nn.Linear(10, 10)
        ddp_model = nn.DistributedDataParallel(model)

        print(f"[Rank {rank}] DDP model created")

        # Create input (same for both ranks)
        x = torch.ones((4, 10))

        output = ddp_model(x)
        loss = output.sum()
        print(f"[Rank {rank}] loss={loss.item():.4f}")

        loss.backward()
        print(f"[Rank {rank}] backward complete")

        # Check grads exist
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = (param.grad * param.grad).sum().item() ** 0.5
                print(f"[Rank {rank}] {name} grad_norm={grad_norm:.6f}")
            else:
                print(f"[Rank {rank}] {name} grad is None!")

        # Verify grads match across ranks
        if model.weight.grad is not None:
            grad_copy = model.weight.grad.clone()
            # Allreduce grad_copy (SUM) and compare with world_size * local grad
            dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM)
            # If grads are identical, grad_copy == world_size * local_grad
            expected = model.weight.grad * world_size
            diff = (grad_copy - expected).abs().sum().item()
            if diff < 1e-5:
                print(f"[Rank {rank}] ✓ PASS: gradients are synchronized (diff={diff:.10f})")
            else:
                print(f"[Rank {rank}] ✗ FAIL: gradients differ (diff={diff:.10f})")

        dist.barrier()
        print(f"[Rank {rank}] Test complete")
    finally:
        _ensure_pg_destroyed()


if __name__ == "__main__":
    test_ddp_cpu()
