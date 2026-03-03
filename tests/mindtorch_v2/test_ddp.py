#!/usr/bin/env python
"""Test DistributedDataParallel with 2 processes.

Uses a simple model that avoids matmul to isolate DDP testing.

Usage:
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=2 RANK=0 python test_ddp.py &
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=2 RANK=1 python test_ddp.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


class SimpleModel(nn.Module):
    """Simple model using elementwise ops (avoids matmul)."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((10,)))

    def forward(self, x):
        return x * self.weight


def test_ddp():
    dist.init_process_group('hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Initialized, world_size={world_size}")

    # Create model and move to NPU
    model = SimpleModel().to(f'npu:{rank}')
    print(f"[Rank {rank}] weight device: {model.weight.device}")

    # Wrap with DDP
    ddp_model = nn.DistributedDataParallel(model)
    print(f"[Rank {rank}] DDP model created")

    # Create input on NPU
    x = torch.ones((4, 10)).to(f'npu:{rank}')

    # Forward
    output = ddp_model(x)
    loss = output.sum()
    print(f"[Rank {rank}] loss={loss.item():.4f}")

    # Backward
    loss.backward()
    print(f"[Rank {rank}] backward complete")

    # Check gradients
    if model.weight.grad is not None:
        grad_sum = model.weight.grad.sum().item()
        print(f"[Rank {rank}] weight grad_sum={grad_sum:.6f}")

        # Verify grads are synchronized:
        # If DDP is working, all ranks should have identical gradients.
        # We verify by allreducing a copy: if grads are identical,
        # sum == world_size * local_grad
        grad_copy = model.weight.grad.clone()
        dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM)
        from mindtorch_v2._functional import mul, add, neg
        expected = mul(model.weight.grad, float(world_size))
        diff = add(grad_copy, neg(expected)).abs().sum().item()
        if diff < 1e-5:
            print(f"[Rank {rank}] PASS: gradients synchronized (diff={diff:.10f})")
        else:
            print(f"[Rank {rank}] FAIL: gradients differ (diff={diff:.10f})")
    else:
        print(f"[Rank {rank}] FAIL: weight grad is None!")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    test_ddp()
