#!/usr/bin/env python
"""Test DDP with register_comm_hook.

Tests custom communication hook for gradient reduction.

Usage:
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=2 RANK=0 python test_ddp_comm_hook.py &
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=2 RANK=1 python test_ddp_comm_hook.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist
import mindtorch_v2.futures as futures


class SimpleModel(nn.Module):
    """Simple model using elementwise ops (avoids matmul)."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((10,)))

    def forward(self, x):
        return x * self.weight


def custom_allreduce_hook(state, bucket):
    """Custom comm hook that does standard allreduce."""
    print(f"[Rank {dist.get_rank()}] Custom hook called for bucket {bucket.index()}")

    # Get the buffer
    buffer = bucket.buffer()

    # Allreduce
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)

    # Divide by world size
    from mindtorch_v2._functional import mul
    result = mul(buffer, 1.0 / state['world_size'])

    # Return a Future
    fut = futures.Future()
    fut.set_result(result)
    return fut


def test_ddp_comm_hook():
    dist.init_process_group('hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] Initialized, world_size={world_size}")

    # Create model and move to NPU
    model = SimpleModel().to(f'npu:{rank}')
    print(f"[Rank {rank}] weight device: {model.weight.device}")

    # Wrap with DDP
    ddp_model = nn.DistributedDataParallel(model)

    # Register custom comm hook
    state = {'world_size': world_size}
    ddp_model.register_comm_hook(state, custom_allreduce_hook)
    print(f"[Rank {rank}] Registered custom comm hook")

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

        # Verify grads are synchronized
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
    test_ddp_comm_hook()
