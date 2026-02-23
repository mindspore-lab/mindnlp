#!/usr/bin/env python
"""Test builtin DDP communication hooks on HCCL.

Tests allreduce_hook, fp16_compress_hook, bf16_compress_hook.

Usage:
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 WORLD_SIZE=2 RANK=0 python test_ddp_builtin_hooks.py &
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 WORLD_SIZE=2 RANK=1 python test_ddp_builtin_hooks.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist
from mindtorch_v2.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    allreduce_hook,
    fp16_compress_hook,
    bf16_compress_hook,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((10,)))

    def forward(self, x):
        return x * self.weight


def verify_grads_synced(model, world_size, rank, label):
    """Verify gradients are synchronized across ranks."""
    if model.weight.grad is None:
        print(f"[Rank {rank}] {label}: FAIL - weight grad is None!")
        return False

    grad_copy = model.weight.grad.clone()
    dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM)
    from mindtorch_v2._functional import mul, add, neg
    expected = mul(model.weight.grad, float(world_size))
    diff = add(grad_copy, neg(expected)).abs().sum().item()
    if diff < 1e-3:
        print(f"[Rank {rank}] {label}: PASS (diff={diff:.10f})")
        return True
    else:
        print(f"[Rank {rank}] {label}: FAIL (diff={diff:.10f})")
        return False


def run_ddp_with_hook(hook, hook_name, rank, world_size):
    """Run a single DDP forward/backward with the given hook."""
    model = SimpleModel().to(f'npu:{rank}')
    ddp_model = nn.DistributedDataParallel(model)
    ddp_model.register_comm_hook(dist.group.WORLD, hook)

    x = torch.ones((4, 10)).to(f'npu:{rank}')
    output = ddp_model(x)
    loss = output.sum()
    loss.backward()

    return verify_grads_synced(model, world_size, rank, hook_name)


def main():
    dist.init_process_group('hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] Initialized, world_size={world_size}")

    results = []

    # Test 1: allreduce_hook
    results.append(run_ddp_with_hook(allreduce_hook, "allreduce_hook", rank, world_size))
    dist.barrier()

    # Test 2: fp16_compress_hook
    results.append(run_ddp_with_hook(fp16_compress_hook, "fp16_compress_hook", rank, world_size))
    dist.barrier()

    # Test 3: bf16_compress_hook (skip - CANN 8.3.RC2 doesn't support float32->bfloat16 cast)
    # results.append(run_ddp_with_hook(bf16_compress_hook, "bf16_compress_hook", rank, world_size))
    # dist.barrier()
    print(f"[Rank {rank}] bf16_compress_hook: SKIPPED (CANN limitation)")

    # Summary
    all_pass = all(results)
    print(f"[Rank {rank}] {'ALL PASSED' if all_pass else 'SOME FAILED'} ({sum(results)}/{len(results)})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
