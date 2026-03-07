"""Gloo DDP world2 semantics for no_sync/find_unused_parameters/static_graph."""

import os
import subprocess
import sys


SCRIPT = r'''
import os, sys
import numpy as np

src_dir = os.environ.get("MINDTORCH_V2_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist


class OptionalMiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x, use_middle=True):
        x = self.fc1(x)
        if use_middle:
            x = self.fc2(x)
        return self.fc3(x)


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="gloo")

# 1) no_sync: first backward should not synchronize, second should.
model_a = nn.Linear(4, 1)
ddp_a = nn.parallel.DistributedDataParallel(model_a)
x0 = torch.tensor(np.full((2, 4), rank + 1, dtype=np.float32))
x1 = torch.tensor(np.full((2, 4), rank + 2, dtype=np.float32))

with ddp_a.no_sync():
    ddp_a(x0).sum().backward()
local_grad_before = model_a.weight.grad.clone()
before_list = [torch.zeros_like(local_grad_before) for _ in range(world_size)]
dist.all_gather(before_list, local_grad_before)
diff_before = (before_list[0] - before_list[1]).abs().sum().item()
assert diff_before > 1e-6

ddp_a(x1).sum().backward()
local_grad_after = model_a.weight.grad.clone()
after_list = [torch.zeros_like(local_grad_after) for _ in range(world_size)]
dist.all_gather(after_list, local_grad_after)
diff_after = (after_list[0] - after_list[1]).abs().sum().item()
assert diff_after < 1e-5

# 2) find_unused_parameters=True with branch-disabled middle layer.
model_b = OptionalMiddle()
ddp_b = nn.parallel.DistributedDataParallel(model_b, find_unused_parameters=True)
x = torch.tensor(np.random.randn(3, 4).astype(np.float32))
ddp_b(x, use_middle=False).sum().backward()
assert model_b.fc2.weight.grad is not None
assert torch.allclose(model_b.fc2.weight.grad, torch.zeros_like(model_b.fc2.weight.grad))

# 3) static_graph=True should keep cached unused set across iterations.
model_c = OptionalMiddle()
ddp_c = nn.parallel.DistributedDataParallel(model_c, static_graph=True)
y = torch.tensor(np.random.randn(3, 4).astype(np.float32))
ddp_c(y, use_middle=False).sum().backward()
cached = ddp_c.reducer._cached_unused_param_indices
assert cached is not None and len(cached) > 0
ddp_c(y, use_middle=False).sum().backward()
assert ddp_c.reducer._cached_unused_param_indices is cached

dist.destroy_process_group()
print(f"[rank {rank}] DDP world2 semantics checks passed")
'''


def test_gloo_ddp_world2_semantics():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29545"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_gloo_ddp_semantics_worker.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    p0 = subprocess.Popen(
        [sys.executable, worker_file],
        env={**env, "RANK": "0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    p1 = subprocess.Popen(
        [sys.executable, worker_file],
        env={**env, "RANK": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    out0, _ = p0.communicate(timeout=120)
    out1, _ = p1.communicate(timeout=120)

    if p0.returncode != 0 or p1.returncode != 0:
        print("=== RANK 0 ===")
        print(out0.decode())
        print("=== RANK 1 ===")
        print(out1.decode())
        raise AssertionError(
            f"ddp world2 worker failed: rank0={p0.returncode}, rank1={p1.returncode}"
        )
