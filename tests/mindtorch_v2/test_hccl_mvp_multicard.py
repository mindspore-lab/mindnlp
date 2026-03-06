"""HCCL MVP baseline verification on 2 NPUs.

Covers minimal distributed capability for production baseline:
- init_process_group / destroy_process_group
- get_rank / get_world_size / barrier
- all_reduce(sum)
- DDP one-step gradient sync
"""

import os
import sys
import subprocess

SCRIPT = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = torch.Device(f"npu:{rank}")

dist.init_process_group(backend="hccl", device_id=device)
assert dist.get_rank() == rank
assert dist.get_world_size() == world_size

# all_reduce sum baseline
x = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(x, op=dist.ReduceOp.SUM)
expected = float(sum(range(1, world_size + 1)))
actual = float(x.to("cpu").item())
assert actual == expected, f"rank={rank} all_reduce mismatch: {actual} vs {expected}"

# DDP one-step
model = nn.Linear(4, 2).to(device)
ddp = nn.parallel.DistributedDataParallel(model)

inp = torch.ones((3, 4), device=device)
loss = ddp(inp).sum()
loss.backward()

assert model.weight.grad is not None
assert model.bias.grad is not None

# verify grad equality across ranks via broadcast from rank0
w_grad = model.weight.grad.clone()
b_grad = model.bias.grad.clone()

dist.broadcast(w_grad, src=0)
dist.broadcast(b_grad, src=0)

wdiff = (model.weight.grad - w_grad).abs().sum().to("cpu").item()
bdiff = (model.bias.grad - b_grad).abs().sum().to("cpu").item()
assert wdiff < 1e-5, f"rank={rank} weight grad mismatch {wdiff}"
assert bdiff < 1e-5, f"rank={rank} bias grad mismatch {bdiff}"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL MVP PASS")
'''


def test_hccl_mvp_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29621"
    env["WORLD_SIZE"] = "2"
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src") + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_mvp_2card_worker.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    procs = []
    for rank in range(2):
        env_rank = {**env, "RANK": str(rank)}
        p = subprocess.Popen(
            [sys.executable, worker_file],
            env=env_rank,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)

    outputs = []
    failed = False
    for rank, p in enumerate(procs):
        try:
            out, _ = p.communicate(timeout=180)
            text = out.decode()
            outputs.append((rank, p.returncode, text))
            if p.returncode != 0:
                failed = True
        except subprocess.TimeoutExpired:
            p.kill()
            outputs.append((rank, -1, "TIMEOUT"))
            failed = True

    for rank, code, text in outputs:
        print(f"=== rank {rank} exit={code} ===")
        print(text)

    assert not failed, "HCCL MVP 2-card worker failed"
