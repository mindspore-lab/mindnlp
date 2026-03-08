"""HCCL split_group repeated calls followed by DDP subgroup training."""

import os
import subprocess
import sys


SCRIPT = r'''
import os, sys
src_dir = os.environ.get("MINDTORCH_V2_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

assert world_size == 2
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

# Repeated split_group invocations to stress sequence bookkeeping and communicator setup.
for step in range(8):
    g = dist.split_group(dist.group.WORLD, color=10 + step, key=0)
    assert g is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_world_size(g) == 2

# Use a split-created group for DDP to verify post-split training stability.
train_group = dist.split_group(dist.group.WORLD, color=999, key=0)
assert train_group is not dist.GroupMember.NON_GROUP_MEMBER

model = nn.Linear(4, 2).to(device)
ddp = nn.parallel.DistributedDataParallel(model, process_group=train_group)

for step in range(3):
    x = torch.ones((3, 4), device=device)
    loss = ddp(x).sum()
    loss.backward()

    ref = model.weight.grad.clone()
    dist.broadcast(ref, src=0, group=train_group)
    diff = (model.weight.grad - ref).abs().sum().to("cpu").item()
    assert diff < 1e-5, f"step={step} rank={rank} grad mismatch {diff}"

    with torch.no_grad():
        model.weight.grad = None
        model.bias.grad = None

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL split->DDP PASS")
'''


def test_hccl_split_group_then_ddp_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29709"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_split_group_then_ddp_2card.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    failed = []
    outputs = []
    procs = []
    for r in range(2):
        p = subprocess.Popen(
            [sys.executable, worker_file],
            env={**env, "RANK": str(r)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)

    for r, p in enumerate(procs):
        out, _ = p.communicate(timeout=420)
        txt = out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    if failed:
        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(f"HCCL split->DDP failed on ranks: {failed}")
