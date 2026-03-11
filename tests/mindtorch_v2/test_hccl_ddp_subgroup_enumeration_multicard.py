"""HCCL DDP with enumeration-created subgroup on 2 NPUs."""

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

subgroup, groups = dist.new_subgroups_by_enumeration([[0, 1]], backend="hccl")
assert subgroup is not dist.GroupMember.NON_GROUP_MEMBER
assert len(groups) == 1
assert dist.get_world_size(subgroup) == 2

# Deterministic init, then DDP on subgroup.
torch.manual_seed(1234)
model = nn.Linear(4, 2).to(device)
ddp = nn.parallel.DistributedDataParallel(model, process_group=subgroup)

x = torch.ones((3, 4), device=device)
loss = ddp(x).sum()
loss.backward()

# Verify synchronized grads across subgroup.
w_ref = model.weight.grad.clone()
b_ref = model.bias.grad.clone()
dist.broadcast(w_ref, src=0, group=subgroup)
dist.broadcast(b_ref, src=0, group=subgroup)
wdiff = (model.weight.grad - w_ref).abs().sum().to("cpu").item()
bdiff = (model.bias.grad - b_ref).abs().sum().to("cpu").item()
assert wdiff < 1e-5, f"rank={rank} subgroup weight grad mismatch {wdiff}"
assert bdiff < 1e-5, f"rank={rank} subgroup bias grad mismatch {bdiff}"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL DDP enumeration subgroup PASS")
'''


def test_hccl_ddp_enumeration_subgroup_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29705"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_ddp_enumeration_subgroup_2card.py"
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
        out, _ = p.communicate(timeout=300)
        txt = out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    if failed:
        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(f"HCCL DDP enumeration subgroup failed on ranks: {failed}")
