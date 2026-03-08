"""HCCL subgroup lifecycle stress: create/use/destroy loops on 2 NPUs."""

import os
import subprocess
import sys


SCRIPT = r'''
import os, sys
src_dir = os.environ.get("MINDTORCH_V2_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

assert world_size == 2
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

for step in range(8):
    # A: explicit new_group lifecycle.
    g_explicit = dist.new_group(ranks=[1, 0], backend="hccl", group_desc=f"hccl_life_explicit_{step}")
    assert g_explicit is not dist.GroupMember.NON_GROUP_MEMBER
    x = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(x, group=g_explicit)
    assert float(x.to("cpu").item()) == 3.0
    dist.destroy_process_group(g_explicit)

    # B: split_group-created lifecycle.
    g_split = dist.split_group(dist.group.WORLD, color=1000 + step, key=0)
    assert g_split is not dist.GroupMember.NON_GROUP_MEMBER
    y = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(y, group=g_split)
    assert float(y.to("cpu").item()) == 3.0
    dist.destroy_process_group(g_split)

    # WORLD barrier between iterations.
    dist.barrier()

# WORLD should remain usable after repeated subgroup destroy.
z = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(z)
assert float(z.to("cpu").item()) == 3.0

dist.destroy_process_group()
print(f"[rank {rank}] HCCL group lifecycle PASS")
'''


def test_hccl_group_lifecycle_create_use_destroy_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29713"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_group_lifecycle_2card.py"
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
        out, _ = p.communicate(timeout=480)
        txt = out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    if failed:
        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(f"HCCL group lifecycle failed on ranks: {failed}")
