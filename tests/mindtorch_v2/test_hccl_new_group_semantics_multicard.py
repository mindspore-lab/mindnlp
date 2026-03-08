"""HCCL new_group semantics on 2 NPUs."""

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

# Explicit non-default rank order.
pg = dist.new_group(ranks=[1, 0], backend="hccl", group_desc="hccl_new_group_10")
assert pg is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(pg) == 2
assert dist.get_group_rank(pg, 1) == 0
assert dist.get_group_rank(pg, 0) == 1
assert dist.get_global_rank(pg, 0) == 1
assert dist.get_global_rank(pg, 1) == 0

# API returns sorted global ranks.
assert dist.get_process_group_ranks(pg) == [0, 1]

# Collective on explicit group.
x = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(x, group=pg)
assert float(x.to("cpu").item()) == 3.0

dist.destroy_process_group()
print(f"[rank {rank}] HCCL new_group semantics PASS")
'''


def test_hccl_new_group_semantics_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29706"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_new_group_semantics_2card.py"
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
        raise AssertionError(f"HCCL new_group semantics failed on ranks: {failed}")
