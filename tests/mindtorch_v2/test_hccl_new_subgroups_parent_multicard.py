"""HCCL new_subgroups(parent_pg) behavior on 2 NPUs."""

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

# Parent group rank mapping: parent group rank 0->global 1, 1->0.
parent = dist.new_group(ranks=[1, 0], backend="hccl", group_desc="hccl_parent_10")
assert parent is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(parent) == 2
assert dist.get_group_rank(parent, 1) == 0
assert dist.get_group_rank(parent, 0) == 1

my_subgroup, all_subgroups = dist.new_subgroups(group_size=2, group=parent, backend="hccl")
assert len(all_subgroups) == 1
assert my_subgroup is not dist.GroupMember.NON_GROUP_MEMBER

# Process-group ranks API returns sorted global ranks, but group-rank mapping should
# still preserve parent-group rank ordering semantics.
assert dist.get_process_group_ranks(my_subgroup) == [0, 1]
assert dist.get_group_rank(my_subgroup, 1) == 0
assert dist.get_group_rank(my_subgroup, 0) == 1

# Basic subgroup collective sanity.
x = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(x, group=my_subgroup)
assert float(x.to("cpu").item()) == 3.0

dist.destroy_process_group()
print(f"[rank {rank}] HCCL new_subgroups(parent) PASS")
'''


def test_hccl_new_subgroups_parent_group_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29696"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_new_subgroups_parent_2card.py"
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
        out, _ = p.communicate(timeout=240)
        txt = out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    if failed:
        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(f"HCCL new_subgroups(parent) failed on ranks: {failed}")
