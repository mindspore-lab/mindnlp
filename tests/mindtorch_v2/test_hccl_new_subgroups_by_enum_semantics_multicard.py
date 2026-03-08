"""HCCL new_subgroups_by_enumeration semantics on 2 NPUs."""

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

my_group, groups = dist.new_subgroups_by_enumeration([[0], [1]], backend="hccl")
assert len(groups) == 2
assert my_group is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(my_group) == 1
assert dist.get_process_group_ranks(my_group) == [rank]
assert dist.get_group_rank(my_group, rank) == 0
assert dist.get_global_rank(my_group, 0) == rank

# Single-rank subgroup all_reduce should be no-op and deterministic.
x = torch.tensor([float(rank + 10)], device=device)
dist.all_reduce(x, group=my_group)
assert float(x.to("cpu").item()) == float(rank + 10)

# Partial membership: rank1 is not part of [0] split.
my_partial, partial_groups = dist.new_subgroups_by_enumeration([[0]], backend="hccl")
assert len(partial_groups) == 1
if rank == 0:
    assert my_partial is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_process_group_ranks(my_partial) == [0]
else:
    assert my_partial is dist.GroupMember.NON_GROUP_MEMBER

dist.destroy_process_group()
print(f"[rank {rank}] HCCL enum semantics PASS")
'''


def test_hccl_new_subgroups_by_enum_semantics_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29700"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_new_subgroups_by_enum_semantics_2card.py"
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
        raise AssertionError(f"HCCL enum semantics failed on ranks: {failed}")
