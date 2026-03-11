"""split_group behavior on gloo backend with 2 local ranks."""

import os
import subprocess
import sys


SCRIPT = r'''
import os, sys
src_dir = os.environ.get("MINDTORCH_V2_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend="gloo")
assert world_size == 2

# Put only rank0 into subgroup.
sg = dist.split_group(dist.group.WORLD, color=(0 if rank == 0 else -1), key=0)
if rank == 0:
    assert sg is not dist.group.WORLD
    assert dist.get_world_size(sg) == 1
    assert dist.get_rank(sg) == 0
    assert dist.get_process_group_ranks(sg) == [0]
    assert dist.get_group_rank(sg, 0) == 0
    assert dist.get_global_rank(sg, 0) == 0
else:
    assert sg is dist.GroupMember.NON_GROUP_MEMBER

# Both ranks into same color should create 2-rank group.
sg2 = dist.split_group(dist.group.WORLD, color=7, key=0)
assert sg2 is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(sg2) == 2
assert dist.get_process_group_ranks(sg2) == [0, 1]
assert dist.get_group_rank(sg2, rank) == rank
assert dist.get_global_rank(sg2, rank) == rank

# Parent PG rank order should be used as key tie-breaker when keys are equal.
parent = dist.new_group(ranks=[1, 0], backend="gloo", group_desc="split_parent_1_0")
assert parent is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_group_rank(parent, 1) == 0
assert dist.get_group_rank(parent, 0) == 1

sg3 = dist.split_group(parent, color=3, key=0)
assert sg3 is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(sg3) == 2
assert dist.get_group_rank(sg3, 1) == 0
assert dist.get_group_rank(sg3, 0) == 1
assert dist.get_global_rank(sg3, 0) == 1
assert dist.get_global_rank(sg3, 1) == 0

# split_group sequence should be per-parent-group.
# Calling split_group only on rank0's parent subgroup must not desync a later
# WORLD split_group call.
parent_single = dist.new_group(ranks=[0], backend="gloo", group_desc="split_parent_single_0")
if rank == 0:
    sg4 = dist.split_group(parent_single, color=11, key=0)
    assert sg4 is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_world_size(sg4) == 1
    assert dist.get_process_group_ranks(sg4) == [0]
else:
    assert parent_single is dist.GroupMember.NON_GROUP_MEMBER

sg5 = dist.split_group(dist.group.WORLD, color=9, key=0)
assert sg5 is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(sg5) == 2
assert dist.get_process_group_ranks(sg5) == [0, 1]

dist.destroy_process_group()
print(f"[rank {rank}] split_group checks passed")
'''


def test_split_group_gloo_world2():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29541"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_split_group_gloo_worker.py"
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

    out0, _ = p0.communicate(timeout=20)
    out1, _ = p1.communicate(timeout=20)

    if p0.returncode != 0 or p1.returncode != 0:
        print("=== RANK 0 ===")
        print(out0.decode())
        print("=== RANK 1 ===")
        print(out1.decode())
        raise AssertionError(
            f"split_group worker failed: rank0={p0.returncode}, rank1={p1.returncode}"
        )
