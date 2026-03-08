"""new_group sequence stability on gloo backend with 2 local ranks."""

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

assert world_size == 2
dist.init_process_group(backend="gloo")

# Rank1 does not join this group, but both ranks must advance group sequence.
solo = dist.new_group(ranks=[0], backend="gloo")
if rank == 0:
    assert solo is not dist.GroupMember.NON_GROUP_MEMBER
else:
    assert solo is dist.GroupMember.NON_GROUP_MEMBER

# This second implicit group creation should not deadlock and should be consistent.
full = dist.new_group(ranks=[0, 1], backend="gloo")
assert full is not dist.GroupMember.NON_GROUP_MEMBER
assert dist.get_world_size(full) == 2
assert dist.get_process_group_ranks(full) == [0, 1]
assert dist.get_group_rank(full, rank) == rank

dist.destroy_process_group()
print(f"[rank {rank}] new_group sequence checks passed")
'''


def test_new_group_sequence_gloo_world2():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29542"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_new_group_sequence_gloo_worker.py"
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
            f"new_group worker failed: rank0={p0.returncode}, rank1={p1.returncode}"
        )
