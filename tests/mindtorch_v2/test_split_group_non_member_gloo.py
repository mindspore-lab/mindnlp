"""split_group non-member parent behavior on gloo backend with 2 local ranks."""

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

# parent is only created on rank0; rank1 gets NON_GROUP_MEMBER sentinel.
parent = dist.new_group(ranks=[0], backend="gloo", group_desc="split_parent_rank0_only")
if rank == 0:
    assert parent is not dist.GroupMember.NON_GROUP_MEMBER
else:
    assert parent is dist.GroupMember.NON_GROUP_MEMBER

# split_group should behave like torch-style non-member behavior: return NON_GROUP_MEMBER
# for the caller when parent_pg is NON_GROUP_MEMBER, rather than raising.
sub = dist.split_group(parent, color=1, key=0)
if rank == 0:
    assert sub is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_world_size(sub) == 1
    assert dist.get_process_group_ranks(sub) == [0]
else:
    assert sub is dist.GroupMember.NON_GROUP_MEMBER

dist.destroy_process_group()
print(f"[rank {rank}] split_group non-member checks passed")
'''


def test_split_group_non_member_parent_gloo_world2():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29543"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_split_group_non_member_gloo_worker.py"
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
            f"split_group non-member worker failed: rank0={p0.returncode}, rank1={p1.returncode}"
        )
