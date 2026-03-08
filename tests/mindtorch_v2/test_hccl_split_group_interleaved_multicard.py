"""HCCL split_group interleaved calls stress on 2 NPUs."""

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

# Build two parent groups with different rank orders.
parent_a = dist.new_group(ranks=[1, 0], backend="hccl", group_desc="hccl_parent_a_10")
parent_b = dist.new_group(ranks=[0, 1], backend="hccl", group_desc="hccl_parent_b_01")
assert parent_a is not dist.GroupMember.NON_GROUP_MEMBER
assert parent_b is not dist.GroupMember.NON_GROUP_MEMBER

for step in range(5):
    # 1) WORLD split: both ranks included.
    w = dist.split_group(dist.group.WORLD, color=step, key=0)
    assert w is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_world_size(w) == 2

    # 2) parent_a split with equal keys: tie-break by parent rank mapping.
    ga = dist.split_group(parent_a, color=100 + step, key=0)
    assert ga is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_world_size(ga) == 2
    assert dist.get_group_rank(ga, 1) == 0
    assert dist.get_group_rank(ga, 0) == 1

    # 3) parent_b split with rank-specific key ordering.
    gb = dist.split_group(parent_b, color=200 + step, key=(0 if rank == 0 else 1))
    assert gb is not dist.GroupMember.NON_GROUP_MEMBER
    assert dist.get_world_size(gb) == 2
    assert dist.get_group_rank(gb, 0) == 0
    assert dist.get_group_rank(gb, 1) == 1

    # 4) Subset split on WORLD with alternating non-member rank.
    s = dist.split_group(dist.group.WORLD, color=(7 if rank == (step % 2) else -1), key=0)
    if rank == (step % 2):
        assert s is not dist.GroupMember.NON_GROUP_MEMBER
        assert dist.get_world_size(s) == 1
        assert dist.get_process_group_ranks(s) == [rank]
    else:
        assert s is dist.GroupMember.NON_GROUP_MEMBER

    # collective sanity on ga/gb
    xa = torch.tensor([float(rank + 1)], device=device)
    xb = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(xa, group=ga)
    dist.all_reduce(xb, group=gb)
    assert float(xa.to("cpu").item()) == 3.0
    assert float(xb.to("cpu").item()) == 3.0

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL split_group interleaved PASS")
'''


def test_hccl_split_group_interleaved_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29708"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_split_group_interleaved_2card.py"
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
        out, _ = p.communicate(timeout=360)
        txt = out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    if failed:
        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(f"HCCL split_group interleaved failed on ranks: {failed}")
