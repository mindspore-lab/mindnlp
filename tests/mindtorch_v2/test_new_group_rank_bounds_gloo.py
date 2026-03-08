"""new_group rank bounds validation on gloo backend."""

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

for bad_ranks in ([-1], [2], [0, 2], [-1, 1]):
    try:
        dist.new_group(ranks=bad_ranks, backend="gloo", group_desc=f"bad_{rank}_{bad_ranks}")
    except ValueError as exc:
        assert "range" in str(exc).lower() or "world_size" in str(exc).lower()
    else:
        raise AssertionError(f"expected ValueError for ranks={bad_ranks}")

dist.destroy_process_group()
print(f"[rank {rank}] new_group bounds checks passed")
'''


def test_new_group_rank_bounds_rejected_gloo_world2():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29545"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_new_group_bounds_gloo_worker.py"
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
            f"new_group bounds worker failed: rank0={p0.returncode}, rank1={p1.returncode}"
        )
