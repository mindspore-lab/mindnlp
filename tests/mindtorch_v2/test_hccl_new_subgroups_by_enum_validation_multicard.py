"""HCCL new_subgroups_by_enumeration input validation on 2 NPUs."""

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

bad_inputs = [
    [[0, 0]],      # duplicate in subgroup
    [[0], [0]],    # overlap across subgroups
    [[-1]],        # negative rank
    [[2]],         # rank >= world_size
]
for ranks_per_subgroup in bad_inputs:
    try:
        dist.new_subgroups_by_enumeration(ranks_per_subgroup, backend="hccl")
    except ValueError as exc:
        msg = str(exc).lower()
        assert "rank" in msg or "duplicate" in msg or "overlap" in msg
    else:
        raise AssertionError(f"expected ValueError for {ranks_per_subgroup}")

dist.destroy_process_group()
print(f"[rank {rank}] HCCL enum validation PASS")
'''


def test_hccl_new_subgroups_by_enum_validation_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29699"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_new_subgroups_by_enum_validation_2card.py"
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
        raise AssertionError(f"HCCL enum validation failed on ranks: {failed}")
