"""HCCL new_subgroups_by_enumeration empty-input validation on 2 NPUs."""

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

for bad in (None, []):
    try:
        dist.new_subgroups_by_enumeration(bad, backend="hccl")
    except ValueError as exc:
        assert "cannot be empty" in str(exc).lower()
    else:
        raise AssertionError(f"expected ValueError for ranks_per_subgroup_list={bad}")

dist.destroy_process_group()
print(f"[rank {rank}] HCCL enum empty validation PASS")
'''


def test_hccl_new_subgroups_by_enum_empty_input_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29701"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_new_subgroups_by_enum_empty_2card.py"
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
        raise AssertionError(f"HCCL enum empty validation failed on ranks: {failed}")
