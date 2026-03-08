"""HCCL monitored_barrier behavior on 2 NPUs."""

import os
import subprocess
import sys


SCRIPT_STRESS = r'''
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

sub = dist.new_group(ranks=[0, 1], backend="hccl", group_desc="hccl_barrier_sub")
assert sub is not dist.GroupMember.NON_GROUP_MEMBER

for step in range(8):
    dist.monitored_barrier(group=dist.group.WORLD)
    dist.monitored_barrier(group=sub)

# Collective sanity after repeated barriers.
x = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(x, group=sub)
assert float(x.to("cpu").item()) == 3.0

dist.destroy_process_group()
print(f"[rank {rank}] HCCL monitored_barrier stress PASS")
'''


SCRIPT_REJECT = r'''
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

try:
    dist.monitored_barrier(wait_all_ranks=True)
except NotImplementedError as exc:
    assert "wait_all_ranks" in str(exc)
else:
    raise AssertionError("expected NotImplementedError for wait_all_ranks=True on hccl")

dist.destroy_process_group()
print(f"[rank {rank}] HCCL monitored_barrier reject PASS")
'''


SCRIPT_TIMEOUT = r'''
import os, sys
from datetime import timedelta
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

for _ in range(4):
    dist.monitored_barrier(timeout=timedelta(seconds=5))

# Check collectives still work.
x = torch.tensor([float(rank + 1)], device=device)
dist.all_reduce(x)
assert float(x.to("cpu").item()) == 3.0

dist.destroy_process_group()
print(f"[rank {rank}] HCCL monitored_barrier timeout PASS")
'''


def _run_two_rank(script_text: str, worker_name: str, master_port: int, timeout_sec: int = 300):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = f"/tmp/{worker_name}.py"
    with open(worker_file, "w") as f:
        f.write(script_text)

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
        out, _ = p.communicate(timeout=timeout_sec)
        txt = out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    if failed:
        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(f"{worker_name} failed on ranks: {failed}")


def test_hccl_monitored_barrier_stress_2card():
    _run_two_rank(SCRIPT_STRESS, "_hccl_monitored_barrier_stress_2card", master_port=29710, timeout_sec=360)


def test_hccl_monitored_barrier_reject_wait_all_ranks_2card():
    _run_two_rank(SCRIPT_REJECT, "_hccl_monitored_barrier_reject_2card", master_port=29711, timeout_sec=240)


def test_hccl_monitored_barrier_timeout_2card():
    _run_two_rank(SCRIPT_TIMEOUT, "_hccl_monitored_barrier_timeout_2card", master_port=29712, timeout_sec=240)
