"""HCCL all_to_all_single invalid split-size validation on 2/4/8 NPUs."""

import os
import subprocess
import sys
import time

import pytest


SCRIPT = r'''
import os, sys, time
src_dir = os.environ.get("MINDTORCH_V2_SRC")
if src_dir:
    sys.path.insert(0, src_dir)

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = torch.Device(f"npu:{rank}")
# Reduce HCCL init burst on large-card jobs.
time.sleep(0.05 * rank)
dist.init_process_group("hccl", device_id=device)

# Baseline profile: send 1 item to self, 2 items to other peers.
input_split = [1 if i == rank else 2 for i in range(world_size)]
output_split = [1 if j == rank else 2 for j in range(world_size)]

# Break pairwise compatibility only on rank 0:
# rank0 -> rank1 send count becomes 3, but rank1's output from rank0 remains 2.
if rank == 0:
    input_split[1] = 3

vals = []
for dst in range(world_size):
    cnt = input_split[dst]
    for k in range(cnt):
        vals.append(float(rank * 1000 + dst * 10 + k))
inp = torch.tensor(vals, device=device)
out = torch.zeros(sum(output_split), device=device)

try:
    dist.all_to_all_single(
        out,
        inp,
        output_split_sizes=output_split,
        input_split_sizes=input_split,
        async_op=True,
    )
except ValueError as exc:
    assert "split mismatch" in str(exc)
else:
    raise AssertionError("expected ValueError for invalid all_to_all_single split pairing")

dist.destroy_process_group()
print(f"[rank {rank}] HCCL invalid split validation {world_size}card PASS")
'''


def _run_once(world_size, master_port):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = f"/tmp/_hccl_all_to_all_single_invalid_split_{world_size}card.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    failed = []
    outputs = []
    procs = []

    for r in range(world_size):
        p = subprocess.Popen(
            [sys.executable, worker_file],
            env={**env, "RANK": str(r)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(p)

    timeout = 420 if world_size <= 4 else 900
    for r, p in enumerate(procs):
        try:
            out, _ = p.communicate(timeout=timeout)
            txt = out.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
            txt = "TIMEOUT\n" + out.decode("utf-8", errors="replace")
        outputs.append(txt)
        if p.returncode != 0:
            failed.append(r)

    return failed, outputs


def _run_case(world_size, master_port):
    retries = 3
    for attempt in range(1, retries + 1):
        failed, outputs = _run_once(world_size, master_port)
        if not failed:
            return

        joined = "\n".join(outputs)
        transient = "resource unavailable" in joined
        if transient and attempt < retries:
            print(
                f"HCCL transient init failure on {world_size} cards, "
                f"retry {attempt}/{retries}"
            )
            time.sleep(5)
            continue

        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(
            f"HCCL invalid split validation {world_size}card failed on ranks: {failed}"
        )


@pytest.mark.parametrize(
    "world_size,master_port",
    [
        (2, 29715),
        (4, 29725),
        (8, 29735),
    ],
)
def test_hccl_all_to_all_single_invalid_split_pairing_multicard(world_size, master_port):
    _run_case(world_size, master_port)
