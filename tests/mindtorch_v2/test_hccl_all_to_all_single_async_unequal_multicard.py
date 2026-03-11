"""HCCL all_to_all_single async unequal-split semantics on 2/4/8 NPUs."""

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

# Pairwise-consistent unequal split profile for all ranks:
# send 1 item to self, 2 items to every other peer.
input_split = [1 if i == rank else 2 for i in range(world_size)]
output_split = [1 if j == rank else 2 for j in range(world_size)]

vals = []
for dst in range(world_size):
    cnt = input_split[dst]
    for k in range(cnt):
        vals.append(float(rank * 1000 + dst * 10 + k))
inp = torch.tensor(vals, device=device)
out = torch.zeros(sum(output_split), device=device)

w = dist.all_to_all_single(
    out,
    inp,
    output_split_sizes=output_split,
    input_split_sizes=input_split,
    async_op=True,
)
assert w is not None
w.wait()

actual = list(out.to("cpu")._numpy_view())
expected = []
for src in range(world_size):
    cnt = 1 if src == rank else 2
    for k in range(cnt):
        expected.append(float(src * 1000 + rank * 10 + k))
assert actual == expected, f"rank={rank} actual={actual}, expected={expected}"

# Repeat once for stability.
out2 = torch.zeros(sum(output_split), device=device)
w2 = dist.all_to_all_single(
    out2,
    inp,
    output_split_sizes=output_split,
    input_split_sizes=input_split,
    async_op=True,
)
w2.wait()
actual2 = list(out2.to("cpu")._numpy_view())
assert actual2 == expected, f"rank={rank} repeat actual={actual2}, expected={expected}"

dist.destroy_process_group()
print(f"[rank {rank}] HCCL all_to_all_single async unequal {world_size}card PASS")
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

    worker_file = f"/tmp/_hccl_all_to_all_single_async_unequal_{world_size}card.py"
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
            f"HCCL all_to_all_single async unequal {world_size}card failed on ranks: {failed}"
        )


@pytest.mark.parametrize(
    "world_size,master_port",
    [
        (2, 29714),
        (4, 29724),
        (8, 29734),
    ],
)
def test_hccl_all_to_all_single_async_unequal_multicard(world_size, master_port):
    _run_case(world_size, master_port)
