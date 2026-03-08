"""HCCL all_to_all_single split/numel validation on 2/4/8 NPUs."""

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

mode = os.environ["CASE_MODE"]

device = torch.Device(f"npu:{rank}")
time.sleep(0.05 * rank)
dist.init_process_group("hccl", device_id=device)

base_in = [1 if i == rank else 2 for i in range(world_size)]
base_out = [1 if j == rank else 2 for j in range(world_size)]

if mode == "input_sum_mismatch":
    input_split = list(base_in)
    output_split = list(base_out)
    # Make split sum larger than input numel.
    input_split[rank] += 1
    inp_numel = sum(base_in)
    out_numel = sum(output_split)
elif mode == "output_sum_mismatch":
    input_split = list(base_in)
    output_split = list(base_out)
    # Make split sum larger than output numel.
    output_split[rank] += 1
    inp_numel = sum(input_split)
    out_numel = sum(base_out)
else:
    raise RuntimeError(f"unexpected mode: {mode}")

inp = torch.zeros(inp_numel, device=device)
out = torch.zeros(out_numel, device=device)

try:
    dist.all_to_all_single(
        out,
        inp,
        output_split_sizes=output_split,
        input_split_sizes=input_split,
        async_op=True,
    )
except ValueError as exc:
    msg = str(exc)
    assert "numel" in msg and "split" in msg, msg
else:
    raise AssertionError("expected ValueError for split sum and tensor numel mismatch")

dist.destroy_process_group()
print(f"[rank {rank}] HCCL split/numel validation {mode} {world_size}card PASS")
'''


def _run_once(world_size, master_port, mode):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = str(world_size)
    env["CASE_MODE"] = mode
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = f"/tmp/_hccl_all_to_all_single_split_numel_validation_{world_size}card.py"
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


def _run_case(world_size, master_port, mode):
    retries = 3
    for attempt in range(1, retries + 1):
        failed, outputs = _run_once(world_size, master_port, mode)
        if not failed:
            return

        joined = "\n".join(outputs)
        transient = "resource unavailable" in joined
        if transient and attempt < retries:
            print(
                f"HCCL transient init failure on {world_size} cards ({mode}), "
                f"retry {attempt}/{retries}"
            )
            time.sleep(5)
            continue

        for r, txt in enumerate(outputs):
            print(f"=== RANK {r} ===")
            print(txt)
        raise AssertionError(
            f"HCCL split/numel validation {mode} {world_size}card failed on ranks: {failed}"
        )


@pytest.mark.parametrize(
    "world_size,master_port",
    [
        (2, 29716),
        (4, 29726),
        (8, 29736),
    ],
)
@pytest.mark.parametrize("mode", ["input_sum_mismatch", "output_sum_mismatch"])
def test_hccl_all_to_all_single_split_numel_validation_multicard(world_size, master_port, mode):
    _run_case(world_size, master_port, mode)
