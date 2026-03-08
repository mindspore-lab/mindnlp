"""HCCL all_to_all_single async unequal-split semantics on 2 NPUs."""

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

# Use legal unequal splits: pairwise send/recv sizes must match across ranks.
# rank0: input_split=[1,3], output_split=[1,3]
# rank1: input_split=[3,1], output_split=[3,1]
if rank == 0:
    input_split = [1, 3]
    output_split = [1, 3]
    inp = torch.tensor([0.0, 10.0, 11.0, 12.0], device=device)
    expected = [0.0, 100.0, 101.0, 102.0]
else:
    input_split = [3, 1]
    output_split = [3, 1]
    inp = torch.tensor([100.0, 101.0, 102.0, 110.0], device=device)
    expected = [10.0, 11.0, 12.0, 110.0]

out = torch.zeros(4, device=device)

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
assert actual == expected, f"rank={rank} actual={actual}, expected={expected}"

# Repeat once to verify reusable stability.
out2 = torch.zeros(4, device=device)
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
print(f"[rank {rank}] HCCL all_to_all_single async unequal PASS")
'''


def test_hccl_all_to_all_single_async_unequal_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29714"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + \
        (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_all_to_all_single_async_unequal_2card.py"
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
        raise AssertionError(f"HCCL all_to_all_single async unequal failed on ranks: {failed}")
