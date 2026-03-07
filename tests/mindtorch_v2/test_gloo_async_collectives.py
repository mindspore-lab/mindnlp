"""Gloo async collective wait semantics with 2 local ranks."""

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

dist.init_process_group(backend="gloo")
assert world_size == 2

# 1) all_gather(async)
src = torch.tensor([float(rank + 1), float(rank + 11)])
gathered = [torch.zeros_like(src) for _ in range(world_size)]
w = dist.all_gather(gathered, src, async_op=True)
w.wait()
assert gathered[0].numpy().tolist() == [1.0, 11.0]
assert gathered[1].numpy().tolist() == [2.0, 12.0]

# 2) all_gather_into_tensor(async)
output = torch.zeros(src.numel() * world_size)
w = dist.all_gather_into_tensor(output, src, async_op=True)
w.wait()
assert output.numpy().tolist() == [1.0, 11.0, 2.0, 12.0]

# 3) reduce_scatter(async)
# rank0: [1, 2, 3, 4], rank1: [10, 20, 30, 40]
flat_in = torch.tensor([1.0, 2.0, 3.0, 4.0]) if rank == 0 else torch.tensor([10.0, 20.0, 30.0, 40.0])
rs_out = torch.zeros(2)
w = dist.reduce_scatter_tensor(rs_out, flat_in, async_op=True)
w.wait()
if rank == 0:
    assert rs_out.numpy().tolist() == [11.0, 22.0]
else:
    assert rs_out.numpy().tolist() == [33.0, 44.0]

# 4) scatter(async)
recv = torch.zeros(2)
scatter_list = [torch.tensor([100.0, 101.0]), torch.tensor([200.0, 201.0])] if rank == 0 else None
w = dist.scatter(recv, scatter_list=scatter_list, src=0, async_op=True)
w.wait()
if rank == 0:
    assert recv.numpy().tolist() == [100.0, 101.0]
else:
    assert recv.numpy().tolist() == [200.0, 201.0]

# 5) gather(async)
send = torch.tensor([float(rank + 5), float(rank + 15)])
glist = [torch.zeros_like(send) for _ in range(world_size)] if rank == 0 else None
w = dist.gather(send, gather_list=glist, dst=0, async_op=True)
w.wait()
if rank == 0:
    assert glist[0].numpy().tolist() == [5.0, 15.0]
    assert glist[1].numpy().tolist() == [6.0, 16.0]

# 6) all_to_all_single(async)
a2a_in = torch.tensor([1.0, 2.0]) if rank == 0 else torch.tensor([10.0, 20.0])
a2a_out = torch.zeros(2)
w = dist.all_to_all_single(a2a_out, a2a_in, async_op=True)
w.wait()
if rank == 0:
    assert a2a_out.numpy().tolist() == [1.0, 10.0]
else:
    assert a2a_out.numpy().tolist() == [2.0, 20.0]

dist.destroy_process_group()
print(f"[rank {rank}] async collective checks passed")
'''


def test_gloo_async_collectives_wait_semantics_world2():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29544"
    env["WORLD_SIZE"] = "2"
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["MINDTORCH_V2_SRC"] = src_dir
    env["PYTHONPATH"] = src_dir + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_gloo_async_collectives_worker.py"
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

    out0, _ = p0.communicate(timeout=120)
    out1, _ = p1.communicate(timeout=120)

    if p0.returncode != 0 or p1.returncode != 0:
        print("=== RANK 0 ===")
        print(out0.decode())
        print("=== RANK 1 ===")
        print(out1.decode())
        raise AssertionError(
            f"async collective worker failed: rank0={p0.returncode}, rank1={p1.returncode}"
        )

