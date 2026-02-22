"""Gloo backend verification: collectives + DDP on CPU with 2 processes."""

import os
import sys
import subprocess

SCRIPT = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# 1. init_process_group with gloo
dist.init_process_group(backend="gloo")
print(f"[rank {rank}] init_process_group OK, backend={dist.get_backend()}")

# 2. all_reduce with SUM
t = torch.tensor([float(rank + 1), float(rank + 2)])
dist.all_reduce(t, op=dist.ReduceOp.SUM)
# rank0: [1,2] + [2,3] = [3,5]; rank1: same
expected_sum = sum(range(1, world_size + 1))
assert float(t[0]) == expected_sum, f"all_reduce failed: {t}"
print(f"[rank {rank}] all_reduce SUM OK: {t}")

# 3. broadcast from rank 0
t2 = torch.tensor([42.0, 43.0]) if rank == 0 else torch.tensor([0.0, 0.0])
dist.broadcast(t2, src=0)
assert float(t2[0]) == 42.0 and float(t2[1]) == 43.0, f"broadcast failed: {t2}"
print(f"[rank {rank}] broadcast OK: {t2}")

# 4. barrier
dist.barrier()
print(f"[rank {rank}] barrier OK")

# 5. DDP on CPU
model = nn.Linear(4, 2)
ddp_model = nn.parallel.DistributedDataParallel(model)

# Forward + backward
x = torch.tensor(np.random.randn(3, 4).astype(np.float32))
loss = ddp_model(x).sum()
loss.backward()

# After backward, gradients should be synchronized across ranks
# DDP already averages gradients. Verify by broadcasting rank 0's grad
# and checking that all ranks have the same values.
grad_w = model.weight.grad._numpy_view().copy()
grad_from_0 = model.weight.grad.clone()
dist.broadcast(grad_from_0, src=0)
grad_from_0_np = grad_from_0._numpy_view()
diff = np.abs(grad_w - grad_from_0_np).sum()
assert diff < 1e-5, f"DDP grads not synced: diff={diff}"
print(f"[rank {rank}] DDP gradient sync OK (diff={diff})")

# 6. all_to_all
# rank 0 sends [10,20] to rank 0 and [30,40] to rank 1
# rank 1 sends [50,60] to rank 0 and [70,80] to rank 1
if rank == 0:
    inp = [torch.tensor([10.0, 20.0]), torch.tensor([30.0, 40.0])]
else:
    inp = [torch.tensor([50.0, 60.0]), torch.tensor([70.0, 80.0])]
out = [torch.zeros(2) for _ in range(world_size)]
dist.all_to_all(out, inp)
if rank == 0:
    assert list(out[0]._numpy_view()) == [10.0, 20.0], f"all_to_all r0[0] fail: {out[0]}"
    assert list(out[1]._numpy_view()) == [50.0, 60.0], f"all_to_all r0[1] fail: {out[1]}"
else:
    assert list(out[0]._numpy_view()) == [30.0, 40.0], f"all_to_all r1[0] fail: {out[0]}"
    assert list(out[1]._numpy_view()) == [70.0, 80.0], f"all_to_all r1[1] fail: {out[1]}"
print(f"[rank {rank}] all_to_all OK: {[o._numpy_view().tolist() for o in out]}")

# 7. all_to_all_single with equal split
# rank 0: [0,1,2,3] -> rank 0 gets [0,1] + [4,5]; rank 1 gets [2,3] + [6,7]
if rank == 0:
    inp_single = torch.tensor([0.0, 1.0, 2.0, 3.0])
else:
    inp_single = torch.tensor([4.0, 5.0, 6.0, 7.0])
out_single = torch.zeros(4)
dist.all_to_all_single(out_single, inp_single)
if rank == 0:
    assert list(out_single._numpy_view()) == [0.0, 1.0, 4.0, 5.0], f"a2a_single r0 fail: {out_single}"
else:
    assert list(out_single._numpy_view()) == [2.0, 3.0, 6.0, 7.0], f"a2a_single r1 fail: {out_single}"
print(f"[rank {rank}] all_to_all_single OK: {out_single._numpy_view().tolist()}")

dist.destroy_process_group()
print(f"[rank {rank}] ALL TESTS PASSED")
'''


def main():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29530"
    env["WORLD_SIZE"] = "2"
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src") + \
        ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    # Write the worker script to a temp file
    worker_file = "/tmp/_gloo_test_worker.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    # Launch rank 0 in background
    env0 = {**env, "RANK": "0"}
    p0 = subprocess.Popen(
        [sys.executable, worker_file],
        env=env0, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Launch rank 1 in foreground
    env1 = {**env, "RANK": "1"}
    p1 = subprocess.Popen(
        [sys.executable, worker_file],
        env=env1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    # Wait for both
    out0, _ = p0.communicate(timeout=120)
    out1, _ = p1.communicate(timeout=120)

    print("=== RANK 0 ===")
    print(out0.decode())
    print("=== RANK 1 ===")
    print(out1.decode())

    if p0.returncode != 0 or p1.returncode != 0:
        print(f"FAILED: rank0={p0.returncode}, rank1={p1.returncode}")
        sys.exit(1)
    else:
        print("ALL RANKS PASSED")


if __name__ == "__main__":
    main()
