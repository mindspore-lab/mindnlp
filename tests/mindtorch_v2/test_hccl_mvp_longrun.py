"""HCCL long-run and timeout-guard MVP tests."""

import os
import sys
import subprocess


SCRIPT_MULTI_STEP = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# deterministic init
seed = 2026
torch.manual_seed(seed)

device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

model = nn.Linear(4, 2).to(device)
ddp = nn.parallel.DistributedDataParallel(model)

for step in range(8):
    # Recreate input tensor each step to ensure a fresh autograd graph.
    x = torch.ones((3, 4), device=device).clone()
    loss = ddp(x).sum()
    loss.backward()

    # verify grad sync each step by comparing against rank0 broadcast
    w_ref = model.weight.grad.clone()
    b_ref = model.bias.grad.clone()
    dist.broadcast(w_ref, src=0)
    dist.broadcast(b_ref, src=0)

    wdiff = (model.weight.grad - w_ref).abs().sum().to("cpu").item()
    bdiff = (model.bias.grad - b_ref).abs().sum().to("cpu").item()
    assert wdiff < 1e-5, f"step={step} rank={rank} weight grad mismatch {wdiff}"
    assert bdiff < 1e-5, f"step={step} rank={rank} bias grad mismatch {bdiff}"

    # simple SGD update
    lr = 0.01
    with torch.no_grad():
        model.weight -= lr * model.weight.grad
        model.bias -= lr * model.bias.grad
        model.weight.grad = None
        model.bias.grad = None

# final parameter consistency check
w = model.weight.detach().clone()
b = model.bias.detach().clone()
dist.broadcast(w, src=0)
dist.broadcast(b, src=0)
assert (model.weight.detach() - w).abs().sum().to("cpu").item() < 1e-5
assert (model.bias.detach() - b).abs().sum().to("cpu").item() < 1e-5

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL multi-step PASS")
'''


SCRIPT_TIMEOUT_GUARD = r'''
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

# Guard scenario: rank1 exits early; rank0 should not hang forever in this script
if rank == 1:
    dist.destroy_process_group()
    print("rank1 early exit")
    sys.exit(0)

# rank0 intentionally avoids collective and exits quickly
time.sleep(1)
dist.destroy_process_group()
print("rank0 graceful exit without hang")
'''


def _run_two_rank_worker(script_text, worker_name, timeout_sec=240):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29651"
    env["WORLD_SIZE"] = "2"
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src") + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = f"/tmp/{worker_name}.py"
    with open(worker_file, "w") as f:
        f.write(script_text)

    procs = []
    for rank in range(2):
        env_rank = {**env, "RANK": str(rank)}
        p = subprocess.Popen([sys.executable, worker_file], env=env_rank, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(p)

    outputs = []
    failed = False
    for rank, p in enumerate(procs):
        try:
            out, _ = p.communicate(timeout=timeout_sec)
            text = out.decode()
            outputs.append((rank, p.returncode, text))
            if p.returncode != 0:
                failed = True
        except subprocess.TimeoutExpired:
            p.kill()
            outputs.append((rank, -1, "TIMEOUT"))
            failed = True

    for rank, code, text in outputs:
        print(f"=== rank {rank} exit={code} ===")
        print(text)

    return failed


def test_hccl_ddp_multi_step_consistency_2card():
    failed = _run_two_rank_worker(SCRIPT_MULTI_STEP, "_hccl_multi_step_worker", timeout_sec=300)
    assert not failed, "HCCL multi-step DDP consistency failed"


def test_hccl_rank_early_exit_no_parent_hang():
    failed = _run_two_rank_worker(SCRIPT_TIMEOUT_GUARD, "_hccl_timeout_guard_worker", timeout_sec=120)
    assert not failed, "HCCL timeout guard scenario failed"
