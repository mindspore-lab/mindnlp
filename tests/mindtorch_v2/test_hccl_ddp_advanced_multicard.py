"""HCCL 2-card advanced DDP coverage for MVP hardening.

Covers key training modes beyond basic smoke cases:
- `find_unused_parameters=True` zero-gradient materialization for unused params
- `static_graph=True` + `gradient_as_bucket_view=True` cache stability
- `no_sync()` accumulation followed by synchronized backward
- multi-step custom comm hook stability
"""

import os
import subprocess
import sys


SCRIPT_UNUSED = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)


class ModelUnused(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x, use_fc2=False):
        x = self.fc1(x)
        if use_fc2:
            x = self.fc2(x)
        return self.fc3(x)


model = ModelUnused().to(device)
ddp = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

x = torch.ones((3, 4), device=device)
loss = ddp(x, use_fc2=False).sum()
loss.backward()

assert model.fc2.weight.grad is not None
assert model.fc2.bias.grad is not None
assert model.fc2.weight.grad.abs().sum().to("cpu").item() < 1e-6
assert model.fc2.bias.grad.abs().sum().to("cpu").item() < 1e-6

# Used parameter gradients should still be synchronized across ranks.
w_ref = model.fc1.weight.grad.clone()
dist.broadcast(w_ref, src=0)
wdiff = (model.fc1.weight.grad - w_ref).abs().sum().to("cpu").item()
assert wdiff < 1e-5, f"rank={rank} used grad mismatch {wdiff}"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL unused-params PASS")
'''


SCRIPT_STATIC_BUCKET = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)


class ModelStatic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.unused = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


model = ModelStatic().to(device)
ddp = nn.parallel.DistributedDataParallel(
    model,
    static_graph=True,
    gradient_as_bucket_view=True,
)

cache_ids = []
for _ in range(3):
    x = torch.ones((3, 4), device=device)
    loss = ddp(x).sum()
    loss.backward()

    cache = ddp.reducer._cached_unused_param_indices
    assert cache is not None and len(cache) > 0
    cache_ids.append(id(cache))

    # Unused param should be materialized as zero gradient tensor.
    assert model.unused.weight.grad is not None
    assert model.unused.weight.grad.abs().sum().to("cpu").item() < 1e-6

    with torch.no_grad():
        for p in model.parameters():
            p.grad = None

assert cache_ids[0] == cache_ids[1] == cache_ids[2]

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL static+bucket PASS")
'''


SCRIPT_NOSYNC = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

model = nn.Linear(4, 2).to(device)
ddp = nn.parallel.DistributedDataParallel(model)

x = torch.ones((3, 4), device=device)
with ddp.no_sync():
    ddp(x).sum().backward()
    ddp(x).sum().backward()

# This step should trigger synchronization.
ddp(x).sum().backward()

w_ref = model.weight.grad.clone()
dist.broadcast(w_ref, src=0)
wdiff = (model.weight.grad - w_ref).abs().sum().to("cpu").item()
assert wdiff < 1e-5, f"rank={rank} no_sync grad mismatch {wdiff}"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL no_sync PASS")
'''


SCRIPT_COMM_HOOK = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist
import mindtorch_v2.futures as futures

rank = int(os.environ["RANK"])
device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)
world_size = dist.get_world_size()


class HookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones((8,)))

    def forward(self, x):
        return x * self.w


def custom_allreduce_hook(state, bucket):
    buf = bucket.buffer()
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    from mindtorch_v2._functional import mul

    fut = futures.Future()
    fut.set_result(mul(buf, 1.0 / state["world_size"]))
    return fut


model = HookModel().to(device)
ddp = nn.parallel.DistributedDataParallel(model)
ddp.register_comm_hook({"world_size": world_size}, custom_allreduce_hook)

for step in range(3):
    x = torch.ones((4, 8), device=device)
    loss = ddp(x).sum()
    loss.backward()

    ref = model.w.grad.clone()
    dist.broadcast(ref, src=0)
    diff = (model.w.grad - ref).abs().sum().to("cpu").item()
    assert diff < 1e-5, f"step={step} rank={rank} hook grad mismatch {diff}"

    with torch.no_grad():
        model.w.grad = None

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL comm_hook PASS")
'''


def _run_two_rank_worker(script_text, worker_name, master_port, timeout_sec=240):
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(master_port)
    env["WORLD_SIZE"] = "2"

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
    env["PYTHONPATH"] = src_root + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

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


def test_hccl_ddp_find_unused_parameters_2card():
    failed = _run_two_rank_worker(SCRIPT_UNUSED, "_hccl_ddp_unused_worker", master_port=29691)
    assert not failed, "HCCL DDP find_unused_parameters failed"


def test_hccl_ddp_static_graph_bucket_view_2card():
    failed = _run_two_rank_worker(SCRIPT_STATIC_BUCKET, "_hccl_ddp_static_bucket_worker", master_port=29692)
    assert not failed, "HCCL DDP static_graph+bucket_view failed"


def test_hccl_ddp_no_sync_2card():
    failed = _run_two_rank_worker(SCRIPT_NOSYNC, "_hccl_ddp_nosync_worker", master_port=29693)
    assert not failed, "HCCL DDP no_sync failed"


def test_hccl_ddp_comm_hook_multistep_2card():
    failed = _run_two_rank_worker(SCRIPT_COMM_HOOK, "_hccl_ddp_comm_hook_worker", master_port=29694)
    assert not failed, "HCCL DDP comm_hook multistep failed"
