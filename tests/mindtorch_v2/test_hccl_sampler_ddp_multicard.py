"""HCCL 2-card integration smoke: DistributedSampler + DataLoader + DDP."""

import os
import sys
import subprocess

SCRIPT = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.nn as nn
import mindtorch_v2.distributed as dist
from mindtorch_v2.utils.data import Dataset, DataLoader, DistributedSampler


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return torch.tensor([float(idx)], dtype=torch.float32)


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

sampler = DistributedSampler(TinyDataset(), num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
loader = DataLoader(TinyDataset(), batch_size=2, sampler=sampler, num_workers=0)

# Verify deterministic rank partition
all_local = []
for batch in loader:
    all_local.extend([int(v) for v in batch.to("cpu").flatten().numpy().tolist()])

if rank == 0:
    assert all_local == [0, 2, 4, 6], f"rank0 partition mismatch: {all_local}"
else:
    assert all_local == [1, 3, 5, 7], f"rank1 partition mismatch: {all_local}"

# DDP one-step over loader batch
model = nn.Linear(1, 1).to(device)
ddp = nn.parallel.DistributedDataParallel(model)

first_batch = None
for batch in loader:
    first_batch = batch.to(device)
    break

loss = ddp(first_batch).sum()
loss.backward()

assert model.weight.grad is not None
assert model.bias.grad is not None

w_ref = model.weight.grad.clone()
b_ref = model.bias.grad.clone()
dist.broadcast(w_ref, src=0)
dist.broadcast(b_ref, src=0)

wdiff = (model.weight.grad - w_ref).abs().sum().to("cpu").item()
bdiff = (model.bias.grad - b_ref).abs().sum().to("cpu").item()
assert wdiff < 1e-5, f"rank={rank} weight grad mismatch {wdiff}"
assert bdiff < 1e-5, f"rank={rank} bias grad mismatch {bdiff}"

dist.barrier()
dist.destroy_process_group()
print(f"[rank {rank}] HCCL sampler+DDP PASS")
'''


def test_hccl_sampler_ddp_2card():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29641"
    env["WORLD_SIZE"] = "2"
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src") + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker_file = "/tmp/_hccl_sampler_ddp_2card_worker.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    procs = []
    for rank in range(2):
        env_rank = {**env, "RANK": str(rank)}
        p = subprocess.Popen([sys.executable, worker_file], env=env_rank, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(p)

    outputs = []
    failed = False
    for rank, p in enumerate(procs):
        out, _ = p.communicate(timeout=180)
        text = out.decode()
        outputs.append((rank, p.returncode, text))
        if p.returncode != 0:
            failed = True

    for rank, code, text in outputs:
        print(f"=== rank {rank} exit={code} ===")
        print(text)

    assert not failed, "HCCL sampler+DDP 2-card worker failed"
