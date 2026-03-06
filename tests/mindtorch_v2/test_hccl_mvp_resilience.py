"""HCCL resilience tests for MVP distributed baseline."""

import os
import sys
import socket
import subprocess
from collections import Counter

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist
from mindtorch_v2.utils.data import Dataset, DistributedSampler


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _set_env(port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)


def _cleanup_pg():
    if dist.is_initialized():
        dist.destroy_process_group()


class RangeDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def test_hccl_reinit_after_destroy_same_process():
    _cleanup_pg()

    # First init/destroy
    p1 = _free_port()
    _set_env(p1)
    dist.init_process_group("hccl", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{p1}", device_id=torch.Device("npu:0"))
    x = torch.tensor([1.0], device="npu:0")
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    assert float(x.to("cpu").item()) == 1.0
    dist.destroy_process_group()
    assert dist.is_initialized() is False

    # Second init/destroy with another port should still work
    p2 = _free_port()
    _set_env(p2)
    dist.init_process_group("hccl", rank=0, world_size=1, init_method=f"tcp://127.0.0.1:{p2}", device_id=torch.Device("npu:0"))
    y = torch.tensor([2.0], device="npu:0")
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    assert float(y.to("cpu").item()) == 2.0
    dist.destroy_process_group()
    assert dist.is_initialized() is False


def test_distributed_sampler_rank_partition_2way():
    ds = RangeDataset(11)
    s0 = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=False, drop_last=False)
    s1 = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=False, drop_last=False)

    i0 = list(iter(s0))
    i1 = list(iter(s1))

    # same number of samples per rank
    assert len(i0) == len(i1) == len(s0) == len(s1)
    # union should cover full dataset; extra entry is one padded duplicate.
    merged = i0 + i1
    counts = Counter(merged)
    for idx in range(11):
        assert counts[idx] >= 1
    assert sum(counts.values()) == 12
    assert sum(1 for v in counts.values() if v > 1) == 1


SCRIPT_FAIL_FAST = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

device = torch.Device(f"npu:{rank}")
dist.init_process_group("hccl", device_id=device)

if rank == 1:
    # Simulate early rank failure before collective
    print("rank1 intentional failure")
    sys.exit(3)

# rank0 does not enter collective to avoid deadlock; just barrier skipped.
dist.destroy_process_group()
print("rank0 clean exit")
'''


def test_hccl_rank_failure_does_not_hang_parent():
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29631"
    env["WORLD_SIZE"] = "2"
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src") + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    worker = "/tmp/_hccl_rank_fail_fast.py"
    with open(worker, "w") as f:
        f.write(SCRIPT_FAIL_FAST)

    procs = []
    for rank in range(2):
        env_rank = {**env, "RANK": str(rank)}
        p = subprocess.Popen([sys.executable, worker], env=env_rank, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append(p)

    outs = []
    for p in procs:
        out, _ = p.communicate(timeout=120)
        outs.append((p.returncode, out.decode()))

    for idx, (code, text) in enumerate(outs):
        print(f"=== rank {idx} exit={code} ===")
        print(text)

    # one rank should fail, one should exit cleanly; parent test should complete quickly
    codes = [c for c, _ in outs]
    assert any(c == 0 for c in codes)
    assert any(c != 0 for c in codes)
