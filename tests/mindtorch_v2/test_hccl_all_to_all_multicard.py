"""HCCL all_to_all verification on NPU with multiple cards."""

import os
import sys
import subprocess

SCRIPT = r'''
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import mindtorch_v2 as torch
import mindtorch_v2.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# 1. init_process_group with hccl, assign each rank its own device
device = torch.Device(f"npu:{rank}")
dist.init_process_group(backend="hccl", device_id=device)
print(f"[rank {rank}] init_process_group OK, backend={dist.get_backend()}, device={device}")

# 2. all_to_all - each rank sends unique data to every other rank
# rank i sends [i*100+j*10, i*100+j*10+1] to rank j
inp = []
for j in range(world_size):
    data = [float(rank * 100 + j * 10), float(rank * 100 + j * 10 + 1)]
    inp.append(torch.tensor(data, device=device))

out = [torch.zeros(2, device=device) for _ in range(world_size)]
dist.all_to_all(out, inp)

# Verify: rank i should receive [j*100+i*10, j*100+i*10+1] from rank j
out_cpu = [o.to("cpu") for o in out]
for j in range(world_size):
    expected = [float(j * 100 + rank * 10), float(j * 100 + rank * 10 + 1)]
    actual = list(out_cpu[j]._numpy_view())
    assert actual == expected, f"rank {rank} from rank {j}: expected {expected}, got {actual}"
print(f"[rank {rank}] all_to_all OK")

# 3. all_to_all_single with equal split
# rank i sends [i*world_size, i*world_size+1, ..., i*world_size+world_size-1]
# rank i should receive [0*world_size+i, 1*world_size+i, ..., (world_size-1)*world_size+i]
inp_single = torch.tensor([float(rank * world_size + j) for j in range(world_size)], device=device)
out_single = torch.zeros(world_size, device=device)
dist.all_to_all_single(out_single, inp_single)

out_single_cpu = out_single.to("cpu")
expected_single = [float(j * world_size + rank) for j in range(world_size)]
actual_single = list(out_single_cpu._numpy_view())
assert actual_single == expected_single, f"rank {rank} all_to_all_single: expected {expected_single}, got {actual_single}"
print(f"[rank {rank}] all_to_all_single OK")

dist.destroy_process_group()
print(f"[rank {rank}] ALL TESTS PASSED")
'''


def run_test(world_size):
    print(f"\n{'='*60}")
    print(f"Testing with {world_size} cards")
    print(f"{'='*60}\n")
    
    env = os.environ.copy()
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(29500 + world_size)  # Different port per test
    env["WORLD_SIZE"] = str(world_size)
    env["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "src") + \
        ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

    # Write the worker script to a temp file
    worker_file = f"/tmp/_hccl_all_to_all_test_{world_size}cards.py"
    with open(worker_file, "w") as f:
        f.write(SCRIPT)

    # Launch all ranks
    processes = []
    for rank in range(world_size):
        env_rank = {**env, "RANK": str(rank)}
        p = subprocess.Popen(
            [sys.executable, worker_file],
            env=env_rank, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        processes.append(p)

    # Wait for all
    outputs = []
    failed = False
    for rank, p in enumerate(processes):
        try:
            out, _ = p.communicate(timeout=180)
            outputs.append((rank, p.returncode, out.decode()))
            if p.returncode != 0:
                failed = True
        except subprocess.TimeoutExpired:
            p.kill()
            outputs.append((rank, -1, "TIMEOUT"))
            failed = True

    # Print outputs
    for rank, code, out in outputs:
        print(f"=== RANK {rank} (exit={code}) ===")
        print(out)

    if failed:
        print(f"\n❌ FAILED: {world_size} cards test")
        return False
    else:
        print(f"\n✅ PASSED: {world_size} cards test")
        return True


def main():
    results = {}
    
    # Test 4 cards
    results[4] = run_test(4)
    
    # Test 8 cards
    results[8] = run_test(8)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for cards, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{cards} cards: {status}")
    
    if all(results.values()):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
