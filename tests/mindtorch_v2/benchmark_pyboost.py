"""Benchmark PyBoost CPU backend performance."""
import time
import sys
sys.path.insert(0, '/Users/lvyufeng/Projects/mindnlp/src')

from mindtorch_v2 import Tensor
import numpy as np


def benchmark_matmul():
    """Benchmark matrix multiplication."""
    print("Matrix Multiplication Benchmark")
    print("-" * 40)
    sizes = [(64, 64), (256, 256), (512, 512)]

    for size in sizes:
        # Create random tensors
        a_np = np.random.randn(size[0], size[1]).astype(np.float32)
        b_np = np.random.randn(size[1], size[0]).astype(np.float32)
        a = Tensor(a_np)
        b = Tensor(b_np)

        # Warmup
        for _ in range(3):
            _ = a @ b

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        elapsed = time.perf_counter() - start

        print(f"  matmul {size}: {elapsed/10*1000:.2f} ms per op")


def benchmark_elementwise():
    """Benchmark elementwise operations."""
    print("\nElementwise Operations Benchmark (1024x1024)")
    print("-" * 40)

    a_np = np.random.randn(1024, 1024).astype(np.float32)
    b_np = np.random.randn(1024, 1024).astype(np.float32)
    a = Tensor(a_np)
    b = Tensor(b_np)

    from mindtorch_v2._dispatch import dispatch

    ops = [
        ("add", lambda: a + b),
        ("mul", lambda: a * b),
        ("exp", lambda: a.exp()),
        ("sin", lambda: dispatch("sin", a)),
        ("sigmoid", lambda: dispatch("sigmoid", a)),
        ("tanh", lambda: dispatch("tanh", a)),
    ]

    for name, op in ops:
        # Warmup
        for _ in range(3):
            _ = op()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = op()
        elapsed = time.perf_counter() - start

        print(f"  {name}: {elapsed/100*1000:.3f} ms per op")


def benchmark_reductions():
    """Benchmark reduction operations."""
    print("\nReduction Operations Benchmark (1024x1024)")
    print("-" * 40)

    a_np = np.random.randn(1024, 1024).astype(np.float32)
    a = Tensor(a_np)

    ops = [
        ("sum", lambda: a.sum()),
        ("mean", lambda: a.mean()),
        ("max", lambda: a.max()),
        ("min", lambda: a.min()),
    ]

    for name, op in ops:
        # Warmup
        for _ in range(3):
            _ = op()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = op()
        elapsed = time.perf_counter() - start

        print(f"  {name}: {elapsed/100*1000:.3f} ms per op")


def benchmark_nn_ops():
    """Benchmark neural network operations."""
    print("\nNeural Network Operations Benchmark")
    print("-" * 40)

    # Embedding lookup
    weight = Tensor(np.random.randn(10000, 256).astype(np.float32))
    indices = Tensor(np.random.randint(0, 10000, (32, 128)).astype(np.int64))

    # Warmup
    from mindtorch_v2._dispatch import dispatch
    for _ in range(3):
        _ = dispatch("embedding", indices, weight)

    # Benchmark embedding
    start = time.perf_counter()
    for _ in range(100):
        _ = dispatch("embedding", indices, weight)
    elapsed = time.perf_counter() - start
    print(f"  embedding (10000x256, batch 32x128): {elapsed/100*1000:.3f} ms per op")

    # Softmax
    x = Tensor(np.random.randn(32, 128, 512).astype(np.float32))
    for _ in range(3):
        _ = dispatch("softmax", x, dim=-1)

    start = time.perf_counter()
    for _ in range(50):
        _ = dispatch("softmax", x, dim=-1)
    elapsed = time.perf_counter() - start
    print(f"  softmax (32x128x512): {elapsed/50*1000:.3f} ms per op")

    # Layer norm
    x = Tensor(np.random.randn(32, 128, 512).astype(np.float32))
    weight = Tensor(np.ones(512).astype(np.float32))
    bias = Tensor(np.zeros(512).astype(np.float32))
    for _ in range(3):
        _ = dispatch("layer_norm", x, (512,), weight, bias)

    start = time.perf_counter()
    for _ in range(50):
        _ = dispatch("layer_norm", x, (512,), weight, bias)
    elapsed = time.perf_counter() - start
    print(f"  layer_norm (32x128x512): {elapsed/50*1000:.3f} ms per op")


if __name__ == "__main__":
    print("=" * 50)
    print("PyBoost CPU Backend Benchmark")
    print("=" * 50)
    benchmark_matmul()
    benchmark_elementwise()
    benchmark_reductions()
    benchmark_nn_ops()
    print("\n" + "=" * 50)
    print("Benchmark complete!")
