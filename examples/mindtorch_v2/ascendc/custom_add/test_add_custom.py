"""End-to-end integration test for AscendC custom operator (AddCustom).

Validates the full pipeline:
    AscendC C++ kernel  ->  ascendc_library() cmake  ->  libadd_custom_kernels.so
        ->  KernelLauncher (aclrtlaunch_add_custom)
        ->  @ascendc_op  ->  dispatch  ->  register_autograd  ->  backward

Prerequisites:
    1. Build the kernel launch library:
       cd examples/mindtorch_v2/ascendc/custom_add && bash build.sh
    2. Run:
       PYTHONPATH=src python examples/mindtorch_v2/ascendc/custom_add/test_add_custom.py

Hardware: Ascend 910B (NPU)
"""
import os
import sys
import struct
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_src_path = os.path.join(_project_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import mindtorch_v2 as torch
from mindtorch_v2._tensor import Tensor
from mindtorch_v2._backends.npu.custom_kernel import (
    KernelLauncher,
    tensor_ptr,
    alloc_like,
    copy_h2d,
    ascendc_op,
)
from mindtorch_v2._backends.npu import runtime as npu_runtime

# ---------------------------------------------------------------------------
# Locate and load the compiled kernel launch library
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_SO = os.path.join(
    _SCRIPT_DIR, "kernel_launch", "build", "lib", "libadd_custom_kernels.so"
)

if not os.path.isfile(_KERNEL_SO):
    print(f"ERROR: {_KERNEL_SO} not found.")
    print("Build it first:")
    print("  cd examples/mindtorch_v2/ascendc/custom_add && bash build.sh")
    sys.exit(1)

print(f"Loading kernel library: {_KERNEL_SO}")
launcher = KernelLauncher(_KERNEL_SO)


def _to_numpy(t):
    """Copy NPU tensor to numpy array."""
    npu_runtime.get_runtime(0).synchronize()
    return t.to("cpu").numpy()


# ---------------------------------------------------------------------------
# Define the custom operator via @ascendc_op
# ---------------------------------------------------------------------------

@ascendc_op("example::add_custom")
def add_custom(x: Tensor, y: Tensor) -> Tensor:
    """Element-wise addition using the AscendC AddCustom kernel."""
    out = alloc_like(x)
    n = x.numel()

    # Pack tiling data: totalLength as uint32_t
    tiling_bytes = struct.pack("<I", n)
    tiling_ptr = copy_h2d(tiling_bytes)

    # block_dim: number of AI cores to use
    # For the kernel, totalLength must be divisible by (block_dim * TILE_LENGTH)
    # TILE_LENGTH=128, so we pick block_dim such that n / block_dim / 128 >= 1
    block_dim = min(8, n // 128)
    if block_dim == 0:
        block_dim = 1

    launcher.launch(
        "add_custom",
        block_dim=block_dim,
        args=[tensor_ptr(x), tensor_ptr(y), tensor_ptr(out), 0, tiling_ptr],
    )
    return out


# ---------------------------------------------------------------------------
# Test 1: Forward pass correctness
# ---------------------------------------------------------------------------
def test_forward():
    print("\n=== Test 1: Forward pass (1024 elements) ===")
    N = 1024
    x = torch.randn(N, dtype=torch.float16, device="npu")
    y = torch.randn(N, dtype=torch.float16, device="npu")

    z = add_custom(x, y)
    npu_runtime.get_runtime(0).synchronize()

    x_np = _to_numpy(x)
    y_np = _to_numpy(y)
    z_np = _to_numpy(z)

    expected = (x_np.astype(np.float32) + y_np.astype(np.float32)).astype(np.float16)
    diff = np.abs(z_np.astype(np.float32) - expected.astype(np.float32)).max()

    print(f"  Shape: {z.shape}")
    print(f"  Max diff vs numpy add: {diff}")
    assert diff < 1e-3, f"Forward pass mismatch: max diff = {diff}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 2: Larger tensor
# ---------------------------------------------------------------------------
def test_larger():
    print("\n=== Test 2: Larger tensor (8192 elements) ===")
    N = 8192
    x = torch.randn(N, dtype=torch.float16, device="npu")
    y = torch.randn(N, dtype=torch.float16, device="npu")

    z = add_custom(x, y)
    npu_runtime.get_runtime(0).synchronize()

    x_np = _to_numpy(x)
    y_np = _to_numpy(y)
    z_np = _to_numpy(z)

    expected = (x_np.astype(np.float32) + y_np.astype(np.float32)).astype(np.float16)
    diff = np.abs(z_np.astype(np.float32) - expected.astype(np.float32)).max()

    print(f"  Max diff: {diff}")
    assert diff < 1e-3, f"Larger tensor mismatch: max diff = {diff}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Test 3: Backward pass via register_autograd
# ---------------------------------------------------------------------------
def test_backward():
    print("\n=== Test 3: Backward pass (register_autograd) ===")

    def setup(ctx, inputs, output):
        pass  # add has trivial gradient, no context needed

    def bwd(ctx, grad_output):
        # d(x + y)/dx = 1, d(x + y)/dy = 1
        return (grad_output, grad_output)

    add_custom.register_autograd(bwd, setup_context=setup)

    N = 1024
    x = torch.randn(N, dtype=torch.float16, device="npu")
    y = torch.randn(N, dtype=torch.float16, device="npu")
    x.requires_grad = True
    y.requires_grad = True

    z = add_custom(x, y)
    loss = z.sum()
    loss.backward()

    assert x.grad is not None, "x.grad is None — backward did not flow"
    assert y.grad is not None, "y.grad is None — backward did not flow"

    npu_runtime.get_runtime(0).synchronize()

    # Gradients should be all-ones (d(sum(x+y))/dx = 1)
    x_grad_np = _to_numpy(x.grad)
    y_grad_np = _to_numpy(y.grad)

    x_grad_diff = np.abs(x_grad_np - 1.0).max()
    y_grad_diff = np.abs(y_grad_np - 1.0).max()
    print(f"  x.grad max diff from 1.0: {x_grad_diff}")
    print(f"  y.grad max diff from 1.0: {y_grad_diff}")
    assert x_grad_diff < 1e-3, f"x.grad incorrect: max diff = {x_grad_diff}"
    assert y_grad_diff < 1e-3, f"y.grad incorrect: max diff = {y_grad_diff}"
    print("  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("AscendC AddCustom E2E Integration Test (Kernel Launch)")
    print("=" * 50)

    test_forward()
    test_larger()
    test_backward()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
