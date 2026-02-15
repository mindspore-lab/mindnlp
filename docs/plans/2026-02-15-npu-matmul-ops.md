# NPU Matmul (MindTorch v2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NPU `matmul` with full PyTorch semantics (1D/2D/batched+broadcast) using ACLNN ctypes, no MindSpore dependency.

**Architecture:** Register a new NPU kernel for `matmul`, implement an ACLNN wrapper, and add a performance-first path that tries direct ACLNN matmul before falling back to per-batch calls with broadcasted batch indices.

**Tech Stack:** Python, ctypes, ACL runtime (`acl`), ACLNN shared libraries.

---

### Task 1: Add failing NPU matmul tests

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_npu_matmul_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="npu")
    out = torch.matmul(a, b)
    assert out.device.type == "npu"
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_1d_2d():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor([1.0, 2.0, 3.0], device="npu")
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="npu")
    out = torch.matmul(a, b)
    assert out.shape == (2,)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))


def test_npu_matmul_batched_broadcast():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    a = torch.tensor(np.arange(2 * 1 * 2 * 3, dtype=np.float32).reshape(2, 1, 2, 3), device="npu")
    b = torch.tensor(np.arange(1 * 4 * 3 * 5, dtype=np.float32).reshape(1, 4, 3, 5), device="npu")
    out = torch.matmul(a, b)
    assert out.shape == (2, 4, 2, 5)
    assert np.allclose(out.to("cpu").numpy(), np.matmul(a.to("cpu").numpy(), b.to("cpu").numpy()))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul_2d -v`
Expected: FAIL (missing NPU matmul kernel).

**Step 3: Write minimal implementation**

_No production code yet (TDD)._

**Step 4: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul_2d -v`
Expected: FAIL.

**Step 5: Commit**

_No commit (tests failing)._ 

---

### Task 2: Add ACLNN matmul wrapper

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Write the failing test**

Reuse the failing test from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul_2d -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
def matmul(a_ptr, b_ptr, out_ptr, a_shape, a_stride, b_shape, b_stride, out_shape, out_stride, dtype, runtime, stream=None):
    global acl
    if acl is None:
        acl = ensure_acl()
    bindings = get_bindings()
    if not bindings.aclnn_matmul_get_workspace or not bindings.aclnn_matmul:
        raise RuntimeError("aclnnMatmul symbols not available")
    a_tensor, a_keep = _create_tensor(bindings, a_shape, a_stride, dtype, a_ptr)
    b_tensor, b_keep = _create_tensor(bindings, b_shape, b_stride, dtype, b_ptr)
    out_tensor, out_keep = _create_tensor(bindings, out_shape, out_stride, dtype, out_ptr)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64(0)
    workspace = None
    try:
        ret = bindings.aclnn_matmul_get_workspace(
            a_tensor,
            b_tensor,
            out_tensor,
            ctypes.c_int8(0),
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMatmulGetWorkspaceSize failed: {ret}")
        if workspace_size.value:
            workspace_ptr, ret = acl.rt.malloc(int(workspace_size.value), 0)
            if ret != 0:
                raise RuntimeError(f"acl.rt.malloc failed: {ret}")
            workspace = workspace_ptr
        ret = bindings.aclnn_matmul(
            ctypes.c_void_p(0 if workspace is None else int(workspace)),
            ctypes.c_uint64(workspace_size.value),
            executor,
            ctypes.c_void_p(int(runtime.stream if stream is None else stream)),
        )
        if ret != 0:
            raise RuntimeError(f"aclnnMatmul failed: {ret}")
        _maybe_sync(runtime)
    finally:
        _defer_executor(executor)
        bindings.acl_destroy_tensor(a_tensor)
        bindings.acl_destroy_tensor(b_tensor)
        bindings.acl_destroy_tensor(out_tensor)
        if workspace is not None:
            runtime.defer_free(workspace)
        _ = (a_keep, b_keep, out_keep)
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul_2d -v`
Expected: FAIL (NPU kernel still missing).

**Step 5: Commit**

_No commit (tests failing)._ 

---

### Task 3: Implement NPU matmul kernel with fallback

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Write the failing test**

Reuse failing tests from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul_2d -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

```python
import math


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def _numel(shape):
    size = 1
    for d in shape:
        size *= d
    return size


def _broadcast_shape(a_shape, b_shape):
    return np.broadcast_shapes(a_shape, b_shape)


def _matmul_out_shape(a_shape, b_shape):
    # mirror meta infer logic
    a_dim = len(a_shape)
    b_dim = len(b_shape)
    if a_dim == 1 and b_dim == 1:
        if a_shape[0] != b_shape[0]:
            raise ValueError("matmul shape mismatch")
        return ()
    if a_dim == 1:
        if b_dim < 2 or b_shape[-2] != a_shape[0]:
            raise ValueError("matmul shape mismatch")
        return b_shape[:-2] + (b_shape[-1],)
    if b_dim == 1:
        if a_shape[-1] != b_shape[0]:
            raise ValueError("matmul shape mismatch")
        return a_shape[:-2] + (a_shape[-2],)
    if a_shape[-1] != b_shape[-2]:
        raise ValueError("matmul shape mismatch")
    batch = _broadcast_shape(a_shape[:-2], b_shape[:-2])
    return batch + (a_shape[-2], b_shape[-1])


def _ensure_contiguous(a):
    return a if a.is_contiguous() else contiguous(a)


def _batch_offset(shape, stride, index):
    offset = 0
    for i, idx in enumerate(index):
        offset += idx * stride[i]
    return offset


def matmul(a, b):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu" or b.device.type != "npu":
        raise ValueError("NPU matmul expects NPU tensors")
    if a.dtype != b.dtype:
        raise ValueError("NPU matmul requires matching dtypes")

    a = _ensure_contiguous(a)
    b = _ensure_contiguous(b)
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)
    out_shape = _matmul_out_shape(a_shape, b_shape)

    # Normalize 1D cases to 2D
    squeeze_left = False
    squeeze_right = False
    if len(a_shape) == 1:
        a_shape = (1, a_shape[0])
        squeeze_left = True
    if len(b_shape) == 1:
        b_shape = (b_shape[0], 1)
        squeeze_right = True

    out_shape_2d = out_shape
    if squeeze_left:
        out_shape_2d = out_shape_2d if out_shape_2d else (1,)
    if squeeze_right:
        out_shape_2d = out_shape_2d if out_shape_2d else (1,)

    out_stride = _contiguous_stride(out_shape_2d)
    out_size = int(_numel(out_shape_2d) if out_shape_2d else 1) * np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)

    a_storage = _unwrap_storage(a)
    b_storage = _unwrap_storage(b)

    # Try direct ACLNN matmul first
    try:
        aclnn.matmul(
            a_storage.data_ptr(),
            b_storage.data_ptr(),
            out_ptr,
            a_shape,
            a.stride,
            b_shape,
            b.stride,
            out_shape_2d,
            out_stride,
            a.dtype,
            runtime,
            stream=stream.stream,
        )
    except RuntimeError:
        # Fallback: batched matmul
        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        batch_shape = _broadcast_shape(a_batch, b_batch)
        if not batch_shape:
            raise
        a_stride = a.stride
        b_stride = b.stride
        out_batch_stride = out_stride[:-2]
        a_item = np.dtype(npu_runtime._dtype_to_numpy(a.dtype)).itemsize

        def iter_indices(shape):
            if not shape:
                yield ()
                return
            total = int(np.prod(shape))
            for flat in range(total):
                idx = []
                rem = flat
                for dim in reversed(shape):
                    idx.append(rem % dim)
                    rem //= dim
                yield tuple(reversed(idx))

        for idx in iter_indices(batch_shape):
            a_idx = tuple(0 if dim == 1 else idx[i] for i, dim in enumerate(a_batch))
            b_idx = tuple(0 if dim == 1 else idx[i] for i, dim in enumerate(b_batch))
            out_idx = idx
            a_off = _batch_offset(a_batch, a_stride[:len(a_batch)], a_idx)
            b_off = _batch_offset(b_batch, b_stride[:len(b_batch)], b_idx)
            out_off = _batch_offset(batch_shape, out_batch_stride, out_idx)
            aclnn.matmul(
                a_storage.data_ptr() + int(a_off * a_item),
                b_storage.data_ptr() + int(b_off * a_item),
                out_ptr + int(out_off * a_item),
                a_shape[-2:],
                a_stride[-2:],
                b_shape[-2:],
                b_stride[-2:],
                out_shape_2d[-2:],
                out_stride[-2:],
                a.dtype,
                runtime,
                stream=stream.stream,
            )

    storage = npu_typed_storage_from_ptr(out_ptr, int(_numel(out_shape_2d) if out_shape_2d else 1), a.dtype, device=a.device)
    out = _wrap_tensor(storage, out_shape_2d, out_stride)
    if squeeze_left:
        out = view_backend.squeeze_first(out)
    if squeeze_right:
        out = view_backend.squeeze_last(out)
    return out
```

Also register the kernel in `src/mindtorch_v2/_backends/npu/__init__.py`:

```python
from .ops import add, mul, matmul, relu, sum_, add_, mul_, relu_, zero_, contiguous
...
registry.register("matmul", "npu", matmul, meta=meta_infer.infer_matmul)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_matmul_2d -v`
Expected: PASS (or skip if NPU missing).

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add npu matmul"
```

---

### Task 4: Update integration notes

**Files:**
- Modify: `docs/plans/npu-op-integration.md`

**Step 1: Write the failing test**

_Not applicable (docs only)._ 

**Step 2: Run test to verify it fails**

_Not applicable._

**Step 3: Write minimal implementation**

Add a brief note to the "Notes" section:

```markdown
- NPU matmul uses a performance-first path: try direct ACLNN matmul, then fallback to per-batch matmul with broadcasted batch indices if ACLNN rejects high-rank shapes.
- Fallback requires contiguous inputs; inputs are made contiguous before fallback to allow pointer offset math.
```

**Step 4: Run test to verify it passes**

_Not applicable._

**Step 5: Commit**

```bash
git add docs/plans/npu-op-integration.md
git commit -m "docs: note npu matmul fallback"
```
