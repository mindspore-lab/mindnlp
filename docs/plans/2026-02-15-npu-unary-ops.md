# NPU Unary Ops (MindTorch v2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NPU kernels for CPU-supported unary ops (float16 + float32) using direct ACLNN bindings, without NumPy in the NPU execution path.

**Architecture:** Extend ACLNN ctypes bindings for each op, implement NPU wrappers in `ops.py` via shared unary/binary helpers, and register kernels in the NPU backend. Tests live in `tests/mindtorch_v2/test_ops_npu.py` with NPU-only parameterized cases. No fallback to CPU or composed ops.

**Tech Stack:** Python, ctypes, ACL runtime (`acl`), ACLNN shared libraries.

---

### Task 1: Add failing NPU unary op tests

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
import pytest
import mindtorch_v2 as torch
import numpy as np


@pytest.mark.parametrize(
    "op_name, numpy_fn",
    [
        ("abs", np.abs),
        ("neg", np.negative),
        ("exp", np.exp),
        ("log", np.log),
        ("sqrt", np.sqrt),
        ("rsqrt", lambda x: 1.0 / np.sqrt(x)),
        ("sin", np.sin),
        ("cos", np.cos),
        ("tanh", np.tanh),
        ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("ceil", np.ceil),
        ("floor", np.floor),
        ("round", np.round),
        ("trunc", np.trunc),
        ("frac", lambda x: x - np.trunc(x)),
        ("log2", np.log2),
        ("log10", np.log10),
        ("exp2", np.exp2),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_unary_ops(op_name, numpy_fn, dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    data = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    x = torch.tensor(data, device="npu", dtype=dtype)
    op = getattr(torch, op_name)
    out = op(x)
    expected = numpy_fn(data).astype(np.float32)
    assert out.device.type == "npu"
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected, atol=1e-3, rtol=1e-3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_unary_ops -v`
Expected: FAIL (missing NPU kernels).

**Step 3: Write minimal implementation**

_No production code yet (TDD)._

**Step 4: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_unary_ops -v`
Expected: FAIL.

**Step 5: Commit**

_No commit (tests failing)._ 

---

### Task 2: Bind ACLNN unary ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Write the failing test**

Reuse failing test from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_unary_ops -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add ACLNN bindings in `AclnnBindings.__init__` for each op:

```python
self.aclnn_abs_get_workspace = _optional_symbol(..., "aclnnAbsGetWorkspaceSize", ...)
self.aclnn_abs = _optional_symbol(..., "aclnnAbs", ...)
# repeat for neg/exp/log/sqrt/rsqrt/sin/cos/tanh/sigmoid/ceil/floor/round/trunc/frac/log2/log10/exp2
```

Then add wrappers matching existing patterns:

```python
def abs(self_ptr, out_ptr, shape, stride, dtype, runtime, stream=None):
    # mirror aclnn.add/relU pattern with GetWorkspaceSize + Execute + cleanup
```

**Step 4: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_unary_ops -v`
Expected: FAIL (kernel not registered yet).

**Step 5: Commit**

_No commit (tests failing)._ 

---

### Task 3: Implement NPU unary ops + register kernels

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Write the failing test**

Reuse failing test from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_unary_ops -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

Add helpers in `ops.py`:

```python
def _unary_op(a, fn):
    runtime = npu_runtime.get_runtime((a.device.index or 0))
    stream = npu_state.current_stream((a.device.index or 0))
    if a.device.type != "npu":
        raise ValueError("NPU unary op expects NPU tensors")
    out_size = _numel(a.shape) * _dtype_itemsize(a.dtype)
    out_ptr = npu_runtime._alloc_device(out_size, runtime=runtime)
    storage = _unwrap_storage(a)
    fn(storage.data_ptr(), out_ptr, a.shape, a.stride, a.dtype, runtime, stream=stream.stream)
    out_storage = npu_typed_storage_from_ptr(out_ptr, _numel(a.shape), a.dtype, device=a.device)
    return _wrap_tensor(out_storage, a.shape, a.stride)
```

Then define each op:

```python
def abs(a): return _unary_op(a, aclnn.abs)
# ... for neg/exp/log/sqrt/rsqrt/sin/cos/tanh/sigmoid/ceil/floor/round/trunc/frac/log2/log10/exp2
```

Register kernels in `npu/__init__.py` using meta infer:

```python
registry.register("abs", "npu", abs, meta=meta_infer.infer_unary)
# ... etc
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_unary_ops -v`
Expected: PASS (or skip if NPU missing).

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add npu unary ops"
```

---

### Task 4: Add NPU pow (binary) + tests

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Write the failing test**

```python
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_npu_pow(dtype):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([1.0, 2.0, 3.0], device="npu", dtype=dtype)
    exp = torch.tensor([2.0, 3.0, 0.5], device="npu", dtype=dtype)
    out = torch.pow(base, exp)
    expected = np.power(base.to("cpu").numpy(), exp.to("cpu").numpy())
    assert np.allclose(out.to("cpu").numpy().astype(np.float32), expected.astype(np.float32), atol=1e-3, rtol=1e-3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_pow -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

Bind ACLNN pow symbols and wrapper, then implement in `ops.py` via `_binary_op` helper:

```python
def _binary_op(a, b, fn):
    # validate device/dtype, allocate out, call fn(self_ptr, other_ptr, out_ptr, shape, stride, dtype, runtime, stream)
```

Register kernel:

```python
registry.register("pow", "npu", pow, meta=meta_infer.infer_binary)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_pow -v`
Expected: PASS (or skip if NPU missing).

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add npu pow"
```
