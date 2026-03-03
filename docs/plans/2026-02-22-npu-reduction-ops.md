# NPU Reduction Ops (MindTorch v2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NPU coverage for CPU-backed reduction ops (all/any/argmax/argmin/amin/amax/count_nonzero) with PyTorch-aligned behavior using ACLNN ctypes only.

**Architecture:** Use ACLNN ArgMax/ArgMin for index reductions and ACLNN MaxDim/MinDim for value reductions; implement all/any/count_nonzero by building a boolean nonzero mask via aclnnEqScalar + aclnnLogicalNot and reducing with aclnnReduceSum to int64, then compare with scalar totals.

**Tech Stack:** MindTorch v2 NPU backend, ACLNN ctypes (aclnn.py), existing NPU runtime allocator, pytest.

---

### Task 1: Add NPU tests for argmax/argmin/amin/amax

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing tests**

```python

def test_npu_amin_amax():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], device="npu")
    expected_min = np.amin(x.to("cpu").numpy(), axis=1)
    expected_max = np.amax(x.to("cpu").numpy(), axis=1)
    np.testing.assert_allclose(torch.amin(x, dim=1).to("cpu").numpy(), expected_min)
    np.testing.assert_allclose(torch.amax(x, dim=1).to("cpu").numpy(), expected_max)


def test_npu_argmax_argmin():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], device="npu")
    expected_max = np.argmax(x.to("cpu").numpy(), axis=1)
    expected_min = np.argmin(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.argmax(x, dim=1).to("cpu").numpy(), expected_max)
    np.testing.assert_array_equal(torch.argmin(x, dim=1).to("cpu").numpy(), expected_min)
    np.testing.assert_array_equal(
        torch.argmax(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_max.reshape(2, 1),
    )
    np.testing.assert_array_equal(
        torch.argmin(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_min.reshape(2, 1),
    )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_amin_amax -v`
Expected: FAIL (missing NPU implementations)

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_argmax_argmin -v`
Expected: FAIL (missing NPU implementations)

---

### Task 2: Add ACLNN bindings for argmax/argmin/max_dim/min_dim

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Add optional symbol bindings in `AclnnBindings.__init__`**

```python
self.aclnn_argmax_get_workspace = _optional_symbol(
    libs,
    "aclnnArgMaxGetWorkspaceSize",
    ctypes.c_int32,
    [ctypes.c_void_p, ctypes.c_int64, ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)],
)
self.aclnn_argmax = _optional_symbol(
    libs,
    "aclnnArgMax",
    ctypes.c_int32,
    [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
)
# Repeat for argmin/max_dim/min_dim with signatures from headers.
```

**Step 2: Implement `argmax`, `argmin`, `max_dim`, `min_dim` helpers**

```python

def argmax(self_ptr, out_ptr, shape, stride, dim, keepdim, out_shape, out_stride, runtime, stream=None):
    # create tensors, allocate workspace, invoke aclnnArgMaxGetWorkspaceSize + aclnnArgMax
```

**Step 3: Add `*_symbols_ok()` helpers for availability checks**

**Step 4: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py
git commit -m "feat(mindtorch_v2): add aclnn argmax/min bindings"
```

---

### Task 3: Implement NPU argmax/argmin/amin/amax + registrations

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Implement `argmax` / `argmin` in `ops.py`**

```python
from ..._dtype import int64 as int64_dtype

# Use aclnn.argmax/argmin
# For dim is None: flatten to 1D via view_backend.reshape and use dim=0.
# Allocate output with int64 dtype and contiguous stride.
```

**Step 2: Implement `amin` / `amax` using aclnn max_dim/min_dim**

```python
# Allocate value output (input dtype) and indices output (int64 dtype)
# Use view_backend.reshape for dim=None, then reshape to keepdim True if needed
```

**Step 3: Register ops**

```python
registry.register("amin", "npu", amin, meta=meta_infer.infer_sum)
registry.register("amax", "npu", amax, meta=meta_infer.infer_sum)
registry.register("argmax", "npu", argmax, meta=meta_infer.infer_argmax)
registry.register("argmin", "npu", argmin, meta=meta_infer.infer_argmax)
```

**Step 4: Run tests**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_amin_amax -v`
Expected: PASS

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_argmax_argmin -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py
git commit -m "feat(mindtorch_v2): add npu argmax/argmin/amin/amax"
```

---

### Task 4: Add NPU tests for all/any/count_nonzero

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing tests**

```python

def test_npu_all_any():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[True, False], [True, True]], device="npu", dtype=torch.bool)
    expected_all = np.all(x.to("cpu").numpy(), axis=1)
    expected_any = np.any(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.all(x, dim=1).to("cpu").numpy(), expected_all)
    np.testing.assert_array_equal(torch.any(x, dim=1).to("cpu").numpy(), expected_any)
    expected_keep = np.all(x.to("cpu").numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(torch.all(x, dim=1, keepdim=True).to("cpu").numpy(), expected_keep)


def test_npu_count_nonzero():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0]], device="npu")
    expected = np.count_nonzero(x.to("cpu").numpy(), axis=1)
    np.testing.assert_array_equal(torch.count_nonzero(x, dim=1).to("cpu").numpy(), expected)
    expected_keep = np.count_nonzero(x.to("cpu").numpy(), axis=1, keepdims=True)
    np.testing.assert_array_equal(
        torch.count_nonzero(x, dim=1, keepdim=True).to("cpu").numpy(),
        expected_keep,
    )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_all_any -v`
Expected: FAIL

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_count_nonzero -v`
Expected: FAIL

---

### Task 5: Implement NPU all/any/count_nonzero + registrations

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Add helper to compute reduction dims/out shape**

```python
# Extract shared logic from sum_ to compute dims, out_shape, out_stride
```

**Step 2: Implement `count_nonzero`**

```python
# mask = logical_not(eq_scalar(a, 0))
# out = reduce_sum(mask, dtype=int64)
```

**Step 3: Implement `all_` / `any_`**

```python
# count = reduce_sum(mask, dtype=int64)
# any = logical_not(eq_scalar(count, 0))
# all = eq_scalar(count, total_elements_per_reduction)
```

**Step 4: Register ops**

```python
registry.register("all", "npu", all_, meta=meta_infer.infer_reduce_bool)
registry.register("any", "npu", any_, meta=meta_infer.infer_reduce_bool)
registry.register("count_nonzero", "npu", count_nonzero, meta=meta_infer.infer_argmax)
```

**Step 5: Run tests**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_all_any -v`
Expected: PASS

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_count_nonzero -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py
git commit -m "feat(mindtorch_v2): add npu all/any/count_nonzero"
```

---

### Task 6: Full verification

**Files:**
- Test: `tests/mindtorch_v2`

**Step 1: Run full MindTorch v2 tests**

Run: `pytest tests/mindtorch_v2 -v`
Expected: PASS

**Step 2: If new special cases discovered, update doc**

- Modify: `docs/plans/2026-02-15-npu-op-integration-update.md`

