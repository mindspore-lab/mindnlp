# NPU Index/Select Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NPU support for `gather`, `index_select`, `take`, `take_along_dim`, and `masked_select` with Torch-aligned semantics and tests.

**Architecture:** Implement ACLNN bindings for gather/masked_select in `aclnn.py`, then build NPU ops in `ops.py` that validate shapes, normalize indices, and call ACLNN. Register ops in `npu/__init__.py` and add NPU tests mirroring CPU behavior.

**Tech Stack:** Python, MindTorch v2 dispatch, ACLNN via ctypes, pytest.

---

### Task 1: Add NPU tests for gather/index_select/take/take_along_dim/masked_select

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**
```python
def test_npu_take():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    index = torch.tensor([0, 3, 1], dtype=torch.int64, device="npu")
    expected = np.take(x.to("cpu").numpy().reshape(-1), index.to("cpu").numpy().astype(np.int64))
    np.testing.assert_allclose(torch.take(x, index).to("cpu").numpy(), expected)
    neg_index = torch.tensor([-1, 0], dtype=torch.int64, device="npu")
    expected_neg = np.take(x.to("cpu").numpy().reshape(-1), neg_index.to("cpu").numpy().astype(np.int64))
    np.testing.assert_allclose(torch.take(x, neg_index).to("cpu").numpy(), expected_neg)
    out_of_range = torch.tensor([4], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.take(x, out_of_range)
```

**Step 2: Run test to verify it fails**
Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_take -vv`
Expected: FAIL (op not implemented for NPU).

**Step 3: Write minimal implementation**
Continue with NPU tests in the same file for:
```python
def test_npu_take_along_dim():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    indices = torch.tensor([[0, 2, 1], [2, 0, 1]], dtype=torch.int64, device="npu")
    expected = np.take_along_axis(x.to("cpu").numpy(), indices.to("cpu").numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.take_along_dim(x, indices, dim=1).to("cpu").numpy(), expected)
    neg_indices = torch.tensor([[-1, 0, 1], [1, -2, 0]], dtype=torch.int64, device="npu")
    expected_neg = np.take_along_axis(x.to("cpu").numpy(), neg_indices.to("cpu").numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.take_along_dim(x, neg_indices, dim=1).to("cpu").numpy(), expected_neg)
    out_of_range = torch.tensor([[3, 0, 1], [1, 2, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.take_along_dim(x, out_of_range, dim=1)
```
```python
def test_npu_index_select():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    index = torch.tensor([2, 0], dtype=torch.int64, device="npu")
    expected = np.take(x.to("cpu").numpy(), index.to("cpu").numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.index_select(x, dim=1, index=index).to("cpu").numpy(), expected)
    neg_index = torch.tensor([-1, 0], dtype=torch.int64, device="npu")
    expected_neg = np.take(x.to("cpu").numpy(), neg_index.to("cpu").numpy().astype(np.int64), axis=1)
    np.testing.assert_allclose(torch.index_select(x, dim=1, index=neg_index).to("cpu").numpy(), expected_neg)
    out_of_range = torch.tensor([3], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.index_select(x, dim=1, index=out_of_range)
    bad_index = torch.tensor([[0, 1]], dtype=torch.int64, device="npu")
    with pytest.raises(ValueError):
        torch.index_select(x, dim=1, index=bad_index)
```
```python
def test_npu_gather():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="npu")
    index = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device="npu")
    expected = np.take_along_axis(x.to("cpu").numpy(), index.to("cpu").numpy(), axis=1)
    np.testing.assert_allclose(torch.gather(x, dim=1, index=index).to("cpu").numpy(), expected)
    neg_index = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=neg_index)
    out_of_range = torch.tensor([[3, 0], [1, 0]], dtype=torch.int64, device="npu")
    with pytest.raises(IndexError):
        torch.gather(x, dim=1, index=out_of_range)
```
```python
def test_npu_masked_select():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1, 2], [3, 4]], device="npu")
    mask = torch.tensor([[True, False], [False, True]], device="npu")
    out = torch.masked_select(x, mask)
    np.testing.assert_array_equal(out.to("cpu").numpy(), np.array([1, 4]))
```

**Step 4: Run test to verify it fails**
Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_take -vv`
Expected: FAIL until NPU ops are implemented.

**Step 5: Commit**
```bash
git add tests/mindtorch_v2/test_ops_npu.py
git commit -m "test(mindtorch_v2): add NPU index/select op tests"
```

---

### Task 2: Add ACLNN bindings for gather + masked_select

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Write the failing test**
Use the tests from Task 1; they should still fail at this stage.

**Step 2: Implement minimal bindings**
Add in `AclnnBindings.__init__`:
```python
self.aclnn_gather_get_workspace = _optional_symbol(
    libs,
    "aclnnGatherGetWorkspaceSize",
    ctypes.c_int32,
    [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
)
self.aclnn_gather = _optional_symbol(
    libs,
    "aclnnGather",
    ctypes.c_int32,
    [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
)
self.aclnn_masked_select_get_workspace = _optional_symbol(
    libs,
    "aclnnMaskedSelectGetWorkspaceSize",
    ctypes.c_int32,
    [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ],
)
self.aclnn_masked_select = _optional_symbol(
    libs,
    "aclnnMaskedSelect",
    ctypes.c_int32,
    [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
)
```
Add helpers:
```python
def gather(self_ptr, index_ptr, out_ptr,
           self_shape, self_stride, self_dtype,
           index_shape, index_stride, index_dtype,
           out_shape, out_stride, out_dtype,
           dim, runtime, stream=None):
    ...
```
```python
def masked_select(self_ptr, mask_ptr, out_ptr,
                  self_shape, self_stride, self_dtype,
                  mask_shape, mask_stride, mask_dtype,
                  out_shape, out_stride, out_dtype,
                  runtime, stream=None):
    ...
```
Each should follow the softmax/mean pattern: create tensors, call `*_get_workspace`, allocate workspace, invoke, defer executor, destroy tensors, defer workspace free.

Add symbol checks:
```python
def gather_symbols_ok():
    ...

def masked_select_symbols_ok():
    ...
```

**Step 3: Run tests**
Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_gather -vv`
Expected: FAIL until ops are wired.

**Step 4: Commit**
```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py
git commit -m "feat(npu): add aclnn gather/masked_select bindings"
```

---

### Task 3: Implement NPU ops + register

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/__init__.py`

**Step 1: Implement ops in `ops.py`**
Add helpers:
```python
def _ensure_int64_indices(indices, name, allow_negative):
    ...
```
```python
def _normalize_indices(indices, dim_size):
    ...
```
```python
def _normalize_dim(dim, ndim):
    ...  # reuse existing helper if present
```
Implement:
- `gather(a, dim, index)`
  - validate `dim`, index dtype, shape match; reject negative or out-of-range indices (raise `IndexError`).
  - call `aclnn.gather`.
- `index_select(a, dim, index)`
  - validate 1D index; allow negatives (wrap), reject out-of-range.
  - build index tensor with expanded shape to match input and call `gather`.
- `take(a, index)`
  - flatten `a`, normalize negatives, reject out-of-range.
  - call `gather` on dim 0.
- `take_along_dim(a, indices, dim)`
  - validate shape; normalize negatives, reject out-of-range.
  - call `gather`.
- `masked_select(a, mask)`
  - validate mask broadcastability to `a` (same shape for now); call `aclnn.masked_select`.

**Step 2: Register ops in `__init__.py`**
```python
registry.register("gather", "npu", gather, meta=meta_infer.infer_gather)
registry.register("index_select", "npu", index_select, meta=meta_infer.infer_index_select)
registry.register("take", "npu", take, meta=meta_infer.infer_take)
registry.register("take_along_dim", "npu", take_along_dim, meta=meta_infer.infer_take_along_dim)
registry.register("masked_select", "npu", masked_select, meta=meta_infer.infer_masked_select)
```

**Step 3: Run tests**
Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_take -vv`
Expected: PASS.

**Step 4: Commit**
```bash
git add src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/__init__.py
git commit -m "feat(mindtorch_v2): add NPU gather/index_select/take ops"
```

---

### Task 4: Update ops coverage table

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Append entries**
Add one line per op: `gather`, `index_select`, `take`, `take_along_dim`, `masked_select` with owner `NPU`, status `done`, and PR link placeholder.

**Step 2: Commit**
```bash
git add docs/plans/ops-coverage.md
git commit -m "docs: update ops coverage for NPU index/select ops"
```

---

### Task 5: Verification + PR

**Step 1: Run NPU tests**
Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_ops_npu.py -vv`
Expected: Failures only from baseline (layer_norm + elementwise batch2).

**Step 2: Push + PR**
```bash
git push origin npu-ops-batch4
```
Create PR to `mindspore-lab/mindnlp` with notes about baseline failures.
