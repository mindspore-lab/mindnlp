# gather/scatter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CPU + meta support for `gather` and `scatter` with PyTorch-aligned behavior and update ops coverage.

**Architecture:** CPU backend uses NumPy `take_along_axis` for gather and indexed assignment for scatter. Meta backend provides shape/dtype inference and validation. Functional APIs dispatch to backend kernels and are exported in `__init__.py`.

**Tech Stack:** Python, NumPy, PyTest, mindtorch v2 dispatch/registry.

---

### Task 1: Write failing tests (CPU + meta)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write CPU tests**

Add:
- `test_gather_cpu` (basic + negative index error)
- `test_scatter_cpu` (tensor src + scalar src)
- Add one out-of-range error test for each (expect `IndexError`).

**Step 2: Write meta tests**

Add:
- `test_meta_gather_shape`
- `test_meta_scatter_shape`
- Add one error case for shape mismatch and dim out of range.

**Step 3: Run tests to verify they fail**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_gather_cpu tests/mindtorch_v2/test_ops_cpu.py::test_scatter_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_gather_shape tests/mindtorch_v2/test_meta_device.py::test_meta_scatter_shape -q`

Expected: FAIL with op not registered/implemented.

---

### Task 2: Add CPU kernels

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`

**Step 1: Implement**

Add:

```python
def _ensure_integer_indices(arr, name):
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must be integer dtype")
    return arr


def _check_index_range(index, dim_size):
    if (index < 0).any() or (index >= dim_size).any():
        raise IndexError("index out of range")


def gather(a, dim, index):
    arr = _to_numpy(a)
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("index shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("index shape mismatch")
    _check_index_range(idx, arr.shape[dim])
    out = np.take_along_axis(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)


def scatter(a, dim, index, src):
    arr = _to_numpy(a).copy()
    idx = _ensure_integer_indices(_to_numpy(index), "index").astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    if dim < 0 or dim >= arr.ndim:
        raise ValueError("dim out of range")
    if idx.ndim != arr.ndim:
        raise ValueError("index shape mismatch")
    for i, size in enumerate(idx.shape):
        if i != dim and size != arr.shape[i]:
            raise ValueError("index shape mismatch")
    _check_index_range(idx, arr.shape[dim])
    if hasattr(src, "shape"):
        src_arr = _to_numpy(src)
    else:
        src_arr = np.array(src, dtype=arr.dtype)
    src_arr = np.broadcast_to(src_arr, idx.shape)
    np.put_along_axis(arr, idx, src_arr, axis=dim)
    return _from_numpy(arr, a.dtype, a.device)
```

**Step 2: Run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_gather_cpu tests/mindtorch_v2/test_ops_cpu.py::test_scatter_cpu -q`

Expected: PASS for CPU tests.

---

### Task 3: Register CPU ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Add imports**

Add `gather`, `scatter` to imports.

**Step 2: Register**

Add:

```python
registry.register("gather", "cpu", gather, meta=meta_infer.infer_gather)
registry.register("scatter", "cpu", scatter, meta=meta_infer.infer_scatter)
```

**Step 3: Re-run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_gather_cpu tests/mindtorch_v2/test_ops_cpu.py::test_scatter_cpu -q`

Expected: PASS.

---

### Task 4: Add meta infer

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Implement**

```python
def infer_gather(a, dim, index):
    if dim < 0:
        dim += len(a.shape)
    if dim < 0 or dim >= len(a.shape):
        raise ValueError("dim out of range")
    if len(index.shape) != len(a.shape):
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    shape = tuple(index.shape)
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)


def infer_scatter(a, dim, index, src):
    if dim < 0:
        dim += len(a.shape)
    if dim < 0 or dim >= len(a.shape):
        raise ValueError("dim out of range")
    if len(index.shape) != len(a.shape):
        raise ValueError("index shape mismatch")
    for i, size in enumerate(index.shape):
        if i != dim and size != a.shape[i]:
            raise ValueError("index shape mismatch")
    return TensorSpec(shape=tuple(a.shape), stride=_contiguous_stride(a.shape), dtype=a.dtype)
```

**Step 2: Run meta tests**

Run:
`pytest tests/mindtorch_v2/test_meta_device.py::test_meta_gather_shape tests/mindtorch_v2/test_meta_device.py::test_meta_scatter_shape -q`

Expected: PASS.

---

### Task 5: Add meta op registration

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Add meta wrapper ops**

In `ops.py` add:

```python
def _meta_gather_meta(a, dim, index):
    return _meta_tensor(index.shape, a.dtype, a.device)


def _meta_scatter_meta(a, dim, index, src):
    return _meta_tensor(a.shape, a.dtype, a.device)
```

**Step 2: Register**

In `meta/__init__.py` add imports and register:

```python
registry.register("gather", "meta", _meta_gather_meta)
registry.register("scatter", "meta", _meta_scatter_meta)
```

---

### Task 6: Functional API + exports

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/__init__.py`

**Step 1: Add functional wrappers**

```python
def gather(a, dim, index):
    return dispatch("gather", a.device.type, a, dim, index)


def scatter(a, dim, index, src):
    return dispatch("scatter", a.device.type, a, dim, index, src)
```

**Step 2: Export in `__init__.py`**

Add to import list and `__all__`: `gather`, `scatter`.

---

### Task 7: Update ops coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Add lines for:
- `aten::gather`
- `aten::scatter`

---

### Task 8: Commit

**Step 1: Git status**

Run: `git status -sb`

**Step 2: Commit**

Run:
```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py \
  src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py \
  src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/ops.py \
  src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_functional.py \
  src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md

git commit -m "feat: add gather scatter"
```

---

### Task 9: Verification + PR prep

**Step 1: Run focused tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_gather_cpu tests/mindtorch_v2/test_ops_cpu.py::test_scatter_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_gather_shape tests/mindtorch_v2/test_meta_device.py::test_meta_scatter_shape -q`

Expected: PASS.

**Step 2: Rebase `ms/master`**

Run:
`git fetch`
`git rebase ms/master`

**Step 3: Push + create PR**

Use correct PR description with real newlines (heredoc).
