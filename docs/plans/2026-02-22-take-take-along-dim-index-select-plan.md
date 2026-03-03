# take/take_along_dim/index_select Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CPU + meta support for `take`, `take_along_dim`, and `index_select` with PyTorch-aligned behavior and update ops coverage.

**Architecture:** CPU backend uses NumPy `take`/`take_along_axis` and wraps outputs via `_from_numpy`. Meta backend adds shape inference for each op. Functional APIs dispatch to backend kernels and exports are added to `__init__.py`.

**Tech Stack:** Python, NumPy, PyTest, mindtorch v2 dispatch/registry.

---

### Task 1: Write failing tests (CPU + meta)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write CPU tests**

Add:
- `test_take_cpu` (basic + negative index)
- `test_take_along_dim_cpu` (basic + negative index)
- `test_index_select_cpu` (basic + negative index)
- Add one out-of-range error test for each (expect `IndexError`).

**Step 2: Write meta tests**

Add:
- `test_meta_take_shape`
- `test_meta_take_along_dim_shape`
- `test_meta_index_select_shape`
- Add one error case for `take_along_dim` shape mismatch and `index_select` non-1D index.

**Step 3: Run tests to verify they fail**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_take_cpu tests/mindtorch_v2/test_ops_cpu.py::test_take_along_dim_cpu tests/mindtorch_v2/test_ops_cpu.py::test_index_select_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_take_shape tests/mindtorch_v2/test_meta_device.py::test_meta_take_along_dim_shape tests/mindtorch_v2/test_meta_device.py::test_meta_index_select_shape -q`

Expected: FAIL with op not registered/implemented.

---

### Task 2: Add CPU kernels

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`

**Step 1: Implement**

Add:

```python
def take(a, index):
    arr = _to_numpy(a).reshape(-1)
    idx = _to_numpy(index).astype(np.int64, copy=False)
    out = np.take(arr, idx)
    return _from_numpy(out, a.dtype, a.device)


def take_along_dim(a, indices, dim):
    arr = _to_numpy(a)
    idx = _to_numpy(indices).astype(np.int64, copy=False)
    if dim < 0:
        dim += arr.ndim
    out = np.take_along_axis(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)


def index_select(a, dim, index):
    arr = _to_numpy(a)
    idx = _to_numpy(index).astype(np.int64, copy=False)
    if idx.ndim != 1:
        raise ValueError("index must be 1D")
    if dim < 0:
        dim += arr.ndim
    out = np.take(arr, idx, axis=dim)
    return _from_numpy(out, a.dtype, a.device)
```

**Step 2: Run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_take_cpu tests/mindtorch_v2/test_ops_cpu.py::test_take_along_dim_cpu tests/mindtorch_v2/test_ops_cpu.py::test_index_select_cpu -q`

Expected: PASS for CPU tests.

---

### Task 3: Register CPU ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Add imports**

Add `take`, `take_along_dim`, `index_select` to imports.

**Step 2: Register**

Add:

```python
registry.register("take", "cpu", take, meta=meta_infer.infer_take)
registry.register("take_along_dim", "cpu", take_along_dim, meta=meta_infer.infer_take_along_dim)
registry.register("index_select", "cpu", index_select, meta=meta_infer.infer_index_select)
```

**Step 3: Re-run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_take_cpu tests/mindtorch_v2/test_ops_cpu.py::test_take_along_dim_cpu tests/mindtorch_v2/test_ops_cpu.py::test_index_select_cpu -q`

Expected: PASS.

---

### Task 4: Add meta infer

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Implement**

```python
def infer_take(a, index):
    return TensorSpec(shape=tuple(index.shape), stride=_contiguous_stride(index.shape), dtype=a.dtype)


def infer_take_along_dim(a, indices, dim):
    if dim < 0:
        dim += len(a.shape)
    if dim < 0 or dim >= len(a.shape):
        raise ValueError("dim out of range")
    for i, s in enumerate(indices.shape):
        if i != dim and s != a.shape[i]:
            raise ValueError("indices shape mismatch")
    return TensorSpec(shape=tuple(indices.shape), stride=_contiguous_stride(indices.shape), dtype=a.dtype)


def infer_index_select(a, dim, index):
    if len(index.shape) != 1:
        raise ValueError("index must be 1D")
    if dim < 0:
        dim += len(a.shape)
    if dim < 0 or dim >= len(a.shape):
        raise ValueError("dim out of range")
    shape = list(a.shape)
    shape[dim] = index.shape[0]
    shape = tuple(shape)
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)
```

**Step 2: Run meta tests**

Run:
`pytest tests/mindtorch_v2/test_meta_device.py::test_meta_take_shape tests/mindtorch_v2/test_meta_device.py::test_meta_take_along_dim_shape tests/mindtorch_v2/test_meta_device.py::test_meta_index_select_shape -q`

Expected: PASS.

---

### Task 5: Add meta op registration

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Add meta wrapper ops**

In `ops.py` add:

```python
def _meta_take_meta(a, index):
    return _meta_tensor(index.shape, a.dtype, a.device)


def _meta_take_along_dim_meta(a, indices, dim):
    return _meta_tensor(indices.shape, a.dtype, a.device)


def _meta_index_select_meta(a, dim, index):
    shape = list(a.shape)
    shape[dim] = index.shape[0]
    return _meta_tensor(tuple(shape), a.dtype, a.device)
```

**Step 2: Register**

In `meta/__init__.py` add imports and register:

```python
registry.register("take", "meta", _meta_take_meta)
registry.register("take_along_dim", "meta", _meta_take_along_dim_meta)
registry.register("index_select", "meta", _meta_index_select_meta)
```

---

### Task 6: Functional API + exports

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/__init__.py`

**Step 1: Add functional wrappers**

```python
def take(a, index):
    return dispatch("take", a.device.type, a, index)


def take_along_dim(a, indices, dim):
    return dispatch("take_along_dim", a.device.type, a, indices, dim)


def index_select(a, dim, index):
    return dispatch("index_select", a.device.type, a, dim, index)
```

**Step 2: Export in `__init__.py`**

Add to import list and `__all__`: `take`, `take_along_dim`, `index_select`.

---

### Task 7: Update ops coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Add lines for:
- `aten::take`
- `aten::take_along_dim`
- `aten::index_select`

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

git commit -m "feat: add take take_along_dim index_select"
```

---

### Task 9: Verification + PR prep

**Step 1: Run focused tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_take_cpu tests/mindtorch_v2/test_ops_cpu.py::test_take_along_dim_cpu tests/mindtorch_v2/test_ops_cpu.py::test_index_select_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_take_shape tests/mindtorch_v2/test_meta_device.py::test_meta_take_along_dim_shape tests/mindtorch_v2/test_meta_device.py::test_meta_index_select_shape -q`

Expected: PASS.

**Step 2: Rebase `ms/master`**

Run:
`git fetch`
`git rebase ms/master`

**Step 3: Push + create PR**

Use correct PR description with real newlines (heredoc).
