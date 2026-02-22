# dstack/row_stack/concatenate + tril_indices/triu_indices Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CPU + meta support for `dstack`, `row_stack`, `concatenate`, `tril_indices`, `triu_indices` with PyTorch-aligned behavior and update ops coverage.

**Architecture:** CPU backend uses NumPy for concrete outputs; meta backend provides shape/dtype inference only. Functional APIs dispatch to backend kernels; Tensor APIs add thin wrappers for the stack/concat ops. Indices ops return a `(2, N)` tensor with dtype defaulting to int64. Layout is accepted but only supports strided.

**Tech Stack:** Python, NumPy, PyTest, mindtorch v2 dispatch/registry.

---

### Task 1: Verify tests are red (TDD baseline)

**Files:**
- Test: `tests/mindtorch_v2/test_ops_cpu.py`
- Test: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Add failing tests**

Add tests:
- `test_concatenate_cpu` (alias of `cat`) + `axis` keyword
- `test_row_stack_cpu` (1D and 2D)
- `test_dstack_cpu` (1D and 2D)
- `test_tril_indices_cpu`
- `test_triu_indices_cpu`

Add meta tests:
- `test_meta_concatenate_shape`
- `test_meta_row_stack_shape`
- `test_meta_dstack_shape`
- `test_meta_tril_triu_indices_shape`

**Step 2: Run tests (expect FAIL)**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_concatenate_cpu tests/mindtorch_v2/test_ops_cpu.py::test_row_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_tril_indices_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_indices_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_concatenate_shape tests/mindtorch_v2/test_meta_device.py::test_meta_row_stack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_dstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_indices_shape -q`

Expected: FAIL with op not registered/implemented.

---

### Task 2: Add CPU kernels (dstack/row_stack/concatenate/tril_indices/triu_indices)

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`

**Step 1: Write minimal implementations**

Add:

```python
def concatenate(tensors, dim=0):
    return cat(tensors, dim=dim)


def row_stack(tensors):
    return vstack(tensors)


def dstack(tensors):
    arrays = [_to_numpy(t) for t in tensors]
    expanded = []
    for a in arrays:
        if a.ndim == 1:
            expanded.append(a.reshape(1, a.shape[0], 1))
        elif a.ndim == 2:
            expanded.append(a.reshape(a.shape[0], a.shape[1], 1))
        else:
            expanded.append(a)
    out = np.concatenate(expanded, axis=2)
    return _from_numpy(out, tensors[0].dtype, tensors[0].device)


def _check_indices_layout(layout):
    if layout is None:
        return
    if isinstance(layout, str):
        if layout != "strided":
            raise ValueError("layout must be strided")
        return
    raise ValueError("layout must be strided")


def tril_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    _check_indices_layout(layout)
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    if device is None:
        device = _to_device(device) if device is not None else None
    r, c = np.tril_indices(row, k=offset, m=col)
    out = np.stack([r, c], axis=0)
    return _from_numpy(out.astype(np.dtype(dtype.np_type)), dtype, device or _default_device())


def triu_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    _check_indices_layout(layout)
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    if device is None:
        device = _to_device(device) if device is not None else None
    r, c = np.triu_indices(row, k=offset, m=col)
    out = np.stack([r, c], axis=0)
    return _from_numpy(out.astype(np.dtype(dtype.np_type)), dtype, device or _default_device())
```

**Step 2: Run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_concatenate_cpu tests/mindtorch_v2/test_ops_cpu.py::test_row_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_tril_indices_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_indices_cpu -q`

Expected: PASS for CPU tests.

---

### Task 3: Register CPU ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Add imports**

Add `dstack`, `row_stack`, `concatenate`, `tril_indices`, `triu_indices` to imports.

**Step 2: Register**

Add:

```python
registry.register("concatenate", "cpu", concatenate, meta=meta_infer.infer_cat)
registry.register("row_stack", "cpu", row_stack, meta=meta_infer.infer_vstack)
registry.register("dstack", "cpu", dstack, meta=meta_infer.infer_dstack)
registry.register("tril_indices", "cpu", tril_indices, meta=meta_infer.infer_tril_triu_indices)
registry.register("triu_indices", "cpu", triu_indices, meta=meta_infer.infer_tril_triu_indices)
```

**Step 3: Re-run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_concatenate_cpu tests/mindtorch_v2/test_ops_cpu.py::test_row_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_tril_indices_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_indices_cpu -q`

Expected: PASS.

---

### Task 4: Add meta infer (dstack + tril/triu indices)

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Implement infer_dstack**

```python
def infer_dstack(tensors):
    shapes = [t.shape for t in tensors]
    expanded = []
    for shape in shapes:
        if len(shape) == 1:
            expanded.append((1, shape[0], 1))
        elif len(shape) == 2:
            expanded.append((shape[0], shape[1], 1))
        else:
            expanded.append(shape)
    first = expanded[0]
    if len(first) < 3:
        raise ValueError("dstack expects 1D, 2D, or 3D+ tensors")
    out_shape = list(first)
    out_shape[2] = sum(s[2] for s in expanded)
    return TensorSpec(shape=tuple(out_shape), stride=_contiguous_stride(out_shape), dtype=tensors[0].dtype)
```

**Step 2: Implement infer_tril_triu_indices**

```python
def infer_tril_triu_indices(row, col, offset=0, dtype=None, device=None, layout=None):
    if row < 0 or col < 0:
        raise ValueError("row and col must be non-negative")
    if dtype is None:
        dtype = int64_dtype
    if offset >= 0:
        count = max(0, min(row, col - offset))
        n = (2 * col - offset - count + 1) * count // 2
    else:
        count = max(0, min(row + offset, col))
        n = (2 * row + offset - count + 1) * count // 2
    shape = (2, n)
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=dtype)
```

**Step 3: Run meta tests**

Run:
`pytest tests/mindtorch_v2/test_meta_device.py::test_meta_concatenate_shape tests/mindtorch_v2/test_meta_device.py::test_meta_row_stack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_dstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_indices_shape -q`

Expected: PASS.

---

### Task 5: Add meta op registration

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Add meta helper for dstack + indices**

In `ops.py` add:

```python
def _meta_dstack_meta(tensors):
    spec = infer_dstack(tensors)
    return _meta_tensor(spec.shape, spec.dtype, tensors[0].device)


def _meta_tril_triu_indices_meta(row, col, offset=0, dtype=None, device=None, layout=None):
    spec = infer_tril_triu_indices(row, col, offset=offset, dtype=dtype, device=device, layout=layout)
    return _meta_tensor(spec.shape, spec.dtype, device or Device("meta"))
```

**Step 2: Register**

In `meta/__init__.py` add imports and register:

```python
registry.register("concatenate", "meta", _meta_cat_meta)
registry.register("row_stack", "meta", _meta_vstack_meta)
registry.register("dstack", "meta", _meta_dstack_meta)
registry.register("tril_indices", "meta", _meta_tril_triu_indices_meta)
registry.register("triu_indices", "meta", _meta_tril_triu_indices_meta)
```

---

### Task 6: Functional + Tensor API

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`

**Step 1: Add functional wrappers**

```python
def concatenate(tensors, dim=0, axis=None):
    if axis is not None:
        dim = axis
    return dispatch("concatenate", tensors[0].device.type, tensors, dim)


def row_stack(tensors):
    return dispatch("row_stack", tensors[0].device.type, tensors)


def dstack(tensors):
    return dispatch("dstack", tensors[0].device.type, tensors)


def tril_indices(row, col, offset=0, *, dtype=None, device=None, layout=None):
    dev = _as_device(device)
    return dispatch("tril_indices", dev, row, col, offset, dtype=dtype, device=dev, layout=layout)


def triu_indices(row, col, offset=0, *, dtype=None, device=None, layout=None):
    dev = _as_device(device)
    return dispatch("triu_indices", dev, row, col, offset, dtype=dtype, device=dev, layout=layout)
```

**Step 2: Add Tensor methods**

```python
def dstack(self, tensors):
    return _functional.dstack(tensors)


def row_stack(self, tensors):
    return _functional.row_stack(tensors)


def concatenate(self, tensors, dim=0, axis=None):
    return _functional.concatenate(tensors, dim=dim, axis=axis)
```

**Step 3: Export in `__init__.py`**

Add to import list and `__all__` for:
`dstack`, `row_stack`, `concatenate`, `tril_indices`, `triu_indices`.

---

### Task 7: Update ops coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Add lines for:
- `aten::dstack`
- `aten::row_stack`
- `aten::concatenate`
- `aten::tril_indices`
- `aten::triu_indices`

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
  src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md

git commit -m "feat: add dstack row_stack concatenate and tri indices"
```

---

### Task 9: Verification + PR prep

**Step 1: Run focused tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_concatenate_cpu tests/mindtorch_v2/test_ops_cpu.py::test_row_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_dstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_tril_indices_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_indices_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_concatenate_shape tests/mindtorch_v2/test_meta_device.py::test_meta_row_stack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_dstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_indices_shape -q`

Expected: PASS.

**Step 2: Rebase `ms/master`**

Run:
`git fetch`
`git rebase ms/master`

**Step 3: Push + create PR**

Use correct PR description with real newlines (heredoc).
