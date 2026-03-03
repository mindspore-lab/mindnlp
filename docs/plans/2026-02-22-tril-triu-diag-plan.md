# Tril/Triu/Diag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add CPU + meta support for `tril`, `triu`, `diag` with PyTorch-aligned behavior and update ops coverage.

**Architecture:** CPU backend uses NumPy (`np.tril/np.triu/np.diag`) and wraps outputs via `_from_numpy`. Meta backend adds shape inference for `diag` and reuses unary meta for `tril/triu`. Functional and Tensor APIs dispatch to backend kernels.

**Tech Stack:** Python, NumPy, PyTest, mindtorch v2 dispatch/registry.

---

### Task 1: Verify tests are red (TDD baseline)

**Files:**
- Test: `tests/mindtorch_v2/test_ops_cpu.py`
- Test: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Run failing tests (CPU + meta)**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_tril_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_cpu tests/mindtorch_v2/test_ops_cpu.py::test_diag_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_shape tests/mindtorch_v2/test_meta_device.py::test_meta_diag_shape -q`

Expected: FAIL with "op not registered/implemented" or similar.

---

### Task 2: Add CPU kernels (tril/triu/diag)

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`

**Step 1: Write minimal implementation**

Add:

```python
def tril(a, diagonal=0):
    out = np.tril(_to_numpy(a), k=diagonal)
    return _from_numpy(out, a.dtype, a.device)

def triu(a, diagonal=0):
    out = np.triu(_to_numpy(a), k=diagonal)
    return _from_numpy(out, a.dtype, a.device)

def diag(a, diagonal=0):
    arr = _to_numpy(a)
    if arr.ndim not in (1, 2):
        raise ValueError("diag expects 1D or 2D tensor")
    out = np.diag(arr, k=diagonal)
    return _from_numpy(out, a.dtype, a.device)
```

**Step 2: Run CPU tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_tril_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_cpu tests/mindtorch_v2/test_ops_cpu.py::test_diag_cpu -q`

Expected: PASS for CPU tests (meta tests may still fail).

---

### Task 3: Register CPU ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Add imports**

Add `tril`, `triu`, `diag` to the import list near other ops.

**Step 2: Register**

Add:

```python
registry.register("tril", "cpu", tril, meta=meta_infer.infer_unary)
registry.register("triu", "cpu", triu, meta=meta_infer.infer_unary)
registry.register("diag", "cpu", diag, meta=meta_infer.infer_diag)
```

**Step 3: Re-run tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_tril_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_cpu tests/mindtorch_v2/test_ops_cpu.py::test_diag_cpu -q`

Expected: PASS.

---

### Task 4: Add meta infer for diag

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Implement infer_diag**

Add:

```python
def infer_diag(a, diagonal=0):
    if len(a.shape) == 1:
        n = a.shape[0]
        size = n + abs(diagonal)
        shape = (size, size)
    elif len(a.shape) == 2:
        m, n = a.shape
        if diagonal >= 0:
            length = max(0, min(m, n - diagonal))
        else:
            length = max(0, min(m + diagonal, n))
        shape = (length,)
    else:
        raise ValueError("diag expects 1D or 2D tensor")
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)
```

**Step 2: Run meta tests**

Run:
`pytest tests/mindtorch_v2/test_meta_device.py::test_meta_diag_shape -q`

Expected: PASS.

---

### Task 5: Add meta op registration (tril/triu/diag)

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Add meta diag helper**

In `ops.py`, add:

```python
def _meta_diag_meta(a, diagonal=0):
    if len(a.shape) == 1:
        size = a.shape[0] + abs(diagonal)
        shape = (size, size)
    elif len(a.shape) == 2:
        m, n = a.shape
        if diagonal >= 0:
            length = max(0, min(m, n - diagonal))
        else:
            length = max(0, min(m + diagonal, n))
        shape = (length,)
    else:
        raise ValueError("diag expects 1D or 2D tensor")
    return _meta_tensor(shape, a.dtype, a.device)
```

**Step 2: Register in meta backend**

In `meta/__init__.py`:
- Import `_meta_diag_meta`
- Register:

```python
registry.register("tril", "meta", _meta_unary_meta)
registry.register("triu", "meta", _meta_unary_meta)
registry.register("diag", "meta", _meta_diag_meta)
```

**Step 3: Run meta tests**

Run:
`pytest tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_shape tests/mindtorch_v2/test_meta_device.py::test_meta_diag_shape -q`

Expected: PASS.

---

### Task 6: Functional + Tensor API

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`

**Step 1: Add functional wrappers**

In `_functional.py` add:

```python
def tril(a, diagonal=0):
    return dispatch("tril", a.device.type, a, diagonal)

def triu(a, diagonal=0):
    return dispatch("triu", a.device.type, a, diagonal)

def diag(a, diagonal=0):
    return dispatch("diag", a.device.type, a, diagonal)
```

**Step 2: Add Tensor methods (if consistent with existing API)**

In `_tensor.py` add:

```python
def tril(self, diagonal=0):
    return _functional.tril(self, diagonal)

def triu(self, diagonal=0):
    return _functional.triu(self, diagonal)

def diag(self, diagonal=0):
    return _functional.diag(self, diagonal)
```

**Step 3: Export in `__init__.py`**

Add to import list and `__all__`: `tril`, `triu`, `diag`.

**Step 4: Run targeted tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_tril_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_cpu tests/mindtorch_v2/test_ops_cpu.py::test_diag_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_shape tests/mindtorch_v2/test_meta_device.py::test_meta_diag_shape -q`

Expected: PASS.

---

### Task 7: Update ops coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Add coverage entries for `aten::tril`, `aten::triu`, `aten::diag` (CPU + meta).

---

### Task 8: Commit

**Files:**
- Modify: files above

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

git commit -m "feat: add tril triu diag ops"
```

---

### Task 9: Verification + PR prep

**Step 1: Run focused tests**

Run:
`pytest tests/mindtorch_v2/test_ops_cpu.py::test_tril_cpu tests/mindtorch_v2/test_ops_cpu.py::test_triu_cpu tests/mindtorch_v2/test_ops_cpu.py::test_diag_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_tril_triu_shape tests/mindtorch_v2/test_meta_device.py::test_meta_diag_shape -q`

Expected: PASS.

**Step 2: Rebase `ms/master`**

Run:
`git fetch`
`git rebase ms/master`

**Step 3: Push + create PR**

Use correct PR description with real newlines (heredoc).
