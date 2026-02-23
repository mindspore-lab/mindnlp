# Masked Select, Nonzero, Where Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CPU/meta support for `masked_select`, `nonzero`, and `where(condition-only)` with torch-aligned behavior, using TDD.

**Architecture:** Add numpy-backed CPU ops, meta shape inference using `0` for variable lengths, and functional/front exports. Register ops with CPU/meta backends and update coverage docs. Tests drive each change.

**Tech Stack:** Python, NumPy, PyTest, MindTorch v2 dispatch.

---

### Task 1: Add failing CPU/meta tests for masked_select

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_masked_select_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    mask = torch.tensor([[True, False], [False, True]])
    out = torch.masked_select(x, mask)
    np.testing.assert_array_equal(out.numpy(), np.array([1, 4]))
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_masked_select_shape():
    x = torch.empty((2, 3), device="meta")
    mask = torch.empty((2, 3), device="meta", dtype=torch.bool)
    out = torch.masked_select(x, mask)
    assert out.shape == (0,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_masked_select_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_masked_select_shape -q`

Expected: FAIL because `masked_select` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def masked_select(a, mask):
    arr = a.numpy()
    mask_arr = mask.numpy().astype(bool)
    out = arr[mask_arr]
    return torch.tensor(out, device=a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_masked_select_meta(a, mask):
    return _meta_tensor((0,), a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_masked_select(a, mask):
    return (0,), a.dtype, a.device
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register masked_select
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register masked_select
```

```python
# src/mindtorch_v2/_functional.py

def masked_select(a, mask):
    return dispatch("masked_select", a.device.type, a, mask)
```

```python
# src/mindtorch_v2/__init__.py
# export masked_select
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_masked_select_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_masked_select_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add masked_select"
```

---

### Task 2: Add failing CPU/meta tests for nonzero (incl. as_tuple)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_nonzero_cpu():
    x = torch.tensor([[0, 1], [2, 0]])
    out = torch.nonzero(x)
    np.testing.assert_array_equal(out.numpy(), np.array([[0, 1], [1, 0]]))


def test_nonzero_as_tuple_cpu():
    x = torch.tensor([[0, 1], [2, 0]])
    out = torch.nonzero(x, as_tuple=True)
    assert isinstance(out, tuple)
    np.testing.assert_array_equal(out[0].numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(out[1].numpy(), np.array([1, 0]))
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_nonzero_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.nonzero(x)
    assert out.shape == (0, 2)


def test_meta_nonzero_as_tuple_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.nonzero(x, as_tuple=True)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (0,)
    assert out[1].shape == (0,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_nonzero_cpu tests/mindtorch_v2/test_ops_cpu.py::test_nonzero_as_tuple_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_nonzero_shape tests/mindtorch_v2/test_meta_device.py::test_meta_nonzero_as_tuple_shape -q`

Expected: FAIL because `nonzero` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def nonzero(a, as_tuple=False):
    arr = a.numpy()
    idx = np.nonzero(arr)
    if as_tuple:
        return tuple(torch.tensor(dim_idx, device=a.device) for dim_idx in idx)
    stacked = np.stack(idx, axis=1).astype(np.int64)
    return torch.tensor(stacked, device=a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_nonzero_meta(a, as_tuple=False):
    if as_tuple:
        return tuple(_meta_tensor((0,), int64_dtype, a.device) for _ in range(len(a.shape)))
    return _meta_tensor((0, len(a.shape)), int64_dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_nonzero(a, as_tuple=False):
    if as_tuple:
        return tuple(((0,), int64_dtype, a.device) for _ in range(len(a.shape)))
    return (0, len(a.shape)), int64_dtype, a.device
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register nonzero
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register nonzero
```

```python
# src/mindtorch_v2/_functional.py

def nonzero(a, as_tuple=False):
    return dispatch("nonzero", a.device.type, a, as_tuple=as_tuple)
```

```python
# src/mindtorch_v2/__init__.py
# export nonzero
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_nonzero_cpu tests/mindtorch_v2/test_ops_cpu.py::test_nonzero_as_tuple_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_nonzero_shape tests/mindtorch_v2/test_meta_device.py::test_meta_nonzero_as_tuple_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add nonzero"
```

---

### Task 3: Add failing test for where(condition-only)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_where_condition_cpu():
    cond = torch.tensor([[True, False], [False, True]])
    out = torch.where(cond)
    assert isinstance(out, tuple)
    np.testing.assert_array_equal(out[0].numpy(), np.array([0, 1]))
    np.testing.assert_array_equal(out[1].numpy(), np.array([0, 1]))
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_where_condition_shape():
    cond = torch.empty((2, 3), device="meta", dtype=torch.bool)
    out = torch.where(cond)
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (0,)
    assert out[1].shape == (0,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_where_condition_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_where_condition_shape -q`

Expected: FAIL because `where(condition-only)` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_functional.py

def where(condition, x=None, y=None):
    if x is None and y is None:
        return nonzero(condition, as_tuple=True)
    return dispatch("where", condition.device.type, condition, x, y)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_where_condition_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_where_condition_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_functional.py
git commit -m "feat: add where(condition)"
```

---

### Task 4: Update coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Append rows for:
- `aten::masked_select`
- `aten::nonzero`
- `aten::where` (condition-only behavior)

**Step 2: Commit**

```bash
git add docs/plans/ops-coverage.md
git commit -m "docs: record masked_select/nonzero/where"
```

---

### Task 5: Final verification

**Step 1: Run focused tests**

Run:
```bash
pytest \
  tests/mindtorch_v2/test_ops_cpu.py::test_masked_select_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_nonzero_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_nonzero_as_tuple_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_where_condition_cpu \
  tests/mindtorch_v2/test_meta_device.py::test_meta_masked_select_shape \
  tests/mindtorch_v2/test_meta_nonzero_shape \
  tests/mindtorch_v2/test_meta_nonzero_as_tuple_shape \
  tests/mindtorch_v2/test_meta_where_condition_shape -q
```
Expected: PASS.

**Step 2: Rebase and push**

```bash
git fetch origin
git rebase origin/ms/master
git push --force-with-lease
```

**Step 3: Create PR**
- Target: `mindspore-lab/mindnlp`, base `ms/master`.
- Ensure PR description uses proper newlines.

