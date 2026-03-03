# Repeat, Repeat_Interleave, Tile Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CPU/meta support for `repeat`, `repeat_interleave`, and `tile` with torch-aligned behavior, using TDD.

**Architecture:** Use NumPy (`np.tile`, `np.repeat`) for CPU kernels, with explicit parameter normalization to match torch semantics. Meta shape inference computes output shapes and validates parameter constraints. Register ops in CPU/meta backends and update coverage docs.

**Tech Stack:** Python, NumPy, PyTest, MindTorch v2 dispatch.

---

### Task 1: Add failing CPU/meta tests for repeat

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_repeat_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    out = torch.repeat(x, (2, 3))
    expected = np.tile(x.numpy(), (2, 3))
    np.testing.assert_array_equal(out.numpy(), expected)
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_repeat_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.repeat(x, (2, 3))
    assert out.shape == (4, 9)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_repeat_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_repeat_shape -q`

Expected: FAIL because `repeat` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def repeat(a, repeats):
    arr = _to_numpy(a)
    out = np.tile(arr, repeats)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_repeat_meta(a, repeats):
    if isinstance(repeats, int):
        repeats = (repeats,)
    if len(repeats) < len(a.shape):
        repeats = (1,) * (len(a.shape) - len(repeats)) + tuple(repeats)
    if len(repeats) != len(a.shape):
        raise ValueError("repeats must match input rank")
    shape = tuple(s * r for s, r in zip(a.shape, repeats))
    return _meta_tensor(shape, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_repeat(a, repeats):
    if isinstance(repeats, int):
        repeats = (repeats,)
    if len(repeats) < len(a.shape):
        repeats = (1,) * (len(a.shape) - len(repeats)) + tuple(repeats)
    if len(repeats) != len(a.shape):
        raise ValueError("repeats must match input rank")
    shape = tuple(s * r for s, r in zip(a.shape, repeats))
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register repeat
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register repeat
```

```python
# src/mindtorch_v2/_functional.py

def repeat(a, repeats):
    return dispatch("repeat", a.device.type, a, repeats)
```

```python
# src/mindtorch_v2/__init__.py
# export repeat
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_repeat_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_repeat_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add repeat"
```

---

### Task 2: Add failing CPU/meta tests for repeat_interleave

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_repeat_interleave_cpu():
    x = torch.tensor([1, 2, 3])
    out = torch.repeat_interleave(x, repeats=2)
    expected = np.repeat(x.numpy(), 2)
    np.testing.assert_array_equal(out.numpy(), expected)

    y = torch.tensor([[1, 2], [3, 4]])
    out = torch.repeat_interleave(y, repeats=2, dim=1)
    expected = np.repeat(y.numpy(), 2, axis=1)
    np.testing.assert_array_equal(out.numpy(), expected)
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_repeat_interleave_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.repeat_interleave(x, repeats=2, dim=1)
    assert out.shape == (2, 6)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_repeat_interleave_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_repeat_interleave_shape -q`

Expected: FAIL because `repeat_interleave` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def repeat_interleave(a, repeats, dim=None):
    arr = _to_numpy(a)
    axis = None if dim is None else dim
    out = np.repeat(arr, repeats, axis=axis)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_repeat_interleave_meta(a, repeats, dim=None):
    shape = list(a.shape)
    if dim is None:
        total = 1
        for s in shape:
            total *= s
        if isinstance(repeats, int):
            total *= repeats
        else:
            total = len(repeats)
        return _meta_tensor((total,), a.dtype, a.device)
    if dim < 0:
        dim += len(shape)
    if isinstance(repeats, int):
        shape[dim] *= repeats
    else:
        shape[dim] = len(repeats)
    return _meta_tensor(tuple(shape), a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_repeat_interleave(a, repeats, dim=None):
    shape = list(a.shape)
    if dim is None:
        total = 1
        for s in shape:
            total *= s
        if isinstance(repeats, int):
            total *= repeats
        else:
            total = len(repeats)
        return TensorSpec(shape=(total,), stride=_contiguous_stride((total,)), dtype=a.dtype)
    if dim < 0:
        dim += len(shape)
    if isinstance(repeats, int):
        shape[dim] *= repeats
    else:
        shape[dim] = len(repeats)
    shape = tuple(shape)
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register repeat_interleave
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register repeat_interleave
```

```python
# src/mindtorch_v2/_functional.py

def repeat_interleave(a, repeats, dim=None):
    return dispatch("repeat_interleave", a.device.type, a, repeats, dim)
```

```python
# src/mindtorch_v2/__init__.py
# export repeat_interleave
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_repeat_interleave_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_repeat_interleave_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add repeat_interleave"
```

---

### Task 3: Add failing CPU/meta tests for tile

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_tile_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    out = torch.tile(x, (2, 3))
    expected = np.tile(x.numpy(), (2, 3))
    np.testing.assert_array_equal(out.numpy(), expected)
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_tile_shape():
    x = torch.empty((2, 3), device="meta")
    out = torch.tile(x, (2, 3))
    assert out.shape == (4, 9)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_tile_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_tile_shape -q`

Expected: FAIL because `tile` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def tile(a, dims):
    arr = _to_numpy(a)
    out = np.tile(arr, dims)
    return _from_numpy(np.ascontiguousarray(out), a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_tile_meta(a, dims):
    if isinstance(dims, int):
        dims = (dims,)
    if len(dims) < len(a.shape):
        dims = (1,) * (len(a.shape) - len(dims)) + tuple(dims)
    if len(dims) != len(a.shape):
        raise ValueError("dims must match input rank")
    shape = tuple(s * d for s, d in zip(a.shape, dims))
    return _meta_tensor(shape, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_tile(a, dims):
    if isinstance(dims, int):
        dims = (dims,)
    if len(dims) < len(a.shape):
        dims = (1,) * (len(a.shape) - len(dims)) + tuple(dims)
    if len(dims) != len(a.shape):
        raise ValueError("dims must match input rank")
    shape = tuple(s * d for s, d in zip(a.shape, dims))
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register tile
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register tile
```

```python
# src/mindtorch_v2/_functional.py

def tile(a, dims):
    return dispatch("tile", a.device.type, a, dims)
```

```python
# src/mindtorch_v2/__init__.py
# export tile
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_tile_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_tile_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add tile"
```

---

### Task 4: Update coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Append rows for:
- `aten::repeat`
- `aten::repeat_interleave`
- `aten::tile`

**Step 2: Commit**

```bash
git add docs/plans/ops-coverage.md
git commit -m "docs: record repeat/repeat_interleave/tile"
```

---

### Task 5: Final verification

**Step 1: Run focused tests**

Run:
```bash
pytest \
  tests/mindtorch_v2/test_ops_cpu.py::test_repeat_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_repeat_interleave_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_tile_cpu \
  tests/mindtorch_v2/test_meta_device.py::test_meta_repeat_shape \
  tests/mindtorch_v2/test_meta_repeat_interleave_shape \
  tests/mindtorch_v2/test_meta_tile_shape -q
```
Expected: PASS.

**Step 2: Rebase and push**

```bash
git fetch ms
git rebase ms/master
git push --force-with-lease
```

**Step 3: Create PR**
- Target: `mindspore-lab/mindnlp`, base `master`.
- Ensure PR description uses proper newlines.

