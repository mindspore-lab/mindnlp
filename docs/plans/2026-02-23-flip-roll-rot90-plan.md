# Flip, Roll, Rot90 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement CPU/meta support for `flip`, `roll`, and `rot90` with torch-aligned behavior, using TDD.

**Architecture:** Use NumPy (`np.flip`, `np.roll`, `np.rot90`) for CPU kernels and meta shape inference for `rot90` axis swap and shape preservation. Register ops in CPU/meta backends and update coverage docs.

**Tech Stack:** Python, NumPy, PyTest, MindTorch v2 dispatch.

---

### Task 1: Add failing CPU/meta tests for flip

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_flip_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    out = torch.flip(x, dims=(0,))
    expected = np.flip(x.numpy(), axis=0)
    np.testing.assert_array_equal(out.numpy(), expected)

    out = torch.flip(x, dims=(0, 1))
    expected = np.flip(x.numpy(), axis=(0, 1))
    np.testing.assert_array_equal(out.numpy(), expected)
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_flip_shape():
    x = torch.empty((2, 3, 4), device="meta")
    out = torch.flip(x, dims=(0, 2))
    assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_flip_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_flip_shape -q`

Expected: FAIL because `flip` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def flip(a, dims):
    arr = _to_numpy(a)
    out = np.flip(arr, axis=dims)
    return _from_numpy(out, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_flip_meta(a, dims):
    return _meta_tensor(a.shape, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_flip(a, dims):
    return TensorSpec(shape=tuple(a.shape), stride=_contiguous_stride(a.shape), dtype=a.dtype)
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register flip
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register flip
```

```python
# src/mindtorch_v2/_functional.py

def flip(a, dims):
    return dispatch("flip", a.device.type, a, dims)
```

```python
# src/mindtorch_v2/__init__.py
# export flip
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_flip_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_flip_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add flip"
```

---

### Task 2: Add failing CPU/meta tests for roll

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_roll_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    out = torch.roll(x, shifts=1, dims=0)
    expected = np.roll(x.numpy(), shift=1, axis=0)
    np.testing.assert_array_equal(out.numpy(), expected)

    out = torch.roll(x, shifts=(1, -1), dims=(0, 1))
    expected = np.roll(x.numpy(), shift=(1, -1), axis=(0, 1))
    np.testing.assert_array_equal(out.numpy(), expected)
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_roll_shape():
    x = torch.empty((2, 3, 4), device="meta")
    out = torch.roll(x, shifts=1, dims=2)
    assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_roll_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_roll_shape -q`

Expected: FAIL because `roll` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def roll(a, shifts, dims=None):
    arr = _to_numpy(a)
    out = np.roll(arr, shift=shifts, axis=dims)
    return _from_numpy(out, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_roll_meta(a, shifts, dims=None):
    return _meta_tensor(a.shape, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_roll(a, shifts, dims=None):
    return TensorSpec(shape=tuple(a.shape), stride=_contiguous_stride(a.shape), dtype=a.dtype)
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register roll
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register roll
```

```python
# src/mindtorch_v2/_functional.py

def roll(a, shifts, dims=None):
    return dispatch("roll", a.device.type, a, shifts, dims)
```

```python
# src/mindtorch_v2/__init__.py
# export roll
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_roll_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_roll_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add roll"
```

---

### Task 3: Add failing CPU/meta tests for rot90

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py

def test_rot90_cpu():
    x = torch.tensor([[1, 2], [3, 4]])
    out = torch.rot90(x, k=1, dims=(0, 1))
    expected = np.rot90(x.numpy(), k=1, axes=(0, 1))
    np.testing.assert_array_equal(out.numpy(), expected)

    out = torch.rot90(x, k=-1, dims=(0, 1))
    expected = np.rot90(x.numpy(), k=-1, axes=(0, 1))
    np.testing.assert_array_equal(out.numpy(), expected)
```

```python
# tests/mindtorch_v2/test_meta_device.py

def test_meta_rot90_shape():
    x = torch.empty((2, 3, 4), device="meta")
    out = torch.rot90(x, k=1, dims=(0, 2))
    assert out.shape == (4, 3, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_rot90_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_rot90_shape -q`

Expected: FAIL because `rot90` is missing.

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu/ops.py

def rot90(a, k=1, dims=(0, 1)):
    arr = _to_numpy(a)
    out = np.rot90(arr, k=k, axes=dims)
    return _from_numpy(out, a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/ops.py

def _meta_rot90_meta(a, k=1, dims=(0, 1)):
    dim0, dim1 = dims
    shape = list(a.shape)
    if k % 2 != 0:
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    return _meta_tensor(tuple(shape), a.dtype, a.device)
```

```python
# src/mindtorch_v2/_backends/meta/infer.py

def infer_rot90(a, k=1, dims=(0, 1)):
    dim0, dim1 = dims
    shape = list(a.shape)
    if k % 2 != 0:
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    shape = tuple(shape)
    return TensorSpec(shape=shape, stride=_contiguous_stride(shape), dtype=a.dtype)
```

```python
# src/mindtorch_v2/_backends/meta/__init__.py
# register rot90
```

```python
# src/mindtorch_v2/_backends/cpu/__init__.py
# register rot90
```

```python
# src/mindtorch_v2/_functional.py

def rot90(a, k=1, dims=(0, 1)):
    return dispatch("rot90", a.device.type, a, k, dims)
```

```python
# src/mindtorch_v2/__init__.py
# export rot90
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_rot90_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_rot90_shape -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py
git commit -m "feat: add rot90"
```

---

### Task 4: Update coverage doc

**Files:**
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Add entries**

Append rows for:
- `aten::flip`
- `aten::roll`
- `aten::rot90`

**Step 2: Commit**

```bash
git add docs/plans/ops-coverage.md
git commit -m "docs: record flip/roll/rot90"
```

---

### Task 5: Final verification

**Step 1: Run focused tests**

Run:
```bash
pytest \
  tests/mindtorch_v2/test_ops_cpu.py::test_flip_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_roll_cpu \
  tests/mindtorch_v2/test_ops_cpu.py::test_rot90_cpu \
  tests/mindtorch_v2/test_meta_device.py::test_meta_flip_shape \
  tests/mindtorch_v2/test_meta_device.py::test_meta_roll_shape \
  tests/mindtorch_v2/test_meta_device.py::test_meta_rot90_shape -q
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

