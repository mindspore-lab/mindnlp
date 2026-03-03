# HStack/VStack/ColumnStack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `hstack`, `vstack`, and `column_stack` with PyTorch-compatible 1D behavior and `out` handling.

**Architecture:** CPU backend uses numpy for stacking/concatenation. `hstack` maps to `cat` (dim=0 for 1D, dim=1 for 2D+). `vstack` for 1D expands to (1, N) then cat dim=0; for 2D+ uses dim=0. `column_stack` for 1D expands to (N, 1) then cat dim=1; for 2D+ uses dim=1. Meta backend computes shapes accordingly. Functional wrappers handle `out` by copying results into provided tensor.

**Tech Stack:** Python, numpy, existing MindTorch v2 dispatch + meta backend.

---

### Task 1: Add CPU Tests For hstack/vstack/column_stack (including out)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`

**Step 1: Write failing tests**

```python

def test_hstack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.hstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.hstack([a, b]).numpy(), expected)


def test_vstack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.vstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.vstack([a, b]).numpy(), expected)


def test_column_stack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.column_stack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(torch.column_stack([a, b]).numpy(), expected)


def test_hstack_out_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = torch.empty((4,))
    result = torch.hstack([a, b], out=out)
    assert result is out
    expected = np.hstack([a.numpy(), b.numpy()])
    np.testing.assert_allclose(out.numpy(), expected)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_hstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_vstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_column_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_hstack_out_cpu -q`

Expected: FAIL with missing attribute errors.

---

### Task 2: Add Meta Shape Tests

**Files:**
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write failing tests**

```python

def test_meta_hstack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.hstack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (4,)


def test_meta_vstack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.vstack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_column_stack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.column_stack([a, b])
    assert out.device.type == "meta"
    assert out.shape == (2, 2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_hstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_vstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_column_stack_shape -q`

Expected: FAIL.

---

### Task 3: Implement CPU Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Implement CPU ops**

```python

def hstack(tensors):
    if tensors[0].dim() == 1:
        return cat(tensors, dim=0)
    return cat(tensors, dim=1)


def vstack(tensors):
    if tensors[0].dim() == 1:
        expanded = [t.reshape((1, t.shape[0])) for t in tensors]
        return cat(expanded, dim=0)
    return cat(tensors, dim=0)


def column_stack(tensors):
    if tensors[0].dim() == 1:
        expanded = [t.reshape((t.shape[0], 1)) for t in tensors]
        return cat(expanded, dim=1)
    return cat(tensors, dim=1)
```

**Step 2: Register ops**
- `hstack`, `vstack`, `column_stack` in CPU backend with meta infer helpers.

---

### Task 4: Implement Meta Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Meta kernels**

```python

def _meta_hstack_meta(tensors):
    if len(tensors[0].shape) == 1:
        shape = (sum(t.shape[0] for t in tensors),)
    else:
        shape = list(tensors[0].shape)
        shape[1] = sum(t.shape[1] for t in tensors)
        shape = tuple(shape)
    return _meta_tensor(shape, tensors[0].dtype, tensors[0].device)


def _meta_vstack_meta(tensors):
    if len(tensors[0].shape) == 1:
        shape = (len(tensors), tensors[0].shape[0])
    else:
        shape = list(tensors[0].shape)
        shape[0] = sum(t.shape[0] for t in tensors)
        shape = tuple(shape)
    return _meta_tensor(shape, tensors[0].dtype, tensors[0].device)


def _meta_column_stack_meta(tensors):
    if len(tensors[0].shape) == 1:
        shape = (tensors[0].shape[0], len(tensors))
    else:
        shape = list(tensors[0].shape)
        shape[1] = sum(t.shape[1] for t in tensors)
        shape = tuple(shape)
    return _meta_tensor(shape, tensors[0].dtype, tensors[0].device)
```

**Step 2: Register meta ops**
- `hstack` -> `_meta_hstack_meta`
- `vstack` -> `_meta_vstack_meta`
- `column_stack` -> `_meta_column_stack_meta`

---

### Task 5: Wire Functional API + Exports + out handling

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Functional wrappers**

```python

def hstack(tensors, out=None):
    result = dispatch("hstack", tensors[0].device.type, tensors)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def vstack(tensors, out=None):
    result = dispatch("vstack", tensors[0].device.type, tensors)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def column_stack(tensors, out=None):
    result = dispatch("column_stack", tensors[0].device.type, tensors)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result
```

- Export in `__init__.py` and update `docs/plans/ops-coverage.md`.

---

### Task 6: Run Tests

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_hstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_vstack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_column_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_hstack_out_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_hstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_vstack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_column_stack_shape -q`

Expected: PASS.

---

### Task 7: Commit

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md
git commit -m "feat: add hstack/vstack/column_stack ops"
```
