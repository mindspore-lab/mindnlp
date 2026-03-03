# Stack/Cat/Concat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `stack`, `cat`, and `concat` with full PyTorch parameter coverage including `out` handling.

**Architecture:** Use numpy for CPU implementations (`np.stack`, `np.concatenate`). Meta kernels compute output shapes and dtypes for meta device. Functional wrappers handle `out` by copying result tensors into provided outputs. Expose top-level APIs and update ops coverage.

**Tech Stack:** Python, numpy, existing MindTorch v2 dispatch + meta backend.

---

### Task 1: Add CPU Tests For stack/cat/concat (including out)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`

**Step 1: Write the failing tests**

```python

def test_stack_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    expected = np.stack([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(torch.stack([a, b], dim=0).numpy(), expected)


def test_cat_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    expected = np.concatenate([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(torch.cat([a, b], dim=0).numpy(), expected)


def test_concat_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    expected = np.concatenate([a.numpy(), b.numpy()], axis=1)
    np.testing.assert_allclose(torch.concat([a, b], dim=1).numpy(), expected)


def test_stack_out_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = torch.empty((2, 2))
    result = torch.stack([a, b], dim=0, out=out)
    assert result is out
    expected = np.stack([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(out.numpy(), expected)


def test_cat_out_cpu():
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0]])
    out = torch.empty((2, 2))
    result = torch.cat([a, b], dim=0, out=out)
    assert result is out
    expected = np.concatenate([a.numpy(), b.numpy()], axis=0)
    np.testing.assert_allclose(out.numpy(), expected)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cat_cpu tests/mindtorch_v2/test_ops_cpu.py::test_concat_cpu tests/mindtorch_v2/test_ops_cpu.py::test_stack_out_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cat_out_cpu -q`

Expected: FAIL with missing attribute errors.

---

### Task 2: Add Meta Shape Tests For stack/cat/concat

**Files:**
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write failing tests**

```python

def test_meta_stack_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0, 4.0], device="meta")
    out = torch.stack([a, b], dim=0)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_cat_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    out = torch.cat([a, b], dim=0)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_concat_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    out = torch.concat([a, b], dim=1)
    assert out.device.type == "meta"
    assert out.shape == (1, 4)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_stack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cat_shape tests/mindtorch_v2/test_meta_device.py::test_meta_concat_shape -q`

Expected: FAIL.

---

### Task 3: Implement CPU Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Implement CPU ops**

```python

def stack(tensors, dim=0):
    arrays = [ _to_numpy(t) for t in tensors ]
    return _from_numpy(np.stack(arrays, axis=dim), tensors[0].dtype, tensors[0].device)


def cat(tensors, dim=0):
    arrays = [ _to_numpy(t) for t in tensors ]
    return _from_numpy(np.concatenate(arrays, axis=dim), tensors[0].dtype, tensors[0].device)
```

- `concat` is an alias of `cat` (use same kernel).

**Step 2: Register ops**
- `stack` / `cat` / `concat` in CPU backend with meta infer helpers.

---

### Task 4: Implement Meta Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Meta kernels**

```python

def _meta_stack_meta(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    return _meta_tensor(tuple(shape), tensors[0].dtype, tensors[0].device)


def _meta_cat_meta(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape[dim] = sum(t.shape[dim] for t in tensors)
    return _meta_tensor(tuple(shape), tensors[0].dtype, tensors[0].device)
```

- `concat` uses `_meta_cat_meta`.

**Step 2: Register meta ops**
- `stack` -> `_meta_stack_meta`
- `cat`/`concat` -> `_meta_cat_meta`

---

### Task 5: Wire Functional/Tensor API + Exports + out handling

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Functional wrappers**

```python

def stack(tensors, dim=0, out=None):
    result = dispatch("stack", tensors[0].device.type, tensors, dim=dim)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def cat(tensors, dim=0, out=None):
    result = dispatch("cat", tensors[0].device.type, tensors, dim=dim)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def concat(tensors, dim=0, out=None):
    return cat(tensors, dim=dim, out=out)
```

- `Tensor` methods are not required.
- Export in `__init__.py` and update `docs/plans/ops-coverage.md`.

---

### Task 6: Run Tests

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_stack_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cat_cpu tests/mindtorch_v2/test_ops_cpu.py::test_concat_cpu tests/mindtorch_v2/test_ops_cpu.py::test_stack_out_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cat_out_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_stack_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cat_shape tests/mindtorch_v2/test_meta_concat_shape -q`

Expected: PASS.

---

### Task 7: Commit

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md
git commit -m "feat: add stack/cat/concat ops"
```
