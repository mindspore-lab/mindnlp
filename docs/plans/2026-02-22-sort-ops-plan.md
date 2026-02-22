# Sort/Argsort/Topk Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `argsort`, `sort`, and `topk` with full PyTorch parameter coverage including `out` handling and multi-output support.

**Architecture:** Reuse the multi-output dispatch path (tuple outputs) for `sort`/`topk`, returning `(values, indices)` tensors. CPU backend uses numpy for sorting/indexing. Meta backend returns shape/dtype specs for values and indices. `out` is handled in functional APIs by copying results into provided tensors (CPU-only) and returning those out tensors.

**Tech Stack:** Python, numpy, existing MindTorch v2 dispatch + pipeline.

---

### Task 1: Add CPU Tests For sort/argsort/topk (including out)

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`

**Step 1: Write the failing tests**

```python

def test_argsort_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    expected = np.argsort(x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argsort(x, dim=1).numpy(), expected)
    expected_desc = np.argsort(-x.numpy(), axis=1)
    np.testing.assert_array_equal(torch.argsort(x, dim=1, descending=True).numpy(), expected_desc)


def test_sort_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    values, indices = torch.sort(x, dim=1)
    expected_indices = np.argsort(x.numpy(), axis=1)
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.numpy(), expected_values)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)


def test_topk_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0], [0.0, -1.0, 5.0]])
    values, indices = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
    expected_indices = np.argsort(-x.numpy(), axis=1)[:, :2]
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(values.numpy(), expected_values)
    np.testing.assert_array_equal(indices.numpy(), expected_indices)


def test_sort_out_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0]])
    out_values = torch.empty((1, 3))
    out_indices = torch.empty((1, 3), dtype=torch.int64)
    values, indices = torch.sort(x, dim=1, out=(out_values, out_indices))
    assert values is out_values
    assert indices is out_indices
    expected_indices = np.argsort(x.numpy(), axis=1)
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(out_values.numpy(), expected_values)
    np.testing.assert_array_equal(out_indices.numpy(), expected_indices)


def test_topk_out_cpu():
    x = torch.tensor([[3.0, 1.0, 2.0]])
    out_values = torch.empty((1, 2))
    out_indices = torch.empty((1, 2), dtype=torch.int64)
    values, indices = torch.topk(x, k=2, dim=1, out=(out_values, out_indices))
    assert values is out_values
    assert indices is out_indices
    expected_indices = np.argsort(-x.numpy(), axis=1)[:, :2]
    expected_values = np.take_along_axis(x.numpy(), expected_indices, axis=1)
    np.testing.assert_allclose(out_values.numpy(), expected_values)
    np.testing.assert_array_equal(out_indices.numpy(), expected_indices)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_argsort_cpu tests/mindtorch_v2/test_ops_cpu.py::test_sort_cpu tests/mindtorch_v2/test_ops_cpu.py::test_topk_cpu tests/mindtorch_v2/test_ops_cpu.py::test_sort_out_cpu tests/mindtorch_v2/test_ops_cpu.py::test_topk_out_cpu -q`

Expected: FAIL with missing attribute errors.

---

### Task 2: Add Meta Shape Tests For sort/argsort/topk

**Files:**
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write failing tests**

```python

def test_meta_sort_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    values, indices = torch.sort(x, dim=1)
    assert values.device.type == "meta"
    assert indices.device.type == "meta"
    assert values.shape == x.shape
    assert indices.shape == x.shape


def test_meta_argsort_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    out = torch.argsort(x, dim=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape


def test_meta_topk_shape():
    x = torch.tensor([[1.0, 2.0, 3.0]], device="meta")
    values, indices = torch.topk(x, k=2, dim=1)
    assert values.device.type == "meta"
    assert indices.device.type == "meta"
    assert values.shape == (1, 2)
    assert indices.shape == (1, 2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_sort_shape tests/mindtorch_v2/test_meta_device.py::test_meta_argsort_shape tests/mindtorch_v2/test_meta_device.py::test_meta_topk_shape -q`

Expected: FAIL.

---

### Task 3: Implement CPU Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Implement CPU ops**

```python

def argsort(a, dim=-1, descending=False):
    arr = _to_numpy(a)
    if descending:
        idx = np.argsort(-arr, axis=dim)
    else:
        idx = np.argsort(arr, axis=dim)
    return _from_numpy(idx.astype(np.int64), int64_dtype, a.device)


def sort(a, dim=-1, descending=False, stable=False):
    arr = _to_numpy(a)
    kind = "stable" if stable else "quicksort"
    if descending:
        idx = np.argsort(-arr, axis=dim, kind=kind)
    else:
        idx = np.argsort(arr, axis=dim, kind=kind)
    values = np.take_along_axis(arr, idx, axis=dim)
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
    )


def topk(a, k, dim=-1, largest=True, sorted=True):
    arr = _to_numpy(a)
    if largest:
        idx = np.argsort(-arr, axis=dim)
    else:
        idx = np.argsort(arr, axis=dim)
    idx = np.take(idx, np.arange(k), axis=dim)
    values = np.take_along_axis(arr, idx, axis=dim)
    if not sorted:
        return (
            _from_numpy(values, a.dtype, a.device),
            _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
        )
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(idx.astype(np.int64), int64_dtype, a.device),
    )
```

**Step 2: Register ops**
- `argsort` returns indices (`infer_argmax` shape, int64 dtype)
- `sort` returns `(values, indices)`
- `topk` returns `(values, indices)`

---

### Task 4: Implement Meta Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Meta kernels**

```python

def _meta_argsort_meta(a, dim=-1, descending=False, stable=False):
    return _meta_tensor(a.shape, int64_dtype, a.device)


def _meta_sort_meta(a, dim=-1, descending=False, stable=False):
    return (
        _meta_tensor(a.shape, a.dtype, a.device),
        _meta_tensor(a.shape, int64_dtype, a.device),
    )


def _meta_topk_meta(a, k, dim=-1, largest=True, sorted=True):
    shape = list(a.shape)
    shape[dim] = k
    shape = tuple(shape)
    return (
        _meta_tensor(shape, a.dtype, a.device),
        _meta_tensor(shape, int64_dtype, a.device),
    )
```

**Step 2: Register meta ops**
- `argsort` -> `_meta_argsort_meta`
- `sort` -> `_meta_sort_meta`
- `topk` -> `_meta_topk_meta`

---

### Task 5: Wire Functional/Tensor API + Exports + out handling

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Functional wrappers**

```python

def argsort(a, dim=-1, descending=False, stable=False, out=None):
    result = dispatch("argsort", a.device.type, a, dim=dim, descending=descending, stable=stable)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def sort(a, dim=-1, descending=False, stable=False, out=None):
    values, indices = dispatch("sort", a.device.type, a, dim=dim, descending=descending, stable=stable)
    if out is not None:
        out_values, out_indices = out
        out_values._storage = values.storage()
        out_values.shape = values.shape
        out_values.stride = values.stride
        out_values.offset = values.offset
        out_indices._storage = indices.storage()
        out_indices.shape = indices.shape
        out_indices.stride = indices.stride
        out_indices.offset = indices.offset
        return out_values, out_indices
    return values, indices


def topk(a, k, dim=-1, largest=True, sorted=True, out=None):
    values, indices = dispatch("topk", a.device.type, a, k, dim=dim, largest=largest, sorted=sorted)
    if out is not None:
        out_values, out_indices = out
        out_values._storage = values.storage()
        out_values.shape = values.shape
        out_values.stride = values.stride
        out_values.offset = values.offset
        out_indices._storage = indices.storage()
        out_indices.shape = indices.shape
        out_indices.stride = indices.stride
        out_indices.offset = indices.offset
        return out_values, out_indices
    return values, indices
```

**Step 2: Tensor methods + exports**
- `Tensor.argsort`, `Tensor.sort`, `Tensor.topk` dispatch to functional.
- Export in `__init__.py`.
- Update `docs/plans/ops-coverage.md` with `aten::argsort`, `aten::sort`, `aten::topk`.

---

### Task 6: Run Tests

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_argsort_cpu tests/mindtorch_v2/test_ops_cpu.py::test_sort_cpu tests/mindtorch_v2/test_ops_cpu.py::test_topk_cpu tests/mindtorch_v2/test_ops_cpu.py::test_sort_out_cpu tests/mindtorch_v2/test_ops_cpu.py::test_topk_out_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_sort_shape tests/mindtorch_v2/test_meta_device.py::test_meta_argsort_shape tests/mindtorch_v2/test_meta_device.py::test_meta_topk_shape -q`

Expected: PASS.

---

### Task 7: Commit

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md
git commit -m "feat: add sort/argsort/topk ops"
```
