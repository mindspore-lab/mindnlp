# Multi-Output Dispatch + Cummax Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-output dispatch support and implement `cumsum`/`cumprod`/`cummax` (CPU + meta + tests).

**Architecture:** Extend the dispatcher/pipeline to accept tuple/list outputs by allowing meta kernels to return a tuple of `TensorSpec` and backend kernels to return tuples of `Tensor`. Add CPU implementations using numpy (`cumsum`, `cumprod`) and a manual `cummax` along a dimension returning `(values, indices)`. Meta kernels mirror shapes/dtypes (values dtype = input dtype, indices dtype = int64). Functional/Tensor APIs expose these ops and tests validate CPU results, meta shapes, and pipeline multi-output behavior.

**Tech Stack:** Python, numpy, existing MindTorch v2 dispatch registry/pipeline.

---

### Task 1: Add CPU Tests For Cum Ops

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`

**Step 1: Write the failing test**

```python
import numpy as np


def test_cumsum_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = np.cumsum(x.numpy(), axis=1)
    np.testing.assert_allclose(torch.cumsum(x, dim=1).numpy(), expected)


def test_cumprod_cpu():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = np.cumprod(x.numpy(), axis=1)
    np.testing.assert_allclose(torch.cumprod(x, dim=1).numpy(), expected)


def test_cummax_cpu():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]])
    values, indices = torch.cummax(x, dim=1)
    expected_vals = np.maximum.accumulate(x.numpy(), axis=1)
    np.testing.assert_allclose(values.numpy(), expected_vals)
    expected_idx = np.array([[0, 1, 1], [0, 0, 2]], dtype=np.int64)
    np.testing.assert_array_equal(indices.numpy(), expected_idx)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_cumsum_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cumprod_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cummax_cpu -q`

Expected: FAIL with `AttributeError: module mindtorch_v2 has no attribute cumsum` (etc).

---

### Task 2: Add Meta Shape Tests For Cum Ops + Multi-Output

**Files:**
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write the failing test**

```python

def test_meta_cum_ops_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    out = torch.cumsum(x, dim=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape
    out = torch.cumprod(x, dim=1)
    assert out.device.type == "meta"
    assert out.shape == x.shape


def test_meta_cummax_shape():
    x = torch.tensor([[1.0, 2.0]], device="meta")
    values, indices = torch.cummax(x, dim=1)
    assert values.device.type == "meta"
    assert indices.device.type == "meta"
    assert values.shape == x.shape
    assert indices.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_cum_ops_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cummax_shape -q`

Expected: FAIL with missing ops or tuple-handling errors.

---

### Task 3: Add Pipeline Test For Multi-Output

**Files:**
- Modify: `tests/mindtorch_v2/test_dispatch_pipeline.py`

**Step 1: Write the failing test**

```python

def test_pipeline_handles_multi_output():
    with torch.pipeline() as pipe:
        x = torch.tensor([1.0, 2.0])
        values, indices = torch.cummax(x, dim=0)
        assert getattr(values, "_pending", False) is True
        assert getattr(indices, "_pending", False) is True
        pipe.flush()
        assert getattr(values, "_pending", False) is False
        assert getattr(indices, "_pending", False) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_handles_multi_output -q`

Expected: FAIL with meta/dispatcher tuple handling error.

---

### Task 4: Extend Dispatcher/Pipeline For Tuple Outputs

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_dispatch/pipeline.py`

**Step 1: Implement minimal tuple support**

- Allow meta kernel to return tuple/list of `TensorSpec`.
- Create pending outputs for each spec and return a tuple from `dispatch_with_keyset`.
- Update `_PendingOp.execute` to assign results to each pending output.
- Update `Pipeline.record` to accept tuple/list `pending` and track each.

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_handles_multi_output -q`

Expected: PASS.

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/_dispatch/pipeline.py tests/mindtorch_v2/test_dispatch_pipeline.py
git commit -m "feat: support multi-output dispatch"
```

---

### Task 5: Implement CPU Cum Ops + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`

**Step 1: Implement CPU kernels**

```python
# cpu/ops.py

def cumsum(a, dim=0):
    return _from_numpy(np.cumsum(_to_numpy(a), axis=dim), a.dtype, a.device)


def cumprod(a, dim=0):
    return _from_numpy(np.cumprod(_to_numpy(a), axis=dim), a.dtype, a.device)


def cummax(a, dim=0):
    arr = _to_numpy(a)
    if dim < 0:
        dim += arr.ndim
    moved = np.moveaxis(arr, dim, 0)
    values = np.empty_like(moved)
    indices = np.empty(moved.shape, dtype=np.int64)
    max_vals = moved[0].copy()
    values[0] = max_vals
    indices[0] = 0
    for i in range(1, moved.shape[0]):
        mask = moved[i] >= max_vals
        max_vals = np.where(mask, moved[i], max_vals)
        values[i] = max_vals
        indices[i] = np.where(mask, i, indices[i - 1])
    values = np.moveaxis(values, 0, dim)
    indices = np.moveaxis(indices, 0, dim)
    return (
        _from_numpy(values, a.dtype, a.device),
        _from_numpy(indices, int64_dtype, a.device),
    )
```

- Register `cumsum`, `cumprod`, `cummax` in CPU backend.

**Step 2: Implement meta kernels**

```python
# meta/ops.py

def _meta_cummax_meta(a, dim=0):
    return (
        _meta_tensor(a.shape, a.dtype, a.device),
        _meta_tensor(a.shape, int64_dtype, a.device),
    )
```

- Register `cumsum`/`cumprod` with `_meta_unary_meta` and `cummax` with `_meta_cummax_meta`.

**Step 3: Run tests**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_cumsum_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cumprod_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cummax_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_cum_ops_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cummax_shape -q`

Expected: PASS.

**Step 4: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/__init__.py tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py
git commit -m "feat: add cumsum/cumprod/cummax cpu ops"
```

---

### Task 6: Wire Functional/Tensor API + Exports + Docs

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Implement API wrappers**

```python
# _functional.py

def cumsum(a, dim=0):
    return dispatch("cumsum", a.device.type, a, dim)


def cumprod(a, dim=0):
    return dispatch("cumprod", a.device.type, a, dim)


def cummax(a, dim=0):
    return dispatch("cummax", a.device.type, a, dim)
```

```python
# _tensor.py

def cumsum(self, dim=0):
    return cumsum_dispatch(self, dim)


def cumprod(self, dim=0):
    return cumprod_dispatch(self, dim)


def cummax(self, dim=0):
    return cummax_dispatch(self, dim)
```

- Export in `__init__.py` and update `docs/plans/ops-coverage.md`.

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_cumsum_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cumprod_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cummax_cpu tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_handles_multi_output -q`

Expected: PASS.

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md
git commit -m "feat: expose cumsum/cumprod/cummax"
```

---

### Task 7: Full Verification

**Step 1: Run focused suite**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_cumsum_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cumprod_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cummax_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_cum_ops_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cummax_shape tests/mindtorch_v2/test_dispatch_pipeline.py::test_pipeline_handles_multi_output -q`

Expected: PASS.

