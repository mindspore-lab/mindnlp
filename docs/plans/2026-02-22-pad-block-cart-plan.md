# PadSequence/BlockDiag/CartesianProd Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `pad_sequence`, `block_diag`, and `cartesian_prod` with PyTorch-compatible behavior and meta support.

**Architecture:** CPU backend uses numpy to build outputs (pad/concat/meshgrid). Meta backend computes shapes only. Functional wrappers expose ops at top-level and update ops coverage. No `out` parameters (PyTorch does not expose them here).

**Tech Stack:** Python, numpy, existing MindTorch v2 dispatch + meta backend.

---

### Task 1: Add CPU Tests

**Files:**
- Modify: `tests/mindtorch_v2/test_ops_cpu.py`

**Step 1: Write failing tests**

```python

def test_pad_sequence_cpu_right():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0])
    out = torch.pad_sequence([a, b], batch_first=True, padding_value=0.0, padding_side="right")
    expected = np.array([[1.0, 2.0], [3.0, 0.0]])
    np.testing.assert_allclose(out.numpy(), expected)


def test_pad_sequence_cpu_left():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0])
    out = torch.pad_sequence([a, b], batch_first=True, padding_value=-1.0, padding_side="left")
    expected = np.array([[1.0, 2.0], [-1.0, 3.0]])
    np.testing.assert_allclose(out.numpy(), expected)


def test_block_diag_cpu():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0]])
    out = torch.block_diag(a, b)
    expected = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    np.testing.assert_allclose(out.numpy(), expected)


def test_cartesian_prod_cpu():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    out = torch.cartesian_prod(a, b)
    expected = np.array([[1.0, 3.0], [1.0, 4.0], [2.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(out.numpy(), expected)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_pad_sequence_cpu_right tests/mindtorch_v2/test_ops_cpu.py::test_pad_sequence_cpu_left tests/mindtorch_v2/test_ops_cpu.py::test_block_diag_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cartesian_prod_cpu -q`

Expected: FAIL with missing attribute errors.

---

### Task 2: Add Meta Shape Tests

**Files:**
- Modify: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write failing tests**

```python

def test_meta_pad_sequence_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0], device="meta")
    out = torch.pad_sequence([a, b], batch_first=True)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)


def test_meta_block_diag_shape():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0]], device="meta")
    out = torch.block_diag(a, b)
    assert out.device.type == "meta"
    assert out.shape == (2, 3)


def test_meta_cartesian_prod_shape():
    a = torch.tensor([1.0, 2.0], device="meta")
    b = torch.tensor([3.0], device="meta")
    out = torch.cartesian_prod(a, b)
    assert out.device.type == "meta"
    assert out.shape == (2, 2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_pad_sequence_shape tests/mindtorch_v2/test_meta_device.py::test_meta_block_diag_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cartesian_prod_shape -q`

Expected: FAIL.

---

### Task 3: Implement CPU Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/__init__.py`

**Step 1: Implement CPU ops**

```python

def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    arrays = [_to_numpy(t) for t in seqs]
    max_len = max(a.shape[0] for a in arrays)
    batch = len(arrays)
    trailing = arrays[0].shape[1:]
    out_shape = (batch, max_len, *trailing) if batch_first else (max_len, batch, *trailing)
    out = np.full(out_shape, padding_value, dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        length = a.shape[0]
        if padding_side == "left":
            start = max_len - length
        else:
            start = 0
        if batch_first:
            out[i, start:start + length, ...] = a
        else:
            out[start:start + length, i, ...] = a
    return _from_numpy(out, seqs[0].dtype, seqs[0].device)


def block_diag(*tensors):
    arrays = [_to_numpy(t) for t in tensors]
    rows = sum(a.shape[0] for a in arrays)
    cols = sum(a.shape[1] for a in arrays)
    out = np.zeros((rows, cols), dtype=arrays[0].dtype)
    r = c = 0
    for a in arrays:
        out[r:r + a.shape[0], c:c + a.shape[1]] = a
        r += a.shape[0]
        c += a.shape[1]
    return _from_numpy(out, tensors[0].dtype, tensors[0].device)


def cartesian_prod(*tensors):
    arrays = [_to_numpy(t) for t in tensors]
    grids = np.meshgrid(*arrays, indexing="ij")
    stacked = np.stack([g.reshape(-1) for g in grids], axis=1)
    return _from_numpy(stacked, tensors[0].dtype, tensors[0].device)
```

**Step 2: Register ops**
- `pad_sequence`, `block_diag`, `cartesian_prod` in CPU backend with meta infer helpers.

---

### Task 4: Implement Meta Kernels + Register

**Files:**
- Modify: `src/mindtorch_v2/_backends/meta/ops.py`
- Modify: `src/mindtorch_v2/_backends/meta/__init__.py`
- Modify: `src/mindtorch_v2/_backends/meta/infer.py`

**Step 1: Meta kernels**

```python

def _meta_pad_sequence_meta(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    max_len = max(t.shape[0] for t in seqs)
    batch = len(seqs)
    trailing = seqs[0].shape[1:]
    shape = (batch, max_len, *trailing) if batch_first else (max_len, batch, *trailing)
    return _meta_tensor(tuple(shape), seqs[0].dtype, seqs[0].device)


def _meta_block_diag_meta(*tensors):
    rows = sum(t.shape[0] for t in tensors)
    cols = sum(t.shape[1] for t in tensors)
    return _meta_tensor((rows, cols), tensors[0].dtype, tensors[0].device)


def _meta_cartesian_prod_meta(*tensors):
    rows = 1
    for t in tensors:
        rows *= t.shape[0]
    cols = len(tensors)
    return _meta_tensor((rows, cols), tensors[0].dtype, tensors[0].device)
```

**Step 2: Register meta ops**
- `pad_sequence` -> `_meta_pad_sequence_meta`
- `block_diag` -> `_meta_block_diag_meta`
- `cartesian_prod` -> `_meta_cartesian_prod_meta`

---

### Task 5: Wire Functional API + Exports + Docs

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Modify: `docs/plans/ops-coverage.md`

**Step 1: Functional wrappers**

```python

def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    return dispatch("pad_sequence", seqs[0].device.type, seqs, batch_first=batch_first, padding_value=padding_value, padding_side=padding_side)


def block_diag(*tensors):
    return dispatch("block_diag", tensors[0].device.type, *tensors)


def cartesian_prod(*tensors):
    return dispatch("cartesian_prod", tensors[0].device.type, *tensors)
```

- Export in `__init__.py` and update `docs/plans/ops-coverage.md`.

---

### Task 6: Run Tests

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_pad_sequence_cpu_right tests/mindtorch_v2/test_ops_cpu.py::test_pad_sequence_cpu_left tests/mindtorch_v2/test_ops_cpu.py::test_block_diag_cpu tests/mindtorch_v2/test_ops_cpu.py::test_cartesian_prod_cpu tests/mindtorch_v2/test_meta_device.py::test_meta_pad_sequence_shape tests/mindtorch_v2/test_meta_device.py::test_meta_block_diag_shape tests/mindtorch_v2/test_meta_device.py::test_meta_cartesian_prod_shape -q`

Expected: PASS.

---

### Task 7: Commit

```bash
git add tests/mindtorch_v2/test_ops_cpu.py tests/mindtorch_v2/test_meta_device.py src/mindtorch_v2/_backends/cpu/ops.py src/mindtorch_v2/_backends/cpu/__init__.py src/mindtorch_v2/_backends/meta/ops.py src/mindtorch_v2/_backends/meta/__init__.py src/mindtorch_v2/_backends/meta/infer.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py docs/plans/ops-coverage.md
git commit -m "feat: add pad_sequence/block_diag/cartesian_prod ops"
```
