# Meta Device Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `device="meta"` tensors with metadata-only storage and meta kernels, plus materialization via `to()`.

**Architecture:** Introduce `MetaStorage`, register meta backend kernels, update device handling and tensor creation, and enforce errors on data access for meta tensors.

**Tech Stack:** Python, pytest

---

### Task 1: MetaStorage and data-access errors

**Files:**
- Modify: `src/mindtorch_v2/_storage.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_meta_device.py::test_meta_tensor_blocks_numpy`

**Step 1: Write the failing test**

```python
import pytest
import mindtorch_v2 as torch


def test_meta_tensor_blocks_numpy():
    t = torch.tensor([1.0, 2.0], device="meta")
    with pytest.raises(RuntimeError, match="meta tensor has no data"):
        _ = t.numpy()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_tensor_blocks_numpy -v`

Expected: FAIL because meta device is not implemented yet.

**Step 3: Write minimal implementation**

Add `MetaStorage` in `src/mindtorch_v2/_storage.py`:

```python
class MetaStorage:
    def __init__(self, shape, device=None, dtype=None):
        self.shape = tuple(shape)
        self.device = device or Device("meta")
        self.dtype = dtype or float32
        self.data = None

    def to(self, device):
        raise RuntimeError("meta tensor has no data")
```

Update `Tensor.numpy()` and `_numpy_view()` to raise `RuntimeError("meta tensor has no data")` when `device.type == "meta"`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_tensor_blocks_numpy -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_meta_device.py
git commit -m "feat: add meta storage and block data access"
```

---

### Task 2: Create meta tensors and materialize with to()

**Files:**
- Modify: `src/mindtorch_v2/_creation.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_meta_device.py::test_meta_to_cpu_materializes`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_meta_to_cpu_materializes():
    t = torch.tensor([1.0, 2.0], device="meta")
    out = t.to("cpu")
    assert out.device.type == "cpu"
    assert out.shape == (2,)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_to_cpu_materializes -v`

Expected: FAIL because meta to cpu not implemented.

**Step 3: Write minimal implementation**

In `Tensor.to()`, if `self.device.type == "meta"` and target is cpu/npu, allocate empty storage with correct shape/dtype:

```python
if self.device.type == "meta":
    arr = np.empty(self.shape, dtype=to_numpy_dtype(self.dtype))
    storage = Storage(arr, device=dev, dtype=self.dtype)
    return Tensor(storage, self.shape, self.stride, self.offset, self.requires_grad)
```

Update `tensor/zeros/ones` to accept `device="meta"` and return a meta tensor (no data allocation).

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_to_cpu_materializes -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_creation.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_meta_device.py
git commit -m "feat: materialize meta tensors with to()"
```

---

### Task 3: Meta backend kernels

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Test: `tests/mindtorch_v2/test_meta_device.py::test_meta_ops_shape_propagation`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_meta_ops_shape_propagation():
    a = torch.tensor([[1.0, 2.0]], device="meta")
    b = torch.tensor([[3.0, 4.0]], device="meta")
    c = a + b
    d = c.relu()
    e = d.sum()
    assert c.device.type == "meta"
    assert d.device.type == "meta"
    assert e.device.type == "meta"
    assert c.shape == a.shape
    assert e.shape == ()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_ops_shape_propagation -v`

Expected: FAIL until meta kernels are registered.

**Step 3: Write minimal implementation**

Register meta kernels for `add/mul/relu/sum` in the dispatcher (using existing meta helpers). Ensure dispatch routes to meta backend when any input device is meta.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_meta_device.py::test_meta_ops_shape_propagation -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_dispatch/registry.py src/mindtorch_v2/_dispatch/dispatcher.py tests/mindtorch_v2/test_meta_device.py
git commit -m "feat: add meta backend kernels"
```
