# Autograd In-place + View Semantics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement torch-aligned autograd in-place semantics with version counters, view tracking, and saved-tensor version validation for CPU and NPU.

**Architecture:** Add a shared `VersionCounter` object per base tensor, attach view metadata to view tensors, and enforce in-place checks before mutation. Saved tensors record the version at save time and validate at backward. In-place ops update the shared counter.

**Tech Stack:** Python, numpy, existing mindtorch_v2 autograd engine.

---

### Task 1: Add version counter + view metadata to Tensor

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_autograd_inplace.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch
import pytest


def test_view_shares_version_counter():
    base = torch.tensor([1.0, 2.0], requires_grad=True)
    view = base.view((2,))
    assert base._version_counter is view._version_counter
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py::test_view_shares_version_counter -v`
Expected: FAIL (attribute missing).

**Step 3: Write minimal implementation**

```python
class VersionCounter:
    def __init__(self, value=0):
        self.value = int(value)


class Tensor:
    def __init__(...):
        self._version_counter = VersionCounter()
        self._base = None
        self._view_meta = None
```

Update view ops in `_backends/common/view.py` to set `_base`, `_view_meta`, and share `_version_counter` with base.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py::test_view_shares_version_counter -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_backends/common/view.py tests/mindtorch_v2/test_autograd_inplace.py
git commit -m "Add version counter and view metadata"
```

---

### Task 2: Enforce in-place checks and bump version

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/cpu/ops.py`
- Test: `tests/mindtorch_v2/test_autograd_inplace.py`

**Step 1: Write failing tests**

```python
import mindtorch_v2 as torch
import pytest


def test_inplace_on_leaf_raises():
    t = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(RuntimeError):
        t.add_(1.0)


def test_inplace_on_view_of_leaf_raises():
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    v = t.view((2,))
    with pytest.raises(RuntimeError):
        v.relu_()


def test_inplace_increments_version():
    t = torch.tensor([1.0])
    v0 = t._version_counter.value
    t.add_(1.0)
    assert t._version_counter.value == v0 + 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py::test_inplace_on_leaf_raises -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Implement `Tensor._check_inplace()` and `Tensor._bump_version()` and call them from in-place ops. Add in-place ops to Tensor: `add_`, `mul_`, `relu_`, `zero_`. Update CPU/NPU backends to provide in-place kernels where available; otherwise do a temp + copy_ (non-inplace fallback) but still treat it as in-place for versioning.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py::test_inplace_on_leaf_raises -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/cpu/ops.py tests/mindtorch_v2/test_autograd_inplace.py
git commit -m "Enforce in-place autograd checks"
```

---

### Task 3: Saved tensor version validation

**Files:**
- Modify: `src/mindtorch_v2/_autograd/node.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Test: `tests/mindtorch_v2/test_autograd_inplace.py`

**Step 1: Write failing test**

```python
import mindtorch_v2 as torch
import pytest


def test_saved_tensor_version_mismatch_raises():
    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b = a.relu()  # saves for backward
    a.add_(1.0)   # in-place mutation should invalidate
    with pytest.raises(RuntimeError):
        b.sum().backward()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py::test_saved_tensor_version_mismatch_raises -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add `SavedTensor` to `node.py` storing tensor + saved version value. Add `Node.save_for_backward()` and `Node.saved_tensors()` that checks version equality and raises if changed. Update relevant backward closures in `_functional.py` to use `save_for_backward` for inputs they need (relu, mul, matmul, sum, view/transpose if needed).

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py::test_saved_tensor_version_mismatch_raises -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_autograd/node.py src/mindtorch_v2/_functional.py tests/mindtorch_v2/test_autograd_inplace.py
git commit -m "Validate saved tensor versions in backward"
```

---

### Task 4: Full autograd inplace + view test sweep

**Files:**
- Test: `tests/mindtorch_v2/test_autograd_inplace.py`

**Step 1: Add NPU coverage test (skip if no NPU)**

```python
import pytest
import mindtorch_v2 as torch


def test_inplace_npu_versioning():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    t = torch.tensor([1.0, 2.0], device="npu")
    v0 = t._version_counter.value
    t.relu_()
    assert t._version_counter.value == v0 + 1
```

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_autograd_inplace.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/mindtorch_v2/test_autograd_inplace.py
git commit -m "Add autograd inplace NPU coverage"
```

---

### Task 5: Full test suite

**Step 1: Run full suite**

Run: `pytest -q tests/mindtorch_v2`
Expected: PASS

**Step 2: Commit (if needed)**

```bash
git status -sb
```

---

Plan complete and saved to `docs/plans/2026-02-14-mindtorch-v2-autograd-inplace-plan.md`.

Two execution options:
1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks.
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints.

Which approach?
