# Dispatch Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Route creation, view, and to() operations through dispatcher so autograd/device/pipeline are unified.

**Architecture:** Creation/view/to are moved into `_functional` and dispatched via registry. View ops register as storage-sharing ops with autograd view grad_fn. `to` becomes a dispatched op using a shared conversion helper.

**Tech Stack:** Python, pytest

---

### Task 1: Creation ops go through dispatch

**Files:**
- Modify: `src/mindtorch_v2/_creation.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Create: `tests/mindtorch_v2/test_dispatch_unify.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2._creation as creation
import mindtorch_v2._dispatch.dispatcher as dispatcher
from mindtorch_v2._storage import typed_storage_from_numpy
from mindtorch_v2._tensor import Tensor
import numpy as np


def test_tensor_creation_dispatch(monkeypatch):
    called = {}

    def fake_dispatch(name, device, *args, **kwargs):
        called["name"] = name
        storage = typed_storage_from_numpy(np.array([1.0], dtype=np.float32), None)
        return Tensor(storage, (1,), (1,))

    monkeypatch.setattr(dispatcher, "dispatch", fake_dispatch)
    _ = creation.tensor([1.0])
    assert called["name"] == "tensor"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_unify.py::test_tensor_creation_dispatch -v`

Expected: FAIL because creation does not call dispatch.

**Step 3: Write minimal implementation**

- Add creation ops to `_functional` (e.g., `tensor`, `zeros`, `ones`, `empty`) that call `dispatch`.
- Update `_creation.py` to delegate to `_functional` instead of creating tensors directly.
- Register `tensor/zeros/ones/empty` for cpu/npu/meta backends (use existing typed storage helpers).

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_unify.py::test_tensor_creation_dispatch -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_creation.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_dispatch/registry.py tests/mindtorch_v2/test_dispatch_unify.py
git commit -m "feat: dispatch creation ops"
```

---

### Task 2: View ops dispatch + autograd

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Create: `src/mindtorch_v2/_backends/view.py`
- Test: `tests/mindtorch_v2/test_view_dispatch.py::test_reshape_autograd`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_reshape_autograd():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    y = x.reshape((4,))
    z = y.sum()
    z.backward()
    assert x.grad.shape == x.shape
    assert x.grad.numpy().tolist() == [[1.0, 1.0], [1.0, 1.0]]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_view_dispatch.py::test_reshape_autograd -v`

Expected: FAIL because view ops don’t dispatch/autograd.

**Step 3: Write minimal implementation**

- Add `reshape/view/transpose` to `_functional` and dispatch them.
- Register view ops in a new `view` backend module for cpu/npu/meta (share storage, adjust shape/stride/offset).
- Add grad_fn for view ops to map gradient back to original shape (reshape/transpose inverse).
- Update `Tensor.reshape/transpose` to call `_functional` variants.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_view_dispatch.py::test_reshape_autograd -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/_dispatch/registry.py src/mindtorch_v2/_backends/view.py tests/mindtorch_v2/test_view_dispatch.py
git commit -m "feat: dispatch view ops with autograd"
```

---

### Task 3: to() becomes a dispatched op

**Files:**
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Test: `tests/mindtorch_v2/test_dispatch_unify.py::test_to_uses_dispatch`

**Step 1: Write the failing test**

```python
import mindtorch_v2._tensor as tensor_mod
import mindtorch_v2._dispatch.dispatcher as dispatcher
from mindtorch_v2._storage import typed_storage_from_numpy
from mindtorch_v2._tensor import Tensor
import numpy as np


def test_to_uses_dispatch(monkeypatch):
    called = {}

    def fake_dispatch(name, device, *args, **kwargs):
        called["name"] = name
        storage = typed_storage_from_numpy(np.array([1.0], dtype=np.float32), None)
        return Tensor(storage, (1,), (1,))

    monkeypatch.setattr(dispatcher, "dispatch", fake_dispatch)
    x = tensor_mod.Tensor(typed_storage_from_numpy(np.array([1.0], dtype=np.float32), None), (1,), (1,))
    _ = x.to("cpu")
    assert called["name"] == "to"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dispatch_unify.py::test_to_uses_dispatch -v`

Expected: FAIL because to() does not dispatch.

**Step 3: Write minimal implementation**

- Add `_functional.to` that calls `dispatch("to", a.device.type, a, device=dev)`.
- Register `to` impls in cpu/npu/meta backends using a shared conversion helper (moved out of Tensor.to to avoid recursion).
- Update `Tensor.to` to only handle pipeline flush and then call `_functional.to`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_unify.py::test_to_uses_dispatch -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_backends/ascend.py tests/mindtorch_v2/test_dispatch_unify.py
git commit -m "feat: dispatch to()"
```
