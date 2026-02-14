# Autograd API + Graph Correctness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fully align autograd graph correctness and API surface with PyTorch (backward flags, retain_grad, detach, hooks, autograd.grad).

**Architecture:** Extend the current autograd engine with saved-tensor version checks, add API methods on Tensor, and implement functional autograd.grad. Keep checks centralized in Node/Tensor to avoid backend divergence.

**Tech Stack:** Python, numpy, existing mindtorch_v2 autograd engine.

---

### Task 1: Backward semantics and scalar-gradient behavior

**Files:**
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Test: `tests/mindtorch_v2/test_autograd_api.py`

**Step 1: Write the failing test**

```python
import pytest
import mindtorch_v2 as torch


def test_backward_requires_grad_for_non_scalar():
    t = torch.ones((2,))
    with pytest.raises(RuntimeError):
        t.backward()


def test_backward_defaults_to_ones_for_scalar():
    t = torch.ones((2,))
    y = t.sum()
    y.backward()
    assert t.grad is not None
    assert t.grad.numpy().tolist() == [1.0, 1.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_backward_requires_grad_for_non_scalar -v`
Expected: FAIL (no error or wrong behavior).

**Step 3: Write minimal implementation**

```python
# in engine.backward
if grad is None:
    if tensor.numel() != 1:
        raise RuntimeError("grad can be implicitly created only for scalar outputs")
    grad = tensor._ones_like()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_backward_requires_grad_for_non_scalar -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_autograd_api.py
git commit -m "Match torch backward scalar grad semantics"
```

---

### Task 2: retain_graph and create_graph flags

**Files:**
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Test: `tests/mindtorch_v2/test_autograd_api.py`

**Step 1: Write the failing test**

```python
import pytest
import mindtorch_v2 as torch


def test_retain_graph_allows_double_backward():
    t = torch.ones((2,))
    y = t.sum()
    y.backward(retain_graph=True)
    # should allow a second backward call
    y.backward(retain_graph=True)
    assert t.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_retain_graph_allows_double_backward -v`
Expected: FAIL (graph freed).

**Step 3: Write minimal implementation**

```python
# add retain_graph and create_graph to engine.backward signature
# keep graph when retain_graph True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_retain_graph_allows_double_backward -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_autograd_api.py
git commit -m "Support retain_graph/create_graph in backward"
```

---

### Task 3: retain_grad for non-leaf tensors

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Test: `tests/mindtorch_v2/test_autograd_api.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_retain_grad_populates_non_leaf_grad():
    t = torch.ones((2,))
    y = t.sum()
    y.retain_grad()
    y.backward()
    assert y.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_retain_grad_populates_non_leaf_grad -v`
Expected: FAIL (grad is None).

**Step 3: Write minimal implementation**

```python
# Tensor.retain_grad() sets a flag
# engine.backward populates grad for non-leaf when flag is set
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_retain_grad_populates_non_leaf_grad -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_autograd_api.py
git commit -m "Implement retain_grad for non-leaf tensors"
```

---

### Task 4: requires_grad_, detach, detach_

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_autograd_api.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_detach_breaks_grad_chain():
    t = torch.ones((2,))
    t.requires_grad_(True)
    y = t.detach()
    assert y.requires_grad is False


def test_detach_inplace():
    t = torch.ones((2,))
    t.requires_grad_(True)
    t.detach_()
    assert t.requires_grad is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_detach_breaks_grad_chain -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Tensor.requires_grad_ sets requires_grad and clears grad_fn if False
# Tensor.detach creates a view sharing storage but no grad_fn
# Tensor.detach_ clears grad_fn and requires_grad on self
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_detach_breaks_grad_chain -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_autograd_api.py
git commit -m "Add requires_grad_ and detach APIs"
```

---

### Task 5: Tensor.register_hook

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Test: `tests/mindtorch_v2/test_autograd_api.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_register_hook_receives_grad():
    t = torch.ones((2,))
    t.requires_grad_(True)
    seen = {}
    def hook(grad):
        seen["grad"] = grad.numpy().tolist()
        return grad
    t.register_hook(hook)
    t.sum().backward()
    assert seen["grad"] == [1.0, 1.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_register_hook_receives_grad -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Tensor.register_hook stores hooks list
# engine.backward applies hooks before accumulation
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_register_hook_receives_grad -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_autograd_api.py
git commit -m "Implement tensor backward hooks"
```

---

### Task 6: torch.autograd.grad

**Files:**
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Modify: `src/mindtorch_v2/_autograd/__init__.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Test: `tests/mindtorch_v2/test_autograd_api.py`

**Step 1: Write the failing test**

```python
import pytest
import mindtorch_v2 as torch


def test_autograd_grad_basic():
    x = torch.ones((2,))
    x.requires_grad_(True)
    y = (x * x).sum()
    (gx,) = torch.autograd.grad(y, (x,))
    assert gx.numpy().tolist() == [2.0, 2.0]


def test_autograd_grad_allow_unused():
    x = torch.ones((2,))
    x.requires_grad_(True)
    y = torch.ones((1,))
    with pytest.raises(RuntimeError):
        torch.autograd.grad(y, (x,))
    gx = torch.autograd.grad(y, (x,), allow_unused=True)[0]
    assert gx is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_autograd_grad_basic -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# Implement grad(outputs, inputs, grad_outputs=None, retain_graph=None,
# create_graph=False, allow_unused=False)
# Use internal engine to compute grads without touching .grad
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_api.py::test_autograd_grad_basic -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_autograd/engine.py src/mindtorch_v2/_autograd/__init__.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_autograd_api.py
git commit -m "Add torch.autograd.grad"
```

---

### Task 7: Full test run

**Step 1: Run full suite**

Run: `pytest -q tests/mindtorch_v2`
Expected: PASS

**Step 2: Commit (if needed)**

```bash
git status -sb
```

---

Plan complete and saved to `docs/plans/2026-02-14-mindtorch-v2-autograd-api-plan.md`.

Two execution options:
1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks.
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints.

Which approach?
