# Dispatch Schema/Alias/Functionalize Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Torch-aligned schema binding, alias resolution, and functionalization dispatch key with exact error messages verified against PyTorch.

**Architecture:** Add an `OpSchema` binder in registry/dispatcher, alias mapping in registry, and a `Functionalize` dispatch key whose kernel rewrites schema-marked in-place ops to functional equivalents.

**Tech Stack:** Python, PyTorch (oracle in tests), existing mindtorch dispatch registry.

---

### Task 1: Add schema representation and binder

**Files:**
- Create: `src/mindtorch_v2/_dispatch/schema.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`

**Step 1: Write failing tests**
- Create: `tests/mindtorch_v2/contract/test_schema_binding.py`

```python
import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_tensor_unexpected_kwarg_message_matches_torch():
    def mt():
        torch.tensor([1.0], dtype=torch.float32, badkw=1)

    def th():
        pt.tensor([1.0], dtype=pt.float32, badkw=1)

    assert_torch_error(mt, th)
```

**Step 2: Run test (expect FAIL)**
Run: `pytest tests/mindtorch_v2/contract/test_schema_binding.py::test_tensor_unexpected_kwarg_message_matches_torch -v`

**Step 3: Implement minimal schema binding**
- Parse a minimal subset of Torch schema needed for current ops (positional + kwargs + defaults).
- Bind args before kernel execution and raise Torch-matching errors.

**Step 4: Re-run test (expect PASS)**

**Step 5: Commit**

---

### Task 2: Wire schema binding into dispatcher

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`

**Step 1: Add failing test for bad positional args**

**Step 2: Run test (FAIL)**

**Step 3: Implement binding call in `dispatch_with_keyset`**

**Step 4: Run test (PASS)**

**Step 5: Commit**

---

### Task 3: Add alias registration + resolution

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Create: `tests/mindtorch_v2/contract/test_alias_resolution.py`

**Step 1: Write failing test**
- Register an alias op name and ensure the canonical kernel is invoked.
- Ensure error messages use the alias name when binding fails.

**Step 2: Run test (FAIL)**

**Step 3: Implement alias mapping**

**Step 4: Run test (PASS)**

**Step 5: Commit**

---

### Task 4: Add Functionalize dispatch key plumbing

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/keys.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Create: `src/mindtorch_v2/_backends/functionalize.py`

**Step 1: Write failing tests**
- Create: `tests/mindtorch_v2/contract/test_functionalize.py`

Cases:
- `add_` should route through functionalize wrapper when enabled.
- Missing functionalize rule for a mutating schema should match torch error.

**Step 2: Run tests (FAIL)**

**Step 3: Implement Functionalize key and wrapper kernel**
- Determine mutating inputs via schema.
- Derive functional op name for common `*_` patterns.
- Otherwise require explicit rule.

**Step 4: Run tests (PASS)**

**Step 5: Commit**

---

### Task 5: Full test suite

Run: `pytest -q tests/mindtorch_v2`
Expected: PASS

Commit if needed.
