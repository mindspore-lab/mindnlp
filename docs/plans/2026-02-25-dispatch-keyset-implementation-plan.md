# Dispatch Keyset Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement torch 2.4-aligned dispatch keyset bitmask semantics with TLS include/exclude masks, preserving custom Pipeline ordering.

**Architecture:** Replace `DispatchKeySet` with a bitmask-backed class, define a torch 2.4 priority list with Pipeline inserted after BackendSelect, and apply TLS include/exclude masks to all dispatch entry points. Composite/PrivateUse keys are added to the enum and global fallthrough to preserve current behavior.

**Tech Stack:** Python, pytest

---

### Task 1: Define torch 2.4 dispatch key enum and priority list

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/keys.py`
- Modify: `src/mindtorch_v2/_dispatch/registry.py`

**Step 1: Write the failing test**

Create `tests/mindtorch_v2/contract/test_dispatch_keyset.py` with:

```python
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet


def test_priority_order_pipeline_after_backendselect():
    keyset = DispatchKeySet.from_mask(
        DispatchKey.BackendSelect | DispatchKey.Pipeline | DispatchKey.CPU
    )
    order = [k for k in keyset.iter_keys()]
    assert order[0] == DispatchKey.BackendSelect
    assert order[1] == DispatchKey.Pipeline
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py -k priority`
Expected: FAIL with ImportError or missing methods on DispatchKeySet.

**Step 3: Write minimal implementation**

In `src/mindtorch_v2/_dispatch/keys.py`:
- Change `DispatchKey` to `IntEnum` with bitmask values (1 << N) following torch 2.4 priority order.
- Add CompositeImplicitAutograd, CompositeExplicitAutograd, PrivateUse1/2/3 keys.
- Define `DISPATCH_KEY_PRIORITY` list in order, insert `Pipeline` after `BackendSelect`.
- Implement `DispatchKeySet` storing `mask` with methods: `from_mask`, `has`, `add`, `remove`, `without`, `iter_keys`, `__contains__`.

In `src/mindtorch_v2/_dispatch/registry.py`:
- Update `resolve_dispatch_key` to return DispatchKey values (bitmask enum) instead of set.
- Add new keys to global fallthrough set.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py -k priority`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/contract/test_dispatch_keyset.py src/mindtorch_v2/_dispatch/keys.py src/mindtorch_v2/_dispatch/registry.py
git commit -m "feat: add dispatch key priority and bitmask keyset"
```

---

### Task 2: Implement TLS include/exclude masks and apply to dispatch paths

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_dispatch/keys.py`
- Test: `tests/mindtorch_v2/contract/test_dispatch_keyset.py`

**Step 1: Write the failing test**

Append to `tests/mindtorch_v2/contract/test_dispatch_keyset.py`:

```python
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet, include_keys, exclude_keys
from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset
from mindtorch_v2._dispatch.registry import registry


def test_tls_exclude_applies_to_dispatch_and_redispatch():
    calls = []

    def cpu_kernel(x):
        calls.append("cpu")
        return x

    def meta_kernel(x):
        calls.append("meta")
        return x

    registry.register_kernel("aten::dummy", DispatchKey.CPU, cpu_kernel)
    registry.register_kernel("aten::dummy", DispatchKey.Meta, meta_kernel)

    keyset = DispatchKeySet.from_mask(DispatchKey.CPU | DispatchKey.Meta)
    with exclude_keys(DispatchKey.Meta):
        dispatch_with_keyset("dummy", keyset, None, 1)
    assert calls == ["cpu"]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py -k exclude`
Expected: FAIL (missing include/exclude context or not applied).

**Step 3: Write minimal implementation**

In `src/mindtorch_v2/_dispatch/keys.py`:
- Add TLS state with `include_mask` and `exclude_mask` stacks.
- Implement internal context managers `include_keys()` and `exclude_keys()` (internal-only for now) returning masks.
- Add `apply_tls_masks(mask)` to compute effective mask.

In `src/mindtorch_v2/_dispatch/dispatcher.py`:
- Apply `apply_tls_masks` to keysets inside `dispatch`, `dispatch_with_keyset`, and `redispatch` before key ordering.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py -k exclude`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/contract/test_dispatch_keyset.py src/mindtorch_v2/_dispatch/keys.py src/mindtorch_v2/_dispatch/dispatcher.py
git commit -m "feat: apply tls include/exclude masks to dispatch"
```

---

### Task 3: Ensure fallthrough semantics remain stable with new keys

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/registry.py`
- Test: `tests/mindtorch_v2/contract/test_dispatch_keyset.py`

**Step 1: Write the failing test**

Append to `tests/mindtorch_v2/contract/test_dispatch_keyset.py`:

```python
from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset
from mindtorch_v2._dispatch.registry import registry


def test_composite_keys_fallthrough():
    calls = []

    def cpu_kernel(x):
        calls.append("cpu")
        return x

    registry.register_kernel("aten::dummy2", DispatchKey.CPU, cpu_kernel)

    keyset = DispatchKeySet.from_mask(
        DispatchKey.CompositeImplicitAutograd | DispatchKey.CPU
    )
    dispatch_with_keyset("dummy2", keyset, None, 1)
    assert calls == ["cpu"]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py -k composite`
Expected: FAIL if CompositeImplicitAutograd not fallthrough or key missing.

**Step 3: Write minimal implementation**

In `src/mindtorch_v2/_dispatch/registry.py`:
- Ensure Composite* and PrivateUse* are added to global fallthrough set.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py -k composite`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/contract/test_dispatch_keyset.py src/mindtorch_v2/_dispatch/registry.py
git commit -m "test: ensure composite keys fallthrough"
```

---

### Task 4: Full contract test run

**Files:**
- Test: `tests/mindtorch_v2/contract/test_dispatch_keyset.py`

**Step 1: Run full test**

Run: `pytest -q tests/mindtorch_v2/contract/test_dispatch_keyset.py`
Expected: PASS.

**Step 2: Commit (if needed)**

```bash
git add tests/mindtorch_v2/contract/test_dispatch_keyset.py
git commit -m "test: add dispatch keyset contract coverage"
```

