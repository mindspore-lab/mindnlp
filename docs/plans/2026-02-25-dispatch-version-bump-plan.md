# Dispatch Version Counter Bump Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move version counter bumping for mutating ops into the dispatch layer with torch-aligned timing and view semantics.

**Architecture:** Identify mutated arguments from schema metadata and bump version counters only after successful execution. Handle pipeline and functionalize paths at their true mutation points to avoid premature or double bumps.

**Tech Stack:** Python, pytest.

---

### Task 1: Add a failing test for dispatch-level bump

**Files:**
- Modify: `tests/mindtorch_v2/test_autograd_inplace.py`

**Step 1: Write the failing test**

```python
def test_dispatch_inplace_bumps_version():
    t = torch.tensor([1.0])
    v0 = t._version_counter.value
    from mindtorch_v2._dispatch.dispatcher import dispatch
    dispatch("add_", t.device.type, t, torch.tensor([1.0]))
    assert t._version_counter.value == v0 + 1
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/test_autograd_inplace.py -k dispatch_inplace_bumps_version`
Expected: FAIL (version does not change).

---

### Task 2: Implement dispatch-level version bumping

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_dispatch/functionalize.py`

**Step 1: Add a helper to compute mutated targets**

```python
def _mutated_targets(schema_obj, args, kwargs):
    # reuse OpSchema params mutates; return list of tensors
```

**Step 2: Add `_bump_versions` helper**

```python
def _bump_versions(schema_obj, args, kwargs):
    # dedupe targets; if view, bump base version counter; else bump self
```

**Step 3: Call `_bump_versions` after kernel success**

- In `dispatch_with_keyset` `_run_kernel()` after `kernel(*args, **impl_kwargs)` returns.
- In `_PendingOp.execute()` after `impl` returns (pipeline case).

**Step 4: Functionalize writeback bump**

- After `_writeback` completes in `functionalize.py`, bump the target’s version (base if view). Ensure no double-bump when functionalize is enabled.

**Step 5: Run targeted test**

Run: `pytest -q tests/mindtorch_v2/test_autograd_inplace.py -k dispatch_inplace_bumps_version`
Expected: PASS.

---

### Task 3: Regression checks

**Step 1: Run inplace tests**

Run: `pytest -q tests/mindtorch_v2/test_autograd_inplace.py`
Expected: PASS.

**Step 2: Commit**

```bash
git add tests/mindtorch_v2/test_autograd_inplace.py \
    src/mindtorch_v2/_dispatch/dispatcher.py \
    src/mindtorch_v2/_dispatch/functionalize.py

git commit -m "feat: bump version counters in dispatch"
```
