# Functionalize NPU/Meta View Writeback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pure NPU view writeback for functionalize using ACLNN slice/assign kernels and align meta behavior with torch (metadata-only writeback).

**Architecture:** Extend `_writeback` to branch by device (CPU/NPU/meta). NPU path resolves view metadata into strided slice parameters and dispatches ACLNN `StridedSliceAssign` (or equivalent) to update base in-place. Meta path updates only metadata and errors when meta kernel is missing.

**Tech Stack:** Python, ACLNN ctypes bindings, pytest, mindtorch_v2 dispatcher/functionalize.

---

### Task 1: Add failing contract test for NPU view writeback

**Files:**
- Modify: `tests/mindtorch_v2/contract/test_functionalize_view_writeback.py`

**Step 1: Write the failing test**

```python
def test_functionalize_writeback_respects_view_npu():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], device="npu")
    view = base.view((2, 2))
    with torch.functionalize():
        view.add_(torch.ones((2, 2), device="npu"))
    assert base.to("cpu").storage().data.tolist() == [2.0, 3.0, 4.0, 5.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py::test_functionalize_writeback_respects_view_npu`
Expected: FAIL with `NotImplementedError` from functionalize NPU view writeback.

**Step 3: Commit**

```bash
git add tests/mindtorch_v2/contract/test_functionalize_view_writeback.py
git commit -m "test: cover functionalize NPU view writeback"
```

### Task 2: Add failing contract test for meta view writeback (metadata-only)

**Files:**
- Modify: `tests/mindtorch_v2/contract/test_functionalize_view_writeback.py`

**Step 1: Write the failing test**

```python
def test_functionalize_writeback_respects_view_meta():
    base = torch.tensor([1.0, 2.0, 3.0, 4.0], device="meta")
    view = base.view((2, 2))
    with torch.functionalize():
        view.add_(torch.ones((2, 2), device="meta"))
    assert base.device.type == "meta"
    assert base.shape == (4,)
    assert view.shape == (2, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py::test_functionalize_writeback_respects_view_meta`
Expected: FAIL due to meta writeback not implemented or metadata mismatch.

**Step 3: Commit**

```bash
git add tests/mindtorch_v2/contract/test_functionalize_view_writeback.py
git commit -m "test: cover functionalize meta view writeback"
```

### Task 3: Add ACLNN bindings for strided slice assign

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Write the failing test**

Add a temporary test in `tests/mindtorch_v2/test_ops_npu.py` to call the new binding via a small wrapper (or invoke via functionalize path once wired). Keep it minimal and remove it after functionalize test covers it.

```python
def test_npu_strided_slice_assign_available():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="npu")
    y = torch.ones((1, 2), device="npu")
    x[:1].copy_(y)
    assert x.to("cpu").numpy().tolist() == [[1.0, 1.0], [3.0, 4.0]]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/test_ops_npu.py::test_npu_strided_slice_assign_available`
Expected: FAIL with missing ACLNN binding or not implemented.

**Step 3: Write minimal implementation**

- Add optional symbols for `aclnnStridedSlice`/`aclnnStridedSliceAssign` (exact names based on CANN docs).
- Create helper `strided_slice_assign(...)` that builds `aclIntArray` for start/end/step and calls ACLNN.
- Ensure all `aclDestroy*` calls run in `finally` and `_maybe_sync` respects `ACL_LAUNCH_BLOCKING`.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/mindtorch_v2/test_ops_npu.py::test_npu_strided_slice_assign_available`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(npu): add ACLNN strided slice assign binding"
```

### Task 4: Implement NPU view writeback in functionalize

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/functionalize.py`
- Modify: `src/mindtorch_v2/_backends/npu/ops.py` (if helper needed)

**Step 1: Write the failing test**

Use Task 1 test (already failing).

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py::test_functionalize_writeback_respects_view_npu`
Expected: FAIL with `NotImplementedError`.

**Step 3: Write minimal implementation**

- In `_writeback`, add NPU branch:
  - Validate `target` and `result` shapes match.
  - Extract `view_meta` from `result` or `target` and compute per-dim `start/end/step` for a strided slice.
  - If view is not representable, raise `RuntimeError("aten::<op> is not implemented for NPU")` with the alias op name.
  - Call ACLNN strided slice assign to write `result` into base storage.
  - Update `target` metadata to match `result` (shape/stride/offset/base/view_meta).

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py::test_functionalize_writeback_respects_view_npu`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/functionalize.py src/mindtorch_v2/_backends/npu/ops.py
git commit -m "feat: functionalize NPU view writeback"
```

### Task 5: Implement meta view writeback behavior

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/functionalize.py`

**Step 1: Write the failing test**

Use Task 2 test (already failing).

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py::test_functionalize_writeback_respects_view_meta`
Expected: FAIL prior to meta handling.

**Step 3: Write minimal implementation**

- In `_writeback`, add meta branch:
  - Update metadata only: `shape/stride/offset/_base/_view_meta` from `result` onto `target`.
  - No data movement.

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/mindtorch_v2/contract/test_functionalize_view_writeback.py::test_functionalize_writeback_respects_view_meta`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/functionalize.py
git commit -m "feat: functionalize meta view writeback"
```

### Task 6: Full test sweep (mindspore env)

**Files:**
- None

**Step 1: Run tests**

Run: `pytest -q tests/mindtorch_v2`
Expected: PASS

**Step 2: Commit**

```bash
git status -sb
```

### Task 7: PR preparation

**Files:**
- None

**Step 1: Push branch**

```bash
git push -u origin functionalize-view-writeback
```

**Step 2: Create PR**

- Title: `v2: functionalize NPU/meta view writeback`
- Body with real newlines (no `\n` escapes).

---
