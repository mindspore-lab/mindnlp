# MindTorch v2 Profiler Parity Batch12 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `key_averages` API parity by exposing per-row objects with torch-like attributes in addition to table rendering.

**Architecture:** Add a lightweight `FunctionEventAvg` row wrapper and iterable behavior for `_KeyAverages`, mapping row fields to commonly-used torch attribute names.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing row-API parity tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_key_averages_iter_returns_row_objects_with_torch_like_attrs`
- `test_key_averages_row_getitem_and_attr_consistency`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_iter_returns_row_objects_with_torch_like_attrs tests/mindtorch_v2/test_profiler.py::test_key_averages_row_getitem_and_attr_consistency -q`

### Task 2: Implement row object API

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add row wrapper class**
- Introduce `FunctionEventAvg`-like wrapper with attributes:
  - `key`, `count`, `self_cpu_time_total`, `cpu_time_total`, `cpu_time`

**Step 2: Make `_KeyAverages` iterable**
- Implement `__iter__`, `__getitem__`, and stable row materialization.

**Step 3: Keep table path unchanged**
- Ensure existing table output and sort behavior remain backward compatible.

**Step 4: Run targeted tests**
- Re-run Task 1 tests.

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add key_averages row object parity api"`

### Task 3: Full validation, PR, and fast merge

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch12-plan.md`

**Step 1: Run full validations**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Record notes and commit**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch12-plan.md`
- `git commit -m "docs: record profiler parity batch12 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`
- Push branch to `origin`
- Create PR to `mindspore-lab/mindnlp`
- Merge PR immediately (`--merge --delete-branch`)
- Create next worktree from latest `ms/master`

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `40 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
