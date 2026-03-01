# MindTorch v2 Profiler Parity Batch18 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `key_averages()` EventList list-like parity by adding remaining high-frequency mutation/query methods (`append`, `extend`, `insert`, `pop`, `remove`, `clear`, `index`).

**Architecture:** Reuse `_KeyAverages` row-object cache as the list-like backing store and implement mutation/query methods directly on that cache. Keep existing aggregation/table behavior unchanged, and avoid broad refactors by scoping this batch to list API compatibility only.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add RED tests for missing list-like APIs

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing mutation test**
- `test_key_averages_supports_append_extend_insert_pop_remove_clear`

**Step 2: Add failing index test**
- `test_key_averages_supports_index_lookup`

**Step 3: Run targeted RED tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_append_extend_insert_pop_remove_clear tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_index_lookup -q`

### Task 2: Implement list-like mutation/query methods

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add methods on `_KeyAverages`**
- `append(value)`
- `extend(values)`
- `insert(index, value)`
- `pop(index=-1)`
- `remove(value)`
- `clear()`
- `index(value, start=0, stop=None)`

**Step 2: Ensure methods operate on row-object cache**
- Keep behavior aligned with Python list semantics for return values/exceptions.

**Step 3: Run targeted GREEN tests**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch18-plan.md`

**Step 1: Run full verification suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add key_averages list mutation apis"`
- `git add docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch18-plan.md`
- `git commit -m "docs: record profiler parity batch18 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run verification suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_append_extend_insert_pop_remove_clear tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_index_lookup -q`
  - Result: `2 failed` (RED), then `2 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `60 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
