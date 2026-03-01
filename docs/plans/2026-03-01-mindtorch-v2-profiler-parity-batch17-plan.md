# MindTorch v2 Profiler Parity Batch17 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `key_averages()` EventList list-like behavior with torch for high-impact collection APIs (`copy`, `sort`, `reverse`, `count`) and EventList re-aggregation entrypoint (`key_averages`).

**Architecture:** Keep `_KeyAverages` as the central EventList wrapper and add list-like methods that operate on stable row-object snapshots. Implement `key_averages(...)` on `_KeyAverages` with torch-compatible keyword acceptance (`group_by_input_shapes` alias) and a minimal re-aggregation contract. Avoid introducing full mutable list parity in this batch to keep scope tight and low-risk.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add RED tests for EventList list-like parity

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing tests for list-like methods**
- `test_key_averages_supports_copy_and_count`
- `test_key_averages_sort_and_reverse_operate_on_row_objects`

**Step 2: Add failing test for EventList re-aggregation API**
- `test_key_averages_event_list_key_averages_accepts_torch_keywords`

**Step 3: Run targeted tests to confirm RED**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_copy_and_count tests/mindtorch_v2/test_profiler.py::test_key_averages_sort_and_reverse_operate_on_row_objects tests/mindtorch_v2/test_profiler.py::test_key_averages_event_list_key_averages_accepts_torch_keywords -q`

### Task 2: Implement minimal EventList list-like APIs

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add stable row cache for list-like operations**
- Maintain an internal row-object cache to support in-place ordering changes for `sort` and `reverse`.

**Step 2: Add list-like APIs**
- `copy()` returns a list of row objects.
- `count(value)` returns count in cached list semantics.
- `sort(*, key=None, reverse=False)` sorts cached row objects.
- `reverse()` reverses cached row objects.

**Step 3: Add EventList-level `key_averages(...)`**
- Accept torch-style kwargs:
  - `group_by_input_shapes` (alias)
  - `group_by_stack_n`
  - `group_by_overload_name` (accepted, ignored for now)
- Return a new `_KeyAverages` instance over the same source events.

**Step 4: Keep existing table and total APIs backward-compatible**
- No behavioral regression for `table`, `self_cpu_time_total`, `total_average`.

**Step 5: Run targeted tests to confirm GREEN**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch17-plan.md`

**Step 1: Run full verification suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add key_averages list-like parity apis"`
- `git add docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch17-plan.md`
- `git commit -m "docs: record profiler parity batch17 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run verification suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_copy_and_count tests/mindtorch_v2/test_profiler.py::test_key_averages_sort_and_reverse_operate_on_row_objects tests/mindtorch_v2/test_profiler.py::test_key_averages_event_list_key_averages_accepts_torch_keywords -q`
  - Result: `3 failed` (RED), then `3 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `58 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
