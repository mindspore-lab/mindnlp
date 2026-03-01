# MindTorch v2 Profiler Parity Batch19 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close remaining high-impact `FunctionEventAvg` compatibility gaps by adding torch-like row fields (`*_str`, `cuda_time`, structural flags/ids) and `add(other)` aggregation behavior.

**Architecture:** Extend `_FunctionEventAvgRow` with derived string-format properties and compatibility fields initialized from grouped row payload defaults. Implement `add(other)` as an in-place-compatible aggregation operation over the underlying row dict and return `self`, matching torch's mutating API shape.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add RED tests for row compatibility fields

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing field exposure test**
- `test_key_averages_row_exposes_torch_compat_fields_and_time_strings`

**Step 2: Add failing add() behavior test**
- `test_key_averages_row_add_merges_counts_and_times`

**Step 3: Run targeted RED tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_row_exposes_torch_compat_fields_and_time_strings tests/mindtorch_v2/test_profiler.py::test_key_averages_row_add_merges_counts_and_times -q`

### Task 2: Implement row compatibility fields and add()

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Extend grouped row defaults**
- Add default keys:
  - `node_id`, `is_legacy`, `is_remote`, `overload_name`
  - `cpu_children`, `cpu_parent`

**Step 2: Extend `_FunctionEventAvgRow` fields**
- Add attributes:
  - `node_id`, `is_legacy`, `is_remote`, `overload_name`
  - `cpu_children`, `cpu_parent`
- Add derived string fields:
  - `cpu_time_str`, `cpu_time_total_str`, `self_cpu_time_total_str`
  - `device_time_str`, `device_time_total_str`, `self_device_time_total_str`
- Add compatibility alias:
  - `cuda_time` (mapped to `device_time`)

**Step 3: Implement `add(other)`**
- Mutate `self._row` numeric totals (`count`, `self_time_ns`, `total_time_ns`, device-time ns and avg ns) by combining `other._row`.
- Refresh derived attributes after aggregation.
- Return `self`.

**Step 4: Run targeted GREEN tests**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch19-plan.md`

**Step 1: Run full verification suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add FunctionEventAvg compatibility fields"`
- `git add docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch19-plan.md`
- `git commit -m "docs: record profiler parity batch19 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run verification suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_row_exposes_torch_compat_fields_and_time_strings tests/mindtorch_v2/test_profiler.py::test_key_averages_row_add_merges_counts_and_times -q`
  - Result: `2 failed` (RED), then `2 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `62 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
