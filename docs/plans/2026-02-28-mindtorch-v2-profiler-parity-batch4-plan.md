# MindTorch v2 Profiler Parity Batch4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `key_averages` parity with torch for grouping controls and sort semantics.

**Architecture:** Extend profiler aggregation API to support `key_averages(group_by_input_shape, group_by_stack_n)` and strict `table(sort_by=...)` validation with torch-like metric keys.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler module.

---

### Task 1: Add failing key_averages parity tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing tests**
- `test_key_averages_supports_group_by_input_shape`
- `test_key_averages_supports_group_by_stack_n`
- `test_key_averages_table_unknown_sort_key_raises`

**Step 2: Run targeted tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_group_by_input_shape tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_group_by_stack_n tests/mindtorch_v2/test_profiler.py::test_key_averages_table_unknown_sort_key_raises -q`

### Task 2: Implement grouping controls and strict sort behavior

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Extend aggregation model**
- Add optional grouping dimensions:
  - input shape key when `group_by_input_shape=True`
  - stack-prefix key when `group_by_stack_n>0`
- Keep default behavior unchanged when both options are disabled.

**Step 2: Extend table sort key mapping**
- Support torch-like keys: `self_cpu_time_total`, `cpu_time_total`, `count`.
- Raise `AttributeError` on unknown sort key.

**Step 3: Wire profile API**
- Change `profile.key_averages(...)` signature to accept `group_by_input_shape` and `group_by_stack_n`.

**Step 4: Run targeted tests**
- Re-run the three tests from Task 1.

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): align key_averages grouping and sort semantics"`

### Task 3: Full validation and notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch4-plan.md` (append verification notes)

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regression tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch4-plan.md`
- `git commit -m "docs: record profiler parity batch4 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add key_averages grouping/sort failing tests
- Task 2: implement `group_by_input_shape`, `group_by_stack_n`, and strict `sort_by` validation
- Task 3: full validation

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_group_by_input_shape tests/mindtorch_v2/test_profiler.py::test_key_averages_supports_group_by_stack_n tests/mindtorch_v2/test_profiler.py::test_key_averages_table_unknown_sort_key_raises -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`25 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- `profile.key_averages(...)` now accepts `group_by_input_shape` and `group_by_stack_n`.
- `table(sort_by=...)` now supports torch-like keys and raises `AttributeError` for unknown keys.
