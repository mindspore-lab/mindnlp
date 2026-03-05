# MindTorch v2 Profiler Parity Batch2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add torch-like schedule behavior to `mindtorch_v2.profiler` and align step-based trace callback semantics while preserving no-`mindspore` dependency and low overhead.

**Architecture:** Introduce `ProfilerAction` and `schedule(...)` helper in profiler API, evaluate action per profiler step, gate event recording by action state, and trigger `on_trace_ready` at `RECORD_AND_SAVE` boundaries.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler runtime.

---

### Task 1: Add schedule API tests (failing first)

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Write failing tests**
- `test_profile_rejects_non_callable_schedule`
- `test_profiler_schedule_filters_recorded_steps`
- `test_profiler_schedule_triggers_trace_ready_on_save_action`
- `test_profiler_schedule_invalid_config_raises`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profile_rejects_non_callable_schedule tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_filters_recorded_steps tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_triggers_trace_ready_on_save_action tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_invalid_config_raises -q`

**Step 3: Commit tests after implementation in later tasks**

### Task 2: Implement `ProfilerAction` + `schedule(...)` helper

**Files:**
- Modify: `src/mindtorch_v2/profiler/common.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`
- Modify: `src/mindtorch_v2/profiler/__init__.py`

**Step 1: Add minimal implementation**
- Add `ProfilerAction` enum (`NONE`, `WARMUP`, `RECORD`, `RECORD_AND_SAVE`).
- Add `schedule(wait, warmup, active, repeat, skip_first)` helper returning callable step->action.
- Validate schedule config (`active > 0`, non-negative integer constraints).

**Step 2: Run targeted tests**
- Run the four schedule tests from Task 1.

**Step 3: Commit**
- `git add src/mindtorch_v2/profiler/common.py src/mindtorch_v2/profiler/profiler.py src/mindtorch_v2/profiler/__init__.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add profiler schedule and action api"`

### Task 3: Integrate schedule semantics into profile lifecycle

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Implement behavior**
- Validate `schedule` argument must be callable or `None`.
- Evaluate current action at start and on every `step()`.
- Record events only for `RECORD`/`RECORD_AND_SAVE` actions.
- Trigger `on_trace_ready(self)` when stepping past `RECORD_AND_SAVE` action.

**Step 2: Run targeted tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_filters_recorded_steps tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_triggers_trace_ready_on_save_action -q`

**Step 3: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): honor schedule actions in profiler step lifecycle"`

### Task 4: Final verification

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch2-plan.md` (append verification notes)

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regression suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/common.py src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Commit verification notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch2-plan.md`
- `git commit -m "docs: record profiler parity batch2 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add schedule-related failing tests
- Task 2: implement `ProfilerAction` and `schedule(...)` API
- Task 3: integrate schedule semantics into profile lifecycle
- Task 4: final validation

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profile_rejects_non_callable_schedule tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_filters_recorded_steps tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_triggers_trace_ready_on_save_action tests/mindtorch_v2/test_profiler.py::test_profiler_schedule_invalid_config_raises -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`18 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/common.py src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- This batch keeps `mindtorch_v2` profiler independent from `mindspore`.
- Implemented schedule actions: `NONE`, `WARMUP`, `RECORD`, `RECORD_AND_SAVE`.
- Recording is gated by schedule action and callback is triggered on `RECORD_AND_SAVE` step boundary.
