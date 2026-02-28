# MindTorch v2 Profiler Parity Batch8 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `key_averages` self-time semantics with torch-style nested event subtraction behavior.

**Architecture:** Compute `self_time_ns` from event interval overlap on same thread by subtracting child durations from parent durations (for both op and scope events), then aggregate with this derived self-time in `key_averages`.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing self-time parity tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_key_averages_self_time_subtracts_nested_scope_time`
- `test_key_averages_self_time_not_greater_than_total`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_self_time_subtracts_nested_scope_time tests/mindtorch_v2/test_profiler.py::test_key_averages_self_time_not_greater_than_total -q`

### Task 2: Implement nested self-time computation

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add per-event self-time derivation**
- Sort events by thread and timeline.
- Build parent-child interval relation via stack on same thread.
- Subtract covered child duration from parent to derive per-event `self_time_ns`.

**Step 2: Use derived self-time in aggregation**
- Replace previous `self_time_ns += duration` with derived value accumulation.
- Keep `total_time_ns` unchanged.

**Step 3: Run targeted tests**
- Re-run Task 1 tests.

**Step 4: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): align key_averages self-time semantics"`

### Task 3: Full validation and notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch8-plan.md`

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regressions**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Record and commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch8-plan.md`
- `git commit -m "docs: record profiler parity batch8 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add self-time parity tests
- Task 2: implement nested self-time subtraction for key_averages
- Task 3: run full profiler/regression/syntax validation

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_self_time_subtracts_nested_scope_time tests/mindtorch_v2/test_profiler.py::test_key_averages_self_time_not_greater_than_total -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`32 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- `key_averages` now derives per-event self-time by subtracting nested child event durations on same thread.
- Aggregation uses derived `self_time_ns` while preserving `total_time_ns` semantics.
