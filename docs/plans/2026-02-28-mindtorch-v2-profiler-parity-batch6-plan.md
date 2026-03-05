# MindTorch v2 Profiler Parity Batch6 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement NPU per-event memory delta capture when `profile_memory=True`, and export memory metadata through profiler events and Chrome trace.

**Architecture:** Extend profiler event metadata pipeline in `src/mindtorch_v2/profiler/profiler.py` to capture NPU memory allocated before/after each recorded event and derive delta; keep all memory instrumentation disabled unless `profile_memory=True`.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler and npu runtime APIs.

---

### Task 1: Add failing tests for NPU `profile_memory` behavior

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_profiler_profile_memory_disabled_has_no_npu_memory_fields`
- `test_profiler_profile_memory_adds_npu_memory_fields` (NPU-gated)
- `test_export_chrome_trace_includes_npu_memory_fields_when_enabled` (NPU-gated)

**Step 2: Run tests to confirm failures**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_profile_memory_disabled_has_no_npu_memory_fields tests/mindtorch_v2/test_profiler.py::test_profiler_profile_memory_adds_npu_memory_fields tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_npu_memory_fields_when_enabled -q`

### Task 2: Implement per-event NPU memory metadata

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add profile-memory runtime gate**
- Store `profile_memory` flag in session/profile state.
- Keep default path unchanged and low-overhead when disabled.

**Step 2: Add NPU memory snapshot helpers**
- Read allocated NPU memory via `mindtorch_v2.npu.memory_allocated()` only when:
  - `profile_memory=True`
  - dispatch event is NPU event
  - session is recording

**Step 3: Attach per-event metadata**
- Add event metadata keys:
  - `npu_memory_allocated_before`
  - `npu_memory_allocated_after`
  - `npu_memory_allocated_delta`
- Guarantee `delta = after - before`.

**Step 4: Run targeted tests**
- Re-run Task 1 tests.

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add NPU per-event profile_memory metadata"`

### Task 3: Validate schedule compatibility and no-regression

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py` (if extra schedule-specific assertion needed)

**Step 1: Optional schedule assertion test**
- Ensure memory metadata appears only in recording phases under schedule.

**Step 2: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 3: Run nearby regressions**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 4: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 5: Commit (if changed)**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "test(mindtorch_v2): cover profile_memory schedule behavior"`

### Task 4: Record verification notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch6-plan.md`

**Step 1: Append validation outputs and conclusions**

**Step 2: Commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch6-plan.md`
- `git commit -m "docs: record profiler parity batch6 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add profile_memory tests (default-off + NPU-enabled + trace export)
- Task 2: implement per-event NPU memory before/after/delta metadata
- Task 3: run profiler and nearby regression validations
- Task 4: record notes

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_profile_memory_disabled_has_no_npu_memory_fields tests/mindtorch_v2/test_profiler.py::test_profiler_profile_memory_adds_npu_memory_fields tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_npu_memory_fields_when_enabled -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`28 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- `profile_memory=True` now records per-op NPU fields:
  - `npu_memory_allocated_before`
  - `npu_memory_allocated_after`
  - `npu_memory_allocated_delta`
- Default path (`profile_memory=False`) does not emit memory fields.
- Export path includes memory metadata under `traceEvents[].args` for NPU events.
