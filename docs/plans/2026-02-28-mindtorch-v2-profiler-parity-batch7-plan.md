# MindTorch v2 Profiler Parity Batch7 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `profile_memory=True` behavior to CPU events with per-event memory delta metadata, aligned with existing NPU implementation style.

**Architecture:** Add CPU memory snapshot support to profiler metadata pipeline in `src/mindtorch_v2/profiler/profiler.py` and emit per-event CPU memory fields alongside existing NPU fields; keep default path unchanged when `profile_memory=False`.

**Tech Stack:** Python, pytest, standard library (`tracemalloc`), mindtorch_v2 profiler runtime.

---

### Task 1: Add failing CPU profile_memory tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_profiler_profile_memory_adds_cpu_memory_fields`
- `test_export_chrome_trace_includes_cpu_memory_fields_when_enabled`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_profile_memory_adds_cpu_memory_fields tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_cpu_memory_fields_when_enabled -q`

### Task 2: Implement CPU per-event memory metadata

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add CPU snapshot helper**
- Use `tracemalloc.get_traced_memory()` current usage value as CPU allocated proxy.
- Start/stop `tracemalloc` only when `profile_memory=True` and CPU activity enabled.

**Step 2: Add per-event CPU fields**
- For CPU op events under `profile_memory=True`, add:
  - `cpu_memory_allocated_before`
  - `cpu_memory_allocated_after`
  - `cpu_memory_allocated_delta`
- Keep `delta = after - before`.

**Step 3: Keep compatibility with NPU memory fields**
- Do not regress existing NPU profile_memory behavior.

**Step 4: Run targeted tests**
- Re-run Task 1 tests.

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add CPU per-event profile_memory metadata"`

### Task 3: Full validation and notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch7-plan.md`

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regressions**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Record and commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch7-plan.md`
- `git commit -m "docs: record profiler parity batch7 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add failing CPU profile_memory tests
- Task 2: implement CPU per-event memory before/after/delta capture
- Task 3: full profiler and regression validation

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_profile_memory_adds_cpu_memory_fields tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_cpu_memory_fields_when_enabled -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`30 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- `profile_memory=True` now emits CPU per-event fields:
  - `cpu_memory_allocated_before`
  - `cpu_memory_allocated_after`
  - `cpu_memory_allocated_delta`
- CPU memory tracking uses `tracemalloc` only during active profiler session and only when needed.
- Existing NPU memory fields remain functional and unchanged.
