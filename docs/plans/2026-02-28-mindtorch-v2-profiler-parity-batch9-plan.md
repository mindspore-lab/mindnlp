# MindTorch v2 Profiler Parity Batch9 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve trace-event model parity by adding runtime/correlation metadata so exported traces are more torch/kineto-like and easier to analyze in Perfetto/Chrome.

**Architecture:** Enrich per-op event metadata with runtime context fields (`correlation_id`, `runtime_name`, `runtime_tid`) and export them through `traceEvents[].args`; maintain backward compatibility and keep disabled fast paths lightweight.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing trace-model parity tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_profiler_events_include_correlation_id_for_ops`
- `test_export_chrome_trace_includes_runtime_correlation_fields`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_events_include_correlation_id_for_ops tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_runtime_correlation_fields -q`

### Task 2: Implement runtime/correlation metadata for op events

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add correlation id generator**
- Assign monotonic correlation id per recorded op event.

**Step 2: Add runtime metadata fields to op events**
- `correlation_id`
- `runtime_name` (initial value: `dispatch_kernel`)
- `runtime_tid` (thread id)

**Step 3: Preserve existing metadata composition**
- Keep shape/stack/memory fields intact.
- Apply only to op events; scope events remain unchanged for now.

**Step 4: Ensure trace export includes new fields**
- Include these metadata keys in `traceEvents[].args`.

**Step 5: Run targeted tests**
- Re-run Task 1 tests.

**Step 6: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): enrich trace events with runtime correlation metadata"`

### Task 3: Full validation and notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch9-plan.md`

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regressions**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Record and commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch9-plan.md`
- `git commit -m "docs: record profiler parity batch9 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add failing tests for runtime/correlation trace metadata
- Task 2: implement op-event correlation/runtime metadata and trace export propagation
- Task 3: full profiler/regression/syntax validation

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_events_include_correlation_id_for_ops tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_runtime_correlation_fields -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`34 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- Added op-level metadata keys:
  - `correlation_id`
  - `runtime_name` (current value: `dispatch_kernel`)
  - `runtime_tid`
- Metadata is available in both `prof.events()` and `traceEvents[].args`.
