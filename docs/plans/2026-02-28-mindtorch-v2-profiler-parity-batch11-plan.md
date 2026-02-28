# MindTorch v2 Profiler Parity Batch11 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve chrome trace structure parity by adding trace metadata events and runtime-category correlated events.

**Architecture:** Keep existing op/scope `ph: X` export path, then append additional metadata (`ph: M`) and runtime correlated events (`cat: runtime`) derived from op correlation fields.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing trace-structure tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_export_chrome_trace_includes_metadata_events`
- `test_export_chrome_trace_includes_runtime_correlated_events`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_metadata_events tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_runtime_correlated_events -q`

### Task 2: Implement trace metadata/runtime event export

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add metadata events**
- Append `ph: M` events for process/thread labeling at export time.

**Step 2: Add runtime correlated events**
- For each op event with `correlation_id`, append `cat: runtime` `ph: X` event with correlation args.

**Step 3: Preserve backward compatibility**
- Keep existing op/scope export layout and args content unchanged.
- Ensure original tests that inspect first event shape still pass.

**Step 4: Run targeted tests**
- Re-run Task 1 tests.

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): enrich chrome trace with metadata and runtime events"`

### Task 3: Full validation and notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch11-plan.md`

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regressions**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Record and commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch11-plan.md`
- `git commit -m "docs: record profiler parity batch11 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add failing trace-structure tests
- Task 2: implement metadata events and runtime-correlated events in trace export
- Task 3: full validation and syntax checks

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_metadata_events tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_runtime_correlated_events -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`38 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- Added trace metadata events (`ph: M`) for process/thread naming.
- Added runtime correlated events (`cat: runtime`, `ph: X`) for op events carrying `correlation_id`.
- Kept existing op/scope `ph: X` events and args behavior intact.
