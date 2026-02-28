# MindTorch v2 Profiler Parity Batch3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement torch-aligned `record_shapes` and `with_stack` runtime behavior for `mindtorch_v2.profiler` with low overhead when flags are disabled.

**Architecture:** Extend event metadata capture in `src/mindtorch_v2/profiler/profiler.py` so op/scope events optionally include input shape metadata and frame-based stack metadata; keep collection gated by `record_shapes`/`with_stack` flags.

**Tech Stack:** Python, pytest, inspect/traceback standard library, mindtorch_v2 profiler runtime.

---

### Task 1: Add failing tests for `record_shapes` and `with_stack`

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Write failing tests**
- `test_profiler_record_shapes_captures_tensor_shapes`
- `test_profiler_with_stack_captures_frame_metadata`
- `test_profiler_default_flags_do_not_emit_shape_or_stack`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_record_shapes_captures_tensor_shapes tests/mindtorch_v2/test_profiler.py::test_profiler_with_stack_captures_frame_metadata tests/mindtorch_v2/test_profiler.py::test_profiler_default_flags_do_not_emit_shape_or_stack -q`

### Task 2: Implement optional shape/stack metadata capture

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Implement shape capture**
- Parse positional/keyword arguments in `dispatch_op_enter`.
- When `record_shapes=True`, attach tensor shapes (and tensor list shapes where applicable) to event metadata.
- Keep metadata lightweight and JSON-serializable.

**Step 2: Implement stack capture (Plan 1)**
- When `with_stack=True`, capture frame-based stack entries in `file:line:function` style.
- Include stack metadata in events for both op and `record_function` scope events.
- Exclude profiler internals from captured stack where practical.

**Step 3: Ensure disabled flags have near-zero overhead**
- Fast-path skip for metadata work if both flags are false.

**Step 4: Run targeted tests**
- Run the three tests from Task 1.

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add profiler record_shapes and with_stack metadata"`

### Task 3: Integrate metadata into exports and snapshots

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Extend `events()` output contract**
- Include optional keys only when enabled (e.g., `input_shapes`, `stack`).

**Step 2: Extend chrome trace args**
- Include shape/stack fields in `traceEvents[].args` when present.
- Keep output valid JSON and backward-compatible when flags are off.

**Step 3: Add export-side tests**
- `test_export_chrome_trace_includes_shape_and_stack_args_when_enabled`

**Step 4: Run targeted tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_shape_and_stack_args_when_enabled -q`

**Step 5: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): export profiler shape and stack metadata to trace"`

### Task 4: Final validation

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch3-plan.md` (append verification notes)

**Step 1: Run profiler test suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regression checks**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax checks**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Commit verification notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch3-plan.md`
- `git commit -m "docs: record profiler parity batch3 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add failing tests for `record_shapes` and `with_stack`
- Task 2: implement optional shape/stack metadata capture
- Task 3: integrate metadata into `events()` and chrome trace args
- Task 4: final validation

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_record_shapes_captures_tensor_shapes tests/mindtorch_v2/test_profiler.py::test_profiler_with_stack_captures_frame_metadata tests/mindtorch_v2/test_profiler.py::test_profiler_default_flags_do_not_emit_shape_or_stack tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_includes_shape_and_stack_args_when_enabled -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`22 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- Metadata collection is fully gated by `record_shapes`/`with_stack`.
- Default path keeps metadata disabled to minimize overhead.
- `dispatch_op_enter` now receives op args/kwargs to build shape metadata.
