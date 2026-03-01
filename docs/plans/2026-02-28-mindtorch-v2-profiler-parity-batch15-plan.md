# MindTorch v2 Profiler Parity Batch15 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add torch-aligned stack export APIs on `key_averages()` EventList wrapper (`supported_export_stacks_metrics` and `export_stacks`).

**Architecture:** Extend `_KeyAverages` with stack-export methods matching torch API and formatting semantics, including supported metric validation and metric-name mapping from cuda/xpu/privateuse1 to device timing fields. Extend row objects with device-time attributes needed by mapped metrics.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: RED tests for stack export parity

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add test for supported metrics list**
- `test_key_averages_supported_export_stacks_metrics_matches_torch_shape`

**Step 2: Add test for unsupported metric error**
- `test_key_averages_export_stacks_rejects_unsupported_metric`

**Step 3: Add test for writing exported stack lines**
- `test_key_averages_export_stacks_writes_stack_lines`

**Step 4: Run targeted tests and confirm fail**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supported_export_stacks_metrics_matches_torch_shape tests/mindtorch_v2/test_profiler.py::test_key_averages_export_stacks_rejects_unsupported_metric tests/mindtorch_v2/test_profiler.py::test_key_averages_export_stacks_writes_stack_lines -q`

### Task 2: GREEN implementation for stack export APIs

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add EventList stack-export APIs**
- Implement `_KeyAverages.supported_export_stacks_metrics()`.
- Implement `_KeyAverages.export_stacks(path, metric)` with torch-like validation and text formatting.

**Step 2: Add row device-time attributes for metric mapping**
- Extend `_FunctionEventAvgRow` with:
  - `self_device_time_total`
  - `device_time_total`
  - `device_time`
- Populate corresponding nanosecond fields in grouped rows and total row defaults.

**Step 3: Run targeted tests and confirm pass**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch15-plan.md`

**Step 1: Run validation suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add key_averages stack export apis"`
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch15-plan.md`
- `git commit -m "docs: record profiler parity batch15 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run validation suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_supported_export_stacks_metrics_matches_torch_shape tests/mindtorch_v2/test_profiler.py::test_key_averages_export_stacks_rejects_unsupported_metric tests/mindtorch_v2/test_profiler.py::test_key_averages_export_stacks_writes_stack_lines -q`
  - Result: `3 failed` (RED), then `3 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `48 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
