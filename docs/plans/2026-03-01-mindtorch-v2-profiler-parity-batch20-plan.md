# MindTorch v2 Profiler Parity Batch20 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining high-value EventList parity gap by adding `key_averages().export_chrome_trace(path)` with torch-compatible trace JSON shape.

**Architecture:** Reuse existing `_KeyAverages` row objects and export a lightweight chrome-trace-compatible JSON with `traceEvents` entries for averaged rows. Keep method behavior minimal and deterministic: create file, write valid JSON, include row-level timing fields and names.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add RED tests for EventList export API

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing test for method availability and JSON validity**
- `test_key_averages_export_chrome_trace_writes_valid_json`

**Step 2: Add failing test for exported event payload basics**
- `test_key_averages_export_chrome_trace_contains_row_entries`

**Step 3: Run targeted RED tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_export_chrome_trace_writes_valid_json tests/mindtorch_v2/test_profiler.py::test_key_averages_export_chrome_trace_contains_row_entries -q`

### Task 2: Implement EventList export_chrome_trace

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add `_KeyAverages.export_chrome_trace(path)`**
- Emit JSON payload with top-level `traceEvents` list.
- Include one event per row object with required fields (`name`, `ph`, `ts`, `dur`, `pid`, `tid`, `cat`, `args`).
- Keep output deterministic and utf-8 encoded.

**Step 2: Ensure basic torch compatibility expectations**
- Method exists on EventList wrapper.
- Output file is parseable JSON and non-empty when rows exist.

**Step 3: Run targeted GREEN tests**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch20-plan.md`

**Step 1: Run full verification suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add key_averages export_chrome_trace"`
- `git add docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch20-plan.md`
- `git commit -m "docs: record profiler parity batch20 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run verification suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_export_chrome_trace_writes_valid_json tests/mindtorch_v2/test_profiler.py::test_key_averages_export_chrome_trace_contains_row_entries -q`
  - Result: `2 failed` (RED), then `2 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `64 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
