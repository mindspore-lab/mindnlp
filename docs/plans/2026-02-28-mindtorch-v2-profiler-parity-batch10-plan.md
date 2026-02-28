# MindTorch v2 Profiler Parity Batch10 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve key_averages/table output parity by adding torch-like CPU time columns and sorting aliases while keeping existing behavior stable.

**Architecture:** Extend `_KeyAverages.table(...)` rendering with additional torch-style columns (`Self CPU`, `CPU total`, `CPU time avg`, `# of Calls`) and support equivalent sort key aliases.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing table-parity tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add tests**
- `test_key_averages_table_includes_torch_like_cpu_columns`
- `test_key_averages_table_accepts_cpu_time_avg_sort_alias`

**Step 2: Run tests to confirm failure**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_table_includes_torch_like_cpu_columns tests/mindtorch_v2/test_profiler.py::test_key_averages_table_accepts_cpu_time_avg_sort_alias -q`

### Task 2: Implement table columns and sort aliases

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Extend sort map aliases**
- Add support for aliases like:
  - `self_cpu_time`
  - `cpu_time`
  - `cpu_time_avg`

**Step 2: Extend table output columns**
- Include torch-like textual columns:
  - `Self CPU`
  - `CPU total`
  - `CPU time avg`
  - `# of Calls`
- Keep previous information available and deterministic ordering.

**Step 3: Run targeted tests**
- Re-run Task 1 tests.

**Step 4: Commit**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): improve key_averages table cpu column parity"`

### Task 3: Full validation and notes

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch10-plan.md`

**Step 1: Run profiler suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`

**Step 2: Run nearby regressions**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

**Step 3: Syntax check**
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 4: Record and commit notes**
- `git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch10-plan.md`
- `git commit -m "docs: record profiler parity batch10 verification notes"`

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: add table-parity failing tests
- Task 2: implement CPU table columns and sort aliases
- Task 3: full validation and syntax checks

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_table_includes_torch_like_cpu_columns tests/mindtorch_v2/test_profiler.py::test_key_averages_table_accepts_cpu_time_avg_sort_alias -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`36 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- Table now includes torch-like CPU columns:
  - `Self CPU`
  - `CPU total`
  - `CPU time avg`
  - `# of Calls`
- Backward-compatible columns preserved for existing tests:
  - `Count`
  - `Total(us)`
- Added sort aliases:
  - `self_cpu_time`
  - `cpu_time`
  - `cpu_time_avg`
