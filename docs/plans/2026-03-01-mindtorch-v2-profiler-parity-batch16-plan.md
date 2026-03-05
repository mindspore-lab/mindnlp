# MindTorch v2 Profiler Parity Batch16 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close high-impact top-level `torch.profiler.profile` API gaps needed for compatibility (`acc_events`, metadata APIs, dynamic collection toggle, memory timeline export baseline behavior).

**Architecture:** Extend `profile` with minimal state and methods to mirror torch API surface while preserving existing tracing flow. Implement strict value-handling behavior for metadata methods and activity-based record toggling in `toggle_collection_dynamic`. Add baseline guard semantics for `export_memory_timeline` matching torch prerequisite checks.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: RED tests for top-level profile API gaps

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add/adjust tests for new APIs**
- `test_profile_exposes_acc_events_flag_and_defaults_false`
- `test_profile_accepts_acc_events_init_kwarg`
- `test_profile_add_metadata_requires_string_value`
- `test_profile_add_metadata_json_accepts_raw_string`
- `test_profile_preset_metadata_json_available_before_start`
- `test_profile_toggle_collection_dynamic_blocks_and_restores_cpu_collection`
- `test_profile_export_memory_timeline_requires_memory_related_flags`

**Step 2: Run targeted tests and confirm fail**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profile_exposes_acc_events_flag_and_defaults_false tests/mindtorch_v2/test_profiler.py::test_profile_accepts_acc_events_init_kwarg tests/mindtorch_v2/test_profiler.py::test_profile_add_metadata_requires_string_value tests/mindtorch_v2/test_profiler.py::test_profile_add_metadata_json_accepts_raw_string tests/mindtorch_v2/test_profiler.py::test_profile_preset_metadata_json_available_before_start tests/mindtorch_v2/test_profiler.py::test_profile_toggle_collection_dynamic_blocks_and_restores_cpu_collection tests/mindtorch_v2/test_profiler.py::test_profile_export_memory_timeline_requires_memory_related_flags -q`

### Task 2: GREEN implementation for missing profile APIs

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add `acc_events` support**
- Accept `acc_events` in `profile.__init__`.
- Expose `self.acc_events` with default `False`.

**Step 2: Add metadata APIs**
- Implement `add_metadata(key, value)` with torch-like string wrapping behavior (`value.replace(...)` path for type consistency).
- Implement `add_metadata_json(key, value)` and `preset_metadata_json(key, value)`.
- Include stored metadata in exported chrome trace metadata events.

**Step 3: Add collection toggle API**
- Implement `toggle_collection_dynamic(enable, activities)` by mutating active activity set.

**Step 4: Add memory timeline baseline API**
- Implement `export_memory_timeline(path, device=None)` with torch-like prerequisite guard:
  - raise `ValueError("record_shapes=True, profile_memory=True, with_stack=True required for memory profiling.")` when flags are incomplete.

**Step 5: Run targeted tests and confirm pass**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch16-plan.md`

**Step 1: Run full validation suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): add profiler top-level parity apis"`
- `git add docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch16-plan.md`
- `git commit -m "docs: record profiler parity batch16 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run validation suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profile_exposes_acc_events_flag_and_defaults_false tests/mindtorch_v2/test_profiler.py::test_profile_accepts_acc_events_init_kwarg tests/mindtorch_v2/test_profiler.py::test_profile_add_metadata_requires_string_value tests/mindtorch_v2/test_profiler.py::test_profile_add_metadata_json_accepts_raw_string tests/mindtorch_v2/test_profiler.py::test_profile_preset_metadata_json_available_before_start tests/mindtorch_v2/test_profiler.py::test_profile_toggle_collection_dynamic_blocks_and_restores_cpu_collection tests/mindtorch_v2/test_profiler.py::test_profile_export_memory_timeline_requires_memory_related_flags -q`
  - Result: `7 failed` (RED), then `7 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `55 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
