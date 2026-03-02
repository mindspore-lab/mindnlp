# MindTorch v2 Profiler Parity Batch21 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce remaining top-level profiler API gaps by exposing torch-compatible scheduler/state attributes and `export_stacks` entrypoint on `profile`.

**Architecture:** Promote existing internal schedule/action state (`_schedule`, `_current_action`, session step) to torch-compatible public attributes (`schedule`, `current_action`, `step_num`, `record_steps`, `action_map`, etc.) and keep them synchronized through `start/step/stop`. Implement `profile.export_stacks(path, metric)` as a delegating wrapper over `self.key_averages().export_stacks(...)`.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add RED tests for top-level scheduler/state parity

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing attribute-availability test**
- `test_profile_exposes_scheduler_state_attributes`

**Step 2: Add failing scheduler-state update test**
- `test_profile_scheduler_state_updates_step_num_and_current_action`

**Step 3: Add failing export_stacks delegating API test**
- `test_profile_export_stacks_delegates_to_key_averages`

**Step 4: Run targeted RED tests**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profile_exposes_scheduler_state_attributes tests/mindtorch_v2/test_profiler.py::test_profile_scheduler_state_updates_step_num_and_current_action tests/mindtorch_v2/test_profiler.py::test_profile_export_stacks_delegates_to_key_averages -q`

### Task 2: Implement public scheduler/state APIs + export_stacks

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Expose profile-level state fields in `__init__`**
- `schedule`, `on_trace_ready`, `step_num`, `current_action`, `record_steps`, `step_rec_fn`, `action_map`
- compatibility placeholders: `profiler`, `mem_tl`, `preset_metadata`, `record_shapes`, `profile_memory`, `with_stack`, `with_flops`, `with_modules`, `experimental_config`, `use_device`, `custom_trace_id_callback`, `execution_trace_observer`

**Step 2: Keep state synchronized through lifecycle**
- Update `step_num` and `current_action` in `start/step` consistently.
- Keep schedule-derived action transitions aligned with existing recording control.

**Step 3: Add `profile.export_stacks(path, metric=...)`**
- Delegate to `self.key_averages().export_stacks(path, metric)`.

**Step 4: Run targeted GREEN tests**
- Re-run Task 1 pytest command.

### Task 3: Validation and integration

**Files:**
- Modify: `docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch21-plan.md`

**Step 1: Run full verification suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Commit feature and docs**
- `git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
- `git commit -m "feat(mindtorch_v2): expose profiler scheduler state apis"`
- `git add docs/plans/2026-03-01-mindtorch-v2-profiler-parity-batch21-plan.md`
- `git commit -m "docs: record profiler parity batch21 verification notes"`

**Step 3: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run verification suite post-rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp`.
- Merge immediately.
- Create next worktree from latest `ms/master`.

## Verification Notes

- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profile_exposes_scheduler_state_attributes tests/mindtorch_v2/test_profiler.py::test_profile_scheduler_state_updates_step_num_and_current_action tests/mindtorch_v2/test_profiler.py::test_profile_export_stacks_delegates_to_key_averages -q`
  - Result: `3 failed` (RED), then `3 passed` (GREEN)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
  - Result: `67 passed`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
  - Result: `10 passed`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
  - Result: pass (exit code 0)
