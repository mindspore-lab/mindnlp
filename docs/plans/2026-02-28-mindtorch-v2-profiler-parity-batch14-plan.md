# MindTorch v2 Profiler Parity Batch14 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `FunctionEventAvg` row-level APIs with torch by adding missing row attributes and removing non-torch `__getitem__` access.

**Architecture:** Extend grouped row materialization in `_KeyAverages` so row dictionaries carry torch-like metadata (device, memory, shape/stack grouping, async/scope flags), then expose these via `_FunctionEventAvgRow` attributes. Preserve existing aggregate/table APIs while removing `__getitem__` from row objects for strict torch semantics.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing tests for row attribute parity and strict access semantics

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing test for default row attribute set**

```python
def test_key_averages_row_exposes_torch_like_default_attributes():
    with torch.profiler.profile() as prof:
        x = torch.ones((4, 4))
        _ = x + x

    row = next(iter(prof.key_averages()))

    assert row.device_type == "CPU"
    assert row.input_shapes == ""
    assert row.stack == []
    assert row.cpu_memory_usage == 0
    assert row.self_cpu_memory_usage == 0
    assert row.device_memory_usage == 0
    assert row.self_device_memory_usage == 0
    assert row.flops == 0
    assert row.is_async is False
    assert row.scope == 0
    assert row.use_device is None
    assert row.is_user_annotation is False
```

**Step 2: Add failing test for memory/grouping-backed row fields**

```python
def test_key_averages_row_populates_memory_and_input_shape_fields():
    with torch.profiler.profile(record_shapes=True, profile_memory=True) as prof:
        x = torch.ones((4, 4))
        _ = x + x

    row = next(iter(prof.key_averages(group_by_input_shape=True)))

    assert isinstance(row.input_shapes, list)
    assert row.cpu_memory_usage >= row.self_cpu_memory_usage
```

**Step 3: Add failing test for removed `__getitem__` compatibility path**

```python
def test_key_averages_row_is_not_subscriptable_like_torch():
    with torch.profiler.profile() as prof:
        x = torch.ones((4, 4))
        _ = x + x

    row = next(iter(prof.key_averages()))

    with pytest.raises(TypeError):
        _ = row["key"]
```

**Step 4: Run targeted tests to verify RED**

Run:
`PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_row_exposes_torch_like_default_attributes tests/mindtorch_v2/test_profiler.py::test_key_averages_row_populates_memory_and_input_shape_fields tests/mindtorch_v2/test_profiler.py::test_key_averages_row_is_not_subscriptable_like_torch -q`

Expected: failures due to missing row attributes and current subscriptable behavior.

### Task 2: Implement minimal row-attribute parity in profiler

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Extend `_KeyAverages._build_rows()` to aggregate additional fields**
- Populate defaults on each row:
  - `input_shapes` default `""` (when not grouped)
  - `stack` default `[]`
  - `cpu_memory_usage`, `self_cpu_memory_usage`, `device_memory_usage`, `self_device_memory_usage`, `flops` default `0`
  - `is_async` default `False`, `scope` default `0`, `use_device` default `None`, `is_user_annotation` default `False`
- When profile metadata exists, sum memory/flops fields per grouped row.

**Step 2: Extend `_FunctionEventAvgRow` attributes**
- Add attribute mapping for all torch-like row fields listed above.
- Keep existing timing/key/count fields unchanged.

**Step 3: Remove `__getitem__` from `_FunctionEventAvgRow`**
- Delete the method so row indexing raises `TypeError`, matching torch.

**Step 4: Run targeted tests to verify GREEN**

Run:
`PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_row_exposes_torch_like_default_attributes tests/mindtorch_v2/test_profiler.py::test_key_averages_row_populates_memory_and_input_shape_fields tests/mindtorch_v2/test_profiler.py::test_key_averages_row_is_not_subscriptable_like_torch -q`

Expected: all pass.

**Step 5: Commit feature changes**

```bash
git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py
git commit -m "feat(mindtorch_v2): align profiler row attributes with torch"
```

### Task 3: Full validation, docs notes, and integration

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch14-plan.md`

**Step 1: Run validation suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Append verification notes**
- Add `## Verification Notes` with exact commands and outcomes.

**Step 3: Commit docs note**

```bash
git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch14-plan.md
git commit -m "docs: record profiler parity batch14 verification notes"
```

**Step 4: Fast-mode integration**
- Rebase latest `ms/master`
- Re-run validation commands after rebase
- Push branch to `origin`
- Create PR to `mindspore-lab/mindnlp` (English, clean line breaks)
- Merge immediately (`gh pr merge --merge --delete-branch`)
- Create next worktree from latest `ms/master`
