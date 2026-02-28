# MindTorch v2 Profiler Parity Batch13 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `key_averages()` EventList aggregate APIs with torch by adding `self_cpu_time_total` and `total_average()` semantics, including empty-session behavior.

**Architecture:** Extend `_KeyAverages` with aggregate helpers that compute totals from cached grouped rows, then expose torch-like aggregate APIs on the EventList wrapper. Reuse the existing row-wrapper type for `total_average()` with a synthetic `Total` row so row-level and total-level unit conventions stay consistent.

**Tech Stack:** Python, pytest, mindtorch_v2 profiler.

---

### Task 1: Add failing parity tests for EventList aggregates

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Add failing test for `self_cpu_time_total` aggregate**

```python
def test_key_averages_exposes_self_cpu_time_total_aggregate():
    with torch.profiler.profile() as prof:
        x = torch.ones((4, 4))
        _ = x + x
        _ = x * x

    rows = prof.key_averages()
    per_row_sum = sum(row.self_cpu_time_total for row in rows)

    assert rows.self_cpu_time_total == pytest.approx(per_row_sum)
```

**Step 2: Add failing test for `total_average()` object semantics**

```python
def test_key_averages_total_average_returns_total_row():
    with torch.profiler.profile() as prof:
        x = torch.ones((4, 4))
        _ = x + x

    rows = prof.key_averages()
    total = rows.total_average()

    assert total.key == "Total"
    assert total.count == sum(row.count for row in rows)
    assert total.self_cpu_time_total == pytest.approx(rows.self_cpu_time_total)
```

**Step 3: Add failing test for empty-session aggregate behavior**

```python
def test_key_averages_empty_events_total_average_is_zero():
    with torch.profiler.profile():
        pass

    rows = prof.key_averages()
    total = rows.total_average()

    assert len(rows) == 0
    assert rows.self_cpu_time_total == 0
    assert total.key == "Total"
    assert total.count == 0
    assert total.self_cpu_time_total == 0
    assert total.cpu_time_total == 0
    assert total.cpu_time == 0.0
```

**Step 4: Run targeted tests to verify RED**

Run:
`PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_exposes_self_cpu_time_total_aggregate tests/mindtorch_v2/test_profiler.py::test_key_averages_total_average_returns_total_row tests/mindtorch_v2/test_profiler.py::test_key_averages_empty_events_total_average_is_zero -q`

Expected: fail with missing `_KeyAverages` attributes (`self_cpu_time_total` / `total_average`).

### Task 2: Implement minimal aggregate APIs in profiler

**Files:**
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Add aggregate-total helper on `_KeyAverages`**
- Build a helper that sums `self_time_ns`, `total_time_ns`, and `count` from `_build_rows()`.
- Return zeroed totals when there are no rows.

**Step 2: Add `self_cpu_time_total` property**
- Expose aggregate self time in microseconds (`ns / 1000.0`).

**Step 3: Add `total_average()` method**
- Construct a synthetic row with `name="Total"` plus aggregate counts/times.
- Return `_FunctionEventAvgRow(total_row)`.
- Ensure empty rows return all zero values and avoid divide-by-zero.

**Step 4: Keep current row/table behavior unchanged**
- Do not modify existing table sorting or formatting behavior in this batch.

**Step 5: Run targeted tests to verify GREEN**

Run:
`PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_exposes_self_cpu_time_total_aggregate tests/mindtorch_v2/test_profiler.py::test_key_averages_total_average_returns_total_row tests/mindtorch_v2/test_profiler.py::test_key_averages_empty_events_total_average_is_zero -q`

Expected: all pass.

**Step 6: Commit feature change**

```bash
git add src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py
git commit -m "feat(mindtorch_v2): add key_averages aggregate parity apis"
```

### Task 3: Full validation and integration

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch13-plan.md`

**Step 1: Run validation suite**
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
- `python -m py_compile src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`

**Step 2: Append verification notes**
- Add a `## Verification Notes` section with exact commands and pass/fail outcomes.

**Step 3: Commit docs note**

```bash
git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch13-plan.md
git commit -m "docs: record profiler parity batch13 verification notes"
```

**Step 4: Fast-mode integration**
- Rebase latest `ms/master`.
- Re-run validation commands after rebase.
- Push branch to `origin`.
- Create PR to `mindspore-lab/mindnlp` with clean English line breaks.
- Merge immediately (`gh pr merge --merge --delete-branch`).
- Create next worktree from latest `ms/master` for the next batch.
