# MindTorch v2 Profiler Parity Batch1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve `mindtorch_v2.profiler` API behavior parity for lifecycle, activity validation, callback semantics, and Chrome trace contract while keeping MindTorch v2 free of any `mindspore` dependency.

**Architecture:** Extend the existing lightweight profiler core in `src/mindtorch_v2/profiler/profiler.py` with stricter API semantics (`activities`, `on_trace_ready`, export fields) and verify behavior with targeted TDD updates in `tests/mindtorch_v2/test_profiler.py`.

**Tech Stack:** Python, pytest, mindtorch_v2 dispatcher/profiler runtime, standard library (`os`, `json`, `threading`, `time`).

---

### Task 1: Activity validation parity (`activities`)

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
def test_profiler_rejects_unknown_activity():
    with pytest.raises(ValueError):
        torch.profiler.profile(activities=["TPU"])
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_rejects_unknown_activity -q`
Expected: FAIL because current implementation silently accepts unknown activity strings.

**Step 3: Write minimal implementation**

- Restrict activities to supported aliases: `CPU`, `NPU`, `CUDA`, `GPU`.
- Normalize `CUDA/GPU` to `NPU`.
- Raise `ValueError` for unknown activity values.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_rejects_unknown_activity -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "test(mindtorch_v2): enforce profiler activity validation"
```

### Task 2: `on_trace_ready` callback behavior parity

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing tests**

```python
def test_on_trace_ready_receives_profiler_instance():
    seen = []

    def callback(prof):
        seen.append(prof)

    with torch.profiler.profile(on_trace_ready=callback) as prof:
        _ = torch.ones((2, 2)) + 1

    assert seen == [prof]


def test_on_trace_ready_type_error_from_callback_is_not_swallowed():
    def callback(prof):
        raise TypeError("callback boom")

    with pytest.raises(TypeError, match="callback boom"):
        with torch.profiler.profile(on_trace_ready=callback):
            _ = torch.ones((2, 2)) + 1
```

**Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_on_trace_ready_receives_profiler_instance tests/mindtorch_v2/test_profiler.py::test_on_trace_ready_type_error_from_callback_is_not_swallowed -q`
Expected: second test FAILs because current implementation catches `TypeError` and re-calls callback incorrectly.

**Step 3: Write minimal implementation**

- Invoke callback exactly once as `on_trace_ready(self)`.
- Do not catch and mask callback `TypeError`.

**Step 4: Run tests to verify pass**

Run: same pytest command as step 2.
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "feat(mindtorch_v2): align profiler on_trace_ready callback semantics"
```

### Task 3: Chrome trace contract parity improvements

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
import os


def test_export_chrome_trace_required_fields_and_pid(tmp_path):
    out = tmp_path / "trace.json"
    with torch.profiler.profile() as prof:
        _ = torch.ones((2, 2)) + 1

    prof.export_chrome_trace(str(out))
    payload = json.loads(out.read_text())
    event = payload["traceEvents"][0]

    for key in ("name", "ph", "ts", "dur", "pid", "tid"):
        assert key in event
    assert event["pid"] == os.getpid()
```

**Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_required_fields_and_pid -q`
Expected: FAIL because current implementation sets `pid=0`.

**Step 3: Write minimal implementation**

- Set `pid` to real process id (`os.getpid()`).
- Keep Chrome trace schema fields stable.

**Step 4: Run test to verify pass**

Run: same pytest command as step 2.
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "feat(mindtorch_v2): improve chrome trace contract fields"
```

### Task 4: Lifecycle and no-op behavior guard tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py` (only if needed)

**Step 1: Write tests first**

```python
def test_record_function_is_noop_when_profiler_inactive():
    with torch.profiler.record_function("noop"):
        x = torch.ones((1,))
    assert x is not None


def test_profiler_step_requires_active_session():
    prof = torch.profiler.profile()
    with pytest.raises(RuntimeError):
        prof.step()
```

**Step 2: Run tests**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_record_function_is_noop_when_profiler_inactive tests/mindtorch_v2/test_profiler.py::test_profiler_step_requires_active_session -q`
Expected: PASS or targeted fixes if regression appears.

**Step 3: Minimal implementation (if required)**

Adjust lifecycle checks only if tests fail.

**Step 4: Re-run tests**

Run: same command as step 2.
Expected: PASS.

**Step 5: Commit (only if code changed)**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "test(mindtorch_v2): add profiler lifecycle guard coverage"
```

### Task 5: Final validation and delivery

**Files:**
- Modify: `docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch1-plan.md` (append verification notes)

**Step 1: Run focused profiler suite**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
Expected: PASS.

**Step 2: Run adjacent regression guard**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`
Expected: PASS.

**Step 3: Syntax sanity**

Run: `python -m py_compile src/mindtorch_v2/profiler/common.py src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py`
Expected: PASS.

**Step 4: Commit validation notes**

```bash
git add docs/plans/2026-02-28-mindtorch-v2-profiler-parity-batch1-plan.md
git commit -m "docs: record profiler parity batch1 verification notes"
```

## Verification Notes

Execution date: 2026-02-28

Completed tasks:
- Task 1: activity validation (`ValueError` on unsupported values)
- Task 2: `on_trace_ready` callback semantics alignment
- Task 3: chrome trace `pid` uses real process id
- Task 4: lifecycle/no-op guard tests

Executed commands and outcomes:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_rejects_unknown_activity -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_on_trace_ready_receives_profiler_instance tests/mindtorch_v2/test_profiler.py::test_on_trace_ready_type_error_from_callback_is_not_swallowed -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_required_fields_and_pid -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_record_function_is_noop_when_profiler_inactive tests/mindtorch_v2/test_profiler.py::test_profiler_step_requires_active_session -q` -> PASS
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q` -> PASS (`14 passed`)
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q` -> PASS (`10 passed`)
- `python -m py_compile src/mindtorch_v2/profiler/common.py src/mindtorch_v2/profiler/profiler.py tests/mindtorch_v2/test_profiler.py` -> PASS

Notes:
- This batch keeps `mindtorch_v2` profiler independent from `mindspore`.
- No dispatcher behavior changes were required in this batch.
