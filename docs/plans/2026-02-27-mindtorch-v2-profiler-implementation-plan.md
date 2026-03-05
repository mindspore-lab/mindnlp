# MindTorch v2 Profiler MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a `mindtorch_v2` profiler MVP aligned with torch core behavior for `profile`, `record_function`, `step`, `key_averages().table`, and `export_chrome_trace`, with op-level coverage across forward/backward/optimizer on CPU+NPU.

**Architecture:** Build a standalone `mindtorch_v2.profiler` module with thread-local active session state and event buffering; integrate low-overhead dispatch hooks at kernel execution path to collect op events; support user scopes via `record_function`; aggregate lazily for `key_averages`; export Chrome trace JSON.

**Tech Stack:** Python (`mindtorch_v2` runtime/dispatcher), pytest, standard library (`time`, `json`, `threading`, `contextlib`).

---

### Task 1: Add failing API and lifecycle tests (CPU)

**Files:**
- Create: `tests/mindtorch_v2/test_profiler.py`
- Test: `tests/mindtorch_v2/test_profiler.py`

**Step 1: Write the failing test**

```python
import mindtorch_v2 as torch


def test_profiler_context_and_step_basics():
    x = torch.ones((2, 2))
    with torch.profiler.profile() as prof:
        y = x + x
        prof.step()
        z = y * y
    assert z is not None
    assert len(prof.events()) >= 2
    assert {e["step"] for e in prof.events()} == {0, 1}
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_context_and_step_basics -q`
Expected: FAIL because `mindtorch_v2.profiler` API does not exist yet.

**Step 3: Write minimal implementation**

Create profiler module skeleton and context manager shape that can enter/exit, collect no-op events list, and expose `step()` + `events()`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_context_and_step_basics -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler src/mindtorch_v2/__init__.py
git commit -m "feat(mindtorch_v2): add profiler api skeleton"
```

### Task 2: Add failing dispatcher op-event tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
def test_profiler_captures_dispatch_ops():
    x = torch.ones((2, 2), requires_grad=True)
    with torch.profiler.profile() as prof:
        y = (x * x).sum()
        y.backward()
    names = [e["name"] for e in prof.events() if e["kind"] == "op"]
    assert any("mul" in n for n in names)
    assert any("sum" in n for n in names)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_captures_dispatch_ops -q`
Expected: FAIL because dispatcher is not instrumented.

**Step 3: Write minimal implementation**

- Add profiler fast-path guard helpers (`is_enabled`, `record_op_start/end`).
- In `dispatch_with_keyset` `_run_kernel`, record op begin/end around kernel call.
- Ensure profiler-disabled path stays a direct branch without extra allocations.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_captures_dispatch_ops -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/profiler/profiler.py
git commit -m "feat(mindtorch_v2): hook profiler op events into dispatcher"
```

### Task 3: Add failing record_function nesting tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
def test_record_function_nesting_and_exception_safe():
    with torch.profiler.profile() as prof:
        with torch.profiler.record_function("outer"):
            with torch.profiler.record_function("inner"):
                _ = torch.ones((2, 2)) + 1
    scopes = [e for e in prof.events() if e["kind"] == "scope"]
    assert [s["name"] for s in scopes] == ["outer", "inner"]
    assert all(s["duration_ns"] >= 0 for s in scopes)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_record_function_nesting_and_exception_safe -q`
Expected: FAIL because record_function is missing or incomplete.

**Step 3: Write minimal implementation**

Implement `record_function(name)` context manager with thread-local stack and guaranteed close in `finally`.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_record_function_nesting_and_exception_safe -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "feat(mindtorch_v2): add record_function nested scope events"
```

### Task 4: Add failing key_averages and table tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
def test_key_averages_table_contains_expected_columns():
    x = torch.ones((4, 4))
    with torch.profiler.profile() as prof:
        _ = x + x
        _ = x * x
    stats = prof.key_averages()
    table = stats.table(sort_by="self_cpu_time_total")
    assert "Name" in table
    assert "Count" in table
    assert "Total" in table
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_table_contains_expected_columns -q`
Expected: FAIL because aggregation/table is missing.

**Step 3: Write minimal implementation**

Implement:
- lazy grouping by `(name, device_type)`
- `count`, `total_time`, `self_time`, `avg_time`
- deterministic table ordering and stable string rendering

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_key_averages_table_contains_expected_columns -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "feat(mindtorch_v2): implement key_averages and table"
```

### Task 5: Add failing chrome trace export tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
import json


def test_export_chrome_trace_json_valid(tmp_path):
    x = torch.ones((2, 2))
    out = tmp_path / "trace.json"
    with torch.profiler.profile() as prof:
        _ = x + x
    prof.export_chrome_trace(str(out))
    payload = json.loads(out.read_text())
    assert "traceEvents" in payload
    assert len(payload["traceEvents"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_json_valid -q`
Expected: FAIL because export is missing.

**Step 3: Write minimal implementation**

Implement `export_chrome_trace(path)` emitting Chrome trace events from stored op/scope events.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_export_chrome_trace_json_valid -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py
git commit -m "feat(mindtorch_v2): add chrome trace export"
```

### Task 6: Add failing optimizer-phase and dependency-guard tests

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `src/mindtorch_v2/profiler/profiler.py`

**Step 1: Write the failing test**

```python
from mindtorch_v2 import nn, optim


def test_profiler_covers_optimizer_ops():
    layer = nn.Linear(2, 1)
    opt = optim.SGD(layer.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])
    with torch.profiler.profile() as prof:
        y = layer(x)
        y.sum().backward()
        opt.step()
    op_names = [e["name"] for e in prof.events() if e["kind"] == "op"]
    assert len(op_names) > 0


def test_profiler_no_mindspore_dependency():
    import inspect
    import mindtorch_v2.profiler.profiler as p
    assert "mindspore" not in inspect.getsource(p)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_covers_optimizer_ops tests/mindtorch_v2/test_profiler.py::test_profiler_no_mindspore_dependency -q`
Expected: first fails if coverage missing; second fails if any forbidden import exists.

**Step 3: Write minimal implementation**

Adjust event recording or API exposure to ensure optimizer-step dispatched ops appear and no mindspore import exists.

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_covers_optimizer_ops tests/mindtorch_v2/test_profiler.py::test_profiler_no_mindspore_dependency -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py src/mindtorch_v2/profiler/profiler.py src/mindtorch_v2/profiler/__init__.py
git commit -m "test(mindtorch_v2): cover optimizer phase and no-mindspore guard"
```

### Task 7: NPU conditional validation and final verification

**Files:**
- Modify: `tests/mindtorch_v2/test_profiler.py`
- Modify: `docs/plans/2026-02-27-mindtorch-v2-profiler-design.md` (append implementation notes if needed)

**Step 1: Write the failing/skip-aware NPU test**

```python
import pytest


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_profiler_npu_event_device_type():
    x = torch.ones((2, 2), device="npu")
    with torch.profiler.profile() as prof:
        _ = x + x
    assert any(e["device_type"] == "NPU" for e in prof.events())
```

**Step 2: Run test to verify behavior**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py::test_profiler_npu_event_device_type -q`
Expected: PASS on NPU machine; SKIP otherwise.

**Step 3: Final verification suite**

Run:
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_profiler.py -q`
- `PYTHONPATH=src pytest tests/mindtorch_v2/test_autograd_api.py tests/mindtorch_v2/test_optim.py -q`

Expected: all pass.

**Step 4: Full mindtorch_v2 UT gate (required before PR)**

Run: `PYTHONPATH=src pytest tests/mindtorch_v2 -q`
Expected: no failures.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_profiler.py docs/plans/2026-02-27-mindtorch-v2-profiler-design.md
git commit -m "feat(mindtorch_v2): finalize profiler mvp validation"
```
