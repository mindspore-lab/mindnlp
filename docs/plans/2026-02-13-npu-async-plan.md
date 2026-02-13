# NPU Async Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make NPU ops async by default, add explicit `torch.npu.synchronize`, and defer workspace frees safely.

**Architecture:** Move per-op synchronization into `runtime.synchronize` and queue workspace frees. Ops run async unless `ACL_LAUNCH_BLOCKING=1` forces sync. Explicit synchronize and NPU->CPU copies are the main sync points.

**Tech Stack:** Python, ACL (ctypes), pytest.

---

### Task 1: Runtime deferred frees + synchronize API

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/runtime.py`
- Modify: `src/mindtorch_v2/npu.py`
- Test: `tests/mindtorch_v2/test_runtime.py`

**Step 1: Write the failing test**

Add to `tests/mindtorch_v2/test_runtime.py`:

```python

def test_runtime_synchronize_drains_deferred(monkeypatch):
    calls = []

    class DummyRT:
        def __init__(self):
            self.stream = "stream"

        def synchronize_stream(self, stream):
            calls.append(("sync", stream))
            return 0

        def free(self, ptr):
            calls.append(("free", ptr))
            return 0

    dummy_acl = types.SimpleNamespace(rt=DummyRT())
    runtime = ascend._Runtime()
    runtime.initialized = True
    runtime.stream = "stream"
    runtime.device_id = 0
    runtime.context = "ctx"
    monkeypatch.setattr(ascend, "acl", dummy_acl)

    runtime.defer_free(111)
    runtime.defer_free(222)
    runtime.synchronize()

    assert calls[0] == ("sync", "stream")
    assert ("free", 111) in calls
    assert ("free", 222) in calls
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_runtime.py::test_runtime_synchronize_drains_deferred -v`
Expected: FAIL (defer_free/synchronize missing)

**Step 3: Write minimal implementation**

In `runtime.py`:
- Add `_deferred_frees` list to `_Runtime`.
- Add methods:

```python
    def defer_free(self, ptr):
        if ptr is None:
            return
        self._deferred_frees.append(ptr)

    def synchronize(self):
        if not self.initialized:
            return
        ret = acl.rt.synchronize_stream(self.stream)
        if ret != ACL_ERROR_CODE:
            raise RuntimeError(f"acl.rt.synchronize_stream failed: {ret}")
        frees = self._deferred_frees
        self._deferred_frees = []
        for ptr in frees:
            acl.rt.free(ptr)
```

In `npu.py`:
- Add:

```python
def synchronize(device=None):
    from ._device import device as Device
    from ._backends.npu import runtime as npu_runtime
    if device is None:
        dev = Device("npu")
    else:
        dev = Device(device) if isinstance(device, str) else device
    runtime = npu_runtime.get_runtime(dev.index or 0)
    runtime.synchronize()
```
- Export in `__all__`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_runtime.py::test_runtime_synchronize_drains_deferred -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/runtime.py src/mindtorch_v2/npu.py tests/mindtorch_v2/test_runtime.py
git commit -m "feat: npu synchronize and deferred frees"
```

---

### Task 2: Remove per-op sync + defer frees in ACLNN wrappers

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/aclnn.py`

**Step 1: Write the failing test**

Add to `tests/mindtorch_v2/test_runtime.py`:

```python

def test_acl_launch_blocking_forces_sync(monkeypatch):
    calls = []

    def fake_sync():
        calls.append("sync")

    monkeypatch.setenv("ACL_LAUNCH_BLOCKING", "1")
    monkeypatch.setattr(ascend, "get_runtime", lambda device_id=0: types.SimpleNamespace(synchronize=fake_sync, stream=0))

    # Call a thin wrapper that should invoke sync when env var set.
    # We will exercise via aclnn.add with mocked bindings in Task 2 implementation.
    assert calls == ["sync"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_runtime.py::test_acl_launch_blocking_forces_sync -v`
Expected: FAIL (sync not invoked)

**Step 3: Write minimal implementation**

In `aclnn.py`:
- Remove `acl.rt.synchronize_stream(stream)` from `add/mul/relu/reduce_sum/inplace_one/inplace_zero`.
- After allocating workspace, replace `acl.rt.free(workspace)` with `runtime.defer_free(workspace)`:
  - Change function signatures to accept `runtime` instead of `stream`:
    `def add(..., runtime):` and use `runtime.stream`.
- If `ACL_LAUNCH_BLOCKING=1`, call `runtime.synchronize()` at end of each op.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_runtime.py::test_acl_launch_blocking_forces_sync -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/aclnn.py tests/mindtorch_v2/test_runtime.py
git commit -m "feat: async aclnn ops with deferred frees"
```

---

### Task 3: Route ops/creation through async ACLNN + implicit sync on CPU access

**Files:**
- Modify: `src/mindtorch_v2/_backends/npu/ops.py`
- Modify: `src/mindtorch_v2/_backends/npu/creation.py`
- Modify: `src/mindtorch_v2/_backends/common/convert.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

Add to `tests/mindtorch_v2/test_ops_npu.py`:

```python
def test_npu_to_cpu_synchronizes(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    calls = []

    def fake_sync():
        calls.append("sync")

    from mindtorch_v2._backends.npu import runtime as npu_runtime
    runtime = npu_runtime.get_runtime(0)
    monkeypatch.setattr(runtime, "synchronize", fake_sync)

    t = torch.ones((1,), device="npu")
    _ = t.to("cpu")
    assert "sync" in calls
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_to_cpu_synchronizes -v`
Expected: FAIL (sync not called)

**Step 3: Write minimal implementation**

- Update `ops.py` and `creation.py` to pass `runtime` into `aclnn.*` wrappers.
- Update `convert.py` NPU->CPU and NPU->NPU (CPU staging) to call `runtime.synchronize()` before D2H copy.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_to_cpu_synchronizes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/npu/ops.py src/mindtorch_v2/_backends/npu/creation.py \
        src/mindtorch_v2/_backends/common/convert.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat: async npu ops with implicit sync on cpu access"
```

---

Plan complete and saved to `docs/plans/2026-02-13-npu-async-plan.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) — Open new session with executing-plans, batch execution with checkpoints

Which approach?
