# ACL Lazy Init + Exit Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Register ACL runtime cleanup at process exit while keeping lazy init and avoiding per-op init/finalize.

**Architecture:** Add atexit registration in `ascend._Runtime.init()` guarded by a module flag. Provide a unit test that stubs `acl` and verifies registration is done once even if `init()` is called twice.

**Tech Stack:** Python, pytest, Ascend ACL Python bindings

---

### Task 1: Add failing test for runtime cleanup registration

**Files:**
- Create: `tests/mindtorch_v2/test_runtime.py`
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Test: `tests/mindtorch_v2/test_runtime.py::test_runtime_init_registers_cleanup_once`

**Step 1: Write the failing test**

```python
import atexit
import types

import mindtorch_v2._backends.ascend as ascend


def test_runtime_init_registers_cleanup_once(monkeypatch):
    calls = []

    class DummyRT:
        def __init__(self, log):
            self.log = log

        def set_device(self, device_id):
            self.log.append(("set_device", device_id))
            return 0

        def create_context(self, device_id):
            self.log.append(("create_context", device_id))
            return "ctx", 0

        def create_stream(self):
            self.log.append(("create_stream",))
            return "stream", 0

        def destroy_stream(self, stream):
            self.log.append(("destroy_stream", stream))
            return 0

        def destroy_context(self, ctx):
            self.log.append(("destroy_context", ctx))
            return 0

        def reset_device(self, device_id):
            self.log.append(("reset_device", device_id))
            return 0

    def init():
        calls.append("init")
        return 0

    def finalize():
        calls.append("finalize")
        return 0

    dummy_acl = types.SimpleNamespace(init=init, finalize=finalize, rt=DummyRT(calls))

    monkeypatch.setattr(ascend, "acl", dummy_acl)
    runtime = ascend._Runtime()
    monkeypatch.setattr(ascend, "_runtime", runtime)
    monkeypatch.setattr(ascend, "_RUNTIME_CLEANUP_REGISTERED", False, raising=False)

    registered = []

    def register(func):
        registered.append(func)

    monkeypatch.setattr(atexit, "register", register)

    runtime.init(0)
    runtime.init(0)

    assert len(registered) == 1
    assert registered[0].__self__ is runtime
    assert calls.count("init") == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_runtime.py::test_runtime_init_registers_cleanup_once -v`

Expected: FAIL because no atexit registration occurs yet.

**Step 3: Write minimal implementation**

Update `src/mindtorch_v2/_backends/ascend.py`:

```python
import atexit

_RUNTIME_CLEANUP_REGISTERED = False


def _register_runtime_cleanup(runtime):
    global _RUNTIME_CLEANUP_REGISTERED
    if _RUNTIME_CLEANUP_REGISTERED:
        return
    atexit.register(runtime.finalize)
    _RUNTIME_CLEANUP_REGISTERED = True
```

Call `_register_runtime_cleanup(self)` at the end of `Runtime.init()` after successful initialization.

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_runtime.py::test_runtime_init_registers_cleanup_once -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/mindtorch_v2/test_runtime.py src/mindtorch_v2/_backends/ascend.py
git commit -m "feat: register acl runtime cleanup on exit"
```
