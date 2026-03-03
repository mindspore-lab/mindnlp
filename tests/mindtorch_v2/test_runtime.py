import atexit
import types
import warnings

import mindtorch_v2._backends.npu.runtime as ascend
import mindtorch_v2 as torch


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


def test_runtime_synchronize_drains_deferred(monkeypatch):
    calls = []

    class DummyRT:
        def set_device(self, device_id):
            calls.append(("set_device", device_id))
            return 0

        def set_context(self, ctx):
            calls.append(("set_context", ctx))
            return 0

    class DummyAlloc:
        def synchronize(self):
            calls.append("alloc_sync")

        def free(self, ptr):
            calls.append(("alloc_free", ptr))

    dummy_acl = types.SimpleNamespace(rt=DummyRT())
    runtime = ascend._Runtime()
    runtime.initialized = True
    runtime.stream = "stream"
    runtime.device_id = 0
    runtime.context = "ctx"

    monkeypatch.setattr(ascend, "acl", dummy_acl)

    from mindtorch_v2._backends.npu import allocator as npu_allocator

    dummy_alloc = DummyAlloc()
    monkeypatch.setattr(npu_allocator, "get_allocator", lambda device_id=0: dummy_alloc)

    runtime.defer_free(111)
    runtime.defer_free(222)
    runtime.synchronize()

    assert "alloc_sync" in calls
    assert ("alloc_free", 111) in calls
    assert ("alloc_free", 222) in calls


def test_acl_launch_blocking_forces_sync(monkeypatch):
    calls = []

    class DummyRT:
        def set_device(self, device_id):
            return 0

        def set_context(self, ctx):
            return 0

    class DummyAlloc:
        def synchronize(self):
            calls.append("alloc_sync")

        def free(self, ptr):
            return None

    class DummyAcl:
        def __init__(self):
            self.rt = DummyRT()

    dummy_acl = DummyAcl()
    runtime = ascend._Runtime()
    runtime.initialized = True
    runtime.stream = "stream"
    runtime.device_id = 0
    runtime.context = "ctx"
    runtime._deferred_frees = []

    monkeypatch.setenv("ACL_LAUNCH_BLOCKING", "1")
    monkeypatch.setattr(ascend, "acl", dummy_acl)
    monkeypatch.setattr(ascend, "get_runtime", lambda device_id=0: runtime)

    from mindtorch_v2._backends.npu import allocator as npu_allocator

    dummy_alloc = DummyAlloc()
    monkeypatch.setattr(npu_allocator, "get_allocator", lambda device_id=0: dummy_alloc)

    from mindtorch_v2._backends.npu import aclnn
    # call internal helper to trigger sync in op wrapper
    aclnn._maybe_sync(runtime)

    assert calls == ["alloc_sync"]


def test_npu_synchronize_uses_runtime(monkeypatch):
    calls = []

    class DummyRuntime:
        def synchronize(self):
            calls.append("sync")

    dummy_runtime = DummyRuntime()

    from mindtorch_v2._backends.npu import runtime as npu_runtime
    monkeypatch.setattr(npu_runtime, "get_runtime", lambda device_id=0: dummy_runtime)

    import mindtorch_v2.npu as npu
    npu.synchronize("npu:0")

    assert calls == ["sync"]


def test_npu_mem_get_info(monkeypatch):
    from mindtorch_v2._backends.npu import runtime as npu_runtime

    class DummyRT:
        def set_device(self, device_id):
            return 0

        def set_context(self, ctx):
            return 0

        def get_mem_info(self, attr):
            return 10, 20, 0

    dummy_acl = types.SimpleNamespace(rt=DummyRT())
    monkeypatch.setattr(npu_runtime, "acl", dummy_acl)
    monkeypatch.setattr(
        npu_runtime,
        "get_runtime",
        lambda device_id=0: types.SimpleNamespace(activate=lambda: None),
    )

    import mindtorch_v2.npu as npu
    free, total = npu.mem_get_info("npu:0")
    assert (free, total) == (10, 20)



def test_npu_is_available_verbose_reports_acl_missing(monkeypatch):
    def fake_get_runtime(device_id=0):
        raise ModuleNotFoundError("No module named acl")

    monkeypatch.setattr(ascend, "get_runtime", fake_get_runtime)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = torch.npu.is_available(verbose=True)

    assert ok is False
    assert any("acl" in str(w.message) for w in caught)
