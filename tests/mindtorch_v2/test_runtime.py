import atexit
import types

import mindtorch_v2._backends.npu.runtime as ascend


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

    assert ("sync", "stream") in calls
    assert ("free", 111) in calls
    assert ("free", 222) in calls
