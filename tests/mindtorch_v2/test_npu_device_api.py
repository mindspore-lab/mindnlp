import pytest
import mindtorch_v2 as torch


def test_get_device_name_from_soc(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_soc_name", lambda: "Ascend910B")
    assert torch.npu.get_device_name("npu:0") == "Ascend910B"


def test_get_device_capability_from_soc(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_soc_name", lambda: "Ascend910B")
    assert torch.npu.get_device_capability("npu:0") == (9, 1)


def test_peer_access_unsupported():
    assert torch.npu.can_device_access_peer(0, 1) is False
    with pytest.raises(RuntimeError):
        torch.npu.enable_peer_access(1)


def test_get_device_properties_schema():
    props = torch.npu.get_device_properties(0)
    assert props.name
    assert hasattr(props, "major")
    assert hasattr(props, "minor")


def test_stream_priority_range_fallback():
    assert torch.npu.stream_priority_range() == (0, 0)


def test_pinned_memory():
    import ctypes
    import mindtorch_v2._backends.npu.runtime as npu_runtime

    buffers = []

    def fake_alloc_host(size):
        buf = (ctypes.c_uint8 * int(size))()
        buffers.append(buf)
        return ctypes.addressof(buf)

    def fake_free_host(ptr):
        return None

    npu_runtime.alloc_host = fake_alloc_host
    npu_runtime.free_host = fake_free_host

    t = torch.tensor([1.0, 2.0])
    tp = t.pin_memory()
    assert tp.is_pinned() is True
    assert t.is_pinned() is False
    assert tp.numpy().tolist() == t.numpy().tolist()


def test_pinned_memory_storage_cleanup(monkeypatch):
    import ctypes
    import mindtorch_v2._backends.npu.runtime as npu_runtime

    calls = {"free": 0}
    buffers = []

    def fake_alloc_host(size):
        buf = (ctypes.c_uint8 * int(size))()
        buffers.append(buf)
        return ctypes.addressof(buf)

    def fake_free_host(ptr):
        calls["free"] += 1

    monkeypatch.setattr(npu_runtime, "alloc_host", fake_alloc_host)
    monkeypatch.setattr(npu_runtime, "free_host", fake_free_host)

    t = torch.tensor([1.0, 2.0])
    tp = t.pin_memory()
    del tp
    import gc

    gc.collect()
    assert calls["free"] >= 1


def test_npu_is_initialized_and_init(monkeypatch):
    import mindtorch_v2._backends.npu.runtime as npu_runtime

    torch.npu._reset_init_for_test()

    class DummyRuntime:
        def __init__(self):
            self.inited = False

        def init(self, device_id=0):
            self.inited = True

    runtime = DummyRuntime()

    monkeypatch.setattr(npu_runtime, "get_runtime", lambda device_id=0: runtime)

    assert torch.npu.is_initialized() is False
    torch.npu.init()
    assert runtime.inited is True
    assert torch.npu.is_initialized() is True


def test_npu_is_initialized_after_set_device(monkeypatch):
    import mindtorch_v2._backends.npu.runtime as npu_runtime

    torch.npu._reset_init_for_test()

    class DummyRuntime:
        def __init__(self):
            self.inited = False

        def init(self, device_id=0):
            self.inited = True

    runtime = DummyRuntime()

    monkeypatch.setattr(npu_runtime, "get_runtime", lambda device_id=0: runtime)

    assert torch.npu.is_initialized() is False
    torch.npu.set_device(0)
    assert torch.npu.is_initialized() is True


def test_to_non_blocking_respects_pinned(monkeypatch):
    import mindtorch_v2._backends.npu.runtime as npu_runtime
    import ctypes

    calls = {}

    def fake_copy_cpu_to_npu(arr, runtime=None, non_blocking=False, stream=None):
        calls["non_blocking"] = non_blocking
        return (123, 0)

    monkeypatch.setattr(npu_runtime, "_copy_cpu_to_npu", fake_copy_cpu_to_npu)

    buffers = []

    def fake_alloc_host(size):
        buf = (ctypes.c_uint8 * int(size))()
        buffers.append(buf)
        return ctypes.addressof(buf)

    npu_runtime.alloc_host = fake_alloc_host
    npu_runtime.free_host = lambda ptr: None

    t = torch.tensor([1.0, 2.0]).pin_memory()
    _ = t.to("npu", non_blocking=True)
    assert calls["non_blocking"] is True


def test_copy_cpu_to_npu_uses_async_when_available(monkeypatch):
    import ctypes
    import numpy as np
    import mindtorch_v2._backends.npu.runtime as npu_runtime

    calls = {}

    class FakeRt:
        def memcpy_async(self, dst, dst_size, src, src_size, kind, stream):
            calls["async"] = True
            calls["stream"] = stream
            return 0

        def memcpy(self, dst, dst_size, src, src_size, kind):
            calls["sync"] = True
            return 0

        def malloc_host(self, size):
            return (1234, 0)

        def free_host(self, ptr):
            return 0

    class FakeAcl:
        def __init__(self):
            self.rt = FakeRt()

    class DummyRuntime:
        stream = 999

        def activate(self):
            return None

        def defer_host_free(self, ptr):
            calls["defer"] = ptr

    fake_acl = FakeAcl()
    monkeypatch.setattr(npu_runtime, "acl", fake_acl)
    monkeypatch.setattr(npu_runtime, "ensure_acl", lambda: fake_acl)
    monkeypatch.setattr(npu_runtime, "_alloc_device", lambda size, runtime=None: 5678)
    monkeypatch.setattr(ctypes, "memmove", lambda *args, **kwargs: None)

    arr = np.zeros((2,), dtype=np.float32)
    npu_runtime._copy_cpu_to_npu(arr, runtime=DummyRuntime(), non_blocking=True, stream=111)

    assert calls.get("async") is True
    assert calls.get("stream") == 111


def test_copy_npu_to_cpu_uses_async_when_available(monkeypatch):
    import numpy as np
    import mindtorch_v2._backends.npu.runtime as npu_runtime

    calls = {}

    class FakeRt:
        def memcpy_async(self, dst, dst_size, src, src_size, kind, stream):
            calls["async"] = True
            calls["stream"] = stream
            return 0

        def memcpy(self, dst, dst_size, src, src_size, kind):
            calls["sync"] = True
            return 0

        def malloc_host(self, size):
            return (1234, 0)

        def free_host(self, ptr):
            calls["free"] = ptr
            return 0

    class FakeAcl:
        def __init__(self):
            self.rt = FakeRt()

    class DummyRuntime:
        stream = 999

        def activate(self):
            return None

        def synchronize_stream(self, stream):
            calls["sync_stream"] = stream

    fake_acl = FakeAcl()
    monkeypatch.setattr(npu_runtime, "acl", fake_acl)
    monkeypatch.setattr(npu_runtime, "ensure_acl", lambda: fake_acl)
    monkeypatch.setattr(
        npu_runtime,
        "_numpy_from_ptr",
        lambda ptr, shape, dtype: np.zeros(shape, dtype=np.float32),
    )

    arr = npu_runtime._copy_npu_to_cpu(
        123,
        8,
        (2,),
        "float32",
        runtime=DummyRuntime(),
        non_blocking=True,
        stream=111,
    )

    assert calls.get("async") is True
    assert calls.get("stream") == 111
    assert calls.get("sync_stream") == 111
    assert arr.shape == (2,)
