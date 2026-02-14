import pytest
import mindtorch_v2 as torch


def test_get_device_name_stub(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_device_name", lambda device=None: "Ascend", raising=False)
    assert torch.npu.get_device_name("npu:0") == "Ascend"


def test_get_device_capability_stub(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_device_capability", lambda device=None: (0, 0), raising=False)
    assert torch.npu.get_device_capability("npu:0") == (0, 0)


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
    t = torch.tensor([1.0, 2.0])
    tp = torch.npu.pin_memory(t)
    assert torch.npu.is_pinned(tp) is True


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
