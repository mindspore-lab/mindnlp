import threading

import mindtorch_v2 as torch


def _stub_runtime(monkeypatch):
    import mindtorch_v2._backends.npu.runtime as npu_runtime
    import mindtorch_v2._backends.npu.state as npu_state

    class FakeRuntime:
        def __init__(self, device_id):
            self.device_id = device_id

        def create_stream(self, priority=0):
            return (self.device_id, int(priority))

        def synchronize_stream(self, stream):
            return None

        def create_event(self, enable_timing, blocking, interprocess):
            return (bool(enable_timing), bool(blocking), bool(interprocess))

        def record_event(self, event, stream):
            return None

        def synchronize_event(self, event):
            return None

        def query_event(self, event):
            return True

        def event_elapsed_time(self, start, end):
            return 0.0

    def fake_get_runtime(device_id=0):
        return FakeRuntime(int(device_id))

    monkeypatch.setattr(npu_runtime, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(npu_state, "get_runtime", fake_get_runtime)
    monkeypatch.setattr(npu_runtime, "_RUNTIMES", {})


def test_npu_device_guard_restores_device(monkeypatch):
    _stub_runtime(monkeypatch)
    torch.npu.set_device(0)
    assert torch.npu.current_device() == 0
    with torch.npu.device(1):
        assert torch.npu.current_device() == 1
    assert torch.npu.current_device() == 0


def test_npu_current_device_thread_local(monkeypatch):
    _stub_runtime(monkeypatch)
    torch.npu.set_device(0)
    assert torch.npu.current_device() == 0
    result = {}

    def worker():
        torch.npu.set_device(1)
        result["dev"] = torch.npu.current_device()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["dev"] == 1
    assert torch.npu.current_device() == 0


def test_npu_default_stream_thread_local(monkeypatch):
    _stub_runtime(monkeypatch)
    s0 = torch.npu.default_stream()
    got = {}

    def worker():
        got["s1"] = torch.npu.default_stream()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert s0 is not got["s1"]


def test_npu_stream_context_switches_device_and_stream(monkeypatch):
    _stub_runtime(monkeypatch)
    s = torch.npu.Stream()
    cur = torch.npu.current_stream()
    with torch.npu.stream(s):
        assert torch.npu.current_stream() is s
        assert torch.npu.current_device() == s.device.index
    assert torch.npu.current_stream() is cur
