import threading

import mindtorch_v2 as torch


def _stub_runtime(monkeypatch):
    import mindtorch_v2._backends.npu.runtime as npu_runtime
    import mindtorch_v2._backends.npu.state as npu_state

    def fake_get_runtime(device_id=0):
        return None

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
