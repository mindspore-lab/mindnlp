import mindtorch_v2 as torch


def test_get_device_name_stub(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_device_name", lambda device=None: "Ascend", raising=False)
    assert torch.npu.get_device_name("npu:0") == "Ascend"


def test_get_device_capability_stub(monkeypatch):
    monkeypatch.setattr(torch.npu, "_get_device_capability", lambda device=None: (0, 0), raising=False)
    assert torch.npu.get_device_capability("npu:0") == (0, 0)
