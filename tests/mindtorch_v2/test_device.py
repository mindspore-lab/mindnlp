# tests/mindtorch_v2/test_device.py
import mindtorch_v2 as torch


def test_device_from_string():
    d = torch.device("cpu")
    assert d.type == "cpu"
    assert d.index is None


def test_device_with_index():
    d = torch.device("cuda", 0)
    assert d.type == "cuda"
    assert d.index == 0


def test_device_from_string_with_index():
    d = torch.device("cuda:1")
    assert d.type == "cuda"
    assert d.index == 1


def test_device_equality():
    assert torch.device("cpu") == torch.device("cpu")
    assert torch.device("cuda", 0) == torch.device("cuda:0")
    assert torch.device("cpu") != torch.device("cuda")


def test_device_repr():
    assert repr(torch.device("cpu")) == "device(type='cpu')"
    assert repr(torch.device("cuda", 0)) == "device(type='cuda', index=0)"


def test_device_hash():
    """Devices can be used as dict keys."""
    d = {torch.device("cpu"): 1, torch.device("cuda:0"): 2}
    assert d[torch.device("cpu")] == 1
