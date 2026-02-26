import mindtorch_v2 as torch


def test_device_parsing_and_repr():
    dev = torch.Device("npu:1")
    assert dev.type == "npu"
    assert dev.index == 1
    assert repr(dev) == "device(type='npu', index=1)"

    dev = torch.Device("cpu")
    assert dev.type == "cpu"
    assert dev.index is None
    assert repr(dev) == "device(type='cpu')"

    dev = torch.Device("meta:1")
    assert dev.type == "meta"
    assert dev.index == 1
    assert repr(dev) == "device(type='meta', index=1)"

    dev = torch.Device("cpu", 1)
    assert dev.type == "cpu"
    assert dev.index == 1
    assert repr(dev) == "device(type='cpu', index=1)"


def test_device_npu_default_index_zero():
    dev = torch.Device("npu")
    assert dev.index == 0
    assert repr(dev) == "device(type='npu', index=0)"
