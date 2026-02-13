import mindtorch_v2 as torch


def test_device_parsing_and_repr():
    dev = torch.Device("npu:1")
    assert dev.type == "npu"
    assert dev.index == 1
    assert repr(dev) == "npu:1"

    dev = torch.Device("cpu")
    assert dev.type == "cpu"
    assert dev.index is None
    assert repr(dev) == "cpu"

    dev = torch.Device("meta:1")
    assert dev.type == "meta"
    assert dev.index == 1
    assert repr(dev) == "meta:1"

    dev = torch.Device("cpu", 1)
    assert dev.type == "cpu"
    assert dev.index == 1
    assert repr(dev) == "cpu:1"
