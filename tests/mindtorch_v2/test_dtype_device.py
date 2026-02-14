import mindtorch_v2 as torch


def test_default_dtype_and_device():
    x = torch.tensor([1, 2, 3])
    assert x.dtype == torch.float32
    assert x.device.type == "cpu"


def test_default_device_set_get_affects_creation():
    prev = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        assert str(torch.get_default_device()) == "meta"
        x = torch.zeros((2, 2))
        assert x.device.type == "meta"
    finally:
        torch.set_default_device(prev)


def test_device_str_repr_eq_hash():
    d1 = torch.Device("npu:0")
    d2 = torch.Device("npu", 0)
    d3 = torch.Device("npu", 1)

    assert str(d1) == "npu:0"
    assert repr(d1) == "device(type='npu', index=0)"
    assert d1 == d2
    assert d1 != d3
    assert hash(d1) == hash(d2)
