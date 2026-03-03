import mindtorch_v2 as torch


def test_to_device_roundtrip():
    x = torch.tensor([1.0, 2.0])
    y = x.to("cpu")
    assert y.device.type == "cpu"
