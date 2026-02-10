import mindtorch_v2 as torch


def test_default_dtype_and_device():
    x = torch.tensor([1, 2, 3])
    assert x.dtype == torch.float32
    assert x.device.type == "cpu"
