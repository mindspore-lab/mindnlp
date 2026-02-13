import mindtorch_v2 as torch


def test_creation_ops():
    x = torch.zeros((2, 3))
    y = torch.ones((2, 3))
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.storage().data.sum() == 0
    assert y.storage().data.sum() == 6


def test_creation_device_index_cpu_meta():
    cpu_tensor = torch.ones((1,), device="cpu:1")
    assert cpu_tensor.device.type == "cpu"
    assert cpu_tensor.device.index == 1

    meta_tensor = torch.ones((1,), device="meta:1")
    assert meta_tensor.device.type == "meta"
    assert meta_tensor.device.index == 1
