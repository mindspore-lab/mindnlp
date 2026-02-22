import numpy as np
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


def test_eye_cpu():
    x = torch.eye(3, 2)
    expected = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    assert x.shape == (3, 2)
    assert x.numpy().tolist() == expected


def test_range_cpu():
    x = torch.range(0.0, 2.0, 0.5)
    expected = np.arange(0.0, 2.0 + 0.5, 0.5)
    np.testing.assert_allclose(x.numpy(), expected)


def test_arange_cpu():
    x = torch.arange(0, 5)
    assert x.shape == (5,)
    assert x.numpy().tolist() == [0, 1, 2, 3, 4]


def test_linspace_cpu():
    x = torch.linspace(0.0, 1.0, 5)
    assert x.shape == (5,)
    assert x.numpy().tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_full_cpu():
    x = torch.full((2, 3), 1.5)
    assert x.shape == (2, 3)
    assert x.numpy().tolist() == [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]
