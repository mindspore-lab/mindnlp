import mindtorch_v2 as torch


def test_creation_ops():
    x = torch.zeros((2, 3))
    y = torch.ones((2, 3))
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.storage.data.sum() == 0
    assert y.storage.data.sum() == 6
