import mindtorch_v2 as torch


def test_view_reshape_transpose_share_storage():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = x.reshape((3, 2))
    z = x.transpose(0, 1)
    assert y.storage() is x.storage()
    assert z.storage() is x.storage()
    assert y.shape == (3, 2)
    assert z.shape == (3, 2)
    assert x.stride != z.stride
