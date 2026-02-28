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


def test_tensor_T_for_2d():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = x.T
    assert y.shape == (3, 2)
    assert y.storage() is x.storage()


def test_tensor_T_for_1d_noop():
    x = torch.tensor([1, 2, 3])
    y = x.T
    assert y.shape == (3,)


def test_tensor_flatten_default_and_partial():
    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    flat_all = x.flatten()
    assert flat_all.shape == (8,)
    assert flat_all.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    flat_partial = x.flatten(1)
    assert flat_partial.shape == (2, 4)
    assert flat_partial.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
