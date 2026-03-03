import mindtorch_v2 as torch


def test_tensor_is_hashable():
    x = torch.tensor([1.0])
    assert isinstance(hash(x), int)
