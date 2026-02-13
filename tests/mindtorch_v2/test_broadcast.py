import mindtorch_v2 as torch


def test_broadcast_add_mul():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0], [2.0]])
    c = torch.add(a, b)
    d = torch.mul(a, b)
    assert c.shape == (2, 3)
    assert d.shape == (2, 3)
