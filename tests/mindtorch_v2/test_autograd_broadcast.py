import mindtorch_v2 as torch


def test_broadcast_backward():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0], [2.0]])
    a.requires_grad = True
    b.requires_grad = True
    c = torch.add(a, b)
    c.sum().backward()
    assert a.grad.shape == (1, 3)
    assert b.grad.shape == (2, 1)
