import mindtorch_v2 as torch


def test_reshape_autograd():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad = True
    y = x.reshape((4,))
    z = y.sum()
    z.backward()
    assert x.grad.shape == x.shape
    assert x.grad.numpy().tolist() == [[1.0, 1.0], [1.0, 1.0]]
