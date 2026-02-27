import mindtorch_v2 as torch


def test_backward_create_graph_allows_second_backward():
    a = torch.ones((2, 2)).requires_grad_()
    out = (a * a).sum()
    out.backward(create_graph=True)
    # Second backward should succeed when create_graph=True (retain_graph implied).
    out.backward()
    assert a.grad is not None


def test_autograd_grad_create_graph_builds_higher_order_graph():
    x = torch.tensor([2.0], requires_grad=True)
    y = x * x

    (gx,) = torch.autograd.grad(y, (x,), create_graph=True)
    assert gx.requires_grad is True
    assert gx.grad_fn is not None

    z = gx * gx
    z.backward()
    assert x.grad is not None


def test_backward_create_graph_keeps_grad_differentiable():
    x = torch.tensor([3.0], requires_grad=True)
    y = x * x
    y.backward(create_graph=True)

    assert x.grad is not None
    assert x.grad.requires_grad is True

    q = x.grad * x
    q.backward()
    assert x.grad is not None
