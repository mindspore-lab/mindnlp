import mindtorch_v2 as torch


def test_backward_create_graph_allows_second_backward():
    a = torch.ones((2, 2)).requires_grad_()
    out = (a * a).sum()
    out.backward(create_graph=True)
    # Second backward should succeed when create_graph=True (retain_graph implied).
    out.backward()
    assert a.grad is not None
