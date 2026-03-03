import pytest

import mindtorch_v2 as torch


def test_backward_twice_without_retain_graph_errors():
    a = torch.ones((2, 2)).requires_grad_()
    out = (a * a).sum()
    out.backward()
    with pytest.raises(
        RuntimeError,
        match=(
            r"Trying to backward through the graph a second time .* retain_graph=True"
        ),
    ):
        out.backward()


def test_backward_twice_with_retain_graph_succeeds():
    a = torch.ones((2, 2)).requires_grad_()
    out = (a * a).sum()
    out.backward(retain_graph=True)
    out.backward(retain_graph=True)
    assert a.grad is not None
