import pytest
import mindtorch_v2 as torch


def test_backward_requires_grad_for_non_scalar():
    t = torch.ones((2,))
    with pytest.raises(RuntimeError):
        t.backward()


def test_backward_defaults_to_ones_for_scalar():
    t = torch.ones((2,))
    t.requires_grad = True
    y = t.sum()
    y.backward()
    assert t.grad is not None
    assert t.grad.numpy().tolist() == [1.0, 1.0]


def test_retain_graph_allows_double_backward():
    t = torch.ones((2,))
    t.requires_grad = True
    y = t.sum()
    y.backward(retain_graph=True)
    y.backward(retain_graph=True)
    assert t.grad is not None


def test_retain_grad_populates_non_leaf_grad():
    t = torch.ones((2,))
    y = t.sum()
    y.retain_grad()
    y.backward()
    assert y.grad is not None
