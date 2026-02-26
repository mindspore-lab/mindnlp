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


def test_detach_breaks_grad_chain():
    t = torch.ones((2,))
    t.requires_grad_(True)
    y = t.detach()
    assert y.requires_grad is False


def test_detach_inplace():
    t = torch.ones((2,))
    t.requires_grad_(True)
    t.detach_()
    assert t.requires_grad is False


def test_register_hook_receives_grad():
    t = torch.ones((2,))
    t.requires_grad_(True)
    seen = {}
    def hook(grad):
        seen["grad"] = grad.numpy().tolist()
        return grad
    t.register_hook(hook)
    t.sum().backward()
    assert seen["grad"] == [1.0, 1.0]


def test_autograd_grad_basic():
    x = torch.ones((2,))
    x.requires_grad_(True)
    y = (x * x).sum()
    (gx,) = torch.autograd.grad(y, (x,))
    assert gx.numpy().tolist() == [2.0, 2.0]


def test_autograd_grad_allow_unused():
    x = torch.ones((2,))
    x.requires_grad_(True)
    y = torch.ones((1,))
    with pytest.raises(RuntimeError):
        torch.autograd.grad(y, (x,))
    gx = torch.autograd.grad(y, (x,), allow_unused=True)[0]
    assert gx is None
