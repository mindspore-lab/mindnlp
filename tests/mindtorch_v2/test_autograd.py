# tests/mindtorch_v2/test_autograd.py
"""Tests for autograd grad mode context managers and Node base class."""

import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2._autograd import Node


def test_grad_enabled_default():
    """Grad is enabled by default."""
    assert torch.is_grad_enabled() == True


def test_no_grad_context():
    """no_grad disables gradient computation."""
    assert torch.is_grad_enabled() == True
    with torch.no_grad():
        assert torch.is_grad_enabled() == False
    assert torch.is_grad_enabled() == True


def test_enable_grad_context():
    """enable_grad re-enables gradient computation."""
    with torch.no_grad():
        assert torch.is_grad_enabled() == False
        with torch.enable_grad():
            assert torch.is_grad_enabled() == True
        assert torch.is_grad_enabled() == False


def test_set_grad_enabled():
    """set_grad_enabled can toggle gradient computation."""
    assert torch.is_grad_enabled() == True
    with torch.set_grad_enabled(False):
        assert torch.is_grad_enabled() == False
    assert torch.is_grad_enabled() == True


def test_no_grad_decorator():
    """no_grad works as a decorator."""
    @torch.no_grad()
    def my_func():
        return torch.is_grad_enabled()

    assert torch.is_grad_enabled() == True
    assert my_func() == False
    assert torch.is_grad_enabled() == True


# ============================================
# Node Base Class Tests
# ============================================

def test_node_base_class():
    """Node is the base class for autograd functions."""
    node = Node()
    assert hasattr(node, 'next_functions')
    assert hasattr(node, 'backward')
    assert node.next_functions == ()


def test_node_next_functions():
    """Node can store next_functions."""
    node1 = Node()
    node2 = Node()
    node2._next_functions = ((node1, 0),)
    assert node2.next_functions == ((node1, 0),)


def test_node_backward_not_implemented():
    """Node.backward raises NotImplementedError by default."""
    node = Node()
    try:
        node.backward((None,))
        assert False, "Should have raised"
    except NotImplementedError:
        pass


# ============================================
# Backward Engine Tests
# ============================================

def test_backward_simple():
    """Simple backward pass computes gradient."""
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x * 2  # y = 2x, dy/dx = 2
    loss = y.sum()  # scalar output
    loss.backward()

    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])


def test_backward_chain():
    """Backward through chain of operations."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y + 1
    loss = z.sum()
    loss.backward()

    # d(loss)/dx = d(sum(2x+1))/dx = 2
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0, 2.0])


def test_backward_multiple_uses():
    """Backward when tensor is used multiple times."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x + x  # y = 2x
    loss = y.sum()
    loss.backward()

    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])


def test_backward_no_grad_tensor():
    """Tensors without requires_grad don't get gradients."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([3.0, 4.0], requires_grad=False)
    z = x + y
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert y.grad is None


def test_backward_retain_graph():
    """Can backward multiple times with retain_graph=True."""
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 2
    loss = y.sum()

    loss.backward(retain_graph=True)
    first_grad = x.grad.numpy().copy()

    loss.backward(retain_graph=True)
    # Gradients accumulate
    np.testing.assert_array_almost_equal(x.grad.numpy(), first_grad * 2)


def test_tensor_backward_method():
    """Tensor.backward() works."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()  # y = x1^2 + x2^2, dy/dx = 2x
    y.backward()

    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 4.0])


# ============================================
# Tensor Hooks Tests
# ============================================

def test_register_hook():
    """Can register gradient hooks on tensors."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    hook_called = [False]
    hook_grad = [None]

    def hook(grad):
        hook_called[0] = True
        hook_grad[0] = grad.numpy().copy()
        return grad

    x.register_hook(hook)

    y = (x * 2).sum()
    y.backward()

    assert hook_called[0] == True
    np.testing.assert_array_almost_equal(hook_grad[0], [2.0, 2.0])


def test_register_hook_modify_grad():
    """Hook can modify gradients."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def double_grad(grad):
        return grad * 2

    x.register_hook(double_grad)

    y = x.sum()
    y.backward()

    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])  # Doubled


def test_remove_hook():
    """Can remove hooks."""
    x = torch.tensor([1.0], requires_grad=True)

    call_count = [0]
    def hook(grad):
        call_count[0] += 1
        return grad

    handle = x.register_hook(hook)

    y = x.sum()
    y.backward(retain_graph=True)
    assert call_count[0] == 1

    handle.remove()

    y.backward()
    assert call_count[0] == 1  # Hook not called again


# ============================================
# Zero Grad and Detach Tests
# ============================================

def test_zero_grad():
    """Tensor.zero_grad_() clears gradients."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x.sum()
    y.backward()

    assert x.grad is not None
    x.zero_grad_()
    assert x.grad is None


def test_detach():
    """Tensor.detach() returns tensor without grad_fn."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    assert y.grad_fn is not None
    z = y.detach()

    assert z.grad_fn is None
    assert z.requires_grad == False

    np.testing.assert_array_almost_equal(y.numpy(), z.numpy())


def test_detach_():
    """Tensor.detach_() detaches in-place."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    assert y.grad_fn is not None
    y.detach_()

    assert y.grad_fn is None
    assert y.requires_grad == False


def test_requires_grad_():
    """Tensor.requires_grad_() sets requires_grad in-place."""
    x = torch.tensor([1.0, 2.0])
    assert x.requires_grad == False

    x.requires_grad_(True)
    assert x.requires_grad == True

    y = x * 2
    y.sum().backward()

    assert x.grad is not None


# ============================================
# Comprehensive Autograd Tests
# ============================================

def test_no_grad_prevents_recording():
    """Operations in no_grad context don't record grad_fn."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    with torch.no_grad():
        y = x * 2

    assert y.grad_fn is None
    assert y.requires_grad == False


def test_leaf_tensor_is_leaf():
    """Leaf tensors (created directly) are leaves."""
    x = torch.tensor([1.0], requires_grad=True)
    assert x.grad_fn is None  # Leaf

    y = x * 2
    assert y.grad_fn is not None  # Not leaf


def test_backward_non_scalar_requires_grad_tensors():
    """backward() on non-scalar requires grad_tensors argument."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    try:
        y.backward()
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "scalar" in str(e)

    # Should work with grad_tensors
    y.backward(gradient=torch.tensor([1.0, 1.0]))
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])


def test_gradient_accumulation():
    """Gradients accumulate across backward calls."""
    x = torch.tensor([1.0], requires_grad=True)

    for _ in range(3):
        y = x.sum()
        y.backward(retain_graph=True)

    np.testing.assert_array_almost_equal(x.grad.numpy(), [3.0])


def test_exp_backward():
    """Backward through exp."""
    x = torch.tensor([0.0, 1.0], requires_grad=True)
    y = torch.exp(x).sum()
    y.backward()

    expected = np.exp([0.0, 1.0])  # d(exp(x))/dx = exp(x)
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)


def test_log_backward():
    """Backward through log."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.log(x).sum()
    y.backward()

    expected = 1.0 / np.array([1.0, 2.0])  # d(log(x))/dx = 1/x
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


def test_sqrt_backward():
    """Backward through sqrt."""
    x = torch.tensor([1.0, 4.0], requires_grad=True)
    y = torch.sqrt(x).sum()
    y.backward()

    expected = 0.5 / np.sqrt([1.0, 4.0])  # d(sqrt(x))/dx = 1/(2*sqrt(x))
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


def test_matmul_backward():
    """Backward through matmul."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    c = torch.matmul(a, b)
    loss = c.sum()
    loss.backward()

    # Gradients should be ones @ b.T and a.T @ ones
    np.testing.assert_array_almost_equal(
        a.grad.numpy(),
        np.ones((2, 2)) @ np.array([[5.0, 7.0], [6.0, 8.0]])
    )


def test_complex_computation():
    """Backward through complex computation graph."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # y = x^2
    y = x * x
    # z = sum(y) / 3 = mean(x^2)
    z = y.sum() / 3.0
    z.backward()

    # d(mean(x^2))/dx = 2x/3
    expected = 2 * np.array([1.0, 2.0, 3.0]) / 3.0
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)
