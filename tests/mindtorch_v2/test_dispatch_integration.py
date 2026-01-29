"""Test dispatch integration with new ops."""
import pytest
import numpy as np


def test_dispatch_uses_new_ops():
    """dispatch() should use new standardized ops."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])

    result = dispatch('add', a, b)

    np.testing.assert_array_equal(result.numpy(), [5.0, 7.0, 9.0])


def test_dispatch_with_autograd():
    """dispatch() should create grad_fn using new ops."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    result = dispatch('add', a, b)

    assert result.requires_grad
    assert result.grad_fn is not None


def test_dispatch_matmul_uses_new_op():
    """dispatch() should route matmul to new op."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0], [1.0]])

    result = dispatch('matmul', a, b)

    expected = [[3.0], [7.0]]
    np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=5)


def test_dispatch_autograd_backward():
    """dispatch() with autograd should compute correct gradients."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    result = dispatch('mul', a, b)

    # Sum and backward
    loss = result.sum()
    loss.backward()

    # grad_a = b, grad_b = a
    np.testing.assert_array_almost_equal(a.grad.numpy(), [4.0, 5.0, 6.0], decimal=5)
    np.testing.assert_array_almost_equal(b.grad.numpy(), [1.0, 2.0, 3.0], decimal=5)


def test_dispatch_new_op_backward_matmul():
    """dispatch() matmul backward computes correct gradients."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)

    result = dispatch('matmul', a, b)

    # Sum and backward
    loss = result.sum()
    loss.backward()

    # grad_a = grad @ b.T = ones @ I = ones
    expected_grad_a = [[1.0, 1.0], [1.0, 1.0]]
    np.testing.assert_array_almost_equal(a.grad.numpy(), expected_grad_a, decimal=5)
