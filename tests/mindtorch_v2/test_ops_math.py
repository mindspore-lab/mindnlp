"""Test math ops via dispatch."""
import numpy as np


def test_add_op_forward():
    """Test add forward via dispatch."""
    import mindtorch_v2 as torch
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = torch.add(a, b)
    np.testing.assert_array_equal(result.numpy(), [5.0, 7.0, 9.0])


def test_add_op_backward():
    """Test add backward via autograd."""
    import mindtorch_v2 as torch
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    result = torch.add(a, b)
    loss = result.sum()
    loss.backward()
    np.testing.assert_array_equal(a.grad.numpy(), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(b.grad.numpy(), [1.0, 1.0, 1.0])


def test_add_op_backward_broadcast():
    """Test add backward with broadcasting."""
    import mindtorch_v2 as torch
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    b = torch.tensor([1.0, 1.0], requires_grad=True)  # (2,)
    result = torch.add(a, b)
    loss = result.sum()
    loss.backward()
    assert a.grad.shape == (2, 2)
    assert b.grad.shape == (2,)
    np.testing.assert_array_equal(b.grad.numpy(), [2.0, 2.0])


def test_sub_op_forward():
    """Test sub forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.sub(torch.tensor([5.0, 3.0]), torch.tensor([1.0, 2.0]))
    np.testing.assert_array_equal(result.numpy(), [4.0, 1.0])


def test_sub_op_backward():
    """Test sub backward via autograd."""
    import mindtorch_v2 as torch
    a = torch.tensor([5.0, 3.0], requires_grad=True)
    b = torch.tensor([1.0, 2.0], requires_grad=True)
    result = torch.sub(a, b)
    loss = result.sum()
    loss.backward()
    np.testing.assert_array_equal(a.grad.numpy(), [1.0, 1.0])
    np.testing.assert_array_equal(b.grad.numpy(), [-1.0, -1.0])


def test_mul_op_forward():
    """Test mul forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.mul(torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
    np.testing.assert_array_equal(result.numpy(), [8.0, 15.0])


def test_mul_op_backward():
    """Test mul backward via autograd."""
    import mindtorch_v2 as torch
    a = torch.tensor([2.0, 3.0], requires_grad=True)
    b = torch.tensor([4.0, 5.0], requires_grad=True)
    result = torch.mul(a, b)
    loss = result.sum()
    loss.backward()
    # grad_a = grad * b, grad_b = grad * a
    np.testing.assert_array_equal(a.grad.numpy(), [4.0, 5.0])
    np.testing.assert_array_equal(b.grad.numpy(), [2.0, 3.0])


def test_div_op_forward():
    """Test div forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.div(torch.tensor([6.0, 10.0]), torch.tensor([2.0, 5.0]))
    np.testing.assert_array_equal(result.numpy(), [3.0, 2.0])


def test_div_op_backward():
    """Test div backward via autograd."""
    import mindtorch_v2 as torch
    a = torch.tensor([6.0, 10.0], requires_grad=True)
    b = torch.tensor([2.0, 5.0], requires_grad=True)
    result = torch.div(a, b)
    loss = result.sum()
    loss.backward()
    # grad_a = grad / b = [0.5, 0.2]
    np.testing.assert_array_almost_equal(a.grad.numpy(), [0.5, 0.2], decimal=5)
    # grad_b = -grad * a / b^2 = [-1.5, -0.4]
    np.testing.assert_array_almost_equal(b.grad.numpy(), [-1.5, -0.4], decimal=5)
