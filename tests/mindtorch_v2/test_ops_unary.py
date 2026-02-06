"""Test unary math ops via dispatch."""
import numpy as np


def test_neg_forward():
    """Test neg forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.neg(torch.tensor([1.0, -2.0, 3.0]))
    np.testing.assert_array_equal(result.numpy(), [-1.0, 2.0, -3.0])


def test_neg_backward():
    """Test neg backward via autograd."""
    import mindtorch_v2 as torch
    x = torch.tensor([5.0, 3.0], requires_grad=True)
    result = torch.neg(x)
    loss = result.sum()
    loss.backward()
    np.testing.assert_array_equal(x.grad.numpy(), [-1.0, -1.0])


def test_exp_forward():
    """Test exp forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.exp(torch.tensor([0.0, 1.0]))
    np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.71828], decimal=4)


def test_exp_backward():
    """Test exp backward via autograd."""
    import mindtorch_v2 as torch
    x = torch.tensor([0.0, 1.0], requires_grad=True)
    result = torch.exp(x)
    loss = result.sum()
    loss.backward()
    # d/dx exp(x) = exp(x)
    np.testing.assert_array_almost_equal(x.grad.numpy(), result.detach().numpy(), decimal=4)


def test_log_forward():
    """Test log forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.log(torch.tensor([1.0, 2.71828]))
    np.testing.assert_array_almost_equal(result.numpy(), [0.0, 1.0], decimal=4)


def test_log_backward():
    """Test log backward via autograd."""
    import mindtorch_v2 as torch
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    result = torch.log(x)
    loss = result.sum()
    loss.backward()
    # d/dx log(x) = 1/x
    np.testing.assert_array_almost_equal(x.grad.numpy(), [1.0, 0.5], decimal=4)


def test_sqrt_forward():
    """Test sqrt forward via dispatch."""
    import mindtorch_v2 as torch
    result = torch.sqrt(torch.tensor([1.0, 4.0, 9.0]))
    np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.0, 3.0], decimal=4)


def test_sqrt_backward():
    """Test sqrt backward via autograd."""
    import mindtorch_v2 as torch
    x = torch.tensor([1.0, 4.0], requires_grad=True)
    result = torch.sqrt(x)
    loss = result.sum()
    loss.backward()
    # d/dx sqrt(x) = 1 / (2 * sqrt(x))
    np.testing.assert_array_almost_equal(x.grad.numpy(), [0.5, 0.25], decimal=4)
