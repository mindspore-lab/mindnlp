"""Test linear algebra ops via dispatch."""
import numpy as np


def test_matmul_forward():
    """Test matmul forward via dispatch."""
    import mindtorch_v2 as torch
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[1.0], [1.0]])
    result = torch.matmul(a, b)
    expected = [[3.0], [7.0]]
    np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=5)


def test_matmul_backward():
    """Test matmul backward via autograd."""
    import mindtorch_v2 as torch
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    result = torch.matmul(a, b)
    loss = result.sum()
    loss.backward()
    # grad_a = grad @ b.T, grad_b = a.T @ grad
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (2, 2)
    assert b.grad.shape == (2, 2)


def test_bmm_forward():
    """Test bmm forward via dispatch."""
    import mindtorch_v2 as torch
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1,2,2) identity
    b = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1,2,2)
    result = torch.bmm(a, b)
    expected = [[[1.0, 2.0], [3.0, 4.0]]]
    np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=5)


def test_bmm_backward():
    """Test bmm backward via autograd."""
    import mindtorch_v2 as torch
    a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    result = torch.bmm(a, b)
    loss = result.sum()
    loss.backward()
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (1, 2, 2)
    assert b.grad.shape == (1, 2, 2)
