# tests/mindtorch_v2/test_backends.py
import numpy as np
import mindtorch_v2 as torch


def test_backend_add():
    """CPU backend can add tensors."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = torch.add(a, b)
    expected = np.array([5.0, 7.0, 9.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_add_scalar():
    """CPU backend can add scalar to tensor."""
    a = torch.tensor([1.0, 2.0, 3.0])
    result = torch.add(a, 10.0)
    expected = np.array([11.0, 12.0, 13.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_sub():
    """CPU backend can subtract tensors."""
    a = torch.tensor([5.0, 6.0, 7.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    result = torch.sub(a, b)
    expected = np.array([4.0, 4.0, 4.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_mul():
    """CPU backend can multiply tensors."""
    a = torch.tensor([2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0])
    result = torch.mul(a, b)
    expected = np.array([10.0, 18.0, 28.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_div():
    """CPU backend can divide tensors."""
    a = torch.tensor([10.0, 20.0, 30.0])
    b = torch.tensor([2.0, 4.0, 5.0])
    result = torch.div(a, b)
    expected = np.array([5.0, 5.0, 6.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)
