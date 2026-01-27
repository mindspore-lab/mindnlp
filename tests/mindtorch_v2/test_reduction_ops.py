# tests/mindtorch_v2/test_reduction_ops.py
import numpy as np
import mindtorch_v2 as torch


def test_sum_all():
    """sum() reduces all elements."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(a)
    assert result.item() == 10.0


def test_sum_dim():
    """sum() can reduce along dimension."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(a, dim=0)
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 6.0])


def test_sum_keepdim():
    """sum() can keep dimensions."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(a, dim=1, keepdim=True)
    assert result.shape == (2, 1)
    np.testing.assert_array_almost_equal(result.numpy(), [[3.0], [7.0]])


def test_mean_all():
    """mean() reduces all elements."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.mean(a)
    assert result.item() == 2.5


def test_mean_dim():
    """mean() can reduce along dimension."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.mean(a, dim=1)
    np.testing.assert_array_almost_equal(result.numpy(), [1.5, 3.5])


def test_max_all():
    """max() returns maximum element."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    result = torch.max(a)
    assert result.item() == 5.0


def test_max_dim():
    """max() can reduce along dimension, returning values and indices."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    values, indices = torch.max(a, dim=1)
    np.testing.assert_array_almost_equal(values.numpy(), [5.0, 3.0])
    np.testing.assert_array_equal(indices.numpy(), [1, 0])


def test_min_all():
    """min() returns minimum element."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    result = torch.min(a)
    assert result.item() == 1.0


def test_min_dim():
    """min() can reduce along dimension."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    values, indices = torch.min(a, dim=1)
    np.testing.assert_array_almost_equal(values.numpy(), [1.0, 2.0])
    np.testing.assert_array_equal(indices.numpy(), [0, 1])


def test_tensor_sum_method():
    """Tensor.sum() method works."""
    a = torch.tensor([1.0, 2.0, 3.0])
    result = a.sum()
    assert result.item() == 6.0


def test_tensor_mean_method():
    """Tensor.mean() method works."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = a.mean()
    assert result.item() == 2.5
