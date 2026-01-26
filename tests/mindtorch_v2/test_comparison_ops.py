# tests/mindtorch_v2/test_comparison_ops.py
import numpy as np
import mindtorch_v2 as torch


def test_eq():
    """Element-wise equality."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 5.0, 3.0])
    result = torch.eq(a, b)
    np.testing.assert_array_equal(result.numpy(), [True, False, True])


def test_ne():
    """Element-wise not equal."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 5.0, 3.0])
    result = torch.ne(a, b)
    np.testing.assert_array_equal(result.numpy(), [False, True, False])


def test_gt():
    """Element-wise greater than."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.gt(a, b)
    np.testing.assert_array_equal(result.numpy(), [False, True, False])


def test_lt():
    """Element-wise less than."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.lt(a, b)
    np.testing.assert_array_equal(result.numpy(), [True, False, False])


def test_ge():
    """Element-wise greater than or equal."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.ge(a, b)
    np.testing.assert_array_equal(result.numpy(), [False, True, True])


def test_le():
    """Element-wise less than or equal."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.le(a, b)
    np.testing.assert_array_equal(result.numpy(), [True, False, True])


def test_tensor_eq_operator():
    """Tensor supports == operator."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a == b
    np.testing.assert_array_equal(result.numpy(), [True, False])


def test_tensor_ne_operator():
    """Tensor supports != operator."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a != b
    np.testing.assert_array_equal(result.numpy(), [False, True])


def test_tensor_gt_operator():
    """Tensor supports > operator."""
    a = torch.tensor([3.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a > b
    np.testing.assert_array_equal(result.numpy(), [True, False])


def test_tensor_lt_operator():
    """Tensor supports < operator."""
    a = torch.tensor([3.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a < b
    np.testing.assert_array_equal(result.numpy(), [False, True])


def test_tensor_ge_operator():
    """Tensor supports >= operator."""
    a = torch.tensor([3.0, 2.0, 5.0])
    b = torch.tensor([1.0, 5.0, 5.0])
    result = a >= b
    np.testing.assert_array_equal(result.numpy(), [True, False, True])


def test_tensor_le_operator():
    """Tensor supports <= operator."""
    a = torch.tensor([3.0, 2.0, 5.0])
    b = torch.tensor([1.0, 5.0, 5.0])
    result = a <= b
    np.testing.assert_array_equal(result.numpy(), [False, True, True])
