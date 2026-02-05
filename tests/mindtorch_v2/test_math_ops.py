# tests/mindtorch_v2/test_math_ops.py
import numpy as np
import mindtorch_v2 as torch


def test_tensor_add_method():
    """Tensor has add method."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = a.add(b)
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 6.0])


def test_tensor_add_operator():
    """Tensor supports + operator."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = a + b
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 6.0])


def test_tensor_sub_operator():
    """Tensor supports - operator."""
    a = torch.tensor([5.0, 6.0])
    b = torch.tensor([1.0, 2.0])
    result = a - b
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 4.0])


def test_tensor_mul_operator():
    """Tensor supports * operator."""
    a = torch.tensor([2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    result = a * b
    np.testing.assert_array_almost_equal(result.numpy(), [8.0, 15.0])


def test_tensor_div_operator():
    """Tensor supports / operator."""
    a = torch.tensor([10.0, 20.0])
    b = torch.tensor([2.0, 4.0])
    result = a / b
    np.testing.assert_array_almost_equal(result.numpy(), [5.0, 5.0])


def test_tensor_neg_operator():
    """Tensor supports unary - operator."""
    a = torch.tensor([1.0, -2.0, 3.0])
    result = -a
    np.testing.assert_array_almost_equal(result.numpy(), [-1.0, 2.0, -3.0])


def test_tensor_pow_operator():
    """Tensor supports ** operator."""
    a = torch.tensor([2.0, 3.0])
    result = a ** 2
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 9.0])


def test_tensor_matmul():
    """Tensor supports @ operator for matmul."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = a @ b
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_exp():
    """torch.exp works."""
    a = torch.tensor([0.0, 1.0, 2.0])
    result = torch.exp(a)
    expected = np.exp([0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected, decimal=5)


def test_log():
    """torch.log works."""
    a = torch.tensor([1.0, np.e, np.e**2])
    result = torch.log(a)
    expected = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_sqrt():
    """torch.sqrt works."""
    a = torch.tensor([1.0, 4.0, 9.0])
    result = torch.sqrt(a)
    np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.0, 3.0])


def test_abs():
    """torch.abs works."""
    a = torch.tensor([-1.0, 2.0, -3.0])
    result = torch.abs(a)
    np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.0, 3.0])
