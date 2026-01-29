"""Test math ops."""
import pytest
import numpy as np


def test_add_op_forward():
    from mindtorch_v2._ops.math_ops import AddOp
    import mindspore
    op = AddOp()
    a = mindspore.Tensor([1.0, 2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0, 6.0])
    result = op.forward(a, b)
    np.testing.assert_array_equal(result.asnumpy(), [5.0, 7.0, 9.0])


def test_add_op_backward():
    from mindtorch_v2._ops.math_ops import AddOp
    import mindspore
    op = AddOp()
    a = mindspore.Tensor([1.0, 2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0, 6.0])
    grad_output = mindspore.Tensor([1.0, 1.0, 1.0])
    grad_a, grad_b = op.backward(grad_output, a, b)
    np.testing.assert_array_equal(grad_a.asnumpy(), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(grad_b.asnumpy(), [1.0, 1.0, 1.0])


def test_add_op_backward_broadcast():
    from mindtorch_v2._ops.math_ops import AddOp
    import mindspore
    op = AddOp()
    a = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2,2)
    b = mindspore.Tensor([1.0, 1.0])  # (2,)
    grad_output = mindspore.Tensor([[1.0, 1.0], [1.0, 1.0]])
    grad_a, grad_b = op.backward(grad_output, a, b)
    assert grad_a.shape == (2, 2)
    assert grad_b.shape == (2,)
    np.testing.assert_array_equal(grad_b.asnumpy(), [2.0, 2.0])


def test_sub_op_forward():
    from mindtorch_v2._ops.math_ops import SubOp
    import mindspore
    op = SubOp()
    result = op.forward(mindspore.Tensor([5.0, 3.0]), mindspore.Tensor([1.0, 2.0]))
    np.testing.assert_array_equal(result.asnumpy(), [4.0, 1.0])


def test_sub_op_backward():
    from mindtorch_v2._ops.math_ops import SubOp
    import mindspore
    op = SubOp()
    grad = mindspore.Tensor([1.0, 1.0])
    a = mindspore.Tensor([5.0, 3.0])
    b = mindspore.Tensor([1.0, 2.0])
    grad_a, grad_b = op.backward(grad, a, b)
    np.testing.assert_array_equal(grad_a.asnumpy(), [1.0, 1.0])
    np.testing.assert_array_equal(grad_b.asnumpy(), [-1.0, -1.0])


def test_mul_op_forward():
    from mindtorch_v2._ops.math_ops import MulOp
    import mindspore
    op = MulOp()
    result = op.forward(mindspore.Tensor([2.0, 3.0]), mindspore.Tensor([4.0, 5.0]))
    np.testing.assert_array_equal(result.asnumpy(), [8.0, 15.0])


def test_mul_op_backward():
    from mindtorch_v2._ops.math_ops import MulOp
    import mindspore
    op = MulOp()
    grad = mindspore.Tensor([1.0, 1.0])
    a = mindspore.Tensor([2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0])
    grad_a, grad_b = op.backward(grad, a, b)
    # grad_a = grad * b, grad_b = grad * a
    np.testing.assert_array_equal(grad_a.asnumpy(), [4.0, 5.0])
    np.testing.assert_array_equal(grad_b.asnumpy(), [2.0, 3.0])


def test_div_op_forward():
    from mindtorch_v2._ops.math_ops import DivOp
    import mindspore
    op = DivOp()
    result = op.forward(mindspore.Tensor([6.0, 10.0]), mindspore.Tensor([2.0, 5.0]))
    np.testing.assert_array_equal(result.asnumpy(), [3.0, 2.0])


def test_div_op_backward():
    from mindtorch_v2._ops.math_ops import DivOp
    import mindspore
    op = DivOp()
    grad = mindspore.Tensor([1.0, 1.0])
    a = mindspore.Tensor([6.0, 10.0])
    b = mindspore.Tensor([2.0, 5.0])
    grad_a, grad_b = op.backward(grad, a, b)
    # grad_a = grad / b = [0.5, 0.2]
    np.testing.assert_array_almost_equal(grad_a.asnumpy(), [0.5, 0.2], decimal=5)
    # grad_b = -grad * a / b^2 = [-1.5, -0.4]
    np.testing.assert_array_almost_equal(grad_b.asnumpy(), [-1.5, -0.4], decimal=5)
