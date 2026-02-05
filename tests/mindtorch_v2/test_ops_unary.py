"""Test unary math ops."""
import numpy as np


def test_neg_forward():
    from mindtorch_v2._ops.math_ops import NegOp
    import mindspore
    op = NegOp()
    result = op.forward(mindspore.Tensor([1.0, -2.0, 3.0]))
    np.testing.assert_array_equal(result.asnumpy(), [-1.0, 2.0, -3.0])

def test_neg_backward():
    from mindtorch_v2._ops.math_ops import NegOp
    import mindspore
    op = NegOp()
    grad = mindspore.Tensor([1.0, 1.0])
    x = mindspore.Tensor([5.0, 3.0])
    (grad_x,) = op.backward(grad, x)
    np.testing.assert_array_equal(grad_x.asnumpy(), [-1.0, -1.0])

def test_exp_forward():
    from mindtorch_v2._ops.math_ops import ExpOp
    import mindspore
    op = ExpOp()
    result = op.forward(mindspore.Tensor([0.0, 1.0]))
    np.testing.assert_array_almost_equal(result.asnumpy(), [1.0, 2.71828], decimal=4)

def test_exp_backward():
    from mindtorch_v2._ops.math_ops import ExpOp
    import mindspore
    op = ExpOp()
    x = mindspore.Tensor([0.0, 1.0])
    exp_x = op.forward(x)
    grad = mindspore.Tensor([1.0, 1.0])
    # backward receives exp(x) as saved tensor (result)
    (grad_x,) = op.backward(grad, exp_x)
    np.testing.assert_array_almost_equal(grad_x.asnumpy(), exp_x.asnumpy(), decimal=4)

def test_log_forward():
    from mindtorch_v2._ops.math_ops import LogOp
    import mindspore
    op = LogOp()
    result = op.forward(mindspore.Tensor([1.0, 2.71828]))
    np.testing.assert_array_almost_equal(result.asnumpy(), [0.0, 1.0], decimal=4)

def test_log_backward():
    from mindtorch_v2._ops.math_ops import LogOp
    import mindspore
    op = LogOp()
    x = mindspore.Tensor([1.0, 2.0])
    grad = mindspore.Tensor([1.0, 1.0])
    (grad_x,) = op.backward(grad, x)
    np.testing.assert_array_almost_equal(grad_x.asnumpy(), [1.0, 0.5], decimal=4)

def test_sqrt_forward():
    from mindtorch_v2._ops.math_ops import SqrtOp
    import mindspore
    op = SqrtOp()
    result = op.forward(mindspore.Tensor([1.0, 4.0, 9.0]))
    np.testing.assert_array_almost_equal(result.asnumpy(), [1.0, 2.0, 3.0], decimal=4)

def test_sqrt_backward():
    from mindtorch_v2._ops.math_ops import SqrtOp
    import mindspore
    op = SqrtOp()
    x = mindspore.Tensor([1.0, 4.0])
    sqrt_x = op.forward(x)
    grad = mindspore.Tensor([1.0, 1.0])
    (grad_x,) = op.backward(grad, sqrt_x)
    np.testing.assert_array_almost_equal(grad_x.asnumpy(), [0.5, 0.25], decimal=4)
