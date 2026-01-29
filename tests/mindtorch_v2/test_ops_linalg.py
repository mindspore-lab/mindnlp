"""Test linear algebra ops."""
import numpy as np


def test_matmul_forward():
    from mindtorch_v2._ops.linalg_ops import MatmulOp
    import mindspore
    op = MatmulOp()
    a = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mindspore.Tensor([[1.0], [1.0]])
    result = op.forward(a, b)
    expected = [[3.0], [7.0]]
    np.testing.assert_array_almost_equal(result.asnumpy(), expected, decimal=5)

def test_matmul_backward():
    from mindtorch_v2._ops.linalg_ops import MatmulOp
    import mindspore
    op = MatmulOp()
    a = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2,2)
    b = mindspore.Tensor([[1.0, 0.0], [0.0, 1.0]])  # (2,2) identity
    grad_output = mindspore.Tensor([[1.0, 0.0], [0.0, 1.0]])
    grad_a, grad_b = op.backward(grad_output, a, b)
    # grad_a = grad @ b.T = grad @ I = grad
    expected_a = [[1.0, 0.0], [0.0, 1.0]]
    np.testing.assert_array_almost_equal(grad_a.asnumpy(), expected_a, decimal=5)
    # grad_b = a.T @ grad
    expected_b = [[1.0, 3.0], [2.0, 4.0]]
    np.testing.assert_array_almost_equal(grad_b.asnumpy(), expected_b, decimal=5)

def test_bmm_forward():
    from mindtorch_v2._ops.linalg_ops import BmmOp
    import mindspore
    import numpy as np_
    op = BmmOp()
    a = mindspore.Tensor(np_.eye(2, dtype=np_.float32).reshape(1, 2, 2))  # (1,2,2)
    b = mindspore.Tensor(np_.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np_.float32))  # (1,2,2)
    result = op.forward(a, b)
    expected = [[[1.0, 2.0], [3.0, 4.0]]]
    np.testing.assert_array_almost_equal(result.asnumpy(), expected, decimal=5)

def test_bmm_backward():
    from mindtorch_v2._ops.linalg_ops import BmmOp
    import mindspore
    import numpy as np_
    op = BmmOp()
    a = mindspore.Tensor(np_.eye(2, dtype=np_.float32).reshape(1, 2, 2))
    b = mindspore.Tensor(np_.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np_.float32))
    grad = mindspore.Tensor(np_.ones((1, 2, 2), dtype=np_.float32))
    grad_a, grad_b = op.backward(grad, a, b)
    assert grad_a.shape == (1, 2, 2)
    assert grad_b.shape == (1, 2, 2)
