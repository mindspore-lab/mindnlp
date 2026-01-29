"""Linear algebra operations."""
from typing import Tuple
import mindspore
from .base import Op
from .pyboost import get_pyboost_op


def _reduce_grad_to_shape(grad: mindspore.Tensor, target_shape: Tuple[int, ...]) -> mindspore.Tensor:
    """Reduce gradient to match target shape by summing over broadcasted dims."""
    sum_op = get_pyboost_op('sum')

    # Sum over leading dims if grad has more dims
    while grad.ndim > len(target_shape):
        grad = sum_op(grad, (0,), False)

    # Sum over dims that were broadcasted (size 1 in target)
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and grad.shape[i] != 1:
            grad = sum_op(grad, (i,), True)

    return grad


class MatmulOp(Op):
    """Matrix multiplication: C = A @ B."""

    def forward(self, a, b):
        return get_pyboost_op('matmul')(a, b)

    def backward(self, grad_output, a, b):
        matmul_op = get_pyboost_op('matmul')
        transpose_op = get_pyboost_op('transpose')

        # Transpose last two dims
        def _transpose_last2(t):
            ndim = t.ndim
            if ndim < 2:
                return t
            perm = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
            return transpose_op(t, perm)

        grad_a = matmul_op(grad_output, _transpose_last2(b))
        grad_b = matmul_op(_transpose_last2(a), grad_output)

        # Reduce gradients to match input shapes (handle broadcasting)
        grad_a = _reduce_grad_to_shape(grad_a, a.shape)
        grad_b = _reduce_grad_to_shape(grad_b, b.shape)

        return grad_a, grad_b


class BmmOp(Op):
    """Batched matrix multiplication."""

    def forward(self, a, b):
        return get_pyboost_op('bmm')(a, b)

    def backward(self, grad_output, a, b):
        bmm_op = get_pyboost_op('bmm')
        transpose_op = get_pyboost_op('transpose')

        b_t = transpose_op(b, (0, 2, 1))
        a_t = transpose_op(a, (0, 2, 1))

        grad_a = bmm_op(grad_output, b_t)
        grad_b = bmm_op(a_t, grad_output)
        return grad_a, grad_b
