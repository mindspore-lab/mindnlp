"""Math operations using pyboost primitives."""
from typing import Tuple
import mindspore
from .base import Op
from .pyboost import get_pyboost_op


def _reduce_grad_to_shape(grad: mindspore.Tensor, target_shape: Tuple[int, ...]) -> mindspore.Tensor:
    """Reduce gradient to match target shape by summing over broadcasted dims."""
    sum_op = get_pyboost_op('sum')

    while grad.ndim > len(target_shape):
        grad = sum_op(grad, (0,), False)

    for i in range(len(target_shape)):
        if target_shape[i] == 1 and grad.shape[i] != 1:
            grad = sum_op(grad, (i,), True)

    return grad


class AddOp(Op):
    """Element-wise addition: c = a + b."""

    def forward(self, a, b):
        return get_pyboost_op('add')(a, b)

    def backward(self, grad_output, a, b):
        # Handle scalar inputs (Python float/int) - they don't have .shape
        a_shape = a.shape if isinstance(a, mindspore.Tensor) else ()
        b_shape = b.shape if isinstance(b, mindspore.Tensor) else ()
        grad_a = _reduce_grad_to_shape(grad_output, a_shape)
        grad_b = _reduce_grad_to_shape(grad_output, b_shape)
        return grad_a, grad_b


class SubOp(Op):
    """Element-wise subtraction: c = a - b."""

    def forward(self, a, b):
        return get_pyboost_op('sub')(a, b)

    def backward(self, grad_output, a, b):
        neg_op = get_pyboost_op('neg')
        # Handle scalar inputs (Python float/int) - they don't have .shape
        a_shape = a.shape if isinstance(a, mindspore.Tensor) else ()
        b_shape = b.shape if isinstance(b, mindspore.Tensor) else ()
        grad_a = _reduce_grad_to_shape(grad_output, a_shape)
        grad_b = _reduce_grad_to_shape(neg_op(grad_output), b_shape)
        return grad_a, grad_b


class MulOp(Op):
    """Element-wise multiplication: c = a * b."""

    def forward(self, a, b):
        return get_pyboost_op('mul')(a, b)

    def backward(self, grad_output, a, b):
        mul_op = get_pyboost_op('mul')
        # Handle scalar inputs (Python float/int) - they don't have .shape
        a_shape = a.shape if isinstance(a, mindspore.Tensor) else ()
        b_shape = b.shape if isinstance(b, mindspore.Tensor) else ()
        grad_a = _reduce_grad_to_shape(mul_op(grad_output, b), a_shape)
        grad_b = _reduce_grad_to_shape(mul_op(grad_output, a), b_shape)
        return grad_a, grad_b


class DivOp(Op):
    """Element-wise division: c = a / b."""

    def forward(self, a, b):
        return get_pyboost_op('div')(a, b)

    def backward(self, grad_output, a, b):
        div_op = get_pyboost_op('div')
        mul_op = get_pyboost_op('mul')
        neg_op = get_pyboost_op('neg')
        pow_op = get_pyboost_op('pow')

        # Handle scalar inputs (Python float/int) - they don't have .shape
        a_shape = a.shape if isinstance(a, mindspore.Tensor) else ()
        b_shape = b.shape if isinstance(b, mindspore.Tensor) else ()

        # Handle scalar b (Python float/int) - convert to tensor for computation
        if not isinstance(b, mindspore.Tensor):
            b = mindspore.Tensor(b, dtype=grad_output.dtype)
        # Handle scalar a (Python float/int) - convert to tensor for computation
        if not isinstance(a, mindspore.Tensor):
            a = mindspore.Tensor(a, dtype=grad_output.dtype)

        grad_a = _reduce_grad_to_shape(div_op(grad_output, b), a_shape)
        b_sq = pow_op(b, mindspore.Tensor(2.0, b.dtype))
        grad_b = _reduce_grad_to_shape(neg_op(div_op(mul_op(grad_output, a), b_sq)), b_shape)
        return grad_a, grad_b


class NegOp(Op):
    """Element-wise negation: y = -x."""

    def forward(self, x):
        return get_pyboost_op('neg')(x)

    def backward(self, grad_output, x):
        return (get_pyboost_op('neg')(grad_output),)


class ExpOp(Op):
    """Element-wise exponential: y = exp(x)."""

    needs_forward_result = True

    def forward(self, x):
        return get_pyboost_op('exp')(x)

    def backward(self, grad_output, exp_x):
        """exp_x = exp(x) from forward result."""
        return (get_pyboost_op('mul')(grad_output, exp_x),)


class LogOp(Op):
    """Element-wise natural logarithm: y = log(x)."""

    def forward(self, x):
        return get_pyboost_op('log')(x)

    def backward(self, grad_output, x):
        return (get_pyboost_op('div')(grad_output, x),)


class SqrtOp(Op):
    """Element-wise square root: y = sqrt(x)."""

    needs_forward_result = True

    def forward(self, x):
        return get_pyboost_op('sqrt')(x)

    def backward(self, grad_output, sqrt_x):
        """sqrt_x = sqrt(x) from forward result."""
        mul_op = get_pyboost_op('mul')
        div_op = get_pyboost_op('div')
        two = mindspore.Tensor(2.0, sqrt_x.dtype)
        return (div_op(grad_output, mul_op(two, sqrt_x)),)


class RsqrtOp(Op):
    """Element-wise reciprocal square root: y = 1/sqrt(x)."""

    def forward(self, x):
        return get_pyboost_op('rsqrt')(x)

    def backward(self, grad_output, x):
        mul_op = get_pyboost_op('mul')
        pow_op = get_pyboost_op('pow')
        neg_op = get_pyboost_op('neg')
        half = mindspore.Tensor(0.5, x.dtype)
        neg_three_half = mindspore.Tensor(-1.5, x.dtype)
        return (mul_op(neg_op(half), mul_op(grad_output, pow_op(x, neg_three_half))),)
