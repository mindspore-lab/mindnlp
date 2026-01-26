"""CPU backend implementations using NumPy.

All ops here are registered with DispatchKey.Backend_CPU.
These implementations use NumPy for computation and wrap results in Tensor.
"""

import numpy as np
from .._dispatch import register_op, DispatchKey
from .._tensor import Tensor


def _to_numpy(x):
    """Convert tensor or scalar to numpy array."""
    if isinstance(x, Tensor):
        return x.numpy()
    return np.asarray(x)


def _wrap_result(arr, dtype=None, device="cpu"):
    """Wrap numpy array as Tensor."""
    return Tensor(arr, dtype=dtype, device=device)


# --- Binary math ops ---

@register_op("add", DispatchKey.Backend_CPU)
def add_cpu(a, b):
    """Element-wise addition."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np + b_np
    return _wrap_result(result)


@register_op("sub", DispatchKey.Backend_CPU)
def sub_cpu(a, b):
    """Element-wise subtraction."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np - b_np
    return _wrap_result(result)


@register_op("mul", DispatchKey.Backend_CPU)
def mul_cpu(a, b):
    """Element-wise multiplication."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np * b_np
    return _wrap_result(result)


@register_op("div", DispatchKey.Backend_CPU)
def div_cpu(a, b):
    """Element-wise division."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np / b_np
    return _wrap_result(result)


@register_op("neg", DispatchKey.Backend_CPU)
def neg_cpu(a):
    """Element-wise negation."""
    a_np = _to_numpy(a)
    return _wrap_result(-a_np)


@register_op("abs", DispatchKey.Backend_CPU)
def abs_cpu(a):
    """Element-wise absolute value."""
    a_np = _to_numpy(a)
    return _wrap_result(np.abs(a_np))


@register_op("pow", DispatchKey.Backend_CPU)
def pow_cpu(a, exponent):
    """Element-wise power."""
    a_np = _to_numpy(a)
    exp_np = _to_numpy(exponent)
    return _wrap_result(np.power(a_np, exp_np))


# --- Unary math ops ---

@register_op("exp", DispatchKey.Backend_CPU)
def exp_cpu(a):
    """Element-wise exponential."""
    a_np = _to_numpy(a)
    return _wrap_result(np.exp(a_np))


@register_op("log", DispatchKey.Backend_CPU)
def log_cpu(a):
    """Element-wise natural logarithm."""
    a_np = _to_numpy(a)
    return _wrap_result(np.log(a_np))


@register_op("sqrt", DispatchKey.Backend_CPU)
def sqrt_cpu(a):
    """Element-wise square root."""
    a_np = _to_numpy(a)
    return _wrap_result(np.sqrt(a_np))


@register_op("sin", DispatchKey.Backend_CPU)
def sin_cpu(a):
    """Element-wise sine."""
    a_np = _to_numpy(a)
    return _wrap_result(np.sin(a_np))


@register_op("cos", DispatchKey.Backend_CPU)
def cos_cpu(a):
    """Element-wise cosine."""
    a_np = _to_numpy(a)
    return _wrap_result(np.cos(a_np))


@register_op("tanh", DispatchKey.Backend_CPU)
def tanh_cpu(a):
    """Element-wise hyperbolic tangent."""
    a_np = _to_numpy(a)
    return _wrap_result(np.tanh(a_np))


@register_op("sigmoid", DispatchKey.Backend_CPU)
def sigmoid_cpu(a):
    """Element-wise sigmoid."""
    a_np = _to_numpy(a)
    return _wrap_result(1.0 / (1.0 + np.exp(-a_np)))


@register_op("relu", DispatchKey.Backend_CPU)
def relu_cpu(a):
    """Element-wise ReLU."""
    a_np = _to_numpy(a)
    return _wrap_result(np.maximum(a_np, 0))


@register_op("matmul", DispatchKey.Backend_CPU)
def matmul_cpu(a, b):
    """Matrix multiplication."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(np.matmul(a_np, b_np))


# --- Reduction ops ---

@register_op("sum", DispatchKey.Backend_CPU)
def sum_cpu(a, dim=None, keepdim=False):
    """Sum reduction."""
    a_np = _to_numpy(a)
    result = np.sum(a_np, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("mean", DispatchKey.Backend_CPU)
def mean_cpu(a, dim=None, keepdim=False):
    """Mean reduction."""
    a_np = _to_numpy(a)
    result = np.mean(a_np, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("max", DispatchKey.Backend_CPU)
def max_cpu(a, dim=None, keepdim=False):
    """Max reduction."""
    a_np = _to_numpy(a)
    if dim is None:
        result = np.max(a_np)
        return _wrap_result(np.array(result))
    else:
        values = np.max(a_np, axis=dim, keepdims=keepdim)
        indices = np.argmax(a_np, axis=dim)
        if keepdim:
            indices = np.expand_dims(indices, axis=dim)
        # Return named tuple-like result
        from collections import namedtuple
        MaxResult = namedtuple('MaxResult', ['values', 'indices'])
        return MaxResult(_wrap_result(values), _wrap_result(indices.astype(np.int64)))


@register_op("min", DispatchKey.Backend_CPU)
def min_cpu(a, dim=None, keepdim=False):
    """Min reduction."""
    a_np = _to_numpy(a)
    if dim is None:
        result = np.min(a_np)
        return _wrap_result(np.array(result))
    else:
        values = np.min(a_np, axis=dim, keepdims=keepdim)
        indices = np.argmin(a_np, axis=dim)
        if keepdim:
            indices = np.expand_dims(indices, axis=dim)
        from collections import namedtuple
        MinResult = namedtuple('MinResult', ['values', 'indices'])
        return MinResult(_wrap_result(values), _wrap_result(indices.astype(np.int64)))


@register_op("prod", DispatchKey.Backend_CPU)
def prod_cpu(a, dim=None, keepdim=False):
    """Product reduction."""
    a_np = _to_numpy(a)
    result = np.prod(a_np, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("argmax", DispatchKey.Backend_CPU)
def argmax_cpu(a, dim=None, keepdim=False):
    """Argmax."""
    a_np = _to_numpy(a)
    result = np.argmax(a_np, axis=dim)
    if keepdim and dim is not None:
        result = np.expand_dims(result, axis=dim)
    return _wrap_result(result.astype(np.int64))


@register_op("argmin", DispatchKey.Backend_CPU)
def argmin_cpu(a, dim=None, keepdim=False):
    """Argmin."""
    a_np = _to_numpy(a)
    result = np.argmin(a_np, axis=dim)
    if keepdim and dim is not None:
        result = np.expand_dims(result, axis=dim)
    return _wrap_result(result.astype(np.int64))


# --- Comparison ops ---

@register_op("eq", DispatchKey.Backend_CPU)
def eq_cpu(a, b):
    """Element-wise equality."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np == b_np)


@register_op("ne", DispatchKey.Backend_CPU)
def ne_cpu(a, b):
    """Element-wise not equal."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np != b_np)


@register_op("gt", DispatchKey.Backend_CPU)
def gt_cpu(a, b):
    """Element-wise greater than."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np > b_np)


@register_op("lt", DispatchKey.Backend_CPU)
def lt_cpu(a, b):
    """Element-wise less than."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np < b_np)


@register_op("ge", DispatchKey.Backend_CPU)
def ge_cpu(a, b):
    """Element-wise greater than or equal."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np >= b_np)


@register_op("le", DispatchKey.Backend_CPU)
def le_cpu(a, b):
    """Element-wise less than or equal."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np <= b_np)
