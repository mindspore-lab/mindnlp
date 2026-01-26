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


@register_op("transpose", DispatchKey.Backend_CPU)
def transpose_cpu(a, dim0, dim1):
    """Transpose two dimensions."""
    a_np = _to_numpy(a)
    return _wrap_result(np.swapaxes(a_np, dim0, dim1))


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


# --- Activation ops ---

@register_op("gelu", DispatchKey.Backend_CPU)
def gelu_cpu(a, approximate='none'):
    """GELU activation."""
    import math
    a_np = _to_numpy(a)
    if approximate == 'tanh':
        # Approximate GELU
        coef = math.sqrt(2.0 / math.pi)
        result = 0.5 * a_np * (1.0 + np.tanh(coef * (a_np + 0.044715 * a_np ** 3)))
    else:
        # Exact GELU using erf
        from scipy.special import erf
        result = 0.5 * a_np * (1.0 + erf(a_np / math.sqrt(2.0)))
    return _wrap_result(result)


@register_op("silu", DispatchKey.Backend_CPU)
def silu_cpu(a):
    """SiLU/Swish activation: x * sigmoid(x)."""
    a_np = _to_numpy(a)
    result = a_np * (1.0 / (1.0 + np.exp(-a_np)))
    return _wrap_result(result)


@register_op("softmax", DispatchKey.Backend_CPU)
def softmax_cpu(a, dim=None):
    """Softmax activation."""
    a_np = _to_numpy(a)
    if dim is None:
        dim = -1
    # Subtract max for numerical stability
    a_max = np.max(a_np, axis=dim, keepdims=True)
    exp_a = np.exp(a_np - a_max)
    result = exp_a / np.sum(exp_a, axis=dim, keepdims=True)
    return _wrap_result(result)


@register_op("log_softmax", DispatchKey.Backend_CPU)
def log_softmax_cpu(a, dim=None):
    """Log softmax."""
    a_np = _to_numpy(a)
    if dim is None:
        dim = -1
    a_max = np.max(a_np, axis=dim, keepdims=True)
    log_sum_exp = a_max + np.log(np.sum(np.exp(a_np - a_max), axis=dim, keepdims=True))
    result = a_np - log_sum_exp
    return _wrap_result(result)


# --- Neural network ops ---

@register_op("embedding", DispatchKey.Backend_CPU)
def embedding_cpu(indices, weight):
    """Embedding lookup."""
    indices_np = _to_numpy(indices).astype(np.int64)
    weight_np = _to_numpy(weight)
    result = weight_np[indices_np]
    return _wrap_result(result)


@register_op("dropout", DispatchKey.Backend_CPU)
def dropout_cpu(a, p=0.5, training=True):
    """Dropout."""
    if not training or p == 0:
        return a
    a_np = _to_numpy(a)
    mask = np.random.binomial(1, 1 - p, a_np.shape) / (1 - p)
    result = a_np * mask
    return _wrap_result(result.astype(a_np.dtype))


@register_op("layer_norm", DispatchKey.Backend_CPU)
def layer_norm_cpu(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Layer normalization."""
    a_np = _to_numpy(a)

    # Determine axes to normalize over
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))

    mean = np.mean(a_np, axis=axes, keepdims=True)
    var = np.var(a_np, axis=axes, keepdims=True)
    result = (a_np - mean) / np.sqrt(var + eps)

    if weight is not None:
        result = result * _to_numpy(weight)
    if bias is not None:
        result = result + _to_numpy(bias)

    return _wrap_result(result)
