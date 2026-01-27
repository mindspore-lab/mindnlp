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
def dropout_cpu(a, p=0.5, training=True, return_mask=False):
    """Dropout."""
    if not training or p == 0:
        if return_mask:
            return a, None
        return a
    a_np = _to_numpy(a)
    mask = np.random.binomial(1, 1 - p, a_np.shape) / (1 - p)
    result = a_np * mask
    result_tensor = _wrap_result(result.astype(a_np.dtype))
    if return_mask:
        return result_tensor, mask
    return result_tensor


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


# --- Tensor manipulation ops ---

@register_op("cat", DispatchKey.Backend_CPU)
def cat_cpu(tensors, dim=0):
    """Concatenate tensors along a dimension."""
    # Filter out empty tensors that have different dimensions (common in KV cache)
    non_empty = []
    for t in tensors:
        arr = _to_numpy(t)
        if arr.size > 0:
            non_empty.append(arr)
        elif arr.ndim > 1:
            # Keep empty tensors only if they have the right shape (just 0 in one dim)
            non_empty.append(arr)

    if len(non_empty) == 0:
        # All tensors are empty, return first one
        return tensors[0] if len(tensors) > 0 else _wrap_result(np.array([]))
    elif len(non_empty) == 1:
        # Only one non-empty tensor, return it
        return _wrap_result(non_empty[0])

    result = np.concatenate(non_empty, axis=dim)
    return _wrap_result(result)


@register_op("stack", DispatchKey.Backend_CPU)
def stack_cpu(tensors, dim=0):
    """Stack tensors along a new dimension."""
    arrays = [_to_numpy(t) for t in tensors]
    result = np.stack(arrays, axis=dim)
    return _wrap_result(result)


@register_op("split", DispatchKey.Backend_CPU)
def split_cpu(tensor, split_size_or_sections, dim=0):
    """Split tensor into chunks."""
    arr = _to_numpy(tensor)
    if isinstance(split_size_or_sections, int):
        # Split into chunks of given size
        size = arr.shape[dim]
        indices = list(range(split_size_or_sections, size, split_size_or_sections))
        splits = np.split(arr, indices, axis=dim)
    else:
        # Split at given section sizes
        indices = np.cumsum(split_size_or_sections[:-1])
        splits = np.split(arr, indices, axis=dim)
    return tuple(_wrap_result(s) for s in splits)


@register_op("chunk", DispatchKey.Backend_CPU)
def chunk_cpu(input, chunks, dim=0):
    """Split tensor into specified number of chunks."""
    arr = _to_numpy(input)
    splits = np.array_split(arr, chunks, axis=dim)
    return tuple(_wrap_result(s) for s in splits)


@register_op("clone", DispatchKey.Backend_CPU)
def clone_cpu(input):
    """Create a copy of tensor with new storage."""
    arr = _to_numpy(input)
    return _wrap_result(arr.copy())


@register_op("where", DispatchKey.Backend_CPU)
def where_cpu(condition, input, other):
    """Select elements based on condition."""
    cond_np = _to_numpy(condition).astype(bool)
    input_np = _to_numpy(input)
    other_np = _to_numpy(other)
    result = np.where(cond_np, input_np, other_np)
    return _wrap_result(result)


# --- Additional math ops ---

@register_op("var", DispatchKey.Backend_CPU)
def var_cpu(a, dim=None, correction=1, keepdim=False):
    """Variance reduction."""
    a_np = _to_numpy(a)
    # Use ddof for Bessel's correction
    result = np.var(a_np, axis=dim, ddof=correction, keepdims=keepdim)
    return _wrap_result(result)


@register_op("std", DispatchKey.Backend_CPU)
def std_cpu(a, dim=None, correction=1, keepdim=False):
    """Standard deviation reduction."""
    a_np = _to_numpy(a)
    result = np.std(a_np, axis=dim, ddof=correction, keepdims=keepdim)
    return _wrap_result(result)


@register_op("clamp", DispatchKey.Backend_CPU)
def clamp_cpu(a, min=None, max=None):
    """Clamp values to range [min, max]."""
    a_np = _to_numpy(a)
    result = np.clip(a_np, min, max)
    return _wrap_result(result)


@register_op("rsqrt", DispatchKey.Backend_CPU)
def rsqrt_cpu(a):
    """Reciprocal square root element-wise."""
    a_np = _to_numpy(a)
    result = 1.0 / np.sqrt(a_np)
    return _wrap_result(result)


@register_op("reciprocal", DispatchKey.Backend_CPU)
def reciprocal_cpu(a):
    """Reciprocal (1/x) element-wise."""
    a_np = _to_numpy(a)
    result = 1.0 / a_np
    return _wrap_result(result)


@register_op("bmm", DispatchKey.Backend_CPU)
def bmm_cpu(a, b):
    """Batched matrix multiplication."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = np.matmul(a_np, b_np)
    return _wrap_result(result)


@register_op("baddbmm", DispatchKey.Backend_CPU)
def baddbmm_cpu(input, batch1, batch2, beta=1, alpha=1):
    """Batched matrix multiply with add: beta*input + alpha*(batch1 @ batch2)."""
    input_np = _to_numpy(input)
    batch1_np = _to_numpy(batch1)
    batch2_np = _to_numpy(batch2)
    result = beta * input_np + alpha * np.matmul(batch1_np, batch2_np)
    return _wrap_result(result)


@register_op("all", DispatchKey.Backend_CPU)
def all_cpu(input, dim=None, keepdim=False):
    """Test if all elements evaluate to True."""
    arr = _to_numpy(input)
    if dim is None:
        result = np.all(arr)
    else:
        result = np.all(arr, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("any", DispatchKey.Backend_CPU)
def any_cpu(input, dim=None, keepdim=False):
    """Test if any element evaluates to True."""
    arr = _to_numpy(input)
    if dim is None:
        result = np.any(arr)
    else:
        result = np.any(arr, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("isin", DispatchKey.Backend_CPU)
def isin_cpu(elements, test_elements, *, assume_unique=False, invert=False):
    """Test if each element is in test_elements."""
    elements_np = _to_numpy(elements)
    test_np = _to_numpy(test_elements)
    result = np.isin(elements_np, test_np, assume_unique=assume_unique, invert=invert)
    return _wrap_result(result)


@register_op("topk", DispatchKey.Backend_CPU)
def topk_cpu(input, k, dim=-1, largest=True, sorted=True):
    """Return the k largest/smallest elements along a dimension."""
    arr = _to_numpy(input)
    if dim < 0:
        dim = arr.ndim + dim

    # Use partition for efficiency
    if largest:
        # Get indices of top k
        indices = np.argpartition(arr, -k, axis=dim)
        # Take the top k indices
        indices = np.take(indices, range(-k, 0), axis=dim)
        # Get the values at those indices
        values = np.take_along_axis(arr, indices, axis=dim)
        # Sort if needed
        if sorted:
            sort_indices = np.argsort(-values, axis=dim)
            values = np.take_along_axis(values, sort_indices, axis=dim)
            indices = np.take_along_axis(indices, sort_indices, axis=dim)
    else:
        # Get indices of bottom k
        indices = np.argpartition(arr, k, axis=dim)
        indices = np.take(indices, range(k), axis=dim)
        values = np.take_along_axis(arr, indices, axis=dim)
        if sorted:
            sort_indices = np.argsort(values, axis=dim)
            values = np.take_along_axis(values, sort_indices, axis=dim)
            indices = np.take_along_axis(indices, sort_indices, axis=dim)

    from collections import namedtuple
    TopKResult = namedtuple('TopKResult', ['values', 'indices'])
    return TopKResult(_wrap_result(values), _wrap_result(indices.astype(np.int64)))


@register_op("multinomial", DispatchKey.Backend_CPU)
def multinomial_cpu(input, num_samples, replacement=False, *, generator=None):
    """Draw samples from a multinomial distribution."""
    probs = _to_numpy(input)

    # Handle 1D case
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        squeeze_result = True
    else:
        squeeze_result = False

    # Normalize probabilities
    probs = probs / probs.sum(axis=-1, keepdims=True)

    # Sample for each row
    results = []
    for row in probs:
        if replacement:
            samples = np.random.choice(len(row), size=num_samples, replace=True, p=row)
        else:
            samples = np.random.choice(len(row), size=num_samples, replace=False, p=row)
        results.append(samples)

    result = np.array(results, dtype=np.int64)
    if squeeze_result:
        result = result.squeeze(0)

    return _wrap_result(result)
