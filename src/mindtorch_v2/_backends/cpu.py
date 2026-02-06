"""CPU backend implementations using MindSpore PyBoost kernels.

All ops here are registered with DispatchKey.Backend_CPU.
Strategy: try pyboost first → fallback to legacy primitive → composite from simple ops.
"""

import numpy as np
import mindspore
from .._dispatch import register_op, DispatchKey
from .._tensor import Tensor
from .pyboost_cpu import (
    add_op, sub_op, mul_op, div_op, neg_op, abs_op,
    pow_op, exp_op, log_op, sqrt_op, rsqrt_op,
    sin_op, cos_op, tanh_op, sigmoid_op,
    relu_op, gelu_op, silu_op,
    matmul_op, bmm_op,
    sum_op, mean_op, max_op as max_dim_op, min_op as min_dim_op,
    equal_op, not_equal_op, greater_op, less_op, greater_equal_op, less_equal_op,
    clone_op, transpose_op, contiguous_op,
    maximum_op, minimum_op, log1p_op, erfinv_op, conj_op,
    ones_like_op, zeros_like_op,
    max_global_op, min_global_op, prod_op,
    reduce_all_op, reduce_any_op, logical_not_op,
    _get_ms_data, _wrap_result,
)


# --- Binary math ops ---

@register_op("add", DispatchKey.Backend_CPU)
def add_cpu(a, b):
    return _wrap_result(add_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("sub", DispatchKey.Backend_CPU)
def sub_cpu(a, b):
    return _wrap_result(sub_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("mul", DispatchKey.Backend_CPU)
def mul_cpu(a, b):
    return _wrap_result(mul_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("div", DispatchKey.Backend_CPU)
def div_cpu(a, b):
    return _wrap_result(div_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("neg", DispatchKey.Backend_CPU)
def neg_cpu(a):
    return _wrap_result(neg_op(_get_ms_data(a)))


@register_op("abs", DispatchKey.Backend_CPU)
def abs_cpu(a):
    return _wrap_result(abs_op(_get_ms_data(a)))


@register_op("pow", DispatchKey.Backend_CPU)
def pow_cpu(a, exponent):
    return _wrap_result(pow_op(_get_ms_data(a), _get_ms_data(exponent)))


# --- Unary math ops ---

@register_op("exp", DispatchKey.Backend_CPU)
def exp_cpu(a):
    return _wrap_result(exp_op(_get_ms_data(a)))


@register_op("log", DispatchKey.Backend_CPU)
def log_cpu(a):
    return _wrap_result(log_op(_get_ms_data(a)))


@register_op("sqrt", DispatchKey.Backend_CPU)
def sqrt_cpu(a):
    return _wrap_result(sqrt_op(_get_ms_data(a)))


@register_op("rsqrt", DispatchKey.Backend_CPU)
def rsqrt_cpu(a):
    return _wrap_result(rsqrt_op(_get_ms_data(a)))


@register_op("reciprocal", DispatchKey.Backend_CPU)
def reciprocal_cpu(a):
    # Composite: 1/x using numpy fallback (OnesLikeExt not available on CPU)
    ms_a = _get_ms_data(a)
    ones = mindspore.Tensor(np.ones(ms_a.shape, dtype=ms_a.asnumpy().dtype))
    return _wrap_result(div_op(ones, ms_a))


@register_op("maximum", DispatchKey.Backend_CPU)
def maximum_cpu(a, b):
    return _wrap_result(maximum_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("minimum", DispatchKey.Backend_CPU)
def minimum_cpu(a, b):
    return _wrap_result(minimum_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("log1p", DispatchKey.Backend_CPU)
def log1p_cpu(a):
    return _wrap_result(log1p_op(_get_ms_data(a)))


@register_op("erfinv", DispatchKey.Backend_CPU)
def erfinv_cpu(a):
    return _wrap_result(erfinv_op(_get_ms_data(a)))


@register_op("conj", DispatchKey.Backend_CPU)
def conj_cpu(a):
    return _wrap_result(conj_op(_get_ms_data(a)))


# --- Trigonometric ops ---

@register_op("sin", DispatchKey.Backend_CPU)
def sin_cpu(a):
    return _wrap_result(sin_op(_get_ms_data(a)))


@register_op("cos", DispatchKey.Backend_CPU)
def cos_cpu(a):
    return _wrap_result(cos_op(_get_ms_data(a)))


@register_op("tanh", DispatchKey.Backend_CPU)
def tanh_cpu(a):
    return _wrap_result(tanh_op(_get_ms_data(a)))


# --- Activation ops ---

@register_op("sigmoid", DispatchKey.Backend_CPU)
def sigmoid_cpu(a):
    return _wrap_result(sigmoid_op(_get_ms_data(a)))


@register_op("relu", DispatchKey.Backend_CPU)
def relu_cpu(a):
    return _wrap_result(relu_op(_get_ms_data(a)))


@register_op("gelu", DispatchKey.Backend_CPU)
def gelu_cpu(a, approximate='none'):
    ms_a = _get_ms_data(a)
    # GeLU pyboost only supports default mode; composite tanh approximate
    if approximate == 'tanh':
        import math
        coef = math.sqrt(2.0 / math.pi)
        # x * 0.5 * (1 + tanh(coef * (x + 0.044715 * x^3)))
        x3 = mul_op(mul_op(ms_a, ms_a), ms_a)
        inner = add_op(ms_a, mul_op(mindspore.Tensor(0.044715, ms_a.dtype), x3))
        inner = mul_op(mindspore.Tensor(coef, ms_a.dtype), inner)
        result = mul_op(mul_op(mindspore.Tensor(0.5, ms_a.dtype), ms_a),
                        add_op(mindspore.Tensor(1.0, ms_a.dtype), tanh_op(inner)))
        return _wrap_result(result)
    return _wrap_result(gelu_op(ms_a))


@register_op("silu", DispatchKey.Backend_CPU)
def silu_cpu(a):
    return _wrap_result(silu_op(_get_ms_data(a)))


@register_op("softmax", DispatchKey.Backend_CPU)
def softmax_cpu(a, dim=None):
    """Composite softmax: exp(x - max) / sum(exp(x - max))."""
    ms_a = _get_ms_data(a)
    if dim is None:
        dim = -1
    # Numerical stability: subtract max
    a_max = max_dim_op(ms_a, dim, True)[0]
    shifted = sub_op(ms_a, a_max)
    exp_a = exp_op(shifted)
    sum_exp = sum_op(exp_a, dim, True)
    return _wrap_result(div_op(exp_a, sum_exp))


@register_op("log_softmax", DispatchKey.Backend_CPU)
def log_softmax_cpu(a, dim=None):
    """Composite log_softmax: x - max - log(sum(exp(x - max)))."""
    ms_a = _get_ms_data(a)
    if dim is None:
        dim = -1
    a_max = max_dim_op(ms_a, dim, True)[0]
    shifted = sub_op(ms_a, a_max)
    exp_a = exp_op(shifted)
    sum_exp = sum_op(exp_a, dim, True)
    log_sum_exp = add_op(a_max, log_op(sum_exp))
    return _wrap_result(sub_op(ms_a, log_sum_exp))


# --- Matrix ops ---

@register_op("matmul", DispatchKey.Backend_CPU)
def matmul_cpu(a, b):
    return _wrap_result(matmul_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("bmm", DispatchKey.Backend_CPU)
def bmm_cpu(a, b):
    return _wrap_result(bmm_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("baddbmm", DispatchKey.Backend_CPU)
def baddbmm_cpu(input, batch1, batch2, beta=1, alpha=1):
    """Composite: beta*input + alpha*(batch1 @ batch2)."""
    ms_input = _get_ms_data(input)
    ms_b1 = _get_ms_data(batch1)
    ms_b2 = _get_ms_data(batch2)
    mm_result = bmm_op(ms_b1, ms_b2)
    if alpha != 1:
        mm_result = mul_op(mindspore.Tensor(alpha, ms_input.dtype), mm_result)
    if beta != 1:
        ms_input = mul_op(mindspore.Tensor(beta, ms_input.dtype), ms_input)
    return _wrap_result(add_op(ms_input, mm_result))


@register_op("transpose", DispatchKey.Backend_CPU)
def transpose_cpu(a, dim0, dim1):
    ms_a = _get_ms_data(a)
    ndim = ms_a.ndim
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return _wrap_result(transpose_op(ms_a, tuple(perm)))


# --- Reduction ops ---

@register_op("sum", DispatchKey.Backend_CPU)
def sum_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    if dim is None:
        dim = tuple(range(ms_a.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    return _wrap_result(sum_op(ms_a, dim, keepdim))


@register_op("mean", DispatchKey.Backend_CPU)
def mean_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    if dim is None:
        dim = tuple(range(ms_a.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    return _wrap_result(mean_op(ms_a, dim, keepdim))


@register_op("max", DispatchKey.Backend_CPU)
def max_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    if dim is None:
        # Global max - use max_global_op
        result = max_global_op(ms_a)
        return _wrap_result(result)
    result = max_dim_op(ms_a, dim, keepdim)
    from collections import namedtuple
    MaxResult = namedtuple('MaxResult', ['values', 'indices'])
    return MaxResult(_wrap_result(result[0]), _wrap_result(result[1]))


@register_op("min", DispatchKey.Backend_CPU)
def min_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    if dim is None:
        # Global min - use min_global_op
        result = min_global_op(ms_a)
        return _wrap_result(result)
    result = min_dim_op(ms_a, dim, keepdim)
    from collections import namedtuple
    MinResult = namedtuple('MinResult', ['values', 'indices'])
    return MinResult(_wrap_result(result[0]), _wrap_result(result[1]))


@register_op("prod", DispatchKey.Backend_CPU)
def prod_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    if dim is None:
        # Global prod - flatten and reduce
        result = prod_op(ms_a.reshape(-1), 0, False)
    else:
        result = prod_op(ms_a, dim, keepdim)
    return _wrap_result(result)


@register_op("argmax", DispatchKey.Backend_CPU)
def argmax_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    # Use NumPy fallback since ArgMaxExt is not registered on CPU
    import numpy as np
    arr = ms_a.asnumpy()
    if dim is None:
        result = np.argmax(arr)
    else:
        result = np.argmax(arr, axis=dim)
        if keepdim:
            result = np.expand_dims(result, axis=dim)
    return _wrap_result(mindspore.Tensor(result))


@register_op("argmin", DispatchKey.Backend_CPU)
def argmin_cpu(a, dim=None, keepdim=False):
    ms_a = _get_ms_data(a)
    # Use NumPy fallback since ArgMinExt is not registered on CPU
    import numpy as np
    arr = ms_a.asnumpy()
    if dim is None:
        result = np.argmin(arr)
    else:
        result = np.argmin(arr, axis=dim)
        if keepdim:
            result = np.expand_dims(result, axis=dim)
    return _wrap_result(mindspore.Tensor(result))


@register_op("var", DispatchKey.Backend_CPU)
def var_cpu(a, dim=None, correction=1, keepdim=False):
    """Composite variance: mean((x - mean)^2) with Bessel correction."""
    ms_a = _get_ms_data(a)
    if dim is None:
        dim = tuple(range(ms_a.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    # Compute mean
    ms_mean = mean_op(ms_a, dim, True)
    # Compute squared deviations
    diff = sub_op(ms_a, ms_mean)
    sq_diff = mul_op(diff, diff)
    # Sum and apply correction
    sq_sum = sum_op(sq_diff, dim, keepdim)
    # Count elements
    n = 1
    for d in dim:
        n *= ms_a.shape[d]
    # Apply Bessel's correction
    denom = max(1, n - correction)
    result = div_op(sq_sum, mindspore.Tensor(float(denom), ms_a.dtype))
    return _wrap_result(result)


@register_op("std", DispatchKey.Backend_CPU)
def std_cpu(a, dim=None, correction=1, keepdim=False):
    """Composite std: sqrt(var)."""
    ms_a = _get_ms_data(a)
    if dim is None:
        dim = tuple(range(ms_a.ndim))
    elif isinstance(dim, int):
        dim = (dim,)
    # Compute variance first
    ms_mean = mean_op(ms_a, dim, True)
    diff = sub_op(ms_a, ms_mean)
    sq_diff = mul_op(diff, diff)
    sq_sum = sum_op(sq_diff, dim, keepdim)
    n = 1
    for d in dim:
        n *= ms_a.shape[d]
    denom = max(1, n - correction)
    var_result = div_op(sq_sum, mindspore.Tensor(float(denom), ms_a.dtype))
    # Take sqrt
    result = sqrt_op(var_result)
    return _wrap_result(result)


# --- Comparison ops ---

@register_op("eq", DispatchKey.Backend_CPU)
def eq_cpu(a, b):
    return _wrap_result(equal_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("ne", DispatchKey.Backend_CPU)
def ne_cpu(a, b):
    return _wrap_result(not_equal_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("gt", DispatchKey.Backend_CPU)
def gt_cpu(a, b):
    return _wrap_result(greater_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("lt", DispatchKey.Backend_CPU)
def lt_cpu(a, b):
    return _wrap_result(less_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("ge", DispatchKey.Backend_CPU)
def ge_cpu(a, b):
    return _wrap_result(greater_equal_op(_get_ms_data(a), _get_ms_data(b)))


@register_op("le", DispatchKey.Backend_CPU)
def le_cpu(a, b):
    return _wrap_result(less_equal_op(_get_ms_data(a), _get_ms_data(b)))


# --- Clamp ops ---

@register_op("clamp", DispatchKey.Backend_CPU)
def clamp_cpu(a, min=None, max=None):
    """Composite clamp using ClampScalar or ClampTensor."""
    ms_a = _get_ms_data(a)
    from mindspore.ops.auto_generate.gen_ops_prim import ClampScalar
    clamp_scalar_op = ClampScalar().set_device('CPU')
    # ClampScalar expects (input, min, max) where min/max can be None
    return _wrap_result(clamp_scalar_op(ms_a, min, max))


# --- Neural network ops ---

@register_op("embedding", DispatchKey.Backend_CPU)
def embedding_cpu(indices, weight):
    """Composite embedding lookup using gather."""
    ms_indices = _get_ms_data(indices)
    ms_weight = _get_ms_data(weight)
    # Embedding is just gather along first dimension
    # weight[indices] - gather rows from weight matrix
    from mindspore.ops.auto_generate.gen_ops_prim import GatherD
    gather_op = GatherD().set_device('CPU')
    try:
        # Flatten indices, gather, then reshape
        orig_shape = ms_indices.shape
        flat_indices = ms_indices.reshape(-1)
        # Expand indices to match weight embedding dim
        expanded = flat_indices.reshape(-1, 1).broadcast_to((-1, ms_weight.shape[1]))
        result = gather_op(ms_weight, 0, expanded.astype(mindspore.int32))
        # Reshape to original indices shape + embedding dim
        final_shape = orig_shape + (ms_weight.shape[1],)
        result = result.reshape(final_shape)
        return _wrap_result(result)
    except Exception:
        # Fallback using numpy indexing
        weight_np = ms_weight.asnumpy()
        indices_np = ms_indices.asnumpy().astype(np.int32)
        result_np = weight_np[indices_np]
        return _wrap_result(mindspore.Tensor(result_np))


@register_op("dropout", DispatchKey.Backend_CPU)
def dropout_cpu(a, p=0.5, training=True, return_mask=False):
    """Composite dropout using random mask."""
    if not training or p == 0:
        if return_mask:
            return a, None
        return a
    ms_a = _get_ms_data(a)
    # Generate random mask
    mask = np.random.random(ms_a.shape) > p
    mask_tensor = mindspore.Tensor(mask.astype(np.float32))
    # Apply mask and scale
    scale = 1.0 / (1.0 - p)
    result = mul_op(ms_a, mask_tensor)
    result = mul_op(result, mindspore.Tensor(scale, ms_a.dtype))
    result_tensor = _wrap_result(result)
    if return_mask:
        return result_tensor, mask
    return result_tensor


@register_op("layer_norm", DispatchKey.Backend_CPU)
def layer_norm_cpu(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Layer normalization using composite ops."""
    ms_a = _get_ms_data(a)
    ms_weight = _get_ms_data(weight) if weight is not None else None
    ms_bias = _get_ms_data(bias) if bias is not None else None

    # Composite implementation using pyboost ops
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))
    # Use mean_op for mean computation
    ms_mean = mean_op(ms_a, axes, True)
    diff = sub_op(ms_a, ms_mean)
    # variance = mean(diff^2)
    var = mean_op(mul_op(diff, diff), axes, True)
    # normalize: diff / sqrt(var + eps)
    eps_tensor = mindspore.Tensor(eps, ms_a.dtype)
    result = div_op(diff, sqrt_op(add_op(var, eps_tensor)))
    if ms_weight is not None:
        result = mul_op(result, ms_weight)
    if ms_bias is not None:
        result = add_op(result, ms_bias)
    return _wrap_result(result)


# --- Tensor manipulation ops ---

@register_op("cat", DispatchKey.Backend_CPU)
def cat_cpu(tensors, dim=0):
    ms_tensors = []
    for t in tensors:
        ms_t = _get_ms_data(t)
        if ms_t.size > 0 or ms_t.ndim > 1:
            ms_tensors.append(ms_t)
    if len(ms_tensors) == 0:
        return tensors[0] if len(tensors) > 0 else _wrap_result(mindspore.Tensor([]))
    if len(ms_tensors) == 1:
        return _wrap_result(ms_tensors[0])
    # Concat primitive requires axis set at creation time, use numpy fallback
    import numpy as np
    np_tensors = [t.asnumpy() for t in ms_tensors]
    result = np.concatenate(np_tensors, axis=dim)
    return _wrap_result(mindspore.Tensor(result))


@register_op("stack", DispatchKey.Backend_CPU)
def stack_cpu(tensors, dim=0):
    ms_tensors = [_get_ms_data(t) for t in tensors]
    # Stack primitive requires axis set at creation time, use numpy fallback
    import numpy as np
    np_tensors = [t.asnumpy() for t in ms_tensors]
    result = np.stack(np_tensors, axis=dim)
    return _wrap_result(mindspore.Tensor(result))


@register_op("split", DispatchKey.Backend_CPU)
def split_cpu(tensor, split_size_or_sections, dim=0):
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled
    from .._autograd.node import Node, AccumulateGrad

    ms_t = _get_ms_data(tensor)
    from mindspore.ops.auto_generate.gen_ops_prim import SplitWithSize, SplitTensor

    # Compute split boundaries
    if isinstance(split_size_or_sections, int):
        size = ms_t.shape[dim]
        num_splits = (size + split_size_or_sections - 1) // split_size_or_sections
        split_sizes = []
        for i in range(num_splits):
            start = i * split_size_or_sections
            end = min(start + split_size_or_sections, size)
            split_sizes.append(end - start)
    else:
        split_sizes = list(split_size_or_sections)

    # Compute start positions
    starts = [0]
    for sz in split_sizes[:-1]:
        starts.append(starts[-1] + sz)

    # Use slicing for each split - this handles autograd through narrow/slice
    results = []
    for i, (start, sz) in enumerate(zip(starts, split_sizes)):
        # Use tensor's __getitem__ which has autograd support
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(start, start + sz)
        result = tensor[tuple(slices)]
        results.append(result)

    return tuple(results)


@register_op("chunk", DispatchKey.Backend_CPU)
def chunk_cpu(input, chunks, dim=0):
    ms_t = _get_ms_data(input)
    from mindspore.ops.auto_generate.gen_ops_prim import Chunk
    chunk_op = Chunk().set_device('CPU')
    try:
        results = chunk_op(ms_t, chunks, dim)
        return tuple(_wrap_result(r) for r in results)
    except Exception:
        # Composite fallback
        size = ms_t.shape[dim]
        chunk_size = (size + chunks - 1) // chunks
        results = []
        for i in range(chunks):
            start = i * chunk_size
            end = min(start + chunk_size, size)
            if start >= size:
                break
            slices = [slice(None)] * ms_t.ndim
            slices[dim] = slice(start, end)
            results.append(ms_t[tuple(slices)])
        return tuple(_wrap_result(r) for r in results)


@register_op("clone", DispatchKey.Backend_CPU)
def clone_cpu(input):
    """Composite clone: copy tensor data."""
    ms_a = _get_ms_data(input)
    # Clone by adding zero - creates new tensor with same values
    result = add_op(ms_a, mindspore.Tensor(0.0, ms_a.dtype))
    return _wrap_result(result)


@register_op("contiguous", DispatchKey.Backend_CPU)
def contiguous_cpu(input):
    """Make tensor contiguous using native CPU op."""
    from .._tensor import Tensor
    if isinstance(input, Tensor):
        if input.is_contiguous():
            return input
        # Use native contiguous op to stay on device
        ms_data = _get_ms_data(input)
        result = contiguous_op(ms_data)
        return _wrap_result(result)
    return input


@register_op("where", DispatchKey.Backend_CPU)
def where_cpu(condition, input, other):
    """Composite where: cond*input + (1-cond)*other."""
    ms_cond = _get_ms_data(condition).astype(mindspore.float32)
    ms_input = _get_ms_data(input)
    ms_other = _get_ms_data(other)
    # where = cond * input + (1 - cond) * other
    one = mindspore.Tensor(1.0, ms_cond.dtype)
    inv_cond = sub_op(one, ms_cond)
    term1 = mul_op(ms_cond, ms_input.astype(ms_cond.dtype))
    term2 = mul_op(inv_cond, ms_other.astype(ms_cond.dtype))
    result = add_op(term1, term2)
    return _wrap_result(result.astype(ms_input.dtype))


# --- all/any ops ---

@register_op("all", DispatchKey.Backend_CPU)
def all_cpu(input, dim=None, keepdim=False):
    """Composite all: reduce using logical AND."""
    ms_a = _get_ms_data(input)
    ms_bool = ms_a.astype(mindspore.bool_)
    from mindspore.ops.auto_generate.gen_ops_prim import ReduceAll
    reduce_all_op = ReduceAll().set_device('CPU')
    try:
        if dim is None:
            # Reduce all dims
            result = reduce_all_op(ms_bool, tuple(range(ms_a.ndim)), keepdim)
        else:
            if isinstance(dim, int):
                dim = (dim,)
            result = reduce_all_op(ms_bool, dim, keepdim)
        return _wrap_result(result)
    except Exception:
        # Fallback: use numpy
        arr = ms_bool.asnumpy()
        if dim is None:
            result = np.all(arr, keepdims=keepdim)
        else:
            result = np.all(arr, axis=dim, keepdims=keepdim)
        return _wrap_result(mindspore.Tensor(result))


@register_op("any", DispatchKey.Backend_CPU)
def any_cpu(input, dim=None, keepdim=False):
    """Composite any: reduce using logical OR."""
    ms_a = _get_ms_data(input)
    ms_bool = ms_a.astype(mindspore.bool_)
    from mindspore.ops.auto_generate.gen_ops_prim import ReduceAny
    try:
        reduce_any_op = ReduceAny().set_device('CPU')
        if dim is None:
            result = reduce_any_op(ms_bool, tuple(range(ms_a.ndim)), keepdim)
        else:
            if isinstance(dim, int):
                dim = (dim,)
            result = reduce_any_op(ms_bool, dim, keepdim)
        return _wrap_result(result)
    except Exception:
        # Fallback: use numpy
        arr = ms_bool.asnumpy()
        if dim is None:
            result = np.any(arr, keepdims=keepdim)
        else:
            result = np.any(arr, axis=dim, keepdims=keepdim)
        return _wrap_result(mindspore.Tensor(result))


@register_op("isin", DispatchKey.Backend_CPU)
def isin_cpu(elements, test_elements, *, assume_unique=False, invert=False):
    # Composite: broadcast comparison
    ms_elem = _get_ms_data(elements)
    ms_test = _get_ms_data(test_elements).reshape(-1)
    # Compare each element against all test elements
    elem_flat = ms_elem.reshape(-1)
    # Use broadcasting: elem_flat[:, None] == ms_test[None, :]
    expanded = elem_flat.reshape(-1, 1)
    test_expanded = ms_test.reshape(1, -1)
    matches = equal_op(expanded, test_expanded)
    result = reduce_any_op(matches, 1, False)
    result = result.reshape(ms_elem.shape)
    if invert:
        result = logical_not_op(result)
    return _wrap_result(result)


@register_op("topk", DispatchKey.Backend_CPU)
def topk_cpu(input, k, dim=-1, largest=True, sorted=True):
    ms_a = _get_ms_data(input)
    from mindspore.ops.auto_generate.gen_ops_prim import TopkExt
    topk_ext_op = TopkExt().set_device('CPU')
    try:
        values, indices = topk_ext_op(ms_a, k, dim, largest, sorted)
        from collections import namedtuple
        TopKResult = namedtuple('TopKResult', ['values', 'indices'])
        return TopKResult(_wrap_result(values), _wrap_result(indices))
    except Exception:
        # Fallback to numpy
        arr = ms_a.asnumpy()
        if dim < 0:
            dim = arr.ndim + dim
        if largest:
            indices = np.argpartition(arr, -k, axis=dim)
            indices = np.take(indices, range(-k, 0), axis=dim)
            values = np.take_along_axis(arr, indices, axis=dim)
            if sorted:
                sort_idx = np.argsort(-values, axis=dim)
                values = np.take_along_axis(values, sort_idx, axis=dim)
                indices = np.take_along_axis(indices, sort_idx, axis=dim)
        else:
            indices = np.argpartition(arr, k, axis=dim)
            indices = np.take(indices, range(k), axis=dim)
            values = np.take_along_axis(arr, indices, axis=dim)
            if sorted:
                sort_idx = np.argsort(values, axis=dim)
                values = np.take_along_axis(values, sort_idx, axis=dim)
                indices = np.take_along_axis(indices, sort_idx, axis=dim)
        from collections import namedtuple
        TopKResult = namedtuple('TopKResult', ['values', 'indices'])
        return TopKResult(
            _wrap_result(mindspore.Tensor(values)),
            _wrap_result(mindspore.Tensor(indices.astype(np.int64)))
        )


@register_op("multinomial", DispatchKey.Backend_CPU)
def multinomial_cpu(input, num_samples, replacement=False, *, generator=None):
    ms_a = _get_ms_data(input)
    # Use NumPy fallback directly as MultinomialExt has API issues
    probs = ms_a.asnumpy()
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    # Normalize probabilities
    probs = probs / probs.sum(axis=-1, keepdims=True)
    # Handle NaN - replace with uniform
    nan_mask = np.isnan(probs)
    if nan_mask.any():
        probs = np.where(nan_mask, 1.0 / probs.shape[-1], probs)
        probs = probs / probs.sum(axis=-1, keepdims=True)
    results = []
    for row in probs:
        samples = np.random.choice(len(row), size=num_samples, replace=replacement, p=row)
        results.append(samples)
    result = np.array(results, dtype=np.int64)
    if squeeze:
        result = result.squeeze(0)
    return _wrap_result(mindspore.Tensor(result))


# --- In-place operations ---
# Note: PyBoost InplaceAddExt/SubExt/Mul/Div are not registered on CPU.
# Use composite implementation that modifies the underlying tensor directly.


@register_op("add_", DispatchKey.Backend_CPU)
def add_inplace_cpu(a, b):
    """In-place add using composite ops."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    if ms_b.shape != ms_a.shape:
        ms_b = ms_b.broadcast_to(ms_a.shape)
    # Compute result and update storage in-place
    result = add_op(ms_a, ms_b)
    a._storage._ms_tensor = result
    a._version += 1
    return a


@register_op("sub_", DispatchKey.Backend_CPU)
def sub_inplace_cpu(a, b):
    """In-place sub using composite ops."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    if ms_b.shape != ms_a.shape:
        ms_b = ms_b.broadcast_to(ms_a.shape)
    result = sub_op(ms_a, ms_b)
    a._storage._ms_tensor = result
    a._version += 1
    return a


@register_op("mul_", DispatchKey.Backend_CPU)
def mul_inplace_cpu(a, b):
    """In-place mul using composite ops."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    if ms_b.shape != ms_a.shape:
        ms_b = ms_b.broadcast_to(ms_a.shape)
    result = mul_op(ms_a, ms_b)
    a._storage._ms_tensor = result
    a._version += 1
    return a


@register_op("div_", DispatchKey.Backend_CPU)
def div_inplace_cpu(a, b):
    """In-place div using composite ops."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    if ms_b.shape != ms_a.shape:
        ms_b = ms_b.broadcast_to(ms_a.shape)
    result = div_op(ms_a, ms_b)
    a._storage._ms_tensor = result
    a._version += 1
    return a


@register_op("zero_", DispatchKey.Backend_CPU)
def zero_inplace_cpu(a):
    """In-place zero using composite ops."""
    ms_a = a._storage.ms_tensor
    # Create zeros by multiplying with 0
    result = mul_op(ms_a, mindspore.Tensor(0.0, ms_a.dtype))
    a._storage._ms_tensor = result
    a._version += 1
    return a


@register_op("fill_", DispatchKey.Backend_CPU)
def fill_inplace_cpu(a, value):
    """In-place fill using composite ops."""
    ms_a = a._storage.ms_tensor
    # Create filled tensor: zeros + value
    zeros = mul_op(ms_a, mindspore.Tensor(0.0, ms_a.dtype))
    result = add_op(zeros, mindspore.Tensor(value, ms_a.dtype))
    a._storage._ms_tensor = result
    a._version += 1
    return a


@register_op("copy_", DispatchKey.Backend_CPU)
def copy_inplace_cpu(a, src, non_blocking=False):
    """In-place copy using composite ops."""
    ms_a = a._storage.ms_tensor
    ms_src = _get_ms_data(src)
    if ms_src.shape != ms_a.shape:
        ms_src = ms_src.broadcast_to(ms_a.shape)
    # Cast to same dtype if needed
    if ms_src.dtype != ms_a.dtype:
        ms_src = ms_src.astype(ms_a.dtype)
    a._storage._ms_tensor = ms_src
    a._version += 1
    return a
