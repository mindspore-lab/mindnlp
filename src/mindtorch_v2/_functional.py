from ._dispatch.dispatcher import dispatch
from ._autograd.grad_mode import GradMode, no_grad
from ._device import device as Device, get_default_device
from ._dtype import to_numpy_dtype

import builtins as _builtins


def add(*args, **kwargs):
    alpha = kwargs.pop("alpha", 1)
    kwargs.pop("out", None)
    if alpha != 1:
        # Pre-multiply: add(a, b, alpha=k) => a + k*b
        a, b = args[0], args[1]
        b = mul(b, alpha)
        return dispatch("add", None, a, b)
    return dispatch("add", None, *args, **kwargs)


def transpose(*args, **kwargs):
    return dispatch("transpose", None, *args, **kwargs)


def reshape(*args, **kwargs):
    return dispatch("reshape", None, *args, **kwargs)


def mul(*args, **kwargs):
    return dispatch("mul", None, *args, **kwargs)


def matmul(*args, **kwargs):
    return dispatch("matmul", None, *args, **kwargs)


def relu(*args, **kwargs):
    return dispatch("relu", None, *args, **kwargs)


def abs(a):
    return dispatch("abs", a.device.type, a)


def neg(a):
    return dispatch("neg", a.device.type, a)


def exp(a):
    return dispatch("exp", a.device.type, a)


def log(a):
    return dispatch("log", a.device.type, a)


def sqrt(a):
    return dispatch("sqrt", a.device.type, a)


def sin(a):
    return dispatch("sin", a.device.type, a)


def cos(a):
    return dispatch("cos", a.device.type, a)


def tan(a):
    return dispatch("tan", a.device.type, a)


def tanh(a):
    return dispatch("tanh", a.device.type, a)


def sigmoid(a):
    return dispatch("sigmoid", a.device.type, a)


def floor(a):
    return dispatch("floor", a.device.type, a)


def ceil(a):
    return dispatch("ceil", a.device.type, a)


def round(a):
    return dispatch("round", a.device.type, a)


def trunc(a):
    return dispatch("trunc", a.device.type, a)


def frac(a):
    return dispatch("frac", a.device.type, a)


def pow(a, b):
    return dispatch("pow", a.device.type, a, b)


def log2(a):
    return dispatch("log2", a.device.type, a)


def log10(a):
    return dispatch("log10", a.device.type, a)


def exp2(a):
    return dispatch("exp2", a.device.type, a)


def rsqrt(a):
    return dispatch("rsqrt", a.device.type, a)


def sign(a):
    return dispatch("sign", a.device.type, a)


def signbit(a):
    return dispatch("signbit", a.device.type, a)


def isnan(a):
    return dispatch("isnan", a.device.type, a)


def isinf(a):
    return dispatch("isinf", a.device.type, a)


def isfinite(a):
    return dispatch("isfinite", a.device.type, a)

def sinh(a):
    return dispatch("sinh", a.device.type, a)


def cosh(a):
    return dispatch("cosh", a.device.type, a)


def asinh(a):
    return dispatch("asinh", a.device.type, a)


def acosh(a):
    return dispatch("acosh", a.device.type, a)


def atanh(a):
    return dispatch("atanh", a.device.type, a)


def erf(a):
    return dispatch("erf", a.device.type, a)


def erfc(a):
    return dispatch("erfc", a.device.type, a)


def softplus(a):
    return dispatch("softplus", a.device.type, a)


def clamp(a, min_val=None, max_val=None):
    return dispatch("clamp", a.device.type, a, min_val, max_val)


def clamp_min(a, min_val):
    return dispatch("clamp_min", a.device.type, a, min_val)


def clamp_max(a, max_val):
    return dispatch("clamp_max", a.device.type, a, max_val)


def relu6(a):
    return dispatch("relu6", a.device.type, a)


def hardtanh(a, min_val=-1.0, max_val=1.0):
    return dispatch("hardtanh", a.device.type, a, min_val, max_val)


def min(a, b):
    return dispatch("min", a.device.type, a, b)


def max(a, b):
    return dispatch("max", a.device.type, a, b)


def amin(a, dim=None, keepdim=False):
    return dispatch("amin", a.device.type, a, dim=dim, keepdim=keepdim)


def amax(a, dim=None, keepdim=False):
    return dispatch("amax", a.device.type, a, dim=dim, keepdim=keepdim)


def fmin(a, b):
    return dispatch("fmin", a.device.type, a, b)


def fmax(a, b):
    return dispatch("fmax", a.device.type, a, b)


def where(cond, x=None, y=None):
    if x is None and y is None:
        return nonzero(cond, as_tuple=True)
    return dispatch("where", x.device.type, cond, x, y)


def atan(a):
    return dispatch("atan", a.device.type, a)


def atan2(a, b):
    return dispatch("atan2", a.device.type, a, b)


def asin(a):
    return dispatch("asin", a.device.type, a)


def acos(a):
    return dispatch("acos", a.device.type, a)


def lerp(a, b, weight):
    return dispatch("lerp", a.device.type, a, b, weight)


def addcmul(a, b, c, value=1.0):
    return dispatch("addcmul", a.device.type, a, b, c, value=value)


def addcdiv(a, b, c, value=1.0):
    return dispatch("addcdiv", a.device.type, a, b, c, value=value)


def logaddexp(a, b):
    return dispatch("logaddexp", a.device.type, a, b)


def logaddexp2(a, b):
    return dispatch("logaddexp2", a.device.type, a, b)


def hypot(a, b):
    return dispatch("hypot", a.device.type, a, b)


def remainder(a, b):
    return dispatch("remainder", a.device.type, a, b)


def fmod(a, b):
    return dispatch("fmod", a.device.type, a, b)


def div(a, b, *, rounding_mode=None):
    return dispatch("div", a.device.type, a, b)


def true_divide(a, b):
    return dispatch("true_divide", a.device.type, a, b)


def mean(a, dim=None, keepdim=False, *, dtype=None, axis=None):
    if axis is not None:
        dim = axis
    result = dispatch("mean", a.device.type, a, dim=dim, keepdim=keepdim)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def std(a, dim=None, keepdim=False, unbiased=True, *, axis=None):
    if axis is not None:
        dim = axis
    return dispatch("std", a.device.type, a, dim=dim, keepdim=keepdim, unbiased=unbiased)


def sinh(a):
    return dispatch("sinh", a.device.type, a)


def cosh(a):
    return dispatch("cosh", a.device.type, a)


def erf(a):
    return dispatch("erf", a.device.type, a)


def erfc(a):
    return dispatch("erfc", a.device.type, a)


def softplus(a):
    return dispatch("softplus", a.device.type, a)


def sum(*args, **kwargs):
    dtype = kwargs.pop("dtype", None)
    kwargs.pop("device", None)
    result = dispatch("sum", None, *args, **kwargs)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def all(a, dim=None, keepdim=False):
    return dispatch("all", a.device.type, a, dim=dim, keepdim=keepdim)


def any(a, dim=None, keepdim=False):
    return dispatch("any", a.device.type, a, dim=dim, keepdim=keepdim)


def argmax(a, dim=None, keepdim=False):
    return dispatch("argmax", a.device.type, a, dim=dim, keepdim=keepdim)


def argmin(a, dim=None, keepdim=False):
    return dispatch("argmin", a.device.type, a, dim=dim, keepdim=keepdim)


def count_nonzero(a, dim=None, keepdim=False):
    return dispatch("count_nonzero", a.device.type, a, dim=dim, keepdim=keepdim)


def nonzero(a, as_tuple=False):
    return dispatch("nonzero", a.device.type, a, as_tuple=as_tuple)


def masked_select(a, mask):
    return dispatch("masked_select", a.device.type, a, mask)


def flip(a, dims):
    return dispatch("flip", a.device.type, a, dims)


def roll(a, shifts, dims=None):
    return dispatch("roll", a.device.type, a, shifts, dims)


def rot90(a, k=1, dims=(0, 1)):
    return dispatch("rot90", a.device.type, a, k, dims)


def repeat(a, repeats):
    return dispatch("repeat", a.device.type, a, repeats)


def repeat_interleave(a, repeats, dim=None):
    return dispatch("repeat_interleave", a.device.type, a, repeats, dim)


def tile(a, dims):
    return dispatch("tile", a.device.type, a, dims)


def flatten(input, start_dim=0, end_dim=-1):
    return input.flatten(start_dim, end_dim)


def logical_and(a, b):
    return dispatch("logical_and", a.device.type, a, b)


def logical_or(a, b):
    return dispatch("logical_or", a.device.type, a, b)


def logical_not(a):
    return dispatch("logical_not", a.device.type, a)






def cumsum(a, dim=0):
    return dispatch("cumsum", a.device.type, a, dim)


def cumprod(a, dim=0):
    return dispatch("cumprod", a.device.type, a, dim)


def cummax(a, dim=0):
    return dispatch("cummax", a.device.type, a, dim)


def argsort(a, dim=-1, descending=False, stable=False, out=None):
    result = dispatch("argsort", a.device.type, a, dim=dim, descending=descending, stable=stable)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def sort(a, dim=-1, descending=False, stable=False, out=None):
    values, indices = dispatch("sort", a.device.type, a, dim=dim, descending=descending, stable=stable)
    if out is not None:
        out_values, out_indices = out
        out_values._storage = values.storage()
        out_values.shape = values.shape
        out_values.stride = values.stride
        out_values.offset = values.offset
        out_values._base = values._base
        out_values._view_meta = values._view_meta
        out_indices._storage = indices.storage()
        out_indices.shape = indices.shape
        out_indices.stride = indices.stride
        out_indices.offset = indices.offset
        out_indices._base = indices._base
        out_indices._view_meta = indices._view_meta
        return out_values, out_indices
    return values, indices


def topk(a, k, dim=-1, largest=True, sorted=True, out=None):
    values, indices = dispatch("topk", a.device.type, a, k, dim=dim, largest=largest, sorted=sorted)
    if out is not None:
        out_values, out_indices = out
        out_values._storage = values.storage()
        out_values.shape = values.shape
        out_values.stride = values.stride
        out_values.offset = values.offset
        out_values._base = values._base
        out_values._view_meta = values._view_meta
        out_indices._storage = indices.storage()
        out_indices.shape = indices.shape
        out_indices.stride = indices.stride
        out_indices.offset = indices.offset
        out_indices._base = indices._base
        out_indices._view_meta = indices._view_meta
        return out_values, out_indices
    return values, indices


def stack(tensors, dim=0, out=None):
    result = dispatch("stack", tensors[0].device.type, tensors, dim=dim)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def cat(tensors, dim=0, out=None):
    result = dispatch("cat", tensors[0].device.type, tensors, dim=dim)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def concat(tensors, dim=0, out=None):
    return cat(tensors, dim=dim, out=out)


def hstack(tensors, out=None):
    result = dispatch("hstack", tensors[0].device.type, tensors)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def vstack(tensors, out=None):
    result = dispatch("vstack", tensors[0].device.type, tensors)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def column_stack(tensors, out=None):
    result = dispatch("column_stack", tensors[0].device.type, tensors)
    if out is not None:
        out._storage = result.storage()
        out.shape = result.shape
        out.stride = result.stride
        out.offset = result.offset
        out._base = result._base
        out._view_meta = result._view_meta
        return out
    return result


def concatenate(tensors, dim=0, axis=None):
    if axis is not None:
        dim = axis
    return dispatch("concatenate", tensors[0].device.type, tensors, dim)


def row_stack(tensors):
    return dispatch("row_stack", tensors[0].device.type, tensors)


def dstack(tensors):
    return dispatch("dstack", tensors[0].device.type, tensors)


def pad_sequence(seqs, batch_first=False, padding_value=0.0, padding_side="right"):
    return dispatch(
        "pad_sequence",
        seqs[0].device.type,
        seqs,
        batch_first=batch_first,
        padding_value=padding_value,
        padding_side=padding_side,
    )


def block_diag(*tensors):
    return dispatch("block_diag", tensors[0].device.type, tensors)


def tril(a, diagonal=0):
    return dispatch("tril", a.device.type, a, diagonal)


def triu(a, diagonal=0):
    return dispatch("triu", a.device.type, a, diagonal)


def diag(a, diagonal=0):
    return dispatch("diag", a.device.type, a, diagonal)


def cartesian_prod(*tensors):
    return dispatch("cartesian_prod", tensors[0].device.type, tensors)


def chunk(a, chunks, dim=0):
    return dispatch("chunk", a.device.type, a, chunks, dim)


def split(a, split_size_or_sections, dim=0):
    return dispatch("split", a.device.type, a, split_size_or_sections, dim)


def vsplit(a, split_size_or_sections):
    return dispatch("vsplit", a.device.type, a, split_size_or_sections)


def hsplit(a, split_size_or_sections):
    return dispatch("hsplit", a.device.type, a, split_size_or_sections)


def dsplit(a, split_size_or_sections):
    return dispatch("dsplit", a.device.type, a, split_size_or_sections)


def take(a, index):
    return dispatch("take", a.device.type, a, index)


def take_along_dim(a, indices, dim):
    return dispatch("take_along_dim", a.device.type, a, indices, dim)


def index_select(a, dim, index):
    return dispatch("index_select", a.device.type, a, dim, index)


def gather(a, dim, index):
    return dispatch("gather", a.device.type, a, dim, index)


def scatter(a, dim, index, src):
    return dispatch("scatter", a.device.type, a, dim, index, src)


def unbind(a, dim=0):
    return dispatch("unbind", a.device.type, a, dim)


def tril_indices(row, col, offset=0, *, dtype=None, device=None, layout=None):
    dev = _as_device(device)
    return dispatch("tril_indices", dev, row, col, offset, dtype=dtype, device=dev, layout=layout)


def triu_indices(row, col, offset=0, *, dtype=None, device=None, layout=None):
    dev = _as_device(device)
    return dispatch("triu_indices", dev, row, col, offset, dtype=dtype, device=dev, layout=layout)


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return dispatch("allclose", a.device.type, a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return dispatch("isclose", a.device.type, a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def equal(a, b):
    return dispatch("equal", a.device.type, a, b)


def _compare_dispatch(op_name, a, b):
    if not hasattr(a, "device"):
        raise TypeError(f"'{op_name}' not supported between instances of '{type(a).__name__}' and '{type(b).__name__}'")
    if not hasattr(b, "device") and not isinstance(b, (int, float, bool)):
        raise TypeError(f"'{op_name}' not supported between instances of 'Tensor' and '{type(b).__name__}'")
    return dispatch(op_name, a.device.type, a, b)


def eq(a, b):
    return _compare_dispatch("eq", a, b)


def ne(a, b):
    return _compare_dispatch("ne", a, b)


def lt(a, b):
    return _compare_dispatch("lt", a, b)


def le(a, b):
    return _compare_dispatch("le", a, b)


def gt(a, b):
    return _compare_dispatch("gt", a, b)


def ge(a, b):
    return _compare_dispatch("ge", a, b)


def logspace(start, end, steps, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("logspace", dev, start, end, steps, dtype=dtype)


def eye(n, m=None, dtype=None, device=None, out=None):
    dev = _as_device(device)
    return dispatch("eye", dev, n, m, dtype=dtype, out=out)


def range(start, end, step=1, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("range", dev, start, end, step, dtype=dtype)


def view(*args, **kwargs):
    return dispatch("view", None, *args, **kwargs)


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    dev = _as_device(device)
    return dispatch("tensor", dev, data, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, dtype=None, device=None, memory_format=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("zeros", dev, shape, dtype=dtype, memory_format=memory_format)


def zeros_like(input, *, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return zeros(input.shape, dtype=dtype, device=device, memory_format=memory_format)


def ones(*shape, dtype=None, device=None, memory_format=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("ones", dev, shape, dtype=dtype, memory_format=memory_format)


def empty(*shape, dtype=None, device=None, memory_format=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("empty", dev, shape, dtype=dtype, memory_format=memory_format)


def randn(*shape, dtype=None, device=None, memory_format=None, generator=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("randn", dev, shape, dtype=dtype, memory_format=memory_format, generator=generator)


def rand(*shape, dtype=None, device=None, memory_format=None, generator=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("rand", dev, shape, dtype=dtype, memory_format=memory_format, generator=generator)


def randint(low, high=None, size=None, *, dtype=None, device=None, requires_grad=False, generator=None):
    dev = _as_device(device)
    return dispatch("randint", dev, low, high=high, size=size, dtype=dtype, requires_grad=requires_grad, generator=generator)


def randperm(n, *, dtype=None, device=None, requires_grad=False, generator=None):
    dev = _as_device(device)
    return dispatch("randperm", dev, n, dtype=dtype, requires_grad=requires_grad, generator=generator)


def arange(start, end=None, step=1, dtype=None, device=None):
    dev = _as_device(device)
    if end is None:
        start, end = 0, start
    return dispatch("arange", dev, start, end, step, dtype=dtype)


def linspace(start, end, steps, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("linspace", dev, start, end, steps, dtype=dtype)


def full(*args, dtype=None, device=None):
    if len(args) >= 2 and not isinstance(args[-1], (tuple, list, int)):
        # full(shape, fill_value) or full(d1, d2, ..., fill_value)
        *shape_args, fill_value = args
    elif len(args) >= 2:
        *shape_args, fill_value = args
    else:
        raise TypeError("full() requires at least a shape and fill_value")
    if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
        shape = shape_args[0]
    else:
        shape = tuple(shape_args)
    dev = _as_device(device)
    return dispatch("full", dev, shape, fill_value, dtype=dtype)


def to(a, device=None, dtype=None, non_blocking=False, copy=False, memory_format=None):
    return dispatch(
        "to",
        a.device,
        a,
        device,
        dtype=dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )



def linalg_qr(a, mode='reduced'):
    return dispatch("linalg_qr", a.device.type, a, mode)


# ---------------------------------------------------------------------------
# Tensor indexing methods
# ---------------------------------------------------------------------------

def narrow(a, dim, start, length):
    return dispatch("narrow", a.device.type, a, dim, start, length)


def select(a, dim, index):
    return dispatch("select", a.device.type, a, dim, index)


def expand(a, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    return dispatch("expand", a.device.type, a, sizes)


def masked_fill(a, mask, value):
    return dispatch("masked_fill", a.device.type, a, mask, value)


def masked_fill_(a, mask, value):
    return dispatch("masked_fill_", a.device.type, a, mask, value)


def index_put_(a, indices, values, accumulate=False):
    return dispatch("index_put_", a.device.type, a, indices, values, accumulate)


def index_put(a, indices, values, accumulate=False):
    return dispatch("index_put", a.device.type, a, indices, values, accumulate)


def index_copy_(a, dim, index, source):
    return dispatch("index_copy_", a.device.type, a, dim, index, source)


def index_fill_(a, dim, index, value):
    return dispatch("index_fill_", a.device.type, a, dim, index, value)


def index_add_(a, dim, index, source, alpha=1.0):
    return dispatch("index_add_", a.device.type, a, dim, index, source, alpha)


def scatter_(a, dim, index, src):
    return dispatch("scatter_", a.device.type, a, dim, index, src)


def scatter_add_(a, dim, index, src):
    return dispatch("scatter_add_", a.device.type, a, dim, index, src)


def masked_scatter_(a, mask, source):
    return dispatch("masked_scatter_", a.device.type, a, mask, source)


def unfold(a, dimension, size, step):
    return dispatch("unfold", a.device.type, a, dimension, size, step)


def squeeze(a, dim=None):
    return dispatch("squeeze", a.device.type, a, dim)


def unsqueeze(a, dim):
    return dispatch("unsqueeze", a.device.type, a, dim)


def permute(a, dims):
    return dispatch("permute", a.device.type, a, dims)


def var(a, dim=None, keepdim=False, unbiased=True):
    return dispatch("var", a.device.type, a, dim=dim, unbiased=unbiased, keepdim=keepdim)


def norm(a, p=2, dim=None, keepdim=False):
    return dispatch("norm", a.device.type, a, p=p, dim=dim, keepdim=keepdim)


def prod(a, dim=None, keepdim=False):
    return dispatch("prod", a.device.type, a, dim=dim, keepdim=keepdim)


def mm(a, b):
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise RuntimeError(
            f"mm: Expected 2-D tensors, got {len(a.shape)}-D and {len(b.shape)}-D"
        )
    return dispatch("matmul", a.device.type, a, b)


def bmm(a, b):
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise RuntimeError(
            f"bmm: Expected 3-D tensors, got {len(a.shape)}-D and {len(b.shape)}-D"
        )
    return dispatch("matmul", a.device.type, a, b)


def floor_divide(a, b):
    return dispatch("floor_divide", a.device.type, a, b)


def ones_like(input, *, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return ones(input.shape, dtype=dtype, device=device, memory_format=memory_format)


def empty_like(input, *, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return empty(input.shape, dtype=dtype, device=device, memory_format=memory_format)


def full_like(input, fill_value, *, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return full(input.shape, fill_value, dtype=dtype, device=device)


def randn_like(input, *, dtype=None, device=None, memory_format=None, generator=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return randn(input.shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator)


def rand_like(input, *, dtype=None, device=None, memory_format=None, generator=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return rand(input.shape, dtype=dtype, device=device, memory_format=memory_format, generator=generator)


def randint_like(input, low=0, high=None, *, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return randint(low, high, size=input.shape, dtype=dtype, device=device)


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    return dispatch("rms_norm", input.device.type, input, normalized_shape, weight, eps)


def normal(mean, std, size=None, *, generator=None, out=None):
    if size is not None:
        result = empty(size)
        result.normal_(float(mean), float(std), generator=generator)
        if out is not None:
            out.copy_(result)
            return out
        return result
    # Tensor mean/std forms
    from ._tensor import Tensor
    mean_is_tensor = isinstance(mean, Tensor)
    std_is_tensor = isinstance(std, Tensor)
    if mean_is_tensor or std_is_tensor:
        if mean_is_tensor and std_is_tensor:
            shape = mean.shape
            device = str(mean.device)
            dt = mean.dtype
        elif mean_is_tensor:
            shape = mean.shape
            device = str(mean.device)
            dt = mean.dtype
        else:
            shape = std.shape
            device = str(std.device)
            dt = std.dtype
        result = randn(shape, dtype=dt, device=device)
        result = add(mul(result, std), mean)
        if out is not None:
            out.copy_(result)
            return out
        return result
    raise TypeError("normal expects at least one of mean/std to be a Tensor, or size to be specified")


# ---------------------------------------------------------------------------
# New ops: math, logical, bitwise, shape, search
# ---------------------------------------------------------------------------

def sub(*args, **kwargs):
    alpha = kwargs.pop("alpha", 1)
    kwargs.pop("out", None)
    if alpha != 1:
        a, b = args[0], args[1]
        b = mul(b, alpha)
        return dispatch("sub", None, a, b)
    return dispatch("sub", None, *args, **kwargs)


def log1p(a):
    return dispatch("log1p", a.device.type, a)


def expm1(a):
    return dispatch("expm1", a.device.type, a)


def reciprocal(a):
    return dispatch("reciprocal", a.device.type, a)


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    return dispatch("addmm", input.device.type, input, mat1, mat2, beta=beta, alpha=alpha)


def maximum(a, b):
    return dispatch("maximum", None, a, b)


def minimum(a, b):
    return dispatch("minimum", None, a, b)


def dot(a, b):
    return dispatch("dot", None, a, b)


def outer(a, b):
    return dispatch("outer", None, a, b)


def inner(a, b):
    return dispatch("inner", None, a, b)


def mv(a, b):
    return dispatch("mv", None, a, b)


def cross(a, b, dim=-1):
    return dispatch("cross", None, a, b, dim)


def tensordot(a, b, dims=2):
    return dispatch("tensordot", None, a, b, dims)


def einsum(equation, *operands):
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    first = operands[0]
    return dispatch("einsum", first.device.type, equation, list(operands))


def logical_and(a, b):
    return dispatch("logical_and", None, a, b)


def logical_or(a, b):
    return dispatch("logical_or", None, a, b)


def logical_not(a):
    return dispatch("logical_not", a.device.type, a)


def logical_xor(a, b):
    return dispatch("logical_xor", None, a, b)


def bitwise_and(a, b):
    return dispatch("bitwise_and", None, a, b)


def bitwise_or(a, b):
    return dispatch("bitwise_or", None, a, b)


def bitwise_xor(a, b):
    return dispatch("bitwise_xor", None, a, b)


def bitwise_not(a):
    return dispatch("bitwise_not", a.device.type, a)


def flatten(a, start_dim=0, end_dim=-1):
    return dispatch("flatten", a.device.type, a, start_dim, end_dim)


def unflatten(a, dim, sizes):
    return dispatch("unflatten", a.device.type, a, dim, sizes)


def broadcast_to(a, shape):
    return dispatch("broadcast_to", a.device.type, a, shape)


def movedim(a, source, destination):
    return dispatch("movedim", a.device.type, a, source, destination)


def moveaxis(a, source, destination):
    return dispatch("moveaxis", a.device.type, a, source, destination)


def diagonal(a, offset=0, dim1=0, dim2=1):
    return dispatch("diagonal", a.device.type, a, offset, dim1, dim2)


def unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None):
    return dispatch("unique", a.device.type, a, sorted, return_inverse, return_counts, dim)


def searchsorted(sorted_seq, values, out_int32=False, right=False, side=None, sorter=None):
    return dispatch(
        "searchsorted", sorted_seq.device.type,
        sorted_seq, values, out_int32, right, side, sorter,
    )


def kthvalue(a, k, dim=-1, keepdim=False):
    return dispatch("kthvalue", a.device.type, a, k, dim, keepdim)


def median(a, dim=None, keepdim=False):
    return dispatch("median", a.device.type, a, dim, keepdim)


def baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return dispatch("baddbmm", input.device.type, input, batch1, batch2, beta=beta, alpha=alpha)


def trace(a):
    return dispatch("trace", a.device.type, a)


def cummin(a, dim):
    return dispatch("cummin", a.device.type, a, dim)


def logsumexp(a, dim, keepdim=False):
    return dispatch("logsumexp", a.device.type, a, dim, keepdim)


def renorm(a, p, dim, maxnorm):
    return dispatch("renorm", a.device.type, a, p, dim, maxnorm)


def _as_device(dev):
    if dev is None:
        return get_default_device()
    if isinstance(dev, str):
        return Device(dev)
    return dev


# ---------------------------------------------------------------------------
# Category B: Wrappers for existing schema+kernel functions
# ---------------------------------------------------------------------------

def nansum(a, dim=None, keepdim=False):
    return dispatch("nansum", a.device.type, a, dim=dim, keepdim=keepdim)


def nanmean(a, dim=None, keepdim=False):
    return dispatch("nanmean", a.device.type, a, dim=dim, keepdim=keepdim)


def det(a):
    return dispatch("det", a.device.type, a)


def dist(a, other, p=2):
    return dispatch("dist", a.device.type, a, other, p)


def matrix_power(a, n):
    return dispatch("matrix_power", a.device.type, a, n)


def argwhere(a):
    return dispatch("argwhere", a.device.type, a)


# ---------------------------------------------------------------------------
# Category C1: Pure-Python functions (no dispatch needed)
# ---------------------------------------------------------------------------

def meshgrid(*tensors, indexing='ij'):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    if indexing not in ('ij', 'xy'):
        raise ValueError(f"meshgrid: indexing must be 'ij' or 'xy', got '{indexing}'")
    if len(tensors) == 0:
        return []
    # For 'xy' indexing, swap the first two inputs, build 'ij', then swap outputs
    if indexing == 'xy' and len(tensors) >= 2:
        swapped = (tensors[1], tensors[0]) + tensors[2:]
        grids = meshgrid(*swapped, indexing='ij')
        grids[0], grids[1] = grids[1], grids[0]
        return grids
    grids = []
    ndim = len(tensors)
    for i, t in enumerate(tensors):
        shape = [1] * ndim
        shape[i] = t.numel()
        reshaped = t.reshape(shape)
        expand_shape = [s.numel() for s in tensors]
        grids.append(reshaped.expand(*expand_shape))
    return grids


def atleast_1d(*tensors):
    if len(tensors) == 1:
        t = tensors[0]
        if t.ndim == 0:
            return t.reshape(1)
        return t
    result = []
    for t in tensors:
        if t.ndim == 0:
            result.append(t.reshape(1))
        else:
            result.append(t)
    return result


def atleast_2d(*tensors):
    if len(tensors) == 1:
        t = tensors[0]
        if t.ndim == 0:
            return t.reshape(1, 1)
        elif t.ndim == 1:
            return t.unsqueeze(0)
        return t
    result = []
    for t in tensors:
        if t.ndim == 0:
            result.append(t.reshape(1, 1))
        elif t.ndim == 1:
            result.append(t.unsqueeze(0))
        else:
            result.append(t)
    return result


def atleast_3d(*tensors):
    if len(tensors) == 1:
        t = tensors[0]
        if t.ndim == 0:
            return t.reshape(1, 1, 1)
        elif t.ndim == 1:
            return t.unsqueeze(0).unsqueeze(-1)
        elif t.ndim == 2:
            return t.unsqueeze(-1)
        return t
    result = []
    for t in tensors:
        if t.ndim == 0:
            result.append(t.reshape(1, 1, 1))
        elif t.ndim == 1:
            result.append(t.unsqueeze(0).unsqueeze(-1))
        elif t.ndim == 2:
            result.append(t.unsqueeze(-1))
        else:
            result.append(t)
    return result


def broadcast_shapes(*shapes):
    if not shapes:
        return ()
    max_ndim = _builtins.max(len(s) for s in shapes)
    result = [1] * max_ndim
    for shape in shapes:
        padded = [1] * (max_ndim - len(shape)) + list(shape)
        for i in _builtins.range(max_ndim):
            if padded[i] == 1:
                continue
            if result[i] == 1:
                result[i] = padded[i]
            elif result[i] != padded[i]:
                raise RuntimeError(
                    f"Shape mismatch: objects cannot be broadcast to a single shape. "
                    f"Mismatch at dim {i}: {result[i]} vs {padded[i]}"
                )
    return tuple(result)


def broadcast_tensors(*tensors):
    if len(tensors) == 1 and isinstance(tensors[0], (list, tuple)):
        tensors = tuple(tensors[0])
    shapes = [t.shape for t in tensors]
    target = broadcast_shapes(*shapes)
    return [t.expand(*target) for t in tensors]


def complex(real, imag):
    import numpy as _np
    r = real.numpy() if hasattr(real, 'numpy') else _np.asarray(real)
    i = imag.numpy() if hasattr(imag, 'numpy') else _np.asarray(imag)
    c = r.astype(_np.float64) + 1j * i.astype(_np.float64)
    from ._dtype import complex128 as _cdouble
    dev = _as_device(real.device if hasattr(real, 'device') else None)
    return dispatch("tensor", dev, c.tolist(), dtype=_cdouble)


def polar(abs, angle):
    import numpy as _np
    a = abs.numpy() if hasattr(abs, 'numpy') else _np.asarray(abs)
    ang = angle.numpy() if hasattr(angle, 'numpy') else _np.asarray(angle)
    c = a * _np.exp(1j * ang)
    from ._dtype import complex128 as _cdouble
    dev = _as_device(abs.device if hasattr(abs, 'device') else None)
    return dispatch("tensor", dev, c.tolist(), dtype=_cdouble)


# ---------------------------------------------------------------------------
# Category C2: Dispatch-based functions (need schema + kernel)
# ---------------------------------------------------------------------------

def diff(a, n=1, dim=-1, prepend=None, append=None):
    return dispatch("diff", a.device.type, a, n=n, dim=dim, prepend=prepend, append=append)


def bincount(a, weights=None, minlength=0):
    return dispatch("bincount", a.device.type, a, weights=weights, minlength=minlength)


def cdist(x1, x2, p=2.0):
    return dispatch("cdist", x1.device.type, x1, x2, p=p)


def aminmax(a, *, dim=None, keepdim=False):
    return dispatch("aminmax", a.device.type, a, dim=dim, keepdim=keepdim)


def quantile(a, q, dim=None, keepdim=False):
    return dispatch("quantile", a.device.type, a, q, dim=dim, keepdim=keepdim)


def nanquantile(a, q, dim=None, keepdim=False):
    return dispatch("nanquantile", a.device.type, a, q, dim=dim, keepdim=keepdim)


def nanmedian(a, dim=None, keepdim=False):
    return dispatch("nanmedian", a.device.type, a, dim=dim, keepdim=keepdim)


def histc(a, bins=100, min=0, max=0):
    return dispatch("histc", a.device.type, a, bins=bins, min=min, max=max)


def histogram(a, bins, *, range=None, weight=None, density=False):
    return dispatch("histogram", a.device.type, a, bins, range=range, weight=weight, density=density)


def bucketize(a, boundaries, *, out_int32=False, right=False):
    return dispatch("bucketize", a.device.type, a, boundaries, out_int32=out_int32, right=right)


def isneginf(a):
    return dispatch("isneginf", a.device.type, a)


def isposinf(a):
    return dispatch("isposinf", a.device.type, a)


def isreal(a):
    return dispatch("isreal", a.device.type, a)


def isin(elements, test_elements):
    return dispatch("isin", elements.device.type, elements, test_elements)


def heaviside(a, values):
    return dispatch("heaviside", a.device.type, a, values)


# ---------------------------------------------------------------------------
# P0 dtype utilities & query functions
# ---------------------------------------------------------------------------

def is_tensor(obj):
    from ._tensor import Tensor
    return isinstance(obj, Tensor)


def is_floating_point(input):
    return input.is_floating_point()


def is_complex(input):
    return input.is_complex()


def numel(input):
    return input.numel()


def square(a):
    return dispatch("square", a.device.type, a)
