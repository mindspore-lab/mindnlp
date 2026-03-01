from ._dispatch.dispatcher import dispatch
from ._autograd.grad_mode import GradMode, no_grad
from ._device import device as Device, get_default_device
from ._dtype import to_numpy_dtype


def add(*args, **kwargs):
    alpha = kwargs.get("alpha", 1)
    if alpha != 1:
        raise NotImplementedError("alpha != 1 not supported yet")
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
    return dispatch("mean", a.device.type, a, dim=dim, keepdim=keepdim)


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
    dtype = kwargs.get("dtype")
    if dtype is not None:
        raise NotImplementedError("sum dtype not supported yet")
    kwargs.pop("device", None)
    return dispatch("sum", None, *args, **kwargs)


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


def zeros(shape, *, dtype=None, device=None, memory_format=None):
    dev = _as_device(device)
    return dispatch("zeros", dev, shape, dtype=dtype, memory_format=memory_format)


def zeros_like(input, *, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    return zeros(input.shape, dtype=dtype, device=device, memory_format=memory_format)


def ones(shape, *, dtype=None, device=None, memory_format=None):
    dev = _as_device(device)
    return dispatch("ones", dev, shape, dtype=dtype, memory_format=memory_format)


def empty(shape, *, dtype=None, device=None, memory_format=None):
    dev = _as_device(device)
    return dispatch("empty", dev, shape, dtype=dtype, memory_format=memory_format)


def randn(*shape, dtype=None, device=None, memory_format=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("randn", dev, shape, dtype=dtype, memory_format=memory_format)


def rand(*shape, dtype=None, device=None, memory_format=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    dev = _as_device(device)
    return dispatch("rand", dev, shape, dtype=dtype, memory_format=memory_format)


def arange(start, end=None, step=1, dtype=None, device=None):
    dev = _as_device(device)
    if end is None:
        start, end = 0, start
    return dispatch("arange", dev, start, end, step, dtype=dtype)


def linspace(start, end, steps, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("linspace", dev, start, end, steps, dtype=dtype)


def full(shape, fill_value, dtype=None, device=None):
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



def _as_device(dev):
    if dev is None:
        return get_default_device()
    if isinstance(dev, str):
        return Device(dev)
    return dev
