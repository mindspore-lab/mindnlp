"""reduction op"""
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp.configs import use_pyboost, DEVICE_TARGET

# argmax
has_argmax = hasattr(mindspore.mint, 'argmax')
def argmax(input, dim=None, keepdim=False):
    if use_pyboost() and has_argmax:
        return mindspore.mint.argmax(input, dim, keepdim)
    return ops.argmax(input, dim, keepdim)

# argmin
has_argmin = hasattr(mindspore.mint, 'argmin')
def argmin(input, dim=None, keepdim=False):
    if use_pyboost() and has_argmin:
        return mindspore.mint.argmin(input, dim, keepdim)
    return ops.argmin(input, dim, keepdim)

# amax
def amax(input, dim, keepdim=False):
    return ops.amax(input, dim, keepdim)

# amin
def amin(input, dim, keepdim=False):
    return ops.amin(input, dim, keepdim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
has_all = hasattr(mindspore.mint, 'all')
def all(input, dim=None, keepdim=False, *, dtype=None):
    if use_pyboost() and has_all:
        return mindspore.mint.all(input, dim, keepdim)
    return ops.all(input, dim, keepdim)

# any
has_any = hasattr(mindspore.mint, 'any')
def any(input, dim=None, keepdim=False):
    if use_pyboost() and has_any:
        return mindspore.mint.any(input, dim, keepdim)
    return ops.any(input, dim)

# max
has_max = hasattr(mindspore.mint, 'max')
def max(*args, **kwargs):
    if use_pyboost() and has_max:
        return mindspore.mint.max(*args, **kwargs)

    input = kwargs.get('input', None) or args[0]
    dim = kwargs.get('dim', None) or args[1]
    keepdim = kwargs.get('keepdim', False) or args[2]
    out = ops.max(input, dim, keepdim)
    if dim is None:
        return out[0]
    return out

# min
has_min = hasattr(mindspore.mint, 'min')
def min(*args, **kwargs):
    if use_pyboost() and has_min:
        return mindspore.mint.min(*args, **kwargs)

    input = kwargs.get('input', None) or args[0]
    dim = kwargs.get('dim', None) or args[1]
    keepdim = kwargs.get('keepdim', False) or args[2]
    out = ops.min(input, dim, keepdim)
    if dim is None:
        return out[0]
    return out


# dist

# logsumexp
def logsumexp(input, dim, keepdim=False):
    return ops.logsumexp(input, dim, keepdim)

# mean
has_mean = hasattr(mindspore.mint, 'mean')
def mean(input, dim=None, keepdim=False, *, dtype=None):
    if use_pyboost() and has_mean:
        return mindspore.mint.mean(input, dim, keepdim, dtype=dtype)
    out = ops.mean(input, dim, keepdim)
    if dtype is not None:
        out = out.astype(dtype)
    return out

# nanmean


# median
has_median = hasattr(mindspore.mint, 'median')
def median(input, dim=-1, keepdim=False):
    if use_pyboost() and has_median:
        return mindspore.mint.median(input, dim, keepdim)
    return ops.median(input, dim, keepdim)

# nanmedian
def nanmedian(input, dim=-1, keepdim=False):
    return ops.nanmedian(input, dim, keepdim)

# mode


# norm
has_norm = hasattr(mindspore.mint, 'norm')
def norm(input, p='fro', dim=None, keepdim=False, dtype=None):
    if use_pyboost() and has_norm:
        return mindspore.mint.norm(input, p, dim, keepdim, dtype=dtype)
    return ops.norm(input, p, dim, keepdim, dtype=dtype)

# nansum
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    return ops.nansum(input, dim, keepdim, dtype=dtype)

# prod
has_prod = hasattr(mindspore.mint, 'prod')
def prod(input, dim=None, keepdim=False, *, dtype=None):
    if use_pyboost() and has_prod:
        return mindspore.mint.prod(input, dim, keepdim, dtype=dtype)
    return ops.prod(input, dim, keepdim).to(dtype)

# quantile
def quantile(input, q, dim=None, keepdim=False, *, interpolation='linear'):
    return ops.quantile(input, q, dim, keepdim)

# nanquantile
def nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear'):
    return ops.quantile(input, q, dim, keepdim)

# std
def std(input, dim=None, *, correction=1, keepdim=False):
    if DEVICE_TARGET == 'GPU':
        unbiased = bool(correction)
        if dim is None:
            dim = ()
        if isinstance(dim, int):
            dim = (dim,)
        _std = _get_cache_prim(ops.ReduceStd)(dim, unbiased, keepdim)
        _std.set_device('CPU')
        return _std(input)[0]
    return ops.std(input, dim, correction, keepdim)

# std_mean
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    return std(input, dim, correction=correction, keepdim=keepdim), \
        mean(input, dim, keepdim)

# sum
has_sum = hasattr(mindspore.mint, 'sum')
def sum(input, dim=None, keepdim=False, *, dtype=None):
    if 0 in input.shape:
        return mindspore.tensor(0, dtype=dtype)
    if use_pyboost() and has_sum:
        return mindspore.mint.sum(input, dim, keepdim, dtype=dtype)
    return ops.sum(input, dim, keepdim, dtype=dtype)

# unique
has_unique = hasattr(mindspore.mint, 'unique')
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if use_pyboost() and has_unique:
        return mindspore.mint.unique(input, sorted, return_inverse, return_counts, dim)

    out, inverse = ops.unique(input)
    outs = (out,)
    if return_inverse:
        outs += (inverse,)
    if return_counts:
        counts = (out == input).sum(0, keepdims=True)
        outs += (counts,)
    return outs if len(outs) > 1 else outs[0]

# unique_consecutive
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    return ops.unique_consecutive(input, return_inverse, return_counts, dim)

# var
def var(input, dim=None, *, correction=1, keepdim=False):
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2)

# var_mean
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2), \
        mean(input, dim, keepdim)

# count_nonzero
has_count_nonzero = hasattr(mindspore.mint, 'count_nonzero')
def count_nonzero(input, dim=None):
    if use_pyboost() and has_count_nonzero:
        return mindspore.mint.count_nonzero(input, dim)
    if dim is None:
        dim = ()
    return ops.count_nonzero(input, dim)
