"""reduction op"""
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp.configs import USE_PYBOOST, DEVICE_TARGET

# argmax
def argmax(input, dim=None, keepdim=False):
    if USE_PYBOOST:
        return mindspore.mint.argmax(input, dim, keepdim)
    return ops.argmax(input, dim, keepdim)

# argmin
def argmin(input, dim=None, keepdim=False):
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
def all(input, dim=None, keepdim=False, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.all(input, dim, keepdim)
    return ops.all(input, dim, keepdim)

# any
def any(input, dim=None, keepdim=False):
    if USE_PYBOOST:
        return mindspore.mint.any(input, dim, keepdim)
    return ops.any(input, dim, keepdim)

# max
def max(input, dim=None, keepdim=False):
    if USE_PYBOOST:
        return mindspore.mint.max(input, dim, keepdim)
    out = ops.max(input, dim, keepdim)
    if dim is None:
        return out[0]
    return out

# min
def min(input, dim=None, keepdim=False):
    if USE_PYBOOST:
        return mindspore.mint.min(input, dim, keepdim)
    out = ops.min(input, dim, keepdim)
    if dim is None:
        return out[0]
    return out

# dist

# logsumexp
def logsumexp(input, dim, keepdim=False):
    return ops.logsumexp(input, dim, keepdim)

# mean
def mean(input, dim=None, keepdim=False, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.mean(input, dim, keepdim, dtype=dtype)
    out = ops.mean(input, dim, keepdim)
    if dtype is not None:
        out = out.astype(dtype)
    return out

# nanmean


# median
def median(input, dim=-1, keepdim=False):
    return ops.median(input, dim, keepdim)

# nanmedian
def nanmedian(input, dim=-1, keepdim=False):
    return ops.nanmedian(input, dim, keepdim)

# mode


# norm
def norm(input, p='fro', dim=None, keepdim=False, dtype=None):
    return ops.norm(input, p, dim, keepdim, dtype=dtype)

# nansum
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    return ops.nansum(input, dim, keepdim, dtype=dtype)

# prod
def prod(input, dim=None, keepdim=False, *, dtype=None):
    if USE_PYBOOST:
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
def sum(input, dim=None, keepdim=False, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.sum(input, dim, keepdim, dtype=dtype)
    return ops.sum(input, dim, keepdim, dtype=dtype)

# unique
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if USE_PYBOOST:
        return mindspore.mint.unique(input, sorted, return_inverse, return_counts, dim)
    return ops.unique(input)

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
def count_nonzero(input, dim=None):
    if dim is None:
        dim = ()
    return ops.count_nonzero(input, dim)
