"""reduction op"""
from collections import namedtuple
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost, DEVICE_TARGET

from ._inner import call_ms_func

max_out = namedtuple('max_out', ['values', 'indices'])
min_out = namedtuple('min_out', ['values', 'indices'])

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
has_amax = hasattr(mindspore.mint, 'amax')
def amax(input, dim, keepdim=False):
    if use_pyboost() and has_amax:
        return mindspore.mint.amax(input, dim, keepdim)
    return ops.amax(input, dim, keepdim)

# amin
has_amin = hasattr(mindspore.mint, 'amin')
def amin(input, dim, keepdim=False):
    if use_pyboost() and has_amin:
        return mindspore.mint.amin(input, dim, keepdim)
    return ops.amin(input, dim, keepdim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
has_all = hasattr(mindspore.mint, 'all')
def all(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    axis = kwargs.get('axis', None)
    keepdims = kwargs.get('keepdims', None)
    if axis is not None:
        dim = axis
    if keepdims:
        keepdim = keepdims

    if use_pyboost() and has_all:
        return mindspore.mint.all(input, dim, keepdim).to(input.dtype)
    return ops.all(input, dim, keepdim).to(input.dtype)

# any
has_any = hasattr(mindspore.mint, 'any')
def any(input, dim=None, keepdim=False, *, out=None):
    if use_pyboost() and has_any:
        if dim is None:
            return call_ms_func(mindspore.mint.any, input, out=out)
        else:
            return call_ms_func(mindspore.mint.any, input, dim, keepdim, out=out)
    return call_ms_func(ops.any, input, dim, out=out)

# max
has_max = hasattr(mindspore.mint, 'max')
def max(*args, **kwargs):
    out = kwargs.pop('out', None)
    if 'dim' in kwargs and 'keepdim' not in kwargs:
        kwargs['keepdim'] = False
    out = mindspore.mint.max(*args, **kwargs)
    if isinstance(out, tuple):
        return max_out(values=out[0], indices=out[1])
    return out

# min
has_min = hasattr(mindspore.mint, 'min')
def min(*args, **kwargs):
    out = kwargs.pop('out', None)
    if 'dim' in kwargs and 'keepdim' not in kwargs:
        kwargs['keepdim'] = False
    out = mindspore.mint.min(*args, **kwargs)
    if isinstance(out, tuple):
        return min_out(values=out[0], indices=out[1])
    return out

# dist


# logsumexp
has_logsumexp = hasattr(mindspore.mint, 'logsumexp')
def logsumexp(input, dim, keepdim=False):
    if use_pyboost() and has_logsumexp:
        return mindspore.mint.logsumexp(input, dim, keepdim)
    return ops.logsumexp(input, dim, keepdim)

# mean
has_mean = hasattr(mindspore.mint, 'mean')
def mean(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    axis = kwargs.get('axis', None)
    if axis is not None:
        dim = axis
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
def norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    if use_pyboost() and has_norm:
        return call_ms_func(mindspore.mint.norm, input, p, dim, keepdim, out=out, dtype=dtype)
    return call_ms_func(ops.norm, input, p, dim, keepdim, out=out, dtype=dtype)

# nansum
has_nansum = hasattr(mindspore.mint, 'nansum')
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    if use_pyboost() and has_nansum:
        return mindspore.mint.nansum(input, dim, keepdim, dtype=dtype)
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
has_std = hasattr(mindspore.mint, 'std')
def std(input, dim=None, *, correction=1, keepdim=False, **kwargs):
    axis = kwargs.get('axis', None)
    if axis is not None:
        dim = axis
    if use_pyboost() and has_std:
        return mindspore.mint.std(input, dim=dim, correction=correction, keepdim=keepdim)
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
has_std_mean = hasattr(mindspore.mint, 'std_mean')
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_std_mean:
        return mindspore.mint.std_mean(input, dim=dim, correction=correction, keepdim=keepdim)
    return std(input, dim, correction=correction, keepdim=keepdim), \
        mean(input, dim, keepdim)

# sum
has_sum = hasattr(mindspore.mint, 'sum')
def sum(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    keepdims = kwargs.pop('keepdims', None)
    if keepdims is not None:
        keepdim = keepdims
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
has_unique_consecutive = hasattr(mindspore.mint, 'unique_consecutive')
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    if use_pyboost() and has_unique_consecutive:
        return mindspore.mint.unique_consecutive(input, return_inverse, return_counts, dim)
    return ops.unique_consecutive(input, return_inverse, return_counts, dim)

# var
has_var = hasattr(mindspore.mint, 'var')
def var(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_var:
        return mindspore.mint.var(input, dim=dim, correction=correction, keepdim=keepdim)
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2)

# var_mean
has_var_mean = hasattr(mindspore.mint, 'var_mean')
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_var_mean:
        return mindspore.mint.var_mean(input, dim=dim, correction=correction, keepdim=keepdim)
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

__all__ = ['all', 'amax', 'amin', 'aminmax', 'any', 'argmax', 'argmin', 'count_nonzero', 'logsumexp', 'max', 'mean', 'median', 'min', 'nanmedian', 'nanquantile', 'nansum', 'norm', 'prod', 'quantile', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive', 'var', 'var_mean']
 