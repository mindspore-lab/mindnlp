"""comparison op"""
from collections import namedtuple
import numpy as np
import mindspore
from mindspore import ops
from ..configs import use_pyboost, ON_ORANGE_PI

from ._inner import call_ms_func

sort_out = namedtuple('sort_out', ['values', 'indices'])
topk_out = namedtuple('topk_out', ['values', 'indices'])
# allclose
has_allclose = hasattr(mindspore.mint, 'allclose')
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    rtol = rtol.item() if isinstance(rtol, mindspore.Tensor) else rtol
    atol = atol.item() if isinstance(atol, mindspore.Tensor) else atol
    if use_pyboost() and has_allclose and not ON_ORANGE_PI:
        return mindspore.mint.allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return np.allclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)

# argsort
has_argsort = hasattr(mindspore.mint, 'argsort')
def argsort(input, dim=-1, descending=False, stable=False):
    if use_pyboost() and has_argsort:
        return mindspore.mint.argsort(input, dim=dim, descending=descending)
    return sort(input, dim=dim, descending=descending, stable=stable)[1]

# eq
has_eq = hasattr(mindspore.mint, 'eq')
def eq(input, other, *, out=None):
    if use_pyboost() and has_eq:
        return call_ms_func(mindspore.mint.eq, input, other, out=out)
    if isinstance(other, str):
        return False
    return call_ms_func(ops.eq, input, other, out=out)

# equal
has_equal = hasattr(mindspore.mint, 'equal')
def equal(input, other):
    if use_pyboost() and has_equal and not ON_ORANGE_PI:
        return mindspore.mint.equal(input, other)
    if input.shape != other.shape:
        return False
    out = eq(input, other)
    return out.all()

# ge
def ge(input, other):
    return ops.ge(input, other)

# gt
has_gt = hasattr(mindspore.mint, 'gt')
def gt(input, other, *, out=None):
    if use_pyboost() and has_gt:
        return call_ms_func(mindspore.mint.gt, input, other, out=out)
    return call_ms_func(ops.gt, input, other, out=out)


# greater
has_greater = hasattr(mindspore.mint, 'greater')
def greater(input, other, *, out=None):
    return gt(input, other, out=out)

# isclose
has_isclose = hasattr(mindspore.mint, 'isclose')
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if use_pyboost() and has_isclose and not ON_ORANGE_PI:
        return mindspore.mint.isclose(input, other, rtol, atol, equal_nan)
    return mindspore.tensor(np.isclose(input.numpy(), other.numpy(), rtol, atol, equal_nan))

# isfinite
has_isfinite = hasattr(mindspore.mint, 'isfinite')
def isfinite(input):
    if use_pyboost() and has_isfinite:
        return mindspore.mint.isfinite(input)
    return ops.isfinite(input)

# isin
def isin(elements, test_elements):
    elements = elements.ravel().expand_dims(-1)
    if isinstance(test_elements, mindspore.Tensor):
        test_elements = test_elements.ravel()
    included = ops.equal(elements, test_elements)
    # F.reduce_sum only supports float
    res = ops.sum(included.int(), -1).astype(mindspore.bool_)

    return res

# isinf
has_isinf = hasattr(mindspore.mint, 'isinf')
def isinf(input):
    if use_pyboost() and has_isinf:
        return mindspore.mint.isinf(input)
    if input.dtype in (mindspore.int32, mindspore.int64):
        input = input.to(mindspore.float32)
    return ops.isinf(input)

# isposinf

# isneginf

# isnan
has_isnan = hasattr(mindspore.mint, 'isnan')
def isnan(input):
    if use_pyboost() and has_isnan:
        return mindspore.mint.isnan(input)
    if input.dtype in (mindspore.int32, mindspore.int64):
        input = input.to(mindspore.float32)
    return ops.isnan(input)

# isreal

# kthvalue

# le
has_le = hasattr(mindspore.mint, 'le')
def le(input, other, *, out=None):
    if use_pyboost() and has_le:
        return call_ms_func(mindspore.mint.le, input, other, out=out)
    return call_ms_func(ops.le, input, other, out=out)

# less_equal
has_less_equal = hasattr(mindspore.mint, 'less_equal')
def less_equal(input, other, *, out=None):
    return le(input, other, out=out)

# lt
has_lt = hasattr(mindspore.mint, 'lt')
def lt(input, other, *, out=None):
    if use_pyboost() and has_lt:
        return call_ms_func(mindspore.mint.lt, input, other, out=out)
    return call_ms_func(ops.lt, input, other, out=out)

# less
has_less = hasattr(mindspore.mint, 'less')
def less(input, other, *, out=None):
    return lt(input, other, out=out)

# maximum
has_maximum = hasattr(mindspore.mint, 'maximum')
def maximum(input, other, *, out=None):
    if use_pyboost() and has_maximum:
        return call_ms_func(mindspore.mint.maximum, input, other, out=out)
    return call_ms_func(ops.maximum, input, other, out=out)

# minimum
has_minimum = hasattr(mindspore.mint, 'minimum')
def minimum(input, other, *, out=None):
    if use_pyboost() and has_minimum:
        return call_ms_func(mindspore.mint.minimum, input, other, out=out)
    return call_ms_func(ops.minimum, input, other, out=out)


# fmax
def fmax(input, other):
    return ops.fmax(input, other)

# fmin
def fmin(input, other):
    return ops.fmin(input, other)

# ne
has_ne = hasattr(mindspore.mint, 'ne')
def ne(input, other, *, out=None):
    if use_pyboost() and has_ne:
        return call_ms_func(mindspore.mint.ne, input, other, out=out)
    return call_ms_func(ops.ne, input, other, out=out)

# not_equal
has_not_equal = hasattr(mindspore.mint, 'not_equal')
def not_equal(input, other):
    return ne(input, other)

# sort
has_sort = hasattr(mindspore.mint, 'sort')
def sort(input, *, dim=-1, descending=False, stable=False):
    if use_pyboost() and has_sort and not ON_ORANGE_PI:
        out = mindspore.mint.sort(input, dim=dim, descending=descending, stable=stable)
    else:
        out = ops.sort(input, dim, descending)
    return sort_out(values=out[0], indices=out[1])

# topk
has_topk = hasattr(mindspore.mint, 'topk')
def topk(input, k, dim=-1, largest=True, sorted=True):
    if use_pyboost() and has_topk:
        out = mindspore.mint.topk(input, int(k), dim, largest, sorted)
    else:
        out = ops.topk(input, k, dim, largest, sorted)
    return topk_out(values=out[0], indices=out[1])

# msort
def msort(input):
    return sort(input, dim=0)

__all__ = [
    'allclose',
    'argsort',
    'eq',
    'equal',
    'ge',
    'gt',
    'greater',
    'isclose',
    'isfinite',
    'isin',
    'isinf',
    # isposinf,
    # isneginf,
    'isnan',
    # isreal,
    # kthvalue,
    'le',
    'less_equal',
    'lt',
    'less',
    'maximum',
    'minimum',
    'fmax',
    'fmin',
    'ne',
    'not_equal',
    'sort',
    'topk',
    'msort',
]
