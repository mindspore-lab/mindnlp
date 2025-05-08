"""comparison op"""
import numpy as np
import mindspore
from mindspore import ops
from mindnlp.configs import use_pyboost

# allclose
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.allclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)

# argsort
def argsort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim=dim, descending=descending, stable=stable)[1]

# eq
has_eq = hasattr(mindspore.mint, 'eq')
def eq(input, other):
    if use_pyboost() and has_eq:
        return mindspore.mint.eq(input, other)
    return ops.eq(input, other)

# equal
def equal(input, other):
    return eq(input, other)

# ge
def ge(input, other):
    return ops.ge(input, other)

# gt
has_gt = hasattr(mindspore.mint, 'gt')
def gt(input, other):
    if use_pyboost() and has_gt:
        return mindspore.mint.gt(input, other)
    return ops.gt(input, other)

# greater
has_greater = hasattr(mindspore.mint, 'greater')
def greater(input, other):
    return gt(input, other)

# isclose
has_isclose = hasattr(mindspore.mint, 'isclose')
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if use_pyboost() and has_isclose:
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
    test_elements = test_elements.ravel()
    included = ops.equal(elements, test_elements)
    # F.reduce_sum only supports float
    res = ops.sum(included.int(), -1).astype(mindspore.bool_)

    return res

# isinf
def isinf(input):
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
def le(input, other):
    if use_pyboost() and has_le:
        return mindspore.mint.le(input, other)
    return ops.le(input, other)

# less_equal
def less_equal(input, other):
    return le(input, other)

# lt
has_lt = hasattr(mindspore.mint, 'lt')
def lt(input, other):
    if use_pyboost() and has_lt:
        return mindspore.mint.lt(input, other)
    return ops.lt(input, other)

# less
def less(input, other):
    return lt(input, other)

# maximum
has_maximum = hasattr(mindspore.mint, 'maximum')
def maximum(input, other):
    if use_pyboost() and has_maximum:
        return mindspore.mint.maximum(input, other)
    return ops.maximum(input, other)

# minimum
has_minimum = hasattr(mindspore.mint, 'minimum')
def minimum(input, other):
    if use_pyboost() and has_minimum:
        return mindspore.mint.minimum(input, other)
    return ops.minimum(input, other)


# fmax
def fmax(input, other):
    return ops.fmax(input, other)

# fmin
def fmin(input, other):
    return ops.fmin(input, other)

# ne
has_ne = hasattr(mindspore.mint, 'ne')
def ne(input, other):
    if use_pyboost() and has_ne:
        return mindspore.mint.ne(input, other)
    return ops.ne(input, other)

# not_equal
def not_equal(input, other):
    return ne(input, other)

# sort
has_sort = hasattr(mindspore.mint, 'sort')
def sort(input, *, dim=-1, descending=False, stable=False):
    if use_pyboost() and has_sort:
        return mindspore.mint.sort(input, dim=dim, descending=descending, stable=stable)
    return ops.sort(input, dim, descending)

# topk
has_topk = hasattr(mindspore.mint, 'topk')
def topk(input, k, dim=-1, largest=True, sorted=True):
    if use_pyboost() and has_topk:
        return mindspore.mint.topk(input, k, dim, largest, sorted)
    return ops.topk(input, k, dim, largest, sorted)

# msort
def msort(input):
    return sort(input, dim=0)
