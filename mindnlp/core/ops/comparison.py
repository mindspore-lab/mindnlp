"""comparison op"""
import numpy as np
import mindspore
from mindspore import ops
from mindnlp.configs import USE_PYBOOST

# allclose
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.allclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)

# argsort
def argsort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim=dim, descending=descending, stable=stable)[1]

# eq
def eq(input, other):
    if USE_PYBOOST:
        return mindspore.mint.eq(input, other)
    return ops.eq(input, other)

# equal
def equal(input, other):
    return eq(input, other)

# ge
def ge(input, other):
    return ops.ge(input, other)

# gt
def gt(input, other):
    if USE_PYBOOST:
        return mindspore.mint.gt(input, other)
    return ops.gt(input, other)

# greater
def greater(input, other):
    return gt(input, other)

# isclose
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if USE_PYBOOST:
        return mindspore.mint.isclose(input, other, rtol, atol, equal_nan)
    return mindspore.tensor(np.isclose(input.numpy(), other.numpy(), rtol, atol, equal_nan))

# isfinite
def isfinite(input):
    if USE_PYBOOST:
        return mindspore.mint.isfinite(input)
    return ops.isfinite(input)

# isin
def isin(elements, test_elements):
    elements = elements.asnumpy()
    test_elements = test_elements.asnumpy()
    mask = np.in1d(elements, test_elements).reshape(elements.shape)
    return mindspore.tensor(mask)

# isinf
def isinf(input):
    if input.dtype in (mindspore.int32, mindspore.int64):
        input = input.to(mindspore.float32)
    return ops.isinf(input)

# isposinf

# isneginf

# isnan
def isnan(input):
    if input.dtype in (mindspore.int32, mindspore.int64):
        input = input.to(mindspore.float32)
    return ops.isnan(input)

# isreal

# kthvalue

# le
def le(input, other):
    if USE_PYBOOST:
        return mindspore.mint.le(input, other)
    return ops.le(input, other)

# less_equal
def less_equal(input, other):
    return le(input, other)

# lt
def lt(input, other):
    if USE_PYBOOST:
        return mindspore.mint.lt(input, other)
    return ops.lt(input, other)

# less
def less(input, other):
    return lt(input, other)

# maximum
def maximum(input, other):
    if USE_PYBOOST:
        return mindspore.mint.maximum(input, other)
    return ops.maximum(input, other)

# minimum
def minimum(input, other):
    if USE_PYBOOST:
        return mindspore.mint.minimum(input, other)
    return ops.minimum(input, other)


# fmax
def fmax(input, other):
    return ops.fmax(input, other)

# fmin
def fmin(input, other):
    return ops.fmin(input, other)

# ne
def ne(input, other):
    if USE_PYBOOST:
        return mindspore.mint.ne(input, other)
    return ops.ne(input, other)

# not_equal
def not_equal(input, other):
    return ne(input, other)

# sort
def sort(input, *, dim=-1, descending=False, stable=False):
    if USE_PYBOOST:
        return mindspore.mint.sort(input, dim=dim, descending=descending, stable=stable)
    return ops.sort(input, dim, descending)

# topk
def topk(input, k, dim=-1, largest=True, sorted=True):
    if USE_PYBOOST:
        return mindspore.mint.topk(input, k, dim, largest, sorted)
    return ops.topk(input, k, dim, largest, sorted)

# msort
def msort(input):
    return sort(input, dim=0)
