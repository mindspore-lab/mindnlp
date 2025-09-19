"""comparison op"""
import numbers
from collections import namedtuple
import mindtorch
from mindtorch.executor import execute

sort_out = namedtuple('sort_out', ['values', 'indices'])
topk_out = namedtuple('topk_out', ['values', 'indices'])

# allclose
def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return isclose(input, other, rtol, atol, equal_nan).all().item()

# argsort
def argsort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim=dim, descending=descending, stable=stable)[1]

# eq
def eq(input, other):
    if not isinstance(other, numbers.Number) and other.device != input.device:
        other = other.to(input.device)
    return execute('eq', input, other)

# equal
def equal(input, other):
    return execute('equal', input, other)

# ge
def ge(input, other):
    return execute('greater_equal', input, other)

# gt
def gt(input, other):
    return execute('greater', input, other)

# greater
def greater(input, other):
    return gt(input, other)

# isclose
def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not isinstance(atol, numbers.Number):
        atol = atol.item()
    return execute('isclose', input, other, rtol, atol, equal_nan)
    

# isfinite
def isfinite(input):
    return execute('isfinite', input)

# isin
def in1d(ar1, ar2, invert=False):
    ar1 = mindtorch.unsqueeze(ar1.ravel(), -1)
    if not isinstance(ar2, numbers.Number):
        ar2 = ar2.ravel()
    included = mindtorch.eq(ar1, ar2)
    # ops.reduce_sum only supports float
    res = mindtorch.sum(included.to(mindtorch.float32), -1).to(mindtorch.bool_)
    if invert:
        res = mindtorch.logical_not(res)
    return res

def isin(elements, test_elements, invert=False):
    res = in1d(elements, test_elements, invert=invert)
    return mindtorch.reshape(res, elements.shape)

# isinf
def isinf(input):
    return execute('isinf', input)

# isposinf

# isneginf

# isnan
def isnan(input):
    return execute('not_equal', input, input)

# isreal

# kthvalue

# le
def le(input, other):
    return execute('less_equal', input, other)

# less_equal
def less_equal(input, other):
    return le(input, other)

# lt
def lt(input, other):
    return execute('less', input, other)

# less
def less(input, other):
    return lt(input, other)

# maximum
def maximum(input, other):
    if isinstance(other, mindtorch.Tensor) and other.device != input.device:
        other = other.to(input.device)
    return execute('maximum', input, other)

# minimum
def minimum(input, other):
    if other.device != input.device:
        other = other.to(input.device)
    return execute('minimum', input, other)

# fmax

# fmin

# ne
def ne(input, other):
    return execute('not_equal', input, other)

# not_equal
def not_equal(input, other):
    return ne(input, other)

# sort
def sort(input, *, dim=-1, descending=False, stable=False):
    out = execute('sort', input, dim, descending, stable)
        
    return sort_out(values=out[0], indices=out[1])

# topk
def topk(input, k, dim=-1, largest=True, sorted=True):
    out = execute('topk', input, k, dim, largest, sorted)
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
    'isinf',
    'isin',
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
    'ne',
    'not_equal',
    'sort',
    'topk',
    'msort',
]
