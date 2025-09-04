"""comparison op"""
import numbers
from collections import namedtuple
from mindnlp import core
from mindnlp.core.executor import execute

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
    return execute('equal', input, other)

# equal
def equal(input, other):
    if input.device.type == 'npu':
        return execute('equal_ext', input, other)
    # if input.shape != other.shape:
    #     return False
    out = eq(input, other)
    return out.all()

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
    return execute('isclose', input, other, rtol, atol, equal_nan)

# isfinite
def isfinite(input):
    return execute('isfinite', input)

# isin
def in1d(ar1, ar2, invert=False):
    ar1 = core.unsqueeze(ar1.ravel(), -1)
    ar2 = ar2.ravel()
    included = core.eq(ar1, ar2)
    # ops.reduce_sum only supports float
    res = core.sum(included.to(core.float32), -1).to(core.bool_)
    if invert:
        res = core.logical_not(res)
    return res

def isin(elements, test_elements, invert=False):
    if elements.device.type != 'cpu':
        res = in1d(elements, test_elements, invert=invert)
        return core.reshape(res, elements.shape)

    return execute('isin', elements, test_elements)

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
    if isinstance(other, core.Tensor) and other.device != input.device:
        other = other.to(input.device)
    return execute('maximum', input, other)

# minimum
def minimum(input, other):
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
    out = execute('sort_ext', input, dim, descending, stable)
    return sort_out(values=out[0], indices=out[1])

# topk
def topk(input, k, dim=-1, largest=True, sorted=True):
    if input.device.type == 'npu':
        out = execute('topk_ext', input, k, dim, largest, sorted)
    else:
        if not largest:
            input = -input
        if dim is None or dim == input.ndim - 1:
            if not largest:
                res = execute('topk', input, k, sorted)
                values, indices = -res[0], res[1]
                return topk_out(values=values, indices=indices)
            out =  execute('topk', input, k, sorted)
            return topk_out(values=out[0], indices=out[1])
        input = input.swapaxes(dim, input.ndim - 1)
        output = execute('topk', input, k, sorted)
        values = output[0].swapaxes(dim, input.ndim - 1)
        indices = output[1].swapaxes(dim, input.ndim - 1)
        if not largest:
            res = (-values, indices)
        else:
            res = (values, indices)
        out = res
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
