from ._dtype import float32
from ._functional import tensor as tensor_dispatch
from ._functional import zeros as zeros_dispatch
from ._functional import ones as ones_dispatch
from ._functional import empty as empty_dispatch
from ._functional import arange as arange_dispatch
from ._functional import linspace as linspace_dispatch
from ._functional import full as full_dispatch


def tensor(data, dtype=float32, device=None, requires_grad=False):
    return tensor_dispatch(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(shape, dtype=float32, device=None):
    return zeros_dispatch(shape, dtype=dtype, device=device)


def ones(shape, dtype=float32, device=None):
    return ones_dispatch(shape, dtype=dtype, device=device)


def empty(shape, dtype=float32, device=None):
    return empty_dispatch(shape, dtype=dtype, device=device)


def arange(start, end=None, step=1, dtype=float32, device=None):
    return arange_dispatch(start, end=end, step=step, dtype=dtype, device=device)


def linspace(start, end, steps, dtype=float32, device=None):
    return linspace_dispatch(start, end, steps, dtype=dtype, device=device)


def full(shape, fill_value, dtype=float32, device=None):
    return full_dispatch(shape, fill_value, dtype=dtype, device=device)
