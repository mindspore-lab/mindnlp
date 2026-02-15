from ._dtype import float32
from ._functional import tensor as tensor_dispatch
from ._functional import zeros as zeros_dispatch
from ._functional import ones as ones_dispatch
from ._functional import empty as empty_dispatch


def tensor(data, dtype=float32, device=None, requires_grad=False):
    return tensor_dispatch(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(shape, dtype=float32, device=None):
    return zeros_dispatch(shape, dtype=dtype, device=device)


def ones(shape, dtype=float32, device=None):
    return ones_dispatch(shape, dtype=dtype, device=device)


def empty(shape, dtype=float32, device=None):
    return empty_dispatch(shape, dtype=dtype, device=device)
