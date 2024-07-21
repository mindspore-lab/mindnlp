"""creation ops"""
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp.configs import USE_PYBOOST

# from_numpy
def from_numpy(ndarray):
    return mindspore.Tensor(ndarray)

# frombuffer

# zeros
def zeros(*size, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.zeros(size, dtype=dtype)
    return ops.zeros(size, dtype)

# zeros_like
def zeros_like(input, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.zeros_like(input, dtype=dtype)
    return ops.zeros_like(input, dtype=dtype)

# ones
def ones(*size, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.ones(size, dtype=dtype)
    return ops.ones(size, dtype)

# ones_like
def ones_like(input, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.ones_like(input, dtype=dtype)
    return ops.ones_like(input, dtype=dtype)

# arange
def arange(start=0, end=None, step=1, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.arange(start, end, step, dtype=dtype)
    return ops.arange(start, end, step, dtype=dtype)

# range
def range(start=0, end=None, step=1, dtype=None):
    return ops.range(start, end, step).to(dtype)

# linspace
def linspace(start, end, steps, *, dtype=None):
    return ops.linspace(start, end, steps).to(dtype)

# logspace
def logspace(start, end, steps, base=10.0, *, dtype=None):
    return ops.logspace(start, end, steps, base, dtype=dtype)

# eye
def eye(n, m=None, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.eye(n, m, dtype)
    return ops.eye(n, n, dtype)

# empty


# empty_like


# empty_strided


# full
def full(size, fill_value, *, dtype=None):
    if USE_PYBOOST:
        return mindspore.mint.ones(size, dtype=dtype) * fill_value
    return ops.full(size, fill_value, dtype=dtype)

# full_like
def full_like(input, fill_value, *, dtype=None):
    return full(input.shape, fill_value, dtype=dtype)

# quantize_per_tensor


# quantize_per_channel


# dequantize


# complex
def complex(real, imag):
    _complex = _get_cache_prim(ops.Complex)()
    return _complex(real, imag)

# polar
def polar(abs, angle):
    return ops.polar(abs, angle)

# heaviside
def heaviside(input, values):
    return ops.heaviside(input, values)
