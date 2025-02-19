"""creation ops"""
import numpy as np
import mindspore
try:
    from mindspore._c_expression import TensorPy as CTensor # pylint: disable=no-name-in-module
except:
    from mindspore._c_expression import Tensor as CTensor # pylint: disable=no-name-in-module


from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp.configs import use_pyboost, ON_ORANGE_PI
from ..utils import get_default_dtype

def as_strided(self, size, stride, storage_offset=None):
    if len(size) != len(stride):
        raise RuntimeError("mismatch in length of strides and shape.")
    index = np.arange(0, size[0]*stride[0], stride[0])
    for i in np.arange(1, len(size)):
        tmp = np.arange(0, size[i]*stride[i], stride[i])
        index = np.expand_dims(index, -1)
        index = index + tmp
    if storage_offset is not None:
        index = index + storage_offset

    if index.size == 0:
        input_indices = mindspore.numpy.empty(index.shape, dtype=mindspore.int32)
    else:
        input_indices = mindspore.tensor(index.astype(np.int32))
    out = ops.gather(self.reshape(-1), input_indices, 0)
    return out

# from_numpy
def from_numpy(ndarray):
    return mindspore.Tensor(ndarray)

# frombuffer

# zeros
_zeros = ops.Zeros()
has_zeros = hasattr(mindspore.mint, 'zeros')
def zeros(*size, dtype=None):
    if dtype is None:
        dtype = get_default_dtype()
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    if use_pyboost() and has_zeros:
        return mindspore.mint.zeros(size, dtype=dtype)
    size = tuple(size)
    return _zeros(size, dtype)

# zeros_like
has_zeros_like = hasattr(mindspore.mint, 'zeros_like')
def zeros_like(input, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    if use_pyboost() and has_zeros_like:
        return mindspore.mint.zeros_like(input, dtype=dtype)
    return ops.zeros_like(input, dtype=dtype)

# ones
_ones = ops.Ones()
has_ones = hasattr(mindspore.mint, 'ones')
def ones(*size, dtype=None):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    if dtype is None:
        dtype = get_default_dtype()
    if use_pyboost() and has_ones:
        return mindspore.mint.ones(size, dtype=dtype)
    return _ones(size, dtype)

# ones_like
has_ones_like = hasattr(mindspore.mint, 'ones_like')
def ones_like(input, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
    if use_pyboost() and has_ones_like:
        return mindspore.mint.ones_like(input, dtype=dtype)
    return ops.ones_like(input, dtype=dtype)

# arange
has_arange = hasattr(mindspore.mint, 'arange')
def arange(start=0, end=None, step=1, *, dtype=None):
    if ON_ORANGE_PI and dtype in (None, mindspore.int64):
        dtype = mindspore.int32
    if use_pyboost() and has_arange:
        return mindspore.mint.arange(start, end, step, dtype=dtype)
    return ops.arange(start, end, step, dtype=dtype)

# range
def range(start=0, end=None, step=1, dtype=None):
    if end is None:
        start, end = 0, start
    out = ops.range(start, end, step)
    if dtype is not None:
        out = out.to(dtype)
    return out

# linspace
has_linspace = hasattr(mindspore.mint, 'linspace')
def linspace(start, end, steps, *, dtype=None):
    if dtype is None:
        dtype = mindspore.float32
    if use_pyboost() and has_linspace:
        return mindspore.Tensor(np.linspace(start, end, steps)).to(dtype)
    return ops.linspace(start, end, steps).to(dtype)

# logspace
def logspace(start, end, steps, base=10.0, *, dtype=None):
    return ops.logspace(start, end, steps, base, dtype=dtype)

# eye
has_eye = hasattr(mindspore.mint, 'eye')
def eye(n, m=None, *, dtype=None):
    if use_pyboost() and has_eye:
        return mindspore.mint.eye(n, m, dtype)
    return ops.eye(n, m, dtype)

# empty
def empty(*size, dtype=None):
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    if dtype is None:
        dtype = get_default_dtype()
    out = CTensor(dtype=dtype, shape=size)
    return mindspore.Tensor(out)

# empty_like


# empty_strided


# full
has_full = hasattr(mindspore.mint, 'full')
def full(size, fill_value, *, dtype=None):
    if use_pyboost() and has_full:
        return mindspore.mint.ones(size, dtype=dtype) * fill_value
    return ops.full(size, fill_value, dtype=dtype)

# full_like
def full_like(input, fill_value, *, dtype=None):
    if dtype is None:
        dtype = input.dtype
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
