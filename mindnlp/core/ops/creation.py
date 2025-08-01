"""creation ops"""
import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16
import mindspore
try:
    from mindspore._c_expression import Tensor as CTensor # pylint: disable=no-name-in-module, import-error
except:
    from mindspore._c_expression import TensorPy as CTensor # pylint: disable=no-name-in-module, import-error

from mindspore._c_expression.typing import Type
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost, ON_ORANGE_PI
from .._bind import get_default_dtype, get_default_device
from .utils import py2dtype
from .other import finfo

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
def zeros(*size, dtype=None, device=None, requires_grad=False, **kwargs):
    if dtype is None:
        dtype = get_default_dtype()
    if not isinstance(dtype, Type):
        dtype = py2dtype[dtype]
    if len(size) == 0:
        size = kwargs.get('size', None)
        if size == () or size == []:
            size = ((),)
    if isinstance(size[0], (tuple, list)):
        size = size[0]

    new_size = ()
    for s in size:
        if not isinstance(s, int):
            s = s.item()
        new_size += (s,)
    if use_pyboost() and has_zeros:
        # if device == 'cpu':
        #     return mindspore.Tensor(np.zeros(size), dtype=dtype)
        return mindspore.mint.zeros(new_size, dtype=dtype)
    size = tuple(size)
    return _zeros(new_size, dtype)

# zeros_like
has_zeros_like = hasattr(mindspore.mint, 'zeros_like')
def zeros_like(input, *, dtype=None, memory_format=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    if use_pyboost() and has_zeros_like:
        return mindspore.mint.zeros_like(input, dtype=dtype)
    return ops.zeros_like(input, dtype=dtype)

# ones
_ones = ops.Ones()
has_ones = hasattr(mindspore.mint, 'ones')
def ones(*size, dtype=None, device=None, **kwargs):
    if len(size) == 0:
        size = kwargs.get('size', None)
        if size == () or size == []:
            size = ((),)

    if isinstance(size[0], (tuple, list)):
        size = size[0]
    if dtype is None:
        dtype = get_default_dtype()
    if not isinstance(dtype, Type):
        dtype = py2dtype[dtype]

    new_size = ()
    for s in size:
        if not isinstance(s, int):
            s = s.item()
        new_size += (s,)
    if use_pyboost() and has_ones:
        return mindspore.mint.ones(new_size, dtype=dtype)
    return _ones(new_size, dtype)

# ones_like
has_ones_like = hasattr(mindspore.mint, 'ones_like')
def ones_like(input, *, dtype=None, device=None, **kwargs):
    if dtype is None:
        dtype = input.dtype
    if use_pyboost() and has_ones_like:
        return mindspore.mint.ones_like(input, dtype=dtype)
    return ops.ones_like(input, dtype=dtype)

# arange
range_op = ops.Range()
has_arange = hasattr(mindspore.mint, 'arange')
def arange(start=0, end=None, step=1, *, dtype=None, device=None):
    if ON_ORANGE_PI and dtype in (None, mindspore.int64):
        dtype = mindspore.int32
    if use_pyboost() and has_arange:
        start = start.item() if isinstance(start, (mindspore.Tensor, np.integer)) else start
        end = end.item() if isinstance(end, (mindspore.Tensor, np.integer)) else end
        step = step.item() if isinstance(step, (mindspore.Tensor, np.integer)) else step
        return mindspore.mint.arange(start, end, step, dtype=dtype)

    if end is None:
        end = start
        start = 0
    start = mindspore.Tensor(start) if not isinstance(start, mindspore.Tensor) else start
    end = mindspore.Tensor(end) if not isinstance(start, mindspore.Tensor) else end
    step = mindspore.Tensor(step) if not isinstance(start, mindspore.Tensor) else step
    out = range_op(start, end, step)
    if dtype:
        out = out.to(dtype)
    return out

# range
def range(start=0, end=None, step=1, dtype=None):
    if end is None:
        start, end = 0, start
    out = ops.range(start, end+1, step)
    if dtype is not None:
        out = out.to(dtype)
    return out

# linspace
has_linspace = hasattr(mindspore.mint, 'linspace')
def linspace(start, end, steps, *, dtype=None, **kwargs):
    if dtype is None:
        dtype = mindspore.float32
    start = start.item() if isinstance(start, mindspore.Tensor) else start
    end = end.item() if isinstance(end, mindspore.Tensor) else end
    steps = steps.item() if isinstance(steps, mindspore.Tensor) else steps
    if use_pyboost() and has_linspace:
        return mindspore.mint.linspace(start, end, steps, dtype=dtype)
    return ops.linspace(start, end, steps).to(dtype)

# logspace
has_logspace = hasattr(mindspore.mint, 'logspace')
def logspace(start, end, steps, base=10.0, *, dtype=None, **kwargs):
    if dtype is None:
        dtype = get_default_dtype()
    if use_pyboost() and has_logspace:
        return mindspore.mint.logspace(start, end, steps, base, dtype=dtype)
    return ops.logspace(float(start), float(end), steps, int(base), dtype=dtype)

# eye
has_eye = hasattr(mindspore.mint, 'eye')
def eye(n, m=None, *, dtype=None, **kwargs):
    if use_pyboost() and has_eye:
        return mindspore.mint.eye(n, m, dtype)
    return ops.eye(n, m, dtype)

# empty
has_empty = hasattr(mindspore.mint, 'empty')
def empty(*size, dtype=None, device=None, requires_grad=False, pin_memory=False, **kwargs):
    size = size or kwargs.get('size', None)
    if device is None:
        device= get_default_device()

    if len(size) > 0 and isinstance(size[0], (tuple, list)):
        size = size[0]

    if dtype is None:
        dtype = get_default_dtype()

    if device:
        if not isinstance(device, str) and hasattr(device, "type"):
            device = device.type
        if device.lower() == 'cpu':
            device = 'CPU'
        elif device.lower() == 'npu':
            device = 'Ascend'
        elif device.lower() == 'cuda':
            device = 'GPU'
        else:
            device = 'meta'

    # To avoid the problem in irecv and recv of using empty.
    if device != 'meta':
        out = mindspore.mint.empty(size, dtype=dtype, device=device)
    else:
        out = CTensor(dtype=dtype, shape=size)
        out = mindspore.Tensor(out)
    if requires_grad:
        out.requires_grad = True
    return out

# empty_like
has_empty_like = hasattr(mindspore.mint, 'empty_like')
def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return mindspore.mint.empty_like(input, dtype=dtype, device=device)

# empty_strided


# full
has_full = hasattr(mindspore.mint, 'full')
def full(size, fill_value, *, dtype=None, device=None, **kwargs):
    new_size = ()
    for s in size:
        if isinstance(s, mindspore.Tensor):
            s = s.item()
        new_size += (s,)
    if isinstance(fill_value, np.generic):
        fill_value = fill_value.item()
    if use_pyboost() and has_full:
        return mindspore.mint.full(new_size, fill_value, dtype=dtype)
    return ops.full(new_size, fill_value, dtype=dtype)

# full_like
has_full_like = hasattr(mindspore.mint, 'full_like')
def full_like(input, fill_value, *, dtype=None, device=None):
    if use_pyboost() and has_full_like:
        return mindspore.mint.full_like(input, fill_value, dtype=dtype)
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
has_polar = hasattr(mindspore.mint, 'polar')
def polar(abs, angle):
    if use_pyboost() and has_polar:
        return mindspore.mint.polar(abs, angle)
    return ops.polar(abs, angle)

# heaviside
def heaviside(input, values):
    return ops.heaviside(input, values)

_TypeDict = {
    mindspore.float16: np.float16,
    mindspore.float32: np.float32,
    mindspore.float64: np.float64,
    mindspore.bfloat16: np_bfloat16,
    mindspore.int8: np.int8,
    mindspore.int16: np.int16,
    mindspore.int32: np.int32,
    mindspore.int64: np.int64,
    mindspore.uint8: np.uint8,
    mindspore.bool_: np.bool_,
    mindspore.complex64: np.complex64,
    mindspore.complex128: np.complex128,
}


def frombuffer(buffer, *, dtype=None, count=-1, offset=0, requires_grad=False):
    np_dtype = _TypeDict[dtype]
    output = np.frombuffer(buffer=buffer, dtype=np_dtype, count=count, offset=offset)
    if dtype == mindspore.bfloat16:
        return mindspore.Tensor(output.astype(np.float32), dtype=dtype)
    return mindspore.Tensor(output, dtype=dtype)


def scalar_tensor(value, dtype, device=None):
    if value == float("-inf"):
        value = finfo(dtype).min
    if value == float("inf"):
        value = finfo(dtype).max
    return mindspore.Tensor(value, dtype=dtype)

__all__ = ['arange', 'as_strided', 'complex', 'empty', 'empty_like',
           'eye', 'from_numpy', 'full', 'full_like', 'frombuffer',
           'heaviside', 'linspace', 'logspace', 'ones', 'ones_like',
           'polar', 'range', 'zeros', 'zeros_like', 'scalar_tensor'
]