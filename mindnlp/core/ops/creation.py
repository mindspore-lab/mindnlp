"""creation ops"""
import numbers
import numpy as np

try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_

from mindnlp import core
from mindnlp.core.executor import execute
from .._bind import get_default_dtype, get_device_in_context

def as_strided(self, size, stride, storage_offset=None):
    return execute('as_strided', self, size, stride, storage_offset)

# from_numpy
def from_numpy(ndarray):
    out = core.Tensor.from_numpy(ndarray)
    out._device = core.device('cpu')
    out._from_numpy = True
    return out

# frombuffer
def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False):
    arr = np.frombuffer(buffer=buffer, dtype=core.dtype2np[dtype], count=count, offset=offset)
    tensor = core.Tensor(arr)
    tensor.requires_grad_(requires_grad)
    return tensor


# zeros
def zeros(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, **kwargs):
    size = kwargs.pop('size', size)
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = get_device_in_context()
    
    if isinstance(device, str):
        device = core.device(device)
    if len(size) > 0 and isinstance(size[0], (tuple, list)):
        size = size[0]

    new_size = ()
    for s in size:
        if not isinstance(s, int):
            s = s.item()
        new_size += (s,)

    output = execute('zeros', new_size, dtype, device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# zeros_like
def zeros_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if device.type == 'cpu':
        return execute('zeros_like', input, device=device, requires_grad=requires_grad, user_created=True)
    return execute('zeros_like_ext', input, dtype,
                   device=device, requires_grad=requires_grad, user_created=True)

# ones
def ones(*size, out=None, dtype=None, layout=None, device=None, requires_grad=False, **kwargs):
    size = kwargs.pop('size', size)
    if dtype is None:
        dtype = get_default_dtype()
    if isinstance(dtype, type):
        dtype = core.py2dtype[dtype]
    if device is None:
        device = get_device_in_context()
    if isinstance(size[0], (tuple, list)):
        size = size[0]
    output = execute('ones', size, dtype,
                     device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# ones_like
def ones_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    if device is None:
        device = input.device
    if device.type == 'cpu':
        return execute('ones_like', input, device=device, requires_grad=requires_grad, user_created=True)
    return execute('ones_like_ext', input, dtype,
                   device=device, requires_grad=requires_grad, user_created=True)

# arange
def arange(start=0, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    if dtype is None:
        dtype = core.py2dtype[type(start)]
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)

    start = start.item() if isinstance(start, (core.Tensor, np.integer)) else start
    end = end.item() if isinstance(end, (core.Tensor, np.integer)) else end
    step = step.item() if isinstance(step, (core.Tensor, np.integer)) else step

    output = execute('arange', start, end, step, dtype, device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# range
def range(start=0, end=None, step=1, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if end is None:
        raise TypeError('range() missing 1 required positional arguments: "end"')
    if dtype is None:
        dtype = core.int64
    if device is None:
        device = get_device_in_context()
    output = execute('range', start, end + 1, step, 1000000,
                     device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# linspace
def linspace(start, end, steps, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)

    start = start.item() if isinstance(start, (core.Tensor, np.integer)) else start
    end = end.item() if isinstance(end, (core.Tensor, np.integer)) else end
    steps = steps.item() if isinstance(steps, (core.Tensor, np.integer)) else steps

    output = execute('lin_space_ext', start, end, steps, dtype,
                        device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# logspace

# eye
def eye(n, m=None, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    if device is None:
        device = get_device_in_context()
    if dtype is None:
        dtype = get_default_dtype()
    if m is None:
        m = n
    output = execute('eye', n, m, dtype,
                     device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# empty
def empty(*size, out=None, dtype=None, layout=None, device=None,
          requires_grad=False, pin_memory=False, memory_format=None, **kwargs):
    size = kwargs.pop('size', size)
    if dtype is None:
        dtype = get_default_dtype()
    if device is None:
        device = get_device_in_context()
    if isinstance(device, str):
        device = core.device(device)
    if len(size) > 0 and isinstance(size[0], (tuple, list)):
        size = size[0]

    if device.type == 'meta':
        output = core.tensor(Tensor_(shape=size, dtype=dtype), device=device)
    else:
        output = execute('empty', size, dtype, device=device)
    if out is None:
        return output
    out.data = output
    return out

# empty_like
def empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if device is None:
        device = input.device
    return empty(input.shape, dtype=input.dtype, layout=layout, device=device, requires_grad=requires_grad)

# empty_strided


# full
def full(size, fill_value, *, out=None, dtype=None, layout=None, device=None, requires_grad=False):
    # if dtype is None:
    #     dtype = get_default_dtype()
    if device is None:
        device = get_device_in_context()
    if device.type == 'cpu':
        output = execute('full', size, fill_value, device=device, requires_grad=requires_grad, user_created=True)
    else:
        if isinstance(fill_value, numbers.Number):
            output = execute('fill_scalar', size, fill_value, dtype,
                             device=device, requires_grad=requires_grad, user_created=True)
        else:
            output = execute('fill_tensor', size, fill_value, dtype,
                             device=device, requires_grad=requires_grad, user_created=True)
    if out is None:
        return output
    out.data = output
    return out

# full_like
def full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    if dtype is None:
        dtype = input.dtype
    return full(input.shape, fill_value, dtype=dtype, layout=layout, device=input.device, requires_grad=requires_grad)

# quantize_per_tensor


# quantize_per_channel


# dequantize


# complex


# polar
def polar(abs, angle, *, out=None):
    output = execute('polar', abs, angle)
    if out is None:
        return output
    out.data = output
    return out


# heaviside

__all__ = ['arange', 'as_strided', 'empty', 'empty_like',
           'eye', 'from_numpy', 'frombuffer', 'full', 'full_like',
           'linspace', 'ones', 'ones_like',
           'polar', 'range', 'zeros', 'zeros_like'
]
