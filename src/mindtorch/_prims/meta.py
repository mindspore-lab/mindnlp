try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_

import math
import numpy as np
import mindtorch

__all__ = []

def arange(start, end, step, dtype):
    out = Tensor_(shape=(math.ceil((end - start) / step), ), dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('arange')

def broadcast_to(input, shape):
    out_shape = ()
    input_shape = input.shape
    if len(input_shape) != shape:
        input_shape = (1,) + input_shape
    for idx, s in enumerate(shape):
        if s == -1:
            s = input_shape[idx]
        out_shape += (s,)

    out = Tensor_(shape=out_shape, dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('broadcast_to')

def zeros(size, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('zeros')

def ones(size, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('ones')

def inplace_uniform(input, *args):
    return input

__all__.append('inplace_uniform')

def inplace_fill_scalar(input, value):
    return input

__all__.append('inplace_fill_scalar')

def inplace_normal(input, *args):
    return input

__all__.append('inplace_normal')

def getitem(input, slice):
    out = input.asnumpy()[slice]
    out = Tensor_(shape=out.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('getitem')

def sub_ext(input, other, alpha):
    if isinstance(input, mindtorch.Tensor):
        return input
    return other

__all__.append('sub_ext')

def pad_v3(input, pad, mode, value):
    out = np.pad(input.asnumpy(), pad, mode, constant_values=value)
    out = Tensor_(shape=out.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('pad_v3')

def abs(input):
    return input

__all__.append('abs')

def cast(input, dtype):
    out = Tensor_(shape=input.shape, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('cast')

def index_select(input, dim, index):
    out = np.take(input.asnumpy(), index.asnumpy(), dim)
    out = Tensor_(shape=out.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('index_select')

def identity(input):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('identity')

def contiguous(input):
    return input

__all__.append('contiguous')

def inplace_copy(input, other):
    return input

__all__.append('inplace_copy')

def div(input, other):
    if isinstance(input, mindtorch.Tensor):
        shape = input.shape
        dtype = input.dtype
    else:
        shape = other.shape
        dtype = other.dtype
    out = Tensor_(shape=shape, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('div')

def pow_scalar_tensor(input, other):
    out = Tensor_(shape=other.shape, dtype=other.dtype)
    return mindtorch.Tensor(out)

__all__.append('pow_scalar_tensor')

def concat(tensors, dim):
    shape = list(tensors[0].shape)
    shape[dim] = sum([t.shape[dim] for t in tensors])
    out = Tensor_(shape=tuple(shape), dtype=tensors[0].dtype)
    return mindtorch.Tensor(out)

__all__.append('concat')

def tril_ext(input, k):
    return input

__all__.append('tril_ext')

def reshape(input, shape):
    out = Tensor_(shape=tuple(shape), dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('reshape')

def linalg_vector_norm(input, p, dim, keepdim, dtype):
    input_shape = list(input.shape)
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        input_shape[d] = 1 if keepdim else 0
    
    new_shape = []
    for s in input_shape:
        if s != 0:
            new_shape.append(s)
    if dtype is None:
        dtype = input.dtype
    out = Tensor_(shape=tuple(new_shape), dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('linalg_vector_norm')

def erfinv(input):
    return input
__all__.append('erfinv')


def stop_gradient(input):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('stop_gradient')

def log(input):
    return input
__all__.append('log')

def mul(input, other):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)
__all__.append('mul')

def randn(size, seed, offset, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('randn')

def zeros_like_ext(input, *args, **kwargs):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return mindtorch.Tensor(out)
__all__.append('zeros_like_ext')

def inplace_add_ext(input, other, alpha):
    return input
__all__.append('inplace_add_ext')

def clamp_scalar(input, *args):
    return input
__all__.append('clamp_scalar')

def expand_dims_view(input, dim):
    input_shape = list(input.shape)
    input_shape.insert(dim, 1)

    out = Tensor_(shape=tuple(input_shape), dtype=input.dtype)
    return mindtorch.Tensor(out)
__all__.append('expand_dims_view')

def floor_div(input, other):
    return input
__all__.append('floor_div')

def sin(input):
    return input

__all__.append('sin')

def cos(input):
    return input

__all__.append('cos')

def triu(input, diagonal):
    return input

__all__.append('triu')

def fill_scalar(size, fill_value, dtype):
    if dtype is None:
        dtype = mindtorch.get_default_dtype()
    out = Tensor_(shape=size, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('fill_scalar')

def sqrt(input):
    return input

__all__.append('sqrt')

def normal_float_float(mean, std, size, seed, offset):
    out = Tensor_(shape=size, dtype=mindtorch.float32)
    return mindtorch.Tensor(out)


__all__.append('normal_float_float')

def stack(tensors, dim):
    x_shape = list(tensors[0].shape)
    x_shape.insert(dim, len(tensors))
    out = Tensor_(shape=tuple(x_shape), dtype=tensors[0].dtype)
    return mindtorch.Tensor(out)

__all__.append('stack')

def argmax_with_value(input, dim, keepdim):
    out_shape = list(input.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)

    indices = Tensor_(shape=out_shape, dtype=mindtorch.int64)
    values = Tensor_(shape=out_shape, dtype=input.dtype)

    return mindtorch.Tensor(indices), mindtorch.Tensor(values)

__all__.append('argmax_with_value')

def tile(input, dims):
    input_shape = input.shape
    out_shape = [input_shape[i] * dims[i] for i in range(input.ndim)]
    out = Tensor_(shape=tuple(out_shape), dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('tile')

def flatten_ext(input, start_dim, end_dim):
    input_shape = list(input.shape)
    if start_dim < 0:
        start_dim = start_dim + input.ndim
    if end_dim < 0:
        end_dim = end_dim + input.ndim

    flatten_shape = input_shape[:start_dim] + input_shape[start_dim:end_dim+1] + input_shape[end_dim+1:]
    out = Tensor_(shape=tuple(flatten_shape), dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('flatten_ext')

def cumsum_ext(input, dim, dtype):
    return input

__all__.append('cumsum_ext')

def squeeze(input, dim):
    input_shape = list(input.shape)
    if isinstance(dim, int):
        dim = (dim,)
    
    new_shape = ()
    for idx, s in enumerate(input_shape):
        if idx not in dim and s != 1:
            new_shape += (s,)

    out = Tensor_(shape=tuple(new_shape), dtype=input.dtype)
    return mindtorch.Tensor(out)

__all__.append('squeeze')

def exp(input):
    return input

__all__.append('exp')

def rand_ext(size, seed, offset, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return mindtorch.Tensor(out)

__all__.append('rand_ext')

def add(input, other):
    return input

__all__.append('add')

def neg(input):
    return input

__all__.append('neg')

def expm1(input):
    return input

__all__.append('expm1')

def reverse_v2(input, dims):
    return input

__all__.append('reverse_v2')

def rsqrt(input):
    return input

__all__.append('rsqrt')

def bitwise_xor_tensor(input, other):
    return input

__all__.append('bitwise_xor_tensor')

def divmod(input, other, rounding_mode):
    if isinstance(input, mindtorch.Tensor):
        return input
    return other

__all__.append('divmod')

def greater_equal(input, other):
    if isinstance(input, mindtorch.Tensor):
        return input
    return other

__all__.append('greater_equal')