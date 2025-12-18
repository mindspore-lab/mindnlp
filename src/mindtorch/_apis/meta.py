import math
import numpy as np
import mindspore
import mindtorch

__all__ = []

def empty(size, dtype):
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out

def empty_like(input, dtype):
    return empty(input.shape, input.dtype)

def arange(start, end, step, dtype):
    out = mindspore.Tensor(init='meta', shape=(math.ceil((end - start) / step), ), dtype=dtype)
    return out

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

    out = mindspore.Tensor(init='meta', shape=out_shape, dtype=input.dtype)
    return out

__all__.append('broadcast_to')

def zeros(size, dtype):
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out

__all__.append('zeros')

def ones(size, dtype):
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out

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

def getitem(input, slice_spec):
    # Handle zero-sized dimensions
    input_shape = input.shape
    slice_type = type(slice(None))  # Get the slice type
    
    # Check if any dimension is 0 and if we're trying to index into it
    if isinstance(slice_spec, tuple):
        for i, s in enumerate(slice_spec):
            if i < len(input_shape) and input_shape[i] == 0:
                # If dimension is 0, check if we're indexing into it (not just using :)
                if isinstance(s, int) or (type(s) == slice_type and s != slice(None)):
                    # Return empty tensor with appropriate shape
                    # Calculate output shape
                    output_shape = []
                    for j, dim_size in enumerate(input_shape):
                        if j < len(slice_spec):
                            s_j = slice_spec[j]
                            if isinstance(s_j, int):
                                # Integer indexing removes the dimension
                                continue
                            elif type(s_j) == slice_type:
                                # Slice indexing keeps the dimension
                                start = s_j.start if s_j.start is not None else 0
                                stop = s_j.stop if s_j.stop is not None else dim_size
                                step = s_j.step if s_j.step is not None else 1
                                if start < 0:
                                    start += dim_size
                                if stop < 0:
                                    stop += dim_size
                                start = max(0, min(start, dim_size))
                                stop = max(start, min(stop, dim_size))
                                size = max(0, (stop - start + step - 1) // step)
                                output_shape.append(size)
                            else:
                                output_shape.append(dim_size)
                        else:
                            output_shape.append(dim_size)
                    return mindspore.Tensor(init='meta', shape=tuple(output_shape), dtype=input.dtype)
    
    # Try to compute output shape using numpy, but handle errors gracefully
    try:
        # Use a dummy array with the same shape to compute output shape
        dummy = np.zeros(input_shape)
        out_shape = dummy[slice_spec].shape
        out = mindspore.Tensor(init='meta', shape=out_shape, dtype=input.dtype)
        return out
    except (IndexError, ValueError):
        # If numpy fails (e.g., zero-sized dimension), compute shape manually
        # For now, return a tensor with the same shape (this is a fallback)
        # In practice, this case should be handled by the check above
        return mindspore.Tensor(init='meta', shape=input_shape, dtype=input.dtype)

__all__.append('getitem')

def sub(input, other, alpha=1.0):
    if isinstance(input, mindtorch.Tensor):
        return input
    return other

__all__.append('sub')

def pad_v3(input, pad, mode, value):
    out = np.pad(np.zeros(input.shape), pad, mode, constant_values=value)
    out = mindspore.Tensor(init='meta', shape=out.shape, dtype=input.dtype)
    return out

__all__.append('pad_v3')

def abs(input):
    return input

__all__.append('abs')

def cast(input, dtype):
    out = mindspore.Tensor(init='meta', shape=input.shape, dtype=dtype)
    return out

__all__.append('cast')

def index_select(input, dim, index):
    out = np.take(np.zeros(input.shape), np.zeros(index.shape, dtype=np.int64), dim)
    out = mindspore.Tensor(init='meta', shape=out.shape, dtype=input.dtype)
    return out

__all__.append('index_select')

def identity(input):
    out = mindspore.Tensor(init='meta', shape=input.shape, dtype=input.dtype)
    return out

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
    out = mindspore.Tensor(init='meta', shape=shape, dtype=dtype)
    return out

__all__.append('div')

def pow(input, other):
    out = mindspore.Tensor(init='meta', shape=other.shape, dtype=other.dtype)
    return out

def concat(tensors, dim=None, axis=None):
    # Support both dim and axis for compatibility
    if axis is not None:
        dim = axis
    if dim is None:
        dim = 0
    shape = list(tensors[0].shape)
    shape[dim] = sum([t.shape[dim] for t in tensors])
    out = mindspore.Tensor(init='meta', shape=tuple(shape), dtype=tensors[0].dtype)
    return out

__all__.append('concat')

def tril(input, k):
    return input

__all__.append('tril')

def reshape(input, shape):
    if -1 in shape:
        out = np.zeros(input.shape).reshape(shape)
        shape = out.shape
    out = mindspore.Tensor(init='meta', shape=tuple(shape), dtype=input.dtype)
    return out

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
    out = mindspore.Tensor(init='meta', shape=tuple(new_shape), dtype=dtype)
    return out

__all__.append('linalg_vector_norm')

def erfinv(input):
    return input
__all__.append('erfinv')


def stop_gradient(input):
    out = mindspore.Tensor(init='meta', shape=input.shape, dtype=input.dtype)
    return out

__all__.append('stop_gradient')

def log(input):
    return input
__all__.append('log')

def mul(input, other):
    if isinstance(input, mindtorch.Tensor):
        shape = input.shape
        dtype = input.dtype
    else:
        shape = other.shape
        dtype = other.dtype

    out = mindspore.Tensor(init='meta', shape=shape, dtype=dtype)
    return out

def inplace_mul(input, other):
    return input

__all__.append('mul')

def randn(size, generator, dtype):
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out

__all__.append('randn')

def zeros_like(input, *args, **kwargs):
    out = mindspore.Tensor(init='meta', shape=input.shape, dtype=input.dtype)
    return out
__all__.append('zeros_like')

def inplace_add(input, other, alpha):
    return input
__all__.append('inplace_add')

def clamp_scalar(input, *args):
    return input
__all__.append('clamp_scalar')

def expand_dims(input, dim):
    input_shape = list(input.shape)
    input_shape.insert(dim, 1)

    out = mindspore.Tensor(init='meta', shape=tuple(input_shape), dtype=input.dtype)
    return out


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
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out

__all__.append('fill_scalar')

def sqrt(input):
    return input

__all__.append('sqrt')

def normal_float_float(mean, std, size, dtype, geneartor):
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out


__all__.append('normal_float_float')


def split_with_size(tensor, split_size_or_sections, dim=0):
    """
    Meta backend: return meta tensors with correct shapes for split_with_size.
    """
    dim = int(dim)
    full_shape = list(tensor.shape)
    total = full_shape[dim]

    if isinstance(split_size_or_sections, int):
        size = split_size_or_sections
        if size <= 0:
            raise ValueError("split_size must be > 0")
        split_sizes = []
        remaining = total
        while remaining > 0:
            split_sizes.append(min(size, remaining))
            remaining -= size
    elif isinstance(split_size_or_sections, (list, tuple)):
        split_sizes = list(split_size_or_sections)
        if sum(split_sizes) != total:
            raise ValueError("sum of split_sizes must equal tensor size along dim")
    else:
        raise TypeError("split_size_or_sections must be int, list or tuple")

    outputs = []
    for sz in split_sizes:
        out_shape = list(full_shape)
        out_shape[dim] = sz
        outputs.append(mindspore.Tensor(init='meta', shape=tuple(out_shape), dtype=tensor.dtype))
    return outputs

__all__.append('split_with_size')

def stack(tensors, dim):
    x_shape = list(tensors[0].shape)
    x_shape.insert(dim, len(tensors))
    out = mindspore.Tensor(init='meta', shape=tuple(x_shape), dtype=tensors[0].dtype)
    return out

__all__.append('stack')

def argmax_with_value(input, dim, keepdim):
    out_shape = list(input.shape)
    if keepdim:
        out_shape[dim] = 1
    else:
        out_shape.pop(dim)

    indices = mindspore.Tensor(init='meta', shape=out_shape, dtype=mindtorch.int64)
    values = mindspore.Tensor(init='meta', shape=out_shape, dtype=input.dtype)

    return indices, values

__all__.append('argmax_with_value')

def tile(input, dims):
    input_shape = input.shape
    out_shape = [input_shape[i] * dims[i] for i in range(input.ndim)]
    out = mindspore.Tensor(init='meta', shape=tuple(out_shape), dtype=input.dtype)
    return out

__all__.append('tile')

def flatten(input, start_dim, end_dim):
    input_shape = list(input.shape)
    if start_dim < 0:
        start_dim = start_dim + input.ndim
    if end_dim < 0:
        end_dim = end_dim + input.ndim

    flatten_shape = input_shape[:start_dim] + input_shape[start_dim:end_dim+1] + input_shape[end_dim+1:]
    out = mindspore.Tensor(init='meta', shape=tuple(flatten_shape), dtype=input.dtype)
    return out

__all__.append('flatten')

def cumsum(input, dim, dtype):
    return input

__all__.append('cumsum')

def squeeze(input, dim):
    input_shape = list(input.shape)
    if isinstance(dim, int):
        dim = (dim,)
    
    new_shape = ()
    for idx, s in enumerate(input_shape):
        if idx not in dim and s != 1:
            new_shape += (s,)

    out = mindspore.Tensor(init='meta', shape=tuple(new_shape), dtype=input.dtype)
    return out

__all__.append('squeeze')

def exp(input):
    return input

__all__.append('exp')

def rand(size, generator, dtype):
    out = mindspore.Tensor(init='meta', shape=size, dtype=dtype)
    return out

__all__.append('rand')

def add(input, other, alpha):
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

def greater(input, other):
    if isinstance(input, mindtorch.Tensor):
        return input
    return other

def less(input, other):
    if isinstance(input, mindtorch.Tensor):
        return input
    return other

def inplace_zero(input):
    return input

def clone(input):
    return input

def select(condition, input, other):
    return input

def logical_not(input):
    return input

def pad(input, pad, mode='constant', value=None):
    size = input.shape
    if len(pad) == 2:
        new_size = size[:-1] + (size[-1] + sum(pad),)
    elif len(pad) == 4:
        new_size = size[:-2] + (size[-2] + pad[2] + pad[3], size[-1] + pad[0] + pad[1])
    elif len(pad) == 6:
        new_size = size[:-3] + (size[-3] + pad[4] + pad[5], size[-2] + pad[2] + pad[3], size[-1] + pad[0] + pad[1])
    else:
        raise ValueError('pad size must be 2, 4 or 6')
 
    out = mindspore.Tensor(init='meta', shape=new_size, dtype=input.dtype)
    return out

def setitem(self, slice, value):
    return self

def meshgrid(args, lambd):
    res = np.meshgrid(*args, indexing=lambd)
    outs = ()
    for r in res:
        out = mindspore.Tensor(init='meta', shape=r.shape, dtype=args[0].dtype)
        out = out
        outs += (out,)
    return outs

def permute(input, dims):
    out = mindspore.Tensor(init='meta', shape=dims, dtype=input.dtype)
    return out