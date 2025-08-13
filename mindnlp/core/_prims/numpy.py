import numbers
import numpy as np
from mindspore import ops
from mindnlp import core

__all__ = []

def empty(size, dtype):
    return core.Tensor.from_numpy(np.empty(size, core.dtype2np[dtype]))

__all__.append('empty')

def ones(size, dtype):
    return core.Tensor.from_numpy(np.ones(size, core.dtype2np[dtype]))

__all__.append('ones')

def zeros(size, dtype):
    return core.Tensor.from_numpy(np.zeros(size, core.dtype2np[dtype]))

__all__.append('zeros')

def arange(start, end, step, dtype):
    return core.Tensor.from_numpy(np.arange(start, end, step, core.dtype2np[dtype]))

__all__.append('arange')

def div(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.divide(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('div')

def pow_scalar_tensor(input, other):
    out = np.power(input, other.numpy())
    return core.Tensor.from_numpy(out)

__all__.append('pow_scalar_tensor')

def mul(input, other):
    if not isinstance(input, numbers.Number):
        input = input.asnumpy()
    elif not isinstance(other, numbers.Number):
        other = other.asnumpy()
    out = np.multiply(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('mul')

def sub_ext(input, other, alpha):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.subtract(input, other * alpha)
    return core.Tensor.from_numpy(out)

__all__.append('sub_ext')

def clamp_scalar(input, min, max):
    out = np.clip(input.numpy(), min, max)
    return core.Tensor.from_numpy(out)

__all__.append('clamp_scalar')

def add(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.add(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('add')

dyn_shape_op = ops.TensorShape().set_device('CPU')
def dyn_shape(self):
    return dyn_shape_op(self)

__all__.append('dyn_shape')

def cast(input, dtype):
    out = input.asnumpy().astype(core.dtype2np[dtype])
    return core.Tensor.from_numpy(out)

__all__.append('cast')

def getitem(input, slice):
    out = input.asnumpy()[slice]
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('getitem')

def setitem(input, slice, value):
    out = input.asnumpy()
    out[slice] = value
    out = core.Tensor.from_numpy(out)
    input.assign_value(out)
    return input

__all__.append('setitem')

def contiguous(input):
    return input

__all__.append('contiguous')

def reshape(input, shape):
    out = np.reshape(input.asnumpy(), shape)
    return core.Tensor.from_numpy(out)

__all__.append('reshape')

def bitwise_and_scalar(input, other):
    out = np.bitwise_and(input.numpy(), other)
    return core.Tensor.from_numpy(out)

__all__.append('bitwise_and_scalar')

def right_shift(input, other):
    out = np.right_shift(input.numpy(), other)
    return core.Tensor.from_numpy(out)

__all__.append('right_shift')

def transpose_ext_view(input, dim0, dim1):
    out = np.swapaxes(input.numpy(), dim0, dim1)
    return core.Tensor.from_numpy(out)

__all__.append('transpose_ext_view')

def expand_dims_view(input, dim):
    out = np.expand_dims(input.numpy(), dim)
    return core.Tensor.from_numpy(out)

__all__.append('expand_dims_view')

def equal(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.equal(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('equal')

def reduce_all(input, dim, keepdim):
    out = np.all(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('reduce_all')

def reduce_any(input, dim, keepdim):
    out = np.any(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('reduce_any')


def sum_ext(input, dim, keepdim, dtype):
    if dtype is not None:
        dtype = core.dtype2np[dtype]
    out = np.sum(input.numpy(), dim, dtype, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('sum_ext')

def full(size, fill_value):
    out = np.full(size, fill_value)
    return core.Tensor.from_numpy(out)

__all__.append('full')

def zeros_like(input):
    out = np.zeros_like(input.numpy())

    return core.Tensor.from_numpy(out)

__all__.append('zeros_like')

broadcast_to_op = ops.Primitive('BroadcastTo').set_device('CPU')
def broadcast_to(input, shape):
    return broadcast_to_op(input, shape)

__all__.append('broadcast_to')

def uniform_real(size):
    out = np.random.rand(*size).astype(np.float32)
    return core.Tensor.from_numpy(out)

__all__.append('uniform_real')

def normal(shape):
    out = np.random.normal(0., 1., shape).astype(np.float32)
    return core.Tensor.from_numpy(out)

__all__.append('normal')

def pad_v3(input, pad, mode, value):
    out = np.pad(input.asnumpy(), pad, mode, constant_values=value)
    return core.Tensor.from_numpy(out)

__all__.append('pad_v3')

def concat(tensors, dim):
    out = np.concatenate([t.numpy() for t in tensors], dim)
    return core.Tensor.from_numpy(out)

__all__.append('concat')

def abs(input):
    out = np.abs(input.numpy())
    return core.Tensor.from_numpy(out)

__all__.append('abs')

def mean_ext(input, dim, keepdim, dtype):
    out = np.mean(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('mean_ext')

def matmul_ext(input, other):
    out = np.matmul(input.numpy(), other.numpy())
    return core.Tensor.from_numpy(out)

__all__.append('matmul_ext')

def max(input):
    out = np.max(input.numpy())

    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('max')

def randint(from_, to, shape, dtype, generator):
    out = np.random.randint(from_, to, shape, dtype=core.dtype2np[dtype])

    return core.Tensor.from_numpy(out)

__all__.append('randint')

def identity(input):
    out = np.copy(input.asnumpy())

    return core.Tensor.from_numpy(out)

__all__.append('identity')

# def non_zero()
def isclose(input, other, rtol, atol, equal_nan):
    out = np.isclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)
    return core.Tensor.from_numpy(out)

__all__.append('isclose')

def non_zero(input):
    out = np.nonzero(input.numpy())
    out = np.stack(out, 1)
    return core.Tensor.from_numpy(out)

__all__.append('non_zero')

def tile(input, dims):
    out = np.tile(input.numpy(), dims)

    return core.Tensor.from_numpy(out[0])

__all__.append('tile')

def squeeze(input, dim):
    out = np.squeeze(input.numpy(), dim)
    return core.Tensor.from_numpy(out)

__all__.append('squeeze')

def index_select(input, dim, index):
    out = np.take(input.asnumpy(), index.asnumpy(), dim)
    return core.Tensor.from_numpy(out)

__all__.append('index_select')

def rand_ext(size, seed, offset, dtype):
    out = np.random.randn(*size).astype(core.dtype2np[dtype])
    return core.Tensor.from_numpy(out[0])

__all__.append('rand_ext')

def inplace_uniform(input, from_, to_, generator_):
    out = np.random.uniform(from_, to_, input.shape).astype(core.dtype2np[input.dtype])
    input.assign_value(core.Tensor.from_numpy(out))
    return input

__all__.append('inplace_uniform')

def inplace_fill_scalar(input, value):
    out = np.full_like(input.numpy(), value)
    input.assign_value(core.Tensor.from_numpy(out))
    return input

__all__.append('inplace_fill_scalar')

def inplace_normal(input, mean, std, seed, offset):
    out = np.random.normal(mean, std, input.shape).astype(core.dtype2np[input.dtype])
    input.assign_value(core.Tensor.from_numpy(out))
    return input

__all__.append('inplace_normal')

def inplace_random(input, from_val=0, to_val=None, seed=None, offset=None):
    # 选择随机数生成器
    rng = np.random
    arr = input.numpy()
    if np.issubdtype(arr.dtype, np.floating):
        # 浮点类型处理
        if to_val is None:
            # 默认 [0, 1) 均匀分布
            rnd = rng.random(size=arr.shape).astype(arr.dtype)
        else:
            rnd = (from_val + (to_val - from_val) * rng.random(size=arr.shape)).astype(arr.dtype)
            
    elif np.issubdtype(arr.dtype, np.integer):
        # 整数类型处理
        from_int = int(from_val)
        
        if to_val is None:
            # 默认范围 [0, dtype.max]
            max_val = np.iinfo(arr.dtype).max
            rnd = rng.randint(0, max_val + 1, size=arr.shape).astype(arr.dtype)
        else:
            # 指定范围 [from_int, to_val)
            to_int = int(to_val)
            
            # 验证参数有效性
            if from_int >= to_int:
                raise ValueError(f"Empty range for integers: from={from_int} >= to={to_int}")
                
            # 处理整数边界问题
            dtype_min = np.iinfo(arr.dtype).min
            dtype_max = np.iinfo(arr.dtype).max
            from_int = np.clip(from_int, dtype_min, dtype_max)
            to_int = np.clip(to_int, dtype_min + 1, dtype_max + 1)
            
            rnd = rng.randint(from_int, to_int, size=arr.shape).astype(arr.dtype)
            
    elif arr.dtype == bool:
        # 布尔类型处理 (忽略 from_val/to_val)
        rnd = rng.random(size=arr.shape) > 0.5
    
    else:
        raise TypeError(f"Unsupported data type: {arr.dtype}")
    
    input.assign_value(core.Tensor.from_numpy(rnd))
    return input

__all__.append('inplace_random')

def inplace_copy(input, other):
    input.assign_value(other)
    return input

__all__.append('inplace_copy')

def softmax(input, dim):
    softmax_op = ops.Softmax(dim).set_device('CPU')
    return softmax_op(input)


__all__.append('softmax')

def topk(input, k, sorted=True):
    topk_op = ops.TopK(sorted).set_device('CPU')
    return topk_op(input, k)

__all__.append('topk')

def sort_ext(input, dim, descending, stable):
    sort_op = ops.Sort(dim, descending).set_device('CPU')
    return sort_op(input)

__all__.append('sort_ext')

def round(input):
    out = np.round(input.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('round')

def isin(elements, test_elements):
    out = np.isin(elements, test_elements)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return core.Tensor.from_numpy(out)

__all__.append('isin')

def ldexp(input, other):
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.ldexp(input.numpy(), other)
    return core.Tensor.from_numpy(out)

__all__.append('ldexp')

def less(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    
    out = input < other
    return core.Tensor.from_numpy(out)

__all__.append('less')

def cumsum_ext(input, dim, dtype):
    if dtype is not None:
        dtype = core.dtype2np[dtype]
    out = np.cumsum(input.numpy(), dim, dtype)

    return core.Tensor.from_numpy(out)

__all__.append('cumsum_ext')

def greater_equal(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    
    out = input >= other
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return core.Tensor.from_numpy(out)

__all__.append('greater_equal')

def masked_fill(input, mask, value):
    out = np.where(mask.numpy(), value, input.numpy())
    return core.Tensor.from_numpy(out)

__all__.append('masked_fill')
