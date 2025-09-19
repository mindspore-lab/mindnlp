import ctypes
import numbers
import numpy as np
import scipy
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
import mindtorch

__all__ = []

def empty(size, dtype):
    return mindtorch.Tensor.from_numpy(np.empty(size, mindtorch.dtype2np[dtype]))

__all__.append('empty')

def ones(size, dtype):
    return mindtorch.Tensor.from_numpy(np.ones(size, mindtorch.dtype2np[dtype]))

__all__.append('ones')

def zeros(size, dtype):
    return mindtorch.Tensor.from_numpy(np.zeros(size, mindtorch.dtype2np[dtype]))

__all__.append('zeros')

def arange(start, end, step, dtype):
    return mindtorch.Tensor.from_numpy(np.arange(start, end, step, mindtorch.dtype2np[dtype]))

__all__.append('arange')

def div(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
        if input.dtype == np.int64:
            input = input.astype(np.int32)
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.divide(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('div')

def pow_scalar_tensor(input, other):
    other = other.numpy()
    out = np.power(input, other)
    if out.dtype == np.float64:
        out = out.astype(np.float32)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('pow_scalar_tensor')

def mul(input, other):
    if not isinstance(input, numbers.Number):
        input = input.asnumpy()
    if not isinstance(other, numbers.Number):
        other = other.asnumpy()

    out = input * other
    if out.dtype == np.float64:
        out = out.astype(np.float32)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('mul')

def sub_ext(input, other, alpha):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.subtract(input, other * alpha)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('sub_ext')

def clamp_scalar(input, min, max):
    out = np.clip(input.numpy(), min, max)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('clamp_scalar')

def add(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.add(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('add')

dyn_shape_op = ops.TensorShape().set_device('CPU')
def dyn_shape(self):
    return dyn_shape_op(self)

__all__.append('dyn_shape')

def cast(input, dtype):
    if input.dtype == dtype:
        return input
    out = input.asnumpy().astype(mindtorch.dtype2np[dtype])
    return mindtorch.Tensor.from_numpy(out)

__all__.append('cast')

def getitem(input, slice):
    if isinstance(slice, tuple):
        new_slice = ()
        for s in slice:
            if isinstance(s, mindtorch.Tensor):
                s = s.numpy()
            new_slice += (s,)
    else:
        new_slice = slice
    out = input.asnumpy()[new_slice]
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('getitem')

def setitem(input, slice, value):
    out = input.asnumpy()
    out[slice] = value
    out = mindtorch.Tensor.from_numpy(out)
    input.assign_value(out)
    return input

__all__.append('setitem')

def contiguous(input):
    return input

__all__.append('contiguous')

def reshape(input, shape):
    out = np.reshape(input.asnumpy(), shape)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('reshape')

def bitwise_and_scalar(input, other):
    out = np.bitwise_and(input.numpy(), other)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('bitwise_and_scalar')


def bitwise_or_tensor(input, other):
    out = np.bitwise_or(input.numpy(), other.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('bitwise_or_tensor')

def right_shift(input, other):
    out = np.right_shift(input.numpy(), other)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('right_shift')

def transpose_ext_view(input, dim0, dim1):
    out = np.swapaxes(input.numpy(), dim0, dim1)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('transpose_ext_view')

def expand_dims_view(input, dim):
    out = np.expand_dims(input.numpy(), dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('expand_dims_view')

def equal(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.equal(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('equal')

def reduce_all(input, dim, keepdim):
    out = np.all(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('reduce_all')

def reduce_any(input, dim, keepdim):
    out = np.any(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('reduce_any')


def sum_ext(input, dim, keepdim, dtype):
    if dtype is not None:
        dtype = mindtorch.dtype2np[dtype]
    out = np.sum(input.numpy(), dim, dtype, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('sum_ext')

def full(size, fill_value):
    out = np.full(size, fill_value)
    if out.dtype == np.float64:
        out = out.astype(np.float32)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('full')

def zeros_like(input):
    out = np.zeros_like(input.numpy())

    return mindtorch.Tensor.from_numpy(out)

__all__.append('zeros_like')

broadcast_to_op = ops.Primitive('BroadcastTo').set_device('CPU')
def broadcast_to(input, shape):
    return broadcast_to_op(input, shape)

__all__.append('broadcast_to')

def uniform_real(size):
    out = np.random.rand(*size).astype(np.float32)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('uniform_real')

def normal(shape):
    out = np.random.normal(0., 1., shape).astype(np.float32)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('normal')

def pad_v3(input_x, padding, mode='constant', value=None):
    pad_op = ops.PadV3(mode=mode, paddings_contiguous=True).set_device('CPU')
    if input_x.dtype == mindtorch.bool:
        input_x = input_x.to(mindtorch.int32)
        value = int(value)
        out = pad_op(input_x, padding, value)
        return cast(out, mindtorch.bool)

    if isinstance(value, (float, int)):
        value = mindtorch.tensor(value, dtype=input_x.dtype)
    return pad_op(input_x, padding, value)

__all__.append('pad_v3')

def concat(tensors, dim):
    out = np.concatenate([t.numpy() for t in tensors], dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('concat')

def abs(input):
    out = np.abs(input.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('abs')

def mean_ext(input, dim, keepdim, dtype):
    out = np.mean(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('mean_ext')

def matmul_ext(input, other):
    out = np.matmul(input.numpy(), other.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('matmul_ext')

def max(input):
    out = np.max(input.numpy())

    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('max')


def min(input):
    out = np.min(input.numpy())

    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('min')


def randint(from_, to, shape, dtype, generator):
    out = np.random.randint(from_, to, shape, dtype=mindtorch.dtype2np[dtype])

    return mindtorch.Tensor.from_numpy(out)

__all__.append('randint')

def identity(input):
    out = np.copy(input.asnumpy())

    return mindtorch.Tensor.from_numpy(out)

__all__.append('identity')

# def non_zero()
def isclose(input, other, rtol, atol, equal_nan):
    out = np.isclose(input.numpy(), other.numpy(), rtol, atol, equal_nan)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('isclose')

def non_zero(input):
    out = np.nonzero(input.numpy())
    out = np.stack(out, 1)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('non_zero')

def tile(input, dims):
    out = np.tile(input.numpy(), dims)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('tile')

def squeeze(input, dim):
    if isinstance(dim, int) and input.shape[dim] != 1:
        return input
    out = np.squeeze(input.numpy(), dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('squeeze')

def index_select(input, dim, index):
    out = np.take(input.asnumpy(), index.asnumpy(), dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('index_select')

def rand_ext(size, seed, offset, dtype):
    out = np.random.randn(*size)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    out = out.astype(mindtorch.dtype2np[dtype])
    return mindtorch.Tensor.from_numpy(out)

__all__.append('rand_ext')

def numpy_to_tensor_overwrite(np_array, torch_tensor):
    if not np_array.flags.c_contiguous:
        np_array = np.ascontiguousarray(np_array)

    tensor_ptr = torch_tensor.data_ptr()
        
    ctypes.memmove(tensor_ptr, np_array.ctypes.data, torch_tensor.nbytes)
    
    return torch_tensor

def inplace_uniform(input, from_, to_, generator_):
    seed, _ = generator_._step(12)
    np.random.seed(seed.item())
    out = np.random.uniform(from_, to_, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)
    return input

__all__.append('inplace_uniform')

def inplace_fill_scalar(input, value):
    out = np.full_like(input.numpy(), value)
    numpy_to_tensor_overwrite(out, input)
    return input

__all__.append('inplace_fill_scalar')

def inplace_fill_tensor(input, value):
    out = np.full_like(input.numpy(), value)
    numpy_to_tensor_overwrite(out, input)
    return input

__all__.append('inplace_fill_tensor')

def inplace_normal(input, mean, std, generator_):
    out = np.random.normal(mean, std, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)

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
    
    numpy_to_tensor_overwrite(rnd, input)

    return input

__all__.append('inplace_random')

def inplace_copy(input, other):
    # input.assign_value(other)
    ctypes.memmove(input.data_ptr(), other.data_ptr(), input.nbytes)

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
    return mindtorch.Tensor.from_numpy(out)

__all__.append('round')

def isin(elements, test_elements):
    out = np.isin(elements, test_elements)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('isin')

def ldexp(input, other):
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.ldexp(input.numpy(), other)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('ldexp')

def less(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    
    out = input < other
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('less')

def cumsum_ext(input, dim, dtype):
    if dtype is not None:
        dtype = mindtorch.dtype2np[dtype]
    out = np.cumsum(input.numpy(), dim, dtype)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('cumsum_ext')

def greater_equal(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    
    out = input >= other
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('greater_equal')

def masked_fill(input, mask, value):
    out = np.where(mask.numpy(), value, input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('masked_fill')

def logical_not(input):
    out = np.logical_not(input.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('logical_not')

def not_equal(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    elif not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.not_equal(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('not_equal')

def less_equal(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    
    out = input <= other
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('less_equal')

def tril_ext(input, diagonal):
    out = np.tril(input.numpy(), diagonal)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('tril_ext')

def randperm_ext(n, seed, offset, dtype):
    out = np.random.permutation(n)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('randperm_ext')

def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq):
    out = np.take(weight.numpy(), input.numpy(), axis=0)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('embedding')

def randn(size, seed, offset, dtype):
    out = np.random.randn(*size).astype(mindtorch.dtype2np[dtype])
    return mindtorch.Tensor.from_numpy(out)

__all__.append('randn')

def erfinv(input):
    out = scipy.special.erfinv(input)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('erfinv')

def inplace_add_ext(input, other, alpha):
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    out = input.numpy() + other * alpha
    numpy_to_tensor_overwrite(out, input)
    return input

__all__.append('inplace_add_ext')

def pow_tensor_scalar(input, other):
    input = input.numpy()
    if input.dtype == np.int64:
        input = input.astype(np.int32)
    out = np.power(input, other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('pow_tensor_scalar')

stop_gradient_op = ops.StopGradient().set_device('CPU')
def stop_gradient(*args):
    return stop_gradient_op(*args)

__all__.append('stop_gradient')

def fmod_scalar(input, other):
    out = np.fmod(input.numpy(), other)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('fmod_scalar')

def argmax_with_value(input, dim, keepdim):
    indices = np.argmax(input.numpy(), dim, keepdims=keepdim)
    values = np.max(input.numpy(), dim, keepdims=keepdim)

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return mindtorch.Tensor.from_numpy(indices), mindtorch.Tensor.from_numpy(values)

__all__.append('argmax_with_value')

def argmin_with_value(input, dim, keepdim):
    indices = np.argmin(input.numpy(), dim, keepdims=keepdim)
    values = np.min(input.numpy(), dim, keepdims=keepdim)

    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return mindtorch.Tensor.from_numpy(indices), mindtorch.Tensor.from_numpy(values)

__all__.append('argmin_with_value')


def argmax_ext(input, dim, keepdim):
    indices = np.argmax(input.numpy(), dim, keepdims=keepdim)
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)
    return mindtorch.Tensor.from_numpy(indices)
__all__.append('argmax_ext')


def log(input):
    out = np.log(input.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('log')

def eye(n, m, dtype):
    out = np.eye(n, m, dtype=mindtorch.dtype2np[dtype])
    return mindtorch.Tensor.from_numpy(out)

__all__.append('eye')

def lin_space_ext(start, end, steps, dtype):
    out = np.linspace(start, end, steps, dtype=mindtorch.dtype2np[dtype])
    return mindtorch.Tensor.from_numpy(out)

__all__.append('lin_space_ext')

def upsample_bilinear2d(input, output_size, scale_factors, align_corners):
    resize = _get_cache_prim(ops.ResizeBilinearV2)(align_corners, not align_corners).set_device('CPU')
    return resize(input, output_size)

__all__.append('upsample_bilinear2d')

def split_with_size(tensor, split_size_or_sections, dim):
    out = np.array_split(tensor.numpy(), np.cumsum(split_size_or_sections[:-1]), dim)
    out = [mindtorch.Tensor.from_numpy(o) for o in out]
    return out

__all__.append('split_with_size')

def floor_div(input, other):
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    out = np.floor_divide(input.numpy(), other)
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('floor_div')

def sin(input):
    out = np.sin(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('sin')

def cos(input):
    out = np.cos(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('cos')

def triu(input, diagonal):
    out = np.triu(input.numpy(), diagonal)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('triu')

def sigmoid(input):
    out = 1 / (1 + np.exp(-input))
    return mindtorch.Tensor.from_numpy(out)

__all__.append('sigmoid')

def neg(input):
    out = -input.numpy()
    return mindtorch.Tensor.from_numpy(out)

__all__.append('neg')

def divmod(input, other, rounding_mode):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()

    if rounding_mode == 'floor':
        out = np.floor_divide(input, other)
    elif rounding_mode == 'trunc':
        out = np.trunc(np.true_divide(input, other)).astype(np.int64)

    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('divmod')

def unstack_ext_view(input, dim):
    arr = input.numpy()
    num_splits = arr.shape[dim]
    outs = np.split(arr, indices_or_sections=np.arange(1, num_splits), axis=dim)
    outs = [mindtorch.Tensor.from_numpy(out) for out in outs]
    return outs

__all__.append('unstack_ext_view')

def stack(tensors, dim):
    out = np.stack([t.numpy() for t in tensors], dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('stack')

def sqrt(input):
    if isinstance(input, numbers.Number):
        input = np.array(input)
    else:
        input = input.numpy()
    out = np.sqrt(input)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('sqrt')

def transpose_view(input, dims):
    out = np.transpose(input.numpy(), dims)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('transpose_view')

def einsum(equation, operands):
    out = np.einsum(equation, *[o.numpy() for o in operands])
    return mindtorch.Tensor.from_numpy(out)

__all__.append('einsum')

def std(input, dim, correction, keepdim):
    out = np.std(input.numpy(), dim, ddof=float(correction), keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('std')

def meshgrid(tensors, indexing):
    outs = np.meshgrid(*[t.numpy() for t in tensors], indexing=indexing)
    new_outs = ()
    for out in outs:
        if not isinstance(out, np.ndarray):
            out = np.array(out)
        new_outs += (mindtorch.Tensor.from_numpy(out),)
    return new_outs

__all__.append('meshgrid')

def repeat_interleave_tensor(input, repeats, dim, _):
    out = np.repeat(input.numpy(), repeats, dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('repeat_interleave_tensor')

def repeat_interleave_int(input, repeats, dim, _):
    out = np.repeat(input.numpy(), repeats, dim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('repeat_interleave_int')

def greater(input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()
    
    out = input > other
    if not isinstance(out, np.ndarray):
        out = np.array(out)

    return mindtorch.Tensor.from_numpy(out)

__all__.append('greater')

def linalg_vector_norm(input, p, dim, keepdim, dtype):
    out = np.linalg.norm(input.numpy(), p, dim, keepdim)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('linalg_vector_norm')

def exp(input):
    out = np.exp(input.numpy())
    if input.dtype == np.int64:
        out = out.astype(np.float32)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('exp')

def expm1(input):
    out = np.expm1(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('expm1')

def ones_like(input):
    out = np.ones_like(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('ones_like')

def reverse_v2(input, dims):
    out = np.flip(input.numpy(), dims)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('reverse_v2')

def rsqrt(input):
    out = np.reciprocal(np.sqrt(input.numpy()))
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('rsqrt')

def bitwise_xor_tensor(input, other):
    out = np.bitwise_xor(input.numpy(), other.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('bitwise_xor_tensor')

def minimum(input, other):
    out = np.minimum(input.numpy(), other.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('minimum')

def maximum(input, other):
    out = np.maximum(input.numpy(), other.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('maximum')

def prod_ext(input, dim, keepdim, dtype):
    out = np.prod(input.numpy(), axis=dim, keepdims=keepdim)
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('prod_ext')

def select(condition, input, other):
    if not isinstance(input, numbers.Number):
        input = input.numpy()
    if not isinstance(other, numbers.Number):
        other = other.numpy()

    out = np.where(condition.numpy(), input, other)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('select')

def dense(input, weight, bias):
    output = np.dot(input.numpy(), weight.numpy().T)
    if bias is not None:
        output += bias
    return mindtorch.Tensor.from_numpy(output)

__all__.append('dense')

def dropout_ext(input, p):
    if p != 0:
        mask = (np.random.rand(*input.shape) < (1 - p))
        out = input.numpy() * mask / (1 - p)
        return mindtorch.Tensor.from_numpy(out), mindtorch.Tensor.from_numpy(mask)
    else:
        return input, None

__all__.append('dropout_ext')

def floor(input):
    out = np.floor(input.numpy())
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('floor')

def chunk(input, chunks, dim):
    out = np.array_split(input.numpy(), chunks, dim)
    out = [mindtorch.Tensor.from_numpy(o) for o in out]
    return out

__all__.append('chunk')

def narrow(input, dim, start, length):
    slices = [slice(None)] * input.ndim
    # 将指定维度的切片修改为 [start: start+length]
    slices[dim] = slice(start, start + length)
    # 应用切片并返回视图
    out = input.numpy()[tuple(slices)]
    return mindtorch.Tensor.from_numpy(out)

__all__.append('narrow')

def roll(input, shifts, dims):
    out = np.roll(input.numpy(), shifts, dims)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('roll')

def outer(input, other):
    out = np.outer(input.numpy(), other.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('outer')

def one_hot_ext(tensor, num_classes=-1):
    if num_classes == -1:
        num_classes = np.max(tensor.numpy()) + 1  # 自动确定类别数[2](@ref)
    
    out = np.eye(num_classes)[tensor.numpy()]
    return mindtorch.Tensor.from_numpy(out)

__all__.append('one_hot_ext')

def log1p(input):
    out = np.log1p(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('log1p')

def gather(input, indices, _dimension):
    out = np.take(input.numpy(), indices.numpy(), _dimension)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('gather')


def layer_norm_ext(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    # 确定需要计算均值和方差的轴
    # 从第一个维度开始到 normalized_shape 所涵盖的维度之前的维度会被保留（即 batch 维度等）
    # 我们需要计算所有不在最后 len(normalized_shape) 个维度上的轴的均值和方差
    input = input.numpy()
    if weight is not None:
        weight = weight.numpy()
    if bias is not None:
        bias = bias.numpy()

    start_axis = input.ndim - len(normalized_shape)
    axes = tuple(range(start_axis, input.ndim))
    
    # 计算均值和方差，并保持维度以便广播
    mean = np.mean(input, axis=axes, keepdims=True)
    var = np.var(input, axis=axes, keepdims=True)
    
    # 标准化: (x - mean) / sqrt(var + eps)
    normalized = (input - mean) / np.sqrt(var + eps)
    
    # 应用可学习的缩放和平移参数 (gamma 和 beta)
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    
    return (mindtorch.Tensor.from_numpy(normalized),)

__all__.append('layer_norm_ext')

def erf(input):
    out = scipy.special.erf(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('erf')

def mse_loss_ext(input, target, reduction='mean'):
    if input.shape != target.shape:
        raise ValueError(f"Input and target must have the same shape. Got input: {input.shape}, target: {target.shape}")

    squared_errors = np.square(input - target)

    if reduction == 'mean':
        loss = np.mean(squared_errors)
    elif reduction == 'sum':
        loss = np.sum(squared_errors)
    elif reduction == 'none':
        loss = squared_errors
    else:
        raise ValueError("Reduction must be 'mean', 'sum', or 'none'.")

    if not isinstance(loss, np.ndarray):
        loss = np.array(loss)
    return mindtorch.Tensor.from_numpy(loss)

__all__.append('mse_loss_ext')

def square(input):
    out = np.square(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('square')

def lgamma(input):
    out = scipy.special.gammaln(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('lgamma')

def gamma(shape, alpha, beta):
    out = np.random.gamma(alpha, 1/beta, shape)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('gamma')

def gather_d(input, dim, index):
    indices = []
    for axis in range(input.ndim):
        if axis == dim:
            indices.append(index)
        else:
            shape = [1] * index.ndim
            shape[axis] = input.shape[axis]
            indices.append(np.arange(input.shape[axis]).reshape(shape))
    
    out = input[tuple(indices)]
    if not isinstance(out, np.ndarray):
        out = np.array(out)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('gather_d')


def log_softmax(x, axis=-1):
    x = x.numpy()
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    exp_x = np.exp(x_shifted)
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    log_sum_exp_x = np.log(sum_exp_x)
    
    out = x_shifted - log_sum_exp_x
    return mindtorch.Tensor.from_numpy(out)

__all__.append('log_softmax')

def nllloss(input, target, weight=None, reduction='mean', ignore_index=-100):
    op = ops.NLLLoss(reduction, ignore_index).set_device('CPU')
    return op(input, target, weight)

__all__.append('nllloss')

def linalg_qr(A, mode):
    # out = np.linalg.qr(A.numpy(), mode)
    # return [mindtorch.Tensor.from_numpy(o) for o in out]
    if mode not in ('reduced', 'complete'):
        raise TypeError(f"For qr, the arg mode must be 'reduced' or 'complete', but got {mode}.")
    qr_ = _get_cache_prim(ops.Qr)(mode == 'complete').set_device('CPU')
    return qr_(A)

__all__.append('linalg_qr')

def diag_ext(input, diagonal):
    out = np.diag(input.numpy(), diagonal)
    return mindtorch.Tensor.from_numpy(out)

__all__.append('diag_ext')

def sign(input):
    out = np.sign(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('sign')

def log2(input):
    out = np.log2(input.numpy())
    return mindtorch.Tensor.from_numpy(out)

__all__.append('log2')
