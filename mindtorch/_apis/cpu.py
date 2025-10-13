import ctypes
import numbers
import math
import numpy as np
import mindspore
from mindspore.ops.auto_generate.gen_ops_prim import Empty
import mindtorch
from .._op_prim.cpu import legacy

empty_op = Empty().set_device('CPU')
def empty(size, dtype):
    return empty_op(size, dtype=dtype, device='CPU')

def inplace_normal(input, mean, std, generator_):
    out = np.random.normal(mean, std, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)

    return input

def select_ext_view(input, dim, index):
    return legacy.select_view(input, index, dim)

def inplace_copy(input, value):
    if value.shape != input.shape:
        value = legacy.fill_v2(input.shape, value)
    # inplace_copy(input, value)
    # t2t_overwrite(input, value)
    # legacy.assign(input, value)
    if hasattr(input, '_base'):
        input._base.assign_value(value)
    input.assign_value(value)
    return input

def fill_scalar(size, fill_value, dtype):
    if dtype is None:
        return legacy.fill_v2(size, mindspore.Tensor(fill_value))
    return legacy.cast(legacy.fill_v2(size, mindspore.Tensor(fill_value)), dtype)

def fill_tensor(size, fill_value, dtype):
    return legacy.cast(legacy.fill_v2(size, fill_value), dtype)


def inplace_fill_scalar(input, value):
    out = np.full_like(input.numpy(), value)
    numpy_to_tensor_overwrite(out, input)
    return input

def zeros_like(input, dtype):
    if dtype is None:
        return legacy.zeros_like(input)
    return legacy.cast(legacy.zeros_like(input), dtype)

def tensor_shape(input):
    return legacy.tensor_shape(input)

def arange(start, end, step, dtype):
    return mindtorch.Tensor.from_numpy(np.arange(start, end, step, mindtorch.dtype2np[dtype]))

def broadcast_to(input, shape):
    return legacy.broadcast_to(input, shape)

def zeros(shape, dtype):
    return legacy.zeros(shape, dtype)

def inplace_uniform(input, from_, to_, generator_):
    seed, _ = generator_._step(12)
    np.random.seed(seed.item())
    out = np.random.uniform(from_, to_, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)
    return input

def sub(input, other, alpha=1):
    return legacy.sub(input, legacy.mul(other, alpha))

def contiguous(input):
    return input

def inplace_zero(input):
    inplace_copy(input, legacy.zeros_like(input))
    return input

def abs(input):
    return legacy.abs(input)

def identity(input):
    return legacy.identity(input)

def clone(input):
    return cast(legacy.mul(input, 1), input.dtype)

def max(input):
    return legacy.reduce_max(input, (), False)

def ones(shape, dtype):
    return legacy.ones(shape, dtype)

def mean(input, dim, keepdim, dtype):
    if dtype is not None:
        input = legacy.cast(input, dtype)
    if dim is None:
        dim = ()
    return legacy.reduce_mean(input, dim, keepdim)

def transpose_view(input, dim0, dim1):
    """
    Transposes the input tensor along the specified dimensions.

    Args:
        input (Tensor): The input tensor.
        dim0 (int): The first dimension to transpose.
        dim1 (int): The second dimension to transpose.

    Returns:
        Tensor: The transposed tensor.
    """
    ranks = list(range(input.ndim))
    rank0 = ranks[dim0]
    rank1 = ranks[dim1]
    ranks[dim0] = rank1
    ranks[dim1] = rank0
    return legacy.transpose(input, tuple(ranks))

def matmul(self, other):
    if self.ndim > 2:
        if self.ndim == other.ndim:
            return legacy.batch_mat_mul(self, other, False, False)
        else:
            self_shape = self.shape
            other_shape = other.shape
            if other.ndim == 2:
                self = reshape(self, (-1, self_shape[-1]))
                out = legacy.mat_mul(self, other, False, False)
                return reshape(out, (*self_shape[:-1], out.shape[-1]))
            if self.ndim == 2:
                other = reshape(other, (-1, other_shape[-1]))
                out = legacy.mat_mul(self, other, False, False)
                return reshape(out, (*other_shape[:-1], out.shape[-1]))
    
    return legacy.mat_mul(self, other, False, False)

def div(input, other):
    return legacy.div(input, other)

def mul(input, other):
    return legacy.mul(input, other)

def reduce_all(input, axis, keepdims):
    return legacy.reduce_all(input, axis, keepdims)

def isclose(input, other, rtol, atol, equal_nan):
    return legacy.is_close(input, other, rtol, atol, equal_nan)

def equal(input, other):
    return legacy.reduce_all(legacy.equal(input, other), (), False)

def eq(input, other):
    return legacy.equal(input, other)


def expand_dims(input, dim):
    return legacy.expand_dims(input, dim)

def tile(input, dims):
    return legacy.tile(input, dims)

py_slice = slice
def slice(self, dim, start, end, step):
    ndim = self.ndim
    begins = [0] * ndim
    ends = [i for i in self.shape]
    strides = [1] * ndim
    begins[dim] = start
    ends[dim] = end
    strides[dim] = step
    return legacy.strided_slice(self, tuple(begins), tuple(ends), tuple(strides), 0, 0, 0, 0, 0)

def pad_v3(input, new_pad, mode, value=None, contiguous=True):
    return legacy.pad_v3(input, new_pad, value, mode, contiguous)

def cumsum(self, dim, dtype):
    if self.shape[dim] == 0:
        return mindtorch.tensor([], dtype=self.dtype, device=self.device)
    return legacy.cum_sum(self, dim, False, False)

def reduce_any(input, axis, keepdims):
    return legacy.reduce_any(input, axis, keepdims)

def concat(tensors, axis):
    return legacy.concat(tensors, axis)

def numpy_to_tensor_overwrite(np_array, tensor):
    if not np_array.flags.c_contiguous:
        np_array = np.ascontiguousarray(np_array)

    tensor_ptr = tensor.data_ptr()
        
    ctypes.memmove(tensor_ptr, np_array.ctypes.data, tensor.nbytes)
    
    return tensor

def t2t_overwrite(input, other):
    other._device = input.device
    ctypes.memmove(input.data_ptr(), other.data_ptr(), input.nbytes)
    return input


def inplace_random(input, from_val=0, to_val=None, generator=None):
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

def gather_d(input, dim, index):
    return legacy.gather_d(input, dim, index)

def reshape(input, shape):
    return legacy.reshape(input, shape)

def flatten(input, start_dim, end_dim):
    if start_dim < 0:
        start_dim = start_dim + input.ndim
    if end_dim < 0:
        end_dim = end_dim + input.ndim
    input_shape = list(input.shape)
    input_shape[start_dim:end_dim+1] = [-1]
    return legacy.reshape(input, tuple(input_shape))

def sort(input, dim, descending, stable):
    return legacy.sort(input, dim, descending)

def gather(input_params, input_indices, axis, batch_dim):
    return legacy.gather(input_params, input_indices, axis, batch_dim)

def randint(low, high, shape, generator, dtype):
    value = legacy.uniform_int(shape,
                                mindspore.tensor(low, dtype=mindspore.int32),
                                mindspore.tensor(high, dtype=mindspore.int32), 0, 0)
    return value

def add(input, other, alpha=1):
    if alpha == 1.0:
        return legacy.add(input, other)
    return legacy.add(input, legacy.mul(other, alpha))

def non_zero(input):
    return legacy.non_zero(input)

def stop_gradient(input):
    return legacy.stop_gradient(input)

def squeeze(input, axis):
    return legacy.squeeze(input, axis)

def softmax(input, axis):
    if axis is None:
        axis = -1
    return legacy.softmax(input, axis)

def topk(input, k, dim, largest, sorted):
    if not largest:
        input = -input
    if dim is None or dim == input.ndim - 1:
        if not largest:
            res = legacy.top_k(input, k, sorted)
            values, indices = -res[0], res[1]
            return values, indices
        return legacy.top_k(input, k, sorted)
    input = transpose_view(input, dim, input.ndim - 1)
    output = legacy.top_k(input, k, sorted)
    values = transpose_view(output[0], dim, input.ndim - 1)
    indices = transpose_view(output[1], dim, input.ndim - 1)
    if not largest:
        res = (-values, indices)
    else:
        res = (values, indices)
    return res

def strided_slice(input, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):
    return legacy.strided_slice(input, tuple(begin), tuple(end), tuple(strides), begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

def strided_slice_grad(input, begin, end, strides, update, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    return legacy.strided_slice_grad(update, input.shape, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

def masked_select(input, mask):
    return legacy.masked_select(input, mask)

def stack(values, axis=0):
    return legacy.stack(values, axis)

def cast(input, dtype):
    return legacy.cast(input, dtype)

def less(input, other):
    return legacy.less(input, other)

def select(condition, x, y):
    return legacy.select(condition, x, y)

def round(input, decimals):
    return legacy.round(input, decimals)

def erfinv(input):
    return legacy.erfinv(input)

def erf(input):
    return legacy.erf(input)

def pow_scalar_tensor(input, exponent):
    return legacy.pow(input, exponent)

def inplace_add(input, other, alpha):
    if alpha != 1:
        return inplace_copy(input, legacy.add(input, legacy.mul(other, alpha)))
    return inplace_copy(input, legacy.add(input, other))

def clamp_scalar(value, min_value, max_value):
    if min_value is not None:
        value = legacy.maximum(value, min_value)
    if max_value is not None:
        value = legacy.minimum(value, max_value)
    return value

def constant_pad_nd(input, pad, value):
    return legacy.pad_v3(input, pad, value, 'constant', True)

def randn(size, generator, dtype):
    return cast(legacy.standard_normal(tuple(size), 0, 0), dtype)

def rand(size, generator, dtype):
    return cast(legacy.uniform_real(tuple(size), 0, 0), dtype)

def tril(input, diagonal):
    return legacy.tril(input, diagonal)

def dense(input, weight, bias=None):
    return legacy.dense(input, weight, bias)

def relu(input):
    return legacy.re_lu(input)

def assign(input, value):
    return inplace_copy(input, value)

def square(input):
    return legacy.square(input)

def log(input):
    if not input.dtype.is_floating_point:
        input = cast(input, mindspore.float32)
    return legacy.log(input)

def permute(input, dims):
    return legacy.transpose(input, dims)

def ones_like(input, dtype):
    if dtype is not None:
        return cast(legacy.ones_like(input), dtype)
    return legacy.ones_like(input)

def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq):
    return cast(legacy.gather(weight, input, 0, 0), weight.dtype)

def linspace(start, end, steps, dtype):
    start = float(start)
    end = float(end)
    return legacy.lin_space(mindspore.Tensor(start), mindspore.Tensor(end), steps)

def masked_fill(input, mask, value):
    if input.dtype.is_floating_point and isinstance(value, numbers.Number):
        value = float(value)
    return legacy.masked_fill(input, mask, value)

def sum(input, dim, keepdim, dtype):
    if dim is None:
        dim = ()
    if input.dtype == mindspore.bool_:
        input = cast(input, mindspore.int64)
    if dtype is None:
        return legacy.reduce_sum(input, dim, keepdim, False)
    return legacy.reduce_sum(input.astype(dtype), dim, keepdim, False)

def conv2d(input, weight, bias=None, stride=1, padding='valid', dilation=1, groups=1):
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, (tuple, list)):
        pad = (padding[0], padding[0], padding[1], padding[1])
    elif isinstance(padding, int):
        pad = (padding,) * 4
    if not isinstance(padding, (int, tuple, list)):
        pad_mode = padding
        pad = (0,) * 4
    
    if isinstance(stride, int):
        stride = (stride,) * 4

    out_channels = weight.shape[0]
    kernel_size = weight.shape[2:]

    output = legacy.conv2_d(
        input, weight,
        out_channels,
        kernel_size,
        1,#mode=1,
        pad_mode, #pad_mode=pad_mode,
        pad, #pad=pad,
        tuple(stride), #stride=tuple(stride),
        dilation, #dilation=dilation,
        groups, #group=groups,
        "NCHW", #data_format="NCHW"
    )
    if bias is not None:
        output = legacy.bias_add(output, bias, "NCHW")
    return output

def conv2d_padding(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return conv2d(input, weight, bias, stride, padding, dilation, groups)

def pow_tensor_scalar(input, scalar):
    return legacy.pow(input, scalar)

def rsqrt(input):
    return legacy.rsqrt(input)

def layer_norm(input, normalized_shape, weight, bias, eps=1e-5):
    if weight is not None:
        begin_axis = input.ndim - weight.ndim
    else:
        begin_axis = -1
    return legacy.layer_norm(input, weight, bias, begin_axis, begin_axis, eps)

def argmin_with_value(input, axis, keep_dims):
    return legacy.arg_min_with_value(input, axis, keep_dims)

def argmax_with_value(input, axis, keep_dims):
    return legacy.arg_max_with_value(input, axis, keep_dims)

def silu(input):
    return legacy.mul(input, legacy.sigmoid(input))

def less_equal(input_x, input_y):
    return legacy.less_equal(input_x, input_y)

def not_equal(input_x, input_y):
    return legacy.not_equal(input_x, input_y)


def logical_not(input):
    return legacy.logical_not(input)

def tensor_scatter_update(input, indices, updates):
    return legacy.tensor_scatter_update(input, indices, updates)

def isinf(input):
    return legacy.is_inf(input)

def gelu(input, approximate):
    if approximate == 'none':
        return mul(mul(input, 0.5), add(erf(div(input, math.sqrt(2.0))), 1.0))
    return legacy.ge_lu(input)

def greater(input_x, input_y):
    return legacy.greater(input_x, input_y)

def greater_equal(input_x, input_y):
    return legacy.greater_equal(input_x, input_y)

def eye(n, m, dtype):
    return legacy.eye(n, m, dtype)

def argmax(input, axis, keep_dims):
    return legacy.arg_max_with_value(input, axis, keep_dims)[0]

def argmin(input, axis, keep_dims):
    return legacy.arg_min_with_value(input, axis, keep_dims)[0]

def exp(input):
    return legacy.exp(input)

def split_with_size(tensor, split_sizes, dim=0):
    chunks = []
    start = 0
    for chunk_size in split_sizes:
        end = start + chunk_size
        slice_obj = [py_slice(None)] * tensor.dim()
        slice_obj[dim] = py_slice(start, end)
        chunks.append(tensor[tuple(slice_obj)])
        start = end

    return tuple(chunks)


def cos(input):
    return legacy.cos(input)

def sigmoid(input):
    return legacy.sigmoid(input)

def sqrt(input):
    return legacy.sqrt(input)

def chunk(input, chunks, dim=0):
    return legacy.split(input, dim, chunks)

def sin(input):
    return legacy.sin(input)

def neg(input):
    return legacy.neg(input)

def bitwise_or_tensor(input_x, input_y):
    return legacy.bitwise_or(input_x, input_y)

def bitwise_and_tensor(input_x, input_y):
    return legacy.bitwise_and(input_x, input_y)

def non_zero_ext(input):
    out = legacy.non_zero(input)
    return unbind(out, 1, out.shape[1])

def unbind(input, dim, num):
    return legacy.unstack(input, dim, num)

def log1p(input):
    return legacy.log1p(input)

def log_softmax(input, axis, dtype):
    if dtype is not None:
        input = input.astype(dtype)
    return legacy.log_softmax(input, axis)

def scatter(input, dim, index, src):
    return legacy.tensor_scatter_elements(input, index, src, dim, "none")

def batch_norm(input, weight, bias, running_mean=None, runnning_var=None, training=False, momentum=0.1, epsilon=1e-5):
    input_ndim = input.ndim
    if input_ndim == 2:
        return legacy.batch_norm(input, weight, bias, running_mean, runnning_var, training, epsilon, momentum, 'NCHW')
    else:
        input = transpose_view(input, 1, -1)
        input_shape = input.shape
        input = reshape(input, (-1, input.shape[-1]))
        outs = legacy.batch_norm(input, weight, bias, running_mean, runnning_var, training, epsilon, momentum, 'NCHW')
        out = reshape(outs[0], (*input_shape[:-1], -1))
        out = transpose_view(out, 1, -1)

        return out, outs[1], outs[2]

def tanh(input):
    return legacy.tanh(input)

def dropout(input, p, seed, offset):
    return legacy.dropout(input, 1-p, 0, 0)

def split_tensor(input, split_size_or_sections, dim):
    if isinstance(split_size_or_sections, int):
        num = input.shape[dim] // split_size_or_sections
        return legacy.split(input, dim, num)

def bmm(input_x, input_y):
    return legacy.batch_mat_mul(input_x, input_y, False, False)

def nllloss(input, target, weight, reduction, ingore_index):
    return legacy.nll_loss(input, target, weight, reduction, ingore_index)

def nllloss_2d(input, target, weight, reduction, ingore_index):
    input = reshape(transpose_view(input, 1, -1), (-1, input.shape[1]))
    target = reshape(target, (-1,))
    out = legacy.nll_loss(input, target, weight, reduction, ingore_index)
    return out


def binary_cross_entropy_with_logits(input, target, weight, posWeight, reduction):
    return legacy.bce_with_logits_loss(input, target, weight, posWeight, reduction)

def std(input, dim, correction, keepdim):
    if dim is None:
        dim = ()
    return legacy.reduce_std(input, dim, bool(correction), keepdim)[0]

def linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    return legacy.lp_norm(x, dim, int(ord), keepdim, 1e-12)

def rfft(input, n=None, dim=-1, norm=None):
    if input.shape[dim] < n:
        pad_inf = (0, n - input.shape[dim])
        pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
        input = constant_pad_nd(input, pad_dims, 0.)
    else:
        input = narrow(input, dim, 0, n)
    return legacy.fft_with_size(input, input.ndim, False, True, norm, True, ())

def narrow(input, dim, start, length):
    begin = [0] * input.ndim
    size = [i for i in input.shape]
    begin[dim] = start
    size[dim] = length
    return legacy.slice(input, begin, size)

def conj(input):
    return legacy.conj(input)

def irfft(input, n, dim, norm):
    if input.shape[dim] < n:
        pad_inf = (0, n - input.shape[dim])
        pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
        input = constant_pad_nd(input, pad_dims, 0.)
    else:
        input = narrow(input, dim, 0, n)
    return legacy.fft_with_size(input, input.ndim, True, True, norm, True, ())

def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    if isinstance(padding, int):
        padding = (0, 0, 0, 0, padding, padding)
    elif isinstance(padding, tuple):
        if len(padding) != 1:
            raise ValueError("For avg_pool1d, padding should be int or tuple of length 1.")
        padding = (0, 0, 0, 0, padding[0], padding[1])
    else:
        raise TypeError("For avg_pool1d, padding should be int or tuple of length 1.")

    if isinstance(stride, tuple):
        if len(stride) != 1:
            raise ValueError("For avg_pool1d, stride should be int or tuple of length 1.")
        stride = stride[0]

    input = expand_dims(input, 2)
    input = expand_dims(input, 2)
    input = legacy.avg_pool3_d(input, (1, 1, kernel_size), (1, 1, stride), 'pad', padding, ceil_mode, count_include_pad, 0, 'NCDHW')
    input = squeeze(input, (2, 3))
    return input

def fmod_scalar(input, other):
    return legacy.floor_mod(input, other)

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, tuple):
        pad = (0, 0, padding[0], padding[0])
    elif isinstance(padding, int):
        pad = (0, 0) + (padding,) * 2
    if not isinstance(padding, (int, tuple)):
        pad_mode = padding
        pad = (0,) * 4

    input = expand_dims(input, 2)
    weight = expand_dims(weight, 2)

    output = legacy.conv2_d(
        input, weight,
        weight.shape[0],
        (1, weight.shape[-1]),
        1,#mode=1,
        pad_mode, #pad_mode=pad_mode,
        pad, #pad=pad,
        (1, stride) if isinstance(stride, int) else (1, *stride), #stride=tuple(stride),
        (1, dilation) if isinstance(dilation, int) else (1, *dilation), #dilation=dilation,
        groups, #group=groups,
        "NCHW", #data_format="NCHW"
    )


    if bias is not None:
        output = legacy.bias_add(output, bias, "NCHW")

    output = squeeze(output, 2)
    return output

def maximum(input, other):
    return legacy.maximum(input, other)

def prod(input, axis, keepdims, dtype):
    if axis is None:
        axis = ()
    return legacy.reduce_prod(input, axis, keepdims)

def mse_loss(input, target, reduction):
    x = square(input - target)
    average_flag = True
    reduce_flag = True
    if reduction == 'sum':
        average_flag = False
    if reduction == 'none':
        reduce_flag = False

    if reduce_flag and average_flag:
        x = mean(x, tuple(range(x.ndim)), False, None)

    if reduce_flag and not average_flag:
        x = sum(x, tuple(range(x.ndim)), False, None)

    return x

def adaptive_avg_pool2d(input, output_size):
    return legacy.adaptive_avg_pool2_d(input, output_size)

def avg_pool2d(input, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if isinstance(padding, int):
        padding = (0, 0, padding, padding, padding, padding)
    elif isinstance(padding, tuple):
        if len(padding) != 1:
            raise ValueError("For avg_pool1d, padding should be int or tuple of length 1.")
        padding = (0, 0, padding[0], padding[1], padding[2], padding[3])
    else:
        raise TypeError("For avg_pool1d, padding should be int or tuple of length 1.")

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    input = expand_dims(input, 2)
    input = legacy.avg_pool3_d(input, (1, *kernel_size), (1, *stride), 'pad', padding, ceil_mode, count_include_pad, 0, 'NCDHW')
    input = squeeze(input, 2)
    return input

def bitwise_or_scalar(input, value):
    return legacy.bitwise_or(input, value)

def floor_div(input, other):
    return legacy.floor_div(input, other)

def minimum(input, other):
    return legacy.minimum(input, other)

def reverse_v2(input, axis):
    if isinstance(axis, int):
        axis = (axis,)
    return legacy.reverse_v2(input, axis)

def divmod(input, other, rounding_mode):
    if rounding_mode == 'floor':
        return legacy.floor_div(input, other)
    elif rounding_mode == 'trunc':
        if isinstance(input, numbers.Number):
            input = mindspore.Tensor(input)
        return legacy.truncate_div(input, other)
    else:
        raise ValueError(f'Invalid rounding mode: {rounding_mode}')

def pow(input, exponent):
    return legacy.pow(input, exponent)


def bitwise_and_scalar(input, value):
    return legacy.bitwise_and(input, value)

def rand_like(input, generator, dtype):
    return rand(input.shape, generator, dtype)

def bincount(input, weights=None, minlength=0):
    if weights is None:
        weights =  mindspore.Tensor(1, dtype=mindspore.int32)
    return legacy.bincount(cast(input, mindspore.int32),
                           mindspore.Tensor(minlength, dtype=mindspore.int32),
                           weights)

def lgamma(input):
    return legacy.lgamma(input)

def _deconv_output_length(pad_mode, filter_size, stride_size, dilation_size, padding):
    """Calculate the width and height of output."""
    length = 0
    filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
    if pad_mode == 'valid':
        if filter_size - stride_size > 0:
            length = filter_size - stride_size
    elif pad_mode == 'pad':
        length = - padding + filter_size - stride_size

    return length


def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, tuple):
        pad = (0, 0, padding[0], padding[0])
    elif isinstance(padding, int):
        pad = (0, 0) + (padding,) * 2
    if not isinstance(padding, (int, tuple)):
        pad_mode = padding
        pad = (0,) * 4

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    in_channel, out_channels = weight.shape[0], weight.shape[1] * groups
    kernel_size = weight.shape[2:]

    n, _, h, w = input.shape
    h_add = _deconv_output_length(pad_mode, kernel_size[0], stride[0], dilation[0], pad[0] + pad[1])
    w_add = _deconv_output_length(pad_mode, kernel_size[1], stride[1], dilation[1], pad[2] + pad[3])

    out = legacy.conv2_d_transpose(
        input, weight,
        (n, out_channels, h * stride[0] + h_add, w * stride[1] + w_add),
        out_channels,
        kernel_size,
        pad_mode,
        pad,
        None,
        1,
        stride,
        dilation,
        groups,
        'NCHW'
    )
    if bias is not None:
        out = legacy.bias_add(out, bias, 'NCHW')
    return out

def expm1(x):
    return legacy.expm1(x)

py_min = min
def min(input):
    return legacy.reduce_min(input, (), False)

def acos(x):
    return legacy.a_cos(x)

def upsample_bilinear2d(input, size=None, scale_factor=None, align_corners=False):
    return legacy.resize_bilinear_v2(input, size, align_corners, not align_corners)

def unstack_view(input, dim):
    return legacy.unstack(input, dim, input.shape[dim])

def triu(input, diagonal=0):
    return legacy.triu(input, diagonal)

def masked_scatter(input, mask, value):
    return legacy.masked_scatter(input, mask, value)

def max_pool2d(input, kernel_size, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    out, indices = legacy.max_pool_with_argmax_v2(input, kernel_size, stride, padding, dilation, ceil_mode, mindspore.int64)

    if return_indices:
        return out, indices
    return out

def baddbmm(input, batch1, batch2, alpha=1, beta=1):
    return add(mul(beta, input), mul(alpha, bmm(batch1, batch2)))

def inplace_fill_tensor(input, value):
    out = np.full_like(input.numpy(), value)
    numpy_to_tensor_overwrite(out, input)
    return input

def softplus(input, beta=1, threshold=20):
    return legacy.softplus(input)

def gather_nd(input, indices):
    return legacy.gather_nd(input, indices)

def unique_consecutive(input, return_inverse, return_counts, dim):
    return legacy.unique_consecutive(input, return_inverse, return_counts, dim)

def meshgrid(input, lambd):
    return legacy.meshgrid(input, lambd)

def addcmul(input, tensor1, tensor2, value=1.0):
    return legacy.addcmul(input, tensor1, tensor2, mindspore.Tensor(value))

def addmm(input, mat1, mat2, alpha=1.0, beta=1.0):
    return add(mul(beta, input), mul(alpha, bmm(mat1, mat2)))

def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    out = legacy.im2_col(input, kernel_size, stride, dilation, padding)
    out_shape = out.shape[:1] + (-1,) + out.shape[-1:]
    out = reshape(out, out_shape)
    return out

def floor(input):
    return legacy.floor(input)

def upsample_nearest2d(input, output_size, scale_factors):
    if output_size is None:
        tuple_len = py_min(len(input.shape) - 2, len(scale_factors))
        output_size = tuple([math.floor(input.shape[i + 2] * scale_factors[i])
                        for i in range(tuple_len)])

    return legacy.resize_nearest_neighbor(input, output_size, False, False)

def upsample_bicubic2d(input, size=None, scale_factor=None, align_corners=False):
    return legacy.resize_bicubic(input, size, align_corners, not align_corners)

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    pad_mode = 'pad'
    pad = padding
    if isinstance(padding, (tuple, list)):
        pad = (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    elif isinstance(padding, int):
        pad = (padding,) * 6
    if not isinstance(padding, (int, tuple, list)):
        pad_mode = padding
        pad = (0,) * 6

    out_channels = weight.shape[0]
    kernel_size = weight.shape[2:]

    output = legacy.conv3_d(input, weight,
                            out_channels,
                            kernel_size,
                            1,
                            pad_mode,
                            pad,
                            tuple(stride),
                            dilation,
                            groups,
                            "NCDHW")
                            
    if bias is not None:
        output = legacy.bias_add(output, bias, 'NCHW')
    return output


    return legacy.conv3_d(input, weight, bias, stride, padding, dilation, groups)

def normal_float_float(mean, std, size, dtype, generator):
    out = np.random.normal(mean, std, size).astype(mindtorch.dtype2np[dtype])
    out = mindspore.Tensor(out)
    return out

def normal_tensor_tensor(mean, std, size, dtype, generator):
    out = np.random.normal(mean.item(), std.item(), size).astype(mindtorch.dtype2np[dtype])
    out = mindspore.Tensor(out)
    return out

def inplace_relu(input):
    return legacy.assign(input, legacy.re_lu(input))

def adaptive_avg_pool1d(input, output_size):
    x_in_shape = input.shape
    width = x_in_shape[2]
    stride = width // output_size
    kernel_size = width - (output_size - 1) * stride
    stride = (1, width // output_size)
    kernel_size = (1, kernel_size)
    input = expand_dims(input, 2)
    input = legacy.avg_pool(input, kernel_size, stride, "VALID", "NCHW")
    input = squeeze(input, 2)
    return input

def remainder_tensor_scalar(input, other):
    out = sub(input, mul(floor_div(input, other), other), 1)
    return out

def outer(input, other):
    input = reshape(input, (-1, 1))
    y = mul(input, other)
    return y

def view_as_complex(input):
    real_part, imag_part = chunk(input, 2, -1)
    return legacy.complex(squeeze(real_part, -1), squeeze(imag_part, -1))

def cdist(x1, x2, p):
    return legacy.cdist(x1, x2, float(p))

def prelu(input, weight):
    return legacy.p_re_lu(input, weight)

def reciprocal(input):
    return legacy.reciprocal(input)

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity):
    loss, log_alpha = legacy.ctc_loss_v2(log_probs, targets, input_lengths, target_lengths, blank, 'none', zero_infinity)
    if reduction == 'sum':
        loss = sum(loss, (), False, None)
    if reduction == 'mean':
        # input_type = loss.dtype
        # target_length_t = target_lengths.clip(1., None)
        # loss = loss.astype("float32")
        loss = div(loss, target_lengths)
        loss = mean(loss, (), False, None)
        # loss = loss.astype(input_type)
    return (loss, log_alpha)

def glu(input, dim=-1):
    return legacy.glu(input, dim)

def one_hot(tensor, num_classes):
    on_value = mindspore.Tensor(1, dtype=tensor.dtype)
    off_value = mindspore.Tensor(0, dtype=tensor.dtype)
    return legacy.one_hot(tensor, num_classes, on_value, off_value, -1)

def polar(abs, angle):
    return legacy.polar(abs, angle)

def scatter_value(input, dim, index, src, reduce='none'):
    if isinstance(src, numbers.Number):
        src = fill_scalar(index.shape, src, dtype=input.dtype)
    return legacy.tensor_scatter_elements(input, index, src, dim, reduce)

def pixel_shuffle(input, upscale_factor):
    idx = input.shape
    length = input.ndim
    pre = idx[:-3]
    c, h, w = idx[-3:]
    c = c // upscale_factor ** 2
    input_perm = pre + (c, upscale_factor, upscale_factor, h, w)
    input = reshape(input, input_perm)
    input_perm = [i for i in range(length - 2)]
    input_perm = input_perm + [length, length - 2, length + 1, length - 1]
    input_perm = tuple(input_perm)
    input = permute(input, input_perm)
    input = reshape(input, (pre + (c, upscale_factor * h, upscale_factor * w)))
    return input

def rms_norm(input, weight, eps=1e-5):
    input_dtype = input.dtype
    input = cast(input, mindspore.float32)
    variance = mean(pow(input, 2), -1, True, None)
    input = mul(input, rsqrt(add(variance, eps, 1)))
    return mul(weight, cast(input, input_dtype))

def count_nonzero(input, dims):
    return legacy.count_non_zero(input, dims)

def index_add_ext(input, dim, index, source, alpha):
    if alpha != 1:
        source = mul(alpha, source)
    return legacy.index_add(input, cast(index, mindspore.int32), source, dim, True, True)

def real(input):
    return legacy.real(input)

def upsample_linear1d(input, output_size, scale_factor, align_corners=False):
    coordinate_transformation_mode = "align_corners" if align_corners else "half_pixel"
    return legacy.resize_linear1_d(input, output_size, coordinate_transformation_mode)

def imag(input):
    return legacy.imag(input)

def bitwise_xor_tensor(input, other):
    return legacy.bitwise_xor(input, other)

def grid_sampler_2d(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    return legacy.grid_sampler2_d(input, grid, mode, padding_mode, align_corners)

def l1_loss(input, target, reduction='mean'):
    loss = abs(sub(input, target))
    if reduction == 'mean':
        return mean(loss, (), False, False)
    elif reduction == 'sum':
        return sum(loss, (), False, False)
    return loss

def leaky_relu(input, negative_slope):
    select_op = maximum
    if negative_slope > 1:
        select_op = minimum
    return select_op(mul(input, negative_slope), input)

def ceil(input):
    return legacy.ceil(input)

def reduce_max(input, axis, keepdims):
    return legacy.reduce_max(input, axis, keepdims)

def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return legacy.nan_to_num(input, nan, posinf, neginf)

def elu(input, alpha):
    return legacy.elu(input, alpha)

def sign(input):
    return legacy.sign(input)

def inplace_fill_diagonal(input, fill_value, wrap):
    inplace_copy(input, legacy.fill_diagonal(input, float(fill_value), wrap))
    return input

def clamp_tensor(value, min_value, max_value):
    if min_value is not None:
        value = legacy.maximum(value, min_value)
    if max_value is not None:
        value = legacy.minimum(value, max_value)
    return value

def lstm(input, h, c, w, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout, proj_size):
    return legacy.lstm(input, h, c, w, input_size, hidden_size, num_layers, has_bias, bidirectional, dropout, proj_size)

def var(input, dim=None, correction=1, keepdim=False):
    if dim is None:
        input_mean = mean(input, (), False, None)
    else:
        input_mean = mean(input, dim=dim, keepdim=True, dtype=None)
    
    # 计算与均值的平方差
    squared_diff = pow(sub(input, input_mean, 1), 2) 
    # 计算方差
    if dim is None:
        variance = mean(squared_diff, (), False, None)
        n = input.numel()  # 总元素个数
    else:
        variance = mean(squared_diff, dim=dim, keepdim=keepdim, dtype=None)
        n = input.size(dim)  # 指定维度的元素个数
    
    # 无偏估计校正
    if correction and n > 1:
        variance = mul(variance, (n / (n - 1)))
    
    return variance

def log2(input):
    return div(log(input), math.log(2))

def bucketize(input, boundaries, right=False):
    epsilon_ = 0. if right else 1.e-6
    boundaries = [boundary + epsilon_ for boundary in boundaries]
    return legacy.bucketize(input, boundaries)

def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return legacy.col2_im(input, output_size, kernel_size, dilation, padding, stride)

def randperm(n, generator, dtype):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return legacy.randperm_v2(n, seed, offset, dtype)

def gamma(shape, alpha, beta):
    out = np.random.gamma(alpha, 1/beta, shape)
    return mindtorch.Tensor.from_numpy(out)

def logical_or(input_x, input_y):
    return legacy.logical_or(input_x, input_y)

def hswish(input):
    return legacy.h_swish(input)

def logical_and(input_x, input_y):
    return legacy.logical_and(input_x, input_y)

def logsigmoid(input):
    output = sigmoid(input)
    ret = log(output)
    return ret

def as_strided(input, size, stride, storage_offset):
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
    out = gather(reshape(input, (-1,)), input_indices, 0, 0)
    return out

def dropout2d(input_x, p):
    return legacy.dropout2_d(input_x, p)

def linalg_qr(input_x, mode):
    full_matrices = 'mode' == 'complete'
    return legacy.qr(input_x, full_matrices)

def diag(input, diagonal):
    out = np.diag(input.numpy(), diagonal)
    return mindtorch.Tensor.from_numpy(out)

def logit(input, eps=1e-5):
    return legacy.logit(input, eps)

def relu6(input):
    return legacy.re_lu6(input)

def logsumexp(input, dim, keepdim=False):
    input_max = legacy.reduce_max(input, dim, keepdim)
    input_exp = exp(sub(input, input_max))
    input_sumexp = sum(input_exp, dim, keepdim, None)
    input_logsumexp = log(input_sumexp)
    if not keepdim:
        input_max = squeeze(input_max, dim)
    return add(input_logsumexp, input_max)

def bernoulli(input, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return legacy.bernoulli(input, 0.5, seed.item(), offset.item())

def right_shift(input, other):
    return legacy.right_shift(input, other)

def histc(input, bins=100, min=0, max=0):
    return legacy.histogram(input, bins, float(min), float(max))

def search_sorted(sorted_sequence, values, sorter, dtype, right):
    return legacy.search_sorted(sorted_sequence, values, sorter, dtype, right)

def scatter_nd_update(input, indices, updates):
    return legacy.scatter_nd_update(input, indices, updates, True)

def triu_indices(row, col, offset, dtype):
    return legacy.triu_indices(row, col, offset, dtype)
