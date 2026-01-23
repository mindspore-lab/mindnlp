import ctypes
import numbers
import math
import numpy as np
import mindspore
import mindtorch
from .._op_prim.cpu import legacy, pyboost

# Lazy import for Function to avoid circular import
_Function = None

def _get_function_class():
    """Lazy import of Function class to avoid circular import."""
    global _Function
    if _Function is None:
        from ..autograd.function import Function
        _Function = Function
    return _Function

# Custom Getitem Function with NumPy-based backward
class GetitemFunction:
    """Custom getitem with NumPy-based backward to avoid MindSpore InplaceCopy issues."""

    _instance = None

    @classmethod
    def apply(cls, input_tensor, slice_spec):
        """Apply getitem with custom backward."""
        Function = _get_function_class()

        # Create a dynamic class that inherits from Function
        class _GetitemOp(Function):
            @staticmethod
            def forward(ctx, input_tensor, slice_spec):
                """Forward pass: perform getitem using NumPy."""
                # Convert to numpy for slicing
                input_np = input_tensor.asnumpy()

                # Store info for backward
                ctx.save_for_backward(input_tensor)
                ctx.slice_spec = slice_spec
                ctx.input_shape = tuple(input_tensor.shape)
                ctx.input_dtype = input_tensor.dtype

                # Perform slicing in numpy
                result_np = input_np[slice_spec]

                # Handle scalar results (numpy scalar types)
                if not isinstance(result_np, np.ndarray):
                    # Convert numpy scalar to Python scalar, then to tensor
                    if hasattr(result_np, 'item'):
                        result_np = result_np.item()
                    result = mindtorch.tensor(result_np, dtype=input_tensor.dtype)
                elif result_np.ndim == 0:
                    # 0-dimensional array - convert to scalar first
                    result = mindtorch.tensor(result_np.item(), dtype=input_tensor.dtype)
                else:
                    # Regular array result
                    result = mindtorch.tensor(result_np, dtype=input_tensor.dtype)

                # Don't copy init attribute - the new tensor already has its own init
                # Copying init causes shape mismatch between wrapper and underlying tensor
                return result

            @staticmethod
            def backward(ctx, grad_output):
                """Backward pass: scatter gradients using NumPy."""
                input_shape = ctx.input_shape
                input_dtype = ctx.input_dtype
                slice_spec = ctx.slice_spec

                # Create zero gradient tensor with input shape
                grad_input_np = np.zeros(input_shape, dtype=mindtorch.dtype2np[input_dtype])

                # Get gradient as numpy
                grad_output_np = grad_output.asnumpy()

                # Scatter gradient to the sliced positions
                # Use numpy's advanced indexing for assignment
                grad_input_np[slice_spec] = grad_output_np

                # Convert back to mindtorch tensor
                grad_input = mindtorch.tensor(grad_input_np, dtype=input_dtype)
                # Don't copy init attribute - the new tensor already has its own init

                return grad_input, None  # None for slice_spec gradient

        return _GetitemOp.apply(input_tensor, slice_spec)

def empty(size, dtype):
    return pyboost.empty_op(size, dtype=dtype, device='CPU')

def empty_like(input, dtype):
    return pyboost.empty_like_op(input, dtype, device='CPU')

def new_empty(input, size, dtype, device):
    """
    Create a new empty tensor with the same type as input tensor.
    
    Args:
        input: The input tensor to base the type on
        size: The size of the new tensor
        dtype: Optional dtype (if None, use input's dtype)
        device: Optional device (if None, use input's device or 'CPU')
    
    Returns:
        A new empty tensor with the specified size and dtype/device
    """
    # Use input's dtype if dtype is None
    if dtype is None:
        dtype = input.dtype
    
    # Use input's device if device is None, or default to 'CPU'
    if device is None:
        # Try to get device from input, default to 'CPU'
        if hasattr(input, 'device') and hasattr(input.device, 'type'):
            device_str = input.device.type
            if device_str.lower() == 'cpu':
                device_str = 'CPU'
        else:
            device_str = 'CPU'
    else:
        # Convert device to string if it's a device object
        if hasattr(device, 'type'):
            device_str = device.type
            if device_str.lower() == 'cpu':
                device_str = 'CPU'
        else:
            device_str = str(device)
            if device_str.lower() == 'cpu':
                device_str = 'CPU'
    
    # Create empty tensor using pyboost
    return pyboost.empty_op(size, dtype=dtype, device=device_str)

def inplace_normal(input, mean, std, generator_):
    out = np.random.normal(mean, std, input.shape).astype(mindtorch.dtype2np[input.dtype])
    numpy_to_tensor_overwrite(out, input)

    return input

def select_ext_view(input, dim, index):
    return pyboost.select_ext_view_op(input, dim, index)

def inplace_copy(input, value):
    # return pyboost.inplace_copy_op(input, value)
    # Preserve init attribute to maintain device information
    init_attr = getattr(input, 'init', None)
    input.data = value
    if init_attr is not None:
        input.init = init_attr
    return input

def inplace_sub(input, other):
    return inplace_copy(input, legacy.sub(input, other))


def raw_sgd(param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat):
    """SGD optimizer step for CPU."""
    return legacy.sgd(param, grad, lr, accum, momentum, stat, dampening, weight_decay, nesterov)

def inplace_index_add(input, dim, index, source, alpha):
    """Inplace index add operation."""
    if alpha != 1:
        source = mul(alpha, source)
    out = legacy.index_add(input, cast(index, mindspore.int32), source, dim, True, True)
    inplace_copy(input, out)
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

def full_like(input, fill_value, dtype=None):
    if dtype is None:
        dtype = input.dtype
    size = input.shape
    if isinstance(fill_value, numbers.Number):
        return fill_scalar(size, fill_value, dtype)
    else:
        return fill_tensor(size, fill_value, dtype)

def tensor_shape(input):
    return legacy.tensor_shape(input)

def arange(start, end, step, dtype=mindspore.int64):
    return mindtorch.Tensor.from_numpy(np.arange(start, end, step, mindtorch.dtype2np[dtype]))

def broadcast_to(input, shape):
    # Handle scalar values by converting them to tensors first
    if not isinstance(input, mindtorch.Tensor):
        input = mindtorch.tensor(input)
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

py_abs = abs
def abs(input):
    return legacy.abs(input)

def identity(input):
    return legacy.identity(input)

class CloneFunction:
    """Custom clone with NumPy-based backward to avoid Clone kernel unregistered error."""

    @classmethod
    def apply(cls, input_tensor):
        """Apply clone with custom backward."""
        Function = _get_function_class()

        # Create a dynamic class that inherits from Function
        class _CloneOp(Function):
            @staticmethod
            def forward(ctx, input_tensor):
                """Forward pass: create a copy using NumPy to avoid MindSpore Clone kernel."""
                # Use numpy copy to completely bypass MindSpore kernel system
                input_np = input_tensor.asnumpy()
                out_np = np.copy(input_np)
                result = mindspore.Tensor.from_numpy(out_np)
                return result

            @staticmethod
            def backward(ctx, grad_output):
                """Backward pass: clone's gradient is simply passed through."""
                # Clone's backward simply passes through the gradient
                # Use numpy to avoid Clone kernel
                if ctx.needs_input_grad[0]:
                    grad_np = grad_output.asnumpy()
                    if not isinstance(grad_np, np.ndarray):
                        grad_np = np.array(grad_np)
                    grad_input = mindspore.Tensor.from_numpy(grad_np)
                    return grad_input
                return None

        return _CloneOp.apply(input_tensor)

def clone(input):
    return CloneFunction.apply(input)

py_max = max
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
    return pyboost.matmul_ext_op(self, other)

def div(input, other):
    return legacy.div(input, other)

def mul(input, other):
    return legacy.mul(input, other)

def inplace_mul(input, other):
    return inplace_copy(input, mul(input, other))

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
    return pyboost.slice_ext_view_op(self, dim, start, end, step)

def pad_v3(input, new_pad, mode, value=None, contiguous=True):
    return legacy.pad_v3(input, new_pad, value, mode, contiguous)

def cumsum(self, dim, dtype):
    if self.shape[dim] == 0:
        return mindspore.tensor([], dtype=self.dtype)
    # MindSpore's CumSum doesn't support bool, cast to int32 first
    if self.dtype == mindspore.bool_:
        result = legacy.cum_sum(cast(self, mindspore.int32), dim, False, False)
        if dtype is not None:
            return cast(result, dtype)
        return result
    if self.dtype == mindspore.int64:
        return cast(legacy.cum_sum(cast(self, mindspore.int32), dim, False, False), mindspore.int64)
    return legacy.cum_sum(self, dim, False, False)

def reduce_any(input, axis, keepdims):
    # When axis is None, reduce over all dimensions
    if axis is None:
        axis = tuple(range(input.ndim))
    return legacy.reduce_any(input, axis, keepdims)

def concat(tensors, axis):
    # Cast all tensors to the same dtype (use the first tensor's dtype)
    if len(tensors) > 1:
        first_dtype = tensors[0].dtype
        tensors = tuple(t.to(first_dtype) if t.dtype != first_dtype else t for t in tensors)
    return legacy.concat(tensors, axis)

def numpy_to_tensor_overwrite(np_array, tensor):
    if not np_array.flags.c_contiguous:
        np_array = np.ascontiguousarray(np_array)

    tensor_ptr = tensor.data_ptr()
        
    ctypes.memmove(tensor_ptr, np_array.ctypes.data, tensor.nbytes)
    
    return tensor

def t2t_overwrite(input, other):
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

def inplace_masked_fill(input, mask, value):
    return inplace_copy(input, masked_fill(input, mask, value))

py_sum = sum
def sum(input, dim, keepdim, dtype):
    if dim is None:
        dim = ()
    if input.dtype == mindspore.bool_:
        input = cast(input, mindspore.int64)
    if dtype is None:
        return legacy.reduce_sum(input, dim, keepdim, False)
    return legacy.reduce_sum(input.astype(dtype), dim, keepdim, False)

def conv2d(input, weight, bias=None, stride=1, padding='valid', dilation=1, groups=1, training=True):
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

def conv2d_padding(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
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
    return legacy.layer_norm(input, weight, bias, begin_axis, begin_axis, eps)[0]

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
    # Cast updates to input dtype to avoid type mismatch
    if hasattr(input, 'dtype') and hasattr(updates, 'dtype') and input.dtype != updates.dtype:
        updates = updates.to(input.dtype)
    return legacy.tensor_scatter_update(input, indices, updates)

def isinf(input):
    # IsInf only supports float types, convert integer tensors to float
    if input.dtype in (mindtorch.int8, mindtorch.int16, mindtorch.int32, mindtorch.int64, 
                       mindtorch.uint8, mindtorch.uint16, mindtorch.uint32, mindtorch.uint64):
        # Integer tensors cannot have inf values, return all False
        result = legacy.zeros_like(input)
        return cast(result, mindtorch.bool)
    return legacy.is_inf(input)

def isin(input, test_elements, assume_unique=False, invert=False):
    """
    Checks if elements of input tensor are in test_elements.

    Args:
        input (Tensor): The input tensor.
        test_elements (Tensor): The tensor to test against.
        assume_unique (bool): If True, assumes that test_elements contains unique elements.
        invert (bool): If True, inverts the result.

    Returns:
        Tensor: The tensor with boolean values indicating whether elements are in test_elements.
    """
    input_shape = input.shape
    input = expand_dims(reshape(input, (-1,)), -1)
    if not isinstance(test_elements, numbers.Number):
        test_elements = reshape(test_elements, (-1,))
    included = eq(input, test_elements)
    # ops.reduce_sum only supports float
    res = cast(sum(included, -1, False, None), mindtorch.bool_)
    if invert:
        res = logical_not(res)
    return reshape(res, input_shape)

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
        chunks.append(getitem(tensor, tuple(slice_obj)))
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

def scatter_reduce(input, dim, index, src, reduce, include_self=True):
    """Scatter reduce operation with support for sum, amax, amin, mean, prod."""
    if reduce == 'sum':
        return legacy.tensor_scatter_elements(input, index, src, dim, "add")

    # Use numpy-based implementation for other reduce operations
    input_np = input.asnumpy().copy()
    index_np = index.asnumpy()
    src_np = src.asnumpy()

    # Get shape info
    ndim = input_np.ndim

    # Create index arrays for advanced indexing
    shape = index_np.shape
    indices = [np.arange(s).reshape((1,) * i + (s,) + (1,) * (ndim - i - 1)) for i, s in enumerate(shape)]
    indices = [np.broadcast_to(idx, shape) for idx in indices]
    indices[dim] = index_np

    if reduce == 'amax':
        if include_self:
            np.maximum.at(input_np, tuple(indices), src_np)
        else:
            fill_val = np.finfo(input_np.dtype).min if np.issubdtype(input_np.dtype, np.floating) else np.iinfo(input_np.dtype).min
            input_np.fill(fill_val)
            np.maximum.at(input_np, tuple(indices), src_np)
    elif reduce == 'amin':
        if include_self:
            np.minimum.at(input_np, tuple(indices), src_np)
        else:
            fill_val = np.finfo(input_np.dtype).max if np.issubdtype(input_np.dtype, np.floating) else np.iinfo(input_np.dtype).max
            input_np.fill(fill_val)
            np.minimum.at(input_np, tuple(indices), src_np)
    elif reduce == 'mean':
        count = np.ones_like(input_np) if include_self else np.zeros_like(input_np)
        np.add.at(input_np, tuple(indices), src_np)
        np.add.at(count, tuple(indices), 1)
        input_np = np.divide(input_np, count, where=count > 0, out=input_np)
    elif reduce == 'prod':
        if include_self:
            np.multiply.at(input_np, tuple(indices), src_np)
        else:
            input_np.fill(1)
            np.multiply.at(input_np, tuple(indices), src_np)
    else:
        raise ValueError(f'do not support reduce: {reduce}')

    return mindtorch.tensor(input_np, dtype=input.dtype, device=input.device)

def batch_norm(input, weight, bias, running_mean=None, running_var=None, training=False, momentum=0.1, epsilon=1e-5):
    num_features = input.shape[1]
    # Handle None values - create default tensors when affine=False or track_running_stats=False
    if weight is None:
        weight = mindtorch.ones(num_features, dtype=input.dtype, device=input.device)
    if bias is None:
        bias = mindtorch.zeros(num_features, dtype=input.dtype, device=input.device)
    if running_mean is None:
        running_mean = mindtorch.zeros(num_features, dtype=input.dtype, device=input.device)
    if running_var is None:
        running_var = mindtorch.ones(num_features, dtype=input.dtype, device=input.device)

    input_ndim = input.ndim
    if input_ndim == 2:
        return legacy.batch_norm(input, weight, bias, running_mean, running_var, training, epsilon, momentum, 'NCHW')
    else:
        input = transpose_view(input, 1, -1)
        input_shape = input.shape
        input = reshape(input, (-1, input.shape[-1]))
        outs = legacy.batch_norm(input, weight, bias, running_mean, running_var, training, epsilon, momentum, 'NCHW')
        out = reshape(outs[0], (*input_shape[:-1], -1))
        out = transpose_view(out, 1, -1)

        return out, outs[1], outs[2]

def tanh(input):
    return legacy.tanh(input)

def dropout(input, p, training=True):
    """
    Returns a tensor with dropout applied element-wise.

    Args:
        input (Tensor): The input tensor.
        p (float): The dropout probability.
        seed (int): The random seed.

    Returns:
        Tensor: The tensor with dropout applied.
    """
    if not training or p==0:
        return input
    # MindSpore's dropout returns (output, mask), extract only output
    return legacy.dropout(input, 1-p, 0, 0)[0]

def split_tensor(input, split_size_or_sections, dim):
    # If split_size_or_sections is a list/tuple, use split_with_size
    if isinstance(split_size_or_sections, (list, tuple)):
        return split_with_size(input, split_size_or_sections, dim)
    # Otherwise, it is an integer specifying the chunk size
    num = input.shape[dim] // split_size_or_sections
    # If split_size is larger than dimension size, return the input as a single-element tuple
    if num == 0:
        return (input,)
    return legacy.split(input, dim, num)

def bmm(input_x, input_y):
    return legacy.batch_mat_mul(input_x, input_y, False, False)

def nllloss(input, target, weight, reduction, ingore_index):
    result = legacy.nll_loss(input, target, weight, reduction, ingore_index)
    # MindSpore NLLLoss returns a tuple (loss, total_weight), extract just the loss
    if isinstance(result, tuple):
        return result[0]
    return result

def nllloss_2d(input, target, weight, reduction, ingore_index):
    input = reshape(transpose_view(input, 1, -1), (-1, input.shape[1]))
    target = reshape(target, (-1,))
    result = legacy.nll_loss(input, target, weight, reduction, ingore_index)
    # MindSpore NLLLoss returns a tuple (loss, total_weight), extract just the loss
    if isinstance(result, tuple):
        return result[0]
    return result


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
        padding = (0, 0, 0, 0, padding[0], padding[0])
    else:
        raise TypeError("For avg_pool1d, padding should be int or tuple of length 1.")

    if isinstance(stride, tuple):
        if len(stride) != 1:
            raise ValueError("For avg_pool1d, stride should be int or tuple of length 1.")
        stride = stride[0]

    if input.ndim == 3:
        input = expand_dims(input, 2)
        input = expand_dims(input, 2)
        input = legacy.avg_pool3_d(input, (1, 1, kernel_size[0]), (1, 1, stride), 'pad', padding, ceil_mode, count_include_pad, 0, 'NCDHW')
        input = squeeze(input, (2, 3))
    elif input.ndim == 2:
        input = expand_dims(input, 1)
        input = expand_dims(input, 1)
        input = expand_dims(input, 1)
        input = legacy.avg_pool3_d(input, (1, 1, kernel_size[0]), (1, 1, stride), 'pad', padding, ceil_mode, count_include_pad, 0, 'NCDHW')
        input = squeeze(input, (1, 2, 3))
    return input

def fmod_scalar(input, other):
    return legacy.floor_mod(input, other)

def fmod_tensor(input, other):
    return legacy.floor_mod(input, other)


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
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

def repeat_interleave_int(input_tensor, repeats, dim, output_size):
    if dim is None:
        input_tensor = flatten(input_tensor, 0, -1)
        dim = 0

    # 确保 dim 是有效的维度
    if dim < 0:
        dim += input_tensor.dim()

    # 将 repeats 统一转换为 LongTensor 并确保其在正确的设备上
    if isinstance(repeats, int):
        repeats_tensor = mindspore.tensor([repeats], dtype=mindtorch.long)
        uniform_repeat = True
    elif isinstance(repeats, (list, tuple)):
        repeats_tensor = mindspore.tensor(repeats, dtype=mindtorch.long)
        uniform_repeat = False
    elif isinstance(repeats, mindtorch.Tensor):
        repeats_tensor = cast(repeats, dtype=mindtorch.long)
        uniform_repeat = False
    else:
        raise TypeError("repeats must be an int, a list, or a mindtorch.Tensor")

    # 获取输入张量在目标维度上的大小
    dim_size = input_tensor.shape[dim] 

    if uniform_repeat:
        # ✅ 优化路径：当所有元素重复次数相同时，使用 expand 和 reshape 避免循环
        # 此方法利用广播机制，非常高效
        unsqueezed_tensor = expand_dims(input_tensor, dim + 1)
        expanded_shape = list(input_tensor.shape)
        expanded_shape[dim] = -1
        expanded_shape.insert(dim + 1, repeats_tensor.item())
        expanded_tensor = broadcast_to(unsqueezed_tensor, expanded_shape)
        
        final_shape = list(input_tensor.shape)
        final_shape[dim] *= repeats_tensor.item()
        output = reshape(expanded_tensor, final_shape)
    else:
        # 🔄 当重复次数不同时，需要构建索引
        # 检查 repeats_tensor 的长度是否与目标维度的长度匹配
        if len(repeats_tensor) != dim_size:
            raise ValueError(f"repeats must have length {dim_size} along dimension {dim}, but got {len(repeats_tensor)}")
        
        # 生成索引：例如 repeats_tensor = [2, 3, 1] -> index = [0, 0, 1, 1, 1, 2]
        # 使用 cumsum 计算总重复次数以预分配空间
        total_repeats = sum(repeats_tensor, 0, False, None).item()
        index = zeros(total_repeats, dtype=mindtorch.long)
        
        # 计算每个块的起始位置
        # start_positions = mindtorch.cat([mindtorch.tensor([0], device=input_tensor.device), mindtorch.cumsum(repeats_tensor, dim=0)[:-1]])
        
        # 使用 scatter 或高级索引填充（这里用循环填充，但可考虑更底层的优化）
        # 注意：对于非常大的非均匀重复，此部分可能成为瓶颈
        current_pos = 0
        for i in range(dim_size):
            repeat_count = repeats_tensor[i].item()
            if repeat_count > 0:
                index[current_pos:current_pos + repeat_count] = i
            current_pos += repeat_count

        output = index_select(input_tensor, dim, index)

    return output

def repeat_interleave_tensor(input, repeats, dim, output_size):
    return repeat_interleave_int(input, repeats, dim, output_size)


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    if weight is None:
        weight = ones([input.shape[1]], dtype=input.dtype)
    if bias is None:
        bias = zeros([input.shape[1]], dtype=input.dtype)

    # pyboost.group_norm_op returns (output, mean, var), we only need output
    return pyboost.group_norm_op(input, num_groups, weight, bias, eps)[0]

def repeat_kv(hidden_states, n_rep: int):
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = broadcast_to(expand_dims(hidden_states, 2), (batch, num_key_value_heads, n_rep, slen, head_dim))
    return reshape(hidden_states, (batch, num_key_value_heads * n_rep, slen, head_dim))

def sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False):
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale

    if enable_gqa:
        key = contiguous(repeat_kv(key, query.shape[-3] // key.shape[-3]))
        value = contiguous(repeat_kv(value, query.shape[-3] // value.shape[-3]))

    attn_bias_shape = (L, S) if attn_mask is None else attn_mask.shape
    attn_bias = zeros(attn_bias_shape, dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = tril(ones((L, S), dtype=mindtorch.bool), diagonal=0)
        attn_bias = masked_fill(attn_bias, logical_not(temp_mask), mindtorch.finfo(attn_bias.dtype).min)

    if attn_mask is not None:
        if attn_mask.dtype == mindtorch.bool:
            if attn_mask.ndim == 3:
                attn_mask = squeeze(attn_mask, 0)
            else:
                attn_mask = attn_mask
            attn_bias = masked_fill(attn_bias, logical_not(attn_mask), mindtorch.finfo(attn_bias.dtype).min)
        else:
            attn_bias = add(attn_mask, attn_bias)

    attn_weight = mul(matmul(query, transpose_view(key, -2, -1)), scale_factor)
    attn_weight = add(attn_weight, attn_bias)
    attn_weight = softmax(attn_weight, -1)
    attn_weight = dropout(attn_weight, dropout_p)
    return matmul(attn_weight, value)

def unstack_view(input, dim):
    # return legacy.unstack(input, dim, input.shape[dim])
    return pyboost.unstack_ext_view_op(input, dim)

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

def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))
    return indices, dimension

def unfold(input, dimension, size, step):
    _indices, _dimension = _get_unfold_indices(input.shape, dimension, size, step)
    indices = mindspore.tensor(_indices)
    output = gather(input, indices, _dimension, 0)
    output = transpose_view(output, _dimension + 1, -1)
    return output

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

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
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

def view_as_real(input):
    real_part = expand_dims(real(input), -1)
    imag_part = expand_dims(imag(input), -1)
    return concat((real_part, imag_part), -1)

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

def inplace_scatter_value(input, dim, index, src, reduce='none'):
    """In-place scatter with scalar or tensor value."""
    if isinstance(src, numbers.Number):
        src = fill_scalar(index.shape, src, dtype=input.dtype)
    result = legacy.tensor_scatter_elements(input, index, src, dim, reduce)
    return inplace_copy(input, result)

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

def pixel_unshuffle(x, downscale_factor):
    batch_size, channels, height, width = x.shape
    
    # 计算新的尺寸
    new_height = height // downscale_factor
    new_width = width // downscale_factor
    new_channels = channels * (downscale_factor ** 2)
    
    # 第一步：重塑张量，将空间维度分解为小块
    # 形状: (N, C, H, W) -> (N, C, new_height, downscale_factor, new_width, downscale_factor)
    x = reshape(x, (batch_size, channels, new_height, downscale_factor, new_width, downscale_factor))
    
    # 第二步：置换维度，将下采样因子维度移到通道维度之后
    # 形状: (N, C, new_height, downscale_factor, new_width, downscale_factor) 
    #    -> (N, C, downscale_factor, downscale_factor, new_height, new_width)
    x = permute(x, (0, 1, 3, 5, 2, 4))
    
    # 第三步：重塑张量，合并通道和下采样因子维度
    # 形状: (N, C, downscale_factor, downscale_factor, new_height, new_width)
    #    -> (N, C * downscale_factor^2, new_height, new_width)
    x = reshape(x, (batch_size, new_channels, new_height, new_width))

    return x

def rms_norm(input, normalized_shape, weight, eps=1e-5):
    if eps is None:
        eps = mindtorch.finfo(input.dtype).eps
    if weight is None:
        weight = ones(normalized_shape, dtype=input.dtype)

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
        return mean(loss, (), False, None)
    elif reduction == 'sum':
        return sum(loss, (), False, None)
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
    # Convert tensor boundaries to list of floats
    if hasattr(boundaries, 'tolist'):
        boundaries = boundaries.tolist()
    boundaries = [float(boundary) + epsilon_ for boundary in boundaries]
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
    # mode can be 'reduced', 'complete', or 'r'
    # For 'complete', use full_matrices=True; for 'reduced' or 'r', use full_matrices=False
    full_matrices = mode == 'complete'
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
    return legacy.scatter_nd_update(input, indices, cast(updates, input.dtype), True)

def triu_indices(row, col, offset, dtype):
    return legacy.triu_indices(row, col, offset, dtype)

def cumprod(input, dim, dtype):
    out = legacy.cum_prod(input, dim, False, False)
    if dtype is not None:
        out = cast(out, dtype)
    return out

def lerp(input, end, weight):
    return legacy.lerp(input, end, weight)

def smooth_l1_loss(input, target, beta=1.0, reduction='none'):
    return legacy.smooth_l1_loss(input, target, beta, reduction)

def index_select(input, dim, index):
    return legacy.gather(input, index, dim, 0)

def custom_circular_pad(x, pad):

    ndim = x.ndim
    n_pad_dims = len(pad) // 2
    assert n_pad_dims <= ndim, "填充参数超过了张量的维度"

    # 按从最后维度向前处理填充
    for dim in range(ndim-1, ndim-1-n_pad_dims, -1):
        # 当前维度的左右填充量
        idx = 2 * (ndim - 1 - dim)  # 在pad元组中的起始位置
        left_pad = pad[idx]
        right_pad = pad[idx + 1]
        
        if left_pad == 0 and right_pad == 0:
            continue  # 跳过该维度
            
        size = x.shape[dim]  # 当前维度的原始长度
        new_size = left_pad + size + right_pad
        
        # 生成循环索引: (index - left_pad) mod size
        index = fmod_scalar(add(arange(0, new_size, 1, mindspore.int64), new_size - left_pad), size)
        index = (index + x.shape[dim]) % x.shape[dim]
        x = index_select(x, dim, index)

    return x


def _reflection_pad(input, pad):
    """reflection pad"""
    out = input
    if len(pad) == 2:
        out = pyboost.reflection_pad_1d_op(input, pad)
    elif len(pad) == 4:
        out = pyboost.reflection_pad_2d_op(input, pad)
    else:
        out = pyboost.reflection_pad_3d_op(input, pad)
    return out

def pad(input, pad, mode='constant', value=None):
    if isinstance(pad, tuple):
        pad = tuple(p if isinstance(p, int) else p.item() for p in pad)

    new_pad = ()
    for idx, pad_v in enumerate(pad):
        if not isinstance(pad_v, int):
            pad_v = pad_v.item()
        if pad_v < 0:
            dim = input.ndim - 1 - idx // 2
            input = narrow(input, dim, 0, input.shape[dim] + pad_v)
            pad_v = 0
        new_pad += (pad_v,)
    if py_sum(new_pad) == 0:
        return input
    if mode == 'circular':
        return custom_circular_pad(input, pad)
    elif mode == 'reflect':
        # return pad_v3(input, new_pad, mode)
        return _reflection_pad(input, pad)
    if value is None:
        value = 0
    if mode == "replicate":
        mode = "edge"
        return pad_v3(input, new_pad, mode)
    if input.dtype.is_floating_point:
        value = float(value)
    elif input.dtype == mindtorch.bool:
        value = bool(value)
    elif input.dtype in [mindtorch.int32, mindtorch.int64]:
        value = int(value)

    return pad_v3(input, new_pad, mode, value)

tensor_1d = mindspore.Tensor([0], dtype=mindtorch.int64)
empty_tensor_1d = mindspore.Tensor(shape=(0,), dtype=mindtorch.int64)
empty_tensor_9d = mindspore.Tensor(shape=(0,)*9, dtype=mindtorch.int64)

def _do_select(self, dim: int, index: int, dim_index: int, self_shape: list):
    """call select view operator"""
    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    dim_size = self_shape[dim]
    if index >= dim_size or index < -dim_size:
        raise IndexError(f"Index {index} is out of bounds for dimension {dim_index} with size {dim_size}")
    index = index + dim_size if index < 0 else index
    return select_ext_view(self, dim, index)


def _do_slice(self, dim: int, index: py_slice, self_shape: list):
    """call slice view operator"""
    def _get_index(index, default):
        if index is None:
            return default
        if mindtorch.is_tensor(index):
            index = int(index)
        return index

    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    step = _get_index(index.step, 1)
    if step <= 0:
        raise ValueError("slice step must be positive")
    start = _get_index(index.start, 0)
    end = _get_index(index.stop, self_shape[dim])
    if start == 0 and end == self_shape[dim] and step == 1:
        return self
    return slice(self, dim, start, end, step)

def _wrap_index_to_tuple(index):
    """Wrap index to tuple"""
    if isinstance(index, tuple):
        return index
    if isinstance(index, list):
        if len(index) < 32 and any(isinstance(i, (mindtorch.Tensor, list, tuple, py_slice, type(None), type(...))) for i in index):
            return tuple(index)
    return (index,)


def _count_indexed_dims(indexes):
    """Count indexed dims"""
    count = 0
    for index in indexes:
        if isinstance(index, mindtorch.Tensor):
            if index.dtype == mindtorch.bool:
                count += index.ndim
            else:
                count += 1
        elif not isinstance(index, (type(None), type(...), bool)):
            count += 1
    return count

def _record_tensor_index(index, remain_indexes, dim):
    """Record indexes remained to be used by aclnnIndex/aclnnIndexPut"""
    if len(remain_indexes) > dim:
        remain_indexes[dim] = index
        return remain_indexes

    while dim > len(remain_indexes):
        # use empty_tensor with dim_num 9 to indicate unused dim
        remain_indexes.append(py_slice(None, None, None))

    remain_indexes.append(index)
    return remain_indexes

def _process_dim_in_multi_dim_index(prev_result, orig_tensor, index, dim, indexed_dims, dim_index, remain_indexes,
                                    prev_shape):
    """Process dim in multi dim index"""
    if isinstance(index, bool):
        result = expand_dims(prev_result, dim)
        index_for_bool = tensor_1d if index else empty_tensor_1d
        _record_tensor_index(index_for_bool, remain_indexes, dim)
        prev_shape.insert(dim, 1)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, int):
        result = _do_select(prev_result, dim, index, dim_index, prev_shape)
        del prev_shape[dim]
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, py_slice):
        result = _do_slice(prev_result, dim, index, prev_shape)
        # current dim in prev_shape will not be used later, ignore it
        dim += 1
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, type(...)):
        dim += (orig_tensor.ndim - indexed_dims)
        return prev_result, dim, remain_indexes, prev_shape
    if index is None:
        result = expand_dims(prev_result, dim)
        prev_shape.insert(dim, 1)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    if isinstance(index, mindtorch.Tensor):
        result = prev_result
        if index.ndim == 0 and index.dtype in (mindtorch.int, mindtorch.long, mindtorch.short, mindtorch.bool):
            if index.dtype in (mindtorch.int, mindtorch.long, mindtorch.short):
                result = _do_select(prev_result, dim, index.item(), dim_index, prev_shape)
                del prev_shape[dim]
                return result, dim, remain_indexes, prev_shape
            # process index with Tensor bool type
            result = expand_dims(prev_result, dim)
            index_for_bool = tensor_1d if index else empty_tensor_1d
            _record_tensor_index(index_for_bool, remain_indexes, dim)
            prev_shape.insert(dim, 1)
            dim += 1
            return result, dim, remain_indexes, prev_shape
        _record_tensor_index(index, remain_indexes, dim)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    raise IndexError(f"Invalid tensor index type {index}")


def _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims):
    """Process indexes in tuple"""
    self_viewed = self
    self_viewed_shape = list(self.shape)
    dim = 0
    # if ON_ORANGE_PI:
    #     if all([isinstance(index, slice) for index in indexes]):
    #         return getitem(self_viewed, tuple(indexes)), remain_indexes
    for i, index in enumerate(indexes):
        if isinstance(index, (list, tuple, np.ndarray)):
            index_np = np.array(index) if isinstance(index, (list, tuple)) else index
            if index_np.dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                                  np.float16, np.float32, np.float64):
                index = mindspore.tensor(index_np, dtype=mindtorch.int64)
            elif index_np.dtype == np.bool_:
                index = mindspore.tensor(index_np, dtype=mindtorch.int64)
            else:
                raise TypeError(f"Index {index} contain unsupported elements")
        self_viewed, dim, remain_indexes, self_viewed_shape = _process_dim_in_multi_dim_index(
            self_viewed, self, index, dim, indexed_dims, i, remain_indexes, self_viewed_shape)
    return self_viewed, remain_indexes



_SLICE_ERROR = (
    'only integers, slices (`:`), ellipsis (`...`), '
    'newaxis (`None`) and integer or boolean arrays are valid indices'
)


def _as_index(idx, need_scalar=True):
    """Helper function to parse idx as an index.
    """
    if isinstance(idx, numbers.Integral):
        return idx, True

    if not isinstance(idx, mindtorch.Tensor):
        idx = mindspore.tensor(idx, dtype=mindtorch.int64)

    if idx.dtype == mindtorch.bool:
        # For multi-dimensional boolean masks, flatten before converting to indices
        if idx.ndim > 1:
            idx = idx.reshape(-1)
        idx = non_zero_ext(idx)
        # non_zero_ext returns a tuple, get the first element
        if isinstance(idx, tuple):
            idx = idx[0]
        idx = idx.reshape(-1)

    if need_scalar and idx.ndim not in (None, 0):
        raise IndexError(_SLICE_ERROR + ', got {!r}'.format(idx))

    if idx.ndim == 0:
        return idx.item(), True

    return idx, False

def moveaxis(a, source, destination):
    """Raises ValueError if source, destination not in (-ndim(a), ndim(a))."""
    if not source and not destination:
        return a

    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    if len(source) != len(destination):
        raise ValueError('The lengths of source and destination must equal')

    a_rank = a.ndim

    def _correct_axis(axis, rank):
        if axis < 0:
            return axis + rank
        return axis

    source = tuple(_correct_axis(axis, a_rank) for axis in source)
    destination = tuple(_correct_axis(axis, a_rank) for axis in destination)

    if a.ndim is not None:
        perm = [i for i in range(a_rank) if i not in source]
        for dest, src in sorted(zip(destination, source)):
            assert dest <= len(perm)
            perm.insert(dest, src)
    else:
        r = range(0, a_rank, 1)

        def _remove_indices(a, b):
            """Remove indices (`b`) from `a`."""
            items = unstack_view(
                sort(stack(b), -1, False, False), 0
            )

            i = 0
            result = []

            for item in items:
                result.append(a[i:item])
                i = item + 1

            result.append(a[i:])

            return concat(result, 0)

        minus_sources = _remove_indices(r, source)
        minus_dest = _remove_indices(r, destination)

        perm = scatter_nd(expand_dims(minus_dest, 1), minus_sources, [a_rank])
        perm = tensor_scatter_update(perm, expand_dims(destination, 1), source)
    a = mindtorch.permute(a, tuple(perm))

    return a

def _cumprod(x, axis=0, exclusive=False, reverse=False):
    x = np.array(x)
    if reverse:
        x = np.flip(x, axis=axis)

    if exclusive:
        shifted_x = np.ones_like(x)
        if axis == 0:
            shifted_x[1:] = x[:-1]
        else:
            shifted_x[:, 1:] = x[:, :-1]
        result = np.cumprod(shifted_x, axis=axis)
    else:
        result = np.cumprod(x, axis=axis)

    if reverse:
        result = np.flip(result, axis=axis)

    return result

def broadcast_shapes(*shapes):
    reversed_shapes = [list(reversed(shape)) for shape in shapes]

    max_dim = py_max(len(shape) for shape in reversed_shapes)

    result_shape = [1] * max_dim

    for i in range(max_dim):
        current_dim_size = 1
        for shape in reversed_shapes:
            if i < len(shape):
                if shape[i] == 1:
                    continue
                if current_dim_size == 1:
                    current_dim_size = shape[i]
                elif current_dim_size != shape[i]:
                    raise ValueError(f"Shapes {shapes} are not broadcastable.")
        result_shape[i] = current_dim_size

    return tuple(reversed(result_shape))

def broadcast_tensors(*tensors):
    target_shape = broadcast_shapes(*[t.shape for t in tensors])
    broadcasted_tensors = [broadcast_to(t, target_shape) for t in tensors]
    return broadcasted_tensors


def _slice_helper(tensor, slice_spec, do_update=False, updates=None):
    """Helper function for __getitem__ and _with_index_update_helper.
    """
    begin, end, strides = [], [], []
    new_axis_mask, shrink_axis_mask = 0, 0
    begin_mask, end_mask = 0, 0
    ellipsis_mask = 0
    advanced_indices = []
    shrink_indices = []
    for index, s in enumerate(slice_spec):
        if isinstance(s, py_slice):
            if s.start is not None:
                begin.append(s.start)
            else:
                begin.append(0)
                begin_mask |= (1 << index)
            if s.stop is not None:
                stop = s.stop
                if stop == -1:
                    stop = tensor.shape[index] - 1
                end.append(stop)
            else:
                end.append(0)
                end_mask |= (1 << index)
            if s.step is not None:
                strides.append(s.step)
            else:
                strides.append(1)
        elif s is Ellipsis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            ellipsis_mask |= (1 << index)
        elif s is None:
            begin.append(0)
            end.append(0)
            strides.append(1)
            new_axis_mask |= (1 << index)
        else:
            s, is_scalar = _as_index(s, False)
            if is_scalar:
                begin.append(s)
                end.append(s + 1)
                strides.append(1)
                shrink_axis_mask |= (1 << index)
                shrink_indices.append(index)
            else:
                begin.append(0)
                end.append(0)
                strides.append(1)
                begin_mask |= (1 << index)
                end_mask |= (1 << index)
                advanced_indices.append((index, s, ellipsis_mask != 0))

    if do_update and not advanced_indices:
        # if 0 in updates.shape:
        #     return tensor
        return strided_slice_update(
            tensor,
            begin,
            end,
            strides,
            updates,
            begin_mask=begin_mask,
            end_mask=end_mask,
            shrink_axis_mask=shrink_axis_mask,
            new_axis_mask=new_axis_mask,
            ellipsis_mask=ellipsis_mask,
        )
    else:
        if updates is not None:
            original_tensor = tensor
        if new_axis_mask != 0:
            tensor = strided_slice_manual(
                tensor,
                begin,
                end,
                strides,
                begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
            )
        else:
            tensor = strided_slice(
                tensor,
                begin,
                end,
                strides,
                begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask
            )

    if not advanced_indices:
        return tensor
    advanced_indices_map = {}
    for index, data, had_ellipsis in advanced_indices:
        if had_ellipsis:
            num_shrink = len([x for x in shrink_indices if x > index])
            dim = index - len(slice_spec) + num_shrink
        else:
            num_shrink = len([x for x in shrink_indices if x < index])
            dim = index - num_shrink
        advanced_indices_map[dim] = data
    dims = sorted(advanced_indices_map.keys())
    dims_contiguous = True
    if len(dims) > 1:
        if dims[0] < 0 and dims[-1] >= 0:  # not all same sign
            dims_contiguous = False
        else:
            for i in range(len(dims) - 1):
                if dims[i] + 1 != dims[i + 1]:
                    dims_contiguous = False
                    break
    indices = [advanced_indices_map[x] for x in dims]
    indices = broadcast_tensors(*indices)
    stacked_indices = stack(indices, -1)
    # Skip the contiguous-dims optimization for update because there is no
    # tf.*scatter* op that supports the `axis` argument.
    if not dims_contiguous or updates is not None:
        if range(len(dims)) != dims:
            tensor = moveaxis(tensor, dims, range(len(dims)))
        tensor_shape_prefix = mindspore.tensor(tensor.shape[: len(dims)])
        stacked_indices = select(
            less(stacked_indices, 0),
            add(stacked_indices, tensor_shape_prefix),
            stacked_indices,
        )
        if updates is None:
            return gather_nd(tensor, stacked_indices)
        else:
            # We only need to move-axis `updates` in the contiguous case becausce
            # only in this case the result dimensions of advanced indexing are in
            # the middle of `updates`. In the non-contiguous case, those dimensions
            # are always at the front.
            if dims_contiguous and updates.ndim > 1:
                batch_size = stacked_indices.ndim - 1
                batch_start = dims[0]
                if batch_start < 0:
                    batch_start += len(dims) - batch_size

                def range_(start, length):
                    return range(start, start + length)

                updates = moveaxis(
                    updates, range_(batch_start, batch_size), range(batch_size)
                )
            # Calculate target shape for updates
            target_shape = stacked_indices.shape[:-1] + tensor.shape[stacked_indices.shape[-1]:]
            # Only broadcast if shapes don't match
            if updates.shape != target_shape:
                # Check if updates can be broadcast to target_shape
                try:
                    updates = updates.broadcast_to(target_shape)
                except:
                    # If broadcast fails, updates might already have the correct data shape
                    # Just reshape/flatten as needed
                    if updates.numel() == np.prod(target_shape):
                        updates = updates.reshape(target_shape)
            tensor = tensor_scatter_update(tensor, stacked_indices, updates)
            if range(len(dims)) != dims:
                tensor = moveaxis(tensor, range(len(dims)), dims)
            return strided_slice_update(
                original_tensor,
                begin,
                end,
                strides,
                tensor,
                begin_mask=begin_mask,
                end_mask=end_mask,
                shrink_axis_mask=shrink_axis_mask,
                new_axis_mask=new_axis_mask,
                ellipsis_mask=ellipsis_mask,
            )

    # Note that gather_nd does not support gathering from inside the array.
    # To avoid shuffling data back and forth, we transform the indices and
    # do a gather instead.
    rank = tensor.ndim
    dims = [(x + rank if x < 0 else x) for x in dims]
    shape_tensor = tensor.shape
    dim_sizes = np.take_along_axis(np.array(shape_tensor), np.array(dims), axis=0)
    if len(dims) == 1:
        stacked_indices = indices[0]
    stacked_indices = stacked_indices.to(mindtorch.int32)
    stacked_indices = select(
        less(stacked_indices, 0), add(stacked_indices, mindspore.tensor(dim_sizes, dtype=stacked_indices.dtype)), stacked_indices
    )
    axis = dims[0]
    if len(dims) > 1:
        index_scaling = _cumprod(dim_sizes, reverse=True, exclusive=True)

        def _tensordot(a, b):
            # TODO(b/168657656): This function should be replaced by
            # tensordot(axis=1) once MatMul has int32 XLA kernel.
            b = broadcast_to(b, a.shape)
            return sum(mul(a,b), -1, False, None)

        stacked_indices = _tensordot(stacked_indices, mindspore.tensor(index_scaling))
        flat_shape = shape_tensor[:axis] + (-1,) + shape_tensor[axis + len(dims) :]
        tensor = reshape(tensor, flat_shape)

    return gather(tensor, stacked_indices, axis, 0)

def _as_spec_tuple(slice_spec):
    """Convert slice_spec to tuple."""
    if isinstance(slice_spec, (list, tuple)):
        is_index = True
        for s in slice_spec:
            if s is None or s is Ellipsis or isinstance(s, (list, tuple, py_slice)):
                is_index = False
                break
        if not is_index:
            return tuple(slice_spec)
    return (slice_spec,)

def getitem(self, slice_spec):
    """Getitem with NumPy-based backward to avoid MindSpore InplaceCopy issues."""
    # Handle boolean indexing
    if (
        isinstance(slice_spec, bool)
        or (
            isinstance(slice_spec, mindtorch.Tensor)
            and slice_spec.dtype == mindtorch.bool
        )
    ):
        if self.shape == slice_spec.shape:
            return masked_select(self, slice_spec)
        slice_spec = non_zero_ext(slice_spec)

    # Use NumPy-based GetitemFunction for proper backward support
    # Convert slice_spec to a format compatible with NumPy
    if not isinstance(slice_spec, tuple):
        slice_spec = _as_spec_tuple(slice_spec)

    # Convert mindtorch tensors in slice_spec to numpy for indexing
    numpy_slice_spec = _convert_slice_spec_to_numpy(slice_spec)

    # Use custom GetitemFunction with NumPy-based backward
    return GetitemFunction.apply(self, numpy_slice_spec)


def _convert_slice_spec_to_numpy(slice_spec):
    """Convert slice_spec to numpy-compatible format."""
    if not isinstance(slice_spec, tuple):
        slice_spec = (slice_spec,)

    result = []
    for s in slice_spec:
        if isinstance(s, mindtorch.Tensor):
            # Convert tensor indices to numpy
            result.append(s.asnumpy())
        elif isinstance(s, py_slice):
            result.append(s)
        elif s is None or s is Ellipsis:
            result.append(s)
        elif isinstance(s, (int, np.integer)):
            result.append(int(s))
        else:
            result.append(s)

    return tuple(result) if len(result) > 1 else result[0]

def setitem(a, slice_spec, updates):
    """Implementation of ndarray setitem using NumPy for correctness."""
    # Convert to numpy, perform setitem, convert back
    a_np = a.asnumpy()

    # Convert slice_spec to numpy-compatible format
    if isinstance(slice_spec, tuple):
        numpy_slice = []
        for s in slice_spec:
            if isinstance(s, mindtorch.Tensor):
                if s.dtype == mindtorch.bool:
                    numpy_slice.append(s.asnumpy())
                else:
                    numpy_slice.append(s.asnumpy())
            else:
                numpy_slice.append(s)
        slice_spec = tuple(numpy_slice)
    elif isinstance(slice_spec, mindtorch.Tensor):
        if slice_spec.dtype == mindtorch.bool:
            slice_spec = slice_spec.asnumpy()
        else:
            slice_spec = slice_spec.asnumpy()

    # Convert updates to numpy
    if isinstance(updates, mindtorch.Tensor):
        updates_np = updates.asnumpy()
    else:
        updates_np = updates

    # Perform numpy setitem
    a_np[slice_spec] = updates_np

    # Convert back and update in-place
    result = mindspore.tensor(a_np, dtype=a.dtype)
    inplace_copy(a, result)
    return a


def strided_slice_manual(x, begin, end, strides, begin_mask=0, end_mask=0,
                ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0):

    x_shape = x.shape
    ndim = len(x_shape)

    full_begin, full_end, full_strides = [], [], []
    dim = 0  # 当前 x 的维度
    i = 0    # 当前 begin/end 索引

    while dim < ndim:
        # ellipsis_mask
        if i < len(begin) and ((ellipsis_mask >> i) & 1):
            remaining_dims = ndim - dim - (len(begin) - i - 1)
            # Shift shrink_axis_mask to account for expanded ellipsis dimensions
            # Only shift if there are dimensions after the ellipsis
            if remaining_dims > 1:
                shrink_axis_mask = shrink_axis_mask << (remaining_dims - 1)
            for _ in range(remaining_dims):
                full_begin.append(0)
                full_end.append(x_shape[dim])
                full_strides.append(1)
                dim += 1
            i += 1
            continue

        # new_axis_mask
        elif i < len(begin) and ((new_axis_mask >> i) & 1):
            full_begin.append(0)
            full_end.append(1)
            full_strides.append(1)
            i += 1
            continue

        else:
            # 自动补齐 begin/end/strides
            b = begin[i] if i < len(begin) else 0
            e = end[i]   if i < len(end) else x_shape[dim]
            s = strides[i] if i < len(strides) else 1
            if b < 0:
                b += x_shape[dim]
                if e == 0:
                    e += x_shape[dim]
            if e < 0:
                e += x_shape[dim]

            # begin_mask / end_mask
            if i < len(begin) and ((begin_mask >> i) & 1):
                b = 0 if s > 0 else x_shape[dim]-1
            if i < len(end) and ((end_mask >> i) & 1):
                e = x_shape[dim] if s > 0 else -1

            # Clamp bounds to ensure indices don't exceed tensor dimensions
            b = py_max(0, py_min(b, x_shape[dim]))
            e = py_max(0, py_min(e, x_shape[dim]))

            full_begin.append(b)
            full_end.append(e)
            full_strides.append(s)

            dim += 1
            i += 1

    # Step 2: generate indices for scatter update
    ranges = [arange(b, e, s) for b, e, s in zip(full_begin, full_end, full_strides)]
    mesh = meshgrid(ranges, 'ij')
    indices = stack(mesh, -1)
    indices = reshape(indices, [-1, ndim])

    x_updated = gather_nd(x, indices)

    # # Step 5: optionally squeeze shrinked axes
    for i in range(ndim-1, -1, -1):
        if (shrink_axis_mask >> i) & 1:
            x_updated = squeeze(x_updated, dim=i)

    return x_updated

def strided_slice_update(x, begin, end, strides, updates,
                         begin_mask=0, end_mask=0,
                         ellipsis_mask=0, new_axis_mask=0,
                         shrink_axis_mask=0):
    x_shape = x.shape
    ndim = len(x_shape)

    full_begin, full_end, full_strides = [], [], []
    dim = 0  # 当前 x 的维度
    i = 0    # 当前 begin/end 索引

    while dim < ndim:
        # ellipsis_mask
        if i < len(begin) and ((ellipsis_mask >> i) & 1):
            remaining_dims = ndim - dim - (len(begin) - i - 1)
            # Shift shrink_axis_mask to account for expanded ellipsis dimensions
            # Only shift if there are dimensions after the ellipsis
            if remaining_dims > 1:
                shrink_axis_mask = shrink_axis_mask << (remaining_dims - 1)
            for _ in range(remaining_dims):
                full_begin.append(0)
                full_end.append(x_shape[dim])
                full_strides.append(1)
                dim += 1
            i += 1
            continue

        # new_axis_mask
        elif i < len(begin) and ((new_axis_mask >> i) & 1):
            full_begin.append(0)
            full_end.append(1)
            full_strides.append(1)
            i += 1
            continue

        else:
            # 自动补齐 begin/end/strides
            b = begin[i] if i < len(begin) else 0
            e = end[i]   if i < len(end) else x_shape[dim]
            s = strides[i] if i < len(strides) else 1
            if b < 0:
                b %= x_shape[dim]
                if e == 0:
                    e += x_shape[dim]
            if e < 0:
                e %= x_shape[dim]
            # begin_mask / end_mask
            if i < len(begin) and ((begin_mask >> i) & 1):
                b = 0 if s > 0 else x_shape[dim]-1
            if i < len(end) and ((end_mask >> i) & 1):
                e = x_shape[dim] if s > 0 else -1

            # Clamp bounds to ensure indices don't exceed tensor dimensions
            b = py_max(0, py_min(b, x_shape[dim]))
            e = py_max(0, py_min(e, x_shape[dim]))

            full_begin.append(b)
            full_end.append(e)
            full_strides.append(s)

            dim += 1
            i += 1

    # Step 2: 计算目标切片 shape（考虑 shrink_axis_mask）
    target_shape = []

    for d, (b, e, s) in enumerate(zip(full_begin, full_end, full_strides)):
        if (shrink_axis_mask >> d) & 1:
            continue
        length = py_max(0, (py_abs(e - b) + py_abs(s) - 1) // py_abs(s))
        target_shape.append(length)

    # Step 3: broadcast updates if scalar
    updates = broadcast_to(updates, target_shape)

    # Step 2: generate indices for scatter update
    ranges = [arange(b, e, s, mindspore.int64) for b, e, s in zip(full_begin, full_end, full_strides)]
    mesh = meshgrid(ranges, 'ij')
    indices = stack(mesh, -1)
    indices = reshape(indices, [-1, ndim])

    # Step 3: flatten updates
    updates_flat = reshape(updates, [-1])
    # if updates.shape[0] == 1 and updates.shape[0] != indices.shape[0]:
    #     updates = updates.broadcast_to((indices.shape[0],))
    # Step 4: apply scatter update
    if x.dtype == mindtorch.bool:
        x_updated = cast(scatter_nd_update(cast(x, mindspore.int32), indices, cast(updates_flat, mindspore.int32)), mindspore.bool_)
    else:
        x_updated = scatter_nd_update(x, indices, updates_flat)

    assign(x, x_updated)
    # # Step 5: optionally squeeze shrinked axes
    # for i in range(ndim-1, -1, -1):
    #     if (shrink_axis_mask >> i) & 1:
    #         x_updated = mindtorch.squeeze(x_updated, dim=i)
    return x_updated

def mish(input):
    return legacy.mish(input)

def selu(input):
    """SELU activation: scale * elu(x, alpha) where alpha=1.67326324, scale=1.05070098"""
    SELU_ALPHA = 1.67326324
    SELU_SCALE = 1.05070098
    return legacy.mul(legacy.elu(input, SELU_ALPHA), SELU_SCALE)

def celu(input, alpha):
    """CELU activation: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))"""
    if alpha == 0:
        raise ZeroDivisionError("ZeroDivisionError: alpha cannot be 0 for CELU")
    return legacy.elu(input, alpha)

def hardsigmoid(input):
    """Hardsigmoid activation: clamp((x + 3) / 6, 0, 1)"""
    x_plus_3 = legacy.add(input, 3.0)
    x_div_6 = legacy.div(x_plus_3, 6.0)
    return clamp_scalar(x_div_6, 0.0, 1.0)

def fast_gelu(x):
    """Fast GELU approximation"""
    return gelu(x, approximate='tanh')

def swiglu(x, dim=-1):
    """Swish-Gated Linear Unit: swish(x[..., :d]) * x[..., d:] where d = x.shape[dim] // 2"""
    split_size = x.shape[dim] // 2
    x1, x2 = legacy.split(x, split_size, dim)
    return legacy.mul(silu(x1), x2)

def upsample_nearest3d(input, output_size, scale_factors):
    return pyboost.upsample_nearest3d_op(input, output_size, scale_factors)

def multinomial(input, num_samples, replacement=False, generator=None):
    """
    Draws samples from a multinomial distribution.

    Arguments:
        input (Tensor): the input tensor containing probabilities or unnormalized log probabilities (must be non-negative, not necessarily summing to one)
        num_samples (int): number of samples to draw
        replacement (bool): whether to draw with replacement or not
        generator (Generator, optional): a pseudorandom number generator for sampling

    Returns:
        Tensor: samples drawn from each row of input (or from the input if 1d)
    """
    # Validate input shape: must be 1d or 2d
    input_shape = input.shape
    input_ndim = len(input_shape)
    if input_ndim not in (1, 2):
        raise RuntimeError(f"multinomial only supports 1D or 2D input, got input of shape {input_shape}")

    # Validate non-negativity
    if (input < 0).any():
        raise RuntimeError("Multinomial probabilities must be non-negative, but got negative probabilities.")

    # Validate probabilities
    if (sum(input, -1, False, None) == 0).any():
        raise RuntimeError("Multinomial probabilities sum to zero for some rows.")

    if input_ndim == 1:
        batch_size = 1
        prob_size = input_shape[0]
    else:
        batch_size, prob_size = input_shape

    if not replacement and num_samples > prob_size:
        raise RuntimeError(
            "multinomial() cannot sample more elements than there are probabilities "
            "when 'replacement=False'"
        )

    # Normalize probabilities along the last dimension
    input_sum = sum(input, dim=-1, keepdim=True, dtype=None)
    probs = div(input, input_sum)

    # Handle sampling
    seed, offset = None, None
    if generator is not None:
        seed, offset = generator._step(12)  # pylint: disable=protected-access

    if replacement:
        return legacy.multinomial(probs, num_samples, seed, offset, mindspore.int64)
    else:
        # Without replacement: use Gumbel-max trick, matching torch.multinomial
        def gumbel(shape, generator, dtype):
            # Uniform in (0,1); avoid 0 for log
            uniform = rand(shape, generator, dtype)
            eps = 1e-10
            uniform = maximum(uniform, eps)
            return -log(-log(uniform))

        # Gumbel + log_prob
        if input_ndim == 1:
            logits = log(probs)
            g = gumbel(logits.shape, generator, logits.dtype)
            y = logits + g
            # largest=True for topk to get max
            _, indices = topk(y, num_samples, dim=-1, largest=True, sorted=True)
            return cast(indices, mindspore.int64)
        else:
            logits = log(probs)
            g = gumbel(logits.shape, generator, logits.dtype)
            y = logits + g
            _, indices = topk(y, num_samples, dim=-1, largest=True, sorted=True)
            return cast(indices, mindspore.int64)
