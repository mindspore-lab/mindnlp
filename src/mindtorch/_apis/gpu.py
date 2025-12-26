import numbers
import math
from itertools import zip_longest

import numpy as np
import mindspore
from mindspore.ops.composite.multitype_ops._compile_utils import _tensor_setitem_origin, _tensor_getitem_origin

import mindtorch
from .._op_prim.gpu import legacy

try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_


def empty(size, dtype):
    return mindspore.Tensor(Tensor_(shape=size, dtype=dtype))

def empty_like(input, dtype=None):
    if dtype is None:
        dtype = input.dtype
    return empty(input.shape, dtype)

def select_ext_view(input, dim, index):
    return legacy.select_view(input, index, dim)

def inplace_copy(input, value):
    if value.shape != input.shape:
        value = legacy.fill_v2(input.shape, value)
    if input.dtype == mindspore.int64:
        legacy.copy_with_slice(input, value)
    else:
        legacy.inplace_update_v2(input, tuple(range(input.shape[0])), value)
    return input

def fill_scalar(size, fill_value, dtype):
    if dtype is None:
        return legacy.fill_v2(size, mindspore.Tensor(fill_value))
    return legacy.cast(legacy.fill_v2(size, mindspore.Tensor(fill_value)), dtype)

def fill_tensor(size, fill_value, dtype):
    if dtype is None:
        return legacy.fill_v2(size, mindspore.Tensor(fill_value))
    return legacy.cast(legacy.fill_v2(size, fill_value), dtype)

def zeros_like(input, dtype):
    if dtype is None:
        return legacy.zeros_like(input)
    return legacy.cast(legacy.zeros_like(input), dtype)

def tensor_shape(input):
    return legacy.tensor_shape(input)

def broadcast_to(input, shape):
    return legacy.broadcast_to(input, shape)

def zeros(shape, dtype):
    return legacy.zeros(shape, dtype)

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
    return legacy.identity(input)

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
    """
    Implements torch.matmul semantics (see: https://pytorch.org/docs/stable/generated/torch.matmul.html).
    Supports scalar, 1D, 2D, and ND tensors following torch's broadcasting rules.
    """
    # Helper to check if scalar
    def is_scalar(x):
        return getattr(x, 'ndim', 0) == 0

    # Scalar cases
    if is_scalar(self) or is_scalar(other):
        return legacy.mul(self, other)

    # 1D x 1D: dot
    if self.ndim == 1 and other.ndim == 1:
        return legacy.reduce_sum(legacy.mul(self, other), (), False)

    # 2D x 1D: matvec
    if self.ndim == 2 and other.ndim == 1:
        other_ = legacy.reshape(other, (-1, 1))
        result = legacy.mat_mul(self, other_, False, False)
        return legacy.reshape(result, (-1,))

    # 1D x 2D: treat 1D as row vector
    if self.ndim == 1 and other.ndim == 2:
        self_ = legacy.reshape(self, (1, -1))
        result = legacy.mat_mul(self_, other, False, False)
        return legacy.reshape(result, (-1,))

    # 2D x 2D: matrix multiplication
    if self.ndim == 2 and other.ndim == 2:
        return legacy.mat_mul(self, other, False, False)

    # ND, broadcast batch dimension per torch
    # We follow torch.matmul broadcasting rules
    # (https://pytorch.org/docs/stable/generated/torch.matmul.html)
    # Suppose: [*, m, n] @ [*, n, p] => [*, m, p]
    # If 1d, view as (1, n) or (n, 1) as appropriate

    # Expand 1D to 2D for batched semantics (for high-dim cases)
    self_was_1d = False
    other_was_1d = False
    if self.ndim == 1:
        self = legacy.reshape(self, (1, self.shape[0]))
        self_was_1d = True
    if other.ndim == 1:
        other = legacy.reshape(other, (other.shape[0], 1))
        other_was_1d = True

    # After this, both are at least 2D

    # Broadcast batch dimensions (all dims except the last 2)
    batch_dim_self = self.shape[:-2]
    batch_dim_other = other.shape[:-2]
    # Broadcast as torch does
    broadcast_shape = []
    for s, o in zip_longest(reversed(batch_dim_self), reversed(batch_dim_other), fillvalue=1):
        broadcast_shape.append(py_max([s, o]))
    broadcast_shape = tuple(reversed(broadcast_shape))

    # Now broadcast self and other to [*broadcast_shape, ...]
    # Only if shapes not already same (or scalars)
    def do_broadcast(tensor, current_batch, target_batch):
        if current_batch == target_batch:
            return tensor
        else:
            new_shape = target_batch + tensor.shape[-2:]
            return broadcast_to(tensor, new_shape)

    self = do_broadcast(self, self.shape[:-2], broadcast_shape)
    other = do_broadcast(other, other.shape[:-2], broadcast_shape)

    # Batched matmul: always at least 3D now (since we padded)
    result = legacy.batch_mat_mul(self, other, False, False)

    # Remove extra dims for 1D inputs per torch spec
    if self_was_1d and other_was_1d:
        return legacy.squeeze(result)
    elif self_was_1d:
        return legacy.squeeze(result, axis=-2)
    elif other_was_1d:
        return legacy.squeeze(result, axis=-1)
    else:
        return result

def div(input, other):
    return legacy.div(input, other)

def mul(input, other):
    # If both input and other are bool or boolean tensors, use bitwise_and
    input_is_bool = (getattr(input, 'dtype', None) == mindspore.bool_) or isinstance(input, bool)
    other_is_bool = (getattr(other, 'dtype', None) == mindspore.bool_) or isinstance(other, bool)
    if input_is_bool and other_is_bool:
        return legacy.bitwise_and(input, other)
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

def pad(input, paddings):
    return legacy.pad(input, paddings)

def cumsum(self, dim, dtype):
    if self.shape[dim] == 0:
        return mindtorch.tensor([], dtype=self.dtype, device=self.device)
    if self.dtype == mindtorch.bool:
        self = cast(self, mindspore.int32)
    return legacy.cum_sum(self, dim, False, False)

def reduce_any(input, axis, keepdims):
    if axis is None:
        axis = ()
    return legacy.reduce_any(input.bool(), axis, keepdims)

def concat(tensors, axis):
    tensors = [cast(tensor, tensors[0].dtype) for tensor in tensors]
    return legacy.concat(tensors, axis)

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
    out = legacy.sort(input, dim, descending)
    return out[0], cast(out[1], mindspore.int64)

def gather(input_params, input_indices, axis, batch_dim):
    return legacy.gather(input_params, input_indices, axis, batch_dim)

def index_select(input, dim, index):
    return legacy.gather(input, index, dim, 0)

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
    if 0 in condition.shape:
        return mindspore.Tensor(Tensor_(shape=condition.shape, dtype=Tensor_(x).dtype))
    if isinstance(x, numbers.Number) or x.ndim == 0:
        x = fill_scalar(condition.shape, x, None)
    if isinstance(y, numbers.Number) or y.ndim == 0:
        y = fill_scalar(condition.shape, y, None)

    return legacy.select(condition, x, y)
    # return add(mul(condition, x), mul(sub(1, condition), y))


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

def inplace_sub(input, other):
    return inplace_copy(input, legacy.sub(input, other))

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
    return legacy.gather(weight, input, 0, 0)

def linspace(start, end, steps, dtype):
    start = float(start)
    end = float(end)
    return legacy.lin_space(mindspore.Tensor(start), mindspore.Tensor(end), steps)

def masked_fill(input, mask, value):
    value = fill_scalar((), value, input.dtype)    
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
    if axis is None:
        axis = -1
    return legacy.arg_min_with_value(input, axis, keep_dims)

def argmax_with_value(input, axis, keep_dims):
    if axis is None:
        axis = -1

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
    if axis is None:
        axis = -1
    return legacy.arg_max_with_value(input, axis, keep_dims)[0]

def argmin(input, axis, keep_dims):
    if axis is None:
        axis = -1
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
    if input.data_ptr() == src.data_ptr():
        src = clone(src)
    return legacy.tensor_scatter_elements(input, index, src, dim, "none")

def batch_norm(input, weight, bias, running_mean=None, running_var=None, training=False, momentum=0.1, epsilon=1e-5):
    if running_mean is None:
        running_mean = ones(input.shape[1], dtype=input.dtype)
    if running_var is None:
        running_var = zeros(input.shape[1], dtype=input.dtype)
    if weight is None:
        weight = ones(input.shape[1], dtype=input.dtype)
    if bias is None:
        bias = zeros(input.shape[1], dtype=input.dtype)

    return legacy.batch_norm(input, weight, bias, running_mean, running_var, training, epsilon, momentum, 'NCHW')

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    """
    Implements Group Normalization by reshaping and calling batch_norm.
    This function reshapes input of shape (N, C, *spatial) into
    (N * num_groups, C // num_groups, -1) so that batch_norm computes
    mean/var across the correct axes, then reshapes back.
    """
    input_shape = input.shape
    N = input_shape[0]
    C = input_shape[1]

    # compute product of spatial dimensions
    spatial_dims = input_shape[2:]
    spatial_size = 1
    for s in spatial_dims:
        spatial_size *= s

    # reshape to (N * num_groups, C // num_groups, spatial_size)
    assert C % num_groups == 0, "C must be divisible by num_groups"
    channels_per_group = C // num_groups
    input_reshaped = reshape(input, (N * num_groups, channels_per_group, spatial_size if spatial_size != 0 else 1))

    # use batch_norm to compute mean/var over batch and spatial dims for each group-channel
    outputs = batch_norm(input_reshaped, None, None, None, None, True, 0.0, eps)

    # reshape back to original
    out = reshape(outputs, input_shape)

    # apply affine parameters if provided
    affine_param_shape = [1] * input.ndim
    affine_param_shape[1] = C
    affine_param_shape = tuple(affine_param_shape)

    if weight is not None and bias is not None:
        out = add(out, reshape(bias, affine_param_shape))
        out = mul(out, reshape(weight, affine_param_shape))
    elif weight is not None:
        out = mul(out, reshape(weight, affine_param_shape))
    elif bias is not None:
        out = add(out, reshape(bias, affine_param_shape))
    return out


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
    return legacy.dropout(input, 1-p, 0, 0)[0]

def split_tensor(input, split_size_or_sections, dim):
    if isinstance(split_size_or_sections, int):
        num = input.shape[dim] // split_size_or_sections
        return legacy.split(input, dim, num)

def bmm(input_x, input_y):
    return legacy.batch_mat_mul(input_x, input_y, False, False)

def nllloss(input, target, weight, reduction, ingore_index):
    # Native implementation for NLLLoss
    # Follow the pattern from functional.py
    target_dim = 1  # Class dimension
    
    # Ensure target has same number of dimensions as input
    target_expanded = target
    if target.ndim == input.ndim - 1:
        target_expanded = expand_dims(target, target_dim)
    
    # Handle ignore_index by creating mask and temporarily replacing with 0
    if ingore_index is not None:
        non_pad_mask = eq(target_expanded, ingore_index)
        target_expanded = masked_fill(target_expanded, non_pad_mask, 0)
    else:
        non_pad_mask = None
    
    # Cast target to int64 for gather_d
    target_expanded = cast(target_expanded, mindspore.int64)
    
    # Gather values at target indices along dimension 1
    loss = neg(gather_d(input, target_dim, target_expanded))
    
    # Apply weights if provided
    if weight is not None:
        # Get weights for each target position
        target_for_weights = squeeze(target_expanded, target_dim) if target_expanded.ndim > 1 else target_expanded
        loss_weights = index_select(weight, 0, target_for_weights)
        if target_expanded.ndim > 1:
            loss_weights = expand_dims(loss_weights, target_dim)
        loss = mul(loss, loss_weights)
    else:
        loss_weights = ones_like(loss, dtype=None)
    
    # Apply ignore_index mask
    if ingore_index is not None and non_pad_mask is not None:
        loss = masked_fill(loss, non_pad_mask, 0.0)
        loss_weights = masked_fill(loss_weights, non_pad_mask, 0.0)
    
    # Squeeze the target_dim
    loss = squeeze(loss, target_dim)
    loss_weights = squeeze(loss_weights, target_dim)
    
    # Apply reduction
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return sum(loss, None, False, None)
    else:  # 'mean'
        total_weight = sum(loss_weights, None, False, None)
        if total_weight > 0:
            return div(sum(loss, None, False, None), total_weight)
        else:
            return sum(loss, None, False, None)

def nllloss_2d(input, target, weight, reduction, ingore_index):
    input = reshape(transpose_view(input, 1, -1), (-1, input.shape[1]))
    target = reshape(target, (-1,))
    out = nllloss(input, target, weight, reduction, ingore_index)
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

def conv1d_padding(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, training=True):
    return conv1d(input, weight, bias, stride, padding, dilation, groups, training)

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
        pad = (padding[0], padding[0], padding[1], padding[1])
    elif isinstance(padding, int):
        pad = (padding,) * 4
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
    input = clone(input)
    indices = non_zero(mask)
    # 如果 src 是 1D，按顺序取值
    updates = narrow(reshape(value, (-1,)), 0, 0, indices.shape[0])
    # 更新 tensor
    out = scatter_nd_update(input, indices, updates)
    return out

def max_pool2d(input, kernel_size, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    out, indices = legacy.max_pool_with_argmax_v2(input, kernel_size, stride, padding, dilation, ceil_mode, mindspore.int64)

    if return_indices:
        return out, indices
    return out

def baddbmm(input, batch1, batch2, alpha=1, beta=1):
    return add(mul(input, beta), mul(bmm(batch1, batch2), alpha))

def softplus(input, beta=1, threshold=20):
    return legacy.softplus(input)

def gather_nd(input, indices):
    return legacy.gather_nd(input, indices)

def meshgrid(input, lambd):
    return legacy.meshgrid(input, lambd)

def addcmul(input, tensor1, tensor2, value=1.0):
    return legacy.addcmul(input, tensor1, tensor2, mindspore.Tensor(value))

def addmm(input, mat1, mat2, alpha=1.0, beta=1.0):
    return add(mul(input, beta), mul(matmul(mat1, mat2), alpha))

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

def inplace_relu(input):
    return inplace_copy(input, legacy.re_lu(input))

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
    if isinstance(boundaries, mindtorch.Tensor):
        boundaries = boundaries.tolist()
    
    if not boundaries:
        return zeros_like(input, mindspore.int64)
    epsilon_ = 0. if right else 1.e-6
    boundaries = [boundary + epsilon_ for boundary in boundaries]
    return cast(legacy.bucketize(input, boundaries), mindspore.int64)

def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    return legacy.col2_im(input, output_size, kernel_size, dilation, padding, stride)

def randperm(n, generator, dtype):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return legacy.randperm_v2(n, seed, offset, dtype)

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
    if (input.sum(-1) == 0).any():
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

def dropout2d(input_x, p):
    return legacy.dropout2_d(input_x, p)

def linalg_qr(input_x, mode):
    full_matrices = 'mode' == 'complete'
    return legacy.qr(input_x, full_matrices)

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
    return legacy.bernoulli(input, seed, offset)

def arange(start, end, step, dtype):
    end = type(start)(end)
    step = type(start)(step)
    if dtype is not None:
        return cast(legacy.range(start, end, step, 1000000), dtype)
    return legacy.range(start, end, step, 1000000)

def inplace_fill_scalar(input, value):
    inplace_copy(input, fill_scalar(input.shape, value, input.dtype))
    return input

def inplace_normal(input, mean, std, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    out = legacy.standard_normal(input.shape, seed.item() + 1, offset.item() + 1)
    value = add(mul(out, std), mean)
    return inplace_copy(input, value)

def inplace_uniform(input, from_, to_, generator_):
    seed, offset = generator_._step(12)  # pylint: disable=protected-access
    if input.dtype.is_floating_point:
        uniform_real = legacy.uniform_real(tuple(input.shape), seed.item() + 1, offset.item() + 1)
        value = add(mul(uniform_real, sub(to_, from_)), from_)
    else:
        value = legacy.uniform_int(input.shape,
                                    mindspore.tensor(from_, dtype=mindspore.int32),
                                    mindspore.tensor(to_, dtype=mindspore.int32), seed.item() + 1, offset.item() + 1)
    return inplace_copy(input, value)

def right_shift(input, other):
    return legacy.right_shift(input, other)

def inplace_fill_tensor(input, value):
    return inplace_copy(input, fill_tensor(input.shape, value, None))
    return input

def search_sorted(sorted_sequence, values, sorter, dtype, right):
    return legacy.search_sorted(sorted_sequence, values, sorter, dtype, right)

def einsum(equation, operands):
    return legacy.einsum(operands, equation)

def unique2(input, sorted, return_inverse, return_counts):
    outs = legacy.unique(input)
    return outs + (None,)

def logaddexp(input, other):
    m = maximum(input, other)
    abs_val = abs(sub(input, other))
    exp_val = exp(neg(abs_val))
    y = add(m, log1p(exp_val))
    return y

def kl_div(input, target, reduction, log_target):
    if log_target:
        target = log(target)

    if reduction == 'batchmean':
        kl_div_sum = legacy.kl_div_loss(input, target, 'sum')
        # shape = input.shape
        # batch_size = shape[0]
        # return div(kl_div_sum, batch_size)
        return kl_div_sum

    if reduction == 'mean':
        kl_div_sum = legacy.kl_div_loss(input, target, 'sum')
        shape = input.shape
        total_size = 1
        for dim in shape:
            total_size = total_size * dim
        return div(kl_div_sum, total_size)

    return legacy.kl_div_loss(input, target, reduction)

def scatter_nd_update(input, indices, updates):
    return legacy.scatter_nd_update(input, indices, updates, True)

def inplace_exponential(self, lambd, generator):

    u = rand(self.shape, generator, self.dtype)
    # 逆变换采样
    out = div(neg(log(sub(1, u))), lambd)
    inplace_copy(self, out)
    return self

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
        input_indices = mindspore.Tensor(Tensor_(index.shape, dtype=mindspore.int32))
    else:
        input_indices = mindspore.tensor(index.astype(np.int32))
    out = gather(reshape(self, (-1,)), input_indices, 0, 0)
    return out

def fft(input, n=None, dim=-1, norm="backward"):
    if norm is None:
        norm="backward"
    if input.shape[dim] < n:
        pad_inf = (0, n - input.shape[dim])
        pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
        input = pad_v3(input, pad_dims, 'constant', 0, True)
    else:
        input = narrow(input, dim, 0, n)
    return legacy.fft_with_size(input, input.ndim, False, False, norm, True, ())

def triu_indices(row, col, offset, dtype):
    return legacy.triu_indices(row, col, offset, dtype)

def cumprod(input, dim, dtype):
    out = legacy.cum_prod(input, dim, False, False)
    if dtype is not None:
        out = cast(out, dtype)
    return out

def lerp(input, end, weight):
    return legacy.lerp(input, end, weight)

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
        return pad_v3(input, new_pad, mode)
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
    if mode == 'constant' and value == 0 and len(new_pad) > 6:
        paddings = ()
        for i in range(input.ndim-1, -1, -1):
            paddings += ((new_pad[2*i], new_pad[2*i+1]),)
        return legacy.pad(input, paddings)
    return pad_v3(input, new_pad, mode, value)

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
    return elu(input, alpha)

def hardsigmoid(input):
    """Hardsigmoid activation: clamp((x + 3) / 6, 0, 1)"""
    x_plus_3 = add(input, 3.0)
    x_div_6 = div(x_plus_3, 6.0)
    return clamp_scalar(x_div_6, 0.0, 1.0)

def fast_gelu(x):
    """Fast GELU approximation"""
    return gelu(x, approximate='tanh')

def swiglu(x, dim=-1):
    """Swish-Gated Linear Unit: swish(x[..., :d]) * x[..., d:] where d = x.shape[dim] // 2"""
    split_size = x.shape[dim] // 2
    x1, x2 = legacy.split(x, split_size, dim)
    return legacy.mul(silu(x1), x2)

def rotary_position_embedding(x, cos, sin, mode=0):
    """Rotary Position Embedding"""
    import mindspore
    return mindspore.ops.auto_generate.gen_ops_def.apply_rotary_pos_emb_(x, cos, sin, mode)

def setitem(self, index, value):
    if isinstance(value, numbers.Number):
        value = mindspore.tensor(value, dtype=self.dtype)
    out = _tensor_setitem_origin(self, index, value)
    if isinstance(out, tuple):
        out = out[0]
    self.data = out
    # legacy.assign(self, out)
    return self

def _do_select(self, dim, index, dim_index, self_shape):
    """call select view operator"""
    if not self_shape:
        raise TypeError("Invalid index of a 0-dim tensor.")
    dim_size = self_shape[dim]
    if index >= dim_size or index < -dim_size:
        raise IndexError(f"Index {index} is out of bounds for dimension {dim_index} with size {dim_size}")
    index = index + dim_size if index < 0 else index
    return select_ext_view(self, dim, index)

def _do_slice(self, dim, index, self_shape):
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
    """Record indexes remained to be used by index operation"""
    if len(remain_indexes) > dim:
        remain_indexes[dim] = index
        return remain_indexes

    while dim > len(remain_indexes):
        # use slice(None, None, None) to indicate unused dim
        remain_indexes.append(py_slice(None, None, None))

    remain_indexes.append(index)
    return remain_indexes

def _process_dim_in_multi_dim_index(prev_result, orig_tensor, index, dim, indexed_dims, dim_index, remain_indexes, prev_shape):
    """Process dim in multi dim index"""
    if isinstance(index, bool):
        result = expand_dims(prev_result, dim)
        index_for_bool = ones((1,), mindtorch.bool_) if index else zeros((0,), mindtorch.bool_)
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
                result = _do_select(prev_result, dim, int(index.item()), dim_index, prev_shape)
                del prev_shape[dim]
                return result, dim, remain_indexes, prev_shape
            # process index with Tensor bool type (scalar)
            result = expand_dims(prev_result, dim)
            index_for_bool = ones((1,), mindtorch.bool_) if index else zeros((0,), mindtorch.bool_)
            _record_tensor_index(index_for_bool, remain_indexes, dim)
            prev_shape.insert(dim, 1)
            dim += 1
            return result, dim, remain_indexes, prev_shape
        # Multi-dimensional boolean tensor or integer tensor
        _record_tensor_index(index, remain_indexes, dim)
        dim += 1
        return result, dim, remain_indexes, prev_shape
    raise IndexError(f"Invalid tensor index type {type(index)}")

def _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims):
    """Process indexes in tuple"""
    self_viewed = self
    self_viewed_shape = list(self.shape)
    dim = 0
    for i, index in enumerate(indexes):
        if isinstance(index, (list, tuple, np.ndarray)):
            index_np = np.array(index) if isinstance(index, (list, tuple)) else index
            if index_np.dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
                                  np.float16, np.float32, np.float64):
                index = mindspore.tensor(index_np, dtype=mindtorch.int64)
            elif index_np.dtype == np.bool_:
                index = mindspore.tensor(index_np, dtype=mindtorch.bool_)
            else:
                raise TypeError(f"Index {index} contain unsupported elements")
        self_viewed, dim, remain_indexes, self_viewed_shape = _process_dim_in_multi_dim_index(
            self_viewed, self, index, dim, indexed_dims, i, remain_indexes, self_viewed_shape)
    return self_viewed, remain_indexes

def getitem(self, index):
    """Handle tensor getitem with proper boolean indexing support"""
    if isinstance(index, bool):
        self_viewed = expand_dims(self, 0)
        index_for_bool = ones((1,), mindtorch.bool_) if index else zeros((0,), mindtorch.bool_)
        return _tensor_getitem_origin(self_viewed, index_for_bool)
    if isinstance(index, int):
        return _do_select(self, 0, index, 0, list(self.shape))
    if isinstance(index, py_slice):
        result = _do_slice(self, 0, index, list(self.shape))
        return result
    if index is None:
        return expand_dims(self, 0)
    if isinstance(index, type(...)):
        return self
    indexes = _wrap_index_to_tuple(index)
    indexed_dims = _count_indexed_dims(indexes)
    if self.ndim < indexed_dims:
        raise IndexError(f"too many indices for tensor with dimension size {self.ndim}")
    remain_indexes = []
    self_viewed, remain_indexes = _process_multi_dim_index(self, indexes, remain_indexes, indexed_dims)
    if not remain_indexes:
        return self_viewed
    
    # Use _tensor_getitem_origin for the remaining indexes
    if len(remain_indexes) == 1:
        return _tensor_getitem_origin(self_viewed, remain_indexes[0])
    else:
        return _tensor_getitem_origin(self_viewed, tuple(remain_indexes))

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
                setitem(index, (py_slice(current_pos, add(current_pos, repeat_count))), i)
            current_pos = add(current_pos, repeat_count)

        output = index_select(input_tensor, dim, index)

    return output

def repeat_interleave_tensor(input, repeats, dim, output_size):
    return repeat_interleave_int(input, repeats, dim, output_size)

def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))
    return indices, dimension

def unfold(input, dimension, size, step):
    _indices, _dimension = _get_unfold_indices(input.shape, dimension, size, step)
    indices = mindspore.tensor(_indices, dtype=mindspore.int64)
    output = gather(input, indices, _dimension, 0)
    output = transpose_view(output, _dimension + 1, -1)
    return output

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
    """
    Scaled Dot Product Attention implementation for GPU.
    """
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale

    if enable_gqa:
        key = contiguous(repeat_kv(key, query.shape[-3] // key.shape[-3]))
        value = contiguous(repeat_kv(value, query.shape[-3] // value.shape[-3]))

    # Compute attention weights first to get the correct shape
    attn_weight = mul(matmul(query, transpose_view(key, -2, -1)), scale_factor)
    
    # Create attention bias with the same shape as attn_weight
    if attn_mask is None:
        attn_bias = zeros(attn_weight.shape, query.dtype)
    else:
        attn_bias = zeros(attn_mask.shape, query.dtype)

    if is_causal:
        assert attn_mask is None, "Cannot use both is_causal and attn_mask"
        temp_mask = tril(ones((L, S), mindtorch.bool), diagonal=0)
        # Broadcast temp_mask to match attn_weight shape if needed
        if temp_mask.shape != attn_weight.shape:
            temp_mask = broadcast_to(temp_mask, attn_weight.shape)
        attn_bias = masked_fill(attn_bias, logical_not(temp_mask), mindtorch.finfo(attn_bias.dtype).min)

    if attn_mask is not None:
        if attn_mask.dtype == mindtorch.bool:
            if attn_mask.ndim == 3:
                attn_mask = squeeze(attn_mask, 0)
            # Broadcast attn_mask to match attn_weight shape if needed
            if attn_mask.shape != attn_weight.shape:
                attn_mask = broadcast_to(attn_mask, attn_weight.shape)
            attn_bias = masked_fill(attn_bias, logical_not(attn_mask), mindtorch.finfo(attn_bias.dtype).min)
        else:
            # Broadcast attn_mask to match attn_weight shape if needed
            if attn_mask.shape != attn_weight.shape:
                attn_mask = broadcast_to(attn_mask, attn_weight.shape)
            attn_bias = add(attn_mask, attn_bias)

    attn_weight = add(attn_weight, attn_bias)
    attn_weight = softmax(attn_weight, -1)
    attn_weight = dropout(attn_weight, dropout_p, training=True)
    return matmul(attn_weight, value)

def raw_sgd(param, grad, lr, dampening, weight_decay, nesterov, accum, momentum, stat):
    return legacy.sgd(param, grad, lr, accum, momentum, stat, dampening, weight_decay, nesterov)

def inplace_index_add(input, dim, index, source, alpha):
    if alpha != 1:
        source = mul(alpha, source)
    out = legacy.index_add(input, cast(index, mindspore.int32), source, dim, True, True)
    inplace_copy(input, out)
    return input

def unique_consecutive(input, return_inverse, return_counts, dim):
    """Native implementation of unique_consecutive for GPU.
    Always returns (output, inverse, counts) tuple - the caller will filter based on flags.
    """
    # Handle empty input
    if input.numel() == 0:
        empty_output = empty((0,), input.dtype)
        empty_idx = zeros(input.shape, mindspore.int64)
        empty_counts = empty((0,), mindspore.int64)
        return empty_output, empty_idx, empty_counts
    
    # Handle dim=None (flatten)
    if dim is None:
        input_flat = reshape(input, (-1,))
        return _unique_consecutive_1d(input_flat, return_inverse, return_counts, input.shape)
    
    # For now, handle multi-dimensional case by processing along the specified dimension
    # This is a simplified version - for full support, we'd need to handle each slice
    # For most use cases, dim=None should be sufficient
    input_flat = reshape(input, (-1,))
    return _unique_consecutive_1d(input_flat, return_inverse, return_counts, input.shape)

def _unique_consecutive_1d(input, return_inverse, return_counts, original_shape):
    """Helper function for 1D unique_consecutive.
    Always returns (output, inverse, counts) tuple.
    """
    input_len = input.shape[0]
    
    if input_len == 0:
        empty_output = empty((0,), input.dtype)
        empty_idx = zeros(original_shape, mindspore.int64)
        empty_counts = empty((0,), mindspore.int64)
        return empty_output, empty_idx, empty_counts
    
    if input_len == 1:
        output = input
        inverse = zeros(original_shape, mindspore.int64)
        counts = ones((1,), mindspore.int64)
        return output, inverse, counts
    
    # Compare consecutive elements: input[1:] != input[:-1]
    input_next = narrow(input, 0, 1, input_len - 1)
    input_prev = narrow(input, 0, 0, input_len - 1)
    diff = not_equal(input_next, input_prev)
    
    # First element is always a change point, add True at the end too
    first_true = ones((1,), mindtorch.bool_)
    last_true = ones((1,), mindtorch.bool_)
    change_mask = concat([first_true, diff, last_true], 0)
    
    # Get indices where changes occur (where mask is True)
    change_indices_nd = non_zero(change_mask)
    if change_indices_nd.shape[0] == 0:
        # All elements are the same
        output = expand_dims(narrow(input, 0, 0, 1), 0)
        inverse = zeros(original_shape, mindspore.int64)
        count_val = cast(ones((1,), mindspore.int32), mindspore.int64) * input_len
        return output, inverse, count_val
    
    # Extract column indices (change_indices_nd is 2D: [N, 1] for 1D input)
    change_indices = squeeze(change_indices_nd, -1) if change_indices_nd.ndim > 1 else change_indices_nd
    change_indices = cast(change_indices, mindspore.int32)
    
    # Extract unique values (all but last change index)
    num_changes = change_indices.shape[0]
    if num_changes > 1:
        unique_indices = narrow(change_indices, 0, 0, num_changes - 1)
        output = index_select(input, 0, unique_indices)
    else:
        output = expand_dims(narrow(input, 0, 0, 1), 0)
    
    # Always build inverse indices using cumsum approach
    # Create a mask where each group gets a unique ID
    # Use cumsum on the change mask to create group IDs
    change_mask_int = cast(change_mask, mindspore.int32)
    # Use legacy.cum_sum directly
    group_ids = legacy.cum_sum(change_mask_int, 0, False, False)
    # Subtract 1 because first element starts at 1
    inverse = sub(group_ids, ones_like(group_ids, None))
    # Take only the first input_len elements
    inverse = narrow(inverse, 0, 0, input_len)
    # Reshape to original shape if needed
    if inverse.shape != original_shape:
        inverse = reshape(inverse, original_shape)
    inverse = cast(inverse, mindspore.int64)
    
    # Always build counts: differences between consecutive change points
    if num_changes > 1:
        # Compute differences: change_indices[i+1] - change_indices[i]
        counts_list = []
        for i in range(num_changes - 1):
            start_idx = int(change_indices[i].item())
            end_idx = int(change_indices[i + 1].item())
            counts_list.append(end_idx - start_idx)
        counts = mindspore.Tensor(counts_list, dtype=mindspore.int64)
    else:
        counts = cast(ones((1,), mindspore.int32), mindspore.int64) * input_len
    
    # Always return all three values
    return output, inverse, counts

def full_like(input, fill_value, dtype=None):
    if dtype is None:
        dtype = input.dtype
    size = input.shape
    if isinstance(fill_value, numbers.Number):
        return fill_scalar(size, fill_value, dtype)
    else:
        return fill_tensor(size, fill_value, dtype)

def isfinite(input):
    return legacy.is_finite(input)