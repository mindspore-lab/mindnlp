import mindspore
from mindspore._c_expression import _empty_instance
from ..configs import use_pyboost, ON_A1, ON_ORANGE_PI
from .._op_prim.ascend import legacy, pyboost


def empty(size, dtype):
    return _empty_instance(size, dtype=dtype, device='Ascend')

def reshape(x, shape):
    """
    Reshape the input tensor to the given shape.

    Args:
        x (Tensor): The input tensor.
        shape (tuple): The target shape.

    Returns:
        Tensor: The reshaped tensor.
    """
    if use_pyboost():
        return pyboost.reshape_op(x, shape)
    else:
        return legacy.reshape(x, shape)

def contiguous(x):
    """
    Returns a contiguous tensor containing the same data as the input tensor.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor: The contiguous tensor.
    """
    if use_pyboost:
        return pyboost.contiguous_op(x)
    else:
        return x

def select_ext_view(input, dim, index):
    """
    Selects a slice from the input tensor along the specified dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension along which to select the slice.
        index (int): The index of the slice to select.

    Returns:
        Tensor: The selected slice.
    """
    if use_pyboost():
        return pyboost.select_ext_view_op(input, dim, index)
    else:
        return legacy.select_view(input, index, dim)

def inplace_copy(self, value):
    """
    Copies the data from the given tensor to the current tensor.

    Args:
        value (Tensor): The tensor from which to copy the data.
    """
    if use_pyboost:
        return pyboost.inplace_copy_op(self, value)
    else:
        self.assign_value(value)
        return self

def slice(input, dim, start, end, step):
    """
    Slices the input tensor along the specified dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension along which to slice.
        start (int): The starting index of the slice.
        end (int): The ending index of the slice.
        step (int): The step size of the slice.

    Returns:
        Tensor: The sliced tensor.
    """
    if use_pyboost():
        return pyboost.slice_ext_op(input, dim, start, end, step)
    else:
        return legacy.slice(input, dim, start, end, step)

def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq):
    """
    Applies embedding to the input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The embedding weight tensor.
        padding_idx (int): The index of the padding element.
        max_norm (float): The maximum norm of the embedding vectors.
        norm_type (float): The p-norm to use for normalization.
        scale_grad_by_freq (bool): Whether to scale the gradient by frequency.
        sparse (bool): Whether to use sparse gradients.

    Returns:
        Tensor: The embedded tensor.
    """
    return pyboost.embedding_op(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq)

def add(input, other, alpha): # pylint: disable=unused-argument
    """
    Adds two tensors element-wise.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The other tensor.
        alpha (float): The scaling factor for the other tensor.

    Returns:
        Tensor: The result of the addition.
    """
    if use_pyboost():
        return pyboost.add_ext_op(input, other, alpha)
    if alpha == 1.0:
        return legacy.add(input, other)
    return legacy.add(input, legacy.mul(other, alpha))

def layer_norm(input, normalized_shape, weight, bias, eps=1e-5):
    """
    Applies layer normalization to the input tensor.

    Args:
        input (Tensor): The input tensor.
        normalized_shape (tuple): The shape of the input tensor to be normalized.
        weight (Tensor): The weight tensor.
        bias (Tensor): The bias tensor.
        eps (float): The epsilon value for numerical stability.

    Returns:
        Tensor: The normalized tensor.
    """
    if use_pyboost():
        return pyboost.layer_norm_ext_op(input, normalized_shape, weight, bias, eps)
    if weight is not None:
        begin_axis = input.ndim - weight.ndim
    else:
        begin_axis = -1
    return legacy.layer_norm(input, weight, bias, begin_axis, begin_axis, eps)

def expand_dims(input, axis):
    """
    Adds an extra dimension to the input tensor.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis along which to add the dimension.

    Returns:
        Tensor: The expanded tensor.
    """
    if use_pyboost():
        return pyboost.expand_dims_op(input, axis)
    return legacy.expand_dims(input, axis)

def cast(input, dtype):
    """
    Casts the input tensor to the specified data type.

    Args:
        input (Tensor): The input tensor.
        dtype (str): The target data type.

    Returns:
        Tensor: The casted tensor.
    """
    return legacy.cast(input, dtype)

def sub(input, other, alpha):
    """
    Subtracts the other tensor from the input tensor.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The tensor to subtract.
        alpha (float): The scale factor for the other tensor.

    Returns:
        Tensor: The result of the subtraction.
    """
    if use_pyboost():
        return pyboost.sub_ext_op(input, other, alpha)
    return legacy.sub(input, legacy.mul(other, alpha))

def mul(input, other):
    """
    Multiplies the input tensor with the other tensor.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The tensor to multiply.

    Returns:
        Tensor: The result of the multiplication.
    """
    if use_pyboost():
        return pyboost.mul_op(input, other)
    return legacy.mul(input, other)

def dense(input, weight, bias=None):
    """
    Performs a dense (fully connected) operation.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.
        bias (Tensor, optional): The bias tensor. Defaults to None.

    Returns:
        Tensor: The result of the dense operation.
    """
    if use_pyboost():
        return pyboost.dense_op(input, weight, bias)
    return legacy.dense(input, weight, bias)

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
    if use_pyboost():
        return pyboost.transpose_ext_view_op(input, dim0, dim1)
    ranks = list(range(input.ndim))
    rank0 = ranks[dim0]
    rank1 = ranks[dim1]
    ranks[dim0] = rank1
    ranks[dim1] = rank0
    return legacy.transpose(input, ranks)

def matmul(input, other):
    """
    Performs a matrix multiplication of the input tensor with another tensor.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The other tensor.

    Returns:
        Tensor: The result of the matrix multiplication.
    """
    if use_pyboost():
        return pyboost.matmul_ext_op(input, other)
    return legacy.mat_mul(input, other)

def div(input, other):
    """
    Divides the input tensor by another tensor.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The other tensor.

    Returns:
        Tensor: The result of the division.
    """
    if use_pyboost():
        return pyboost.div_op(input, other)
    return legacy.div(input, other)

def divmod(input, other, rounding_mode):
    """
    Divides the input tensor by another tensor and returns both the quotient and the remainder.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The other tensor.
        rounding_mode (str): The rounding mode to use.

    Returns:
        Tuple[Tensor, Tensor]: The quotient and the remainder.
    """
    if use_pyboost():
        return pyboost.divmod_op(input, other, rounding_mode)
    if rounding_mode == 'floor':
        return legacy.floor_div(input, other)
    elif rounding_mode == 'trunc':
        return legacy.truncate_div(input, other)
    else:
        raise ValueError(f'Invalid rounding mode: {rounding_mode}')

def softmax(input, axis=-1):
    """
    Computes the softmax of the input tensor along the specified axis.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis along which to compute the softmax.

    Returns:
        Tensor: The softmax of the input tensor.
    """
    if use_pyboost():
        return pyboost.softmax_impl(input, axis)
    return legacy.softmax(input, axis)

def permute(input, axes=None):
    """
    Transposes the dimensions of the input tensor according to the specified axes.

    Args:
        input (Tensor): The input tensor.
        axes (Tuple[int]): The axes to transpose.

    Returns:
        Tensor: The transposed tensor.
    """
    if use_pyboost():
        return pyboost.transpose_view_op(input, axes)
    return legacy.transpose(input, axes)

def gelu(input, approximate):
    """
    Computes the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor: The GELU activation of the input tensor.
    """
    if use_pyboost():
        return pyboost.gelu_ext_op(input, approximate)
    return legacy.ge_lu(input)

def tanh(input):
    """
    Computes the hyperbolic tangent of the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor: The hyperbolic tangent of the input tensor.
    """
    if use_pyboost():
        return pyboost.tanh_op(input)
    return legacy.tanh(input)

def broadcast_to(input, shape):
    """
    Broadcasts the input tensor to the specified shape.

    Args:
        input (Tensor): The input tensor.
        shape (Tuple[int]): The shape to broadcast to.

    Returns:
        Tensor: The broadcasted tensor.
    """
    if use_pyboost():
        return pyboost.broadcast_to_view_op(input, shape)
    return legacy.broadcast_to(input, shape)

def split_tensor(tensor, split_size_or_sections, dim):
    """
    Splits a tensor into multiple sub-tensors.

    Args:
        tensor (Tensor): The input tensor.
        split_size_or_sections (Union[int, Tuple[int]]): The size or number of sections to split the tensor into.
        dim (int): The dimension along which to split the tensor.

    Returns:
        List[Tensor]: The list of split sub-tensors.
    """
    if use_pyboost():
        return pyboost.split_tensor_op(tensor, split_size_or_sections, dim)
    return legacy.split(tensor, split_size_or_sections, dim)

def squeeze(input, dim):
    """
    Removes dimensions of size 1 from the shape of the input tensor.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, Tuple[int]]): The dimensions to squeeze.

    Returns:
        Tensor: The squeezed tensor.
    """
    if use_pyboost():
        return pyboost.squeeze_impl(input, dim)
    return legacy.squeeze(input, dim)

def zeros(shape, dtype):
    """
    Returns a tensor filled with zeros.

    Args:
        shape (Union[int, Tuple[int]]): The shape of the tensor.
        dtype (str): The data type of the tensor.

    Returns:
        Tensor: The tensor filled with zeros.
    """
    return legacy.zeros(shape, dtype)

def equal(input, other):
    """
    Returns a tensor with boolean values, indicating element-wise equality.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The tensor to compare with.

    Returns:
        Tensor: The tensor with boolean values.
    """
    if use_pyboost():
        return pyboost.equal_ext_op(input, other)
    return legacy.equal(input, other).all()

def eq(input, other):
    """
    Returns a tensor with boolean values, indicating element-wise equality.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The tensor to compare with.

    Returns:
        Tensor: The tensor with boolean values.
    """
    if use_pyboost():
        return pyboost.equal_op(input, other)
    return legacy.equal(input, other)


def sum(input, dim, keepdim, dtype):
    """
    Returns the sum of elements over a specified dimension.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, Tuple[int]]): The dimensions to sum over.
        keepdim (bool): Whether to keep the dimensions of size one.

    Returns:
        Tensor: The tensor with summed elements.
    """
    if use_pyboost():
        return pyboost.sum_ext_op(input, dim, keepdim, dtype)
    return legacy.reduce_sum(input.astype(dtype), dim, keepdim)

def dropout(input, p, seed, offset):
    """
    Returns a tensor with dropout applied element-wise.

    Args:
        input (Tensor): The input tensor.
        p (float): The dropout probability.
        seed (int): The random seed.

    Returns:
        Tensor: The tensor with dropout applied.
    """
    if use_pyboost():
        return pyboost.dropout_ext_op(input, p, seed, offset)
    return legacy.dropout(input, 1-p, 0, 0)

def clone(input):
    """
    Returns a copy of the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor: The copied tensor.
    """
    if use_pyboost():
        return pyboost.clone_op(input)
    return legacy.identity(input)

def inplace_normal(input, mean, std, generator):
    """
    Returns a tensor with normal distribution applied element-wise.

    Args:
        input (Tensor): The input tensor.
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.
        seed (int): The random seed.

    Returns:
        Tensor: The tensor with normal distribution applied.
    """
    seed, offset = generator._step(12)
    if use_pyboost():
        return pyboost.inplace_normal_op(input, mean, std, seed, offset)
    return legacy.normal(input, mean, std, 0, 0)

def reduce_all(input, dim, keepdim):
    """
    Returns the sum of all elements in the tensor.

    Args:
        input (Tensor): The input tensor.
        dim (int): The dimension to reduce.
        keepdim (bool): Whether to keep the reduced dimension.

    Returns:
        Tensor: The tensor with the sum of all elements.
    """
    if use_pyboost():
        return pyboost.reduce_all_impl(input, dim, keepdim)
    return legacy.reduce_all(input, dim, keepdim)

def masked_fill(input, mask, value):
    """
    Fills elements of the input tensor with the specified value where the mask is True.

    Args:
        input (Tensor): The input tensor.
        mask (Tensor): The mask tensor.
        value (float): The value to fill.

    Returns:
        Tensor: The tensor with elements filled.
    """
    if use_pyboost():
        return pyboost.masked_fill_op(input, mask, value)
    return legacy.masked_fill(input, mask, value)

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
    if use_pyboost():
        return pyboost.isin(input, test_elements, assume_unique, invert)
    return legacy.isin(input, test_elements, assume_unique, invert)

def pad_v3(input, new_pad, mode, value=None, contiguous=True):
    if input.dtype == mindspore.bool_:
        input = cast(input, mindspore.int8)
        out = legacy.pad_v3(input, new_pad, int(value), mode, contiguous)
        return cast(out, mindspore.bool_)
    return legacy.pad_v3(input, new_pad, value, mode, contiguous)

def log_softmax(input, axis=-1, dtype=None):
    """
    Computes the log softmax of the input tensor along the specified axis.

    Args:
        input (Tensor): The input tensor.
        axis (int): The axis along which to compute the log softmax.
        dtype (dtype): The data type of the output tensor.

    Returns:
        Tensor: The tensor with log softmax values.
    """
    if use_pyboost():
        return pyboost.log_softmax_impl(input, axis)
    return legacy.log_softmax(input, axis)

def not_equal(input, other):
    """
    Computes the element-wise comparison of two tensors for inequality.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The other tensor.

    Returns:
        Tensor: The tensor with boolean values indicating whether elements are not equal.
    """
    if use_pyboost():
        return pyboost.not_equal_op(input, other)
    return legacy.not_equal(input, other)

def chunk(input, chunks, dim=0):
    """
    Splits a tensor into a specified number of chunks.

    Args:
        input (Tensor): The input tensor.
        chunks (int): The number of chunks to split the tensor into.
        dim (int): The dimension along which to split the tensor.

    Returns:
        Tensor: The tensor split into chunks.
    """
    if use_pyboost():
        return pyboost.chunk_op(input, chunks, dim)
    return legacy.split(input, dim, chunks)

def ones(shape, dtype):
    """
    Returns a tensor filled with ones.

    Args:
        shape (tuple): The shape of the tensor.
        dtype (dtype): The data type of the tensor.

    Returns:
        Tensor: The tensor filled with ones.
    """
    return legacy.ones(shape, dtype)

def greater(input, other):
    """
    Returns a tensor with boolean values indicating whether elements in the input tensor are greater than those in the other tensor.

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The other tensor.

    Returns:
        Tensor: The tensor with boolean values indicating whether elements are greater.
    """
    if use_pyboost():
        return pyboost.greater_op(input, other)
    return legacy.greater(input, other)

def randint(low, high, shape, generator, dtype):
    """
    Returns a tensor filled with random integers from low (inclusive) to high (exclusive).

    Args:
        low (int): The lower bound of the range.
        high (int): The upper bound of the range.
        shape (tuple): The shape of the tensor.
        dtype (dtype): The data type of the tensor.

    Returns:

        Tensor: The tensor filled with random integers.
    """
    seed, offset = generator._step(12)

    if use_pyboost():
        return pyboost.randint_op(low, high, shape, seed, offset, dtype)
    value = legacy.uniform_int(shape,
                                mindspore.tensor(low, dtype=mindspore.int32),
                                mindspore.tensor(high, dtype=mindspore.int32), 0, 0)
    return value
    
def nllloss(input, target, weight, reduction, ingore_index):
    if use_pyboost():
        return pyboost.nllloss_impl(input, target, weight, reduction, ingore_index)
    return legacy.nll_loss(input, target, weight, reduction, ingore_index)

def clamp_scalar(value, min_value, max_value):
    if use_pyboost():
        return pyboost.clamp_scalar_op(value, min_value, max_value)
    if min_value is not None:
        value = legacy.maximum(value, min_value)
    if max_value is not None:
        value = legacy.minimum(value, max_value)
    return value

def cumsum(self, dim, dtype):
    if use_pyboost():
        return pyboost.cumsum_ext_op(self, dim, dtype)
    return legacy.cum_sum(self, dim, False, False)

def reduce_any(input, axis, keepdims):
    if use_pyboost():
        return pyboost.reduce_any_impl(input, axis, keepdims)
    return legacy.reduce_any(input, axis, keepdims)

def concat(tensors, axis):
    if use_pyboost():
        return pyboost.concat_impl(tensors, axis)
    return legacy.concat(tensors, axis)

def gather_d(input, dim, index):
    if use_pyboost():
        return pyboost.gather_d_op(input, dim, index)
    return legacy.gather_d(input, dim, index)

def greater_equal(input, other):
    if use_pyboost():
        return pyboost.greater_equal_op(input, other)
    return legacy.greater_equal(input, other)

def less(input, other):
    if use_pyboost():
        return pyboost.less_op(input, other)
    return legacy.less(input, other)

def less_equal(input, other):
    if use_pyboost():
        return pyboost.less_equal_op(input, other)
    return legacy.less_equal(input, other)

def select(condition, input, other):
    if use_pyboost():
        return pyboost.select_op(condition, input, other)
    return legacy.select(condition, input, other)

def mean(input, axis, keepdims, dtype):
    if use_pyboost():
        return pyboost.mean_ext_op(input, axis, keepdims, dtype)
    return legacy.reduce_mean(input, axis, keepdims)

def index(input, index):
    if use_pyboost():
        return pyboost.index_op(input, index)
    return legacy.index(input, index)

def scatter(input, dim, index, src):
    if use_pyboost():
        return pyboost.scatter_op(input, dim, index, src)
    return legacy.tensor_scatter_elements(input, index, src, dim)

def tril(input, diagonal=0):
    if use_pyboost():
        return pyboost.tril_ext_op(input, diagonal)
    return legacy.tril(input, diagonal)

def triu(input, diagonal=0):
    if use_pyboost():
        return pyboost.triu_impl(input, diagonal)
    return legacy.triu(input, diagonal)

def inplace_index_put(input, indices, values, accumulate):
    if use_pyboost():
        return pyboost.inplace_index_put_op(input, indices, values, accumulate)
    return legacy.tensor_scatter_elements(input, indices, values, accumulate)

def zeros_like(input, dtype):
    if use_pyboost():
        return pyboost.zeros_like_ext_op(input, dtype)
    return legacy.zeros_like(input)

def ones_like(input, dtype):
    if use_pyboost():
        return pyboost.ones_like_ext_op(input, dtype)
    return legacy.ones_like(input)

def tile(input, multiples):
    return legacy.tile(input, multiples)

def arange(start, end, step, dtype):
    if use_pyboost():
        return pyboost.arange_op(start, end, step, dtype)
    return legacy.range(start, end, step, 100000)

def fill_scalar(input, value, dtype):
    if use_pyboost():
        return pyboost.fill_scalar_op(input, value, dtype)
    return legacy.fill(input, value)

def stop_gradient(input):
    return legacy.stop_gradient(input)

def isinf(input):
    if use_pyboost():
        return pyboost.isinf_op(input)
    return legacy.is_inf(input)

def sort(input, dim, descending, stable):
    if use_pyboost():
        return pyboost.sort_ext_op(input, dim, descending, stable)
    return legacy.sort(input, dim, descending)

def prod(input, axis, keepdims, dtype):
    if use_pyboost():
        return pyboost.prod_ext_op(input, axis, keepdims, dtype)
    return legacy.reduce_prod(input, axis, keepdims)

def isclose(input, other, rtol, atol, equal_nan):
    if use_pyboost():
        return pyboost.isclose_impl(input, other, rtol, atol, equal_nan)
    return legacy.is_close(input, other, rtol, atol, equal_nan)

def argmax(input, axis, keepdims):
    if use_pyboost():
        return pyboost.argmax_ext_op(input, axis, keepdims)
    return legacy.argmax(input, axis, keepdims)

def argmin(input, axis, keepdims):
    if use_pyboost():
        return pyboost.argmin_ext_op(input, axis, keepdims)
    return legacy.argmin(input, axis, keepdims)


def bmm(input, other):
    if use_pyboost():
        return pyboost.bmm_ext_op(input, other)
    return legacy.batch_mat_mul(input, other)

def topk(input, k, dim, largest, sorted):
    if use_pyboost():
        return pyboost.topk_ext_op(input, k, dim, largest, sorted)

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

def logical_not(input):
    if use_pyboost():
        return pyboost.logical_not_op(input)
    return legacy.logical_not(input)

def rand(size, generator, dtype):
    seed, offset = generator._step(12)
    if use_pyboost():
        return pyboost.rand_ext_op(size, seed, offset, dtype)
    return legacy.uniform_real(size, 0, 0)

def inplace_uniform(input, from_, to, generator):
    seed, offset = generator._step(12)
    if use_pyboost():
        return pyboost.uniform_ext_op(input, from_, to, seed, offset)

    if input.dtype.is_floating_point:
        out = legacy.uniform_real(input.shape, 0, 0)
        value = legacy.add(legacy.mul(out, (legacy.sub(to, from_))), from_)
    else:
        value = legacy.uniform_int(input.shape,
                                    mindspore.tensor(from_, dtype=mindspore.int32),
                                    mindspore.tensor(to, dtype=mindspore.int32), 0, 0)
    input.assign_value(legacy.cast(value, input.dtype))

def bitwise_or_tensor(input, other):
    if use_pyboost():
        return pyboost.bitwise_or_tensor_op(input, other)
    return legacy.bitwise_or(input, other)

def bitwise_and_tensor(input, other):
    if use_pyboost():
        return pyboost.bitwise_and_tensor_op(input, other)
    return legacy.bitwise_and(input, other)

def bitwise_or_scalar(input, other):
    if use_pyboost():
        return pyboost.bitwise_or_scalar_op(input, other)
    return legacy.bitwise_or(input, other)


def max(input):
    if use_pyboost():
        return pyboost.max_op(input)
    return legacy.reduce_max(input, (), False)

def stack(tensors, axis=0):
    if use_pyboost():
        return pyboost.stack_ext_impl(tensors, axis)
    return legacy.stack(tensors, axis)

def narrow(input, dim, start, length):
    if use_pyboost():
        return pyboost.narrow_op(input, dim, start, length)
    begin = [0] * input.ndim
    size = [i for i in input.shape]
    begin[dim] = start
    size[dim] = length
    return legacy.slice(input, begin, size)

def std(input, dim, correction, keepdim):
    if use_pyboost():
        return pyboost.std_op(input, dim, correction, keepdim)
    return legacy.reduce_std(input, dim, keepdim)


def log(input):
    if use_pyboost():
        return pyboost.log_op(input)
    return legacy.log(input)

def gather(input_params, input_indices, axis, batch_dim):
    return legacy.gather(input_params, input_indices, axis, batch_dim)

def non_zero_ext(input):
    if use_pyboost():
        return pyboost.non_zero_ext_op(input)
    return legacy.non_zero(input)

def binary_cross_entropy_with_logits(input, target, weight, posWeight, reduction):
    if use_pyboost():
        return pyboost.binary_cross_entropy_with_logits_impl(input, target, weight, posWeight, reduction)
    return legacy.bce_with_logits_loss(input, target, weight, posWeight, reduction)

def rand_like(input, generator, dtype):
    seed, offset = generator._step(123)
    if use_pyboost():
        return pyboost.rand_like_ext_op(input, seed, offset, dtype)
    return rand(input.shape, dtype)

def floor_div(input, other):
    if use_pyboost():
        return pyboost.floor_div_op(input, other)
    return legacy.floor_div(input, other)

def inplace_fill_scalar(input, value):
    if use_pyboost():
        return pyboost.inplace_fill_scalar_op(input, value)
    input.assign_value(fill_scalar(input.shape, value, input.dtype))
    return input

def linalg_vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    if use_pyboost():
        return pyboost.linalg_vector_norm_op(x, ord, dim, keepdim, dtype)

def non_zero(input):
    if use_pyboost():
        return pyboost.non_zero_op(input)
    return legacy.non_zero(input)

def fmod_scalar(input, other):
    if use_pyboost():
        return pyboost.fmod_scalar_op(input, other)
    return legacy.floor_mod(input, other)

def inplace_zero(input):
    if use_pyboost():
        return pyboost.inplace_zero_op(input)
    input.assign_value(zeros(input.shape, input.dtype))
    return input

def mse_loss(input, target, reduction):
    if use_pyboost():
        return pyboost.mse_loss_ext_op(input, target, reduction)

def abs(input):
    if use_pyboost():
        return pyboost.abs_op(input)
    return legacy.abs(input)

def bincount(input, weights=None, minlength=0):
    if use_pyboost():
        return pyboost.bincount_ext_op(input, weights, minlength)
    return legacy.bincount(input, minlength, weights)

def bitwise_and_scalar(input, other):
    if use_pyboost():
        return pyboost.bitwise_and_scalar_op(input, other)
    return legacy.bitwise_and(input, other)

def argmax_with_value(input, axis, keep_dims):
    if use_pyboost():
        return pyboost.argmax_with_value_impl(input, axis, keep_dims)
    return legacy.argmax(input, axis, keep_dims)

def index_select(input, dim, index):
    if use_pyboost():
        return pyboost.index_select_op(input, dim, index)
    return legacy.gather(input, index, dim, 0)

def min(input):
    if use_pyboost():
        return pyboost.min_op(input)
    return legacy.reduce_min(input, (), False)

def minimum(input, other):
    if use_pyboost():
        return pyboost.minimum_op(input, other)
    return legacy.minimum(input, other)

def argmin_with_value(input, axis, keep_dims):
    if use_pyboost():
        return pyboost.argmin_with_value_impl(input, axis, keep_dims)
    return legacy.argmin(input, axis, keep_dims)

def flatten(input, start_dim, end_dim):
    if use_pyboost():
        return pyboost.flatten_ext_op(input, start_dim, end_dim)
    if start_dim < 0:
        start_dim = start_dim + input.ndim
    if end_dim < 0:
        end_dim = end_dim + input.ndim
    input_shape = list(input.shape)
    input_shape[start_dim:end_dim] = [-1]
    return legacy.reshape(input, tuple(input_shape))

def conv2d_padding(input, weight, bias=None, stride=1, padding='valid', dilation=1, groups=1):
    if use_pyboost():
        return pyboost.conv2d_padding_op(input, weight, bias, stride, padding, dilation, groups)
    return legacy.conv2d(input, weight, bias, stride, padding, dilation, groups)

def conv2d(input, weight, bias=None, stride=1, padding='valid', dilation=1, groups=1):
    if use_pyboost():
        return pyboost.conv2d_ext_op(input, weight, bias, stride, padding, dilation, groups)
    return legacy.conv2d(input, weight, bias, stride, padding, dilation, groups)

def cos(input):
    if use_pyboost():
        return pyboost.cos_op(input)
    return legacy.cos(input)

def pow_tensor_scalar(input, exponent):
    if use_pyboost():
        return pyboost.pow_tensor_scalar_op(input, exponent)
    return legacy.pow(input, exponent)

def sin(input):
    if use_pyboost():
        return pyboost.sin_op(input)
    return legacy.sin(input)

def batch_norm(input, weight, bias, running_mean=None, runnning_var=None, training=False, momentum=0.1, epsilon=1e-5):
    if use_pyboost():
        return pyboost.batch_norm_ext_op(input, weight, bias, running_mean, runnning_var, training, momentum, epsilon)
    return legacy.batch_norm(input, weight, bias, running_mean, runnning_var, training, momentum, epsilon, 'NHWC')

def silu(input):
    if use_pyboost():
        return pyboost.silu_op(input)
    return legacy.silu(input)

def rsqrt(input):
    if use_pyboost():
        return pyboost.rsqrt_op(input)
    return legacy.rsqrt(input)

def sqrt(input):
    if use_pyboost():
        return pyboost.sqrt_op(input)
    return legacy.sqrt(input)

def masked_scatter(input, mask, value):
    return legacy.masked_scatter(input, mask, value)

def neg(input):
    if use_pyboost():
        return pyboost.neg_op(input)
    return legacy.neg(input)

def log1p(input):
    if use_pyboost():
        return pyboost.log1p_op(input)
    return legacy.log1p(input)

def pow_scalar_tensor(input, scalar):
    if use_pyboost():
        return pyboost.pow_scalar_tensor_op(input, scalar)
    return legacy.pow(input, scalar)

def adaptive_avg_pool2d(input, output_size):
    if use_pyboost():
        return pyboost.adaptive_avg_pool2d_ext_op(input, output_size)
    return legacy.adaptive_avg_pool2_d(input, output_size)


def exp(input):
    if use_pyboost():
        return pyboost.exp_op(input)
    return legacy.exp(input)

def sigmoid(input):
    if use_pyboost():
        return pyboost.sigmoid_op(input)
    return legacy.sigmoid(input)

def constant_pad_nd(input, pad, value=0.0):
    if use_pyboost():
        return pyboost.constant_pad_nd_op(input, pad, value)

def rfft(input, n=None, dim=-1, norm=None):
    if use_pyboost():
        return pyboost.rfft_op(input, n, dim, norm)
    if input.shape[dim] < n:
        pad_inf = (0, n - input.shape[dim])
        pad_dims = (0, 0) * (input.ndim - (dim + 1)) + pad_inf
        input = constant_pad_nd(input, pad_dims)
    else:
        input = narrow(input, dim, 0, n)
    return legacy.fft_with_size(input, input.ndim, False, True, norm, True, ())

def avg_pool2d(input, kernel_size, stride, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if use_pyboost():
        return pyboost.avg_pool2d_op(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

def conj(input):
    if use_pyboost():
        return pyboost.conj_op(input)
    return legacy.conj(input)

def fill_tensor(size, value, dtype):
    if use_pyboost():
        return pyboost.fill_tensor_op(size, value, dtype)
    return legacy.fill_v2(size, value)

def maximum(input, other):
    if use_pyboost():
        return pyboost.maximum_op(input, other)
    return legacy.maximum(input, other)

def irfft(input, n, dim, norm):
    if use_pyboost():
        return pyboost.irfft_op(input, n, dim, norm)
    return legacy.fft_with_size(input, input.ndim, True, True, norm)

def randn(size, generator, dtype):
    if use_pyboost():
        seed, offset = generator._step(12)
        return pyboost.randn_op(size, seed, offset, dtype)
    return cast(legacy.standard_normal(size, 0, 0), dtype)

def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    if use_pyboost():
        return pyboost.avg_pool1d_op(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
    return legacy.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)

def pow(input, exponent):
    if use_pyboost():
        return pyboost.pow_op(input, exponent)
    return legacy.pow(input, exponent)

def roll(input, shifts, axis):
    if use_pyboost():
        return pyboost.roll_impl(input, shifts, axis)
    return legacy.roll(input, shifts, axis)

def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if use_pyboost():
        return pyboost.conv1d_ext_op(input, weight, bias, stride, padding, dilation, groups)
    return legacy.conv1d(input, weight, bias, pad, stride, dilation)

def conv1d_padding(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if use_pyboost():
        return pyboost.conv1d_padding_op(input, weight, bias, stride, padding, dilation, groups)
    return legacy.conv1d(input, weight, bias, pad, stride, dilation)

def square(input):
    if use_pyboost():
        return pyboost.square_op(input)
    return legacy.square(input)

def lgamma(input):
    return legacy.lgamma(input)

def reverse_v2(input, axis):
    if isinstance(axis, int):
        axis = (axis,)
    if use_pyboost():
        return pyboost.reverse_v2_impl(input, axis)
    return legacy.reverse_v2(input, axis)

def unique_consecutive(input, return_inverse, return_counts, dim):
    if use_pyboost():
        return pyboost.unique_consecutive_impl(input, return_inverse, return_counts, dim)
    return legacy.unique_consecutive(input, return_inverse, return_counts, dim)

def split_with_size(input, size, dim=0):
    if use_pyboost():
        return pyboost.split_with_size_op(input, size, dim)
    return legacy.split_with_size(input, size, dim)

def softplus(input, beta=1, threshold=20):
    if use_pyboost():
        return pyboost.softplus_ext_op(input, beta, threshold)
    return legacy.softplus(input, beta, threshold)

def remainder_tensor_scalar(input, other):
    if use_pyboost():
        return pyboost.remainder_tensor_scalar_op(input, other)
    out = input - floor_div(input, other) * other
    return out

def baddbmm(input, batch1, batch2, alpha=1, beta=1):
    if use_pyboost():
        return pyboost.baddbmm_op(input, batch1, batch2, alpha, beta)
    return legacy.baddbmm(input, batch1, batch2, alpha, beta)

def floor(input):
    if use_pyboost():
        return pyboost.floor_op(input)
    return legacy.floor(input)

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    if use_pyboost():
        return pyboost.conv_transpose2d_op(input, weight, bias, stride, padding, output_padding, groups, dilation)
    return legacy.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)

def relu(input):
    if use_pyboost():
        return pyboost.relu_op(input)
    return legacy.re_lu(input)

def max_pool2d(input, kernel_size, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    # out, indices = legacy.max_pool_with_argmax_v2(input, kernel_size, stride, padding, dilation, ceil_mode)

    out, indices = legacy.max_pool_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
    if return_indices:
        return out, indices
    return out

def upsample_bilinear2d(input, size=None, scale_factor=None, align_corners=False):
    if use_pyboost():
        return pyboost.upsample_bilinear2d_op(input, size, scale_factor, align_corners)
    return legacy.resize_bilinear_v2(input, size, scale_factor, align_corners)

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    if use_pyboost():
        return pyboost.group_norm_op(input, num_groups, weight, bias, eps)
    return legacy.group_norm(input, num_groups, eps, affine)

def nllloss_2d(input, target, weight, reduction='mean', ignore_index=-100):
    if use_pyboost():
        return pyboost.nllloss_2d_op(input, target, weight, reduction, ignore_index)
    return legacy.nll_loss(input, target, weight, ignore_index, reduction)

def inplace_relu(input):
    if use_pyboost():
        return pyboost.inplace_relu_op(input)
    return legacy.assign(input, legacy.re_lu(input))

def expm1(input):
    if use_pyboost():
        return pyboost.expm1_op(input)
    return legacy.expm1(input)

def upsample_bicubic2d(input, size=None, scale_factor=None, align_corners=False):
    if use_pyboost():
        return pyboost.upsample_bicubic2d_op(input, size, scale_factor, align_corners)
    return legacy.resize_bicubic(input, size, scale_factor, align_corners)

def acos(input):
    if use_pyboost():
        return pyboost.acos_op(input)
    return legacy.acos(input)

def cdist(x1, x2, p):
    return legacy.cdist(x1, x2, float(p))

def unstack_view(input, dim):
    if use_pyboost():
        return pyboost.unstack_ext_view_op(input, dim)
    return legacy.unstack(input, dim, input.shape[dim])

def l1_loss(input, target, reduction='mean'):
    if use_pyboost():
        return pyboost.l1_loss_ext_op(input, target, reduction)
    return legacy.l1(input, target, reduction)

def diag(input, diagonal):
    if use_pyboost():
        return pyboost.diag_ext_op(input, diagonal)
    return legacy.diag(input, diagonal)

def logsigmoid(input):
    if use_pyboost():
        return pyboost.logsigmoid_op(input)
    return legacy.logsigmoid(input)

def one_hot(tensor, num_classes):
    if use_pyboost():
        on_value = mindspore.Tensor(1, dtype=tensor.dtype)
        off_value = mindspore.Tensor(0, dtype=tensor.dtype)
        return pyboost.one_hot_ext_impl(tensor, num_classes, on_value, off_value, -1)
    return legacy.one_hot(tensor, num_classes, on_value, off_value, -1)

def var(input, dim=None, correction=1, keepdim=False):
    if use_pyboost():
        return pyboost.var_op(input, dim, correction, keepdim)
    return legacy.var(input, dim, correction, keepdim)

def linspace(start, end, steps, dtype=None):
    if use_pyboost():
        return pyboost.lin_space_ext_op(start, end, steps, dtype)
    return legacy.lin_space(start, end, steps)

def masked_select(input, mask):
    if use_pyboost():
        return pyboost.masked_select_op(input, mask)
    return legacy.masked_select(input, mask)

def glu(input, dim=-1):
    if use_pyboost():
        return pyboost.glu_impl(input, dim)
    return legacy.glu(input, dim)

def scatter_value(input, dim, index, src, reduce='none'):
    if use_pyboost():
        return pyboost.scatter_value_op(input, dim, index, src, reduce)
    return legacy.scatter(input, dim, index, src, reduce)

def unique_dim(input, sorted, return_inverse, dim):
    if use_pyboost():
        return pyboost.unique_dim_op(input, sorted, return_inverse, dim)
    return legacy.unique_dim(input, sorted, return_inverse, dim)

def inplace_add(input, other, alpha):
    if use_pyboost():
        return pyboost.inplace_add_ext_op(input, other, alpha)
    return legacy.inplace_add(input, other)

def logsumexp(input, dim, keepdim):
    if use_pyboost():
        return pyboost.logsumexp_op(input, dim, keepdim)
    return legacy.logsumexp(input, dim, keepdim)

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

def inplace_exponential(self, lambd, generator):
    seed, offset = generator._step(12)
    if use_pyboost():
        return pyboost.inplace_exponential_op(self, lambd, seed, offset)
    return legacy.expo(self, lambd, generator)

def im2col(input, kernel_size, dilation=1, padding=0, stride=1):
    if use_pyboost() and not ON_A1:
        return pyboost.im2col_ext_op(input, kernel_size, dilation, padding, stride)
    out = legacy.im2_col(input, kernel_size, stride, dilation, padding)
    out_shape = out.shape[:1] + (-1,) + out.shape[-1:]
    out = reshape(out, out_shape)
    return out

def upsample_nearest2d(input, output_size, scale_factors):
    if use_pyboost():
        return pyboost.upsample_nearest2d_op(input, output_size, scale_factors)
    return legacy.upsample_nearest2d(input, scale_factor, align_corners)

def addmm(input, mat1, mat2, alpha=1.0, beta=1.0):
    if use_pyboost():
        return pyboost.addmm_op(input, mat1, mat2, alpha, beta)
    return legacy.addmm(input, mat1, mat2, alpha, beta)

def meshgrid(input, lambd):
    if use_pyboost():
        return pyboost.meshgrid_impl(input, lambd)
    return legacy.meshgrid(input, lambd)

def adaptive_avg_pool1d(input, output_size):
    if use_pyboost():
        return pyboost.adaptive_avg_pool1d_op(input, output_size)
    return legacy.adaptive_avg_pool1d(input, output_size)

def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if use_pyboost():
        return pyboost.conv3d_ext_op(input, weight, bias, stride, padding, dilation, groups)
    return legacy.conv3d(input, weight, bias, stride, padding, dilation, groups)

def outer(input, other):
    if use_pyboost():
        return pyboost.outer_op(input, other)
    return legacy.outer(input, other)

def addcmul(input, tensor1, tensor2, value=1.0):
    if use_pyboost():
        return pyboost.addcmul_op(input, tensor1, tensor2, value)
    return legacy.addcmul(input, tensor1, tensor2, value)

def prelu(input, weight):
    if use_pyboost():
        return pyboost.prelu_op(input, weight)
    return legacy.p_re_lu(input, weight)

def reciprocal(input):
    if use_pyboost():
        return pyboost.reciprocal_op(input)
    return legacy.reciprocal(input)

def index_add_ext(input, dim, index, source, alpha):
    if use_pyboost():
        return pyboost.index_add_ext_op(input, dim, index, source, alpha)
    return legacy.index_add(input, dim, index, source, alpha)

def polar(abs, angle):
    if use_pyboost():
        return pyboost.polar_op(abs, angle)
    return legacy.polar(abs, angle)

def upsample_linear1d(input, output_size, scale_factor, align_corners=False):
    if use_pyboost():
        return pyboost.upsample_linear1d_op(input, output_size, scale_factor, align_corners)
    return legacy.upsample_linear1d(input, output_size, scale_factor, align_corners)

def grid_sampler_2d(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    if use_pyboost():
        return pyboost.grid_sampler_2d_impl(input, grid, mode, padding_mode, align_corners)
    return legacy.grid_sampler_2d(input, grid, mode, padding_mode, align_corners)

def pixel_shuffle(input, upscale_factor):
    if use_pyboost():
        return pyboost.pixel_shuffle_op(input, upscale_factor)
    return legacy.pixel_shuffle(input, upscale_factor)

def view_as_complex(input):
    real_part, imag_part = chunk(input, 2, -1)
    return legacy.complex(squeeze(real_part, -1), squeeze(imag_part, -1))

def rms_norm(input, weight, eps=1e-5):
    if use_pyboost():
        return pyboost.rms_norm_impl(input, weight, eps)[0]
    input_dtype = input.dtype
    input = cast(input, mindspore.float32)
    variance = mean(pow(input, 2), -1, True, None)
    input = mul(input, rsqrt(add(variance, eps, 1)))
    return mul(weight, cast(input, input_dtype))

def normal_float_float(mean, std, size, dtype, generator):
    seed, offset = generator._step(12)
    if use_pyboost():
        return pyboost.normal_float_float_op(mean, std, size, seed, offset)

def real(input):
    if use_pyboost():
        return pyboost.real_op(input)
    return legacy.real(input)

def imag(input):
    return legacy.imag(input)

def leaky_relu(input, negative_slope):
    if use_pyboost():
        return pyboost.leaky_relu_ext_op(input, negative_slope)
    return legacy.leaky_relu(input, negative_slope)

def ceil(input):
    if use_pyboost():
        return pyboost.ceil_op(input)
    return legacy.ceil(input)

def erf(input):
    if use_pyboost():
        return pyboost.erf_op(input)
    return legacy.erf(input)

def cross(input, other, dim):
    if use_pyboost():
        return pyboost.cross_impl(input, other, dim)
    return legacy.cross(input, other, dim)

def elu(input, alpha):
    if use_pyboost():
        return pyboost.elu_ext_impl(input, alpha)
    return legacy.elu(input, alpha)

def reduce_max(input, axis, keepdims):
    if use_pyboost():
        return pyboost.reduce_max_impl(input, axis, keepdims)
    return legacy.reduce_max(input, axis, keepdims)

def dynamic_rnn(x, w, b, seq_length, init_h, init_c):
    return legacy.dynamic_rnn(x, w, b, seq_length, init_h, init_c,
                              'LSTM', 'UNIDIRECTIONAL', 1, False, 1.0, -1.0, 0, True, 'tanh', 0.0, True)

def nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return legacy.nan_to_num(input, nan, posinf, neginf)

def round(input, decimals):
    if use_pyboost():
        return pyboost.round_op(input, decimals)
    return legacy.round(input, decimals)

def fftn(input, s=None, dim=None, norm=None):
    if use_pyboost():
        return pyboost.fftn_op(input, s, dim, norm)

def eye(n, m=None, dtype=None):
    if use_pyboost():
        return pyboost.eye_op(n, m, dtype)
    return legacy.eye(n, m, dtype)

def erfinv(input):
    if use_pyboost():
        return pyboost.erfinv_op(input)
    return legacy.erfinv(input)

def logit(input, eps=1e-5):
    return legacy.logit(input, eps)

def bitwise_xor_tensor(input, other):
    if use_pyboost():
        return pyboost.bitwise_xor_tensor_op(input, other)
    return legacy.bitwise_xor(input, other)

def unique2(input, sorted, return_inverse, return_counts):
    if use_pyboost():
        return pyboost.unique2_op(input, sorted, return_inverse, return_counts)
    return legacy.unique(input, sorted, return_inverse, return_counts)

def sign(input):
    if use_pyboost():
        return pyboost.sign_op(input)
    return legacy.sign(input)

def log2(input):
    if use_pyboost():
        return pyboost.log2_op(input)
    return legacy.log2(input)

def bucketize(input, boundaries, right=False):
    epsilon_ = 0. if right else 1.e-6
    boundaries = [boundary + epsilon_ for boundary in boundaries]
    return legacy.bucketize(input, boundaries)

def inplace_fill_diagonal(input, fill_value, wrap):
    if use_pyboost():
        return pyboost.inplace_fill_diagonal_op(input, fill_value, wrap)
    return legacy.fill_diagonal(input, fill_value, wrap)

def clamp_tensor(input, min, max):
    if use_pyboost():
        return pyboost.clamp_tensor_op(input, min, max)

def hswish(input):
    if use_pyboost():
        return pyboost.hswish_op(input)
    return legacy.h_swish(input)

def logical_and(input, other):
    if use_pyboost():
        return pyboost.logical_and_op(input, other)
    return legacy.logical_and(input, other)

def as_strided(input, size, stride, storage_offset):
    if use_pyboost():
        return pyboost.as_strided_op(input, size, stride, storage_offset)
    return legacy.as_strided(input, size, stride, storage_offset)

def relu6(input):
    if use_pyboost():
        return pyboost.relu6_op(input)
    return legacy.re_lu6(input)

def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    if use_pyboost():
        return pyboost.col2im_ext_op(input, output_size, kernel_size, dilation, padding, stride)
    return legacy.col2im(input, output_size, kernel_size, dilation, padding, stride)

def flash_attention_score(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix, actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value, pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode):
    if use_pyboost():
        return pyboost.flash_attention_score_impl(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix, actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value, pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode)
    return legacy.flash_attention_score(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix, actual_seq_qlen, actual_seq_kvlen, head_num, keep_prob, scale_value, pre_tokens, next_tokens, inner_precise, input_layout, sparse_mode)

def prompt_flash_attention(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise):
    return pyboost.prompt_flash_attention_impl(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value, pre_tokens, next_tokens, input_layout, num_key_value_heads, sparse_mode, inner_precise)

def incre_flash_attention(query, key, value, attn_mask, actual_seq_lengths, pse_shift, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, kv_padding_size, num_heads, input_layout, scale_value, num_key_value_heads, block_size, inner_precise):
    return pyboost.incre_flash_attention_impl(query, key, value, attn_mask, actual_seq_lengths, pse_shift, dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset, block_table, kv_padding_size, num_heads, input_layout, scale_value, num_key_value_heads, block_size, inner_precise)

def randperm(n, generator, dtype):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    if use_pyboost():
        return pyboost.randperm_ext_op(n, seed, offset, dtype)
    return legacy.randperm(n, seed)

def logical_or(input_x, input_y):
    if use_pyboost():
        return pyboost.logical_or_op(input_x, input_y)
    return legacy.logical_or(input_x, input_y)

def dropout2d(input_x, p):
    return legacy.dropout2_d(input_x, p)

def linalg_qr(input_x, mode):
    if use_pyboost():
        return pyboost.linalg_qr_op(input_x, mode)
    full_matrices = 'mode' == 'complete'
    return legacy.qr(input_x, full_matrices)

def bernoulli(input, generator):
    seed, offset = generator._step(12)
    if use_pyboost():
        return pyboost.bernoulli_ext_op(input, seed, offset)
    return legacy.bernoulli(input, seed, offset)

def multinomial(input, num_samples, replacement, generator):
    seed, offset = generator._step(12)  # pylint: disable=protected-access
    return pyboost.multinomial_ext_op(input, num_samples, replacement, seed, offset)

def right_shift(input, other):
    if isinstance(other, int):
        other = mindspore.Tensor(other, dtype=input.dtype)
    if use_pyboost():
        return pyboost.right_shift_op(input, other)
    return legacy.right_shift(input, other)

def histc(input, bins=100, min=0, max=0):
    if use_pyboost():
        return pyboost.histc_ext_op(input, bins, float(min), float(max))
    return legacy.histogram(input, bins, float(min), float(max))

def dist_comm_barrier(group):
    return pyboost.dist_comm_barrier_op(group)

def new_empty(input, size, dtype):
    return pyboost.new_empty_op(input, size, dtype, 'Ascend')

def new_ones(input, size, dtype):
    return pyboost.new_ones_op(input, size, dtype)

def kl_div(input, target, reduction, log_target):
    return pyboost.kl_div_op(input, target, reduction, log_target)

def repeat_interleave_int(input, repeats, dim, output_size):
    return pyboost.repeat_interleave_int_op(input, repeats, dim, output_size)

def repeat_interleave_tensor(input, repeats, dim, output_size):
    return pyboost.repeat_interleave_tensor_op(input, repeats, dim, output_size)

def triu_indices(row, col, offset, dtype):
    return legacy.triu_indices(row, col, offset, dtype)
