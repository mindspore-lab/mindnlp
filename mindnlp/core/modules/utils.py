"""module utils"""
from typing import List, Sequence, TypeVar

import mindspore
from mindspore._c_expression import typing # pylint: disable=no-name-in-module, import-error
from mindnlp.core import ops, nn

T = TypeVar("T")

def _rindex(sequence: Sequence[T], obj: T) -> int:
    """
    Return zero-based index in the sequence of the last item whose value is equal to obj.  Raises a
    ValueError if there is no such item.

    # Parameters

    sequence : `Sequence[T]`
    obj : `T`

    # Returns

    `int`
        zero-based index associated to the position of the last item equal to obj
    """
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == obj:
            return i

    raise ValueError(f"Unable to find {obj} in sequence {sequence}.")

def info_value_of_dtype(dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == mindspore.bool_:
        raise TypeError("Does not support torch.bool")
    elif isinstance(dtype, typing.Float):
        return ops.finfo(dtype)
    else:
        return ops.iinfo(dtype)


def min_value_of_dtype(dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return float(info_value_of_dtype(dtype).min)


def max_value_of_dtype(dtype):
    """
    Returns the maximum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return float(info_value_of_dtype(dtype).max)


def tiny_value_of_dtype(dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype in (mindspore.float32, mindspore.float64):
        return 1e-13
    elif dtype == mindspore.float16:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

def masked_softmax(
    vector: mindspore.Tensor,
    mask: mindspore.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
) -> mindspore.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        while mask.ndim < vector.ndim:
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = nn.functional.softmax(masked_vector, dim=dim)
    return result

def _get_combination(combination: str, tensors: List[mindspore.Tensor]) -> mindspore.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            return first_tensor * second_tensor
        elif operation == "/":
            return first_tensor / second_tensor
        elif operation == "+":
            return first_tensor + second_tensor
        elif operation == "-":
            return first_tensor - second_tensor
        else:
            raise ValueError("Invalid operation: " + operation)


def _get_combination_and_multiply(
    combination: str, tensors: List[mindspore.Tensor], weight: nn.Parameter
) -> mindspore.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return ops.matmul(tensors[index], weight)
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == "*":
            if first_tensor.ndim > 4 or second_tensor.ndim > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.ndim, second_tensor.ndim) - 1
            if first_tensor.ndim == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.ndim == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = ops.matmul(intermediate, second_tensor.swapaxes(-1, -2))
            if result.ndim == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == "/":
            if first_tensor.ndim > 4 or second_tensor.ndim > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.ndim, second_tensor.ndim) - 1
            if first_tensor.ndim == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.ndim == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = ops.matmul(intermediate, second_tensor.pow(-1).swapaxes(-1, -2))
            if result.ndim == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == "+":
            return ops.matmul(first_tensor, weight) + ops.matmul(second_tensor, weight)
        elif operation == "-":
            return ops.matmul(first_tensor, weight) - ops.matmul(second_tensor, weight)
        else:
            raise ValueError("Invalid operation: " + operation)

def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with [`combine_tensors`](./util.md#combine_tensors).
    This function computes the resultant dimension when calling `combine_tensors(combination, tensors)`,
    when the tensor dimension is known.  This is necessary for knowing the sizes of weight matrices
    when building models that use `combine_tensors`.

    # Parameters

    combination : `str`
        A comma-separated list of combination pieces, like `"1,2,1*2"`, specified identically to
        `combination` in `combine_tensors`.
    tensor_dims : `List[int]`
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to `combine_tensors`.
    """
    if len(tensor_dims) > 9:
        raise ValueError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    return sum(_get_combination_dim(piece, tensor_dims) for piece in combination.split(","))


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise ValueError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ValueError('Tensor dims must match for operation "{}"'.format(operation))
        return first_tensor_dim

def combine_tensors_and_multiply(
    combination: str, tensors: List[mindspore.Tensor], weights: nn.Parameter
) -> mindspore.Tensor:
    """
    Like [`combine_tensors`](./util.md#combine_tensors), but does a weighted (linear)
    multiplication while combining. This is a separate function from `combine_tensors`
    because we try to avoid instantiating large intermediate tensors during the combination,
    which is possible because we know that we're going to be multiplying by a weight vector in the end.

    # Parameters

    combination : `str`
        Same as in `combine_tensors`
    tensors : `List[mindspore.Tensor]`
        A list of tensors to combine, where the integers in the `combination` are (1-indexed)
        positions in this list of tensors.  These tensors are all expected to have either three or
        four dimensions, with the final dimension being an embedding.  If there are four
        dimensions, one of them must have length 1.
    weights : `torch.nn.Parameter`
        A vector of weights to use for the combinations.  This should have shape (combined_dim,),
        as calculated by `get_combined_dim`.
    """
    if len(tensors) > 9:
        raise ValueError("Double-digit tensor lists not currently supported")
    combination = combination.replace("x", "1").replace("y", "2")
    pieces = combination.split(",")
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far : (dims_so_far + combination_dim)]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result
