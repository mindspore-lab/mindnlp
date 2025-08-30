"""other op"""
import numpy as np
import mindspore
from mindspore.ops import gather
from mindnlp import core
from mindnlp.core.executor import execute
from ..configs import ON_A1

# atleast_2d


# atleast_3d


# bincount
def bincount(input, weights=None, minlength=0):
    return execute('bincount_ext', input, weights, minlength)

# block_diag


# broadcast_tensors
def broadcast_tensors(*tensors):
    target_shape = broadcast_shapes(*[t.shape for t in tensors])
    broadcasted_tensors = [t.broadcast_to(target_shape) for t in tensors]
    return broadcasted_tensors


# broadcast_to
def broadcast_to(input, shape):
    return execute('broadcast_to', input, shape)


# broadcast_shapes
def broadcast_shapes(*shapes):
    reversed_shapes = [list(reversed(shape)) for shape in shapes]

    max_dim = max(len(shape) for shape in reversed_shapes)

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

# bucketize

# cartesian_prod


# cdist
def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    return execute('cdist', x1, x2, p)

# clone
def clone(input, *, memory_format=core.preserve_format):
    if input.device.type == 'npu':
        return execute('clone', input)
    return execute('identity', input)


# combinations


# corrcoef


# cov


# cross

# cummax

# cummin

# cumprod

# cumsum
def cumsum(input, dim, dtype=None):
    if input.dtype in [core.int64, core.bool]:
        return execute('cumsum_ext', input.int(), dim, None).long()
    if dtype is not None and dtype == core.int64:
        return execute('cumsum_ext', input, dim, None).long()
    return execute('cumsum_ext', input, dim, dtype)

# diag
def diag(input, diagonal=0, *, out=None):
    return execute('diag_ext', input, diagonal)

# diag_embed


# diagflat


# diagonal
def diagonal(input, offset=0, dim1=0, dim2=1):
    return execute('diagonal', input, offset, dim1, dim2)

# diff
def _diff_is_scalar_or_scalar_tensor(value):
    """judge the value"""
    if isinstance(value, int):
        return True

    if isinstance(value, core.Tensor) and value.shape == ():
        return True

    return False

def _diff_helper(input, n, dim):
    """calculate the forward difference"""
    out_len = input.shape[dim] - 1
    is_bool = (input.dtype == core.bool)
    result = input

    for _ in range(n):  # pylint: disable=unused-variable
        if is_bool:
            result = core.logical_xor(core.narrow(result, dim, 1, out_len), core.narrow(result, dim, 0, out_len))
        else:
            result = core.sub(core.narrow(result, dim, 1, out_len), core.narrow(result, dim, 0, out_len))

        if out_len == 0:
            break
        out_len -= 1

    return result


def _diff_prepend_append_on_dim(input, prepend, append, dim):
    """append tensor on dim"""
    if prepend is not None and append is None:
        return core.cat((prepend, input), dim)

    if prepend is None and append is not None:
        return core.cat((input, append), dim)

    return core.cat((prepend, input, append), dim)


def diff(input, n=1, dim=-1, prepend=None, append=None):
    if (prepend is None and append is None) or n == 0:
        return _diff_helper(input, n, dim)

    input = _diff_prepend_append_on_dim(input, prepend, append, dim)
    return _diff_helper(input, n, dim)

def _einsum_convert_sublist_to_label(num, ell_num=False):
    """Convert sublist to label."""
    if num == Ellipsis or ell_num and num == 52:
        return '...'
    if 0 <= num < 26:
        return chr(num + ord('A'))
    if 26 <= num < 52:
        return chr(num + ord('a') - 26)
    raise ValueError(
        f'For einsum, the number in sublist must be in range [0, 52), but got {num}')


def _einsum_convert_label_to_index(label):
    """Convert label to index."""
    label_num = ord(label)
    if ord('A') <= label_num <= ord('Z'):
        return label_num - ord('A')
    if ord('a') <= label_num <= ord('z'):
        return label_num - ord('a') + 26
    if label_num == ord('.'):
        return 52
    raise ValueError(
        f'For einsum, the label in equation must be in [a-zA-Z] or ., but got {label}')


def _einsum_convert_sublist(equation, *operands):
    """Convert the sublist to an equation operand if the received input is a sublist format."""
    if isinstance(equation, core.Tensor):
        equation_tmp = ''
        for i, lst in enumerate(operands):
            if i % 2 == 0:
                for _, num in enumerate(lst):
                    equation_tmp += _einsum_convert_sublist_to_label(num)
                if i in (len(operands) - 1, len(operands) - 2):
                    continue
                equation_tmp += ','
        if len(operands) % 2 == 0:
            equation_tmp += '->'
            for _, num in enumerate(operands[-1]):
                equation_tmp += _einsum_convert_sublist_to_label(num)
            operands_tmp = list([equation]) + list(operands[1:-1:2])
        else:
            operands_tmp = list([equation]) + list(operands[1::2])
        equation = equation_tmp
        operands = tuple(operands_tmp)
    if len(operands) == 0:  # pylint: disable=len-as-condition
        raise ValueError(
            "For einsum, the 'operands' must have at least one operand.")
    return equation, operands


def _einsum_check_inputargs(equation, operands):
    """Check equation and operands."""
    if not isinstance(equation, str):
        raise TypeError(
            f"For einsum, 'equation' must be a str, but got {type(equation)}.")
    for operand in operands:
        if not isinstance(operand, core.Tensor):
            raise TypeError(
                f"For einsum, members of 'operands' must be Tensor, but got {type(operand)}.")


def _einsum_parse_equation(equation):
    """Parse equation."""
    l_equation = ''
    r_equation = ''
    equation = equation.replace(' ', '')

    if '->' in equation:
        l_equation, r_equation = equation.split('->', 1)
        if l_equation == '':
            raise ValueError(
                'For einsum, equation must contain characters to the left fo the arrow.')
    else:
        l_equation = equation

    if ',' in l_equation:
        l_equationlst = l_equation.split(",")
    else:
        l_equationlst = [l_equation]

    l_equationlst = []

    for subequation in l_equation.split(','):
        if '.' in subequation and ('...' not in subequation or subequation.count('.') != 3):
            raise ValueError(f"For einsum, an ellipsis in the equation must include three continuous \'.\', "
                             f"and can only be found once.")
        subequation_lst = [_einsum_convert_label_to_index(label) for label in subequation.replace('...', '.')]
        l_equationlst.append(subequation_lst)

    if "." in r_equation and ('...' not in r_equation or r_equation.count('.') != 3):
        raise ValueError(f"For einsum, an ellipsis in the equation must include three continuous \'.\', "
                         f"and can only be found once.")
    r_equationlst = [_einsum_convert_label_to_index(label) for label in r_equation.replace('...', '.')]

    return l_equationlst, r_equationlst, ('->' in equation)


def _einsum_parse_labels(l_equationlst, operands):
    """Parse left script of equation."""
    align_rank = 0
    max_labels = 53
    ellipsis_dimnum = 0
    labels_count = [0] * max_labels

    if len(operands) != len(l_equationlst):
        raise ValueError(f"For einsum, 'operands' is not equal to specified in the 'equation', "
                         f"but got {len(operands)} and {len(l_equationlst)}.")

    for idx, sub_equ in enumerate(l_equationlst):
        start_dim = 0
        label_num = 0
        operand_shape = list(operands[idx].shape)
        for label in sub_equ:
            dim_num = 1
            label_num += 1
            end_dim = start_dim + 1

            # Label is ellipsis
            if label == 52:
                end_dim = len(operand_shape) - len(sub_equ) + label_num
                dim_num = end_dim - start_dim
                if ellipsis_dimnum != 0 and ellipsis_dimnum != dim_num:
                    raise ValueError(f"For einsum, an ellipsis in 'equation' can only represent the same numbers of "
                                     f"dimensions in 'operands'.")
                ellipsis_dimnum = dim_num
            if labels_count[label] == 0:
                align_rank += dim_num
            labels_count[label] += 1
            start_dim += dim_num
        if label_num != len(sub_equ) or start_dim != len(operand_shape):
            raise ValueError(f"For einsum, the numbers of labels specified in the 'equation' does not match "
                             f"'operands[{idx}]'.")
    return ellipsis_dimnum, labels_count, align_rank


def _einsum_infer_output(r_equationlst, arrow_exist, ellipsis_dimnum, labels_count):
    """Parse right script of equation and infer output shape."""
    idx = 0
    idle_idx = -1
    output_rank = 0
    labels_perm_idx = [idle_idx] * 53

    if arrow_exist:
        for label in r_equationlst:
            if labels_count[label] != 0:
                if labels_perm_idx[label] != idle_idx:
                    raise ValueError(f"For einsum, '{_einsum_convert_sublist_to_label(label, True)}' or {label} in "
                                     f"sublist format has appears more than once in output subscript.")
                dimnum = 1
                if label == 52:
                    dimnum = ellipsis_dimnum
                labels_perm_idx[label] = idx
                output_rank += dimnum
                idx += dimnum
            else:
                raise ValueError(f"For einsum, the label to the right of arrow in the 'equation' must appear on "
                                 f"left, but '{_einsum_convert_sublist_to_label(label, True)}' does not.")
    else:
        if labels_count[52] != 0:
            output_rank += ellipsis_dimnum
            labels_perm_idx[52] = idx
            idx += ellipsis_dimnum
        for label, count in enumerate(labels_count):
            if count == 1:
                output_rank += 1
                labels_perm_idx[label] = idx
                idx += 1

    for label, count in enumerate(labels_count):
        if count != 0 and labels_perm_idx[label] == idle_idx:
            labels_perm_idx[label] = idx
            idx += 1

    return output_rank, labels_perm_idx


def _einsum_adjust_operands(operands, l_equationlst, ellipsis_dimnum, labels_perm_idx, align_rank):
    """Align operands to output as possible."""
    # Unsqueeze miss dimensions to make all operands has same rank, compute diagonal if operand has same label.
    # Then use _labels_perm_idx to transpose all operands to align dimensions with output.
    adjust_operands = []
    for idx, operand in enumerate(operands):
        idle_dim = -1
        align_axis = [idle_dim] * align_rank
        label_dims = [idle_dim] * 53
        dim = 0

        for label in l_equationlst[idx]:
            if label_dims[label] != idle_dim:
                operand = core.diagonal(operand, 0, label_dims[label], dim)
                diag_perm = []
                diag_dim = 0
                for i in range(len(operand.shape)):
                    if i == label_dims[label]:
                        diag_perm.append(len(operand.shape) - 1)
                    else:
                        diag_perm.append(diag_dim)
                        diag_dim += 1
                operand = core.permute(operand, tuple(diag_perm))
            else:
                label_dims[label] = dim
                if label == 52:
                    for ell_idx in range(ellipsis_dimnum):
                        align_axis[labels_perm_idx[label] + ell_idx] = dim
                        dim += 1
                else:
                    align_axis[labels_perm_idx[label]] = dim
                    dim += 1
        if len(operand.shape) < align_rank:
            for i, axis in enumerate(align_axis):
                if axis == idle_dim:
                    align_axis[i] = dim
                    dim += 1
            missing_dims = [1] * (align_rank - len(operand.shape))
            operand_shape = list(operand.shape) + missing_dims
            operand = core.reshape(operand, operand_shape)
        operand = core.permute(operand, tuple(align_axis))
        adjust_operands.append(operand)
    return adjust_operands


def _einsum_find_dimlastop(align_rank, operands, adjust_operands):
    """Find dim last operand."""
    dim_last_op = [0] * align_rank
    has_zero_dim = False
    for dim in range(align_rank):
        broadcast_dim = adjust_operands[0].shape[dim]
        for idx in range(1, len(adjust_operands)):
            other_dim = adjust_operands[idx].shape[dim]
            if broadcast_dim != other_dim and broadcast_dim != 1 and other_dim != 1:
                err_msg = "For einsum, operands do not broadcast after align to output [shapes :origin -> adjust]:"
                for i in range(len(operands)):
                    err_msg += f" {operands[i].shape} -> {adjust_operands[i].shape}"
                raise ValueError(err_msg)
            if other_dim != 1:
                dim_last_op[dim] = idx
                broadcast_dim = other_dim
        has_zero_dim = has_zero_dim or broadcast_dim == 0
    return dim_last_op, has_zero_dim


def _einsum_multiplication(sum_dims, l_tensor, r_tensor):
    """Compute bmm for einsum."""
    batch_dims = []
    lonly_dims = []
    ronly_dims = []
    batch_size = 1
    lonly_size = 1
    ronly_size = 1
    sum_size = 1

    l_shape = l_tensor.shape
    r_shape = r_tensor.shape

    # Compute sum if dim is in sum_dims and get shapes for bmm
    for i in range(len(l_shape)):
        sum_l = l_shape[i] > 1
        sum_r = r_shape[i] > 1
        if i in sum_dims:
            if sum_l and sum_r:
                sum_size *= l_shape[i]
            elif sum_l:
                l_tensor = core.sum(l_tensor, i, True)
            elif sum_r:
                r_tensor = core.sum(r_tensor, i, True)
        elif sum_l and sum_r:
            batch_dims.append(i)
            batch_size *= l_shape[i]
        elif sum_l:
            lonly_dims.append(i)
            lonly_size *= l_shape[i]
        else:
            ronly_dims.append(i)
            ronly_size *= r_shape[i]

    # Compute the einsum bmm operators pipeline.
    # The whole operators pipeline is transpose(in) -> reshape(in) -> bmm(in) -> reshape(out) -> transpose(out).
    l_reshape_shape = (batch_size, lonly_size, sum_size)
    r_reshape_shape = (batch_size, sum_size, ronly_size)

    out_reshape_shape = [l_shape[dim] for dim in batch_dims]
    out_reshape_shape += [l_shape[dim] for dim in lonly_dims]
    out_reshape_shape += [1 for _ in sum_dims]
    out_reshape_shape += [r_shape[dim] for dim in ronly_dims]

    l_perm_axis = batch_dims + lonly_dims + sum_dims + ronly_dims
    r_perm_axis = batch_dims + sum_dims + ronly_dims + lonly_dims
    out_perm_axis = [-1] * len(out_reshape_shape)

    out_dim = 0
    for idx in range(len(l_perm_axis)):
        out_perm_axis[l_perm_axis[idx]] = out_dim
        out_dim += 1

    l_tensor = core.permute(l_tensor, tuple(l_perm_axis))
    l_tensor = core.reshape(l_tensor, l_reshape_shape)

    r_tensor = core.permute(r_tensor, tuple(r_perm_axis))
    r_tensor = core.reshape(r_tensor, r_reshape_shape)

    output = core.bmm(l_tensor, r_tensor)
    output = core.reshape(output, out_reshape_shape)
    output = core.permute(output, tuple(out_perm_axis))

    output_origin_shape = output.shape
    output_squeeze_shape = []
    for dim in range(len(output_origin_shape)):
        if dim not in sum_dims:
            output_squeeze_shape.append(output_origin_shape[dim])

    return core.reshape(output, output_squeeze_shape)


def _einsum(equation, operands):
    '''Einsum main process'''
    _l_equationlst, _r_equationlst, _arrow_exist = _einsum_parse_equation(
        equation)
    _ellipsis_dimnum, _labels_count, _align_rank = _einsum_parse_labels(
        _l_equationlst, operands)
    _output_rank, _labels_perm_idx = _einsum_infer_output(
        _r_equationlst, _arrow_exist, _ellipsis_dimnum, _labels_count)
    _adjust_operands = _einsum_adjust_operands(operands, _l_equationlst, _ellipsis_dimnum, _labels_perm_idx,
                                               _align_rank)
    _dim_last_op, _has_zero_dim = _einsum_find_dimlastop(
        _align_rank, operands, _adjust_operands)
    _result = _adjust_operands[0]

    # Fast path if operands has zero dim.
    if _has_zero_dim:
        output_shape = []
        for dim in range(_output_rank):
            output_shape.append(_adjust_operands[_dim_last_op[dim]].shape[dim])
        return core.zeros(output_shape, dtype=_result.dtype)

    # Sum or squeeze dimensions that is 1 for all rest operands.
    _reduce_dim = _output_rank
    for dim in range(_output_rank, _align_rank):
        if _dim_last_op[dim] == 0:
            if _result.shape[_reduce_dim] == 1:
                _result = core.squeeze(_result, _reduce_dim)
            else:
                _result = core.sum(_result, _reduce_dim)
        else:
            _reduce_dim += 1

    # Compute multiplication if operands are more than two.
    for i in range(1, len(_adjust_operands)):
        operand = _adjust_operands[i]
        dim = _output_rank
        sum_dims = []
        for j in range(_output_rank, _align_rank):
            if _dim_last_op[j] < i:
                operand = core.squeeze(operand, dim)
            elif _dim_last_op[j] == i:
                if _result.shape[dim] == 1:
                    operand = core.sum(operand, dim)
                    _result = core.squeeze(_result, dim)
                else:
                    sum_dims.append(dim)
                    dim += 1
            else:
                dim += 1

        if sum_dims == []:
            _result = core.mul(_result, operand)
        elif len(sum_dims) == len(_result.shape):
            _result = core.dot(core.flatten(_result), core.flatten(operand))
        else:
            _result = _einsum_multiplication(sum_dims, _result, operand)

    return _result


def einsum(equation, *operands):
    r"""
    According to the Einstein summation Convention (Einsum),
    the product of the input tensor elements is summed along the specified dimension.
    You can use this operator to perform diagonal, reducesum, transpose, matmul, mul, inner product operations, etc.

    Note:
        The sublist format is also supported. For example, einsum_ext(op1, sublist1, op2, sublist2, ..., sublist_out).
        In this format, equation can be derived by the sublists which are made up of Python's Ellipsis and list of
        integers in [0, 52). Each operand is followed by a sublist and an output sublist is at the end.
        Dynamic shape, dynamic rank input is not supported in `graph mode (mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        equation (str): Notation based on the Einstein summation convention, represent the operation you want to do.
            the value can contain only letters, commas, ellipsis and arrow. The letters(must be in [a-zA-Z]) represent
            input tensor dimension, commas(,) represent separate tensors, ellipsis indicates the tensor dimension that
            you do not care about, the left of the arrow indicates the input tensors, and the right of it indicates the
            desired output dimension. If there are no arrows in the equation, the letters that appear exactly once in
            the equation will be part of the output, sorted in increasing alphabetical order. The output is computed by
            multiplying the input operands element-wise, with their dimensions aligned based on the letters, and then
            summing out the dimensions whose letters are not part of the output. If there is one arrow in the equation,
            the output letters must appear at least once for some input operand and at most once for the output.
        operands (Tensor): Input tensor used for calculation. The dtype of the tensor must be the same.

    Returns:
        Tensor, the shape of it can be obtained from the `equation` , and the dtype is the same as input tensors.

    Raises:
        TypeError: If `equation` is invalid, or the `equation` does not match the input tensor.
        ValueError: If the number in sublist is not in [0, 52) in sublist format.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> equation = "i->"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        7.0
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> equation = "i,i->i"
        >>> output = ops.einsum_ext(equation, x, y)
        >>> print(output)
        [ 2. 8. 12.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> output = ops.einsum_ext(equation, x, y)
        >>> print(output)
        [[16. 22.]
         [37. 52.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->ji"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->j"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        [5. 7. 9.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "...->"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        21.0
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float32)
        >>> equation = "j,i->ji"
        >>> output = ops.einsum_ext(equation, x, y)
        >>> print(output)
        [[ 2. 4. 1.]
         [ 4. 8. 2.]
         [ 6. 12. 3.]]
        >>> x = mindspore.Tensor([1, 2, 3, 4], mindspore.float32)
        >>> y = mindspore.Tensor([1, 2], mindspore.float32)
        >>> output = ops.einsum_ext(x, [..., 1], y, [..., 2], [..., 1, 2])
        >>> print(output)
        [[1. 2.]
         [2. 4.]
         [3. 6.]
         [4. 8.]]
    """
    if isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    if operands[0].device.type != 'npu':
        return execute('einsum', equation, operands)
    _equation, _operands = _einsum_convert_sublist(equation, *operands)
    _einsum_check_inputargs(_equation, _operands)
    return _einsum(_equation, _operands)

# flatten
def flatten(input, start_dim=0, end_dim=-1):
    if input.device.type == 'cpu':
        if end_dim < 0:
            end_dim = input.ndim + end_dim
        new_shape = input.shape[:start_dim] + (-1,) + input.shape[end_dim + 1:]
        return input.reshape(new_shape)
    return execute('flatten_ext', input, start_dim, end_dim)


# flip
def flip(input, dims):
    return execute('reverse_v2', input, dims)


# fliplr


# flipud


# kron


# rot90


# gcd


# histc


# histogram


# histogramdd


# meshgrid
def meshgrid(*tensors, indexing=None):
    if isinstance(tensors[0], (tuple, list)):
        tensors = tensors[0]
    if indexing is None:
        indexing = 'ij'
    return execute('meshgrid', tensors, indexing)


# lcm


# logcumsumexp

# ravel
def ravel(input):
    return input.reshape(-1)

# renorm


# repeat_interleave
def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    if input.device.type == 'npu' and ON_A1:

        if isinstance(repeats, core.Tensor):
            repeats = repeats.tolist()
        if not isinstance(repeats, (tuple, list)):
            repeats = (repeats,)
        for index, element in enumerate(repeats):
            if not isinstance(element, int):
                raise TypeError(f"For 'Tensor.repeat', each element in {repeats} should be int, but got "
                                f"{type(element)} at index {index}.")
        if dim is None:
            input = input.ravel()
            dim = 0

        dim = dim + input.ndim if dim < 0 else dim

        if len(repeats) == 1:
            repeats = repeats[0]
            if repeats == 0:
                return Tensor_(input.dtype, (0,))
            if input.dtype == mindspore.bool_:
                input = input.to(mindspore.int32)
                out = execute('repeat_elements', input, repeats, dim)
                return out.to(mindspore.bool_)
            return execute('repeat_elements', input, repeats, dim)
        size = input.shape[dim]
        if len(repeats) != size:
            raise ValueError(f"For 'Tensor.repeat', the length of 'repeats' must be the same as the shape of the "
                                f"original tensor in the 'axis' dimension, but got the length of 'repeats' "
                                f"{len(repeats)}, the shape of the original tensor in the 'axis' dimension {size}.")
        subs = core.split(input, 1, dim)
        repeated_subs = []
        for sub, rep in zip(subs, repeats):
            if rep != 0:
                repeated_subs.append(execute('repeat_elements', sub, rep, dim))
        return core.concat(repeated_subs, dim)

    if isinstance(repeats, int):
        return execute('repeat_interleave_int', input, repeats, dim, None)
    return execute('repeat_interleave_tensor', input, repeats, dim, None)


# roll
def roll(input, shifts, dims=None):
    return execute('roll', input, shifts, dims)


# searchsorted
def searchsorted(
    sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None
):
    dtype = core.int32 if bool(out_int32) else core.int64
    if (side == "left" and right is True):
        raise ValueError(f"For 'searchsorted', side and right can't be set to opposites,"
                         f"got side of left while right was True.")
    if side == "right":
        right = True
    return execute('search_sorted', sorted_sequence, values, sorter,
                   dtype_to_type_id('SearchSorted', 'dtype', dtype), right)

# tensordot

# trace

# tril
def tril(input, diagonal=0):
    return execute('tril_ext', input, diagonal)


# tril_indices

# triu
def triu(input, diagonal=0):
    return execute('triu', input, diagonal)

# triu_indices


# unflatten
def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return x.reshape(new_shape)

# vander


# view_as_real
def view_as_real(input):
    real_part = input.real.unsqueeze(-1)
    imag_part = input.imag.unsqueeze(-1)
    return core.concat((real_part, imag_part), -1)


# view_as_complex
def view_as_complex(input):
    return execute('view_as_complex', input)

# resolve_conj


# resolve_neg


def masked_fill(input, mask, value):
    if isinstance(value, float):
        if value == -float('inf'):
            value = finfo(input.dtype).min
        if value == float('inf'):
            value = finfo(input.dtype).max

    if isinstance(value, core.Tensor) and input.device != value.device:
        value = value.to(input.device)
    return execute('masked_fill', input, mask, value)

class finfo:
    def __init__(self, bits, min, max, eps, tiny, smallest_normal, resolution, dtype):
        self.bits = bits
        self.min = min
        self.max = max
        self.eps = eps
        self.tiny = tiny
        self.smallest_normal = smallest_normal
        self.resolution = resolution
        self.dtype = dtype

class iinfo:
    def __init__(self, bits, min, max, dtype):
        self.bits = bits
        self.min = min
        self.max = max
        self.dtype = dtype


finfo_dtype = {
    mindspore.bfloat16: finfo(
        bits=16,
        resolution=0.01,
        min=-3.38953e38,
        max=3.38953e38,
        eps=0.0078125,
        smallest_normal=1.17549e-38,
        tiny=1.17549e-38,
        dtype="bfloat16",
    ),
    mindspore.float16: finfo(
        bits=16,
        resolution=0.001,
        min=-65504,
        max=65504,
        eps=0.000976562,
        smallest_normal=6.10352e-05,
        tiny=6.10352e-05,
        dtype="float16",
    ),
    mindspore.float32: finfo(
        bits=32,
        resolution=1e-06,
        min=-3.40282e38,
        max=3.40282e38,
        eps=1.19209e-07,
        smallest_normal=1.17549e-38,
        tiny=1.17549e-38,
        dtype="float32",
    ),
    mindspore.float64: finfo(
        bits=64,
        resolution=1e-15,
        min=-1.79769e308,
        max=1.79769e308,
        eps=2.22045e-16,
        smallest_normal=2.22507e-308,
        tiny=2.22507e-308,
        dtype='float64',
    ),
}


def finfo(dtype):
    return finfo_dtype[dtype]


iinfo_dtype = {
    mindspore.int64: iinfo(bits=64, min=-9223372036854775808, max=9223372036854775807, dtype='int64'),
    mindspore.int32: iinfo(bits=32, min=-2147483648, max=2147483647, dtype='int32')
}

def iinfo(dtype):
    return iinfo_dtype[dtype]

def iinfo(dtype):
    return np.iinfo(core.dtype2np[dtype])


def contains(self, key):
    r"""
    Args:
        self (object): The object instance on which the method is called.
        key (object): The key to be checked for containment in the object.

    Returns:
        None: This function returns None, indicating whether the key is contained in the object.

    Raises:
        None
    """
    eq_res = eq(self, key)
    res = any(eq_res)
    return bool(res)


def stop_gradient(input):
    return execute('stop_gradient', input)


def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    return indices, dimension


def unfold(input, dimension, size, step):
    _indices, _dimension = _get_unfold_indices(input.shape, dimension, size, step)
    indices = core.tensor(_indices, device=input.device)
    output = execute('gather', input, indices, _dimension)
    output = core.swapaxes(output, _dimension + 1, -1)
    return output


def contiguous(input):
    return execute('contiguous', input)

def dyn_shape(input):
    return execute('dyn_shape', input)

def cross(input, other, dim=None, *, out=None):
    if dim is None:
        dim = -65530
    return execute('cross', input, other, dim)


__all__ = [
    "bincount",
    "broadcast_shapes",
    "broadcast_tensors",
    "broadcast_to",
    "cdist",
    "clone",
    "contains",
    "cross",
    "cumsum",
    "diag",
    "diagonal",
    "einsum",
    "finfo",
    "flatten",
    "flip",
    "iinfo",
    "masked_fill",
    "meshgrid",
    "repeat_interleave",
    "roll",
    "searchsorted",
    "stop_gradient",
    "tril",
    "triu",
    "unflatten",
    "unfold",
    "contiguous",
    "ravel",
    "dyn_shape",
    "diff",
    'view_as_complex',
    'view_as_real'
]
