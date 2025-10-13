"""other op"""
import numpy as np
import mindspore
import mindtorch
from mindtorch.executor import execute
from ..configs import ON_A2

# atleast_2d


# atleast_3d


# bincount
def bincount(input, weights=None, minlength=0):
    return execute('bincount', input, weights, minlength)

# block_diag


# broadcast_tensors
def broadcast_tensors(*tensors):
    target_shape = broadcast_shapes(*[t.shape for t in tensors])
    broadcasted_tensors = [t.broadcast_to(target_shape) for t in tensors]
    return broadcasted_tensors


# broadcast_to
def broadcast_to(input, shape):
    if input.shape == shape:
        return input

    new_shape = ()
    for s in shape:
        if not isinstance(s, int):
            s = s.item()
        new_shape += (s,)
    return execute('broadcast_to', input, new_shape)


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
def bucketize(input, boundaries, *, out_int32=False, right=False, out=None):
    if isinstance(boundaries, mindtorch.Tensor):
        boundaries = boundaries.tolist()
    
    if not boundaries:
        return mindtorch.zeros_like(input)
    out = execute('bucketize', input, boundaries, right)
    if out_int32:
        return out.to(mindtorch.int32)
    return out

# cartesian_prod


# cdist
def cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    return execute('cdist', x1, x2, p)

# clone
def clone(input, *, memory_format=mindtorch.preserve_format):
    return execute('clone', input)


# combinations


# corrcoef


# cov


# cross

# cummax

# cummin

# cumprod

# cumsum
def cumsum(input, dim=None, dtype=None, **kwargs):
    dim = kwargs.pop('axis', dim)
    if input.dtype in [mindtorch.int64, mindtorch.bool]:
        return execute('cumsum', input.int(), dim, None).long()
    if dtype is not None and dtype == mindtorch.int64:
        return execute('cumsum', input, dim, None).long()
    return execute('cumsum', input, dim, dtype)

# diag
def my_diag(input_tensor, diagonal=0):
    """
    手动实现 torch.diag 的功能
    参数:
        input_tensor: 输入张量，可以是一维（向量）或二维（矩阵）
        diagonal: 对角线的位置，0为主对角线，正数为上对角线，负数为下对角线
    返回:
        根据输入维度返回对角矩阵或对角线元素
    """
    if input_tensor.dim() == 1:  # 输入是向量，构建对角矩阵
        n = input_tensor.size(0)
        output = mindtorch.zeros(n, n, dtype=input_tensor.dtype, device=input_tensor.device)
        for i in range(n):
            output[i, i] = input_tensor[i]
        return output
        
    elif input_tensor.dim() == 2:  # 输入是矩阵，提取对角线元素
        rows, cols = input_tensor.shape
        if diagonal >= 0:
            diag_len = min(rows, cols - diagonal)
        else:
            diag_len = min(rows + diagonal, cols)
        
        if diag_len <= 0:  # 对角线长度无效则返回空张量
            return mindtorch.tensor([], dtype=input_tensor.dtype, device=input_tensor.device)
            
        output = mindtorch.zeros(diag_len, dtype=input_tensor.dtype, device=input_tensor.device)
        for i in range(diag_len):
            if diagonal >= 0:
                output[i] = input_tensor[i, i + diagonal]
            else:
                output[i] = input_tensor[i - diagonal, i]
        return output
        
    else:
        raise RuntimeError("输入张量必须是一维或二维")

def diag(input, diagonal=0, *, out=None):
    if input.device.type == 'cuda':
        return my_diag(input, diagonal)
    return execute('diag', input, diagonal)

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

    if isinstance(value, mindtorch.Tensor) and value.shape == ():
        return True

    return False

def _diff_helper(input, n, dim):
    """calculate the forward difference"""
    out_len = input.shape[dim] - 1
    is_bool = (input.dtype == mindtorch.bool)
    result = input

    for _ in range(n):  # pylint: disable=unused-variable
        if is_bool:
            result = mindtorch.logical_xor(mindtorch.narrow(result, dim, 1, out_len), mindtorch.narrow(result, dim, 0, out_len))
        else:
            result = mindtorch.sub(mindtorch.narrow(result, dim, 1, out_len), mindtorch.narrow(result, dim, 0, out_len))

        if out_len == 0:
            break
        out_len -= 1

    return result


def _diff_prepend_append_on_dim(input, prepend, append, dim):
    """append tensor on dim"""
    if prepend is not None and append is None:
        return mindtorch.cat((prepend, input), dim)

    if prepend is None and append is not None:
        return mindtorch.cat((input, append), dim)

    return mindtorch.cat((prepend, input, append), dim)


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
    if isinstance(equation, mindtorch.Tensor):
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
        if not isinstance(operand, mindtorch.Tensor):
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
                operand = mindtorch.diagonal(operand, 0, label_dims[label], dim)
                diag_perm = []
                diag_dim = 0
                for i in range(len(operand.shape)):
                    if i == label_dims[label]:
                        diag_perm.append(len(operand.shape) - 1)
                    else:
                        diag_perm.append(diag_dim)
                        diag_dim += 1
                operand = mindtorch.permute(operand, tuple(diag_perm))
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
            operand = mindtorch.reshape(operand, operand_shape)
        operand = mindtorch.permute(operand, tuple(align_axis))
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
                l_tensor = mindtorch.sum(l_tensor, i, True)
            elif sum_r:
                r_tensor = mindtorch.sum(r_tensor, i, True)
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

    l_tensor = mindtorch.permute(l_tensor, tuple(l_perm_axis))
    l_tensor = mindtorch.reshape(l_tensor, l_reshape_shape)

    r_tensor = mindtorch.permute(r_tensor, tuple(r_perm_axis))
    r_tensor = mindtorch.reshape(r_tensor, r_reshape_shape)

    output = mindtorch.bmm(l_tensor, r_tensor)
    output = mindtorch.reshape(output, out_reshape_shape)
    output = mindtorch.permute(output, tuple(out_perm_axis))

    output_origin_shape = output.shape
    output_squeeze_shape = []
    for dim in range(len(output_origin_shape)):
        if dim not in sum_dims:
            output_squeeze_shape.append(output_origin_shape[dim])

    return mindtorch.reshape(output, output_squeeze_shape)


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
        return mindtorch.zeros(output_shape, dtype=_result.dtype)

    # Sum or squeeze dimensions that is 1 for all rest operands.
    _reduce_dim = _output_rank
    for dim in range(_output_rank, _align_rank):
        if _dim_last_op[dim] == 0:
            if _result.shape[_reduce_dim] == 1:
                _result = mindtorch.squeeze(_result, _reduce_dim)
            else:
                _result = mindtorch.sum(_result, _reduce_dim)
        else:
            _reduce_dim += 1

    # Compute multiplication if operands are more than two.
    for i in range(1, len(_adjust_operands)):
        operand = _adjust_operands[i]
        dim = _output_rank
        sum_dims = []
        for j in range(_output_rank, _align_rank):
            if _dim_last_op[j] < i:
                operand = mindtorch.squeeze(operand, dim)
            elif _dim_last_op[j] == i:
                if _result.shape[dim] == 1:
                    operand = mindtorch.sum(operand, dim)
                    _result = mindtorch.squeeze(_result, dim)
                else:
                    sum_dims.append(dim)
                    dim += 1
            else:
                dim += 1

        if sum_dims == []:
            _result = mindtorch.mul(_result, operand)
        elif len(sum_dims) == len(_result.shape):
            _result = mindtorch.dot(mindtorch.flatten(_result), mindtorch.flatten(operand))
        else:
            _result = _einsum_multiplication(sum_dims, _result, operand)

    return _result


def einsum(equation, *operands):
    r"""
    According to the Einstein summation Convention (Einsum),
    the product of the input tensor elements is summed along the specified dimension.
    You can use this operator to perform diagonal, reducesum, transpose, matmul, mul, inner product operations, etc.

    Note:
        The sublist format is also supported. For example, einsum(op1, sublist1, op2, sublist2, ..., sublist_out).
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
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        7.0
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> equation = "i,i->i"
        >>> output = ops.einsum(equation, x, y)
        >>> print(output)
        [ 2. 8. 12.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> output = ops.einsum(equation, x, y)
        >>> print(output)
        [[16. 22.]
         [37. 52.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->ji"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->j"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        [5. 7. 9.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "...->"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        21.0
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float32)
        >>> equation = "j,i->ji"
        >>> output = ops.einsum(equation, x, y)
        >>> print(output)
        [[ 2. 4. 1.]
         [ 4. 8. 2.]
         [ 6. 12. 3.]]
        >>> x = mindspore.Tensor([1, 2, 3, 4], mindspore.float32)
        >>> y = mindspore.Tensor([1, 2], mindspore.float32)
        >>> output = ops.einsum(x, [..., 1], y, [..., 2], [..., 1, 2])
        >>> print(output)
        [[1. 2.]
         [2. 4.]
         [3. 6.]
         [4. 8.]]
    """
    if isinstance(operands[0], (list, tuple)):
        operands = operands[0]
    # if operands[0].device.type == 'cuda':
    #     return execute('einsum', equation, operands, device=operands[0].device)
    _equation, _operands = _einsum_convert_sublist(equation, *operands)
    _einsum_check_inputargs(_equation, _operands)
    return _einsum(_equation, _operands)

# flatten
def flatten(input, start_dim=0, end_dim=-1):
    return execute('flatten', input, start_dim, end_dim)


# flip
def flip(input, dims):
    return execute('reverse_v2', input, dims)


# fliplr


# flipud


# kron


# rot90


# gcd


# histc
def manual_histc_searchsorted(input_tensor, bins=100, min=0, max=0):
    """
    使用 searchsorted 实现 histc，适用于浮点数，更精确地模拟边界。
    """
    if min == 0 and max == 0:
        min = input_tensor.min().item()
        max = input_tensor.max().item()

    bin_width = (max - min) / bins
    # 生成 bin 的右边界（除了最后一个，因为histc的最后一个bin是闭区间[2,5]）
    bin_edges = mindtorch.linspace(min, max, bins + 1, device=input_tensor.device)
    # 调整最后一个区间的右边界为无穷大，以确保等于max的值被包含在最后一个bin
    # 同时，其他区间保持左闭右开
    bin_edges[-1] = float('inf') 

    flattened = input_tensor.view(-1)
    # 找到每个元素应该插入到 bin_edges 中的位置，然后减1得到 bin 索引
    # side='right' 表示返回的是使得 sorted_sequence[i-1] < v <= sorted_sequence[i] 成立的索引 i
    indices = mindtorch.searchsorted(bin_edges, flattened, side='right') - 1

    # 处理小于 min 的值（索引会变成 -1）
    valid_mask = (indices >= 0)
    indices_valid = indices[valid_mask]
    # 同样需要确保索引不超过 bins-1（理论上由于bin_edges[-1]=inf，不会超过，但保险起见）
    indices_valid = mindtorch.clamp(indices_valid, 0, bins - 1)

    # 使用 bincount 统计有效的索引
    histogram = mindtorch.bincount(indices_valid, minlength=bins)
    return histogram.float() # 保持与 histc 输出类型一致

def histc(input, bins=100, min=0, max=0):
    if input.device.type == 'cuda':
        return manual_histc_searchsorted(input, bins, min, max)
    return execute('histc', input, bins, min, max)

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
def efficient_repeat_interleave(input_tensor, repeats, dim=None):
    """
    高效实现 mindtorch.repeat_interleave 的功能，支持 repeats 为 int 或 list/tensor。
    
    参数:
        input_tensor (Tensor): 输入张量。
        repeats (int 或 list 或 Tensor): 每个元素的重复次数。
        dim (int, optional): 沿着哪个维度进行重复。如果为None，则先将输入张量展平。
    
    返回:
        Tensor: 重复后的张量。
    """
    if dim is None:
        input_tensor = input_tensor.flatten()
        dim = 0

    # 确保 dim 是有效的维度
    if dim < 0:
        dim += input_tensor.dim()

    # 将 repeats 统一转换为 LongTensor 并确保其在正确的设备上
    if isinstance(repeats, int):
        repeats_tensor = mindtorch.tensor([repeats], device=input_tensor.device, dtype=mindtorch.long)
        uniform_repeat = True
    elif isinstance(repeats, (list, tuple)):
        repeats_tensor = mindtorch.tensor(repeats, device=input_tensor.device, dtype=mindtorch.long)
        uniform_repeat = False
    elif isinstance(repeats, mindtorch.Tensor):
        repeats_tensor = repeats.to(device=input_tensor.device, dtype=mindtorch.long)
        uniform_repeat = False
    else:
        raise TypeError("repeats must be an int, a list, or a mindtorch.Tensor")

    # 获取输入张量在目标维度上的大小
    dim_size = input_tensor.size(dim)
    
    if uniform_repeat:
        # ✅ 优化路径：当所有元素重复次数相同时，使用 expand 和 reshape 避免循环
        # 此方法利用广播机制，非常高效
        unsqueezed_tensor = input_tensor.unsqueeze(dim + 1)
        expanded_shape = list(input_tensor.shape)
        expanded_shape[dim] = -1
        expanded_shape.insert(dim + 1, repeats_tensor.item())
        expanded_tensor = unsqueezed_tensor.expand(*expanded_shape)
        
        final_shape = list(input_tensor.shape)
        final_shape[dim] *= repeats_tensor.item()
        output = expanded_tensor.reshape(*final_shape)
    else:
        # 🔄 当重复次数不同时，需要构建索引
        # 检查 repeats_tensor 的长度是否与目标维度的长度匹配
        if len(repeats_tensor) != dim_size:
            raise ValueError(f"repeats must have length {dim_size} along dimension {dim}, but got {len(repeats_tensor)}")
        
        # 生成索引：例如 repeats_tensor = [2, 3, 1] -> index = [0, 0, 1, 1, 1, 2]
        # 使用 cumsum 计算总重复次数以预分配空间
        total_repeats = repeats_tensor.sum().item()
        index = mindtorch.zeros(total_repeats, dtype=mindtorch.long, device=input_tensor.device)
        
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

        output = input_tensor.index_select(dim, index)

    return output

def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    if input.device.type == 'npu' and ON_A2:
        if isinstance(repeats, int):
            return execute('repeat_interleave_int', input, repeats, dim, None)
        return execute('repeat_interleave_tensor', input, repeats, dim, None)
    return efficient_repeat_interleave(input, repeats, dim)


# roll
def roll(input, shifts, dims=None):
    if input.device.type == 'npu':
        return execute('roll', input, shifts, dims)
    # 处理 dims 为 None 的情况：先展平，操作后再恢复形状[4,6](@ref)
    if dims is None:
        original_shape = input.shape
        flattened = input.flatten()
        rolled_flattened = roll(flattened, shifts, dims=0)
        return rolled_flattened.reshape(original_shape)
    
    # 确保 shifts 和 dims 为元组以便统一处理[1,6](@ref)
    if not isinstance(shifts, tuple):
        shifts = (shifts,)
    if not isinstance(dims, tuple):
        dims = (dims,)
    
    # 检查 shifts 和 dims 长度是否匹配
    if len(shifts) != len(dims):
        raise ValueError("shifts 和 dims 必须具有相同的长度")
    
    result = input.clone()  # 创建输入张量的副本以避免修改原张量
    
    # 对每个需要移动的维度依次进行处理[2](@ref)
    for shift, dim in zip(shifts, dims):
        # 确保维度有效
        if dim >= result.dim():
            raise ValueError("维度索引超出张量的维度范围")
        
        # 获取该维度的长度
        dim_size = result.size(dim)
        # 处理负的 shift 值：正向移动 shift + dim_size 等同于反向移动 dim_size - shift
        effective_shift = shift % dim_size
        if effective_shift == 0:
            continue  # 移动 0 步，无需操作
        
        # 沿指定维度切片并重新拼接[1,3](@ref)
        # 将张量沿该维度分成两部分：[第一部分: 从开始到 (dim_size - effective_shift)], [第二部分: 从 (dim_size - effective_shift) 到结束]
        # 然后交换这两部分的位置
        slices_pre = [slice(None)] * result.dim()
        slices_pre[dim] = slice(dim_size - effective_shift, None)
        part1 = result[slices_pre]
        
        slices_post = [slice(None)] * result.dim()
        slices_post[dim] = slice(0, dim_size - effective_shift)
        part2 = result[slices_post]
        
        # 沿该维度拼接两部分
        result = mindtorch.concat((part1, part2), dim)
    
    return result

# searchsorted
def searchsorted(
    sorted_sequence, values, *, out_int32=False, right=False, side=None, sorter=None
):
    dtype = mindtorch.int32 if bool(out_int32) else mindtorch.int64
    if (side == "left" and right is True):
        raise ValueError(f"For 'searchsorted', side and right can't be set to opposites,"
                         f"got side of left while right was True.")
    if side == "right":
        right = True
    return execute('search_sorted', sorted_sequence, values, sorter, dtype, right)

# tensordot

# trace

# tril
def tril(input, diagonal=0):
    return execute('tril', input, diagonal)


# tril_indices

# triu
def triu(input, diagonal=0):
    return execute('triu', input, diagonal)

# triu_indices
def triu_indices(row, col, offset=0, *, dtype=mindtorch.long, device='cpu', layout=mindtorch.strided):
    return execute('triu_indices', row, col, offset, dtype, device=device)


# unflatten
def unflatten(x, dim, sizes):
    new_shape = x.shape[:dim] + sizes
    return x.reshape(new_shape)

# vander


# view_as_real
def view_as_real(input):
    real_part = input.real.unsqueeze(-1)
    imag_part = input.imag.unsqueeze(-1)
    return mindtorch.concat((real_part, imag_part), -1)


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

    if isinstance(value, mindtorch.Tensor) and input.device != value.device:
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
    return np.iinfo(mindtorch.dtype2np[dtype])


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

def detach(input):
    return stop_gradient(input)

def _get_unfold_indices(input_shape, dimension, size, step):
    if dimension < 0:
        dimension += len(input_shape)
    indices = []
    for i in range(0, input_shape[dimension] - size + 1, step):
        indices.append(list(range(i, i + size)))

    return indices, dimension


def unfold(input, dimension, size, step):
    _indices, _dimension = _get_unfold_indices(input.shape, dimension, size, step)
    indices = mindtorch.tensor(_indices, device=input.device)
    output = execute('gather', input, indices, _dimension, 0)
    output = mindtorch.moveaxis(output, _dimension + 1, -1)
    return output


def contiguous(input):
    return execute('contiguous', input)

def dyn_shape(input):
    return execute('tensor_shape', input)

def cross(input, other, dim=None, *, out=None):
    if dim is None:
        dim = -65530
    return execute('cross', input, other, dim)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    dot_product = mindtorch.sum(x1 * x2, dim=dim)
    
    # 2. 计算L2范数 (||x|| 和 ||y||)
    norm_vec1 = mindtorch.norm(x1, p=2, dim=dim)
    norm_vec2 = mindtorch.norm(x2, p=2, dim=dim)
    
    # 3. 计算余弦相似度: (x · y) / (||x|| * ||y|| + eps)
    cosine_sim = dot_product / (norm_vec1 * norm_vec2 + eps)
    
    return cosine_sim

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
    "triu_indices",
    "unflatten",
    "unfold",
    "contiguous",
    "ravel",
    "dyn_shape",
    "diff",
    'view_as_complex',
    'view_as_real',
    'bucketize',
    'cosine_similarity',
    'detach',
    'histc'
]
