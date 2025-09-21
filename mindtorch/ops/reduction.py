"""reduction op"""
from collections import namedtuple

import mindtorch
from mindtorch.executor import execute

max_out = namedtuple('max_out', ['values', 'indices'])
min_out = namedtuple('min_out', ['values', 'indices'])

# argmax
def argmax(input, dim=None, keepdim=False, **kwargs):
    dim = kwargs.pop('axis', dim)
    return execute('argmax', input, dim, keepdim)

# argmin
def argmin(input, dim=None, keepdim=False):
    return execute('argmin', input, dim, keepdim)

# amax
def amax(input, dim, keepdim=False):
    return execute('reduce_max', input, dim, keepdim)

# amin
def amin(input, dim, keepdim=False):
    return execute('reduce_min', input, dim, keepdim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
def all(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    dim = kwargs.pop('axis', dim)
    keepdim = kwargs.pop('keepdims', keepdim)
    return execute('reduce_all', input, dim, keepdim)

# any
def any(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return execute('reduce_any', input, dim, keepdim)

# max
def max(input, dim=None, keepdim=False, *, out=None, **kwargs):
    dim = kwargs.pop('axis', dim)
    if dim is None and not keepdim:
        return execute('max', input)
    if mindtorch.is_tensor(dim):
        return mindtorch.maximum(input, dim)
    output = execute('argmax_with_value', input, dim, keepdim)
    if out is None:
        return max_out(values=output[1], indices=output[0])

    out[0].data = output[1]
    out[1].data = output[0]
    return out

# min
def min(input, dim=None, keepdim=False, *, out=None):
    if dim is None and not keepdim:
        return execute('min', input)
    if mindtorch.is_tensor(dim):
        return mindtorch.minimum(input, dim)
    output = execute('argmin_with_value', input, dim, keepdim)
    if out is None:
        return min_out(values=output[1], indices=output[0])

    out[0].data = output[0]
    out[1].data = output[1]
    return out

# dist

# logsumexp
def logsumexp(input, dim, keepdim=False):
    return execute('logsumexp', input, dim, keepdim)

# mean
def mean(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    dim = kwargs.pop('axis', dim)
    return execute('mean', input, dim, keepdim, dtype)

# nanmean


# median
def median(input, dim=-1, keepdim=False):
    if dim is None:
        return execute('median', input)
    return execute('median_dim', input, dim, keepdim)

# nanmedian


# mode


# norm
def vector_norm(input, p=2, dim=None, keepdim=False, *, dtype=None):
    if float(p) in [0.0, 1.0, 2.0, 3.0]:
        return execute('linalg_vector_norm', input, float(p), dim, keepdim, dtype)
    if input.dtype in [mindtorch.bfloat16, mindtorch.float16, mindtorch.float32]:
        if dtype is None:
            return execute('lp_norm_v2', input, p, dim, keepdim, 0.0)
        return execute('lp_norm_v2', input, p, dim, keepdim, 0.0).to(dtype)

    cast_dtype = input.dtype if dtype is None else dtype
    input = input.to(mindtorch.float32)
    return execute('lp_norm_v2', input, p, dim, keepdim, 0.0).to(cast_dtype)

def matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None):
    ndim = A.ndim
    row_axis, col_axis = _check_matrix_norm_axis(dim, ndim)
    _check_matrix_norm_ord(ord)
    if ord == 'fro':
        return vector_norm(A, 2, dim, keepdim, dtype=dtype)
    if ord == 'nuc':
        res = _multi_svd_norm(A, row_axis, col_axis, 'sum')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord == 2:
        res = _multi_svd_norm(A, row_axis, col_axis, 'amax')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord == -2:
        res = _multi_svd_norm(A, row_axis, col_axis, 'amin')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord in [float('inf'), -float('inf')]:
        row_axis, col_axis = col_axis, row_axis
    if not keepdim and col_axis > row_axis:
        col_axis -= 1
    if ord < 0:
        return amin(vector_norm(A, 1, row_axis, keepdim, dtype=dtype), col_axis, keepdim)
    return amax(vector_norm(A, 1, row_axis, keepdim, dtype=dtype), col_axis, keepdim)

def norm(input, p='fro', dim=None, keepdim=False, dtype=None):
    if not isinstance(input, mindtorch.Tensor):
        raise TypeError(f"For `norm`, the `input` must be Tensor!, but get {type(input)}.")
    if isinstance(p, (bool, int, float)):
        return vector_norm(input, p, dim, keepdim, dtype=dtype)
    if p == 'fro':
        if isinstance(dim, (list, tuple)) and len(dim) > 2:
            raise ValueError(f"For `norm`, the size of `dim` cannot be greater than 2 "
                             f"when the norm mode is `fro`.")
        return execute('linalg_vector_norm', input, 2.0, dim, keepdim,
                       dtype if dtype is None else dtype)
    if p == 'nuc':
        dim = tuple(range(input.ndim)) if dim is None else dim
        return matrix_norm(input, p, dim, keepdim, dtype=dtype)
    raise ValueError(f"For `norm`, the value of `p` must be one of [int, float, inf, -inf, 'fro', 'nuc',] "
                     f"but got `{p}`.")

# nansum
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    return execute('nansum', input, dim, keepdim, dtype)

# prod
def prod(input, dim=None, keepdim=False, *, dtype=None):
    return execute('prod', input, dim, keepdim, dtype)

# quantile

# nanquantile

# std
def my_std(input_tensor, dim=None, unbiased=True, keepdim=False):
    """
    手动实现类似 torch.std 的功能，计算张量的标准差。
    
    参数:
        input_tensor (torch.Tensor): 输入张量。
        dim (int 或 tuple, 可选): 要计算标准差的维度。默认为 None，计算全局标准差。
        unbiased (bool, 可选): 是否使用无偏估计 (贝塞尔校正)。默认为 True。
        keepdim (bool, 可选): 输出是否保持输入张量的维度。默认为 False。
        
    返回:
        torch.Tensor: 包含标准差值的张量。
    """
    # 处理空张量输入
    if input_tensor.numel() == 0:
        raise ValueError("my_std(): input tensor is empty")
    
    # 如果未指定 dim，则计算全局标准差
    if dim is None:
        # 计算均值
        mean = input_tensor.mean()
        # 计算与均值的平方差
        squared_diff = (input_tensor - mean) ** 2
        # 计算平方差的平均值（方差）
        # 根据 unbiased 选择分母
        n = input_tensor.numel()
        divisor = n - 1 if unbiased else n
        variance = squared_diff.sum() / divisor
        # 标准差是方差的平方根
        std_dev = mindtorch.sqrt(variance)
        return std_dev

    # 如果指定了 dim，则沿指定维度计算标准差
    else:
        # 计算沿指定维度的均值，keepdim=True 为了广播
        mean = input_tensor.mean(dim=dim, keepdim=True)
        # 计算平方差
        squared_diff = (input_tensor - mean) ** 2
        # 计算沿指定维度的平方差和
        sum_squared_diff = squared_diff.sum(dim=dim, keepdim=keepdim)
        # 获取沿指定维度缩减后的元素数
        n = input_tensor.size(dim) if isinstance(dim, int) else mindtorch.prod(mindtorch.tensor([input_tensor.size(d) for d in dim])).item()
        divisor = (n - 1) if unbiased else n
        # 计算方差
        variance = sum_squared_diff / divisor
        # 标准差是方差的平方根
        std_dev = mindtorch.sqrt(variance)
        return std_dev

def std(input, dim=None, *, correction=1, keepdim=False, **kwargs):
    dim = kwargs.pop('axis', dim)
    if input.device.type == 'cuda':
        return my_std(input, dim, bool(correction), keepdim)
    return execute('std', input, dim, correction, keepdim)

# std_mean
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    return execute('std_mean', input, dim, correction, keepdim)

# sum
def sum(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    dim = kwargs.pop('axis', dim)
    if 0 in input.shape:
        return mindtorch.tensor(0, dtype=dtype, device=input.device)
    return execute('sum', input, dim, keepdim, dtype)

# unique
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if dim is None:
        y, inverse, counts = execute('unique2',
            input, sorted, return_inverse, return_counts)
    else:
        y, inverse, counts = execute('unique_dim', input, sorted, return_inverse, dim)
    if return_inverse and return_counts:
        return y, inverse, counts
    if return_inverse:
        return y, inverse
    if return_counts:
        return y, counts
    return y

def unique_consecutive_optimized(input, return_inverse=False, return_counts=False, dim=None):
    """
    优化版的 torch.unique_consecutive 手动实现。
    """
    if dim is None:
        input_flat = input.flatten()
        return _unique_consecutive_1d(input_flat, return_inverse, return_counts)
    else:
        return _unique_consecutive_nd(input, dim, return_inverse, return_counts)

def _unique_consecutive_1d(input, return_inverse, return_counts):
    """处理一维张量的优化实现"""
    if input.numel() == 0:
        return _handle_empty_input(input, return_inverse, return_counts)
    
    # 找到变化点的位置
    diff = input[1:] != input[:-1]
    change_points = mindtorch.cat([
        mindtorch.tensor([True], device=input.device), 
        diff, 
        mindtorch.tensor([True], device=input.device)
    ])
    change_indices = mindtorch.where(change_points)[0]
    
    # 提取唯一值
    unique_values = input[change_indices[:-1]]
    
    # 准备返回结果
    result = [unique_values]
    
    # 处理逆向索引
    if return_inverse:
        inverse_indices = mindtorch.repeat_interleave(
            mindtorch.arange(len(unique_values), device=input.device), 
            mindtorch.diff(change_indices)
        )
        result.append(inverse_indices)
    
    # 处理计数
    if return_counts:
        counts = mindtorch.diff(change_indices)
        result.append(counts)
    
    return result[0] if len(result) == 1 else tuple(result)

def _unique_consecutive_nd(input, dim, return_inverse, return_counts):
    """处理多维张量的实现"""
    # 将目标维度移动到最后一维
    input_transposed = input.transpose(dim, -1)
    original_shape = input_transposed.shape
    input_2d = input_transposed.reshape(-1, original_shape[-1])
    
    results = []
    for i in range(input_2d.shape[0]):
        slice_result = _unique_consecutive_1d(input_2d[i], return_inverse, return_counts)
        if isinstance(slice_result, tuple):
            results.append(slice_result)
        else:
            results.append((slice_result,))
    
    # 重组结果
    return _reconstruct_nd_results(results, original_shape, dim, return_inverse, return_counts)

def _handle_empty_input(input, return_inverse, return_counts):
    """处理空输入的情况"""
    empty_tensor = mindtorch.tensor([], dtype=input.dtype, device=input.device)
    if return_inverse and return_counts:
        return empty_tensor, mindtorch.tensor([], dtype=mindtorch.long, device=input.device), mindtorch.tensor([], dtype=mindtorch.long, device=input.device)
    elif return_inverse:
        return empty_tensor, mindtorch.tensor([], dtype=mindtorch.long, device=input.device)
    elif return_counts:
        return empty_tensor, mindtorch.tensor([], dtype=mindtorch.long, device=input.device)
    else:
        return empty_tensor

def _reconstruct_nd_results(results, original_shape, dim, return_inverse, return_counts):
    """
    重组多维处理结果
    """
    # 确定最大唯一值长度（用于填充）
    max_unique_len = max(len(result[0]) for result in results)
    batch_size = original_shape[0]  # 第一维的大小（其他维度的乘积）
    
    # 重组唯一值张量
    unique_dtype = results[0][0].dtype
    unique_device = results[0][0].device
    
    # 创建输出唯一值张量
    unique_output_shape = list(original_shape)
    unique_output_shape[-1] = max_unique_len
    unique_output = mindtorch.full(unique_output_shape, 0, dtype=unique_dtype, device=unique_device)
    
    # 填充唯一值张量
    for i, result in enumerate(results):
        unique_slice = result[0]
        unique_output[i, :len(unique_slice)] = unique_slice
    
    # 重塑回原始形状（不包括被处理的维度）
    final_unique_shape = list(original_shape[:-1]) + [max_unique_len]
    unique_output = unique_output.reshape(final_unique_shape)
    
    # 如果需要恢复原始维度顺序
    if dim != -1:
        # 计算原始维度顺序
        dim_perm = list(range(unique_output.dim()))
        # 将最后一个维度移回原始位置
        dim_perm.append(dim_perm.pop(-1))
        # 调整维度顺序
        unique_output = unique_output.permute(dim_perm)
    
    # 处理返回结果
    output_results = [unique_output]
    
    # 处理逆向索引
    if return_inverse:
        inverse_shape = list(original_shape)
        inverse_output = mindtorch.zeros(inverse_shape, dtype=mindtorch.long, device=unique_device)
        
        for i, result in enumerate(results):
            if len(result) > 1:  # 确保有逆向索引
                inverse_slice = result[1]
                inverse_output[i, :len(inverse_slice)] = inverse_slice
        
        # 重塑逆向索引到原始形状
        inverse_output = inverse_output.reshape(original_shape[:-1] + [original_shape[-1]])
        
        # 调整维度顺序
        if dim != -1:
            inverse_output = inverse_output.permute(dim_perm)
        
        output_results.append(inverse_output)
    
    # 处理计数
    if return_counts:
        counts_shape = list(original_shape[:-1]) + [max_unique_len]
        counts_output = mindtorch.zeros(counts_shape, dtype=mindtorch.long, device=unique_device)
        
        for i, result in enumerate(results):
            counts_index = 2 if return_inverse else 1  # 确定计数在结果中的位置
            if len(result) > counts_index:
                counts_slice = result[counts_index]
                counts_output[i, :len(counts_slice)] = counts_slice
        
        # 调整计数张量的维度顺序
        if dim != -1:
            counts_output = counts_output.permute(dim_perm)
        
        output_results.append(counts_output)
    
    # 返回适当的结果组合
    if len(output_results) == 1:
        return output_results[0]
    else:
        return tuple(output_results)


def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    if input.device.type == 'cuda':
        return unique_consecutive_optimized(input, return_inverse, return_counts, dim)
    output, idx, counts = execute('unique_consecutive', input, return_inverse, return_counts, dim)
    if return_inverse and return_counts:
        return output, idx, counts
    if return_inverse:
        return output, idx
    if return_counts:
        return output, counts
    return output

# var
def var(input, dim=None, *, correction=1, keepdim=False):
    return execute('var', input, dim, correction, keepdim)

# var_mean
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    return execute('var_mean', input, dim, correction, keepdim)

# count_nonzero
def count_nonzero(input, dim=-1):
    return execute('count_nonzero', input, dim)

__all__ = ['all', 'amax', 'amin', 'aminmax', 'any', 'argmax', 'argmin', 'count_nonzero',
           'logsumexp', 'max', 'mean', 'median', 'min', 'nansum',
           'norm', 'prod', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive',
           'var', 'var_mean']
 