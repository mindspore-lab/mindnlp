"""reduction op"""
import numbers
from collections import namedtuple
import mindspore
from mindspore import ops
from mindspore.ops._primitive_cache import _get_cache_prim
from ..configs import use_pyboost, DEVICE_TARGET

from ._inner import call_ms_func
from mindnlp import core

max_out = namedtuple('max_out', ['values', 'indices'])
min_out = namedtuple('min_out', ['values', 'indices'])

# argmax
has_argmax = hasattr(mindspore.mint, 'argmax')
def argmax(input, dim=None, keepdim=False):
    if use_pyboost() and has_argmax:
        return mindspore.mint.argmax(input, dim, keepdim)
    return ops.argmax(input, dim, keepdim)

# argmin
has_argmin = hasattr(mindspore.mint, 'argmin')
def argmin(input, dim=None, keepdim=False):
    if use_pyboost() and has_argmin:
        return mindspore.mint.argmin(input, dim, keepdim)
    return ops.argmin(input, dim, keepdim)

# amax
has_amax = hasattr(mindspore.mint, 'amax')
def amax(input, dim=(), keepdim=False):
    if use_pyboost() and has_amax:
        return mindspore.mint.amax(input, dim, keepdim)
    return ops.amax(input, dim, keepdim)

# amin
has_amin = hasattr(mindspore.mint, 'amin')
def amin(input, dim, keepdim=False):
    if use_pyboost() and has_amin:
        return mindspore.mint.amin(input, dim, keepdim)
    return ops.amin(input, dim, keepdim)

# aminmax
def aminmax(input, *, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return amin(input, dim, keepdim), amax(input, dim, keepdim)

# all
has_all = hasattr(mindspore.mint, 'all')
def all(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    axis = kwargs.get('axis', None)
    keepdims = kwargs.get('keepdims', None)
    if axis is not None:
        dim = axis
    if keepdims:
        keepdim = keepdims

    if use_pyboost() and has_all:
        return mindspore.mint.all(input, dim, keepdim).to(input.dtype)
    return ops.all(input, dim, keepdim).to(input.dtype)

# any
has_any = hasattr(mindspore.mint, 'any')
def any(input, dim=None, keepdim=False, *, out=None):
    if use_pyboost() and has_any:
        if dim is None:
            return call_ms_func(mindspore.mint.any, input, out=out)
        else:
            return call_ms_func(mindspore.mint.any, input, dim, keepdim, out=out)
    return ops.any(input, dim, keepdim)

# max
has_max = hasattr(mindspore.mint, 'max')
def max(*args, **kwargs):
    out = kwargs.pop('out', None)
    if 'dim' in kwargs and 'keepdim' not in kwargs:
        kwargs['keepdim'] = False
    if 'axis' in kwargs:
        kwargs['dim'] = kwargs.pop('axis')
    out = mindspore.mint.max(*args, **kwargs)
    if isinstance(out, tuple):
        return max_out(values=out[0], indices=out[1])
    return out

# min
has_min = hasattr(mindspore.mint, 'min')
def min(*args, **kwargs):
    out = kwargs.pop('out', None)
    if 'dim' in kwargs and 'keepdim' not in kwargs:
        kwargs['keepdim'] = False
    out = mindspore.mint.min(*args, **kwargs)
    if isinstance(out, tuple):
        return min_out(values=out[0], indices=out[1])
    return out

# dist


# logsumexp
has_logsumexp = hasattr(mindspore.mint, 'logsumexp')
def logsumexp(input, dim, keepdim=False):
    if use_pyboost() and has_logsumexp:
        return mindspore.mint.logsumexp(input, dim, keepdim)
    return ops.logsumexp(input, dim, keepdim)

# mean
has_mean = hasattr(mindspore.mint, 'mean')
def mean(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    axis = kwargs.get('axis', None)
    if axis is not None:
        dim = axis
    if use_pyboost() and has_mean:
        return mindspore.mint.mean(input, dim, keepdim, dtype=dtype)
    out = ops.mean(input, dim, keepdim)
    if dtype is not None:
        out = out.astype(dtype)
    return out

# nanmean


# median
has_median = hasattr(mindspore.mint, 'median')
def median(input, dim=-1, keepdim=False):
    if use_pyboost() and has_median:
        return mindspore.mint.median(input, dim, keepdim)
    return ops.median(input, dim, keepdim)

# nanmedian
def nanmedian(input, dim=-1, keepdim=False):
    return ops.nanmedian(input, dim, keepdim)

# mode


# norm
has_norm = hasattr(mindspore.mint, 'norm')
def norm(input, p='fro', dim=None, keepdim=False, out=None, dtype=None):
    if use_pyboost() and has_norm:
        return call_ms_func(mindspore.mint.norm, input, p, dim, keepdim, out=out, dtype=dtype)
    if p == 'fro':
        p = None
    return ops.norm(input, p, dim, keepdim, dtype=dtype)

# nansum
has_nansum = hasattr(mindspore.mint, 'nansum')
def nansum(input, dim=None, keepdim=False, *, dtype=None):
    if use_pyboost() and has_nansum:
        return mindspore.mint.nansum(input, dim, keepdim, dtype=dtype)
    return ops.nansum(input, dim, keepdim, dtype=dtype)

# prod
has_prod = hasattr(mindspore.mint, 'prod')
def prod(input, dim=None, keepdim=False, *, dtype=None):
    if use_pyboost() and has_prod:
        return mindspore.mint.prod(input, dim, keepdim, dtype=dtype)
    return ops.prod(input, dim, keepdim).to(dtype)

# quantile
def quantile_output_shape(
    original_dim,
    input_tensor,
    q,
    keepdim,
    wrapped_dim
):
    """
    计算分位数函数的输出形状
    
    参数:
        original_dim: 原始维度（None表示展平）
        input_tensor: 输入张量
        q: 分位数张量
        keepdim: 是否保留维度
        wrapped_dim: 处理后的维度索引
    """
    # 计算输出形状: q大小 + 缩减维度后的大小
    out_shape = []
    
    if original_dim is not None and input_tensor.dim() > 0:
        # 保留原始维度结构
        out_shape = list(input_tensor.shape)
        if keepdim:
            out_shape[wrapped_dim] = 1
        else:
            del out_shape[wrapped_dim]
    elif keepdim:
        # 当展平但需保留维度时创建全1形状
        out_shape = [1] * input_tensor.dim()
    
    if q.dim() > 0:
        # 添加分位数维度到最前面
        out_shape.insert(0, q.numel())
    
    return out_shape


def quantile(
    input_tensor,
    q,
    dim = None,
    keepdim: bool = False,
    interpolation: str = 'linear',
    ignore_nan: bool = False
):
    """
    PyTorch分位数函数的完整实现
    
    参数:
        input_tensor: 输入数据
        q: 分位数(0-1之间)
        dim: 计算维度
        keepdim: 是否保留维度
        interpolation: 插值模式 ('linear', 'lower', 'higher', 'nearest', 'midpoint')
        ignore_nan: 是否忽略NaN值
        
    返回:
        计算得到的分位数
    """
    if isinstance(q, numbers.Number):
        q = core.tensor(q, dtype=input_tensor.dtype)
    # ===== 1. 输入验证 =====
    device = input_tensor.device
    dtype = input_tensor.dtype

    # 验证分位数范围
    if device.type == 'cpu':
        if not core.all((q >= 0) & (q <= 1)):
            raise ValueError("quantile() q values must be in the range [0, 1]")
    
    # ===== 2. 维度处理 =====
    wrapped_dim = dim if dim is not None else 0
    original_dim = dim
    
    if dim is not None:
        # 验证维度有效性
        if dim < 0:
            dim = input_tensor.dim() + dim
        if dim < 0 or dim >= input_tensor.dim():
            raise ValueError(f"Dimension out of range (expected to be in range [{-input_tensor.dim()}, {input_tensor.dim()-1}])")
        wrapped_dim = dim
    
    # 计算输出形状
    out_shape = quantile_output_shape(original_dim, input_tensor, q, keepdim, wrapped_dim)
    
    # ===== 3. 数据预处理 =====
    # 处理标量分位数
    q_scalar = q.dim() == 0
    q = q.reshape(-1)  # 确保q是1D
    
    # 展平或重排维度
    if dim is None:
        # 展平整个张量
        sorted_x, _ = input_tensor.flatten().sort()
    elif wrapped_dim == input_tensor.dim() - 1:
        # 当目标维度已是最后一维时直接排序
        sorted_x, _ = input_tensor.sort(dim=wrapped_dim)
    else:
        # 将目标维度移到末尾再排序
        transposed = input_tensor.transpose(wrapped_dim, -1).unsqueeze(-1)
        sorted_x, _ = transposed.sort(dim=-2)
        sorted_x = sorted_x.squeeze(-1)
    
    # ===== 4. 分位数计算核心 =====
    n = sorted_x.shape[-1]
    
    # 处理空输入
    if n == 0:
        result = core.full(out_shape, float('nan'), device=device, dtype=dtype)
        return result
    
    # 计算排名位置 (考虑NaN处理)
    if ignore_nan:
        # 计算非NaN数量
        non_nan_count = (~sorted_x.isnan()).sum(dim=-1, keepdim=True)
        ranks = q * (non_nan_count - 1)
        ranks = core.clamp(ranks, min=0)  # 防止负索引
    else:
        last_index = n - 1
        # 广播处理NaN标记
        nan_mask = sorted_x.isnan().any(dim=-1, keepdim=True)
        # 扩展q和nan_mask到相同形状
        expanded_q = q.view(1, -1).expand(*sorted_x.shape[:-1], q.numel())
        nan_mask = nan_mask.expand_as(expanded_q)
        # 计算基础排名
        ranks = expanded_q * last_index
        # 对包含NaN的行使用最后索引
        ranks = core.where(nan_mask, core.tensor(last_index, device=device), ranks)
    
    # 根据插值模式调整排名
    if interpolation == 'lower':
        ranks = core.floor(ranks)
    elif interpolation == 'higher':
        ranks = core.ceil(ranks)
    elif interpolation == 'nearest':
        ranks = core.round(ranks)
    
    # 确保排名在有效范围内
    ranks = core.clamp(ranks, 0, n - 1)
    
    # 获取下界索引和值
    ranks_below = ranks.to(core.int64)
    values_below = sorted_x.gather(-1, ranks_below)
    
    # ===== 5. 插值处理 =====
    if interpolation in ['linear', 'midpoint']:
        # 计算插值权重
        weights = core.full_like(ranks, 0.5) if interpolation == 'midpoint' else ranks - ranks_below
        
        # 获取上界值
        ranks_above = core.ceil(ranks).to(core.int64)
        values_above = sorted_x.gather(-1, ranks_above)
        
        # 线性插值: result = (1 - weight)*below + weight*above
        values_below = values_below.lerp(values_above, weights)
    
    # ===== 6. 形状调整 =====
    if q_scalar:
        # 标量分位数：移除分位数维度
        values_below = values_below.squeeze(-1)
    else:
        # 多分位数：移动分位数维度到最前面
        values_below = values_below.movedim(-1, 0)
    
    # 恢复原始输出形状
    if values_below.shape != tuple(out_shape):
        values_below = values_below.reshape(out_shape)
    
    return values_below

# nanquantile
def nanquantile(input, q, dim=None, keepdim=False, *, interpolation='linear'):
    return ops.nanquantile(input, q, dim, keepdim)

# std
has_std = hasattr(mindspore.mint, 'std')
def std(input, dim=None, *, correction=1, keepdim=False, **kwargs):
    axis = kwargs.get('axis', None)
    if axis is not None:
        dim = axis
    if use_pyboost() and has_std:
        return mindspore.mint.std(input, dim=dim, correction=correction, keepdim=keepdim)
    if DEVICE_TARGET == 'GPU':
        unbiased = bool(correction)
        if dim is None:
            dim = ()
        if isinstance(dim, int):
            dim = (dim,)
        _std = _get_cache_prim(ops.ReduceStd)(dim, unbiased, keepdim)
        _std.set_device('CPU')
        return _std(input)[0]
    return ops.std(input, dim, correction, keepdim)

# std_mean
has_std_mean = hasattr(mindspore.mint, 'std_mean')
def std_mean(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_std_mean:
        return mindspore.mint.std_mean(input, dim=dim, correction=correction, keepdim=keepdim)
    return std(input, dim, correction=correction, keepdim=keepdim), \
        mean(input, dim, keepdim)

# sum
has_sum = hasattr(mindspore.mint, 'sum')
def sum(input, dim=None, keepdim=False, *, dtype=None, **kwargs):
    keepdims = kwargs.pop('keepdims', None)
    if keepdims is not None:
        keepdim = keepdims
    if 0 in input.shape:
        return mindspore.tensor(0, dtype=dtype)
    if use_pyboost() and has_sum:
        return mindspore.mint.sum(input, dim, keepdim, dtype=dtype)
    return ops.sum(input, dim, keepdim, dtype=dtype)

# unique
has_unique = hasattr(mindspore.mint, 'unique')
def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if use_pyboost() and has_unique:
        return mindspore.mint.unique(input, sorted, return_inverse, return_counts, dim)

    out, inverse = ops.unique(input)
    outs = (out,)
    if return_inverse:
        outs += (inverse,)
    if return_counts:
        counts = (out == input).sum(0, keepdims=True)
        outs += (counts,)
    return outs if len(outs) > 1 else outs[0]

# unique_consecutive
has_unique_consecutive = hasattr(mindspore.mint, 'unique_consecutive')
def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    if use_pyboost() and has_unique_consecutive:
        return mindspore.mint.unique_consecutive(input, return_inverse, return_counts, dim)
    return ops.unique_consecutive(input, return_inverse, return_counts, dim)

# var
has_var = hasattr(mindspore.mint, 'var')
def var(input, dim=None, *, correction=1, keepdim=False, **kwargs):
    correction = int(kwargs.pop('unbiased', correction))
    if use_pyboost and has_var:
        return mindspore.mint.var(input, dim=dim, correction=correction, keepdim=keepdim)
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2)

# var_mean
has_var_mean = hasattr(mindspore.mint, 'var_mean')
def var_mean(input, dim=None, *, correction=1, keepdim=False):
    if use_pyboost and has_var_mean:
        return mindspore.mint.var_mean(input, dim=dim, correction=correction, keepdim=keepdim)
    return pow(std(input, dim, correction=correction, keepdim=keepdim), 2), \
        mean(input, dim, keepdim)

# count_nonzero
has_count_nonzero = hasattr(mindspore.mint, 'count_nonzero')
def count_nonzero(input, dim=None):
    if use_pyboost() and has_count_nonzero:
        return mindspore.mint.count_nonzero(input, dim)
    if dim is None:
        dim = ()
    return ops.count_nonzero(input, dim)

__all__ = ['all', 'amax', 'amin', 'aminmax', 'any', 'argmax', 'argmin', 'count_nonzero', 'logsumexp', 'max', 'mean', 'median', 'min', 'nanmedian', 'nanquantile', 'nansum', 'norm', 'prod', 'quantile', 'std', 'std_mean', 'sum', 'unique', 'unique_consecutive', 'var', 'var_mean']
 