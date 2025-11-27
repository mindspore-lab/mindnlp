import sys
from typing import Optional, Sequence, Tuple, Union, Callable
import functools

import math
import numpy as np
import torch
import torch.distributed._functional_collectives
import mindspore as ms
import mindspore.ops as mops
import mindspore.numpy as mnp

from torch4ms.ops import ops_registry
from torch4ms.ops import op_base, mappings
from torch4ms import minterop
from torch4ms.view import View

# Keys are OpOverload, value is a callable that takes
# Tensor
all_ops = {}


def op(*aten, **kwargs):
    def inner(func):
        for a in aten:
            ops_registry.register_torch_dispatch_op(a, func, **kwargs)
            # 移除了JAX特定的实现部分
        return func

    return inner


@op(
    torch.ops.aten.view_copy,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten.reshape,
)
def _aten_unsafe_view(x, shape):
    return mnp.reshape(x, shape)


@op(torch.ops.aten.add)
@op(torch.ops.aten.add.Scalar)
def _aten_add(x, y, *, alpha=1):
    res = x + y * alpha
    if isinstance(x, float) or isinstance(y, float):
        new_dtype = mappings.t2ms_dtype(torch.get_default_dtype())
        res = res.astype(new_dtype)
    return res


@op(
    torch.ops.aten.copy_, is_jax_function=False, is_view_op=True, needs_env=True
)
def _aten_copy(x, y, memory_format=None, env=None):
    if y.device.type == "cpu":
        y = env.to_mindspore(y)

    if isinstance(x, View):
        x.update(y)
        return x
    return y


@op(torch.ops.aten.sub)
@op(torch.ops.aten.sub.Scalar)
def _aten_sub(x, y, *, alpha=1):
    return x - y * alpha


@op(torch.ops.aten.mul)
@op(torch.ops.aten.mul.Scalar)
def _aten_mul(x, y):
    return x * y


@op(torch.ops.aten.div)
@op(torch.ops.aten.div.Scalar)
def _aten_div(x, y, rounding_mode=None):
    # MindSpore的除法默认行为类似PyTorch的'trunc'模式
    if rounding_mode == 'floor':
        return ops.floor_div(x, y)
    return x / y


@op(torch.ops.aten.pow)
@op(torch.ops.aten.pow.Scalar)
def _aten_pow(x, y):
    return ops.pow(x, y)


@op(torch.ops.aten.neg)
def _aten_neg(x):
    return -x


@op(torch.ops.aten.abs)
def _aten_abs(x):
    return ops.abs(x)


@op(torch.ops.aten.sum)
def _aten_sum(x, dim=None, keepdim=False, dtype=None):
    return ops.reduce_sum(x, axis=dim, keep_dims=keepdim)


@op(torch.ops.aten.mean)
def _aten_mean(x, dim=None, keepdim=False, dtype=None):
    return ops.reduce_mean(x, axis=dim, keep_dims=keepdim)


@op(torch.ops.aten.max)
def _aten_max(x, dim=None, keepdim=False):
    if dim is None:
        return ops.max(x)
    values, indices = ops.max(x, axis=dim, keep_dims=keepdim)
    return values, indices


@op(torch.ops.aten.min)
def _aten_min(x, dim=None, keepdim=False):
    if dim is None:
        return ops.min(x)
    values, indices = ops.min(x, axis=dim, keep_dims=keepdim)
    return values, indices


@op(torch.ops.aten.argmax)
def _aten_argmax(x, dim=None, keepdim=False):
    return ops.argmax(x, axis=dim, keep_dims=keepdim)


@op(torch.ops.aten.argmin)
def _aten_argmin(x, dim=None, keepdim=False):
    return ops.argmin(x, axis=dim, keep_dims=keepdim)


@op(torch.ops.aten.matmul)
def _aten_matmul(x, y):
    return ops.matmul(x, y)


@op(torch.ops.aten.t)
def _aten_t(x):
    return ops.transpose(x, (1, 0))


@op(torch.ops.aten.transpose)
def _aten_transpose(x, dim0, dim1):
    return ops.transpose(x, (dim0, dim1))


@op(torch.ops.aten.permute)
def _aten_permute(x, dims):
    return ops.transpose(x, dims)


@op(torch.ops.aten.reshape)
def _aten_reshape(x, shape):
    return ops.reshape(x, shape)


@op(torch.ops.aten.expand)
def _aten_expand(x, size, implicit=False):
    return ops.broadcast_to(x, size)


@op(torch.ops.aten.squeeze)
def _aten_squeeze(x, dim=None):
    if dim is None:
        return ops.squeeze(x)
    return ops.squeeze(x, dim)


@op(torch.ops.aten.unsqueeze)
def _aten_unsqueeze(x, dim):
    return ops.unsqueeze(x, dim)


@op(torch.ops.aten.cat)
def _aten_cat(tensors, dim=0):
    return ops.concat(tensors, axis=dim)


@op(torch.ops.aten.stack)
def _aten_stack(tensors, dim=0):
    return ops.stack(tensors, axis=dim)


@op(torch.ops.aten.split)
def _aten_split(x, split_size_or_sections, dim=0):
    return ops.split(x, split_size_or_sections, axis=dim)


@op(torch.ops.aten.chunk)
def _aten_chunk(x, chunks, dim=0):
    return ops.chunk(x, chunks, axis=dim)


@op(torch.ops.aten.exp)
def _aten_exp(x):
    return ops.exp(x)


@op(torch.ops.aten.log)
def _aten_log(x):
    return ops.log(x)


@op(torch.ops.aten.log10)
def _aten_log10(x):
    return ops.log10(x)


@op(torch.ops.aten.log2)
def _aten_log2(x):
    return ops.log2(x)


@op(torch.ops.aten.sin)
def _aten_sin(x):
    return ops.sin(x)


@op(torch.ops.aten.cos)
def _aten_cos(x):
    return ops.cos(x)


@op(torch.ops.aten.tan)
def _aten_tan(x):
    return ops.tan(x)


@op(torch.ops.aten.asin)
def _aten_asin(x):
    return ops.asin(x)


@op(torch.ops.aten.acos)
def _aten_acos(x):
    return ops.acos(x)


@op(torch.ops.aten.atan)
def _aten_atan(x):
    return ops.atan(x)


@op(torch.ops.aten.atan2)
def _aten_atan2(x, y):
    return ops.atan2(x, y)


@op(torch.ops.aten.sinh)
def _aten_sinh(x):
    return ops.sinh(x)


@op(torch.ops.aten.cosh)
def _aten_cosh(x):
    return ops.cosh(x)


@op(torch.ops.aten.tanh)
def _aten_tanh(x):
    return ops.tanh(x)


@op(torch.ops.aten.relu)
def _aten_relu(x):
    return ops.relu(x)


@op(torch.ops.aten.leaky_relu)
def _aten_leaky_relu(x, negative_slope=0.01):
    return ops.leaky_relu(x, negative_slope)


@op(torch.ops.aten.sigmoid)
def _aten_sigmoid(x):
    return ops.sigmoid(x)


@op(torch.ops.aten.softmax)
def _aten_softmax(x, dim=-1, dtype=None):
    return ops.softmax(x, axis=dim)


@op(torch.ops.aten.log_softmax)
def _aten_log_softmax(x, dim=-1, dtype=None):
    return ops.log_softmax(x, axis=dim)


@op(torch.ops.aten.layer_norm)
def _aten_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return ops.layer_norm(x, normalized_shape, weight, bias, eps)


@op(torch.ops.aten.batch_norm)
def _aten_batch_norm(x, running_mean, running_var, weight=None, bias=None, 
                    training=False, momentum=0.1, eps=1e-5):
    # MindSpore的BatchNorm接口略有不同，这里做简单实现
    if training:
        # 训练模式下的BatchNorm
        mean = ops.mean(x, (0, 2, 3), keep_dims=True)
        var = ops.var(x, (0, 2, 3), keep_dims=True)
        # 更新running_mean和running_var（简化实现）
        if running_mean is not None:
            running_mean = running_mean * (1 - momentum) + mean.squeeze() * momentum
        if running_var is not None:
            running_var = running_var * (1 - momentum) + var.squeeze() * momentum
    else:
        # 推理模式
        mean = ops.reshape(running_mean, (1, -1, 1, 1))
        var = ops.reshape(running_var, (1, -1, 1, 1))
    
    # 归一化
    x_norm = (x - mean) / ops.sqrt(var + eps)
    
    # 应用缩放和偏移
    if weight is not None:
        x_norm = x_norm * ops.reshape(weight, (1, -1, 1, 1))
    if bias is not None:
        x_norm = x_norm + ops.reshape(bias, (1, -1, 1, 1))
    
    return x_norm


@op(torch.ops.aten.gelu)
def _aten_gelu(x, approximate='none'):
    if approximate == 'tanh':
        return ops.gelu(x, approximate=True)
    return ops.gelu(x, approximate=False)


@op(torch.ops.aten.silu)
def _aten_silu(x):
    return ops.silu(x)


@op(torch.ops.aten.dropout)
def _aten_dropout(x, p=0.5, training=True):
    if training:
        return ops.dropout(x, p)
    return x


@op(torch.ops.aten.flatten)
def _aten_flatten(x, start_dim=0, end_dim=-1):
    return ops.flatten(x, start_dim, end_dim)


@op(torch.ops.aten.eye)
def _aten_eye(n, m=None, *, dtype=None, layout=None, device=None, requires_grad=False):
    if m is None:
        m = n
    return ops.eye(n, m, dtype=dtype or ms.float32)


@op(torch.ops.aten.zeros_like)
def _aten_zeros_like(x, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return ops.zeros_like(x, dtype=dtype)


@op(torch.ops.aten.ones_like)
def _aten_ones_like(x, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    return ops.ones_like(x, dtype=dtype)


@op(torch.ops.aten.fill_)
def _aten_fill_(x, value):
    return ops.fill(x, value)


@op(torch.ops.aten.copy_)
def _aten_copy_(x, y, memory_format=None):
    return ops.assign(x, y)


@op(torch.ops.aten.eq)
def _aten_eq(x, y):
    return ops.equal(x, y)


@op(torch.ops.aten.ne)
def _aten_ne(x, y):
    return ops.not_equal(x, y)


@op(torch.ops.aten.lt)
def _aten_lt(x, y):
    return ops.less(x, y)


@op(torch.ops.aten.le)
def _aten_le(x, y):
    return ops.less_equal(x, y)


@op(torch.ops.aten.gt)
def _aten_gt(x, y):
    return ops.greater(x, y)


@op(torch.ops.aten.ge)
def _aten_ge(x, y):
    return ops.greater_equal(x, y)


@op(torch.ops.aten.clamp)
def _aten_clamp(x, min=None, max=None):
    return ops.clip_by_value(x, min, max)


@op(torch.ops.aten.erf)
def _aten_erf(x):
    return ops.erf(x)


@op(torch.ops.aten.erfc)
def _aten_erfc(x):
    return ops.erfc(x)


@op(torch.ops.aten.sqrt)
def _aten_sqrt(x):
    return ops.sqrt(x)


@op(torch.ops.aten.rsqrt)
def _aten_rsqrt(x):
    return ops.rsqrt(x)


@op(torch.ops.aten.cumsum)
def _aten_cumsum(x, dim, dtype=None):
    return ops.cumsum(x, axis=dim)


@op(torch.ops.aten.cumprod)
def _aten_cumprod(x, dim, dtype=None):
    return ops.cumprod(x, axis=dim)


# 暂时注释掉unique操作符，因为当前PyTorch版本不支持torch.ops.aten.unique
# @op(torch.ops.aten.unique)
def _aten_unique(x, sorted=True, return_inverse=False, return_counts=False, dim=None):
    unique_values, indices = ops.unique(x, return_idx=True, dim=dim)
    result = [unique_values]
    
    if return_inverse:
        # MindSpore不直接支持return_inverse，这里做简化处理
        # 实际应用中可能需要更复杂的实现
        result.append(indices)
    
    if return_counts:
        # MindSpore不直接支持return_counts，这里做简化处理
        counts = ops.ones_like(unique_values, dtype=ms.int32)
        result.append(counts)
    
    return result[0] if len(result) == 1 else tuple(result)


@op(torch.ops.aten.where)
def _aten_where(condition, x, y):
    return ops.where(condition, x, y)


@op(torch.ops.aten.masked_select)
def _aten_masked_select(x, mask):
    return ops.masked_select(x, mask)


@op(torch.ops.aten.masked_fill)
def _aten_masked_fill(x, mask, value):
    return ops.masked_fill(x, mask, value)


@op(torch.ops.aten.index_select)
def _aten_index_select(x, dim, index):
    return ops.gather(x, dim, index)


@op(torch.ops.aten.scatter)
def _aten_scatter(input, dim, index, src):
    return ops.scatter(input, dim, index, src)


@op(torch.ops.aten.scatter_add)
def _aten_scatter_add(input, dim, index, src):
    return ops.scatter_add(input, dim, index, src)


@op(torch.ops.aten.narrow)
def _aten_narrow(x, dim, start, length):
    return ops.slice(x, (start,), (length,), axis=dim)


@op(torch.ops.aten.select)
def _aten_select(x, dim, index):
    return ops.take(x, index, dim=dim)


@op(torch.ops.aten.meshgrid)
def _aten_meshgrid(*tensors, indexing='ij'):
    return ops.meshgrid(*tensors, indexing=indexing)


@op(torch.ops.aten.unbind)
def _aten_unbind(x, dim=0):
    return ops.unbind(x, axis=dim)


@op(torch.ops.aten.bmm)
def _aten_bmm(x, y):
    return ops.bmm(x, y)


@op(torch.ops.aten.einsum)
def _aten_einsum(equation, *operands):
    return ops.einsum(equation, *operands)


@op(torch.ops.aten.linalg_norm)
def _aten_linalg_norm(x, ord=None, dim=None, keepdim=False, dtype=None):
    if ord is None:
        ord = 'fro'  # 默认使用Frobenius范数
    return ops.norm(x, ord=ord, axis=dim, keep_dims=keepdim)


@op(torch.ops.aten.linalg_inv)
def _aten_linalg_inv(x):
    return ops.inv(x)


@op(torch.ops.aten.linalg_svd)
def _aten_linalg_svd(x, full_matrices=True, compute_uv=True):
    u, s, v = ops.svd(x, full_matrices=full_matrices)
    if compute_uv:
        return u, s, v
    return s