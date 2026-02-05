import sys
from typing import Optional, Sequence, Tuple, Union, Callable
import functools

import math
import numpy as np
import torch
import torch.distributed._functional_collectives
import mindspore as ms
import mindspore.ops as ops
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
    torch.ops.aten.copy_, is_mindspore_function=False, is_view_op=True, needs_env=True
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


@op(torch.ops.aten.squeeze, is_mindspore_function=True)
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
    # MindSpore的BatchNorm接口略有不同，这里使用MindSpore的reduce操作实现
    if training:
        # 训练模式下的BatchNorm - 计算当前batch的均值和方差
        # 使用MindSpore的reduce_mean计算均值
        mean = ops.reduce_mean(x, axis=(0, 2, 3), keep_dims=True)
        # 计算方差：E[(x - E[x])^2]
        centered = x - mean
        var = ops.reduce_mean(centered * centered, axis=(0, 2, 3), keep_dims=True)
        # 更新running_mean和running_var（简化实现）
        if running_mean is not None:
            running_mean = running_mean * (1 - momentum) + mean.squeeze() * momentum
        if running_var is not None:
            running_var = running_var * (1 - momentum) + var.squeeze() * momentum
    else:
        # 推理模式 - 使用running statistics
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

@op(
    torch.ops.aten.convolution,
    torch.ops.aten.convolution.out,
    is_mindspore_function=True
)
def _aten_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, *, out=None):
    """
    实现PyTorch的aten.convolution算子，这是所有卷积操作的底层实现
    参考 JAX/XLA 的实现逻辑，使用 MindSpore API
    
    注意：函数签名必须与 PyTorch 的 aten::convolution 完全匹配：
    (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
    
    注意：由于使用了is_mindspore_function=True，input、weight、bias等参数
    应该已经通过env.t2ms_iso转换成了MindSpore Tensor，这里不需要再次转换
    
    Args:
        input: 输入张量（MindSpore Tensor）
        weight: 权重张量（MindSpore Tensor）
        bias: 偏置张量（MindSpore Tensor 或 None）
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        transposed: 是否为转置卷积
        output_padding: 输出填充（仅用于转置卷积）
        groups: 组数
    
    Returns:
        卷积结果（MindSpore Tensor）
    """
    # 转置卷积目前不支持，抛出异常
    if transposed:
        raise NotImplementedError("Transposed convolution (deconvolution) is not yet implemented")
    
    # 注意：由于使用了is_mindspore_function=True，input、weight、bias等参数
    # 应该已经通过env.t2ms_iso转换成了MindSpore Tensor，但为了安全起见，我们再次确保它们是 MindSpore Tensor
    # 如果参数仍然是 torch4ms.Tensor，提取其底层的 MindSpore Tensor
    import mindspore as ms
    import torch4ms.tensor as torch4ms_tensor
    
    # 转换 input
    if isinstance(input, ms.Tensor):
        # 已经是 MindSpore Tensor，直接使用
        pass
    elif isinstance(input, torch4ms_tensor.Tensor):
        # torch4ms.Tensor，提取底层 MindSpore Tensor
        input = input._elem if hasattr(input, '_elem') else input.mindspore()
    elif hasattr(input, '_elem') and isinstance(input._elem, ms.Tensor):
        # 可能是 View 或其他包装类型
        input = input._elem
    elif hasattr(input, 'mindspore'):
        # 有 mindspore() 方法
        input = input.mindspore()
    else:
        # 尝试直接转换为 MindSpore Tensor
        try:
            input = ms.Tensor(input)
        except:
            raise TypeError(f"Unable to convert input to MindSpore Tensor. Type: {type(input)}")
    
    # 转换 weight
    if isinstance(weight, ms.Tensor):
        pass
    elif isinstance(weight, torch4ms_tensor.Tensor):
        weight = weight._elem if hasattr(weight, '_elem') else weight.mindspore()
    elif hasattr(weight, '_elem') and isinstance(weight._elem, ms.Tensor):
        weight = weight._elem
    elif hasattr(weight, 'mindspore'):
        weight = weight.mindspore()
    else:
        try:
            weight = ms.Tensor(weight)
        except:
            raise TypeError(f"Unable to convert weight to MindSpore Tensor. Type: {type(weight)}")
    
    # 转换 bias（如果提供）
    if bias is not None:
        if isinstance(bias, ms.Tensor):
            pass
        elif isinstance(bias, torch4ms_tensor.Tensor):
            bias = bias._elem if hasattr(bias, '_elem') else bias.mindspore()
        elif hasattr(bias, '_elem') and isinstance(bias._elem, ms.Tensor):
            bias = bias._elem
        elif hasattr(bias, 'mindspore'):
            bias = bias.mindspore()
        else:
            try:
                bias = ms.Tensor(bias)
            except:
                raise TypeError(f"Unable to convert bias to MindSpore Tensor. Type: {type(bias)}")
    
    # 计算空间维度数量：weight的维度减1（去掉out_channel和in_channel）
    num_shape_dim = weight.ndim - 2
    
    # 确保参数格式正确 - 参考参考代码的逻辑
    if isinstance(stride, int):
        stride = (stride,) * num_shape_dim
    elif isinstance(stride, list):
        stride = tuple(stride)
    elif not isinstance(stride, tuple):
        stride = tuple(stride) if hasattr(stride, '__iter__') else (stride,) * num_shape_dim
    
    # 确保stride长度正确
    if len(stride) == 1 and num_shape_dim > 1:
        stride = stride * num_shape_dim
    
    if isinstance(padding, int):
        padding = (padding,) * num_shape_dim
    elif isinstance(padding, list):
        padding = tuple(padding)
    elif not isinstance(padding, tuple):
        padding = tuple(padding) if hasattr(padding, '__iter__') else (padding,) * num_shape_dim
    
    # 确保padding长度正确
    if len(padding) == 1 and num_shape_dim > 1:
        padding = padding * num_shape_dim
    
    # 处理dilation参数
    if isinstance(dilation, bool):
        dilation = 1
    if isinstance(dilation, int):
        dilation = (dilation,) * num_shape_dim
    elif isinstance(dilation, list):
        dilation = tuple(dilation)
    elif not isinstance(dilation, tuple):
        dilation = tuple(dilation) if hasattr(dilation, '__iter__') else (dilation,) * num_shape_dim
    
    # 确保dilation长度正确
    if len(dilation) == 1 and num_shape_dim > 1:
        dilation = dilation * num_shape_dim
    
    # 处理groups参数
    if isinstance(groups, bool):
        groups = 1 if groups else 0
    groups = int(groups)
    
    # 处理output_padding（用于转置卷积，目前不使用）
    if isinstance(output_padding, int):
        output_padding = (output_padding,) * num_shape_dim
    elif isinstance(output_padding, list):
        output_padding = tuple(output_padding)
    elif not output_padding:
        output_padding = (0,) * num_shape_dim
    
    # 根据空间维度数量选择对应的卷积操作
    # 使用MindSpore的nn层API（参考 torchax 的实现方式，统一使用nn层以确保兼容性）
    import mindspore.nn as nn
    
    if num_shape_dim == 1:
        # 1D卷积 - 通过转换为2D卷积实现（参考 MindSpore 内部实现）
        # 输入: [N, C_in, L] -> [N, C_in, 1, L]
        # 权重: [C_out, C_in, K] -> [C_out, C_in, 1, K]
        # 使用 Conv2D 进行卷积
        # 输出: [N, C_out, 1, L_out] -> [N, C_out, L_out]
        
        # 扩展输入维度：在位置2插入维度1
        input_2d = ops.expand_dims(input, 2)  # [N, C_in, L] -> [N, C_in, 1, L]
        
        # 扩展权重维度：在位置2插入维度1
        weight_2d = ops.expand_dims(weight, 2)  # [C_out, C_in, K] -> [C_out, C_in, 1, K]
        
        # 准备padding：对于2D，padding格式为(上,下,左,右)，1D只需要左右padding
        pad_val = int(padding[0])
        pad_2d = (0, 0, pad_val, pad_val)  # (上=0, 下=0, 左=pad, 右=pad)
        
        # 使用Conv2D进行卷积
        conv2d_op = nn.Conv2d(
            in_channels=int(weight.shape[1] * groups),
            out_channels=int(weight.shape[0]),
            kernel_size=(1, int(weight.shape[2])),  # (height=1, width=kernel_size)
            stride=(1, int(stride[0])),  # (height_stride=1, width_stride=stride)
            pad_mode='pad',
            padding=pad_2d,
            dilation=(1, int(dilation[0])),  # (height_dilation=1, width_dilation=dilation)
            group=groups,
            has_bias=(bias is not None)
        )
        
        # 设置权重和偏置
        conv2d_op.weight.set_data(weight_2d)
        if bias is not None:
            conv2d_op.bias.set_data(bias)
        
        # 执行2D卷积
        res_2d = conv2d_op(input_2d)  # [N, C_out, 1, L_out]
        
        # 压缩输出维度：移除位置2的维度
        res = ops.squeeze(res_2d, 2)  # [N, C_out, 1, L_out] -> [N, C_out, L_out]
    
    elif num_shape_dim == 2:
        # 2D卷积 - 使用 nn.Conv2d
        # 将padding转换为元组格式（MindSpore的Conv2d接受int或tuple）
        if isinstance(padding, int):
            pad_val = padding
        elif len(padding) == 1:
            pad_val = padding[0]
        elif len(padding) == 2:
            # 对于2D，如果提供2个值，使用第一个（MindSpore的padding参数）
            pad_val = padding[0] if padding[0] == padding[1] else padding
        else:
            pad_val = tuple(padding[:2])
        
        # 创建临时Conv2d层
        conv2d_layer = nn.Conv2d(
            in_channels=int(weight.shape[1] * groups),
            out_channels=int(weight.shape[0]),
            kernel_size=tuple(weight.shape[2:]),
            stride=tuple(stride),
            pad_mode='pad',
            padding=pad_val,
            dilation=tuple(dilation),
            group=groups,
            has_bias=(bias is not None)
        )
        
        # 设置权重和偏置
        conv2d_layer.weight.set_data(weight)
        if bias is not None:
            conv2d_layer.bias.set_data(bias)
        
        res = conv2d_layer(input)
    
    elif num_shape_dim == 3:
        # 3D卷积 - 使用 nn.Conv3d
        # 将padding转换为元组格式
        if isinstance(padding, int):
            pad_val = padding
        elif len(padding) == 1:
            pad_val = padding[0]
        elif len(padding) == 3:
            # 对于3D，如果提供3个值且都相同，使用单个值
            if padding[0] == padding[1] == padding[2]:
                pad_val = padding[0]
            else:
                pad_val = tuple(padding[:3])
        else:
            pad_val = tuple(padding[:3])
        
        # 创建临时Conv3d层
        conv3d_layer = nn.Conv3d(
            in_channels=int(weight.shape[1] * groups),
            out_channels=int(weight.shape[0]),
            kernel_size=tuple(weight.shape[2:]),
            stride=tuple(stride),
            pad_mode='pad',
            padding=pad_val,
            dilation=tuple(dilation),
            group=groups,
            has_bias=(bias is not None)
        )
        
        # 设置权重和偏置
        conv3d_layer.weight.set_data(weight)
        if bias is not None:
            conv3d_layer.bias.set_data(bias)
        
        res = conv3d_layer(input)
    
    else:
        raise NotImplementedError(f"Convolution not supported for {num_shape_dim}D spatial dimensions (weight.ndim={weight.ndim})")
    
    # 如果提供了 out 参数，将结果写入 out
    if out is not None:
        # 将结果复制到 out 参数中
        if hasattr(out, '_elem'):
            # torch4ms.Tensor - 直接更新 _elem
            out._elem = res
            return out
        elif isinstance(out, ms.Tensor):
            # MindSpore Tensor - 使用 assign_value 或直接赋值
            try:
                out.assign_value(res)
                return out
            except:
                # 如果 assign_value 不可用，尝试直接赋值
                out = res
                return out
        else:
            # 其他类型，尝试直接赋值
            return res
    
    return res

@op(torch.ops.aten.randn, needs_env=True)
def _aten_randn(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    env=None,
):
    # 使用MindSpore的随机数生成替代PyTorch的randn
    # *size 意味着size是一个元组，包含所有尺寸参数
    # 处理size参数：确保它是一个包含整数的元组
    if not size:
        shape = (1,)
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        # 如果传入的是单个列表或元组，如 torch.randn((2, 3))
        shape = tuple(int(x) for x in size[0])
    else:
        # 如果传入的是多个参数，如 torch.randn(2, 3)
        shape = tuple(int(x) for x in size)
    # MindSpore的normal操作生成标准正态分布的随机数
    tensor = ops.normal(shape=shape, mean=0.0, stddev=1.0)
    
    # 转换dtype（如果需要）
    if dtype is not None:
        ms_dtype = mappings.t2ms_dtype(dtype) if isinstance(dtype, torch.dtype) else dtype
        tensor = tensor.astype(ms_dtype)
    
    # 如果提供了env，使用环境进行转换
    if env is not None:
        return env.ms2t_iso(tensor)
    
    # 否则直接返回MindSpore Tensor
    return tensor


# ==================== 参考 torchax 添加的算子实现 ====================

@op(torch.ops.aten.clone)
def _aten_clone(x, memory_format=None):
    """克隆张量"""
    return mnp.copy(x)


@op(torch.ops.aten.trunc)
def _aten_trunc(x):
    """截断函数，将每个元素向零取整"""
    return mnp.trunc(x).astype(x.dtype)


@op(torch.ops.aten.index_copy)
def _aten_index_copy(x, dim, indexes, source):
    """在指定维度上，使用索引从source复制值到x"""
    if x.ndim == 0:
        return source
    if x.ndim == 1:
        source = mnp.squeeze(source)
    if dim < 0:
        dim += x.ndim
    
    # 构建索引
    indices = [mnp.arange(x.shape[i]) if i != dim else indexes for i in range(x.ndim)]
    indices = mnp.meshgrid(*indices, indexing='ij')
    result = mnp.copy(x)
    result[tuple(indices)] = source
    return result


@op(torch.ops.aten.mm)
def _aten_mm(x, y):
    """矩阵乘法"""
    return ops.matmul(x, y)


@op(torch.ops.aten.detach)
@op(torch.ops.aten.positive)
def _aten_detach(x):
    """分离梯度，在MindSpore中返回原张量"""
    return x


@op(torch.ops.aten.log1p)
def _aten_log1p(x):
    """log(1 + x)"""
    return mnp.log1p(x)


@op(torch.ops.aten.expm1)
def _aten_expm1(x):
    """exp(x) - 1"""
    return mnp.expm1(x)


@op(torch.ops.aten.floor)
def _aten_floor(x):
    """向下取整"""
    return mnp.floor(x).astype(x.dtype)


@op(torch.ops.aten.ceil)
def _aten_ceil(x):
    """向上取整"""
    return mnp.ceil(x).astype(x.dtype)


@op(torch.ops.aten.round)
def _aten_round(x, decimals=0):
    """四舍五入"""
    return mnp.round(x, decimals)


@op(torch.ops.aten.sign)
def _aten_sign(x):
    """符号函数"""
    return mnp.sign(x)


@op(torch.ops.aten.reciprocal)
def _aten_reciprocal(x):
    """倒数 1/x"""
    return 1.0 / x


@op(torch.ops.aten.prod)
def _aten_prod(x, dim=None, keepdim=False, dtype=None):
    """乘积"""
    if dtype is not None:
        x = x.astype(mappings.t2ms_dtype(dtype))
    # 使用 MindSpore 的 ReduceProd 类（类似 ReduceMin/ReduceMax）
    # 注意：MindSpore 可能没有 ops.reduce_prod 函数形式，所以直接使用类形式
    if dim is None:
        reduce_prod_op = ops.ReduceProd(keep_dims=keepdim)
        return reduce_prod_op(x)
    reduce_prod_op = ops.ReduceProd(keep_dims=keepdim)
    return reduce_prod_op(x, axis=dim)


@op(torch.ops.aten.var)
def _aten_var(x, dim=None, unbiased=True, keepdim=False, correction=None):
    """方差"""
    # correction 参数对应 ddof (delta degrees of freedom)
    ddof = 1 if (unbiased or correction == 1) else (correction or 0)
    return mnp.var(x, axis=dim, ddof=ddof, keepdims=keepdim)


@op(torch.ops.aten.std)
def _aten_std(x, dim=None, unbiased=True, keepdim=False, correction=None):
    """标准差"""
    ddof = 1 if (unbiased or correction == 1) else (correction or 0)
    return mnp.std(x, axis=dim, ddof=ddof, keepdims=keepdim)


@op(torch.ops.aten.all)
def _aten_all(x, dim=None, keepdim=False):
    """所有元素为True"""
    # 使用 MindSpore 的 ReduceAll 类（类似 ReduceProd/ReduceMin/ReduceMax）
    if dim is None:
        reduce_all_op = ops.ReduceAll(keep_dims=keepdim)
        return reduce_all_op(x)
    reduce_all_op = ops.ReduceAll(keep_dims=keepdim)
    return reduce_all_op(x, axis=dim)


@op(torch.ops.aten.any)
def _aten_any(x, dim=None, keepdim=False):
    """任意元素为True"""
    # 使用 MindSpore 的 ReduceAny 类（类似 ReduceAll）
    if dim is None:
        reduce_any_op = ops.ReduceAny(keep_dims=keepdim)
        return reduce_any_op(x)
    reduce_any_op = ops.ReduceAny(keep_dims=keepdim)
    return reduce_any_op(x, axis=dim)


@op(torch.ops.aten.logical_and)
def _aten_logical_and(x, y):
    """逻辑与"""
    return mnp.logical_and(x, y)


@op(torch.ops.aten.logical_or)
def _aten_logical_or(x, y):
    """逻辑或"""
    return mnp.logical_or(x, y)


@op(torch.ops.aten.logical_not)
def _aten_logical_not(x):
    """逻辑非"""
    return mnp.logical_not(x)


@op(torch.ops.aten.logical_xor)
def _aten_logical_xor(x, y):
    """逻辑异或"""
    return mnp.logical_xor(x, y)


@op(torch.ops.aten.isnan)
def _aten_isnan(x):
    """检查是否为NaN"""
    return mnp.isnan(x)


@op(torch.ops.aten.isfinite)
def _aten_isfinite(x):
    """检查是否为有限数"""
    return mnp.isfinite(x)


@op(torch.ops.aten.isinf)
def _aten_isinf(x):
    """检查是否为无穷大"""
    return mnp.isinf(x)


@op(torch.ops.aten.diag)
def _aten_diag(x, diagonal=0):
    """提取或构造对角线矩阵"""
    return mnp.diag(x, k=diagonal)


@op(torch.ops.aten.diagflat)
def _aten_diagflat(x, offset=0):
    """从扁平输入创建对角矩阵"""
    return mnp.diagflat(x, k=offset)


@op(torch.ops.aten.triu)
def _aten_triu(x, diagonal=0):
    """上三角矩阵"""
    return mnp.triu(x, k=diagonal)


@op(torch.ops.aten.tril)
def _aten_tril(x, diagonal=0):
    """下三角矩阵"""
    return mnp.tril(x, k=diagonal)


@op(torch.ops.aten.diagonal)
def _aten_diagonal(x, offset=0, dim1=0, dim2=1):
    """提取对角线"""
    return mnp.diagonal(x, offset=offset, axis1=dim1, axis2=dim2)


@op(torch.ops.aten.repeat)
def _aten_repeat(x, repeats):
    """重复张量"""
    if isinstance(repeats, int):
        repeats = (repeats,) * x.ndim
    return mnp.tile(x, repeats)


@op(torch.ops.aten.take)
def _aten_take(x, index):
    """从扁平化张量中取索引元素"""
    flat_x = mnp.ravel(x)
    return flat_x[index]


@op(torch.ops.aten.index_add)
def _aten_index_add(x, dim, index, source):
    """在指定维度上，使用索引将source加到x"""
    if x.ndim == 0:
        return x + source
    if x.ndim == 1:
        source = mnp.squeeze(source)
    if dim < 0:
        dim += x.ndim
    
    # 构建索引并执行索引加法
    indices = [mnp.arange(x.shape[i]) if i != dim else index for i in range(x.ndim)]
    indices = mnp.meshgrid(*indices, indexing='ij')
    result = mnp.copy(x)
    result[tuple(indices)] = result[tuple(indices)] + source
    return result


@op(torch.ops.aten.gather)
def _aten_gather(x, dim, index):
    """从输入张量中收集值"""
    # MindSpore 的 gather 函数签名略有不同
    return ops.gather_elements(x, dim, index)


@op(torch.ops.aten.embedding)
def _aten_embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    """嵌入层"""
    # 使用 MindSpore 的 embedding lookup
    # 注意：MindSpore 的 embedding 可能需要不同的实现方式
    return ops.gather(weight, indices, axis=0)


@op(torch.ops.aten.embedding_renorm_)
def _aten_embedding_renorm_(weight, indices, max_norm, norm_type):
    """重新归一化嵌入权重"""
    unique_indices = mnp.unique(indices)
    # 获取对应的嵌入向量
    embedded = ops.gather(weight, unique_indices, axis=0)
    # 计算范数
    norm = mnp.linalg.norm(embedded, ord=norm_type, axis=1)
    # 找到需要缩放的索引
    mask = norm > max_norm
    scale = max_norm / (norm + 1e-7)
    # 应用缩放
    weight = weight.at[unique_indices].multiply(mnp.where(mask, scale, 1.0)[:, None])
    return weight


@op(torch.ops.aten.narrow)
@op(torch.ops.aten.narrow_copy)
def _aten_narrow(x, dim, start, length):
    """窄化张量（切片）"""
    indices = [slice(None)] * x.ndim
    indices[dim] = slice(start, start + length)
    return x[tuple(indices)]


@op(torch.ops.aten.slice)
def _aten_slice(x, dim=0, start=None, end=None, step=1):
    """切片操作"""
    if dim < 0:
        dim += x.ndim
    if end == sys.maxsize or end is None:
        end = x.shape[dim]
    indices = [slice(None)] * x.ndim
    indices[dim] = slice(start, end, step)
    return x[tuple(indices)]


@op(torch.ops.aten.select)
def _aten_select(x, dim, index):
    """选择指定维度的单个索引"""
    indices = [slice(None)] * x.ndim
    indices[dim] = index
    return x[tuple(indices)]


@op(torch.ops.aten.index_select)
def _aten_index_select(x, dim, index):
    """根据索引选择元素"""
    if x.shape == ():
        return x
    return ops.gather(x, index, axis=dim)


@op(torch.ops.aten.masked_fill)
def _aten_masked_fill(x, mask, value):
    """使用mask填充值"""
    return mnp.where(mask, value, x)


@op(torch.ops.aten.masked_scatter)
def _aten_masked_scatter(x, mask, source):
    """使用mask从source散布值"""
    mask_flat = mnp.ravel(mask)
    source_flat = mnp.ravel(source)
    x_flat = mnp.ravel(x)
    true_indices = mnp.where(mask_flat)[0]
    x_flat = x_flat.at[true_indices[:len(source_flat)]].set(source_flat[:len(true_indices)])
    return x_flat.reshape(x.shape)


@op(torch.ops.aten.scatter)
def _aten_scatter(x, dim, index, src):
    """散布操作"""
    # MindSpore 的 scatter 实现
    return ops.scatter(x, dim, index, src)


@op(torch.ops.aten.scatter_add)
def _aten_scatter_add(x, dim, index, src):
    """散布加法"""
    return ops.scatter_add(x, dim, index, src)


@op(torch.ops.aten.topk)
def _aten_topk(x, k, dim=None, largest=True, sorted=True):
    """Top-K操作"""
    if dim is None:
        dim = -1
    
    # 尝试使用 ops.topk（如果支持 dim 参数）
    if hasattr(ops, 'topk'):
        try:
            values, indices = ops.topk(x, k, dim=dim, largest=largest, sorted=sorted)
            return values, indices
        except (TypeError, AttributeError):
            pass
    
    # 回退到 ops.top_k（可能不支持 dim 参数）
    # 如果 dim 不是最后一个维度，需要先转置
    original_shape = x.shape
    if dim != -1 and dim != len(original_shape) - 1:
        # 将目标维度移到最后
        perm = list(range(len(original_shape)))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        x_transposed = ops.transpose(x, perm)
        if largest:
            values, indices = ops.top_k(x_transposed, k)
        else:
            values, indices = ops.top_k(-x_transposed, k)
            values = -values
        # 如果 sorted=False，需要反转结果（top_k 默认返回降序）
        if not sorted and largest:
            values = mnp.flip(values, axis=-1)
            indices = mnp.flip(indices, axis=-1)
        # 转置回来
        values = ops.transpose(values, perm)
        indices = ops.transpose(indices, perm)
        return values, indices
    else:
        # dim 是最后一个维度，直接使用
        if largest:
            values, indices = ops.top_k(x, k)
        else:
            values, indices = ops.top_k(-x, k)
            values = -values
        # 如果 sorted=False，需要反转结果
        if not sorted and largest:
            values = mnp.flip(values, axis=-1)
            indices = mnp.flip(indices, axis=-1)
        return values, indices


@op(torch.ops.aten.sort)
def _aten_sort(x, dim=-1, descending=False, stable=False):
    """排序"""
    if x.shape == ():
        return (x, mnp.array(0, dtype=mnp.int64))
    # 使用 MindSpore 的 ops.sort，它返回 (values, indices)
    values, indices = ops.sort(x, axis=dim, descending=descending)
    return values, indices.astype(mnp.int64)


@op(torch.ops.aten.argsort)
def _aten_argsort(x, dim=-1, descending=False, stable=False):
    """返回排序索引"""
    # 使用 MindSpore 的 ops.argsort
    indices = ops.argsort(x, axis=dim, descending=descending)
    return indices.astype(mnp.int64)


@op(torch.ops.aten.amax)
def _aten_amax(x, dim=None, keepdim=False):
    """沿维度求最大值"""
    # MindSpore 的 reduce_max 使用 keepdim 而不是 keep_dims
    if dim is None:
        return ops.reduce_max(x)
    reduce_max_op = ops.ReduceMax(keep_dims=keepdim)
    return reduce_max_op(x, axis=dim)


@op(torch.ops.aten.amin)
def _aten_amin(x, dim=None, keepdim=False):
    """沿维度求最小值"""
    # MindSpore 的 reduce_min 需要使用 ReduceMin 类
    if dim is None:
        return ops.reduce_min(x)
    reduce_min_op = ops.ReduceMin(keep_dims=keepdim)
    return reduce_min_op(x, axis=dim)


@op(torch.ops.aten.maximum)
def _aten_maximum(x, y):
    """逐元素最大值"""
    return mnp.maximum(x, y)


@op(torch.ops.aten.minimum)
def _aten_minimum(x, y):
    """逐元素最小值"""
    return mnp.minimum(x, y)


@op(torch.ops.aten.fmod)
def _aten_fmod(x, y):
    """逐元素浮点取模"""
    return mnp.fmod(x, y)


@op(torch.ops.aten.remainder)
def _aten_remainder(x, y):
    """逐元素取余"""
    return mnp.remainder(x, y)


@op(torch.ops.aten.flip)
def _aten_flip(x, dims):
    """翻转张量"""
    if dims is None:
        dims = tuple(range(x.ndim))
    return mnp.flip(x, axis=dims)


@op(torch.ops.aten.roll)
def _aten_roll(x, shifts, dims=None):
    """滚动张量"""
    return mnp.roll(x, shifts, axis=dims)


@op(torch.ops.aten.broadcast_to)
def _aten_broadcast_to(x, shape):
    """广播到指定形状"""
    return mnp.broadcast_to(x, shape)


@op(torch.ops.aten.contiguous)
def _aten_contiguous(x, memory_format=None):
    """返回连续张量（MindSpore中通常已经是连续的）"""
    return x


@op(torch.ops.aten.zero_)
def _aten_zero_(x):
    """原地置零"""
    # 使用 zeros_like 创建全零张量，保持相同的形状和类型
    return ops.zeros_like(x)


@op(torch.ops.aten.fill_)
def _aten_fill_(x, value):
    """原地填充值"""
    # 使用 ones_like 然后乘以 value，或者直接使用广播
    # 创建一个与 x 形状相同、值为 value 的张量
    return ops.ones_like(x) * value


@op(torch.ops.aten.logaddexp)
def _aten_logaddexp(x, y):
    """log(exp(x) + exp(y))"""
    return mnp.logaddexp(x, y)


@op(torch.ops.aten.logaddexp2)
def _aten_logaddexp2(x, y):
    """log2(2^x + 2^y)"""
    return mnp.logaddexp2(x, y)


@op(torch.ops.aten.hypot)
def _aten_hypot(x, y):
    """直角三角形的斜边长度 sqrt(x^2 + y^2)"""
    return mnp.hypot(x, y)


@op(torch.ops.aten.nextafter)
def _aten_nextafter(x, y):
    """返回x朝向y的下一个浮点数"""
    # MindSpore 可能不直接支持 nextafter，使用近似实现
    return x  # 简化实现


@op(torch.ops.aten.lerp)
def _aten_lerp(start, end, weight):
    """线性插值"""
    return start + weight * (end - start)


def _broadcast_shape(shape1, shape2):
    """计算两个形状的广播后形状（PyTorch风格）"""
    # 从右到左对齐维度
    max_ndim = max(len(shape1), len(shape2))
    # 左填充维度为1
    shape1_padded = [1] * (max_ndim - len(shape1)) + list(shape1)
    shape2_padded = [1] * (max_ndim - len(shape2)) + list(shape2)
    
    # 计算广播后的形状
    broadcast_shape = []
    for s1, s2 in zip(shape1_padded, shape2_padded):
        if s1 == s2:
            broadcast_shape.append(s1)
        elif s1 == 1:
            broadcast_shape.append(s2)
        elif s2 == 1:
            broadcast_shape.append(s1)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
    return tuple(broadcast_shape)


def _broadcast_tensors(*tensors):
    """将多个张量广播到相同的形状"""
    if len(tensors) == 0:
        return []
    if len(tensors) == 1:
        return tensors
    
    # 计算所有张量的广播后形状
    shapes = [tuple(t.shape) for t in tensors]
    # 从右到左对齐，找到最大维度数
    max_ndim = max(len(s) for s in shapes)
    
    # 左填充维度为1
    padded_shapes = []
    for s in shapes:
        padded = [1] * (max_ndim - len(s)) + list(s)
        padded_shapes.append(padded)
    
    # 计算广播后的形状
    broadcast_shape = []
    for i in range(max_ndim):
        dims = [s[i] for s in padded_shapes]
        # 找到非1的维度，如果都相同或其中一个为1，则可以广播
        non_one_dims = [d for d in dims if d != 1]
        if len(non_one_dims) == 0:
            broadcast_shape.append(1)
        elif len(set(non_one_dims)) == 1:
            broadcast_shape.append(non_one_dims[0])
        else:
            # 检查是否有冲突
            if len(set(non_one_dims)) > 1:
                raise ValueError(f"Cannot broadcast shapes {shapes}")
            broadcast_shape.append(non_one_dims[0])
    
    target_shape = tuple(broadcast_shape)
    
    # 广播所有张量
    broadcasted = []
    for t in tensors:
        if tuple(t.shape) == target_shape:
            broadcasted.append(t)
        else:
            broadcasted.append(mnp.broadcast_to(t, target_shape))
    
    return broadcasted


@op(torch.ops.aten.addcmul)
def _aten_addcmul(input, tensor1, tensor2, *, value=1):
    """执行 tensor1 * tensor2 * value + input，支持广播"""
    # 将所有张量广播到相同的形状
    input_bc, tensor1_bc, tensor2_bc = _broadcast_tensors(input, tensor1, tensor2)
    # 执行计算
    return input_bc + value * tensor1_bc * tensor2_bc


@op(torch.ops.aten.addcdiv)
def _aten_addcdiv(input, tensor1, tensor2, *, value=1):
    """执行 tensor1 / tensor2 * value + input，支持广播"""
    # 将所有张量广播到相同的形状
    input_bc, tensor1_bc, tensor2_bc = _broadcast_tensors(input, tensor1, tensor2)
    # 执行计算
    return input_bc + value * (tensor1_bc / tensor2_bc)


@op(torch.ops.aten.triu_indices)
def _aten_triu_indices(row, col, offset=0, dtype=None, device=None):
    """返回上三角矩阵的索引"""
    a, b = mnp.triu_indices(row, offset, col)
    return mnp.stack((a, b))


@op(torch.ops.aten.tril_indices)
def _aten_tril_indices(row, col, offset=0, dtype=None, device=None):
    """返回下三角矩阵的索引"""
    a, b = mnp.tril_indices(row, offset, col)
    return mnp.stack((a, b))


# ==================== 张量创建算子 ====================

@op(torch.ops.aten.zeros, needs_env=True)
def _aten_zeros(*size, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建全零张量"""
    if not size:
        shape = (1,)
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = tuple(int(x) for x in size[0])
    else:
        shape = tuple(int(x) for x in size)
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    tensor = ops.zeros(shape, dtype=ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.ones, needs_env=True)
def _aten_ones(*size, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建全一张量"""
    if not size:
        shape = (1,)
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = tuple(int(x) for x in size[0])
    else:
        shape = tuple(int(x) for x in size)
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    tensor = ops.ones(shape, dtype=ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.full, needs_env=True)
def _aten_full(size, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建填充指定值的张量"""
    if isinstance(size, int):
        shape = (size,)
    elif isinstance(size, (list, tuple)):
        shape = tuple(int(x) for x in size)
    else:
        shape = tuple(int(x) for x in size)
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    # 使用ones_like然后乘以fill_value，或者使用mnp.full
    try:
        tensor = mnp.full(shape, fill_value, dtype=ms_dtype)
    except:
        # 回退方案：使用ones然后乘以fill_value
        tensor = ops.ones(shape, dtype=ms_dtype) * fill_value
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.full_like)
def _aten_full_like(input, fill_value, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    """创建与输入张量形状相同、填充指定值的张量"""
    if dtype is not None:
        ms_dtype = mappings.t2ms_dtype(dtype) if isinstance(dtype, torch.dtype) else dtype
    else:
        ms_dtype = input.dtype if hasattr(input, 'dtype') else ms.float32
    
    shape = input.shape if hasattr(input, 'shape') else tuple(input.shape)
    # 使用ones_like然后乘以fill_value，或者使用mnp.full
    try:
        return mnp.full(shape, fill_value, dtype=ms_dtype)
    except:
        # 回退方案：使用ones_like然后乘以fill_value
        return ops.ones_like(input, dtype=ms_dtype) * fill_value


@op(torch.ops.aten.empty, needs_env=True)
def _aten_empty(*size, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建未初始化的张量（MindSpore中创建全零张量）"""
    if not size:
        shape = (1,)
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = tuple(int(x) for x in size[0])
    else:
        shape = tuple(int(x) for x in size)
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    # MindSpore没有真正的未初始化张量，使用全零代替
    tensor = ops.zeros(shape, dtype=ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.empty_like)
def _aten_empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=None):
    """创建与输入张量形状相同的未初始化张量"""
    if dtype is not None:
        ms_dtype = mappings.t2ms_dtype(dtype) if isinstance(dtype, torch.dtype) else dtype
    else:
        ms_dtype = input.dtype if hasattr(input, 'dtype') else ms.float32
    
    shape = input.shape if hasattr(input, 'shape') else tuple(input.shape)
    # MindSpore没有真正的未初始化张量，使用全零代替
    return ops.zeros(shape, dtype=ms_dtype)


@op(torch.ops.aten.arange, needs_env=True)
def _aten_arange(*args, step=None, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建等差数列张量
    
    PyTorch的arange签名：
    - arange(end) -> 从0到end-1
    - arange(start, end) -> 从start到end-1
    - arange(start, end, step) -> 从start开始，步长为step，到end-1
    
    注意：aten.ops.aten.arange可能以位置参数形式调用
    """
    # 处理参数：根据位置参数数量判断
    if len(args) == 1:
        # arange(end)
        start_val = 0
        end_val = args[0]
        step_val = step if step is not None else 1
    elif len(args) == 2:
        # arange(start, end)
        start_val = args[0]
        end_val = args[1]
        step_val = step if step is not None else 1
    elif len(args) >= 3:
        # arange(start, end, step)
        start_val = args[0]
        end_val = args[1]
        step_val = args[2] if len(args) > 2 else (step if step is not None else 1)
    else:
        raise ValueError("arange requires at least 1 argument")
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.int64
    tensor = mnp.arange(start_val, end_val, step_val, dtype=ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.linspace, needs_env=True)
def _aten_linspace(start, end, steps, *, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建线性空间张量"""
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    tensor = mnp.linspace(start, end, steps, dtype=ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.rand, needs_env=True)
def _aten_rand(*size, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建均匀分布随机数张量 [0, 1)"""
    if not size:
        shape = (1,)
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        shape = tuple(int(x) for x in size[0])
    else:
        shape = tuple(int(x) for x in size)
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    # MindSpore使用uniform生成[0, 1)区间的均匀分布
    # 注意：MindSpore的uniform可能需要使用不同的API
    try:
        # 尝试使用ops.uniform
        tensor = ops.uniform(shape, ms_dtype, 0.0, 1.0)
    except (AttributeError, TypeError):
        # 回退方案：使用normal然后转换，或者使用numpy
        import numpy as np
        np_array = np.random.rand(*shape).astype(np.float32)
        tensor = ms.Tensor(np_array, dtype=ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


@op(torch.ops.aten.normal, needs_env=True)
def _aten_normal(mean, std, size=None, *, generator=None, out=None, dtype=None, layout=None, device=None, requires_grad=False, pin_memory=False, env=None):
    """创建正态分布随机数张量"""
    if size is None:
        if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
            shape = (1,)
        else:
            # mean和std是张量，使用它们的形状
            shape = mean.shape if hasattr(mean, 'shape') else tuple(mean.shape)
    else:
        if isinstance(size, int):
            shape = (size,)
        elif isinstance(size, (list, tuple)):
            shape = tuple(int(x) for x in size)
        else:
            shape = tuple(int(x) for x in size)
    
    ms_dtype = mappings.t2ms_dtype(dtype) if dtype is not None else ms.float32
    
    # 如果mean和std是标量
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        tensor = ops.normal(shape=shape, mean=float(mean), stddev=float(std))
    else:
        # mean和std是张量，需要广播
        mean_tensor = mean if hasattr(mean, 'shape') else mnp.array(mean)
        std_tensor = std if hasattr(std, 'shape') else mnp.array(std)
        # 简化实现：使用固定mean和std
        tensor = ops.normal(shape=shape, mean=0.0, stddev=1.0) * std_tensor + mean_tensor
    
    if dtype is not None:
        tensor = tensor.astype(ms_dtype)
    
    if env is not None:
        return env.ms2t_iso(tensor)
    return tensor


# ==================== 索引和查找算子 ====================

@op(torch.ops.aten.nonzero)
def _aten_nonzero(input, *, as_tuple=False):
    """查找非零元素的索引"""
    indices = ops.nonzero(input)
    if as_tuple:
        # 返回元组形式，每个维度一个张量
        if indices.shape[0] == 0:
            # 如果没有非零元素，返回空张量的元组
            return tuple(ops.zeros((0,), dtype=ms.int64) for _ in range(input.ndim))
        # 转置并分离每个维度
        indices_t = ops.transpose(indices, (1, 0))
        return tuple(indices_t[i] for i in range(input.ndim))
    return indices


@op(torch.ops.aten.unique_consecutive)
def _aten_unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    """查找连续的唯一值"""
    # 手动实现unique_consecutive（MindSpore可能没有这个函数）
    # 先将input转换为numpy数组进行计算，然后转换回MindSpore
    if hasattr(input, 'asnumpy'):
        input_np = input.asnumpy()
    elif hasattr(input, 'numpy'):
        input_np = input.numpy()
    else:
        input_np = np.array(input)
    
    if dim is None:
        # 展平处理
        input_flat_np = input_np.flatten()
        if len(input_flat_np) == 0:
            unique_vals_np = input_flat_np
            inverse_indices_np = np.array([], dtype=np.int64) if return_inverse else None
            counts_np = np.array([], dtype=np.int64) if return_counts else None
        else:
            # 找到连续不同的值
            if len(input_flat_np) == 1:
                mask_np = np.array([True])
            else:
                # 第一个元素总是保留，然后找出与前面不同的元素
                diff = input_flat_np[1:] != input_flat_np[:-1]
                mask_np = np.concatenate([np.array([True]), diff])
            
            unique_vals_np = input_flat_np[mask_np]
            
            if return_inverse:
                # 计算inverse_indices：每个元素在unique_vals中的索引
                inverse_indices_np = np.cumsum(mask_np.astype(np.int64)) - 1
            else:
                inverse_indices_np = None
            
            if return_counts:
                # 计算每个唯一值的连续出现次数
                input_len = len(input_flat_np)
                if input_len == 1:
                    counts_np = np.array([1], dtype=np.int64)
                else:
                    # 找出mask中True的位置
                    true_indices_np = np.where(mask_np)[0]
                    num_unique = len(true_indices_np)
                    if num_unique == 1:
                        counts_np = np.array([input_len], dtype=np.int64)
                    else:
                        # 计算每个区间的长度：相邻True位置之间的差值
                        intervals = []
                        for i in range(len(true_indices_np)):
                            if i == len(true_indices_np) - 1:
                                # 最后一个区间：从当前位置到末尾
                                intervals.append(input_len - int(true_indices_np[i]))
                            else:
                                # 中间区间：相邻位置之间的差值
                                intervals.append(int(true_indices_np[i+1]) - int(true_indices_np[i]))
                        counts_np = np.array(intervals, dtype=np.int64)
            else:
                counts_np = None
        
        # 转换回MindSpore Tensor
        unique_vals = ms.Tensor(unique_vals_np, dtype=input.dtype if hasattr(input, 'dtype') else ms.float32)
        if return_inverse and inverse_indices_np is not None:
            inverse_indices = ms.Tensor(inverse_indices_np, dtype=ms.int64)
        else:
            inverse_indices = None
        if return_counts and counts_np is not None:
            counts = ms.Tensor(counts_np, dtype=ms.int64)
        else:
            counts = None
    else:
        # 沿指定维度处理 - 简化实现，只处理1D情况
        if input.ndim == 1:
            return _aten_unique_consecutive(input, return_inverse, return_counts, dim=None)
        else:
            # 对于多维情况，简化处理
            raise NotImplementedError("unique_consecutive with dim parameter for multi-dimensional tensors is not fully implemented")
    
    result = [unique_vals]
    if return_inverse and inverse_indices is not None:
        result.append(inverse_indices)
    if return_counts and counts is not None:
        result.append(counts)
    
    return result[0] if len(result) == 1 else tuple(result)


@op(torch.ops.aten.bincount)
def _aten_bincount(input, weights=None, minlength=0):
    """计算每个值的出现次数"""
    # MindSpore的bincount实现
    if weights is None:
        return ops.bincount(input, minlength=minlength)
    else:
        return ops.bincount(input, weights=weights, minlength=minlength)


@op(torch.ops.aten.bucketize)
def _aten_bucketize(input, boundaries, *, out_int32=False, right=False):
    """将输入值分桶到边界定义的区间"""
    # MindSpore可能没有直接的bucketize，使用searchsorted实现
    if right:
        # right=True: 使用右边界（包含右边界）
        indices = mnp.searchsorted(boundaries, input, side='right')
    else:
        # right=False: 使用左边界（包含左边界）
        indices = mnp.searchsorted(boundaries, input, side='left')
    
    if out_int32:
        indices = indices.astype(ms.int32)
    return indices


@op(torch.ops.aten.searchsorted)
def _aten_searchsorted(sorted_sequence, self, *, out_int32=False, right=False):
    """在排序序列中搜索插入位置"""
    if right:
        indices = mnp.searchsorted(sorted_sequence, self, side='right')
    else:
        indices = mnp.searchsorted(sorted_sequence, self, side='left')
    
    if out_int32:
        indices = indices.astype(ms.int32)
    return indices


# ==================== 其他实用算子 ====================

@op(torch.ops.aten.unfold)
def _aten_unfold(input, dimension, size, step):
    """展开张量（滑动窗口）"""
    # MindSpore可能没有直接的unfold，使用slice和stack实现
    dim = dimension
    if dim < 0:
        dim += input.ndim
    
    input_shape = list(input.shape)
    num_windows = (input_shape[dim] - size) // step + 1
    
    # 创建索引列表
    slices = []
    for i in range(num_windows):
        start_idx = i * step
        end_idx = start_idx + size
        indices = [slice(None)] * input.ndim
        indices[dim] = slice(start_idx, end_idx)
        slices.append(input[tuple(indices)])
    
    # 堆叠所有切片
    result = ops.stack(slices, axis=dim)
    return result


@op(torch.ops.aten.as_strided)
def _aten_as_strided(input, size, stride, storage_offset=None):
    """使用指定的步长创建视图（简化实现）"""
    # 这是一个复杂的操作，MindSpore可能不完全支持
    # 这里提供一个简化实现
    if storage_offset is not None and storage_offset != 0:
        # 如果有偏移，先切片
        indices = [slice(None)] * input.ndim
        indices[0] = slice(storage_offset, None)
        input = input[tuple(indices)]
    
    # 简化：只处理基本的reshape和transpose
    # 实际实现需要更复杂的stride处理
    return ops.reshape(input, size)


@op(torch.ops.aten.unsafe_chunk)
def _aten_unsafe_chunk(input, chunks, dim=0):
    """不安全的分块操作（与chunk相同）"""
    return ops.chunk(input, chunks, axis=dim)
