try:
    from mindspore._c_expression import TensorPy as Tensor_
except:
    from mindspore._c_expression import Tensor as Tensor_

import math
import numpy as np
from mindnlp import core

__all__ = []

def arange(start, end, step, dtype):
    out = Tensor_(shape=(math.ceil((end - start) / step), ), dtype=dtype)
    return core.Tensor(out)

__all__.append('arange')

import math
from typing import Tuple, Union

def infer_broadcast_shape(input_shape: Tuple[int, ...], 
                          target_shape: Tuple[Union[int, None], ...]) -> Tuple[int, ...]:
    """
    推断 torch.broadcast_to 的输出形状
    
    参数:
        input_shape: 输入张量的形状元组 (例如 (3, 1))
        target_shape: 目标广播形状元组 (可包含None表示自动推断维度)
    
    返回:
        广播后的输出形状元组
        
    异常:
        ValueError: 当广播不兼容时
    """
    # 处理 None 值（自动维度推断）
    final_target_shape = []
    for i, dim in enumerate(target_shape):
        if dim is None:
            # 查找可以推断的维度位置
            candidates = [j for j, d in enumerate(target_shape) if d is None]
            if len(candidates) > 1:
                raise ValueError(f"多个None维度 {candidates}，无法明确推断")
            final_target_shape.append(None)
        elif dim < -1:
            raise ValueError(f"维度大小不能为负数 (除-1外)，发现 {dim}")
        else:
            final_target_shape.append(dim)
    
    # 计算需要推断的总元素数量
    def count_product(shape, exclude_none=True):
        prod = 1
        for dim in shape:
            if dim == 0:
                return 0  # 任何维度为0结果即为0
            if dim is not None and not (exclude_none and dim == -1):
                prod *= max(1, dim)  # -1视为1用于计数
        return prod
    
    # 验证维度数量兼容性
    ndim_input = len(input_shape)
    ndim_target = len(final_target_shape)
    
    if ndim_input > ndim_target:
        raise ValueError(
            f"输入维度({ndim_input})多于目标维度({ndim_target})，"
            f"无法广播: {input_shape} -> {final_target_shape}"
        )
    
    # 创建对齐后的形状（左侧填充1）
    aligned_input_shape = (1,) * (ndim_target - ndim_input) + input_shape
    inferred_target_shape = list(final_target_shape)
    known_product = 1
    
    # 第一遍：收集已知信息
    for i in range(ndim_target):
        target_dim = inferred_target_shape[i]
        input_dim = aligned_input_shape[i]
        
        if target_dim == -1:
            # 标记需要推断的维度
            inferred_target_shape[i] = None
        elif target_dim is not None:
            # 验证维度兼容性
            if target_dim == 0:
                if input_dim not in (0, 1):
                    raise ValueError(
                        f"维度 {i}: 目标维度为0时输入维度必须为0或1, "
                        f"但得到 {input_dim} -> {target_dim}"
                    )
            else:  # 正数维度
                if input_dim != 1 and input_dim != target_dim:
                    raise ValueError(
                        f"维度 {i}: 大小 {input_dim} 无法广播到 {target_dim}"
                    )
            known_product *= target_dim

    # 第二遍：推断维度
    total_elements = math.prod([d for d in input_shape if d != 0])
    inferred_product = known_product
    
    # 统计需要推断的维度数量
    none_indices = [i for i, d in enumerate(inferred_target_shape) if d is None]
    num_infer = len(none_indices)
    
    if num_infer > 0:
        # 计算需要推断的总元素量
        required_total = total_elements
        
        # 当输入有0维时的特殊情况
        if 0 in input_shape:
            if required_total != 0:
                raise ValueError("含0维输入广播时无法推断非0维度")
            # 所有推断维度必须为0
            for i in none_indices:
                inferred_target_shape[i] = 0
        else:
            if inferred_product == 0 and required_total > 0:
                raise ValueError(
                    "无法将非0输入广播到含0维的目标形状: "
                    f"{input_shape} -> {inferred_target_shape}"
                )
            
            # 计算推断维度的乘积
            infer_product = required_total // inferred_product if inferred_product != 0 else 0
            
            if infer_product * inferred_product != required_total:
                raise ValueError(
                    f"元素总数不兼容: 输入有 {total_elements} 元素, "
                    f"但目标形状仅能容纳 {inferred_product * infer_product} 元素"
                )
            
            # 检查是否可以整数划分
            for i in none_indices:
                # 仅当有1个-1时可以推断
                if num_infer == 1:
                    inferred_target_shape[i] = infer_product
                else:
                    # 多维度无法自动推断
                    raise ValueError(
                        f"多个维度({len(none_indices)})需要推断: {none_indices} "
                        "但未指定足够约束条件"
                    )
    
    # 转换为确定形状元组
    result_shape = tuple(
        d if d is not None else -1  # 保留-1表示未指定
        for d in inferred_target_shape
    )
    
    return result_shape

def broadcast_to(input, shape):
    out_shape = infer_broadcast_shape(input.shape, shape)
    out = Tensor_(shape=out_shape, dtype=input.dtype)
    return core.Tensor(out)

__all__.append('broadcast_to')

def zeros(size, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return core.Tensor(out)

__all__.append('zeros')

def ones(size, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return core.Tensor(out)

__all__.append('ones')

def inplace_uniform(input, *args):
    return input

__all__.append('inplace_uniform')

def inplace_fill_scalar(input, value):
    return input

__all__.append('inplace_fill_scalar')

def inplace_normal(input, *args):
    return input

__all__.append('inplace_normal')

def getitem(input, slice):
    out = input.asnumpy()[slice]
    out = Tensor_(shape=out.shape, dtype=input.dtype)
    return core.Tensor(out)

__all__.append('getitem')

def sub_ext(input, other, alpha):
    return input

__all__.append('sub_ext')

def pad_v3(input, pad, mode, value):
    out = np.pad(input.asnumpy(), pad, mode, constant_values=value)
    out = Tensor_(shape=out.shape, dtype=input.dtype)
    return core.Tensor(out)

__all__.append('pad_v3')

def abs(input):
    return input

__all__.append('abs')

def cast(input, dtype):
    out = Tensor_(shape=input.shape, dtype=dtype)
    return core.Tensor(out)

__all__.append('cast')

def index_select(input, dim, index):
    out = np.take(input.asnumpy(), index.asnumpy(), dim)
    out = Tensor_(shape=out.shape, dtype=input.dtype)
    return core.Tensor(out)

__all__.append('index_select')

def identity(input):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return core.Tensor(out)

__all__.append('identity')

def contiguous(input):
    return input

__all__.append('contiguous')

def inplace_copy(input, other):
    return input

__all__.append('inplace_copy')

def div(input, other):
    if isinstance(input, core.Tensor):
        shape = input.shape
        dtype = input.dtype
    else:
        shape = other.shape
        dtype = other.dtype
    out = Tensor_(shape=shape, dtype=dtype)
    return core.Tensor(out)

__all__.append('div')

def pow_scalar_tensor(input, other):
    out = Tensor_(shape=other.shape, dtype=other.dtype)
    return core.Tensor(out)

__all__.append('pow_scalar_tensor')

def concat(tensors, dim):
    shape = list(tensors[0].shape)
    shape[dim] = sum([t.shape[dim] for t in tensors])
    out = Tensor_(shape=tuple(shape), dtype=tensors[0].dtype)
    return core.Tensor(out)

__all__.append('concat')

def tril_ext(input, k):
    return input

__all__.append('tril_ext')

def reshape(input, shape):
    out = Tensor_(shape=tuple(shape), dtype=input.dtype)
    return core.Tensor(out)

__all__.append('reshape')

def linalg_vector_norm(input, p, dim, keepdim, dtype):
    input_shape = list(input.shape)
    if isinstance(dim, int):
        dim = (dim,)
    for d in dim:
        input_shape[d] = 1 if keepdim else 0
    
    new_shape = []
    for s in input_shape:
        if s != 0:
            new_shape.append(s)
    if dtype is None:
        dtype = input.dtype
    out = Tensor_(shape=tuple(new_shape), dtype=dtype)
    return core.Tensor(out)

__all__.append('linalg_vector_norm')

def erfinv(input):
    return input
__all__.append('erfinv')


def stop_gradient(input):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return core.Tensor(out)

__all__.append('stop_gradient')

def log(input):
    return input
__all__.append('log')

def mul(input, other):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return core.Tensor(out)
__all__.append('mul')

def randn(size, seed, offset, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return core.Tensor(out)

__all__.append('randn')

def zeros_like_ext(input, *args, **kwargs):
    out = Tensor_(shape=input.shape, dtype=input.dtype)
    return core.Tensor(out)
__all__.append('zeros_like_ext')

def inplace_add_ext(input, other, alpha):
    return input
__all__.append('inplace_add_ext')

def clamp_scalar(input, *args):
    return input
__all__.append('clamp_scalar')

def expand_dims_view(input, dim):
    input_shape = list(input.shape)
    input_shape.insert(dim, 1)

    out = Tensor_(shape=tuple(input_shape), dtype=input.dtype)
    return core.Tensor(out)
__all__.append('expand_dims_view')

def floor_div(input, other):
    return input
__all__.append('floor_div')

def sin(input):
    return input

__all__.append('sin')

def cos(input):
    return input

__all__.append('cos')

def triu(input, diagonal):
    return input

__all__.append('triu')

def fill_scalar(size, fill_value, dtype):
    out = Tensor_(shape=size, dtype=dtype)
    return core.Tensor(out)
__all__.append('fill_scalar')

def sqrt(input):
    return input

__all__.append('sqrt')
