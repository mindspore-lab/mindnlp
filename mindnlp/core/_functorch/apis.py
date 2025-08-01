import functools
from typing import Callable

from .vmap import (
    _check_randomness_arg,
    vmap_impl,
)

def vmap(
    func: Callable,
    in_dims = 0,
    out_dims = 0,
    randomness: str = "error",
    *,
    chunk_size=None,
) -> Callable:
    # from torch.compiler import is_compiling

    # _check_randomness_arg(randomness)
    # if not (chunk_size is None or chunk_size > 0):
    #     raise ValueError(
    #         f"vmap: chunk_size should be None or greater than 0. (got {chunk_size})"
    #     )

    # def wrapped(*args, **kwargs):
    #     return vmap_impl(
    #         func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs
    #     )

    # if not is_compiling():
    #     wrapped = functools.wraps(func)(wrapped)

    # return wrapped
    def batched_func(*args):
        # 标准化输入维度参数
        if isinstance(in_dims, int):
            in_dims_tuple = (in_dims,) * len(args)
        else:
            in_dims_tuple = in_dims
            if len(in_dims_tuple) != len(args):
                raise ValueError(f"输入的in_dims长度({len(in_dims_tuple)})与参数数量({len(args)})不匹配")
        
        # 识别并验证批处理大小
        batch_size = None
        for i, dim in enumerate(in_dims_tuple):
            if dim is not None:
                if batch_size is None:
                    batch_size = args[i].shape[dim]
                elif args[i].shape[dim] != batch_size:
                    raise ValueError(f"不一致的批处理大小: "
                                     f"参数 {i} 有大小 {args[i].shape[dim]}, "
                                     f"期望 {batch_size}")
        
        # 如果没有批处理维度，设置批处理大小为1
        if batch_size is None:
            batch_size = 1
        
        # 重新排列所有输入，使批处理维度位于第0位
        reordered_args = []
        reshaped_shapes = []
        
        for arg, dim in zip(args, in_dims_tuple):
            if dim is None:
                # 无批处理维度：添加伪批处理维度并进行广播
                # 使用unsqueeze而不是expand来保持梯度
                expanded = arg.unsqueeze(0)
                if batch_size > 1:
                    expanded = expanded.expand(batch_size, *[-1]*arg.ndim)
                reordered_args.append(expanded)
                reshaped_shapes.append(None)  # 标记为无原始维度
            else:
                # 有批处理维度：将其移动到维度0
                # 创建新维度顺序: [dim, 0, 1, ..., dim-1, dim+1, ...]
                dims_order = [dim] + [d for d in range(arg.ndim) if d != dim]
                permuted = arg.permute(*dims_order)
                reordered_args.append(permuted)
                reshaped_shapes.append(arg.shape)  # 保存原始形状
                
        # 处理函数可能返回元组的情况
        result = func(*reordered_args)
        
        # 调整输出维度的函数
        def adjust_out_dims(tensor):
            if tensor.size(0) != batch_size:
                # 如果函数返回了标量或没有批处理维度
                return tensor.unsqueeze(out_dims).expand(
                    *[batch_size if i == out_dims else -1 
                      for i in range(tensor.ndim + 1)]
                )
            
            if out_dims == 0:
                return tensor
            
            # 创建将维度0移动到out_dims位置的新顺序
            new_order = list(range(1, tensor.ndim))
            new_order.insert(out_dims, 0)
            return tensor.permute(*new_order)
        
        # 处理不同类型输出
        if isinstance(result, tuple):
            return tuple(adjust_out_dims(r) for r in result)
        else:
            return adjust_out_dims(result)
    
    return batched_func