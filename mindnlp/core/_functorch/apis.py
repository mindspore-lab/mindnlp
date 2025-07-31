from typing import Callable
from mindnlp import core

def vmap(
    func: Callable,
    in_dims = 0,
    out_dims = 0,
    randomness: str = "error",
    *,
    chunk_size=None,
) -> Callable:
    def batched_func(*args):
        # 统一处理in_dims格式
        if not isinstance(in_dims, tuple):
            in_dims_tuple = (in_dims,) * len(args)
        else:
            in_dims_tuple = in_dims
            
        # 验证输入维度一致性
        batch_sizes = set()
        for i, (arg, dim) in enumerate(zip(args, in_dims_tuple)):
            if dim is not None:
                batch_sizes.add(arg.shape[dim])
        
        if len(batch_sizes) > 1:
            raise ValueError(f"不一致的批处理大小: {batch_sizes}")
        batch_size = next(iter(batch_sizes)) if batch_sizes else 1
        
        # 收集单个样本的结果
        results = []
        for b in range(batch_size):
            # 为当前批次构造输入
            single_args = []
            for arg, dim in zip(args, in_dims_tuple):
                if dim is None:
                    single_args.append(arg)
                else:
                    # 切片获取当前批次的样本
                    slices = [slice(None)] * arg.ndim
                    slices[dim] = b
                    single_args.append(arg[tuple(slices)])
            
            # 调用原始函数
            result = func(*single_args)
            results.append(result)
        
        # 堆叠结果并调整维度
        stacked = core.stack(results, dim=out_dims)
        return stacked
    
    return batched_func