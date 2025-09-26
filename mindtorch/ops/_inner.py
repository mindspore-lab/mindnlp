"""inner ops"""
from mindtorch.executor import execute

def cast(input, dtype):
    return execute('cast', input, dtype)

def depend(*args):
    return execute('depend', *args)

def npu_get_float_status_v2(status):
    return execute('npu_get_float_status_v2', status)

def npu_clear_float_status_v2(status):
    return execute('npu_clear_float_status_v2', status)

def all_finite(inputs):
    return execute('all_finite', inputs)

def custom_masked_scatter_vec(input, mask, source):    
    output = input.clone()
    if mask.sum() == 0:
        return output
    output[mask] = source.flatten() # 关键的一行：向量化赋值
    return output

def masked_scatter(input, mask, source):
    if input.device.type == 'cuda':
        return custom_masked_scatter_vec(input, mask, source)
    return execute('masked_scatter', input, mask, source)

__all__ = [
    'cast', 'depend', 'masked_scatter',
    'npu_get_float_status_v2', 'npu_clear_float_status_v2',
    'all_finite'
]
