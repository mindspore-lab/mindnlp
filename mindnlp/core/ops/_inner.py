"""inner ops"""
from mindnlp.core.executor import execute

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

def masked_scatter(input, mask, source):
    return execute('masked_scatter', input, mask, source)

__all__ = [
    'cast', 'depend', 'masked_scatter',
    'npu_get_float_status_v2', 'npu_clear_float_status_v2',
    'all_finite'
]
