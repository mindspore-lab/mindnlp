"""inner ops"""
from mindnlp.core.executor import execute

def cast(input, dtype):
    return execute('cast', input, dtype)

def assign(input, other):
    return execute('assign', input, other)

def depend(*args):
    return execute('depend', *args)

def npu_get_float_status_v2(status):
    return execute('npu_get_float_status_v2', status)

def npu_clear_float_status_v2(status):
    return execute('npu_clear_float_status_v2', status)


__all__ = ['cast', 'assign', 'depend', 'npu_get_float_status_v2', 'npu_clear_float_status_v2']
