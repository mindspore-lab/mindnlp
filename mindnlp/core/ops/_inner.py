"""inner ops"""
import mindspore
from mindspore import ops
from ..configs import use_pyboost

def cast(input, dtype):
    return ops.cast(input, dtype)

def assign(input, other):
    return ops.assign(input, other)

def call_ms_func(func_name, *args, **kwargs):
    out = kwargs.pop('out', None)
    if out is None:
        return func_name(*args, **kwargs)
    else:
        tmp = func_name(*args, **kwargs)
        return out.copy_(tmp)

__all__ = ['cast', 'assign']
