"""inner ops"""
from mindspore import ops

def cast(input, dtype):
    return ops.cast(input, dtype)

def assign(input, other):
    return ops.assign(input, other)

__all__ = ['cast', 'assign']
