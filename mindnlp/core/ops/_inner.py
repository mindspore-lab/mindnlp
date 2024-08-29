"""inner ops"""
import mindspore
from mindspore import ops
from mindnlp.configs import USE_PYBOOST

def cast(input, dtype):
    return ops.cast(input, dtype)

def assign(input, other):
    return ops.assign(input, other)

def pad(input, pad, mode='constant', value=0.0):
    if USE_PYBOOST:
        return mindspore.mint.nn.functional.pad(input, pad, mode, value)
    if mode == 'reflect':
        return ops.pad(input, pad, mode)
    return ops.pad(input, pad, mode, value)

__all__ = ['cast', 'assign']
