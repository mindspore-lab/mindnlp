"""inner ops"""
import mindspore
from mindspore import ops
from mindnlp.configs import use_pyboost

def cast(input, dtype):
    return ops.cast(input, dtype)

def assign(input, other):
    return ops.assign(input, other)

def pad(input, pad, mode='constant', value=0.0):
    if use_pyboost():
        return mindspore.mint.nn.functional.pad(input, pad, mode, value)
    if mode == 'reflect':
        return ops.pad(input, pad, mode)
    return ops.pad(input, pad, mode, value)

__all__ = ['cast', 'assign']
