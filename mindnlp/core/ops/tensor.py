"""tensor op"""
import mindspore
from mindspore._c_expression import typing # pylint: disable=no-name-in-module, import-error

def is_floating_point(input):
    return isinstance(input.dtype, typing.Float)

def is_tensor(input):
    return isinstance(input, mindspore.Tensor)

def numel(input):
    return input.numel()

def as_tensor(data, dtype=None, **kwarg):
    return mindspore.Tensor(data, dtype)

__all__ = ['as_tensor', 'is_floating_point', 'is_tensor', 'numel']