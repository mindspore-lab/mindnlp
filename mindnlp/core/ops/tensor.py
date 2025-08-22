"""tensor op"""
import mindspore
from mindspore._c_expression import typing # pylint: disable=no-name-in-module, import-error

from mindnlp import core

def is_floating_point(input):
    return isinstance(input.dtype, typing.Float)

def is_tensor(input):
    return isinstance(input, mindspore.Tensor)

def numel(input):
    return input.numel()

def as_tensor(data, dtype=None, **kwarg):
    return core.tensor(data, dtype=dtype)

def is_complex(input):
    return input.dtype.is_complex

__all__ = ['as_tensor', 'is_floating_point', 'is_tensor', 'numel', 'is_complex']