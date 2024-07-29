# pylint: disable=no-name-in-module
"""tensor op"""
import mindspore
from mindspore._c_expression import typing

def is_floating_point(input):
    return isinstance(input.dtype, typing.Float)

def is_tensor(input):
    return isinstance(input, mindspore.Tensor)

def numel(input):
    return input.numel()
