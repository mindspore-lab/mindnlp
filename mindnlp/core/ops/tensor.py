# pylint: disable=no-name-in-module
"""tensor op"""

from mindspore._c_expression import typing

def is_floating_point(input):
    return isinstance(input.dtype, typing.Float)
