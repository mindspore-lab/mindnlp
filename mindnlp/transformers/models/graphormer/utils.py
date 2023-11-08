from typing import Optional

from mindspore import Parameter
from mindspore.common.initializer import (
    initializer, Zero, Normal, Constant, XavierUniform)


def initializer_decorator(generator):
    def func(param: Parameter, *args, **kwargs):
        return initializer(generator(*args, **kwargs), param.shape, param.dtype)
    return func


init_zero = initializer_decorator(Zero)
init_normal = initializer_decorator(Normal)
init_constant = initializer_decorator(Constant)
init_xavier_uniform = initializer_decorator(XavierUniform)
