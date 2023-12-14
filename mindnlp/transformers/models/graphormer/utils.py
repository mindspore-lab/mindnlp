# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Graphormer utils
"""
from mindspore import Parameter
from mindspore.common.initializer import (
    initializer, Initializer, Zero, Normal, Constant, XavierUniform)


def initializer_decorator(generator: Initializer):
    """
    A decorator function that uses a given generator to initialize a tensor

    Args:
        generator (Initializer): The initializer for generating the tensor

    Returns:
        A function `func(param: Parameter, *args, **kwargs)` that initializes
        a tensor matching the shape and dtype of `param`
    """
    def func(param: Parameter, *args, **kwargs):
        return initializer(generator(*args, **kwargs), param.shape, param.dtype)
    return func


init_zero = initializer_decorator(Zero)
init_normal = initializer_decorator(Normal)
init_constant = initializer_decorator(Constant)
init_xavier_uniform = initializer_decorator(XavierUniform)
