# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindNLP Activations"""

from collections import OrderedDict
from mindspore import nn


class ClassInstantier(OrderedDict):
    r"""
    Class Instantier
    """

    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    """
    Excitation equation matrix
    """
    'relu': nn.ReLU,
    'gelu': (nn.GELU, {"approximate": False}),
    'gelu_new': nn.GELU,
    'gelu_approximate': nn.GELU,
    "swish": nn.SiLU,  # MindSpore的SiLU激活函数是Swish函数
    "gelu_10": nn.GELU,  # MindSpore的GELU激活函数不支持设置最大值和最小值
    "gelu_fast": nn.FastGelu,
    "gelu_python": nn.GELU,  # MindSpore的GELU激活函数不支持选择是否使用Python实现
    "linear": nn.ReLU,  # MindSpore没有Linear激活函数，使用ReLU代替
    "mish": nn.Mish,
    "quick_gelu": nn.FastGelu,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    """
    Obtained parameters required for outputting self. activation in the SequenceSummary class
    :param activation_string:
    :return:
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
