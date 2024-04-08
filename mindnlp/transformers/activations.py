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
import math
from functools import partial
from collections import OrderedDict
from mindspore import nn, ops, Tensor


class QuickGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def construct(self, input: Tensor) -> Tensor:
        return input * ops.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Cell):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def construct(self, x: Tensor) -> Tensor:
        return ops.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def construct(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + ops.tanh(self.precomputed_constant * (input + 0.044715 * ops.pow(input, 3))))


class MishActivation(nn.Cell):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def construct(self, input: Tensor) -> Tensor:
        return input * ops.tanh(ops.softplus(input))


class LinearActivation(nn.Cell):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def construct(self, input: Tensor) -> Tensor:
        return input


class LaplaceActivation(nn.Cell):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def construct(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + ops.erf(input))


class ReLUSquaredActivation(nn.Cell):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def construct(self, input):
        relu_applied = ops.relu(input)
        squared = ops.square(relu_applied)
        return squared



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
    'gelu_pytorch_tanh': nn.GELU,
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

gelu_new = partial(ops.gelu, approximate='tanh')
silu = ops.silu
gelu = ops.gelu
