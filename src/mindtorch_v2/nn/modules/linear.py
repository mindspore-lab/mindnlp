"""Linear layer module."""

import math
import numpy as np
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class Identity(Module):
    """A placeholder identity operator."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Linear(Module):
    """Applies a linear transformation: y = xW^T + b.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If False, layer will not learn an additive bias. Default: True
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight with Kaiming uniform
        k = 1.0 / in_features
        bound = math.sqrt(k)
        weight_np = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def __repr__(self):
        return f'Linear({self.extra_repr()})'
