"""Normalization modules."""

from typing import List, Union
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class LayerNorm(Module):
    """Applies Layer Normalization.

    Args:
        normalized_shape: input shape from an expected input of size
        eps: a value added to the denominator for numerical stability
        elementwise_affine: whether to learn affine parameters
    """

    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: tuple
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5,
                 elementwise_affine: bool = True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(normalized_shape))
            self.bias = Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

    def __repr__(self):
        return f'LayerNorm({self.extra_repr()})'
