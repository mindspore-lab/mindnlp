"""Dropout modules."""

from ..module import Module
from .. import functional as F
from ..._tensor import Tensor


class Dropout(Module):
    """Randomly zeroes elements with probability p during training."""

    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'
