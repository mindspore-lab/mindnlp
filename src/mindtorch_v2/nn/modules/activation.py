"""Activation function modules."""

from ..module import Module
from .. import functional as F
from ..._tensor import Tensor


class ReLU(Module):
    """Applies ReLU: max(0, x)."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class GELU(Module):
    """Applies GELU activation."""

    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)


class SiLU(Module):
    """Applies SiLU (Swish): x * sigmoid(x)."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)


class Sigmoid(Module):
    """Applies sigmoid: 1 / (1 + exp(-x))."""

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(Module):
    """Applies tanh activation."""

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class Softmax(Module):
    """Applies softmax along a dimension."""

    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(Module):
    """Applies log softmax along a dimension."""

    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'
