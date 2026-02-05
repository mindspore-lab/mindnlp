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


class LeakyReLU(Module):
    """Applies LeakyReLU: max(0, x) + negative_slope * min(0, x)."""

    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}, inplace={self.inplace}'


class ELU(Module):
    """Applies ELU activation."""

    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.elu(input, alpha=self.alpha, inplace=self.inplace)


class CELU(Module):
    """Applies CELU activation."""

    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.celu(input, alpha=self.alpha, inplace=self.inplace)


class SELU(Module):
    """Applies SELU activation."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, inplace=self.inplace)


class PReLU(Module):
    """Applies PReLU (Parametric ReLU)."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        import numpy as np
        from ..parameter import Parameter
        import mindtorch_v2 as torch
        self.weight = Parameter(torch.tensor(np.full(num_parameters, init, dtype=np.float32)))

    def forward(self, input: Tensor) -> Tensor:
        return F.prelu(input, self.weight)


class Mish(Module):
    """Applies Mish: x * tanh(softplus(x))."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.mish(input, inplace=self.inplace)


class Hardswish(Module):
    """Applies Hardswish activation."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardswish(input, inplace=self.inplace)


class Hardsigmoid(Module):
    """Applies Hardsigmoid activation."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardsigmoid(input, inplace=self.inplace)


class Softplus(Module):
    """Applies Softplus: log(1 + exp(beta * x)) / beta."""

    __constants__ = ['beta', 'threshold']
    beta: float
    threshold: float

    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: Tensor) -> Tensor:
        return F.softplus(input, beta=self.beta, threshold=self.threshold)


class Hardtanh(Module):
    """Applies Hardtanh activation."""

    __constants__ = ['min_val', 'max_val', 'inplace']
    min_val: float
    max_val: float
    inplace: bool

    def __init__(self, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.hardtanh(input, min_val=self.min_val, max_val=self.max_val, inplace=self.inplace)


class ReLU6(Module):
    """Applies ReLU6: min(max(0, x), 6)."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu6(input, inplace=self.inplace)

