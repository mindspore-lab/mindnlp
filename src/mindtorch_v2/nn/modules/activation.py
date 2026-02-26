from ..module import Module
from .. import functional as F


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)


class GELU(Module):
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self):
        return f'approximate={repr(self.approximate)}'


class SiLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.silu(input, inplace=self.inplace)


class Sigmoid(Module):
    def forward(self, input):
        return F.sigmoid(input)


class Tanh(Module):
    def forward(self, input):
        return F.tanh(input)


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmax(input, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class LogSoftmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.log_softmax(input, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input):
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self):
        return f'negative_slope={self.negative_slope}'


class ELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return F.elu(input, self.alpha, self.inplace)

    def extra_repr(self):
        return f'alpha={self.alpha}'


class Mish(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.mish(input, inplace=self.inplace)


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        super().__init__()
        from ..parameter import Parameter
        from ..._creation import tensor
        self.num_parameters = num_parameters
        self.weight = Parameter(tensor([init] * num_parameters))

    def forward(self, input):
        return F.prelu(input, self.weight)

    def extra_repr(self):
        return f'num_parameters={self.num_parameters}'
