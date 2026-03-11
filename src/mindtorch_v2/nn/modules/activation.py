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


class ReLU6(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input)


class Hardtanh(Module):
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input):
        return F.hardtanh(input, self.min_val, self.max_val)

    def extra_repr(self):
        return f'min_val={self.min_val}, max_val={self.max_val}'


class LogSigmoid(Module):
    def forward(self, input):
        return F.logsigmoid(input)


class Hardswish(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.hardswish(input, inplace=self.inplace)


class Hardsigmoid(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.hardsigmoid(input, inplace=self.inplace)


class SELU(Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.selu(input, inplace=self.inplace)


class CELU(Module):
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input):
        return F.celu(input, self.alpha, self.inplace)

    def extra_repr(self):
        return f'alpha={self.alpha}'


class Softplus(Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold)

    def extra_repr(self):
        return f'beta={self.beta}, threshold={self.threshold}'


class Softsign(Module):
    def forward(self, input):
        return F.softsign(input)


class Threshold(Module):
    def __init__(self, threshold, value, inplace=False):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, input):
        return F.threshold(input, self.threshold, self.value, self.inplace)

    def extra_repr(self):
        return f'threshold={self.threshold}, value={self.value}'


class GLU(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.glu(input, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class Softmax2d(Module):
    def forward(self, input):
        return F.softmax2d(input)


class Softmin(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.softmin(input, self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class Tanhshrink(Module):
    def forward(self, input):
        return F.tanhshrink(input)


class Softshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return F.softshrink(input, self.lambd)

    def extra_repr(self):
        return f'lambd={self.lambd}'


class Hardshrink(Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, input):
        return F.hardshrink(input, self.lambd)

    def extra_repr(self):
        return f'lambd={self.lambd}'


class RReLU(Module):
    def __init__(self, lower=1.0/8, upper=1.0/3, inplace=False):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.rrelu(input, self.lower, self.upper, self.training, self.inplace)

    def extra_repr(self):
        return f'lower={self.lower}, upper={self.upper}'

