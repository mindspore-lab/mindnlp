from ..module import Module
from .. import functional as F


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self):
        return f'p={self.p}, inplace={self.inplace}'


class Dropout1d(Dropout):
    pass


class Dropout2d(Dropout):
    pass


class Dropout3d(Module):
    """Randomly zero out entire channels of a 5D input (N, C, D, H, W).

    Each channel is zeroed out independently with probability ``p``.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout3d(input, self.p, self.training, self.inplace)

    def extra_repr(self):
        return f'p={self.p}, inplace={self.inplace}'


class AlphaDropout(Module):
    """Apply Alpha Dropout over the input, preserving self-normalizing properties.

    Alpha Dropout is designed for inputs that are the output of SELU activation.
    It randomly sets elements to the negative saturation value of SELU.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.alpha_dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self):
        return f'p={self.p}, inplace={self.inplace}'
