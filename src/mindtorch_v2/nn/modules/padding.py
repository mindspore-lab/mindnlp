from ..module import Module
from .. import functional as F


class _PadNd(Module):
    def __init__(self, padding, value=0.0):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self):
        return f'padding={self.padding}'


class ZeroPad1d(_PadNd):
    def __init__(self, padding):
        super().__init__(padding, 0.0)


class ZeroPad2d(_PadNd):
    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, 0.0)


class ConstantPad1d(_PadNd):
    def __init__(self, padding, value):
        super().__init__(padding, value)

    def extra_repr(self):
        return f'padding={self.padding}, value={self.value}'


class ConstantPad2d(_PadNd):
    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, value)

    def extra_repr(self):
        return f'padding={self.padding}, value={self.value}'


class ReflectionPad1d(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def extra_repr(self):
        return f'padding={self.padding}'


class ReflectionPad2d(Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = padding

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def extra_repr(self):
        return f'padding={self.padding}'
