"""Padding modules for neural networks."""

import numpy as np
from ..module import Module
from ..._tensor import Tensor


class _PadNd(Module):
    """Base class for padding modules."""

    def __init__(self, padding, value=0.0):
        super().__init__()
        self.padding = padding
        self.value = value

    def extra_repr(self):
        return f'padding={self.padding}'


class ZeroPad1d(_PadNd):
    """Pads the input tensor boundaries with zero using the specified padding."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding, value=0.0)

    def forward(self, input):
        # input: (N, C, L)
        x = input.numpy()
        left, right = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (left, right)), mode='constant', constant_values=0)
        return Tensor(result.astype(x.dtype))


class ZeroPad2d(_PadNd):
    """Pads the input tensor boundaries with zero.

    Args:
        padding: The size of the padding. If is int, uses the same padding in all boundaries.
                 If is tuple of 4 ints, uses (left, right, top, bottom).
    """

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, value=0.0)

    def forward(self, input):
        # input: (N, C, H, W)
        x = input.numpy()
        left, right, top, bottom = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)
        return Tensor(result.astype(x.dtype))


class ZeroPad3d(_PadNd):
    """Pads the input tensor boundaries with zero using the specified padding."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding, padding, padding)
        super().__init__(padding, value=0.0)

    def forward(self, input):
        # input: (N, C, D, H, W)
        x = input.numpy()
        left, right, top, bottom, front, back = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)),
                       mode='constant', constant_values=0)
        return Tensor(result.astype(x.dtype))


class ConstantPad1d(_PadNd):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding, value=value)

    def forward(self, input):
        x = input.numpy()
        left, right = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (left, right)), mode='constant', constant_values=self.value)
        return Tensor(result.astype(x.dtype))

    def extra_repr(self):
        return f'padding={self.padding}, value={self.value}'


class ConstantPad2d(_PadNd):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, value=value)

    def forward(self, input):
        x = input.numpy()
        left, right, top, bottom = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)),
                       mode='constant', constant_values=self.value)
        return Tensor(result.astype(x.dtype))

    def extra_repr(self):
        return f'padding={self.padding}, value={self.value}'


class ConstantPad3d(_PadNd):
    """Pads the input tensor boundaries with a constant value."""

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding, padding, padding)
        super().__init__(padding, value=value)

    def forward(self, input):
        x = input.numpy()
        left, right, top, bottom, front, back = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)),
                       mode='constant', constant_values=self.value)
        return Tensor(result.astype(x.dtype))

    def extra_repr(self):
        return f'padding={self.padding}, value={self.value}'


class ReflectionPad1d(_PadNd):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding, value=None)

    def forward(self, input):
        x = input.numpy()
        left, right = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (left, right)), mode='reflect')
        return Tensor(result.astype(x.dtype))


class ReflectionPad2d(_PadNd):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, value=None)

    def forward(self, input):
        x = input.numpy()
        left, right, top, bottom = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='reflect')
        return Tensor(result.astype(x.dtype))


class ReflectionPad3d(_PadNd):
    """Pads the input tensor using the reflection of the input boundary."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding, padding, padding)
        super().__init__(padding, value=None)

    def forward(self, input):
        x = input.numpy()
        left, right, top, bottom, front, back = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)), mode='reflect')
        return Tensor(result.astype(x.dtype))


class ReplicationPad1d(_PadNd):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding)
        super().__init__(padding, value=None)

    def forward(self, input):
        x = input.numpy()
        left, right = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (left, right)), mode='edge')
        return Tensor(result.astype(x.dtype))


class ReplicationPad2d(_PadNd):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super().__init__(padding, value=None)

    def forward(self, input):
        x = input.numpy()
        left, right, top, bottom = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='edge')
        return Tensor(result.astype(x.dtype))


class ReplicationPad3d(_PadNd):
    """Pads the input tensor using replication of the input boundary."""

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding, padding, padding)
        super().__init__(padding, value=None)

    def forward(self, input):
        x = input.numpy()
        left, right, top, bottom, front, back = self.padding
        result = np.pad(x, ((0, 0), (0, 0), (front, back), (top, bottom), (left, right)), mode='edge')
        return Tensor(result.astype(x.dtype))


__all__ = [
    'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d',
    'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
    'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d',
]
