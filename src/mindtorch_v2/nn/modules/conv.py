import functools
import operator

from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from .. import functional as F


def _single(x):
    return (x,) if isinstance(x, int) else tuple(x)


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def _make_nested(flat, shape):
    if len(shape) == 1:
        return flat[:shape[0]]
    size = functools.reduce(operator.mul, shape[1:], 1)
    return [_make_nested(flat[i * size:(i + 1) * size], shape[1:]) for i in range(shape[0])]


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        if transposed:
            weight_shape = [in_channels, out_channels // groups] + list(kernel_size)
        else:
            weight_shape = [out_channels, in_channels // groups] + list(kernel_size)
        total = functools.reduce(operator.mul, weight_shape, 1)
        self.weight = Parameter(tensor(_make_nested([0.0] * total, weight_shape)))
        if bias:
            self.bias = Parameter(tensor([0.0] * out_channels))
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}'
             f', stride={self.stride}')
        if self.padding != (0,) * len(self.padding):
            s += f', padding={self.padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += f', dilation={self.dilation}'
        if self.groups != 1:
            s += f', groups={self.groups}'
        if self._parameters.get('bias') is None:
            s += ', bias=False'
        return s


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding) if not isinstance(padding, str) else padding
        dilation = _single(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, False, (0,), groups, bias, padding_mode, device, dtype)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding) if not isinstance(padding, str) else padding
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, False, (0, 0), groups, bias, padding_mode, device, dtype)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvTranspose1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 device=None, dtype=None):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups, bias, padding_mode, device, dtype)

    def forward(self, input, output_size=None):
        return F.conv_transpose1d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)


class ConvTranspose2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros',
                 device=None, dtype=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, True, output_padding, groups, bias, padding_mode, device, dtype)

    def forward(self, input, output_size=None):
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)
