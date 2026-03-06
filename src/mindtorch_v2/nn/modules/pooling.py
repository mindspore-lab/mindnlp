from ..module import Module
from .. import functional as F


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

    def extra_repr(self):
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}')


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

    def extra_repr(self):
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}')


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return F.max_pool3d(input, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)

    def extra_repr(self):
        return (f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}')


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad, self.divisor_override)

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'


class AvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return F.avg_pool3d(input, self.kernel_size, self.stride, self.padding,
                            self.ceil_mode, self.count_include_pad)

    def extra_repr(self):
        return f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}'


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool1d(input, self.output_size)

    def extra_repr(self):
        return f'output_size={self.output_size}'


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool2d(input, self.output_size)

    def extra_repr(self):
        return f'output_size={self.output_size}'


class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        return F.adaptive_avg_pool3d(input, self.output_size)

    def extra_repr(self):
        return f'output_size={self.output_size}'


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size, return_indices=False):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)
        self.return_indices = return_indices

    def forward(self, input):
        return F.adaptive_max_pool2d(input, self.output_size, self.return_indices)

    def extra_repr(self):
        return f'output_size={self.output_size}'


class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size, return_indices=False):
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        # Delegate to 2D: unsqueeze last dim, pool, squeeze
        x = input.unsqueeze(-1)  # (N, C, L) -> (N, C, L, 1)
        out_size = (self.output_size, 1)
        y = F.adaptive_max_pool2d(x, out_size, self.return_indices)
        if self.return_indices:
            return y[0].squeeze(-1), y[1].squeeze(-1)
        return y.squeeze(-1)

    def extra_repr(self):
        return f'output_size={self.output_size}'
