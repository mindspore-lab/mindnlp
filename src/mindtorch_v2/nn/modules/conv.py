"""Convolutional layer modules."""

import math
import numpy as np
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class Conv1d(Module):
    """1D convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weight
        k = groups / (in_channels * self.kernel_size[0])
        bound = math.sqrt(k)
        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0])
        weight_np = np.random.uniform(-bound, bound, weight_shape).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.conv1d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class Conv2d(Module):
    """2D convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weight
        k = groups / (in_channels * self.kernel_size[0] * self.kernel_size[1])
        bound = math.sqrt(k)
        weight_shape = (out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        weight_np = np.random.uniform(-bound, bound, weight_shape).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class Conv3d(Module):
    """3D convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weight
        k = groups / (in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
        bound = math.sqrt(k)
        weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        weight_np = np.random.uniform(-bound, bound, weight_shape).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class ConvTranspose1d(Module):
    """1D transposed convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weight (note: shape is different from Conv1d)
        k = groups / (out_channels * self.kernel_size[0])
        bound = math.sqrt(k)
        weight_shape = (in_channels, out_channels // groups, self.kernel_size[0])
        weight_np = np.random.uniform(-bound, bound, weight_shape).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, output_size=None):
        return F.conv_transpose1d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)


class ConvTranspose2d(Module):
    """2D transposed convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weight
        k = groups / (out_channels * self.kernel_size[0] * self.kernel_size[1])
        bound = math.sqrt(k)
        weight_shape = (in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1])
        weight_np = np.random.uniform(-bound, bound, weight_shape).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, output_size=None):
        return F.conv_transpose2d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)


class ConvTranspose3d(Module):
    """3D transposed convolution layer."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding, output_padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        # Initialize weight
        k = groups / (out_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])
        bound = math.sqrt(k)
        weight_shape = (in_channels, out_channels // groups, *self.kernel_size)
        weight_np = np.random.uniform(-bound, bound, weight_shape).astype(np.float32)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_np = np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input, output_size=None):
        return F.conv_transpose3d(input, self.weight, self.bias, self.stride,
                                  self.padding, self.output_padding, self.groups, self.dilation)
