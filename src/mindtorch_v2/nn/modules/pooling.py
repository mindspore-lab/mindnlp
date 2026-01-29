"""Pooling modules."""

import numpy as np
from ..module import Module
import mindtorch_v2 as torch


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, input):
        from .. import functional as F
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, input):
        from .. import functional as F
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding)


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, input):
        from .. import functional as F
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, input):
        x = input.numpy()
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        N, C, H, W = x.shape
        out_h = (H - kh) // sh + 1
        out_w = (W - kw) // sw + 1

        out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
        for i in range(out_h):
            for j in range(out_w):
                out[:, :, i, j] = np.mean(x[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3))

        return torch.Tensor(out)


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        x = input.numpy()
        N, C, L = x.shape
        out_l = self.output_size

        out = np.zeros((N, C, out_l), dtype=x.dtype)
        for i in range(out_l):
            start = int(np.floor(i * L / out_l))
            end = int(np.ceil((i + 1) * L / out_l))
            out[:, :, i] = np.mean(x[:, :, start:end], axis=2)

        return torch.Tensor(out)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif output_size is None:
            self.output_size = (1, 1)
        else:
            self.output_size = output_size

    def forward(self, input):
        x = input.numpy()
        N, C, H, W = x.shape
        out_h, out_w = self.output_size

        out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start = int(np.floor(i * H / out_h))
                h_end = int(np.ceil((i + 1) * H / out_h))
                w_start = int(np.floor(j * W / out_w))
                w_end = int(np.ceil((j + 1) * W / out_w))
                out[:, :, i, j] = np.mean(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return torch.Tensor(out)


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size, return_indices=False):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.return_indices = return_indices

    def forward(self, input):
        x = input.numpy()
        N, C, H, W = x.shape
        out_h, out_w = self.output_size

        out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
        for i in range(out_h):
            for j in range(out_w):
                h_start = int(np.floor(i * H / out_h))
                h_end = int(np.ceil((i + 1) * H / out_h))
                w_start = int(np.floor(j * W / out_w))
                w_end = int(np.ceil((j + 1) * W / out_w))
                out[:, :, i, j] = np.max(x[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))

        return torch.Tensor(out)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        shape = input.shape
        end = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim
        new_shape = list(shape[:self.start_dim])
        new_shape.append(-1)
        if end + 1 < len(shape):
            new_shape.extend(shape[end + 1:])
        return input.reshape(tuple(new_shape))
