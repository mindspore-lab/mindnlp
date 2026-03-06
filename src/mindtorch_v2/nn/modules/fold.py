from ..module import Module
from .. import functional as F


class Fold(Module):
    r"""Combines an array of sliding local blocks into a large containing tensor.

    See :func:`torch.nn.functional.fold` for details and output shape.

    Args:
        output_size (int or tuple): the shape of the spatial dimensions of the
            output (i.e., ``output.sizes()[2:]``)
        kernel_size (int or tuple): the size of the sliding blocks
        dilation (int or tuple, optional): a parameter that controls the
            stride of elements within the neighborhood. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
            both sides of input. Default: 0
        stride (int or tuple, optional):  the stride of the sliding blocks in
            the input spatial dimensions. Default: 1
    """

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return F.fold(input, self.output_size, self.kernel_size,
                      self.dilation, self.padding, self.stride)

    def extra_repr(self):
        return (f'output_size={self.output_size}, kernel_size={self.kernel_size}, '
                f'dilation={self.dilation}, padding={self.padding}, stride={self.stride}')


class Unfold(Module):
    r"""Extracts sliding local blocks from a batched input tensor.

    See :func:`torch.nn.functional.unfold` for details and output shape.

    Args:
        kernel_size (int or tuple): the size of the sliding blocks
        dilation (int or tuple, optional): a parameter that controls the
            stride of elements within the neighborhood. Default: 1
        padding (int or tuple, optional): implicit zero paddings on both sides
            of input. Default: 0
        stride (int or tuple, optional):  the stride of the sliding blocks in
            the input spatial dimensions. Default: 1
    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return F.unfold(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)

    def extra_repr(self):
        return (f'kernel_size={self.kernel_size}, dilation={self.dilation}, '
                f'padding={self.padding}, stride={self.stride}')
