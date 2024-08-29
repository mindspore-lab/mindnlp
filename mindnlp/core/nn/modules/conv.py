# coding=utf-8
"""conv"""
import math
from typing import Optional, Tuple, Union, List
from mindspore import Tensor, Parameter, ops as mops
from .module import Module
from ..common_types import _size_2_t, _size_1_t
from ._utils import _single, _pair, _reverse_repeat_tuple
from .. import init
from .. import functional as F
from ... import ops


class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 dtype=None) -> None:
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
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
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(ops.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(ops.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(ops.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class Conv1d(_ConvNd):
    r"""Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        dtype=None
    ) -> None:
        factory_kwargs = {'dtype': dtype}
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

        pad_mode = 'valid'
        pad = padding
        if isinstance(padding, tuple):
            if padding[0] != 0:
                pad_mode = 'pad'
            pad = (0, 0, padding[0], padding[0])
        elif isinstance(padding, int):
            if padding != 0:
                pad_mode = 'pad'
            pad = (0, 0) + (padding,) * 2
        if not isinstance(padding, (int, tuple)):
            pad_mode = padding
            pad = (0,) * 4

        if self.padding_mode != 'zeros':
            pad_mode = 'valid'
            pad = (0,) * 4
        self.conv2d = mops.Conv2D(out_channel=self.out_channels,
                                kernel_size=(1,) + self.kernel_size,
                                mode=1,
                                pad_mode=pad_mode,
                                pad=pad,
                                stride=(1,) + self.stride,
                                dilation=(1,) + self.dilation,
                                group=self.groups)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        input = input.expand_dims(2)
        output = self.conv2d(input, self.weight.expand_dims(2))

        if self.bias is not None:
            output = mops.bias_add(output, self.bias)

        output = output.squeeze(2)
        return output


class Conv2d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype=None
    ) -> None:
        factory_kwargs = {'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

        pad_mode = 'pad'
        pad = padding
        if isinstance(padding, tuple):
            pad = (padding[0], padding[0], padding[1], padding[1])
        elif isinstance(padding, int):
            pad = (padding,) * 4
        if not isinstance(padding, (int, tuple)):
            pad_mode = padding
            pad = (0,) * 4

        self.conv2d = mops.Conv2D(out_channel=self.out_channels,
                                kernel_size=self.kernel_size,
                                mode=1,
                                pad_mode=pad_mode,
                                pad=pad,
                                stride=self.stride,
                                dilation=self.dilation,
                                group=self.groups)
    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            input = ops.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        output = self.conv2d(input, weight)
        if bias is not None:
            output = mops.bias_add(output, bias)
        return output

    def forward(self, input):
        return self._conv_forward(input, self.weight, self.bias)



class Conv3d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype=None
    ) -> None:
        factory_kwargs = {'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = dilation
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

        pad_mode = 'pad'
        pad = padding
        if isinstance(padding, tuple):
            pad = (padding[0], padding[0], padding[1], padding[1])
        elif isinstance(padding, int):
            pad = (padding,) * 6
        if not isinstance(padding, (int, tuple)):
            pad_mode = padding
            pad = (0,) * 6

        self.conv3d = mops.Conv3D(out_channel=self.out_channels,
                                kernel_size=self.kernel_size,
                                mode=1,
                                pad_mode=pad_mode,
                                pad=pad,
                                stride=self.stride,
                                dilation=self.dilation,
                                group=self.groups)

    def forward(self, input):
        if self.padding_mode != 'zeros':
            input = ops.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        output = self.conv3d(input, self.weight)
        if self.bias is not None:
            output = mops.bias_add(output, self.bias)
        return output

# class Conv3d(_ConvNd):
#     r"""Applies a 3D convolution over an input signal composed of several input
#     planes.

#     In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
#     and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:

#     .. math::

#         \begin{array}{ll}
#         out(N_i, C_{out_j})  = bias(C_{out_j})
#                        + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
#         \end{array}

#     where :math:`\star` is the valid 3D `cross-correlation`_ operator

#     | :attr:`stride` controls the stride for the cross-correlation.
#     | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
#       for :attr:`padding` number of points.
#     | :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
#       It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
#     | :attr:`groups` controls the connections between inputs and outputs. `in_channels` and `out_channels`
#       must both be divisible by `groups`.
#     |       At groups=1, all inputs are convolved to all outputs.
#     |       At groups=2, the operation becomes equivalent to having two conv layers
#                  side by side, each seeing half the input channels,
#                  and producing half the output channels, and both subsequently concatenated.
#             At groups=`in_channels`, each input channel is convolved with its own set of filters
#                  (of size `out_channels // in_channels`).

#     The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

#         - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
#         - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
#           the second `int` for the height dimension and the third `int` for the width dimension

#     .. note::

#          Depending of the size of your kernel, several (of the last)
#          columns of the input might be lost, because it is a valid `cross-correlation`_,
#          and not a full `cross-correlation`_.
#          It is up to the user to add proper padding.

#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int or tuple): Size of the convolving kernel
#         stride (int or tuple, optional): Stride of the convolution
#         padding (int or tuple, optional): Zero-padding added to both sides of the input
#         dilation (int or tuple, optional): Spacing between kernel elements
#         groups (int, optional): Number of blocked connections from input channels to output channels
#         bias (bool, optional): If True, adds a learnable bias to the output

#     Shape:
#         - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
#         - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where
#           :math:`D_{out} = floor((D_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
#           :math:`H_{out} = floor((H_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
#           :math:`W_{out} = floor((W_{in}  + 2 * padding[2] - dilation[2] * (kernel\_size[2] - 1) - 1) / stride[2] + 1)`

#     Attributes:
#         weight (Tensor): the learnable weights of the module of shape
#                          (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
#         bias (Tensor):   the learnable bias of the module of shape (out_channels)

#     Examples::

#         >>> # With square kernels and equal stride
#         >>> m = nn.Conv3d(16, 33, 3, stride=2)
#         >>> # non-square kernels and unequal stride and with padding
#         >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
#         >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
#         >>> output = m(input)

#     .. _cross-correlation:
#         https://en.wikipedia.org/wiki/Cross-correlation

#     .. _link:
#         https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True):
#         kernel_size = _triple(kernel_size)
#         stride = _triple(stride)
#         padding = _triple(padding)
#         dilation = _triple(dilation)
#         super(Conv3d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _triple(0), groups, bias)

#     def forward(self, input):
#         return ops.conv3d(input, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)


class _ConvTransposeNd(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, dtype=None) -> None:
        if padding_mode != 'zeros':
            raise ValueError(f'Only "zeros" padding mode is supported for {self.__class__.__name__}')

        factory_kwargs = {'dtype': dtype}
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode, **factory_kwargs)

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
                        stride: List[int], padding: List[int], kernel_size: List[int],
                        num_spatial_dims: int, dilation: Optional[List[int]] = None) -> List[int]:
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    f"ConvTranspose{num_spatial_dims}D: for {input.dim()}D input, output_size must have {num_spatial_dims} or {num_non_spatial_dims + num_spatial_dims} elements (got {len(output_size)})")

            min_sizes = []
            max_sizes = []
            for d in range(num_spatial_dims):
                dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        f"requested an output size of {output_size}, but valid sizes range "
                        f"from {min_sizes} to {max_sizes} (for an input of {input.size()[2:]})")

            res = []
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

class ConvTranspose1d(_ConvTransposeNd):
    """Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    | :attr:`stride` controls the stride for the cross-correlation.
    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points.
    | If :attr:`output_padding` is non-zero, then the output is implicitly zero-padded on one side
      for :attr:`output_padding` number of points.
    | :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
    | :attr:`groups` controls the connections between inputs and outputs. `in_channels` and `out_channels`
      must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv layers
                 side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently concatenated.
            At groups=`in_channels`, each input channel is convolved with its own set of filters
                 (of size `out_channels // in_channels`).

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        output_padding (int or tuple, optional): Zero-padding added to one side of the output
        groups (int, optional): Number of blocked connections from input channels to output channels
        bias (bool, optional): If True, adds a learnable bias to the output
        dilation (int or tuple, optional): Spacing between kernel elements

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + kernel\_size + output\_padding`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode: str = 'zeros'):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super(ConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

        pad_mode = 'pad'
        pad = padding
        if isinstance(padding, tuple):
            pad = (0, 0, padding[0], padding[0])
        elif isinstance(padding, int):
            pad = (0, 0) + (padding,) * 2
        if not isinstance(padding, (int, tuple)):
            pad_mode = padding
            pad = (0,) * 4

        # cause Conv2DTranspose's out_channel refers to Conv2D's out_channel.
        self.conv2d_transpose = mops.Conv2DTranspose(out_channel=self.out_channels,
                                                    kernel_size=(1,) + self.kernel_size,
                                                    mode=1,
                                                    pad_mode=pad_mode,
                                                    pad=pad,
                                                    stride=(1,) + self.stride,
                                                    dilation=(1,) + self.dilation,
                                                    group=self.groups)
        self.h_add = _deconv_output_length(pad_mode, 1, 1, 1, pad[0] + pad[1])
        self.w_add = _deconv_output_length(pad_mode, kernel_size[0], stride[0], dilation[0], pad[2] + pad[3])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]
        input = mops.expand_dims(input, 2)
        n, _, h, w = input.shape
        conv2d_trans_ret = self.conv2d_transpose(input, self.weight.expand_dims(2),
                                                 (n, self.out_channels,
                                                  h + self.h_add,
                                                  w * self.stride[0] + self.w_add))
        if self.bias is not None:
            conv2d_trans_ret = mops.bias_add(conv2d_trans_ret, self.bias)

        conv2d_trans_ret = conv2d_trans_ret.squeeze(2)
        conv2d_trans_ret = ops.pad(conv2d_trans_ret, (0,) + output_padding, value=0.)
        return conv2d_trans_ret


def _deconv_output_length(pad_mode, filter_size, stride_size, dilation_size, padding):
    """Calculate the width and height of output."""
    length = 0
    filter_size = filter_size + (filter_size - 1) * (dilation_size - 1)
    if pad_mode == 'valid':
        if filter_size - stride_size > 0:
            length = filter_size - stride_size
    elif pad_mode == 'pad':
        length = - padding + filter_size - stride_size

    return length

class ConvTranspose2d(_ConvTransposeNd):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation).

    | :attr:`stride` controls the stride for the cross-correlation.
    | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
      for :attr:`padding` number of points.
    | If :attr:`output_padding` is non-zero, then the output is implicitly zero-padded on one side
      for :attr:`output_padding` number of points.
    | :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
    | :attr:`groups` controls the connections between inputs and outputs. `in_channels` and `out_channels`
      must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv layers
                 side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently concatenated.
            At groups=`in_channels`, each input channel is convolved with its own set of filters
                 (of size `out_channels // in_channels`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution
        padding (int or tuple, optional): Zero-padding added to both sides of the input
        output_padding (int or tuple, optional): Zero-padding added to one side of the output
        groups (int, optional): Number of blocked connections from input channels to output channels
        bias (bool, optional): If True, adds a learnable bias to the output
        dilation (int or tuple, optional): Spacing between kernel elements

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
          :math:`W_{out} = (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (in_channels, out_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> h.size()
        torch.Size([1, 16, 6, 6])
        >>> output = upsample(h, output_size=input.size())
        >>> output.size()
        torch.Size([1, 16, 12, 12])

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros', dtype=None):
        factory_kwargs = {'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode, **factory_kwargs)

        pad_mode = 'pad'
        pad = padding
        if isinstance(padding, tuple):
            pad = (padding[0], padding[0], padding[1], padding[1])
        elif isinstance(padding, int):
            pad = (padding,) * 4
        if not isinstance(padding, (int, tuple)):
            pad_mode = padding
            pad = (0,) * 4

        # cause Conv2DTranspose's out_channel refers to Conv2D's out_channel.
        self.conv2d_transpose = mops.Conv2DTranspose(out_channel=in_channels,
                                                    kernel_size=kernel_size,
                                                    mode=1,
                                                    pad_mode=pad_mode,
                                                    pad=pad,
                                                    stride=stride,
                                                    dilation=dilation,
                                                    group=groups)

        self.h_add = _deconv_output_length(pad_mode, kernel_size[0], stride[0], dilation[0], pad[0] + pad[1])
        self.w_add = _deconv_output_length(pad_mode, kernel_size[1], stride[1], dilation[1], pad[2] + pad[3])

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        n, _, h, w = input.shape
        conv2d_trans_ret = self.conv2d_transpose(input, self.weight,
                                                 (n, self.out_channels,
                                                  h * self.stride[0] + self.h_add,
                                                  w * self.stride[1] + self.w_add))
        if self.bias is not None:
            conv2d_trans_ret = mops.bias_add(conv2d_trans_ret, self.bias)

        conv2d_trans_ret = ops.pad(conv2d_trans_ret, output_padding, value=0.)

        return conv2d_trans_ret


# class ConvTranspose3d(_ConvTransposeNd):
#     r"""Applies a 3D transposed convolution operator over an input image composed of several input
#     planes.
#     The transposed convolution operator multiplies each input value element-wise by a learnable kernel,
#     and sums over the outputs from all input feature planes.

#     This module can be seen as the gradient of Conv3d with respect to its input.
#     It is also known as a fractionally-strided convolution or
#     a deconvolution (although it is not an actual deconvolution operation).

#     | :attr:`stride` controls the stride for the cross-correlation.
#     | If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
#       for :attr:`padding` number of points.
#     | If :attr:`output_padding` is non-zero, then the output is implicitly zero-padded on one side
#       for :attr:`output_padding` number of points.
#     | :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
#       It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
#     | :attr:`groups` controls the connections between inputs and outputs. `in_channels` and `out_channels`
#       must both be divisible by `groups`.
#     |       At groups=1, all inputs are convolved to all outputs.
#     |       At groups=2, the operation becomes equivalent to having two conv layers
#                  side by side, each seeing half the input channels,
#                  and producing half the output channels, and both subsequently concatenated.
#             At groups=`in_channels`, each input channel is convolved with its own set of filters
#                  (of size `out_channels // in_channels`).

#     The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
#     can either be:

#         - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
#         - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
#           the second `int` for the height dimension and the third `int` for the width dimension

#     .. note::

#          Depending of the size of your kernel, several (of the last)
#          columns of the input might be lost, because it is a valid `cross-correlation`_,
#          and not a full `cross-correlation`_.
#          It is up to the user to add proper padding.

#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int or tuple): Size of the convolving kernel
#         stride (int or tuple, optional): Stride of the convolution
#         padding (int or tuple, optional): Zero-padding added to both sides of the input
#         output_padding (int or tuple, optional): Zero-padding added to one side of the output
#         groups (int, optional): Number of blocked connections from input channels to output channels
#         bias (bool, optional): If True, adds a learnable bias to the output
#         dilation (int or tuple, optional): Spacing between kernel elements

#     Shape:
#         - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
#         - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where
#           :math:`D_{out} = (D_{in} - 1) * stride[0] - 2 * padding[0] + kernel\_size[0] + output\_padding[0]`
#           :math:`H_{out} = (H_{in} - 1) * stride[1] - 2 * padding[1] + kernel\_size[1] + output\_padding[1]`
#           :math:`W_{out} = (W_{in} - 1) * stride[2] - 2 * padding[2] + kernel\_size[2] + output\_padding[2]`

#     Attributes:
#         weight (Tensor): the learnable weights of the module of shape
#                          (in_channels, out_channels, kernel_size[0], kernel_size[1], kernel_size[2])
#         bias (Tensor):   the learnable bias of the module of shape (out_channels)

#     Examples::

#         >>> # With square kernels and equal stride
#         >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
#         >>> # non-square kernels and unequal stride and with padding
#         >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
#         >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
#         >>> output = m(input)

#     .. _cross-correlation:
#         https://en.wikipedia.org/wiki/Cross-correlation

#     .. _link:
#         https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, output_padding=0, groups=1, bias=True, dilation=1):
#         kernel_size = _triple(kernel_size)
#         stride = _triple(stride)
#         padding = _triple(padding)
#         dilation = _triple(dilation)
#         output_padding = _triple(output_padding)
#         super(ConvTranspose3d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             True, output_padding, groups, bias)

#     def forward(self, input, output_size=None):
#         output_padding = self._output_padding(input, output_size)
#         return F.conv_transpose3d(
#             input, self.weight, self.bias, self.stride, self.padding,
#             output_padding, self.groups, self.dilation)


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap
