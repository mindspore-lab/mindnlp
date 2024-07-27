# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""complex convolution implementation"""
import numbers
from typing import Callable, Tuple, Union

from mindspore.common.tensor import Tensor
from mindspore.common.initializer import Initializer
from mindspore import ops as P

from ..hypercomplex._hc_conv_impl import _BaseConvImpl as BaseConvImpl

class _ConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for complex numbers.

    Applies complex-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})
        - \text{ccor}(\text{Im(kernel)}, \text{Im(inp)})\\
        \text{Im(ccor)} = \text{ccor}(\text{Im(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Im(inp)})
        \end{align}

    where :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the complex-valued input tensors, :math:`\text{kernel}` is a complex weight matrix with the same
    data type as the :math:`inp` created by the layer, :math:`\text{Re(...)}` and :math:`\text{Im(...)}`
    are respectively real and imaginary parts of the complex-valued expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the complex convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **imag** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the imaginary part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the imaginary parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def forward(self,
                  conv_fn: Callable,
                  real: Tensor,
                  imag: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:
        r"""
        Constructs the complex convolution operation in the specified mode.
        
        Args:
            self (_ConvImpl): The instance of the _ConvImpl class.
            conv_fn (Callable): The convolution function to apply to the input tensors.
            real (Tensor): The real part of the input tensor.
            imag (Tensor): The imaginary part of the input tensor.
            pad_mode (str): The padding mode to use during convolution.
            padding (Tuple[int, ...]): The padding values for each dimension.
            stride (Tuple[int, ...]): The stride values for each convolution operation.
            dilation (Tuple[int, ...]): The dilation values for each dimension.
            group (int): The number of groups to split the input tensor for group convolution.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the real and imaginary parts of the output tensor after complex convolution.
        
        Raises:
            None.
        """
        out_rr = conv_fn(real, self.weight_x, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)
        out_ii = conv_fn(imag, self.weight_y, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)
        out_ri = conv_fn(real, self.weight_y, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)
        out_ir = conv_fn(imag, self.weight_x, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)

        out_r = out_rr - out_ii
        out_i = out_ri + out_ir

        return out_r, out_i


class _KaratsubaConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for complex numbers.

    Applies complex-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{C1} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})\\
        \text{C2} = \text{ccor}(\text{Im(kernel)}, \text{Im(inp)})\\
        \text{C3} = \text{ccor}(\text{Re(kernel)} + \text{Im(kernel)}, \text{Re(inp)} + \text{Im(inp)})\\
        \text{Re(out)} = C1 - C2\\
        \text{Im(out)} = C3 - C1 - C2,
        \end{align}

    where :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the complex-valued input tensors, :math:`\text{kernel}` is a complex weight matrix with the same
    data type as the :math:`inp` created by the layer, :math:`\text{Re(...)}` and :math:`\text{Im(...)}`
    are respectively real and imaginary parts of the complex-valued expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the complex convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **imag** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the imaginary part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the imaginary parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def forward(self,
                  conv_fn: Callable,
                  real: Tensor,
                  imag: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:
        r"""
        Constructs a complex convolution operation using the Karatsuba algorithm.
        
        Args:
            self (_KaratsubaConvImpl): The instance of the _KaratsubaConvImpl class.
            conv_fn (Callable): The convolution function to apply. It should accept input tensor, weight tensor, and various convolution parameters.
            real (Tensor): The real part of the input tensor.
            imag (Tensor): The imaginary part of the input tensor.
            pad_mode (str): The padding mode to use during convolution, e.g., 'valid' or 'same'.
            padding (Tuple[int, ...]): The padding values to apply to the input tensor.
            stride (Tuple[int, ...]): The stride values for the convolution operation.
            dilation (Tuple[int, ...]): The dilation values for the convolution operation.
            group (int): The number of groups to split the input channels and weights into.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing two tensors - the real and imaginary parts of the output tensor after the complex convolution operation.
        
        Raises:
            - ValueError: If the dimensions of the input tensors are not compatible for convolution.
            - TypeError: If the input parameters are of incorrect types.
            - RuntimeError: If there is an issue during the convolution operation.
        """
        c1 = conv_fn(real, self.weight_x, pad_mode=pad_mode, padding=padding,
                     stride=stride, dilation=dilation, group=group)
        c2 = conv_fn(imag, self.weight_y, pad_mode=pad_mode, padding=padding,
                     stride=stride, dilation=dilation, group=group)
        c3 = conv_fn(real + imag, self.weight_x + self.weight_y, pad_mode=pad_mode, padding=padding,
                     stride=stride, dilation=dilation, group=group)

        out_r = c1 - c2
        out_i = c3 - c1 - c2

        return out_r, out_i


class _ReImConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for complex numbers.

    Applies complex-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{inp_cat} = \text{cat}(\text{Re(inp)}, \text{Im(inp)}) \\
        \text{K1} = \text{cat}(\text{Re(kernel)}, \text{-Im(kernel)}) \\
        \text{K2} = \text{cat}(\text{Im(kernel)}, \text{Re(kernel)}) \\
        \text{Re(ccor)} = \text{ccor}(\text{K1}, \text{Re(inp_cat)}) \\
        \text{Im(ccor)} = \text{ccor}(\text{K2}, \text{Re(inp_cat)})
        \end{align}

    where :math:`ccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the complex-valued input tensors, :math:`\text{kernel}` is a complex weight matrix with the same
    data type as the :math:`inp` created by the layer, :math:`\text{cat}` is concatenation along the channel axis,
    :math:`\text{Re(...)}` and :math:`\text{Im(...)}` are respectively real and imaginary parts of the complex-valued
    expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and imaginary parts of the kernel.
        factory_kwargs (dict): Additional parameters, which must include data_format.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the complex convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **imag** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the imaginary part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the imaginary parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        r"""
        Initializes the _ReImConvImpl object.
        
        Args:
            self (object): The instance of the _ReImConvImpl class.
            weight_init (Union[Tensor, str, Initializer, numbers.Number]): The weight initialization method. It can be a Tensor, a string, an Initializer object, or a number.
            weight_shape (tuple): The shape of the weight.
            **factory_kwargs: Additional keyword arguments.
        
        Returns:
            None. This method does not return any value.
        
        Raises:
            ValueError: If the data format specified in the 'data_format' keyword argument is unsupported.
        """
        super().__init__(weight_init, weight_shape, **factory_kwargs)
        data_format = factory_kwargs.get('data_format', "nchw")
        self.c_idx = data_format.lower().find('c')
        if self.c_idx < 0:
            raise ValueError(f"Data format {data_format} is unsupported")

    def forward(self,
                  conv_fn: Callable,
                  real: Tensor,
                  imag: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:
        """
        Constructs the real and imaginary parts of the convolution operation.
        
        Args:
            self (object): The instance of the _ReImConvImpl class.
            conv_fn (Callable): The convolution function to be used, which should accept input, weight, and other convolution parameters.
            real (Tensor): The input tensor containing the real part of the complex data.
            imag (Tensor): The input tensor containing the imaginary part of the complex data.
            pad_mode (str): The padding mode to be used in the convolution operation.
            padding (Tuple[int, ...]): The padding to be applied to the input tensor. The length of the tuple should be compatible with the input tensor dimensions.
            stride (Tuple[int, ...]): The stride of the convolution operation along each dimension of the input tensor.
            dilation (Tuple[int, ...]): The dilation of the convolution operation along each dimension of the input tensor.
            group (int): The number of groups for grouped convolution. Should be set to 1 for standard convolution.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the real and imaginary parts of the convolution operation.
        
        Raises:
            (Any exceptions that the function may raise should be documented here.)
        """
        inp = P.concat([real, imag], axis=self.c_idx)
        weight_y_neg = P.neg(self.weight_y)
        w1 = P.concat([self.weight_x, weight_y_neg], axis=self.c_idx)
        w2 = P.concat([self.weight_y, self.weight_x], axis=self.c_idx)
        out_r = conv_fn(inp, w1, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        out_i = conv_fn(inp, w2, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        return out_r, out_i
