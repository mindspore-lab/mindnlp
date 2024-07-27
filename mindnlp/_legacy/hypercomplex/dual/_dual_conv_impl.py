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
"""Dual Convolution Implementation"""
import numbers
from typing import Callable, Tuple, Union

from mindspore.common.tensor import Tensor
from mindspore.common.initializer import Initializer
from mindspore import ops as P

from ..hypercomplex._hc_conv_impl import _BaseConvImpl as BaseConvImpl

class _ConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for dual numbers.

    Applies dual-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})\\
        \text{Du(ccor)} = \text{ccor}(\text{Du(kernel)}, \text{Re(inp)})
        + \text{ccor}(\text{Re(kernel)}, \text{Du(inp)}),
        \end{align}

    where and :math:`cccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer. :math:`\text{Re(...)}` and :math:`\text{Du(...)}`
    are respectively real and dual parts of the dual-valued expression inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and dual parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the dual convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **dual** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the dual part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the dual parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def forward(self,
                  conv_fn: Callable,
                  real: Tensor,
                  dual: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:
        """
        Constructs the convolution operation for the given real and dual tensors.
        
        Args:
            self: The instance of the _ConvImpl class.
            conv_fn (Callable): The convolution function to apply.
            real (Tensor): The input tensor for the real part of the operation.
            dual (Tensor): The input tensor for the dual part of the operation.
            pad_mode (str): The padding mode to use for the convolution operation.
            padding (Tuple[int, ...]): The padding to apply to the input tensors.
            stride (Tuple[int, ...]): The stride to apply to the input tensors.
            dilation (Tuple[int, ...]): The dilation to apply to the input tensors.
            group (int): The number of groups for grouped convolution.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the output tensors for the real and dual parts of the operation.
        
        Raises:
            - Any exceptions raised by the convolution function 'conv_fn' when applied to the input tensors 'real' and 'dual'.
        """
        out_r = conv_fn(real, self.weight_x, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        out_rd = conv_fn(real, self.weight_y, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)
        out_dr = conv_fn(dual, self.weight_x, pad_mode=pad_mode, padding=padding,
                         stride=stride, dilation=dilation, group=group)

        out_d = out_rd + out_dr
        return out_r, out_d


class _ReDuConvImpl(BaseConvImpl):
    r"""
    The implementor class of the convolution layer for dual numbers.

    Applies dual-valued convolution transformation. This layer implements the operation as:

    .. math::
        \begin{align}
        \text{inp_cat} = \text{cat}(\text{Re(inp)}, \text{Du(inp)}) \\
        \text{K} = \text{cat}(\text{Du(kernel)}, \text{Re(kernel)}) \\
        \text{Re(ccor)} = \text{ccor}(\text{Re(kernel)}, \text{Re(inp)})\\
        \text{Du(ccor)} = \text{ccor}(\text{K}, \text{Re(inp_cat)})
        \end{align}

    where and :math:`cccor` is the real-valued `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`inp` is the dual input tensors, :math:`\text{kernel}` is a dual weight matrix with the same
    data type as the :math:`inp` created by the layer, :math:`\text{cat}` is concatenation along the channel axis.
    :math:`\text{Re(...)}` and :math:`\text{Du(...)}` are respectively real and dual parts of the dual-valued expression
    inside the parentheses.

    Args:
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `inp`. The values of str refer to the function `initializer`. Default: 'normal'.
        weight_shape (tuple): The set of int numbers that defines the shape of real and dual parts of the kernel.

    Inputs:
        - **conv_op** (Callable) - the function of the real-valued convolution to be used for decomposition
          of the dual convolution transformation. For example, mindspore.ops.operations.Conv2D(...) may be passed
        - **real** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the real part of the input.
        - **dual** (Tensor) - Tensor of shape :math:`(N, C_{in}, *, ..., *)` or :math:`(N, *, ..., *, C_{in})`,
          which defines the dual part of the input.

    Outputs:
        Tuple of two tensors, each of shape :math:`(N, C_{out}, *, ..., *)` or :math:`(N, *, ..., *, C_{out})`,
        which represents the real and the dual parts of the output.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    def __init__(self,
                 weight_init: Union[Tensor, str, Initializer, numbers.Number],
                 weight_shape: tuple,
                 **factory_kwargs) -> None:
        r"""Initializes the _ReDuConvImpl class.
        
        Args:
            weight_init (Union[Tensor, str, Initializer, numbers.Number]): The weight initialization for the convolution layer. It can be a Tensor, a string representing the weight initialization method, an
Initializer object, or a number. 
            weight_shape (tuple): The shape of the weight tensor for the convolution layer.
            **factory_kwargs: Additional keyword arguments to configure the convolution layer.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: Raised if the data format specified in the factory_kwargs is unsupported.
        """
        super().__init__(weight_init, weight_shape, **factory_kwargs)
        data_format = factory_kwargs.get('data_format', 'nchw')
        self.c_idx = data_format.lower().find('c')
        if self.c_idx < 0:
            raise ValueError(f"Data format {data_format} is unsupported")

    def forward(self,
                  conv_fn: Callable,
                  real: Tensor,
                  dual: Tensor,
                  pad_mode: str,
                  padding: Tuple[int, ...],
                  stride: Tuple[int, ...],
                  dilation: Tuple[int, ...],
                  group: int) -> Tuple[Tensor, Tensor]:
        r"""
        Constructs the ReDuConv operation on the given input tensors.
        
        Args:
            self (_ReDuConvImpl): An instance of the _ReDuConvImpl class.
            conv_fn (Callable): The convolution function to use.
            real (Tensor): The real input tensor.
            dual (Tensor): The dual input tensor.
            pad_mode (str): The padding mode to use during convolution. Valid values are 'valid', 'same', or 'full'.
            padding (Tuple[int, ...]): The padding values for each dimension of the input tensor.
            stride (Tuple[int, ...]): The stride values for each dimension of the input tensor.
            dilation (Tuple[int, ...]): The dilation values for each dimension of the input tensor.
            group (int): The group size for group convolution.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the real and dual output tensors.
        
        Raises:
            None.
        
        Note:
            The 'conv_fn' parameter should be a callable that performs convolution operation on the given tensors. 
            The 'real' and 'dual' tensors should have the same number of dimensions and sizes.
            The 'pad_mode' parameter determines how the input tensors are padded before convolution. 
            The 'padding', 'stride', and 'dilation' parameters specify the respective values for each dimension of the tensors.
            The 'group' parameter defines the group size for group convolution operation.
        
        Example:
            >>> conv_fn = torch.nn.functional.conv2d
            >>> real_tensor = torch.randn(1, 3, 32, 32)
            >>> dual_tensor = torch.randn(1, 3, 32, 32)
            >>> pad_mode = 'same'
            >>> padding = (1, 1)
            >>> stride = (1, 1)
            >>> dilation = (1, 1)
            >>> group = 1
            >>> _ReDuConvImpl.forward(self, conv_fn, real_tensor, dual_tensor, pad_mode, padding, stride, dilation, group)
            (tensor([...]), tensor([...]))
        """
        out_r = conv_fn(real, self.weight_x, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        inp = P.concat([real, dual], axis=self.c_idx)
        w = P.concat([self.weight_y, self.weight_x], axis=self.c_idx)
        out_d = conv_fn(inp, w, pad_mode=pad_mode, padding=padding,
                        stride=stride, dilation=dilation, group=group)
        return out_r, out_d
