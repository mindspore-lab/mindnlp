# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MindSpore BiT model. Also supports backbone for ViT hybrid."""

import collections
import math
from typing import Optional, Tuple

import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, HeNormal

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from ...backbone_utils import BackboneMixin
from .configuration_bit import BitConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "BitConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/bit-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/bit-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"


def get_padding_value(padding=None, kernel_size=7, stride=1, dilation=1) -> Tuple[Tuple, bool]:
    r"""
    Utility function to get the tuple padding value given the kernel_size and padding.

    Args:
        padding (Union[`str`, `int`], *optional*):
            Padding value, can be either `"same"`, `"valid"`. If a different value is provided the default padding from
            PyTorch is used.
        kernel_size (`int`, *optional*, defaults to 7):
            Kernel size of the convolution layers.
        stride (`int`, *optional*, defaults to 1):
            Stride value of the convolution layers.
        dilation (`int`, *optional*, defaults to 1):
            Dilation value of the convolution layers.
    """
    dynamic = False
    if padding is None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        return padding, dynamic

    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0:
                # static case, no extra overhead
                padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding, dynamic


class WeightStandardizedConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Includes mindspore.TensorFlow compatible SAME padding. Used for ViT Hybrid model.

    Paper: [Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization](https://arxiv.org/abs/1903.10520v2)
    """
    def __init__(
        self,
        in_channel,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        eps=1e-6,
    ):
        """
        This method initializes an instance of the WeightStandardizedConv2d class.

        Args:
            self (object): The instance of the class.
            in_channel (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolutional kernel.
            stride (int, optional): The stride for the convolution operation. Default is 1.
            padding (int, optional): The padding to apply. Default is 0.
            dilation (int, optional): The dilation rate for the convolution operation. Default is 1.
            groups (int, optional): The number of groups for grouped convolution. Default is 1.
            bias (bool, optional): Whether to include bias in the convolution operation. Default is False.
            eps (float, optional): Small value to avoid division by zero. Default is 1e-06.

        Returns:
            None.

        Raises:
            TypeError: If any of the input parameters are of incorrect type.
            ValueError: If the values of parameters are out of expected range or invalid.
            RuntimeError: If an error occurs during the initialization process.
        """
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            pad_mode='pad' if padding != 0 else 'valid',
            padding=padding,
            dilation=dilation,
            group=groups,
            has_bias=bias,
        )
        if is_dynamic:
            self.pad = DynamicPad2d(kernel_size, stride, dilation)
        else:
            self.pad = None
        self.eps = eps

    def construct(self, hidden_state):
        """
        Constructs a weighted standardized convolutional operation.

        Args:
            self (WeightStandardizedConv2d): The instance of the WeightStandardizedConv2d class.
            hidden_state (tensor): The input tensor representing the hidden state.

        Returns:
            None: This method does not return any value directly, but modifies the hidden state tensor in place.

        Raises:
            None.
        """
        if self.pad is not None:
            hidden_state = self.pad(hidden_state)
        input_weight = self.weight.reshape(1, self.out_channels, -1)
        weight = ops.batch_norm(
            input_weight, ops.ones(self.out_channels), ops.zeros(self.out_channels),
                                ops.ones(self.out_channels), ops.zeros(self.out_channels),
                                training=True, momentum=0.0, eps=self.eps
        ).reshape_as(self.weight)
        hidden_state = ops.conv2d(
            hidden_state, weight, self.bias, self.stride, self.pad_mode, self.padding, self.dilation, self.group
        )
        return hidden_state


class BitGroupNormActivation(nn.GroupNorm):
    r"""
    A module that combines group normalization with an activation function.
    """
    def __init__(self, config, num_channels, eps=1e-5, affine=True, apply_activation=True):
        """
        Initializes an instance of the BitGroupNormActivation class.

        Args:
            self (BitGroupNormActivation): The instance of the class.
            config: The configuration object.
            num_channels (int): The number of input channels.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-05.
            affine (bool, optional): If True, applies learnable affine transformation. Defaults to True.
            apply_activation (bool, optional): If True, applies activation function specified in the configuration.
                If False, applies identity function. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super(BitGroupNormActivation, self).__init__(config.num_groups, num_channels, eps=eps, affine=affine)
        if apply_activation:
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = nn.Identity()

    def construct(self, hidden_state):
        """
        Constructs the hidden state of the BitGroupNormActivation.

        Args:
            self (BitGroupNormActivation): An instance of the BitGroupNormActivation class.
            hidden_state: The hidden state to be processed.
                It can be any valid input that can be processed by the _cal_output and activation methods.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_state = self._cal_output(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class DynamicPad2d(nn.Cell):
    r"""
    A module that wraps dynamic padding of any input, given the parameters of the convolutional layer and the input
    hidden states.
    """
    def __init__(self, kernel_size, stride, dilation, value=0):
        """Initializes an instance of the DynamicPad2d class.

        Args:
            self (DynamicPad2d): The current instance of the DynamicPad2d class.
            kernel_size (int or tuple): The size of the kernel used for padding.
                If an int is provided, it is converted to a tuple with the same value in both dimensions. (default: 0)
            stride (int or tuple): The stride used for padding.
                If an int is provided, it is converted to a tuple with the same value in both dimensions. (default: 0)
            dilation (int or tuple): The dilation used for padding.
                If an int is provided, it is converted to a tuple with the same value in both dimensions. (default: 0)
            value (int): The value used for padding. (default: 0)

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        # Safety checkers
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.value = value

        def compute_padding(x, kernel_size, stride, dilation):
            return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

        self.compute_padding = compute_padding

    def __call__(self, input):
        """
        This method is called when an instance of the DynamicPad2d class is used as a function.
        It performs dynamic padding on the input tensor based on the kernel size, stride, dilation, and value specified
        in the class instance.

        Args:
            self (DynamicPad2d): The instance of the DynamicPad2d class.
            input (tensor): The input tensor to be dynamically padded.
                It should be a tensor with shape [batch_size, channels, height, width].

        Returns:
            None: This method does not return any value explicitly.
                However, it modifies the input tensor in place by applying dynamic padding.

        Raises:
            ValueError: If the input tensor does not have the expected shape [batch_size, channels, height, width].
            RuntimeError: If an error occurs during the dynamic padding process.
        """
        # Get width and height
        input_height, input_width = input.shape[-2:]

        # Compute the padding values
        padding_height = self.compute_padding(input_height, self.kernel_size[0], self.stride[0], self.dilation[0])
        padding_width = self.compute_padding(input_width, self.kernel_size[1], self.stride[1], self.dilation[1])

        # apply pad
        if padding_height > 0 or padding_width > 0:
            input = ops.pad(
                input,
                [
                    padding_width // 2,
                    padding_width - padding_width // 2,
                    padding_height // 2,
                    padding_height - padding_height // 2,
                ],
                value=self.value,
            )
        return input


class BitMaxPool2d(nn.Cell):
    """Tensorflow like 'SAME' wrapper for 2D max pooling"""
    def __init__(
        self,
        kernel_size: int,
        stride=None,
        dilation=1,
        ceil_mode=False,
        padding=(0, 0),
        padding_value=0,
        use_dynamic_padding=True,
    ):
        """
        Initializes a BitMaxPool2d object.

        Args:
            self (BitMaxPool2d): The BitMaxPool2d instance.
            kernel_size (int): The size of the sliding window kernel. Can be a single integer or a tuple of two integers.
            stride (int, optional): The stride of the sliding window kernel.
                Can be a single integer or a tuple of two integers. Defaults to None.
            dilation (int, optional): The dilation rate of the sliding window kernel.
                Can be a single integer or a tuple of two integers. Defaults to 1.
            ceil_mode (bool, optional): Whether to use ceil mode for the output size calculation.
                Defaults to False.
            padding (tuple, optional): The padding to be applied to the input.
                Can be a tuple of two integers or a single integer. Defaults to (0, 0).
            padding_value (int, optional): The value used for padding if `use_dynamic_padding` is True. Defaults to 0.
            use_dynamic_padding (bool, optional): Whether to apply dynamic padding using `DynamicPad2d` or not.
                Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size if isinstance(kernel_size, collections.abc.Iterable) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        self.dilation = dilation if isinstance(dilation, collections.abc.Iterable) else (dilation, dilation)
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.pad_mode = 'pad' if sum(padding) > 0 else 'valid'
        if use_dynamic_padding:
            self.pad = DynamicPad2d(kernel_size, stride, dilation, padding_value)
        else:
            self.pad = nn.Identity()

    def construct(self, hidden_states):
        """
        Constructs a BitMaxPool2d object.

        This method takes two parameters: self and hidden_states.

        Args:
            self (BitMaxPool2d): The current instance of the BitMaxPool2d class.
            hidden_states (Tensor): The input tensor of shape (batch_size, channels, height, width)
                representing the hidden states.

        Returns:
            None.

        Raises:
            ValueError: If the input tensor `hidden_states` is not a 4-dimensional tensor.
            ValueError: If the input tensor `hidden_states` has a negative height or width.
            ValueError: If the kernel size is not a positive integer or a tuple of two positive integers.
            ValueError: If the stride is not a positive integer or a tuple of two positive integers.
            ValueError: If the padding is not a tuple of two non-negative integers.
            ValueError: If the dilation is not a positive integer or a tuple of two positive integers.
            TypeError: If the ceil_mode is not a boolean value.
        """
        hidden_states = self.pad(hidden_states)
        return ops.max_pool2d(
            hidden_states, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode
        )


class BitEmbeddings(nn.Cell):
    """
    BiT Embeddings (stem) composed of a single aggressive convolution.
    """
    def __init__(self, config: BitConfig):
        """
        Initializes an instance of the BitEmbeddings class.

        Args:
            self: The instance of the class.
            config (BitConfig): A BitConfig object containing configuration parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.convolution = WeightStandardizedConv2d(
            config.num_channels,
            config.embedding_size,
            kernel_size=7,
            stride=2,
            eps=1e-8,
            padding=config.global_padding,
        )

        self.pooler = BitMaxPool2d(kernel_size=3, stride=2, use_dynamic_padding=config.embedding_dynamic_padding)

        # Use the same padding strategy as convolutional layers
        if config.global_padding is not None and config.global_padding.upper() == "SAME":
            self.pad = nn.Identity()
        else:
            self.pad = nn.ConstantPad2d(padding=(1, 1, 1, 1), value=0.0)

        if not config.layer_type == "preactivation":
            self.norm = BitGroupNormActivation(config, num_channels=config.embedding_size)
        else:
            self.norm = nn.Identity()

        self.num_channels = config.num_channels

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the bit embeddings for the given pixel values.

        Args:
            self (BitEmbeddings): An instance of the BitEmbeddings class.
            pixel_values (mindspore.Tensor):
                A tensor containing pixel values.
                The pixel_values parameter should be of shape (batch_size, num_channels, height, width).

                - batch_size (int): The number of images in the batch.
                - num_channels (int): The number of channels in each image.
                - height (int): The height of each image.
                - width (int): The width of each image.

                The pixel_values should have the same number of channels as the configuration set in the class.
                Raises a ValueError if the number of channels does not match.

        Returns:
            mindspore.Tensor: The constructed bit embeddings tensor.

        Raises:
            ValueError: If the channel dimension of the pixel values does not match with the one set in the configuration.
        """
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embedding = self.convolution(pixel_values)

        embedding = self.pad(embedding)

        embedding = self.norm(embedding)

        embedding = self.pooler(embedding)

        return embedding


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input: mindspore.Tensor, drop_prob: float = 0.0, training: bool = False) -> mindspore.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ops.rand(shape, dtype=input.dtype)
    random_tensor = random_tensor.floor()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->Bit
class BitDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initializes a new instance of the BitDropPath class.

        Args:
            self: The instance of the BitDropPath class.
            drop_prob (Optional[float]): The probability of dropping a bit. If not provided, the default value is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a new tensor by applying drop path regularization to the given hidden states.

        Args:
            self (BitDropPath): An instance of the BitDropPath class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, ..., hidden_size).
                It represents the hidden states of a neural network layer.

        Returns:
            mindspore.Tensor: A tensor of the same shape and dtype as the input tensor.
                It contains the modified hidden states after applying drop path regularization.

        Raises:
            TypeError: If the input tensor is not an instance of mindspore.Tensor.
            ValueError: If the shape of the input tensor is invalid or incompatible with the neural network layer.
            RuntimeError: If the drop path regularization is attempted during inference mode.

        Note:
            Drop path regularization randomly sets a fraction of the hidden unit activations to zero during training,
            which helps in regularization and prevents overfitting to the training data. The drop probability
            is specified by the 'drop_prob' attribute of the BitDropPath instance.

        Example:
            ```python
            >>> drop_path = BitDropPath()
            >>> hidden_states = mindspore.Tensor(np.random.randn(32, 64, 256), mindspore.float32)
            >>> output = drop_path.construct(hidden_states)
            ```
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        Returns a string representation of the BitDropPath object.

        Args:
            self (BitDropPath): The BitDropPath object itself.

        Returns:
            str: A string representation of the BitDropPath object, containing the probability of dropping a bit.

        Raises:
            None.

        This method returns a string representation of the BitDropPath object, specifically the probability of
        dropping a bit.
        The returned string is formatted as 'p=drop_prob', where 'drop_prob' is the probability of dropping a bit.
        """
        return "p={}".format(self.drop_prob)


def make_div(value, divisor=8):
    """
    Args:
        value (int): The input value for which the division needs to be performed.
        divisor (int, optional): The divisor used for division. Defaults to 8.

    Returns:
        int: The new value after performing the division operation.

    Raises:
        None
    """
    min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class BitPreActivationBottleneckLayer(nn.Cell):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """
    def __init__(
        self,
        config,
        in_channels,
        out_channels=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        drop_path_rate=0.0,
        is_first_layer=False,
    ):
        """
        Initializes a BitPreActivationBottleneckLayer instance.

        Args:
            self: The instance of the class.
            config: A configuration object containing settings for the layer.
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. If not provided, defaults to in_channels.
            bottle_ratio (float): Ratio used for bottleneck layer compression.
            stride (int): Stride value for convolution operations.
            dilation (int): Dilation rate for convolution operations.
            first_dilation (int, optional): Dilation rate for the first convolution layer.
                Defaults to dilation if not provided.
            groups (int): Number of groups for grouped convolutions.
            drop_path_rate (float): Probability of applying drop path regularization.
            is_first_layer (bool): Flag indicating if the layer is the first in the network.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        first_dilation = first_dilation or dilation

        out_channels = out_channels or in_channels
        mid_channels = make_div(out_channels * bottle_ratio)

        if is_first_layer:
            self.downsample = BitDownsampleConv(
                config,
                in_channels,
                out_channels,
                stride=stride,
                preact=True,
            )
        else:
            self.downsample = None

        self.norm1 = BitGroupNormActivation(config, in_channels)
        self.conv1 = WeightStandardizedConv2d(in_channels, mid_channels, 1, eps=1e-8, padding=config.global_padding)

        self.norm2 = BitGroupNormActivation(config, num_channels=mid_channels)
        self.conv2 = WeightStandardizedConv2d(
            mid_channels, mid_channels, 3, stride=stride, groups=groups, eps=1e-8, padding=config.global_padding
        )

        self.norm3 = BitGroupNormActivation(config, mid_channels)
        self.conv3 = WeightStandardizedConv2d(mid_channels, out_channels, 1, eps=1e-8, padding=config.global_padding)

        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def construct(self, hidden_states):
        """
        The 'construct' method initializes the BitPreActivationBottleneckLayer class.

        Args:
            self (object): The instance of the BitPreActivationBottleneckLayer class.
            hidden_states (tensor): A tensor representing the hidden states.

        Returns:
            None.

        Raises:
            ValueError: If the downsample operation encounters an issue.
            RuntimeError: If the convolutions encounter runtime issues.
            TypeError: If the input parameters are of incorrect types.
        """
        hidden_states_preact = self.norm1(hidden_states)

        # shortcut branch
        shortcut = hidden_states
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states_preact)

        # residual branch
        hidden_states = self.conv1(hidden_states_preact)
        hidden_states = self.conv2(self.norm2(hidden_states))
        hidden_states = self.conv3(self.norm3(hidden_states))
        hidden_states = self.drop_path(hidden_states)
        return hidden_states + shortcut


class BitBottleneckLayer(nn.Cell):
    """Non Pre-activation bottleneck block, equivalent to V1.5/V1b bottleneck. Used for ViT Hybrid."""
    def __init__(
        self,
        config,
        in_channels,
        out_channels=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        drop_path_rate=0.0,
        is_first_layer=False,
    ):
        """
        Initializes a BitBottleneckLayer object.

        Args:
            self: The BitBottleneckLayer object being initialized.
            config: An object containing configuration parameters.
            in_channels (int): The number of input channels.
            out_channels (int, optional): The number of output channels. If not provided, it defaults to the value of in_channels.
            bottle_ratio (float): The ratio of the bottleneck width to the output channels.
            stride (int): The stride value for the convolutional layers.
            dilation (int): The dilation value for the middle convolutional layer.
            first_dilation (int, optional): The dilation value for the first convolutional layer. If not provided, it defaults to the value of dilation.
            groups (int): The number of groups for the middle convolutional layer.
            drop_path_rate (float): The dropout rate for the drop path layer.
            is_first_layer (bool): Indicates if this is the first layer of the network.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        first_dilation = first_dilation or dilation

        out_channels = out_channels or in_channels
        mid_chs = make_div(out_channels * bottle_ratio)

        if is_first_layer:
            self.downsample = BitDownsampleConv(
                config,
                in_channels,
                out_channels,
                stride=stride,
                preact=False,
            )
        else:
            self.downsample = None

        self.conv1 = WeightStandardizedConv2d(in_channels, mid_chs, 1, eps=1e-8, padding=config.global_padding)
        self.norm1 = BitGroupNormActivation(config, num_channels=mid_chs)
        self.conv2 = WeightStandardizedConv2d(
            mid_chs,
            mid_chs,
            3,
            stride=stride,
            dilation=first_dilation,
            groups=groups,
            eps=1e-8,
            padding=config.global_padding,
        )
        self.norm2 = BitGroupNormActivation(config, num_channels=mid_chs)
        self.conv3 = WeightStandardizedConv2d(mid_chs, out_channels, 1, eps=1e-8, padding=config.global_padding)
        self.norm3 = BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)
        self.drop_path = BitDropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = ACT2FN[config.hidden_act]

    def construct(self, hidden_states):
        """
        The 'construct' method in the class 'BitBottleneckLayer' performs a series of operations on
        the input 'hidden_states' to construct a new hidden state and returns the result.

        Args:
            self: The instance of the BitBottleneckLayer class.
            hidden_states (Tensor): The input hidden states on which the method operates.
                It is of type Tensor and represents the intermediate hidden states of the model.
                There are no specific restrictions on the input.

        Returns:
            Tensor: The updated hidden states after the operations have been performed.
                It is of type Tensor and represents the modified hidden states.

        Raises:
            None
        """
        # shortcut branch
        shortcut = hidden_states
        if self.downsample is not None:
            shortcut = self.downsample(hidden_states)

        # residual
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.conv3(hidden_states)
        hidden_states = self.norm3(hidden_states)

        hidden_states = self.drop_path(hidden_states)
        hidden_states = self.activation(hidden_states + shortcut)
        return hidden_states


class BitDownsampleConv(nn.Cell):

    """
    This class represents a BitDownsampleConv module in a neural network. It is a subclass of nn.Cell.

    BitDownsampleConv applies down-sampling to the input tensor using a combination of weight-standardized convolution
    and bit group normalization activation.

    Attributes:
        conv (WeightStandardizedConv2d): An instance of the WeightStandardizedConv2d class that performs
            a weight-standardized convolution operation on the input tensor.
        norm (nn.Identity or BitGroupNormActivation): An instance of either nn.Identity or BitGroupNormActivation class,
            depending on the value of the preact parameter. If preact is True, nn.Identity is used, otherwise
            BitGroupNormActivation is used for applying bit group normalization activation.

    Methods:
        __init__(self, config, in_channels, out_channels, stride=1, preact=True):
            Initializes a BitDownsampleConv instance with the specified parameters.

            Args:

            - config (Config): The configuration object containing various settings.

                - in_channels (int): The number of input channels.
                - out_channels (int): The number of output channels.
                - stride (int, optional): The stride value for the convolution operation. Defaults to 1.
                - preact (bool, optional): If True, nn.Identity is used for normalization,
                otherwise BitGroupNormActivation is used. Defaults to True.

        construct(self, x):
            Applies down-sampling to the input tensor x by performing weight-standardized convolution followed by normalization.

            Args:

            - x (Tensor): The input tensor to be down-sampled.

            Returns:

            - Tensor: The down-sampled output tensor.
    """
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        stride=1,
        preact=True,
    ):
        """
        Initializes an instance of the BitDownsampleConv class.

        Args:
            self (BitDownsampleConv): The instance of the class.
            config: The configuration object containing various settings.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            stride (int): The stride for the convolution operation. Default is 1.
            preact (bool): Indicates whether to apply preactivation. Default is True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.conv = WeightStandardizedConv2d(
            in_channels, out_channels, 1, stride=stride, eps=1e-8, padding=config.global_padding
        )
        self.norm = (
            nn.Identity()
            if preact
            else BitGroupNormActivation(config, num_channels=out_channels, apply_activation=False)
        )

    def construct(self, x):
        """
        Constructs the BitDownsampleConv object.

        Args:
            self (BitDownsampleConv): The instance of the BitDownsampleConv class.
            x (any): The input data to be processed. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        return self.norm(self.conv(x))


class BitStage(nn.Cell):
    """
    A ResNet v2 stage composed by stacked layers.
    """
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        stride,
        dilation,
        depth,
        bottle_ratio=0.25,
        layer_dropout=None,
    ):
        """
        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the hyperparameters for the network.
            in_channels (int): The number of input channels for the layer.
            out_channels (int): The number of output channels for the layer.
            stride (int): The value specifying the stride for the convolution operation.
            dilation (int): The value specifying the dilation for the convolution operation.
            depth (int): The depth of the layer.
            bottle_ratio (float, optional): The ratio of bottleneck channels to the output channels. Default is 0.25.
            layer_dropout (float, None): The dropout rate for the layer. If None, no dropout is applied.

        Returns:
            None.

        Raises:
            NotImplementedError: If the layer type specified in the config is not supported.
            ValueError: If the dilation value is not 1 or 2.
        """
        super().__init__()

        first_dilation = 1 if dilation in (1, 2) else 2

        # Get the layer type
        if config.layer_type == "bottleneck":
            layer_cls = BitBottleneckLayer
        else:
            layer_cls = BitPreActivationBottleneckLayer

        prev_chs = in_channels
        self.layers = nn.SequentialCell()
        for layer_idx in range(depth):
            # Get the current hyper-parameters
            stride, drop_path_rate, is_first_layer = self._get_updated_hyperparameters(
                layer_idx, stride, layer_dropout
            )

            self.layers.append(
                layer_cls(
                    config,
                    prev_chs,
                    out_channels,
                    stride=stride,
                    dilation=dilation,
                    bottle_ratio=bottle_ratio,
                    first_dilation=first_dilation,
                    drop_path_rate=drop_path_rate,
                    is_first_layer=is_first_layer,
                ),
            )
            prev_chs = out_channels
            first_dilation = dilation

    def _get_updated_hyperparameters(self, layer_idx, stride, layer_dropout):
        r"""
        Get the new hyper-parameters with respect to the previous ones and the index of the current layer.
        """
        if layer_dropout:
            drop_path_rate = layer_dropout[layer_idx]
        else:
            drop_path_rate = 0.0

        if layer_idx != 0:
            stride = 1

        is_first_layer = layer_idx == 0

        return stride, drop_path_rate, is_first_layer

    def construct(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct method in the BitStage class.

        Args:
            self: BitStage instance.
                The instance of the BitStage class.
            input: mindspore.Tensor
                The input tensor to be processed.

        Returns:
            mindspore.Tensor
                The processed tensor after going through the BitStage layers.

        Raises:
            None
        """
        hidden_state = input
        for _, layer in enumerate(self.layers):
            hidden_state = layer(hidden_state)
        return hidden_state


class BitEncoder(nn.Cell):

    """
    The `BitEncoder` class is a subclass of `nn.Cell` and represents an encoder module for the Bit model.
    It is responsible for encoding the input hidden state through a series of stages.

    Attributes:
        stages (nn.CellList): A list of BitStage instances representing each stage of the encoder.


    Methods:
        __init__:
            Initializes a new instance of the `BitEncoder` class.

        _get_updated_hyperparameters:
            Calculates and returns the updated hyperparameters for the given stage.

        construct:
            Constructs the encoder module by iterating through each stage and applying them to the input hidden state.

    """
    def __init__(self, config: BitConfig):
        """
        Initializes an instance of the BitEncoder class.

        Args:
            self: The BitEncoder instance.
            config (BitConfig): The configuration object that specifies the hyperparameters for the BitEncoder.
                The config parameter must be an instance of the BitConfig class and should contain the following attributes:

                - embedding_size (int): The size of the input embeddings.
                - depths (list[int]): A list of integers representing the depths of each BitStage.
                - hidden_sizes (list[int]): A list of integers representing the hidden sizes of each BitStage.
                - drop_path_rate (float): The drop path rate for the BitStages.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.stages = nn.CellList([])

        prev_chs = config.embedding_size

        # These needs to stay hardcoded
        current_stride = 4
        dilation = 1

        layer_dropouts = [
            x.tolist()
            for x in mindspore.Tensor(np.linspace(0, config.drop_path_rate, sum(config.depths))).split(config.depths)
        ]

        for stage_idx, (current_depth, current_hidden_size, layer_dropout) in enumerate(
            zip(config.depths, config.hidden_sizes, layer_dropouts)
        ):
            # Get the updated hyper params
            out_channels, stride, dilation = self._get_updated_hyperparameters(
                stage_idx, current_stride, current_hidden_size, dilation, config
            )

            stage = BitStage(
                config,
                prev_chs,
                out_channels,
                stride=stride,
                dilation=dilation,
                depth=current_depth,
                layer_dropout=layer_dropout,
            )

            prev_chs = out_channels
            current_stride *= stride

            setattr(self.stages, str(stage_idx), stage)

    def _get_updated_hyperparameters(self, stage_idx, current_stride, current_hidden_size, dilation, config):
        """
        This method '_get_updated_hyperparameters' updates the hyperparameters based on the given parameters.

        Args:
            self (object): The instance of the BitEncoder class.
            stage_idx (int): The index of the current stage. It is used to determine the stride value.
            current_stride (int): The current stride value used for calculations.
            current_hidden_size (int): The current hidden size value used for calculations.
            dilation (int): The dilation value used for calculations.
            config (object): The configuration object containing width factor and output stride values.

        Returns:
            None.

        Raises:
            None.
        """
        out_channels = make_div(current_hidden_size * config.width_factor)
        stride = 1 if stage_idx == 0 else 2
        if current_stride >= config.output_stride:
            dilation *= stride
            stride = 1
        return out_channels, stride, dilation

    def construct(
        self, hidden_state: mindspore.Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        """
        Constructs the BitEncoder model.

        Args:
            self (BitEncoder): An instance of the BitEncoder class.
            hidden_state (mindspore.Tensor): The initial hidden state tensor.
            output_hidden_states (bool, optional): Whether to output hidden states at each stage.
                Defaults to False.
            return_dict (bool, optional): Whether to return the output as a dictionary.
                Defaults to True.

        Returns:
            BaseModelOutputWithNoAttention: An instance of the BaseModelOutputWithNoAttention class
                containing the last hidden state and the hidden states at each stage.

        Raises:
            None.

        """
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class BitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BitConfig
    base_model_prefix = "bit"
    main_input_name = "pixel_values"

    def _init_weights(self, cell):
        """
        This method initializes the weights of the given cell based on its type.

        Args:
            self: The instance of the BitPreTrainedModel class.
            cell: An instance of a neural network cell (e.g., nn.Conv2d, nn.BatchNorm2d, nn.GroupNorm).
                It represents the cell for which the weights are initialized.

        Returns:
            None.

        Raises:
            TypeError: If the 'cell' parameter is not an instance of nn.Conv2d, nn.BatchNorm2d, or nn.GroupNorm.
            ValueError: If the 'cell' parameter is provided with an unsupported type.
            RuntimeError: If the weight initialization fails due to any runtime issues.
        """
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(initializer(HeNormal(), cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class BitModel(BitPreTrainedModel):

    """
    The BitModel class represents a model for processing pixel values using Bit embeddings and encoding techniques.
    It inherits from the BitPreTrainedModel and includes methods for initialization and
    constructing the model output with pooling and no attention.

    Attributes:
        config: The configuration for the model.
        embedder: Instance of BitEmbeddings for embedding the input pixel values.
        encoder: Instance of BitEncoder for encoding the embedded values.
        norm: Instance of BitGroupNormActivation for applying normalization to the hidden state.
        pooler: Instance of nn.AdaptiveAvgPool2d for pooling the last hidden state.

    Methods:
        __init__(self, config): Initializes the BitModel with the provided configuration.
        construct(self, pixel_values, output_hidden_states, return_dict): Constructs the model output with pooling
            and no attention based on the input pixel values and optional flags for outputting hidden states and
            using a return dictionary.
    """
    def __init__(self, config):
        """Initializes a BitModel instance.

        Args:
            self (BitModel): An instance of the BitModel class.
            config (object): A configuration object containing various settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embedder = BitEmbeddings(config)

        self.encoder = BitEncoder(config)
        self.norm = (
            BitGroupNormActivation(config, num_channels=config.hidden_sizes[-1])
            if config.layer_type == "preactivation"
            else nn.Identity()
        )

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self, pixel_values: mindspore.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        """
        Constructs the BitModel by processing the given pixel values.

        Args:
            self: The instance of the BitModel class.
            pixel_values (mindspore.Tensor): The input tensor containing pixel values.
            output_hidden_states (bool, optional): Whether to include the hidden states in the output. Defaults to None.
            return_dict (bool, optional): Whether to return the output as a dictionary. Defaults to None.

        Returns:
            BaseModelOutputWithPoolingAndNoAttention: An object containing the constructed BitModel output,
                including the last hidden state, pooled output, and hidden states.

        Raises:
            None.

        Note:
            - The `output_hidden_states` parameter,
            if provided, overrides the `output_hidden_states` configuration of the BitModel instance.
            - The `return_dict` parameter,
            if provided, overrides the `use_return_dict` configuration of the BitModel instance.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        last_hidden_state = self.norm(last_hidden_state)

        pooled_output = self.pooler(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class BitForImageClassification(BitPreTrainedModel):

    """
    BitForImageClassification is a class that represents a model for image classification using a Bit (Big Transfer) architecture.
    It inherits from BitPreTrainedModel and provides functionalities for image classification tasks.

    Attributes:
        num_labels (int): The number of labels for classification.
        bit (BitModel): BitModel instance for feature extraction.
        classifier (nn.SequentialCell): Neural network layers for classification.

    Methods:
        __init__:
            Initializes the BitForImageClassification instance with the given configuration.

        construct:
            Constructs the image classifier model with optional inputs and returns the output with or without attention.

            Parameters:

            - pixel_values (mindspore.Tensor): Tensor of shape `(batch_size, channels, height, width)` representing input images.
            - labels (mindspore.Tensor): Tensor of shape `(batch_size,)` representing labels for classification/regression.
                Indices should be in `[0, ..., config.num_labels - 1]`. For classification, a classification loss is computed (Cross-Entropy).
            - output_hidden_states (bool): Flag to indicate whether to output hidden states.
            - return_dict (bool): Flag to specify the format of the returned output.

        Returns:
            ImageClassifierOutputWithNoAttention: Output containing loss, logits, and hidden states if specified.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BitForImageClassification class.

        Args:
            self (BitForImageClassification): The current instance of the BitForImageClassification class.
            config: The configuration object containing various settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bit = BitModel(config)
        # classification head
        self.classifier = nn.SequentialCell(
            nn.Flatten(),
            nn.Dense(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)


class BitBackbone(BitPreTrainedModel, BackboneMixin):

    """
    A BitBackbone class represents the backbone of a Bit model, which is a pre-trained image classification model.

    This class inherits from the BitPreTrainedModel and BackboneMixin classes.

    The BitBackbone class has the following methods:

    - __init__(self, config): Initializes the BitBackbone instance with the provided configuration.
    - construct(self, pixel_values, output_hidden_states, return_dict): Constructs the backbone model and returns the feature maps and hidden states.

    Example:
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> from PIL import Image
        >>> import requests
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ...
        >>> processor = AutoImageProcessor.from_pretrained("google/resnetnv2-50")
        >>> model = AutoBackbone.from_pretrained("google/resnetnv2-50")
        ...
        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```

    Note:
        In the above example, the BitBackbone class is used to extract feature maps and hidden states from an image using a pre-trained Bit model.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BitBackbone class.

        Args:
            self: The instance of the BitBackbone class.
            config:
                A configuration object containing the settings for the BitBackbone model.
                It should be an instance of the Config class and contain the following attributes:

                - embedding_size (int): The size of the input embedding.
                - hidden_sizes (list): A list of integers representing the sizes of hidden layers.
                
        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        super()._init_backbone(config)

        self.bit = BitModel(config)
        self.num_features = [config.embedding_size] + config.hidden_sizes

        # initialize weights and apply final processing
        self.post_init()

    def construct(
        self, pixel_values: mindspore.Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        Returns:
            BackboneOutput

        Example:
            ```python
            >>> from transformers import AutoImageProcessor, AutoBackbone
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> processor = AutoImageProcessor.from_pretrained("google/resnetnv2-50")
            >>> model = AutoBackbone.from_pretrained("google/resnetnv2-50")
            ...
            >>> inputs = processor(image, return_tensors="pt")
            >>> outputs = model(**inputs)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.bit(pixel_values, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )

__all__ = [
    "BitForImageClassification",
    "BitModel",
    "BitPreTrainedModel",
    "BitBackbone",
]
