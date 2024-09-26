# coding=utf-8
# Copyright 2022 BNRist (Tsinghua University), TKLNDST (Nankai University) and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore Visual Attention Network (VAN) model."""

import math
from collections import OrderedDict
from typing import Optional, Tuple, Union

import mindspore
from mindnlp.core.nn import Parameter

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from .configuration_van import VanConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "VanConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "Visual-Attention-Network/van-base"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "Visual-Attention-Network/van-base"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


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


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath with ConvNext->Van
class VanDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initialize a new instance of the VanDropPath class.
        
        Args:
            self: The instance of the VanDropPath class.
            drop_prob (Optional[float]): The probability of dropping a path during training. 
                If set to None, no paths will be dropped. Should be a float value between 0 and 1, inclusive.
        
        Returns:
            None.
        
        Raises:
            None
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a new tensor by applying drop_path operation to the given hidden states.
        
        Args:
            self (VanDropPath): An instance of the VanDropPath class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
        
        Returns:
            mindspore.Tensor: A tensor representing the output of the drop_path operation.
        
        Raises:
            None.
        
        Note:
            The drop_path operation randomly sets a fraction of the hidden states to zero during training.
            This helps in regularizing the model and preventing overfitting. The drop probability is controlled by
            the 'drop_prob' attribute of the VanDropPath class.

        Example:
            ```python
            >>> drop_path = VanDropPath()
            >>> hidden_states = mindspore.Tensor([[1, 2, 3], [4, 5, 6]], mindspore.float32)
            >>> output = drop_path.forward(hidden_states)
            >>> print(output)
            [[1, 0, 3], [4, 0, 6]]
            ```
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        Return a string representation of the probability of dropping nodes during training.

        Args:
            self (VanDropPath): An instance of the VanDropPath class.

        Returns:
            str: A string representation of the probability of dropping nodes during training.

        Raises:
            None.

        This method returns a formatted string representation of the drop probability of the VanDropPath instance.
        The drop probability is obtained from the `drop_prob` attribute of the instance. The returned string is of the
        form 'p={}', where '{}' is replaced by the actual drop probability value.

        Example:
            If the `drop_prob` attribute of the instance is 0.3, the method will return the string "p=0.3".
        """
        return "p={}".format(self.drop_prob)


class VanOverlappingPatchEmbedder(nn.Module):
    """
    Downsamples the input using a patchify operation with a `stride` of 4 by default making adjacent windows overlap by
    half of the area. From [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 7, stride: int = 4):
        """
        Initializes a VanOverlappingPatchEmbedder object.

        Args:
            self: The instance of the class.
            in_channels (int): Number of input channels for the convolutional layer.
            hidden_size (int): Number of output channels from the convolutional layer.
            patch_size (int, optional): Size of the patch/kernel for the convolutional layer. Default is 7.
            stride (int, optional): Stride value for the convolution operation. Default is 4.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        self.normalization = nn.BatchNorm2d(hidden_size)

    def forward(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a hidden state tensor using the provided input tensor.

        Args:
            self (VanOverlappingPatchEmbedder): An instance of the VanOverlappingPatchEmbedder class.
            input (mindspore.Tensor): The input tensor to be processed.
                It should have shape (batch_size, channels, height, width).

        Returns:
            mindspore.Tensor: The hidden state tensor obtained from the input tensor after applying convolution and
                normalization. It has the same shape as the input tensor.

        Raises:
            None.

        Note:
            - The 'convolution' method is applied to the input tensor to obtain an intermediate hidden state tensor.
            - The 'normalization' method is then applied to the intermediate hidden state tensor to obtain the final
            hidden state tensor.
        """
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class VanMlpLayer(nn.Module):
    """
    MLP with depth-wise convolution, from [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        hidden_act: str = "gelu",
        dropout_rate: float = 0.5,
    ):
        """
        Initializes an instance of the VanMlpLayer class.

        Args:
            self: The object itself.
            in_channels (int): The number of input channels.
                This specifies the number of channels in the input tensor.
            hidden_size (int): The size of the hidden layer.
                This determines the number of output channels of the first convolutional layer.
            out_channels (int): The number of output channels.
                This specifies the number of channels in the output tensor.
            hidden_act (str, optional): The activation function for the hidden layer. Defaults to 'gelu'.
                This specifies the activation function to be used in the hidden layer.
                Supported options are 'gelu', 'relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'leaky_relu'.
            dropout_rate (float, optional): The dropout rate. Defaults to 0.5.
                This specifies the probability of an element to be zeroed in the dropout layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.in_dense = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.dropout1 = nn.Dropout(dropout_rate)
        self.out_dense = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards a multi-layer perceptron (MLP) layer in the VanMlpLayer class.

        Args:
            self (VanMlpLayer): The instance of the VanMlpLayer class.
            hidden_state (mindspore.Tensor): The input hidden state tensor to be processed by the MLP layer.

        Returns:
            mindspore.Tensor: The output tensor after processing through the MLP layer.

        Raises:
            None
        """
        hidden_state = self.in_dense(hidden_state)
        hidden_state = self.depth_wise(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.out_dense(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        return hidden_state


class VanLargeKernelAttention(nn.Module):
    """
    Basic Large Kernel Attention (LKA).
    """
    def __init__(self, hidden_size: int):
        """
        Initializes an instance of the VanLargeKernelAttention class.

        Args:
            self: The instance of the class.
            hidden_size (int): The size of the hidden layer. Specifies the number of hidden units in the neural network.
                It is used to define the dimensions of the convolutional layers within the attention mechanism.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size)
        self.depth_wise_dilated = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=7, dilation=3, padding=9, groups=hidden_size
        )
        self.point_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the attention mechanism in the VanLargeKernelAttention class.

        Args:
            self (VanLargeKernelAttention): An instance of the VanLargeKernelAttention class.
            hidden_state (mindspore.Tensor): The hidden state tensor representing the input data.

        Returns:
            mindspore.Tensor: The transformed hidden state tensor after passing through the attention mechanism.

        Raises:
            None
        """
        hidden_state = self.depth_wise(hidden_state)
        hidden_state = self.depth_wise_dilated(hidden_state)
        hidden_state = self.point_wise(hidden_state)
        return hidden_state


class VanLargeKernelAttentionLayer(nn.Module):
    """
    Computes attention using Large Kernel Attention (LKA) and attends the input.
    """
    def __init__(self, hidden_size: int):
        """
        Initializes a VanLargeKernelAttentionLayer instance with the specified hidden size.

        Args:
            self: The instance of the VanLargeKernelAttentionLayer class.
            hidden_size (int): The size of the hidden state, representing the dimensionality of the input feature space.
                It must be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = VanLargeKernelAttention(hidden_size)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards an attention mechanism in the VanLargeKernelAttentionLayer class.

        Args:
            self: The instance of the VanLargeKernelAttentionLayer class.
            hidden_state (mindspore.Tensor): The hidden state tensor on which the attention mechanism is applied.

        Returns:
            mindspore.Tensor: The attended tensor resulting from applying attention to the hidden state.

        Raises:
            No specific exceptions are raised by this method.
        """
        attention = self.attention(hidden_state)
        attended = hidden_state * attention
        return attended


class VanSpatialAttentionLayer(nn.Module):
    """
    Van spatial attention layer composed by projection (via conv) -> act -> Large Kernel Attention (LKA) attention ->
    projection (via conv) + residual connection.
    """
    def __init__(self, hidden_size: int, hidden_act: str = "gelu"):
        """
        Initializes an instance of the VanSpatialAttentionLayer class.

        Args:
            hidden_size (int): The size of the hidden layer.
            hidden_act (str, optional): The activation function to be used in the pre_projection layer. Defaults to 'gelu'.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.pre_projection = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(hidden_size, hidden_size, kernel_size=1)),
                    ("act", ACT2FN[hidden_act]),
                ]
            )
        )
        self.attention_layer = VanLargeKernelAttentionLayer(hidden_size)
        self.post_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards a spatial attention layer in the VanSpatialAttentionLayer class.

        Args:
            self: The instance of the VanSpatialAttentionLayer class.
            hidden_state (mindspore.Tensor): The input hidden state tensor to be processed.
                It represents the feature map of the input data and should be a tensor of shape
                [batch_size, channels, height, width].

        Returns:
            mindspore.Tensor: The processed hidden state tensor after applying the spatial attention mechanism.
                It has the same shape as the input hidden_state tensor.

        Raises:
            ValueError: If the input hidden_state tensor is not a valid mindspore.Tensor.
            RuntimeError: If an error occurs during the processing of the spatial attention mechanism.
        """
        residual = hidden_state
        hidden_state = self.pre_projection(hidden_state)
        hidden_state = self.attention_layer(hidden_state)
        hidden_state = self.post_projection(hidden_state)
        hidden_state = hidden_state + residual
        return hidden_state


class VanLayerScaling(nn.Module):
    """
    Scales the inputs by a learnable parameter initialized by `initial_value`.
    """
    def __init__(self, hidden_size: int, initial_value: float = 1e-2):
        """
        Initializes a new instance of the VanLayerScaling class.

        Args:
            self: The object itself.
            hidden_size (int): The size of the hidden layer.
            initial_value (float, optional): The initial value for the weight parameter. Default is 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.weight = Parameter(initial_value * ops.ones((hidden_size)), requires_grad=True)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method is part of the VanLayerScaling class and is used to perform scaling on the hidden_state tensor.

        Args:
            self (VanLayerScaling): The instance of the VanLayerScaling class.
            hidden_state (mindspore.Tensor): The input tensor representing the hidden state.
                It is expected to be a tensor of type mindspore.Tensor.

        Returns:
            mindspore.Tensor: Returns a tensor of type mindspore.Tensor which is the result of scaling the
                input hidden_state tensor.

        Raises:
            None.
        """
        # unsqueezing for broadcasting
        hidden_state = self.weight.unsqueeze(-1).unsqueeze(-1) * hidden_state
        return hidden_state


class VanLayer(nn.Module):
    """
    Van layer composed by normalization layers, large kernel attention (LKA) and a multi layer perceptron (MLP).
    """
    def __init__(
        self,
        config: VanConfig,
        hidden_size: int,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.5,
    ):
        """
        Initializes an instance of the VanLayer class.

        Args:
            self: The object itself.
            config (VanConfig): An object containing configuration settings for the layer.
            hidden_size (int): The size of the hidden layer.
            mlp_ratio (int, optional): The ratio of the hidden size to the output size of the MLP layer. Defaults to 4.
            drop_path_rate (float, optional): The rate at which to apply drop path regularization. Defaults to 0.5.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.drop_path = VanDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.pre_normomalization = nn.BatchNorm2d(hidden_size)
        self.attention = VanSpatialAttentionLayer(hidden_size, config.hidden_act)
        self.attention_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)
        self.post_normalization = nn.BatchNorm2d(hidden_size)
        self.mlp = VanMlpLayer(
            hidden_size, hidden_size * mlp_ratio, hidden_size, config.hidden_act, config.dropout_rate
        )
        self.mlp_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct method in the VanLayer class.

        This method forwards the output tensor by applying a series of operations to the input hidden state.

        Args:
            self: Instance of the VanLayer class.
            hidden_state (mindspore.Tensor): The input hidden state tensor on which the operations are performed.

        Returns:
            mindspore.Tensor: The output tensor after applying the operations on the input hidden state.

        Raises:
            None.
        """
        residual = hidden_state
        # attention
        hidden_state = self.pre_normomalization(hidden_state)
        hidden_state = self.attention(hidden_state)
        hidden_state = self.attention_scaling(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        # residual connection
        hidden_state = residual + hidden_state
        residual = hidden_state
        # mlp
        hidden_state = self.post_normalization(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = self.mlp_scaling(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        # residual connection
        hidden_state = residual + hidden_state
        return hidden_state


class VanStage(nn.Module):
    """
    VanStage, consisting of multiple layers.
    """
    def __init__(
        self,
        config: VanConfig,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        stride: int,
        depth: int,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ):
        """
        __init__

        Initializes a new instance of the VanStage class.

        Args:
            self: The current object instance.
            config (VanConfig): An instance of VanConfig class containing configuration parameters.
            in_channels (int): The number of input channels.
            hidden_size (int): The size of the hidden layer.
            patch_size (int): The size of the patch.
            stride (int): The stride for patching.
            depth (int): The depth of the network.
            mlp_ratio (int, optional): The ratio for the multi-layer perceptron. Defaults to 4.
            drop_path_rate (float, optional): The rate for drop path regularization. Defaults to 0.0.

        Returns:
            None.

        Raises:
            TypeError: If any of the input arguments does not match the expected type.
            ValueError: If any of the input arguments does not meet the specified restrictions.
        """
        super().__init__()
        self.embeddings = VanOverlappingPatchEmbedder(in_channels, hidden_size, patch_size, stride)
        self.layers = nn.Sequential(
            *[
                VanLayer(
                    config,
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )
        self.normalization = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_state: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the hidden state tensor for the VanStage class.

        Args:
            self: An instance of the VanStage class.
            hidden_state (mindspore.Tensor): A tensor representing the hidden state.
                It should have a shape of (batch_size, hidden_size, height, width).

        Returns:
            mindspore.Tensor: A tensor representing the forwarded hidden state.
                It has a shape of (batch_size, hidden_size, height, width).

        Raises:
            None.
        """
        hidden_state = self.embeddings(hidden_state)
        hidden_state = self.layers(hidden_state)
        # rearrange b c h w -> b (h w) c
        batch_size, hidden_size, height, width = hidden_state.shape
        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        hidden_state = self.normalization(hidden_state)
        # rearrange  b (h w) c- > b c h w
        hidden_state = hidden_state.view(batch_size, height, width, hidden_size).permute(0, 3, 1, 2)
        return hidden_state


class VanEncoder(nn.Module):
    """
    VanEncoder, consisting of multiple stages.
    """
    def __init__(self, config: VanConfig):
        """
        Initializes a VanEncoder object.

        Args:
            self: The instance of the class.
            config (VanConfig): An object containing configuration parameters for the VanEncoder.
                It includes the following attributes:

                - patch_sizes (List[int]): List of patch sizes for each stage.
                - strides (List[int]): List of stride values for each stage.
                - hidden_sizes (List[int]): List of hidden layer sizes for each stage.
                - depths (List[int]): List of depths for each stage.
                - mlp_ratios (List[int]): List of MLP expansion ratios for each stage.
                - drop_path_rate (float): Drop path rate for the encoder.

        Returns:
            None.

        Raises:
            AssertionError: If the config parameter is not of type VanConfig.
            TypeError: If any of the config attributes are not of the expected types.
            ValueError: If the drop_path_rate value is out of range or invalid.
        """
        super().__init__()
        self.stages = nn.ModuleList([])
        patch_sizes = config.patch_sizes
        strides = config.strides
        hidden_sizes = config.hidden_sizes
        depths = config.depths
        mlp_ratios = config.mlp_ratios
        drop_path_rates = [x.item() for x in ops.linspace(0, config.drop_path_rate, sum(config.depths))]

        for num_stage, (patch_size, stride, hidden_size, depth, mlp_expantion, drop_path_rate) in enumerate(
            zip(patch_sizes, strides, hidden_sizes, depths, mlp_ratios, drop_path_rates)
        ):
            is_first_stage = num_stage == 0
            in_channels = hidden_sizes[num_stage - 1]
            if is_first_stage:
                in_channels = config.num_channels
            self.stages.append(
                VanStage(
                    config,
                    in_channels,
                    hidden_size,
                    patch_size=patch_size,
                    stride=stride,
                    depth=depth,
                    mlp_ratio=mlp_expantion,
                    drop_path_rate=drop_path_rate,
                )
            )

    def forward(
        self,
        hidden_state: mindspore.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        """
        Construct method in the VanEncoder class.

        Args:
            self: The instance of the class.
            hidden_state (mindspore.Tensor): The input hidden state tensor.
            output_hidden_states (bool, optional): A flag indicating whether to output hidden states. Defaults to False.
            return_dict (bool, optional): A flag indicating whether to return the output as a dictionary. Defaults to True.

        Returns:
            Union[Tuple, BaseModelOutputWithNoAttention]: The forwarded output, which is either a tuple of hidden
                state and all hidden states or an instance of BaseModelOutputWithNoAttention.

        Raises:
            None
        """
        all_hidden_states = () if output_hidden_states else None

        for _, stage_module in enumerate(self.stages):
            hidden_state = stage_module(hidden_state)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)


class VanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = VanConfig
    base_model_prefix = "van"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class VanModel(VanPreTrainedModel):

    """
    The VanModel class represents a model for processing pixel values using the VanEncoder and providing various
    output representations. It inherits from the VanPreTrainedModel class and includes methods for initialization and
    forwarding the model's output. The forwardor initializes the model with the provided configuration, while the
    forward method processes the pixel values and returns the output representation. The class provides flexibility
    for handling hidden states and returning output in the form of BaseModelOutputWithPoolingAndNoAttention.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the VanModel class.

        Args:
            self: The object itself.
            config (object): The configuration object that contains various settings for the model.
                This object should have the following attributes:

                - hidden_sizes (list): A list of integers representing the sizes of hidden layers.
                - layer_norm_eps (float): A small value used for numerical stability in layer normalization.

                The config object is required for the proper initialization of the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        self.encoder = VanEncoder(config)
        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        """
        Constructs the encoder outputs and pooled output from the given pixel values.

        Args:
            self (VanModel): The instance of the VanModel class.
            pixel_values (Optional[mindspore.Tensor]): The input pixel values.
                If provided, it should be a Tensor.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
                If None, the value is taken from self.config.output_hidden_states.
            return_dict (Optional[bool]): Whether to return the output as a dictionary.
                If None, the value is taken from self.config.use_return_dict.

        Returns:
            Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]: A tuple containing the last hidden state and the
                pooled output, along with the encoder hidden states if return_dict is False. Otherwise, it
                returns a BaseModelOutputWithPoolingAndNoAttention object.

        Raises:
            None
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        # global average pooling, n c w h -> n c
        pooled_output = last_hidden_state.mean(dim=[-2, -1])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class VanForImageClassification(VanPreTrainedModel):

    """
    VanForImageClassification is a class that represents a model for image classification using a pre-trained VanModel
    for feature extraction and a classifier for final prediction. It inherits from VanPreTrainedModel and implements
    methods for model initialization and inference.

    Attributes:
        van (VanModel): The VanModel instance used for feature extraction.
        classifier (nn.Module): The classifier module for predicting the final output based on the extracted features.

    Methods:
        __init__:
            Initializes the VanForImageClassification model with the given configuration.

        forward:
            Constructs the model for image classification.

            Args:

            - pixel_values (Optional[mindspore.Tensor]): The input pixel values representing the image.
            - labels (Optional[mindspore.Tensor]): Labels for computing the image classification/regression loss.
            - output_hidden_states (Optional[bool]): Flag to output hidden states.
            - return_dict (Optional[bool]): Flag to determine if the return should be a dictionary.

            Returns:

            - Union[Tuple, ImageClassifierOutputWithNoAttention]: Tuple of output elements or
            ImageClassifierOutputWithNoAttention object.

    Example:
        ```python
        >>> model = VanForImageClassification(config)
        >>> output = model.forward(pixel_values, labels, output_hidden_states, return_dict)
        ```

    Note:
        The forward method computes the loss based on the labels and the model's prediction, and returns the output
        based on the configured settings.
    """
    def __init__(self, config):
        """
        __init__

        Initializes an instance of the VanForImageClassification class.

        Args:
            self: The instance of the class.
            config: A configuration object containing parameters for the van model and classification.
                This parameter is of type 'config' and is used to configure the van model and classifier.
                It should be an instance of the configuration class and must be provided.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.van = VanModel(config)
        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.van(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.config.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
