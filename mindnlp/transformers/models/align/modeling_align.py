# coding=utf-8
# Copyright 2023 The Google Research Team Authors and The HuggingFace Team. All rights reserved.
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
# ============================================================================
# pylint: disable=unexpected-keyword-arg
""" MindSpore ALIGN model."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal, XavierUniform

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "kakaobrain/align-base"
_CONFIG_FOR_DOC = "AlignConfig"


ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "kakaobrain/align-base",
    # See all ALIGN models at https://hf-mirror.com/models?filter=align
]

@dataclass
class AlignVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`mindspore.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    image_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class AlignTextModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`mindspore.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    text_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class AlignOutput(ModelOutput):
    """
    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`mindspore.Tensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`mindspore.Tensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`AlignTextModel`].
        image_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The output of [`AlignVisionModel`].
        text_model_output(`BaseModelOutputWithPoolingAndCrossAttentions`):
            The output of the [`AlignTextModel`].
        vision_model_output(`BaseModelOutputWithPoolingAndNoAttention`):
            The output of the [`AlignVisionModel`].
    """
    loss: Optional[mindspore.Tensor] = None
    logits_per_image: mindspore.Tensor = None
    logits_per_text: mindspore.Tensor = None
    text_embeds: mindspore.Tensor = None
    image_embeds: mindspore.Tensor = None
    text_model_output: BaseModelOutputWithPoolingAndCrossAttentions = None
    vision_model_output: BaseModelOutputWithPoolingAndNoAttention = None

    def to_tuple(self) -> Tuple[Any]:
        """
        Converts the AlignOutput instance to a tuple representation.
        
        Args:
            self: The instance of the AlignOutput class. It represents the object to be converted to a tuple.
        
        Returns:
            Tuple[Any]: A tuple containing the values of the AlignOutput instance. The method excludes 'text_model_output' and 
                'vision_model_output' keys and recursively converts any nested AlignOutput instances to tuples.
        
        Raises:
            None.
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: mindspore.Tensor) -> mindspore.Tensor:
    '''
    This function calculates the contrastive loss used in deep learning models.
    
    Args:
        logits (mindspore.Tensor): The input tensor representing the logits. It is a 1-D tensor of shape (N,),
            where N is the number of classes. The logits are usually the output of the model before applying
            the softmax function.

    Returns:
        mindspore.Tensor: The computed contrastive loss as a scalar tensor.

    Raises:
        None.
    '''
    return F.cross_entropy(logits, ops.arange(len(logits)), label_smoothing=0.1)


def align_loss(similarity: mindspore.Tensor) -> mindspore.Tensor:
    """
    Args:
        similarity (mindspore.Tensor): A tensor representing the similarity between two sets of embeddings.
            It is used to calculate the contrastive loss for alignment. The tensor should have the shape
            (batch_size, embedding_size).

    Returns:
        mindspore.Tensor: A tensor representing the alignment loss, which is the average of the contrastive loss calculated for captions and images.

    Raises:
        None
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


# Copied from transformers.models.efficientnet.modeling_efficientnet.round_filters with EfficientNet->AlignVision
def round_filters(config: AlignVisionConfig, num_channels: int):
    r"""
    Round number of filters based on depth multiplier.
    """
    divisor = config.depth_divisor
    num_channels *= config.width_coefficient
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


# Copied from transformers.models.efficientnet.modeling_efficientnet.correct_pad
def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
    r"""
    Utility function to get the tuple padding value for the depthwise convolution.

    Args:
        kernel_size (`int` or `tuple`):
            Kernel size of the convolution layers.
        adjust (`bool`, *optional*, defaults to `True`):
            Adjusts padding value to apply to right and bottom sides of the input.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    return (correct[1], correct[1], correct[0], correct[0])


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetEmbeddings with EfficientNet->AlignVision
class AlignVisionEmbeddings(nn.Module):
    r"""
    A module that corresponds to the stem module of the original work.
    """
    def __init__(self, config: AlignVisionConfig):
        """
        Initializes an instance of the AlignVisionEmbeddings class.

        Args:
            self: The instance of the class.
            config (AlignVisionConfig): The configuration object containing various parameters for the alignment vision.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes the AlignVisionEmbeddings instance by setting the values of various attributes.
            It takes two parameters: 'self' which refers to the instance of the class, and 'config' which is
            an object of type AlignVisionConfig.

            The 'self.out_dim' attribute is set to the rounded value of the 'config' parameter by calling the 'round_filters' method.
            The 'self.padding' attribute is set to an instance of nn.ZeroPad2d with a padding of (0, 1, 0, 1).

            The 'self.convolution' attribute is set to an instance of nn.Conv2d with the following parameters:

            - 'config.num_channels' as the input channels
            - 'self.out_dim' as the output channels
            - kernel size of 3
            - stride of 2
            - pad_mode set to 'valid'
            - bias set to False

            The 'self.batchnorm' attribute is set to an instance of nn.BatchNorm2d with the following parameters:

            - 'self.out_dim' as the number of channels
            - 'config.batch_norm_eps' as the epsilon value for numerical stability
            - 'config.batch_norm_momentum' as the momentum value for batch normalization
            
            The 'self.activation' attribute is set to the value corresponding to 'config.hidden_act' in the ACT2FN dictionary.
        """
        super().__init__()

        self.out_dim = round_filters(config, 32)
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        self.convolution = nn.Conv2d(
            config.num_channels, self.out_dim, kernel_size=3, stride=2, padding="valid", bias=False
        )
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the aligned vision embeddings for the given pixel values.

        Args:
            self: An instance of the AlignVisionEmbeddings class.
            pixel_values (mindspore.Tensor): A tensor containing the pixel values of the images.
                It should have shape (batch_size, channels, height, width).

        Returns:
            mindspore.Tensor: A tensor containing the aligned vision embeddings. It has the same shape as the input tensor.

        Raises:
            None.

        This method performs the following steps to forward the aligned vision embeddings:

        1. Padding: The pixel_values tensor is padded to ensure that the dimensions are compatible with the subsequent convolution operation.
        2. Convolution: The padded tensor is convolved using a predefined set of filters to extract features.
        3. Batch Normalization: The features tensor is normalized to improve the stability and speed of the training process.
        4. Activation: The normalized features are passed through an activation function to introduce non-linearity.
        5. The resulting tensor is returned as the aligned vision embeddings.
        """
        features = self.padding(pixel_values)
        features = self.convolution(features)
        features = self.batchnorm(features)
        features = self.activation(features)

        return features


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetDepthwiseConv2d with EfficientNet->AlignVision
class AlignVisionDepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetExpansionLayer with EfficientNet->AlignVision
class AlignVisionExpansionLayer(nn.Module):
    r"""
    This corresponds to the expansion phase of each block in the original implementation.
    """
    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int):
        """
        Initialize the AlignVisionExpansionLayer.

        Args:
            self: The instance of the AlignVisionExpansionLayer class.
            config (AlignVisionConfig): An instance of AlignVisionConfig containing configuration settings.
            in_dim (int): The input dimension for the expansion layer.
            out_dim (int): The output dimension for the expansion layer.

        Returns:
            None.

        Raises:
            ValueError: If the configuration settings are invalid or incompatible.
            TypeError: If the input or output dimension is not an integer.
            RuntimeError: If there is an issue with the execution of the method.
        """
        super().__init__()
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        self.expand_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards an expansion layer for align vision.

        Args:
            self: The instance of the AlignVisionExpansionLayer class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states. 
                It should be a tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels, and H, W are
                the height and width of the input feature map.

        Returns:
            mindspore.Tensor: 
                Returns a tensor representing the expanded hidden states after the expansion layer operations.

        Raises:
            ValueError: If the input hidden_states tensor is not of the expected shape (N, C, H, W).
            RuntimeError: If any runtime error occurs during the expansion layer operations.
        """
        # Expand phase
        hidden_states = self.expand_conv(hidden_states)
        hidden_states = self.expand_bn(hidden_states)
        hidden_states = self.expand_act(hidden_states)

        return hidden_states


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetDepthwiseLayer with with EfficientNet->AlignVision
class AlignVisionDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """
    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ):
        """
        Initializes an instance of the AlignVisionDepthwiseLayer class.

        Args:
            config (AlignVisionConfig): An instance of AlignVisionConfig class containing configuration parameters.
            in_dim (int): The number of input channels.
            stride (int): The stride value for convolution operation.
            kernel_size (int): The size of the kernel for convolution operation.
            adjust_padding (bool): A boolean flag indicating whether to adjust padding.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.stride = stride
        conv_pad = "valid" if self.stride == 2 else "same"
        padding = correct_pad(kernel_size, adjust=adjust_padding)

        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        self.depthwise_conv = AlignVisionDepthwiseConv2d(
            in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False
        )
        self.depthwise_norm = nn.BatchNorm2d(
            num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the depthwise convolutional layer for aligning vision, applying convolution, normalization, and activation operations.

        Args:
            self: An instance of the AlignVisionDepthwiseLayer class.
            hidden_states (mindspore.Tensor): The input tensor containing hidden states for the depthwise convolution operation.

        Returns:
            mindspore.Tensor: The output tensor after applying depthwise convolution, normalization, and activation operations.

        Raises:
            None
        """
        # Depthwise convolution
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


# Copied from transformers.models.efficientnet.modeling_efficientnet.EfficientNetSqueezeExciteLayer with with EfficientNet->AlignVision
class AlignVisionSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """
    def __init__(self, config: AlignVisionConfig, in_dim: int, expand_dim: int, expand: bool = False):
        """
        Initialize the AlignVisionSqueezeExciteLayer.

        Args:
            self: The instance of the class.
            config (AlignVisionConfig): An instance of AlignVisionConfig containing configuration parameters.
            in_dim (int): The input dimension.
            expand_dim (int): The dimension to expand to.
            expand (bool, optional): A flag indicating whether to expand the dimension. Defaults to False.

        Returns:
            None.

        Raises:
            ValueError: If the input dimensions are not valid.
            TypeError: If any of the arguments are of incorrect types.
        """
        super().__init__()
        self.dim = expand_dim if expand else in_dim
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))

        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=1,
            padding="same",
        )
        self.act_reduce = ACT2FN[config.hidden_act]
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the AlignVisionSqueezeExciteLayer.

        This method applies a series of operations to the input hidden_states in order to forward the AlignVisionSqueezeExciteLayer.

        Args:
            self (AlignVisionSqueezeExciteLayer): An instance of the AlignVisionSqueezeExciteLayer class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.

        Returns:
            mindspore.Tensor: The output tensor after applying the series of operations to the input hidden states tensor.

        Raises:
            None.
        """
        inputs = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)

        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        hidden_states = ops.mul(inputs, hidden_states)

        return hidden_states


class AlignVisionFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """
    def __init__(
        self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool
    ):
        """
        Initializes an instance of the AlignVisionFinalBlockLayer class.

        Args:
            self: The instance of the class.
            config (AlignVisionConfig): The configuration object for AlignVision.
            in_dim (int): The number of input channels.
            out_dim (int): The number of output channels.
            stride (int): The stride value for convolution operation.
            drop_rate (float): The dropout rate.
            id_skip (bool): Indicates whether to skip the identity connection.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.apply_dropout = stride == 1 and not id_skip
        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",
            bias=False,
        )
        self.project_bn = nn.BatchNorm2d(
            num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, embeddings: mindspore.Tensor, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the final block layer for alignment vision.

        Args:
            self (AlignVisionFinalBlockLayer): The instance of the AlignVisionFinalBlockLayer class.
            embeddings (mindspore.Tensor): The tensor representing embeddings to be added to hidden_states.
            hidden_states (mindspore.Tensor): The tensor representing the hidden states of the model.

        Returns:
            mindspore.Tensor: 
                The tensor representing the final hidden states after processing in the final block layer.

        Raises:
            None.
        """
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.project_bn(hidden_states)

        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings

        return hidden_states


class AlignVisionBlock(nn.Module):
    r"""
    This corresponds to the block module of original the EfficientNet vision encoder implementation.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """
    def __init__(
        self,
        config: AlignVisionConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
        id_skip: bool,
        adjust_padding: bool,
    ):
        """
        Initializes an instance of the AlignVisionBlock class.

        Args:
            self: The instance of the class.
            config (AlignVisionConfig): The configuration object for the AlignVision model.
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            stride (int): The stride value for the depthwise convolutional layer.
            expand_ratio (int): The expansion ratio for the input dimension.
            kernel_size (int): The kernel size for the depthwise convolutional layer.
            drop_rate (float): The dropout rate.
            id_skip (bool): Whether to use skip connections in the final block layer.
            adjust_padding (bool): Whether to adjust padding in the depthwise convolutional layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.expand_ratio = expand_ratio
        self.expand = self.expand_ratio != 1
        expand_in_dim = in_dim * expand_ratio

        if self.expand:
            self.expansion = AlignVisionExpansionLayer(
                config=config, in_dim=in_dim, out_dim=expand_in_dim
            )

        self.depthwise_conv = AlignVisionDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )
        self.squeeze_excite = AlignVisionSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )
        self.projection = AlignVisionFinalBlockLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
            id_skip=id_skip,
        )

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the AlignVisionBlock for processing hidden states.

        Args:
            self: An instance of the AlignVisionBlock class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.

        Returns:
            mindspore.Tensor: The processed hidden states tensor.

        Raises:
            None.

        This method takes the input hidden states and performs a series of operations to process them. 
        The method first assigns the input tensor to the variable 'embeddings'. 
        Then, if the 'expand_ratio' attribute of the AlignVisionBlock instance is not equal to 1, 
        the hidden states tensor is passed through the 'expansion' function to expand its dimensions. 
        After that, the expanded tensor is passed through the 'depthwise_conv' function to perform depthwise convolution. 
        The resulting tensor is then passed through the 'squeeze_excite' function for squeeze-and-excitation. 
        Finally, the method applies the 'projection' function to combine the original 'embeddings' tensor and the processed hidden states tensor. 
        The resulting tensor is returned as the output of the method.
        """
        embeddings = hidden_states
        # Expansion and depthwise convolution phase
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        hidden_states = self.depthwise_conv(hidden_states)

        # Squeeze and excite phase
        hidden_states = self.squeeze_excite(hidden_states)
        hidden_states = self.projection(embeddings, hidden_states)
        return hidden_states


class AlignVisionEncoder(nn.Module):
    r"""
    Forward propogates the embeddings through each vision encoder (EfficientNet) block.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
    """
    def __init__(self, config: AlignVisionConfig):
        """
        Initializes an instance of the AlignVisionEncoder class with the provided configuration.

        Args:
            self (AlignVisionEncoder): The instance of the AlignVisionEncoder class.
            config (AlignVisionConfig):
                An instance of AlignVisionConfig containing the configuration parameters for the encoder.
            
                - depth_coefficient (float): A coefficient used for computing the number of repeated blocks.
                - in_channels (list): List of input channel dimensions for each block.
                - out_channels (list): List of output channel dimensions for each block.
                - strides (list): List of stride values for each block.
                - kernel_sizes (list): List of kernel sizes for each block.
                - expand_ratios (list): List of expansion ratios for each block.
                - num_block_repeats (list): List of integers representing the number of times each block should be repeated.
                - depthwise_padding (set): Set of block numbers on which depthwise padding should be adjusted.
                - drop_connect_rate (float): The rate at which to apply drop connect regularization.

        Returns:
            None.

        Raises:
            ValueError: If any of the configuration parameters are invalid or missing.
            TypeError: If the configuration parameters are not of the expected types.
        """
        super().__init__()
        self.depth_coefficient = config.depth_coefficient

        def round_repeats(repeats):
            # Round number of block repeats based on depth multiplier.
            return int(math.ceil(self.depth_coefficient * repeats))

        num_base_blocks = len(config.in_channels)
        num_blocks = sum(round_repeats(n) for n in config.num_block_repeats)

        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]

            for j in range(round_repeats(config.num_block_repeats[i])):
                id_skip = j == 0
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = not curr_block_num in config.depthwise_padding
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                block = AlignVisionBlock(
                    config=config,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                    id_skip=id_skip,
                    adjust_padding=adjust_padding,
                )
                blocks.append(block)
                curr_block_num += 1

        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        """
        Construct method in the AlignVisionEncoder class.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input tensor containing the hidden states.
            output_hidden_states (Optional[bool]): A boolean flag indicating whether to output hidden states. Defaults to False.
            return_dict (Optional[bool]): A boolean flag indicating whether to return the output as a dictionary. Defaults to True.

        Returns:
            BaseModelOutputWithPoolingAndNoAttention: An instance of the BaseModelOutputWithPoolingAndNoAttention class containing the forwarded hidden states.

        Raises:
            None.
        """
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# Copied from transformers.models.bert.modeling_bert.BertEmbeddings with Bert->AlignText
class AlignTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes an instance of the AlignTextEmbeddings class.

        Args:
            self: The instance of the class.
            config (object):
                An object containing configuration parameters.

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden state.
                - pad_token_id (int): The index of the padding token.
                - max_position_embeddings (int): The maximum number of positions.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - position_embedding_type (str, optional): The type of position embeddings. Defaults to 'absolute'.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer('position_ids', ops.broadcast_to(ops.arange(config.max_position_embeddings), (1, -1)))
        self.register_buffer('token_type_ids', ops.zeros(*self.position_ids.shape, dtype=mindspore.int64))

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        Construct the aligned text embeddings.

        Args:
            self (AlignTextEmbeddings): An instance of the AlignTextEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): Tensor containing the input token IDs. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): Tensor containing the token type IDs. Default is None.
            position_ids (Optional[mindspore.Tensor]): Tensor containing the position IDs. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): Tensor containing the input embeddings. Default is None.
            past_key_values_length (int): Length of past key values. Default is 0.

        Returns:
            mindspore.Tensor: Tensor containing the aligned text embeddings.

        Raises:
            None.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in forwardor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = ops.broadcast_to(buffered_token_type_ids, (input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(*input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->AlignText
class AlignTextSelfAttention(nn.Module):

    """
    AlignTextSelfAttention

    This class represents a self-attention module for aligning text. It is designed for use in neural network models and inherits from the nn.Module class.

    Attributes:
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): The total size of all attention heads.
        query (nn.Linear): The linear transformation layer for the query.
        key (nn.Linear): The linear transformation layer for the key.
        value (nn.Linear): The linear transformation layer for the value.
        dropout (nn.Dropout): The dropout layer for attention probabilities.
        position_embedding_type (str): The type of position embedding used.
        distance_embedding (nn.Embedding): The embedding layer for distance information.
        is_decoder (bool): Indicates whether the module is used as a decoder.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the AlignTextSelfAttention class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the attention mechanism. It must have the following attributes:

                - hidden_size (int): The size of the hidden states.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (optional, int): The size of the embeddings if different from hidden_size (default: None).
                - attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
                - position_embedding_type (optional, str): The type of position embedding to use (default: None).
                  If not provided, it is obtained from the config object.
                - max_position_embeddings (int): The maximum number of positions for relative position embeddings.
                - is_decoder (bool): Whether the attention mechanism is used as a decoder.

            position_embedding_type (optional, str): The type of position embedding to use. If not provided, it defaults
            to 'absolute' if not specified in the config object.

        Returns:
            None

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and no embedding size is specified.

        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def swapaxes_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Swaps and permutes axes of a given tensor to align the text self-attention scores in the class named 'AlignTextSelfAttention'.

        Args:
            self (AlignTextSelfAttention): An instance of the AlignTextSelfAttention class.
            x (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: A tensor of shape (batch_size, num_attention_heads, sequence_length, attention_head_size), 
                where the axes have been swapped and permuted for aligning the text self-attention scores.

        Raises:
            None.

        This method takes in a tensor 'x' and performs the following operations:

        1. Computes the new shape of the tensor by appending the number of attention heads and attention head size
        dimensions to the existing shape.
        2. Reshapes the tensor 'x' to the new shape calculated in the previous step.
        3. Permutes the axes of the tensor 'x' to align the text self-attention scores, specifically the second and
        third dimensions are swapped.

        The method then returns the tensor 'x' with the swapped and permuted axes, which can be used for further
        computations within the AlignTextSelfAttention class.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        '''
        Constructs the self-attention mechanism for aligning text in the AlignTextSelfAttention class.

        Args:
            self (AlignTextSelfAttention): The instance of the AlignTextSelfAttention class.
            hidden_states (mindspore.Tensor): 
                The input tensor of shape (batch_size, sequence_length, hidden_size) representing 
                the hidden states of the input sequence.
            attention_mask (Optional[mindspore.Tensor]): 
                The optional input tensor of shape (batch_size, sequence_length) representing 
                the attention mask. Default is None.
            head_mask (Optional[mindspore.Tensor]): 
                The optional input tensor of shape (num_attention_heads, sequence_length, sequence_length) representing 
                the head mask. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): 
                The optional input tensor of shape (batch_size, encoder_sequence_length, hidden_size) representing 
                the hidden states of the encoder sequence. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): 
                The optional input tensor of shape (batch_size, encoder_sequence_length) representing 
                the attention mask for the encoder sequence. Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): 
                The optional input tensor of shape (2, batch_size, num_attention_heads, past_sequence_length, head_size) representing 
                the past key-value states. Default is None.
            output_attentions (Optional[bool]): Whether to output attention probabilities. Default is False.

        Returns:
            Tuple[mindspore.Tensor]: 
                A tuple containing the output context layer tensor of shape (batch_size, sequence_length, hidden_size) 
                and optionally the attention probabilities tensor of shape (batch_size, num_attention_heads, sequence_length, sequence_length), 
                and if self.is_decoder is True, the past key-value states tensor of shape (2, batch_size, num_attention_heads, sequence_length, head_size).

        Raises:
            None.
        '''
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.swapaxes_for_scores(self.key(encoder_hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = mindspore.tensor(key_length - 1, dtype=mindspore.int64).view(
                    -1, 1
                )
            else:
                position_ids_l = ops.arange(query_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(key_length, dtype=mindspore.int64).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = ops.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in AlignTextModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->AlignText
class AlignTextSelfOutput(nn.Module):

    """
    A class representing the output of self-aligning text data.

    This class is used to perform operations on text data for self-alignment. It includes layers for dense transformation,
    layer normalization, and dropout. The input consists of hidden states and an input tensor, and the output is the
    transformed hidden states after applying dense transformation, dropout, and layer normalization.
    """
    def __init__(self, config):
        """
        This method initializes an instance of the AlignTextSelfOutput class.

        Args:
            self: The instance of the AlignTextSelfOutput class.
            config: An object containing configuration parameters for the alignment text self output.
                It is expected to have the following attributes:
            
                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden state.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required attributes.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the aligned text self output.

        Args:
            self (AlignTextSelfOutput): An instance of the AlignTextSelfOutput class.
            hidden_states (mindspore.Tensor):
                The hidden states.
            
                - Shape: (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor):
                The input tensor.
            
                - Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor:
                The aligned text self output tensor.
            
                - Shape: (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->AlignText
class AlignTextAttention(nn.Module):

    """
    A class representing an align text attention mechanism for neural networks.

    This class implements an attention mechanism for aligning text sequences in neural networks.
    It includes methods for initializing the attention mechanism, pruning attention heads, and forwarding the attention output.

    This class inherits from nn.Module.

    Attributes:
        self: AlignTextSelfAttention
            The self-attention mechanism for aligning text sequences.
        output: AlignTextSelfOutput
            The output mechanism for processing attention outputs.
        pruned_heads: set
            A set containing the indices of pruned attention heads.

    Methods:
        __init__:
            Initializes the AlignTextAttention instance with the provided configuration and position embedding type.
        prune_heads:
            Prunes the specified attention heads from the self-attention mechanism.
        forward:
            Constructs the attention output based on the given input tensors and parameters.

    Returns:
        Tuple[mindspore.Tensor]:
            A tuple containing the attention output tensor and any additional outputs generated during the attention process.

    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the AlignTextAttention class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing settings and parameters for the attention mechanism.
            position_embedding_type (str, optional): Specifies the type of position embedding to be used. Defaults to None.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.self = AlignTextSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = AlignTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'AlignTextAttention'.

        Args:
            self: An instance of the class 'AlignTextAttention'. This parameter is used to access the attributes and methods of the class instance.

            heads: A list of integers representing the indices of attention heads to be pruned. If the list is empty, the method returns without performing any pruning.

        Returns:
            None.

        Raises:
            ValueError: If the length of the parameter 'heads' is not equal to 0 (indicating that there are attention heads to be pruned).
            AttributeError: If any attribute accessed within the method is not found or accessible.
            TypeError: If the provided 'heads' parameter is not a list of integers.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        This method forwards attention output based on the input hidden states and optional parameters for the AlignTextAttention class.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor specifying which elements should be attended to.
            head_mask (Optional[mindspore.Tensor]): Optional tensor to mask specific attention heads.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional tensor of hidden states from an encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional tensor for encoder attention mask.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple of past key and value tensors.
            output_attentions (Optional[bool]): Optional flag to output the attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor.

        Raises:
            None
        """
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->AlignText
class AlignTextIntermediate(nn.Module):

    """
    Represents a neural network module for aligning text with intermediate processing steps.

    This class inherits from nn.Module and provides methods for initializing the module with configuration parameters
    and forwarding the neural network with intermediate processing steps.

    The class includes an initialization method that sets up the dense layers based on the provided configuration.
    It also forwards the neural network by applying the intermediate activation function to the hidden states after passing through the dense layer.

    Attributes:
        dense (nn.Linear): Dense layer for processing hidden states.
        intermediate_act_fn (Activation function): Function for intermediate activation of hidden states.

    Methods:
        __init__: Initializes the neural network module with the given configuration.
        forward: Constructs the neural network by processing the hidden states.

    Note:
        The class is designed for aligning text with intermediate processing steps in a neural network architecture.
    """
    def __init__(self, config):
        """
        Initializes an instance of the AlignTextIntermediate class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' that holds the configuration settings.
                This parameter is required for initializing the class and configuring its behavior.
                It should be an instance of the 'Config' class defined elsewhere.
                The 'Config' class should have the following attributes:

                - hidden_size: An integer representing the size of the hidden layer.
                - intermediate_size: An integer representing the size of the intermediate layer.
                - hidden_act: Either a string indicating the activation function for the hidden layer,
                  or a callable object representing the activation function itself.
                  If it is a string, it should be one of the activation functions defined in the ACT2FN dictionary.
                  If it is a callable object, it will be used directly as the activation function.
                  Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the intermediate representation of hidden states for aligning text in the AlignTextIntermediate class.

        Args:
            self (AlignTextIntermediate): Instance of the AlignTextIntermediate class.
            hidden_states (mindspore.Tensor): The hidden states tensor to be processed. It represents the input tensor
            containing hidden states.

        Returns:
            mindspore.Tensor: The processed hidden states tensor after passing through the dense layer and activation function.
            It represents the intermediate representation of the hidden states for aligning text.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->AlignText
class AlignTextOutput(nn.Module):

    """
    AlignTextOutput class represents a neural network cell for aligning text output.
    This class inherits from nn.Module and contains methods for initializing and forwarding the align text output.

    Attributes:
        config (object): The configuration object for the align text output.

    Methods:
        __init__:
            Initializes the align text output cell with the given configuration.

        forward:
            Constructs the align text output using the provided hidden states and input tensor.

        Args:
            hidden_states (mindspore.Tensor): The hidden states tensor.
            input_tensor (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The aligned text output tensor.
    """
    def __init__(self, config):
        """
        Initialize the AlignTextOutput class with the provided configuration.

        Args:
            self (AlignTextOutput): The instance of the AlignTextOutput class.
            config: An object containing configuration parameters for the AlignTextOutput class.
                It must have the following attributes:

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            ValueError: If the config parameter is missing any of the required attributes.
            TypeError: If the config parameter is not of the expected type.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the output tensor by performing a series of operations on the input hidden states and tensor.

        Args:
            self: An instance of the AlignTextOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor to be processed.
                This tensor contains the encoded information from the input text.
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden states.
                This tensor is typically the original input tensor to be aligned with the processed hidden states.

        Returns:
            mindspore.Tensor: A tensor representing the aligned output of the hidden states and input tensor.
                This tensor is the result of processing the hidden states through dense layers, dropout, and layer normalization,
                and then adding it to the input tensor.

        Raises:
            TypeError: If the input hidden_states or input_tensor is not of type mindspore.Tensor.
            ValueError: If the dimensions of hidden_states and input_tensor are not compatible for addition.
            RuntimeError: If an error occurs during the execution of the dense, dropout, or LayerNorm operations.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->AlignText
class AlignTextLayer(nn.Module):

    """
    This class represents an AlignTextLayer for processing text sequences with attention mechanisms in a neural network model.

    This class inherits from nn.Module and implements methods for initializing the layer,
    forwarding the layer with attention mechanisms, and performing feed-forward chunk processing.

    Attributes:
        chunk_size_feed_forward (int): The chunk size used for feed-forward processing.
        seq_len_dim (int): The dimension of the sequence length.
        attention (AlignTextAttention): An instance of AlignTextAttention for handling attention mechanisms.
        is_decoder (bool): Flag indicating whether the layer is used as a decoder model.
        add_cross_attention (bool): Flag indicating whether cross-attention is added to the model.
        crossattention (AlignTextAttention): An instance of AlignTextAttention for cross-attention if added.
        intermediate (AlignTextIntermediate): An instance of AlignTextIntermediate for intermediate processing.
        output (AlignTextOutput): An instance of AlignTextOutput for final output processing.

    Methods:
        __init__: Initializes the AlignTextLayer with configuration settings.

        forward:
            Constructs the layer with attention mechanisms and handles cross-attention if added.

        feed_forward_chunk: Performs feed-forward chunk processing on the attention output.

    Raises:
        ValueError:
            Raised under specific conditions such as incorrect usage as a decoder model or missing cross-attention layers.

    """
    def __init__(self, config):
        """
        Initializes an instance of the AlignTextLayer class.

        Args:
            self: The instance of the AlignTextLayer class.
            config:
                A configuration object containing parameters for the AlignTextLayer.

                - Type: Config object
                - Purpose: Contains settings and hyperparameters for the AlignTextLayer.
                - Restrictions: Must be provided for proper initialization.

        Returns:
            None.

        Raises:
            ValueError:
                Raised if self.add_cross_attention is True and self.is_decoder is False.

                - Purpose: Ensures that cross attention is only added when the model is used as a decoder.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AlignTextAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = AlignTextAttention(config, position_embedding_type="absolute")
        self.intermediate = AlignTextIntermediate(config)
        self.output = AlignTextOutput(config)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Constructs the AlignTextLayer.

        This layer is responsible for aligning text using self-attention and cross-attention mechanisms.

        Args:
            self (AlignTextLayer): The instance of the AlignTextLayer class.
            hidden_states (mindspore.Tensor): The input hidden states. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask. Shape: (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask. Shape: (num_attention_heads, sequence_length, sequence_length).
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder. Shape: (batch_size, encoder_sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder. Shape: (batch_size, encoder_sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key-value pairs. Shape: ((self_attention_past_key, self_attention_past_value), (cross_attention_past_key,
            cross_attention_past_value)).
            output_attentions (Optional[bool]): Whether to output attentions. Default: False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the outputs of the layer. The first element is the layer output tensor.
                Shape: (batch_size, sequence_length, hidden_size). If the layer is a decoder, the tuple also contains the present key-value pairs.

        Raises:
            ValueError:
                If `encoder_hidden_states` are passed, but the layer is not instantiated with cross-attention layers
                by setting `config.add_cross_attention=True`.
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        This method performs a feed-forward chunk operation on the given attention output.

        Args:
            self (AlignTextLayer): An instance of the AlignTextLayer class.
            attention_output:
                The input attention output tensor.

                - Type: Tensor
                - Purpose: Represents the attention output from a previous layer.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->AlignText
class AlignTextEncoder(nn.Module):

    """
    This class represents an AlignTextEncoder that inherits from nn.Module.

    The AlignTextEncoder initializes with a configuration and forwards the encoder layer with align text functionality.
    It supports gradient checkpointing during training and provides options to output hidden states, attentions, and cross-attentions.
    The encoder can handle various input tensors such as hidden states, attention masks, head masks, encoder hidden states,
    encoder attention masks, past key values, and caching.

    The forward method processes the input tensors through the encoder layers, applying gradient checkpointing if enabled during training.
    It iterates through each layer to generate hidden states and optional outputs like next decoder cache,
    all hidden states, self-attentions, and cross-attentions. The method returns the desired outputs based on the return_dict flag.

    This class provides a flexible and efficient way to encode text data using alignment techniques within a neural network architecture.
    """
    def __init__(self, config):
        """
        Initializes an instance of the AlignTextEncoder class.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing parameters for the AlignTextEncoder.

                - Type: dict
                - Purpose: Specifies the configuration settings for the AlignTextEncoder.
                - Restrictions: Must be a valid dictionary object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([AlignTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        This method forwards the AlignTextEncoder model.

        Args:
            self: The instance of the AlignTextEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states.
            attention_mask (Optional[mindspore.Tensor]): Optional attention mask tensor. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): Optional head mask tensor. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional encoder hidden states tensor. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional encoder attention mask tensor. Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple of past key values. Defaults to None.
            use_cache (Optional[bool]): Optional boolean flag for caching. Defaults to None.
            output_attentions (Optional[bool]): Optional boolean flag to output attentions. Defaults to False.
            output_hidden_states (Optional[bool]): Optional boolean flag to output hidden states. Defaults to False.
            return_dict (Optional[bool]): Optional boolean flag to return a dictionary. Defaults to True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]: The forwarded output of the model.

        Raises:
            Warning: Raised if `use_cache=True` is incompatible with gradient checkpointing. Sets `use_cache=False` in such case.
            Other exceptions may be raised during the execution of the method based on internal conditions.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert -> AlignText
class AlignTextPooler(nn.Module):

    """
    AlignTextPooler

    This class represents a text pooler that aligns the input hidden states and performs pooling operation. It inherits from the nn.Module class.

    Attributes:
        dense (nn.Linear): A fully connected layer that maps the hidden states to a specific size.
        activation (nn.Tanh): An activation function that applies the hyperbolic tangent to the pooled output.

    Methods:
        __init__:
            Initializes the AlignTextPooler instance with the given configuration.

        forward:
            Constructs the pooled output by aligning the input hidden states and applying pooling.

    """
    def __init__(self, config):
        """
        Initializes the AlignTextPooler class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the AlignTextPooler.

                - Type: object
                - Purpose: Specifies the configuration settings for the AlignTextPooler.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """Constructs the aligned text pooler.

        This method takes two parameters: self and hidden_states.

        Args:
            self: An instance of the AlignTextPooler class.
            hidden_states (mindspore.Tensor): The hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the input hidden states for the pooler.

        Returns:
            mindspore.Tensor: The pooled output tensor of shape (batch_size, hidden_size).
                It represents the output pooled tensor after applying the alignment and pooling operations.

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class AlignPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlignConfig
    base_model_prefix = "align"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Linear, nn.Conv2d)):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, AlignModel):
            cell.text_projection.weight.set_data(initializer(XavierUniform(), cell.text_projection.weight.shape,
                                                             cell.text_projection.weight.dtype))
            cell.text_projection.bias[:] = 0
            cell.text_projection._is_initialized = True
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class AlignTextModel(AlignPreTrainedModel):

    """
    The `AlignTextModel` class represents a model for aligning text.
    It includes methods for initializing the model, getting and setting input embeddings, and forwarding the model for inference.

    The `__init__` method initializes the model with the provided configuration and sets up
    the embeddings, encoder, and pooler layers based on the configuration parameters.

    The `get_input_embeddings` method retrieves the word embeddings used as input to the model.

    The `set_input_embeddings` method allows for setting custom word embeddings as input to the model.

    The `forward` method forwards the model for inference based on the input parameters such as
    input tokens, attention mask, token type ids, etc.
    It returns the model outputs including the last hidden state and pooled output.

    The class also includes examples of how to use the model for text alignment tasks.

    This class inherits from `AlignPreTrainedModel`.
    """
    config_class = AlignTextConfig

    def __init__(self, config: AlignTextConfig, add_pooling_layer: bool = True):
        """
        Initializes an instance of AlignTextModel.

        Args:
            self: The instance of the AlignTextModel class.
            config (AlignTextConfig): An instance of AlignTextConfig containing configuration parameters.
            add_pooling_layer (bool, optional): A flag indicating whether to add a pooling layer. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = AlignTextEmbeddings(config)
        self.encoder = AlignTextEncoder(config)

        self.pooler = AlignTextPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the AlignTextModel.

        Args:
            self: The instance of the AlignTextModel class.

        Returns:
            None: This method returns None as it retrieves the input embeddings without any transformations.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the AlignTextModel.

        Args:
            self (AlignTextModel): The instance of the AlignTextModel class.
            value (any): The input embeddings value to be set for the model. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Returns:
            Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, AlignTextModel
            ...
            >>> model = AlignTextModel.from_pretrained("kakaobrain/align-base")
            >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = ops.ones(batch_size, seq_length)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = ops.broadcast_to(buffered_token_type_ids, (batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(*input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class AlignVisionModel(AlignPreTrainedModel):

    """
    This class represents an AlignVision model for vision tasks, which includes functionalities for processing images
    and generating embeddings using a vision encoder.

    The model supports different pooling strategies for extracting features from the encoded image representations.

    It inherits from AlignPreTrainedModel and provides methods for initializing the model, accessing input embeddings,
    and forwarding the model output.

    The model's forwardor takes an AlignVisionConfig object as a parameter to configure the model's behavior.
    It initializes the model's components including embeddings and encoder based on the provided configuration,
    and sets up the pooling strategy based on the specified pooling type in the configuration.

    The 'get_input_embeddings' method returns the input embeddings generated by the model's convolutional layers for further processing.

    The 'forward' method processes input pixel values to generate embeddings using the model's embeddings and encoder components.
    It then applies the pooling strategy to extract features from the encoded image representations.
    The method returns the last hidden state, pooled output, and additional encoder outputs based on the specified return format.

    The class provides examples in the docstring to demonstrate how to use the model for image processing tasks,
    including loading an image, processing it with the model, and accessing the output hidden states
    and pooled output for further analysis.
    """
    config_class = AlignVisionConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: AlignVisionConfig):
        """
        Initializes an instance of the AlignVisionModel class.

        Args:
            self: The instance of the class.
            config (AlignVisionConfig): An object containing configuration parameters for the model.

        Returns:
            None

        Raises:
            ValueError: If the 'pooling_type' in the config is not one of ['mean', 'max'].

        Description:
            This method initializes an instance of the AlignVisionModel class.
            It takes in a config object which contains the configuration parameters for the model.
            The 'config' parameter is of type AlignVisionConfig.

            Inside the method, the superclass's __init__ method is called with the 'config' parameter.
            The 'config' is then assigned to the 'self.config' attribute.

            The method also initializes the 'embeddings' attribute with an instance of AlignVisionEmbeddings,
            passing in the 'config' parameter. Similarly, the 'encoder' attribute is initialized with an instance
            of AlignVisionEncoder, passing in the 'config' parameter.

            The 'pooler' attribute is dynamically set based on the value of the 'pooling_type' in the 'config'.

            - If 'pooling_type' is set to 'mean', the 'pooler' attribute is set to a partial function 'ops.mean'
            with the specified axis and keep_dims parameters.
            - If 'pooling_type' is set to 'max', the 'pooler' attribute is set to an instance of nn.MaxPool2d
            with the specified 'hidden_dim' and 'ceil_mode' parameters.
            - If the 'pooling_type' in the 'config' is not one of ['mean', 'max'], a ValueError is raised.

            Finally, the 'post_init' method is called.

            This method does not return any value.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = AlignVisionEmbeddings(config)
        self.encoder = AlignVisionEncoder(config)

        # Final pooling layer
        if config.pooling_type == "mean":
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """
        Retrieve the input embeddings from the AlignVisionModel.

        Args:
            self (AlignVisionModel): The instance of the AlignVisionModel class.

        Returns:
            nn.Module: The input embeddings extracted from the vision model's convolution layer.

        Raises:
            None.
        """
        return self.vision_model.embeddings.convolution

    def forward(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        r"""
        Returns:
            Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, AlignVisionModel
            ...
            >>> model = AlignVisionModel.from_pretrained("kakaobrain/align-base")
            >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled CLS states
            ```
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Apply pooling
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        # Reshape (batch_size, projection_dim, 1 , 1) -> (batch_size, projection_dim)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class AlignModel(AlignPreTrainedModel):

    """
    The `AlignModel` class is a model for aligning text and image embeddings.
    It is designed to compute image-text similarity scores using pre-trained text and vision models.
    The class inherits from the `AlignPreTrainedModel` class.

    Attributes:
        `projection_dim`: The dimension of the projection layer.
        `text_embed_dim`: The dimension of the text embeddings.
        `text_model`: An instance of the `AlignTextModel` class for processing text inputs.
        `vision_model`: An instance of the `AlignVisionModel` class for processing image inputs.
        `text_projection`: A dense layer for projecting the text embeddings.
        `temperature`: A parameter for scaling the similarity scores.

    Methods:
        `__init__`: Initializes the `AlignModel` class.
        `get_text_features`: Computes the text embeddings.
        `get_image_features`: Computes the image embeddings.
        `forward`: Constructs the model and computes the image-text similarity scores.

    Please see the code examples in the docstrings of each method for usage details.
    """
    config_class = AlignConfig

    def __init__(self, config: AlignConfig):
        '''
        Initializes the AlignModel with the specified configuration.

        Args:
            self: The instance of the AlignModel class.
            config (AlignConfig):
                An object containing the configuration settings for the AlignModel.

                - text_config (AlignTextConfig): The configuration settings for the text model.
                - vision_config (AlignVisionConfig): The configuration settings for the vision model.
                - projection_dim (int): The dimension for the projection.

        Returns:
            None.

        Raises:
            ValueError: If the config.text_config is not of type AlignTextConfig.
            ValueError: If the config.vision_config is not of type AlignVisionConfig.
        '''
        super().__init__(config)

        if not isinstance(config.text_config, AlignTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type AlignTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, AlignVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type AlignVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size

        self.text_model = AlignTextModel(text_config)
        self.vision_model = AlignVisionModel(vision_config)

        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim)
        self.register_buffer('temperature', mindspore.tensor(self.config.temperature_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            text_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
                applying the projection layer to the pooled output of [`AlignTextModel`].

        Example:
            ```python
            >>> from transformers import AutoTokenizer, AlignModel
            ...
            >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
            >>> tokenizer = AutoTokenizer.from_pretrained("kakaobrain/align-base")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
            ```
        """
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = text_outputs[0][:, 0, :]
        text_features = self.text_projection(last_hidden_state)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            image_features (`mindspore.Tensor` of shape `(batch_size, output_dim`):
                The image embeddings obtained by applying the projection layer to the pooled output of [`AlignVisionModel`].

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, AlignModel
            ...
            >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
            >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="pt")
            ...
            >>> image_features = model.get_image_features(**inputs)
            ```
        """
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_features = vision_outputs[1]  # pooled_output

        return image_features

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AlignOutput]:
        r"""
        Returns:
            Union[Tuple, AlignOutput]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, AlignModel
            ...
            >>> model = AlignModel.from_pretrained("kakaobrain/align-base")
            >>> processor = AutoProcessor.from_pretrained("kakaobrain/align-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(
            ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
            ... )
            ...
            >>> outputs = model(**inputs)
            >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            ```
        """
        # Use ALIGN model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[0][:, 0, :]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = ops.matmul(text_embeds, image_embeds.t()) / self.temperature
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = align_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return AlignOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

__all__ = [
    "ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST",
    "AlignModel",
    "AlignPreTrainedModel",
    "AlignTextModel",
    "AlignVisionModel",
]
