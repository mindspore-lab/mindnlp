# coding=utf-8
# Copyright 2022 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore CvT model."""


import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import TruncatedNormal

from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ....utils import logging
from .configuration_cvt import CvtConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "CvtConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/cvt-13"
_EXPECTED_OUTPUT_SHAPE = [1, 384, 14, 14]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/cvt-13"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


@dataclass
class BaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`mindspore.Tensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """
    last_hidden_state: mindspore.Tensor = None
    cls_token_value: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None


# Copied from transformers.models.beit.modeling_beit.drop_path
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


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
class CvtDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initializes an instance of the CvtDropPath class.
        
        Args:
            self: The instance of the class.
            drop_prob (Optional[float]): The probability of dropping a connection during training. Defaults to None.
                Must be a float value between 0 and 1, inclusive.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a modified version of the input hidden_states tensor using the drop_path operation.
        
        Args:
            self (CvtDropPath): The instance of the CvtDropPath class.
            hidden_states (mindspore.Tensor): The input tensor representing hidden states.
                It should be a tensor of arbitrary shape and type.
            
        Returns:
            mindspore.Tensor: A tensor of the same shape and type as the input hidden_states tensor,
                but with the drop_path operation applied.
        
        Raises:
            ValueError: If the input hidden_states tensor is not a valid mindspore.Tensor object.
            RuntimeError: If an error occurs during the drop_path operation.
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        This method provides a string representation for the CvtDropPath class.
        
        Args:
            self: CvtDropPath instance. Represents the current instance of the CvtDropPath class.
        
        Returns:
            str: A string representing the drop probability of the CvtDropPath instance.
        
        Raises:
            None.
        """
        return "p={}".format(self.drop_prob)


class CvtEmbeddings(nn.Cell):
    """
    Construct the CvT embeddings.
    """
    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate):
        """
        Initializes an instance of the CvtEmbeddings class.
        
        Args:
            self: The object instance.
            patch_size (int): The size of the patches to be extracted from the input image.
            num_channels (int): The number of input channels in the image.
            embed_dim (int): The dimension of the embedded representation.
            stride (int): The stride of the convolution operation.
            padding (int): The amount of padding to be added to the input image.
            dropout_rate (float): The dropout rate to be applied to the convolutional embeddings.
        
        Returns:
            None.
        
        Raises:
            None.
        
        """
        super().__init__()
        self.convolution_embeddings = CvtConvEmbeddings(
            patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def construct(self, pixel_values):
        """
        Constructs the hidden state using convolutional embeddings.
        
        Args:
            self (CvtEmbeddings): The instance of the CvtEmbeddings class.
            pixel_values (array-like): An array-like object containing pixel values for image data.
        
        Returns:
            numpy.ndarray: The hidden state constructed using convolutional embeddings.
        
        Raises:
            ValueError: If the pixel_values parameter is empty or not valid.
            TypeError: If the pixel_values parameter is not array-like.
            RuntimeError: If an unexpected error occurs during the construction process.
        """
        hidden_state = self.convolution_embeddings(pixel_values)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtConvEmbeddings(nn.Cell):
    """
    Image to Conv Embedding.
    """
    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        """
        __init__
        
        Initializes the CvtConvEmbeddings class.
        
        Args:
            self: The instance of the class.
            patch_size (int or tuple): The size of the patch or kernel used for convolution.
                If an int is provided, the patch will be square.
                If a tuple is provided, it should contain two integers representing the height and width of the patch.
            num_channels (int): The number of input channels for the convolutional layer.
            embed_dim (int): The dimensionality of the output embedding.
            stride (int or tuple): The stride of the convolution operation.
                If an int is provided, the same stride is used in both dimensions.
                If a tuple is provided, it should contain two integers
                representing the stride in the height and width dimensions.
            padding (int or tuple): The amount of padding to be added to the input data for the convolution operation.
                If an int is provided, the same padding is added to both dimensions.
                If a tuple is provided, it should contain two integers representing the padding
                in the height and width dimensions.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, pad_mode='pad', has_bias=True)
        self.normalization = nn.LayerNorm(embed_dim)

    def construct(self, pixel_values):
        """
        Constructs the pixel embeddings for a given set of pixel values.

        Args:
            self (CvtConvEmbeddings): An instance of the CvtConvEmbeddings class.
            pixel_values (torch.Tensor): A tensor containing the pixel values of the image.
                It should have the shape (batch_size, num_channels, height, width).

        Returns:
            None: This method modifies the pixel_values tensor in-place.

        Raises:
            None.
        """
        pixel_values = self.projection(pixel_values)
        batch_size, num_channels, height, width = pixel_values.shape
        hidden_size = height * width
        # rearrange "b c h w -> b (h w) c"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        if self.normalization:
            pixel_values = self.normalization(pixel_values)
        # rearrange "b (h w) c" -> b c h w"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return pixel_values


class CvtSelfAttentionConvProjection(nn.Cell):

    """
    CvtSelfAttentionConvProjection represents a class for performing convolution and normalization operations
    on input data. This class inherits from nn.Cell and provides methods for initializing the
    convolution and normalization layers, as well as for constructing the output from the input hidden state.

    Attributes:
        embed_dim (int): The dimension of the input embedding.
        kernel_size (int): The size of the convolutional kernel.
        padding (int): The amount of padding to apply to the input data.
        stride (int): The stride of the convolution operation.

    Methods:
        __init__: Initializes the CvtSelfAttentionConvProjection class with the specified parameters.
        construct: Constructs the output from the input hidden state by applying convolution and normalization operations.

    """
    def __init__(self, embed_dim, kernel_size, padding, stride):
        """
        Initializes a new instance of the CvtSelfAttentionConvProjection class.

        Args:
            self (CvtSelfAttentionConvProjection): The object itself.
            embed_dim (int): The number of channels in the input and output tensors.
            kernel_size (int or Tuple[int, int]): The size of the convolving kernel.
            padding (int or Tuple[int, int]): The amount of padding added to the input.
            stride (int or Tuple[int, int]): The stride of the convolution.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.convolution = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            pad_mode='pad',
            stride=stride,
            has_bias=False,
            group=embed_dim,
        )
        self.normalization = nn.BatchNorm2d(embed_dim)

    def construct(self, hidden_state):
        """
        Constructs a hidden state using convolution, normalization, and projection in the CvtSelfAttentionConvProjection class.

        Args:
            self (CvtSelfAttentionConvProjection): An instance of the CvtSelfAttentionConvProjection class.
            hidden_state (any): The input hidden state.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class CvtSelfAttentionLinearProjection(nn.Cell):

    """
    The 'CvtSelfAttentionLinearProjection' class is a Python class that inherits from the 'nn.Cell' class.
    It represents a linear projection operation applied to hidden states in a self-attention mechanism.

    Attributes:
        None.

    Methods:
        construct(hidden_state): Applies a linear projection to the input hidden state.

    """
    def construct(self, hidden_state):
        """
        Constructs a linear projection of hidden state for self-attention in the CvtSelfAttentionLinearProjection class.

        Args:
            self (CvtSelfAttentionLinearProjection): The instance of the CvtSelfAttentionLinearProjection class.
            hidden_state (torch.Tensor): The hidden state tensor with shape (batch_size, num_channels, height, width),
                where batch_size is the number of samples in the batch, num_channels is the number of channels,
                height is the height of the hidden state tensor, and width is the width of the hidden state tensor.

        Returns:
            torch.Tensor: The linearly projected hidden state tensor with shape (batch_size, hidden_size, num_channels),
                where batch_size is the number of samples in the batch, hidden_size is the product of height and width
                of the hidden state tensor, and num_channels is the number of channels.
                The tensor is permuted to have the dimensions (batch_size, hidden_size, num_channels).

        Raises:
            None
        """
        batch_size, num_channels, height, width = hidden_state.shape
        hidden_size = height * width
        # rearrange " b c h w -> b (h w) c"
        hidden_state = hidden_state.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_state


class CvtSelfAttentionProjection(nn.Cell):

    """
    A class representing the projection layer for self-attention in a Convolutional Transformer network.

    This class is responsible for projecting the input hidden state using convolutional and linear projections.
    It provides methods to initialize the projections and apply them sequentially to the input hidden state.

    Attributes:
        embed_dim (int): The dimensionality of the input embeddings.
        kernel_size (int): The size of the convolutional kernel.
        padding (int): The amount of padding to apply during convolution.
        stride (int): The stride of the convolution operation.
        projection_method (str): The method used for projection, default is 'dw_bn' (depthwise batch normalization).

    Methods:
        __init__:
            Initializes the projection layer with the specified parameters.

        construct:
            Applies the convolutional projection followed by the linear projection to the input hidden state.
            Returns the projected hidden state.

    Note:
        This class inherits from nn.Cell and is designed to be used within a Convolutional Transformer network.
    """
    def __init__(self, embed_dim, kernel_size, padding, stride, projection_method="dw_bn"):
        """
        Initializes an instance of the CvtSelfAttentionProjection class.

        Args:
            self (CvtSelfAttentionProjection): The instance of the class.
            embed_dim (int): The dimensionality of the input embeddings.
            kernel_size (int): The size of the convolutional kernel.
            padding (int): The amount of padding to be added to the input.
            stride (int): The stride value for the convolution operation.
            projection_method (string, optional): The method used for projection. Defaults to 'dw_bn'.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        if projection_method == "dw_bn":
            self.convolution_projection = CvtSelfAttentionConvProjection(embed_dim, kernel_size, padding, stride)
        self.linear_projection = CvtSelfAttentionLinearProjection()

    def construct(self, hidden_state):
        """
        Constructs the self-attention projection for the CvtSelfAttentionProjection class.

        Args:
            self (CvtSelfAttentionProjection): The instance of the CvtSelfAttentionProjection class.
            hidden_state (Tensor): The hidden state tensor to be projected.

        Returns:
            None: The method modifies the hidden_state in-place after applying convolution and linear projections.

        Raises:
            None.
        """
        hidden_state = self.convolution_projection(hidden_state)
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state


class CvtSelfAttention(nn.Cell):

    """
    This class represents a Convolutional Self-Attention layer for a neural network model. It inherits from the nn.Cell class.

    Attributes:
        num_heads (int): The number of attention heads.
        embed_dim (int): The dimension of the input embeddings.
        kernel_size (int): The size of the convolutional kernel.
        padding_q (int): The amount of padding for the query projection convolution.
        padding_kv (int): The amount of padding for the key and value projection convolutions.
        stride_q (int): The stride for the query projection convolution.
        stride_kv (int): The stride for the key and value projection convolutions.
        qkv_projection_method (str): The projection method used for the query, key, and value projections.
        qkv_bias (bool): Indicates whether bias is added to the query, key, and value projections.
        attention_drop_rate (float): The dropout rate for the attention scores.
        with_cls_token (bool): Indicates whether a classification token is included in the input.

    Methods:
        __init__(self, num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, with_cls_token=True, **kwargs):
            Initializes the CvtSelfAttention instance.

        rearrange_for_multi_head_attention(self, hidden_state):
            Rearranges the input hidden state for multi-head attention computations.

        construct(self, hidden_state, height, width):
            Constructs the CvtSelfAttention layer by performing convolutional projections, multi-head attention calculations, and output rearrangement.

    Note:
        - The CvtSelfAttention layer assumes that the input hidden state is a 4D tensor with shape (batch_size, hidden_size, height, width).
        - The attention_score and attention_probs computations make use of the Einstein summation convention (einsum).
        - The context output is a 3D tensor with shape (batch_size, hidden_size, num_heads * head_dim).
    """
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        with_cls_token=True,
        **kwargs,
    ):
        """
        __init__

        Initializes the CvtSelfAttention class.

        Args:
            self: The instance of the class.
            num_heads (int): The number of attention heads.
            embed_dim (int): The dimension of the input embeddings.
            kernel_size (int): The size of the convolutional kernel.
            padding_q (int): The padding size for the query projection.
            padding_kv (int): The padding size for the key and value projections.
            stride_q (int): The stride for the query projection.
            stride_kv (int): The stride for the key and value projections.
            qkv_projection_method (str): The method used for query, key, and value projections.
                Can be 'avg' or any other specific projection method.
            qkv_bias (bool): Indicates whether bias is applied to the query, key, and value projections.
            attention_drop_rate (float): The dropout rate for attention weights.
            with_cls_token (bool, optional): Indicates whether the class token is included. Defaults to True.

        Returns:
            None.

        Raises:
            ValueError: If embed_dim is not a positive integer.
            ValueError: If num_heads is not a positive integer.
            ValueError: If kernel_size, padding_q, padding_kv, stride_q, or stride_kv is not a positive integer.
            ValueError: If qkv_projection_method is not 'avg' or a valid specific projection method.
            ValueError: If attention_drop_rate is not in the range [0, 1].
            TypeError: If with_cls_token is not a boolean value.
        """
        super().__init__()
        self.scale = embed_dim**-0.5
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.convolution_projection_query = CvtSelfAttentionProjection(
            embed_dim,
            kernel_size,
            padding_q,
            stride_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
        )
        self.convolution_projection_key = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )
        self.convolution_projection_value = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )

        self.projection_query = nn.Dense(embed_dim, embed_dim, has_bias=qkv_bias)
        self.projection_key = nn.Dense(embed_dim, embed_dim, has_bias=qkv_bias)
        self.projection_value = nn.Dense(embed_dim, embed_dim, has_bias=qkv_bias)

        self.dropout = nn.Dropout(p=attention_drop_rate)

    def rearrange_for_multi_head_attention(self, hidden_state):
        """
        Method: rearrange_for_multi_head_attention

        In the class CvtSelfAttention, this method rearranges the hidden state tensor for multi-head attention computation.

        Args:
            self (CvtSelfAttention): The instance of the CvtSelfAttention class.
                This parameter is required for accessing the attributes and methods of the class.
            hidden_state (torch.Tensor):
                The input hidden state tensor of shape (batch_size, hidden_size, _).

                - batch_size (int): The number of sequences in the batch.
                - hidden_size (int): The dimensionality of the hidden state.
                - _ (int): Placeholder dimension for compatibility with the transformer architecture.

                This tensor represents the input hidden state that needs to be rearranged for multi-head attention computation.

        Returns:
            None:
                This method does not return any value.
                It rearranges the hidden state tensor in place and does not create a new tensor.

        Raises:
            None:
                This method does not explicitly raise any exceptions.
        """
        batch_size, hidden_size, _ = hidden_state.shape
        head_dim = self.embed_dim // self.num_heads
        # rearrange 'b t (h d) -> b h t d'
        return hidden_state.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def construct(self, hidden_state, height, width):
        """
        Constructs the self-attention context for the CvtSelfAttention class.

        Args:
            self: An instance of the CvtSelfAttention class.
            hidden_state (Tensor): The hidden state tensor of shape (batch_size, hidden_size, num_channels).
                It represents the input features.
            height (int): The height of the hidden state tensor.
            width (int): The width of the hidden state tensor.

        Returns:
            Tensor: The context tensor of shape (batch_size, hidden_size, num_heads * head_dim).
                It represents the output context after applying self-attention mechanism.

        Raises:
            None.
        """
        if self.with_cls_token:
            cls_token, hidden_state = ops.split(hidden_state, [1, height * width], 1)
        batch_size, hidden_size, num_channels = hidden_state.shape
        # rearrange "b (h w) c -> b c h w"
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)

        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)

        if self.with_cls_token:
            query = ops.cat((cls_token, query), axis=1)
            key = ops.cat((cls_token, key), axis=1)
            value = ops.cat((cls_token, value), axis=1)

        head_dim = self.embed_dim // self.num_heads

        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        attention_score = ops.einsum("bhlk,bhtk->bhlt", query, key) * self.scale
        attention_probs = ops.softmax(attention_score, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context = ops.einsum("bhlt,bhtv->bhlv", attention_probs, value)
        # rearrange"b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).view(batch_size, hidden_size, self.num_heads * head_dim)
        return context


class CvtSelfOutput(nn.Cell):
    """
    The residual connection is defined in CvtLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, embed_dim, drop_rate):
        """
        Initializes an instance of the CvtSelfOutput class.

        Args:
            self (CvtSelfOutput): The instance of the class.
            embed_dim (int): The dimension of the embedding.
            drop_rate (float): The dropout rate to be applied.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=drop_rate)

    def construct(self, hidden_state, input_tensor):
        """
        Constructs the output of the CvtSelfOutput class.

        Args:
            self (CvtSelfOutput): An instance of the CvtSelfOutput class.
            hidden_state (Tensor): The hidden state to be processed.
                This tensor represents the current state of the model and is expected to have shape (batch_size, hidden_size).
                It serves as input to the dense layer and will be transformed.
            input_tensor (Tensor): The input tensor to the method.
                This tensor represents additional input to the construct method and can be of any shape.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtAttention(nn.Cell):

    """
    This class represents an attention mechanism for the Cvt model.
    It includes methods for initializing the attention mechanism, pruning specific attention heads,
    and constructing the attention output.

    Attributes:
        num_heads (int): Number of attention heads.
        embed_dim (int): Dimension of the input embeddings.
        kernel_size (int): Size of the convolutional kernel.
        padding_q (int): Padding size for query tensor.
        padding_kv (int): Padding size for key and value tensors.
        stride_q (int): Stride size for query tensor.
        stride_kv (int): Stride size for key and value tensors.
        qkv_projection_method (str): Method for projecting query, key, and value tensors.
        qkv_bias (bool): Whether to include bias in query, key, and value projections.
        attention_drop_rate (float): Dropout rate for attention scores.
        drop_rate (float): Dropout rate for output.
        with_cls_token (bool): Whether to include a classification token in the input.

    Methods:
        __init__(num_heads, embed_dim, kernel_size, padding_q, padding_kv, stride_q, stride_kv, qkv_projection_method, qkv_bias, attention_drop_rate, drop_rate, with_cls_token=True):
            Initializes the attention mechanism with the given parameters.

        prune_heads(heads):
            Prunes specified attention heads based on the provided indices.

        construct(hidden_state, height, width):
            Constructs the attention output using the input hidden state and spatial dimensions.

    Inherits from:
        nn.Cell
    """
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        with_cls_token=True,
    ):
        """
        Initializes a CvtAttention instance with the specified parameters.

        Args:
            self (CvtAttention): The current instance of the CvtAttention class.
            num_heads (int): The number of attention heads to use.
            embed_dim (int): The dimension of the input embeddings.
            kernel_size (int): The size of the convolutional kernel.
            padding_q (int): Padding size for query tensor.
            padding_kv (int): Padding size for key and value tensors.
            stride_q (int): Stride size for query tensor.
            stride_kv (int): Stride size for key and value tensors.
            qkv_projection_method (str): The method used for query, key, value projection.
            qkv_bias (bool): Flag indicating whether to include bias in query, key, value projection.
            attention_drop_rate (float): The dropout rate applied to attention weights.
            drop_rate (float): The dropout rate applied to the output.
            with_cls_token (bool): Flag indicating whether to include a classification token.

        Returns:
            None.

        Raises:
            ValueError: If num_heads is not a positive integer.
            ValueError: If embed_dim is not a positive integer.
            ValueError: If kernel_size is not a positive integer.
            ValueError: If padding_q is not a non-negative integer.
            ValueError: If padding_kv is not a non-negative integer.
            ValueError: If stride_q is not a positive integer.
            ValueError: If stride_kv is not a positive integer.
            ValueError: If qkv_projection_method is not a string.
            ValueError: If attention_drop_rate is not in the range [0, 1].
            ValueError: If drop_rate is not in the range [0, 1].
        """
        super().__init__()
        self.attention = CvtSelfAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
        )
        self.output = CvtSelfOutput(embed_dim, drop_rate)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'CvtAttention' and is used to prune the attention
        heads based on the provided 'heads' parameter.

        Args:
            self (object): The instance of the 'CvtAttention' class.
            heads (list): A list containing the indices of attention heads to be pruned.
                If the list is empty, no pruning is performed.

        Returns:
            None.

        Raises:
            ValueError: If the length of the 'heads' list is invalid or if any of the provided indices are out of range.
            TypeError: If the 'heads' parameter is not a list or if any of the internal operations encounter unexpected data types.
            RuntimeError: If an unexpected error occurs during the pruning process.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(self, hidden_state, height, width):
        """
        Constructs an attention output based on the given hidden state, height, and width.

        Args:
            self (CvtAttention): An instance of the CvtAttention class.
            hidden_state: The hidden state used for attention computation.
            height (int): The height of the attention output.
            width (int): The width of the attention output.

        Returns:
            None.

        Raises:
            None.
        """
        self_output = self.attention(hidden_state, height, width)
        attention_output = self.output(self_output, hidden_state)
        return attention_output


class CvtIntermediate(nn.Cell):

    """
    Represents an intermediate layer in a Convolutional Vision Transformer (CVT) network.

    This class defines an intermediate layer in a CVT network that consists of a dense layer followed by
    a GELU activation function. The intermediate layer is used to process the hidden states in the network.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
        mlp_ratio (float): The ratio used to determine the size of the hidden layer in the dense layer.

    Methods:
        __init__(self, embed_dim, mlp_ratio):
            Initializes the CvtIntermediate object with the specified embedding dimension and MLP ratio.
        construct(self, hidden_state):
            Constructs the intermediate layer by applying a dense layer and GELU activation function to the input hidden state.

    Inherits from:
        nn.Cell
    """
    def __init__(self, embed_dim, mlp_ratio):
        """
        Initializes an instance of the CvtIntermediate class.

        Args:
            self (CvtIntermediate): The instance of the class.
            embed_dim (int): The dimension of the embedding.
            mlp_ratio (float): The ratio used to calculate the hidden dimension of the MLP.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(embed_dim, int(embed_dim * mlp_ratio))
        self.activation = nn.GELU()

    def construct(self, hidden_state):
        """
        Constructs the hidden state of the CvtIntermediate class.

        Args:
            self (CvtIntermediate): An instance of the CvtIntermediate class.
            hidden_state: The hidden state to be processed. It should be a tensor or array-like object.

        Returns:
            None: This method modifies the hidden state in-place.

        Raises:
            None.

        This method takes in the 'hidden_state' and applies transformations to it in order to construct the
        hidden state of the CvtIntermediate class.
        The 'hidden_state' is first passed through a dense layer using the 'self.dense' function.
        Then, the resulting tensor is passed through the activation function specified by the 'self.activation' attribute.
        The modified hidden state is returned as the output of this method.

        Note that this method modifies the hidden state in-place and does not create a new object.
        """
        hidden_state = self.dense(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class CvtOutput(nn.Cell):

    """
    The 'CvtOutput' class represents a conversion output module that is used in neural network models.

    This class inherits from the 'nn.Cell' class, which is a base class for all neural network cells in the MindSpore framework.

    Methods:
        __init__(self, embed_dim, mlp_ratio, drop_rate):
            Initializes a new instance of the 'CvtOutput' class.

            Args:

            - embed_dim (int): The dimension of the embedded vectors.
            - mlp_ratio (float): The ratio used to calculate the dimension of the MLP intermediate layer.
            - drop_rate (float): The probability of an element to be zeroed in the dropout layer.

        construct(self, hidden_state, input_tensor):
            Constructs the conversion output module by applying operations to the input tensors.

            Args:

            - hidden_state (Tensor): The hidden state tensor.
            - input_tensor (Tensor): The input tensor.

            Returns:

            - Tensor: The final hidden state tensor obtained after applying the conversion operations.
    """
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        """
        Initialize the CvtOutput class.

        Args:
            self: The instance of the class.
            embed_dim (int): The dimension of the embedding.
            mlp_ratio (float): The ratio used to determine the hidden layer size in the MLP.
            drop_rate (float): The dropout rate applied to the output.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(int(embed_dim * mlp_ratio), embed_dim)
        self.dropout = nn.Dropout(p=drop_rate)

    def construct(self, hidden_state, input_tensor):
        """
        Constructs the output of the CvtOutput class.

        Args:
            self (CvtOutput): An instance of the CvtOutput class.
            hidden_state (tensor): The hidden state tensor.
                This tensor represents the current state of the model and is used as input for further processing.
                It should have a shape compatible with the dense layer.
            input_tensor (tensor): The input tensor.
                This tensor represents the input data and is added to the hidden state tensor.
                It should have the same shape as the hidden state tensor.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = hidden_state + input_tensor
        return hidden_state


class CvtLayer(nn.Cell):
    """
    CvtLayer composed by attention layers, normalization and multi-layer perceptrons (mlps).
    """
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        mlp_ratio,
        drop_path_rate,
        with_cls_token=True,
    ):
        """
        Initializes an instance of the CvtLayer class.

        Args:
            self: The object instance.
            num_heads (int): The number of attention heads.
            embed_dim (int): The dimensionality of the embedding.
            kernel_size (int): The kernel size for the attention computation.
            padding_q (int): The padding size for queries.
            padding_kv (int): The padding size for key and value.
            stride_q (int): The stride size for queries.
            stride_kv (int): The stride size for key and value.
            qkv_projection_method (str): The method used for query, key, and value projection.
            qkv_bias (bool): Whether to include bias in query, key, and value projection.
            attention_drop_rate (float): The dropout rate for attention weights.
            drop_rate (float): The dropout rate for the output tensor.
            mlp_ratio (float): The ratio of the hidden size to the input size in the intermediate layer.
            drop_path_rate (float): The dropout rate for the residual connection.
            with_cls_token (bool): Whether to include a classification token.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = CvtAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
        )

        self.intermediate = CvtIntermediate(embed_dim, mlp_ratio)
        self.output = CvtOutput(embed_dim, mlp_ratio, drop_rate)
        self.drop_path = CvtDropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_before = nn.LayerNorm(embed_dim)
        self.layernorm_after = nn.LayerNorm(embed_dim)

    def construct(self, hidden_state, height, width):
        """
        This method constructs a layer in the CvtLayer class.

        Args:
            self (object): The instance of the CvtLayer class.
            hidden_state (tensor): The hidden state of the layer.
            height (int): The height of the input tensor.
            width (int): The width of the input tensor.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_state is not a valid tensor.
            TypeError: If height and width are not integer values.
            RuntimeError: If an unexpected error occurs during the execution of the method.
        """
        self_attention_output = self.attention(
            self.layernorm_before(hidden_state),  # in Cvt, layernorm is applied before self-attention
            height,
            width,
        )
        attention_output = self_attention_output
        attention_output = self.drop_path(attention_output)

        # first residual connection
        hidden_state = attention_output + hidden_state

        # in Cvt, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output)
        return layer_output


class CvtStage(nn.Cell):

    """
    The CvtStage class represents a stage in the Cross Vision Transformer (Cvt) model. It inherits from nn.Cell and
    is designed to handle the processing and transformation of input data within a specific stage of the Cvt model.

    This class includes methods for initializing the stage with configuration and stage information, as well as
    constructing the hidden state through a series of operations involving embeddings, layer processing,
    and token manipulation.

    The class supports the configuration of parameters such as patch size, stride, number of channels,
    embedding dimensions, padding, dropout rates, depth, number of heads, kernel size, attention and
    multi-layer perceptron (MLP) settings, and the inclusion of a classification (cls) token.

    The construct method is responsible for processing the hidden state by applying the configured embeddings,
    manipulating the hidden state based on the existence of a cls token, and iterating through the
    layers to transform the hidden state. Additionally, it handles the splitting and reshaping of the hidden state
    before returning the updated hidden state and cls token.

    Overall, the CvtStage class provides a structured and configurable framework for managing the transformation of
    data within a specific stage of the Cvt model.
    """
    def __init__(self, config, stage):
        """
        This method initializes an instance of the CvtStage class.

        Args:
            self: The instance of the CvtStage class.
            config (object): The configuration object containing various parameters such as patch size, stride,
                number of channels, embedding dimensions, padding, dropout rate, depth, number of heads, kernel
                size, padding for query, key, and value, stride for key and value, stride for query, method
                for QKV projection, QKV bias, attention dropout rate, drop rate, drop path rate, MLP ratio,
                and presence of a classification token.
            stage (int): The stage of the CvtStage.

        Returns:
            None.

        Raises:
            ValueError: If the config.cls_token[self.stage] does not exist or is not a valid value.
            TypeError: If the config.drop_path_rate[self.stage] is not a valid type.
            IndexError: If the drop_path_rates[self.stage] does not exist or is not a valid index.
            TypeError: If any of the parameters in the CvtLayer instantiation are of an invalid type.
        """
        super().__init__()
        self.config = config
        self.stage = stage
        if self.config.cls_token[self.stage]:
            self.cls_token = Parameter(ops.randn(1, 1, self.config.embed_dim[-1]))

        self.embedding = CvtEmbeddings(
            patch_size=config.patch_sizes[self.stage],
            stride=config.patch_stride[self.stage],
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],
            embed_dim=config.embed_dim[self.stage],
            padding=config.patch_padding[self.stage],
            dropout_rate=config.drop_rate[self.stage],
        )

        drop_path_rates = [x.item() for x in ops.linspace(0, config.drop_path_rate[self.stage], config.depth[stage])]

        self.layers = nn.SequentialCell(
            *[
                CvtLayer(
                    num_heads=config.num_heads[self.stage],
                    embed_dim=config.embed_dim[self.stage],
                    kernel_size=config.kernel_qkv[self.stage],
                    padding_q=config.padding_q[self.stage],
                    padding_kv=config.padding_kv[self.stage],
                    stride_kv=config.stride_kv[self.stage],
                    stride_q=config.stride_q[self.stage],
                    qkv_projection_method=config.qkv_projection_method[self.stage],
                    qkv_bias=config.qkv_bias[self.stage],
                    attention_drop_rate=config.attention_drop_rate[self.stage],
                    drop_rate=config.drop_rate[self.stage],
                    drop_path_rate=drop_path_rates[self.stage],
                    mlp_ratio=config.mlp_ratio[self.stage],
                    with_cls_token=config.cls_token[self.stage],
                )
                for _ in range(config.depth[self.stage])
            ]
        )

    def construct(self, hidden_state):
        """
        Constructs the hidden state for the CvtStage class.

        Args:
            self (CvtStage): The instance of the CvtStage class.
            hidden_state: The hidden state input for constructing the hidden state. It should be a tensor.

        Returns:
            tuple:
                A tuple containing the constructed hidden state and cls_token.

                The hidden state is a tensor with dimensions (batch_size, num_channels, height, width), representing
                the constructed hidden state. The cls_token is a tensor with dimensions (batch_size, 1, num_channels),
                representing the cls_token if it exists, otherwise it is None.

        Raises:
            None.
        """
        cls_token = None
        hidden_state = self.embedding(hidden_state)
        batch_size, num_channels, height, width = hidden_state.shape
        # rearrange b c h w -> b (h w) c"
        hidden_state = hidden_state.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        if self.config.cls_token[self.stage]:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            hidden_state = ops.cat((cls_token, hidden_state), axis=1)

        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width)
            hidden_state = layer_outputs

        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = ops.split(hidden_state, [1, height * width], 1)
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        return hidden_state, cls_token


class CvtEncoder(nn.Cell):

    """
    This class represents a converter encoder used for converting pixel values to hidden states. It is a subclass of nn.Cell.

    Attributes:
        config (Config): The configuration object for the CvtEncoder.
        stages (nn.CellList): A list of CvtStage instances representing the stages of the converter encoder.

    Methods:
        __init__(self, config)
            Initializes a new instance of the CvtEncoder class.

            Args:

            - config (Config): The configuration object for the CvtEncoder.

        construct(self, pixel_values, output_hidden_states=False, return_dict=True)
            Constructs the converter encoder model.

            Args:

            - pixel_values (tensor): The input pixel values.
            - output_hidden_states (bool): Whether to output all hidden states. Defaults to False.
            - return_dict (bool): Whether to return the model output as a dictionary. Defaults to True.

            Returns:

            - BaseModelOutputWithCLSToken: The model output containing the last hidden state, the cls token value, and
            all hidden states.
    """
    def __init__(self, config):
        """
        Initializes an instance of the CvtEncoder class.

        Args:
            self: The instance of the class.
            config (object): The configuration object that holds the parameters for the encoder.
                This object is used to configure the behavior of the encoder.
                It must be an instance of the Config class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.stages = nn.CellList([])
        for stage_idx in range(len(config.depth)):
            self.stages.append(CvtStage(config, stage_idx))

    def construct(self, pixel_values, output_hidden_states=False, return_dict=True):
        """
        Constructs the CvTEncoder.

        Args:
            self (CvtEncoder): The instance of the CvtEncoder class.
            pixel_values (Any): The input pixel values.
            output_hidden_states (bool): Whether to output hidden states or not. Defaults to False.
            return_dict (bool): Whether to return the result as a dictionary or not. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values

        cls_token = None
        for _, (stage_module) in enumerate(self.stages):
            hidden_state, cls_token = stage_module(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        return BaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )


class CvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CvtConfig
    base_model_prefix = "cvt"
    main_input_name = "pixel_values"
    _no_split_modules = ["CvtLayer"]
    _keys_to_ignore_on_load_unexpected = [r'num_batches_tracked']

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Dense, nn.Conv2d)):
            module.weight.initialize(TruncatedNormal(self.config.initializer_range))
            if module.bias is not None:
                module.bias.initialize('zeros')
        elif isinstance(module, nn.LayerNorm):
            module.bias.initialize('zeros')
            module.weight.initialize('ones')
        elif isinstance(module, CvtStage):
            if self.config.cls_token[module.stage]:
                module.cls_token.initialize(TruncatedNormal(self.config.initializer_range))


class CvtModel(CvtPreTrainedModel):

    """
    CvtModel is a model class that represents a Convolutional Vision Transformer (Cvt) model for processing visual data.
    This class inherits from CvtPreTrainedModel and provides functionalities for initializing the model, pruning heads,
    and constructing the model output.

    Attributes:
        config (CvtConfig): The configuration object for the model.
        encoder (CvtEncoder): The encoder component of the CvtModel responsible for processing input data.

    Methods:
        __init__:
            Initializes the CvtModel instance with the provided configuration.

        _prune_heads:
            Prunes specified heads of the model based on the provided dictionary of layer numbers and heads to prune.

        construct:
            Constructs the model output by processing the input pixel values and returning the output hidden states.
            If pixel_values is not provided, a ValueError is raised.
            The output format is determined based on the return_dict flag and the model configuration.
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the CvtModel class.

        Args:
            self (object): The instance of the CvtModel class.
            config (object): The configuration object containing model settings and parameters.
            add_pooling_layer (bool, optional): A flag indicating whether to add a pooling layer. Default is True.

        Returns:
            None.

        Raises:
            ValueError: If the provided config is invalid or missing required parameters.
            TypeError: If the provided config is not of the expected type.
            RuntimeError: If an error occurs during initialization.
        """
        super().__init__(config)
        self.config = config
        self.encoder = CvtEncoder(config)
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCLSToken]:
        """
        Constructs the CvtModel.

        Args:
            self (CvtModel): The instance of the CvtModel class.
            pixel_values (Optional[mindspore.Tensor]): The pixel values of the input image. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary output. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithCLSToken]:
                The constructed model output.

                - If `return_dict` is False, a tuple is returned containing the sequence output and any additional
                encoder outputs.
                - If `return_dict` is True, a BaseModelOutputWithCLSToken object is returned, which includes
                the last hidden state, cls token value, and hidden states.

        Raises:
            ValueError: If `pixel_values` is not specified.

        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )


class CvtForImageClassification(CvtPreTrainedModel):

    """
    CvtForImageClassification is a class that represents a model for image classification utilizing the Cvt architecture.
    It inherits from the CvtPreTrainedModel class and provides methods for constructing the model and computing
    image classification/regression loss.

    Attributes:
        num_labels (int): Number of labels for classification
        cvt (CvtModel): CvtModel instance used for image processing
        layernorm (nn.LayerNorm): Layer normalization module
        classifier (nn.Dense or nn.Identity): Classifier module for final predictions

    Methods:
        __init__(self, config): Initializes the CvtForImageClassification model with the provided configuration.
        construct(self, pixel_values, labels, output_hidden_states, return_dict):
            Constructs the model and computes loss for image classification.

            Parameters:

            - pixel_values (Optional[mindspore.Tensor]): Tensor containing pixel values of images
            - labels (Optional[mindspore.Tensor]): Tensor containing labels for computing classification/regression loss
            - output_hidden_states (Optional[bool]): Flag to indicate whether to output hidden states
            - return_dict (Optional[bool]): Flag to indicate whether to return output as a dictionary

    Returns:
        Union[Tuple, ImageClassifierOutputWithNoAttention]: Tuple containing loss and output if return_dict is False.
          Otherwise, returns an ImageClassifierOutputWithNoAttention instance.

    Notes:
        - The 'construct' method handles the processing of input pixel values, computation of logits,
        and determination of loss based on the configuration settings.
        - The loss calculation depends on the problem type (regression, single_label_classification,
        or multi_label_classification) and the number of labels.
        - The final output includes logits and optionally hidden states depending on the return_dict flag.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the CvtForImageClassification class.

        Args:
            self: The object itself.
            config:
                An instance of the class Config containing the configuration settings.

                - Type: object
                - Purpose: Stores the configuration settings for the model.
                - Restrictions: Must be a valid instance of the Config class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.cvt = CvtModel(config, add_pooling_layer=False)
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])
        # Classifier head
        self.classifier = (
            nn.Dense(config.embed_dim[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        cls_token = outputs[1]
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # rearrange "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)

        sequence_output_mean = sequence_output.mean(axis=1)
        logits = self.classifier(sequence_output_mean)

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
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

__all__ = [
    "CvtForImageClassification",
    "CvtModel",
    "CvtPreTrainedModel",
]
