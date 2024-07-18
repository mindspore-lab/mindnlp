# coding=utf-8
# Copyright 2021 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore BEiT model."""


import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...backbone_utils import BackboneMixin
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from .configuration_beit import BeitConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "BeitConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/beit-base-patch16-224-pt22k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/beit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

BEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/beit-base-patch16-224",
    # See all BEiT models at https://hf-mirror.com/models?filter=beit
]


@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Class for outputs of [`BeitModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
            *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
            will be returned.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
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


class BeitDropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        """
        Initializes an instance of the BeitDropPath class.
        
        Args:
            self: The instance of the BeitDropPath class.
            drop_prob (Optional[float]): The probability of dropping a connection. 
                It is a floating point number representing the probability of dropping a connection during training. 
                If not provided, the default value is None.
        
        Returns:
            None.
        
        Raises:
            None
        """
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a new tensor by applying drop path regularization to the input hidden states.
        
        Args:
            self (BeitDropPath): An instance of the BeitDropPath class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states of the model.
                Expected to be a tensor of shape (batch_size, sequence_length, hidden_size).
                The hidden states to which drop path regularization will be applied.
        
        Returns:
            mindspore.Tensor: A new tensor resulting from applying drop path regularization to the input hidden states.
                The same shape as the input tensor (hidden_states).
        
        Raises:
            ValueError: If the drop probability is not within the range [0, 1].
            TypeError: If the input hidden_states is not a valid mindspore.Tensor object.
        """
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """
        This method generates a string representation for the 'BeitDropPath' class instance.
        
        Args:
            self: 'BeitDropPath' instance. Represents the current instance of the 'BeitDropPath' class.
        
        Returns:
            str: A string representation of the 'BeitDropPath' instance with the drop probability value.
        
        Raises:
            None.
        """
        return "p={}".format(self.drop_prob)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class BeitEmbeddings(nn.Cell):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """
    def __init__(self, config: BeitConfig) -> None:
        """
        Initialize the BeitEmbeddings class.
        
        Args:
            self: The instance of the BeitEmbeddings class.
            config (BeitConfig): An instance of the BeitConfig class containing configuration parameters for the embeddings.
                This parameter is required for initializing the embeddings.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()

        self.cls_token = Parameter(ops.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = Parameter(ops.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = Parameter(ops.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, pixel_values: mindspore.Tensor, bool_masked_pos: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        """
        Construct method in the BeitEmbeddings class.
        
        This method constructs embeddings for the input pixel values and optional masked positions.
        
        Args:
            self: The object itself.
            pixel_values (mindspore.Tensor): The input pixel values for which embeddings need to be constructed.
            bool_masked_pos (Optional[mindspore.Tensor]): An optional tensor containing boolean masked positions. Defaults to None.
        
        Returns:
            mindspore.Tensor: The constructed embeddings as a tensor.
            Tuple[int, int]: A tuple containing the height and width of the patches.
        
        Raises:
            None.
        """
        embeddings, (patch_height, patch_width) = self.patch_embeddings(
            pixel_values, self.position_embeddings[:, 1:, :] if self.position_embeddings is not None else None
        )
        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.position_embeddings is not None:
            cls_tokens = cls_tokens + self.position_embeddings[:, :1, :]

        embeddings = ops.cat((cls_tokens, embeddings), axis=1)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)


class BeitPatchEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config):
        """
        Args:
            self (object): The instance of the BeitPatchEmbeddings class.
            config (object): 
                An object containing configuration parameters for the patch embeddings.
                
                - image_size (int or tuple): The size of the input image. 
                
                    - If int, it represents the height and width of the square image. 
                    - If tuple, it represents the height and width of the rectangular image.

                - patch_size (int or tuple): The size of the patch to be extracted from the input image.
                
                    - If int, it represents the height and width of the square patch. 
                    - If tuple, it represents the height and width of the rectangular patch.

                - num_channels (int): The number of input channels in the image.
                - hidden_size (int): The desired size of the hidden dimension for the patch embeddings.

        Returns:
            None.

        Raises:
            TypeError: If the image_size or patch_size parameters are not of type int or tuple.
            ValueError: If the image_size or patch_size parameters are not valid for calculating the number of patches.
            ValueError: If the patch_size does not evenly divide the image_size, resulting in incomplete patches.
        """
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def construct(self, pixel_values: mindspore.Tensor, position_embedding: Optional[mindspore.Tensor] = None) -> mindspore.Tensor:
        '''
        Constructs patch embeddings for the Beit model.

        Args:
            self (BeitPatchEmbeddings): The instance of the BeitPatchEmbeddings class.
            pixel_values (mindspore.Tensor): 
                The input tensor representing the pixel values of the image with shape (batch_size, num_channels, height, width).
            position_embedding (Optional[mindspore.Tensor]): 
                An optional input tensor representing the position embeddings with shape (1, patch_height, patch_width, -1). 
                Default is None.

        Returns:
            mindspore.Tensor: A tensor representing the constructed patch embeddings with shape (batch_size, num_patches, embedding_dim).
            tuple: A tuple containing the height and width of the patches.

        Raises:
            ValueError: If the number of channels in the pixel_values tensor does not match the num_channels set in the configuration.
        '''
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values)
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]

        if position_embedding is not None:
            # interpolate the position embedding to the corresponding size
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(
                0, 3, 1, 2
            )
            position_embedding = ops.interpolate(
                position_embedding, size=(patch_height, patch_width), mode="bicubic"
            )
            embeddings = embeddings + position_embedding

        embeddings = embeddings.flatten(start_dim=2).swapaxes(1, 2)

        return embeddings, (patch_height, patch_width)


class BeitSelfAttention(nn.Cell):

    """
    Represents the self-attention mechanism for the Beit model.

    This class implements the self-attention mechanism for the Beit model, 
    which is used to process input data and compute the output context layer. 
    It includes methods for initializing the self-attention layer, swapping axes for attention scores, 
    and constructing the self-attention mechanism based on the input data and optional parameters.

    Attributes:
        num_attention_heads (int): Number of attention heads in the self-attention mechanism.
        attention_head_size (int): Size of each attention head.
        all_head_size (int): Total size of all attention heads combined.
        query (nn.Dense): Fully connected layer for generating query vectors.
        key (nn.Dense): Fully connected layer for generating key vectors.
        value (nn.Dense): Fully connected layer for generating value vectors.
        dropout (nn.Dropout): Dropout layer for attention probabilities.
        relative_position_bias (BeitRelativePositionBias): Relative position bias for 
            incorporating positional information in attention scores.

    Methods:
        swapaxes_for_scores(x): Method for swapping axes in the input tensor to calculate attention scores.
        construct(hidden_states, head_mask, output_attentions, relative_position_bias): Method for constructing 
            the self-attention mechanism based on input data and optional parameters.

    Raises:
        ValueError: If the hidden size is not a multiple of the number of attention heads.

    Returns:
        Tuple[mindspore.Tensor]: Tuple containing the context layer and optionally the attention probabilities.
    """
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        """
        Initializes the BeitSelfAttention instance.

        Args:
            self: The instance of the BeitSelfAttention class.
            config (BeitConfig): The configuration for the Beit model. 
                It specifies the hidden size, number of attention heads, and other relevant parameters.
            window_size (Optional[tuple]): The size of the window for relative position bias. 
                Defaults to None if not provided.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads 
                and the 'embedding_size' attribute is not present in the config.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size, has_bias=False)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

        if window_size:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

    def swapaxes_for_scores(self, x):
        """
        This method 'swapaxes_for_scores' is defined in the class 'BeitSelfAttention' and is used to perform 
        a specific operation on the input tensor 'x'.

        Args:
            self: Represents the instance of the class. 
                It is automatically passed to the method when it is called. 
                The 'self' parameter is used to access and modify class attributes and methods.
            x: Represents the input tensor on which the operation will be performed. 
                It is expected to be a multi-dimensional array. 
                The shape of the input tensor should be compatible with the specified operations. 
                No specific restrictions apply to this parameter.

        Returns:
            a tensor of the same shape as the input tensor 'x':
                but with the axes swapped according to the defined logic. 
                The return value is of type 'None' because the operation is performed in place on the input tensor.

        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """Constructs the self-attention mechanism for the Beit model.

        Args:
            self (BeitSelfAttention): The instance of the BeitSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states to the self-attention mechanism.
            head_mask (Optional[mindspore.Tensor], optional): 
                An optional mask tensor for masking certain attention heads. Defaults to None.
            output_attentions (bool, optional): 
                A flag indicating whether to output the attention scores. Defaults to False.
            relative_position_bias (Optional[BeitRelativePositionBias], optional): 
                An optional relative position bias tensor. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]: 
                tuple containing the context layer tensor and optionally the attention scores tensor.

        Raises:
            ValueError: If the attention scores or context layer computation encounters an invalid dimensionality mismatch.
            RuntimeError: If an error occurs during the computation of the attention mechanism or context layer.
            TypeError: If the input parameters are of incorrect types or incompatible with the operations within the method.
        """
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.swapaxes_for_scores(self.key(hidden_states))
        value_layer = self.swapaxes_for_scores(self.value(hidden_states))
        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias().unsqueeze(0)

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class BeitSelfOutput(nn.Cell):
    """
    The residual connection is defined in BeitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes a BeitSelfOutput object with the provided configuration.

        Args:
            self (BeitSelfOutput): The instance of the BeitSelfOutput class.
            config (BeitConfig):
                An instance of BeitConfig containing the configuration parameters for the self output layer.
            
                - hidden_size (int): The size of the hidden layer.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor, gamma=None) -> mindspore.Tensor:
        """
        This method 'construct' is defined within the 'BeitSelfOutput' class and is responsible for 
        constructing the hidden states using the provided input tensors and optional gamma parameter.

        Args:
            self: The instance of the 'BeitSelfOutput' class.
            hidden_states (mindspore.Tensor): A tensor representing the hidden states to be processed.
            input_tensor (mindspore.Tensor): A tensor representing the input data used in the construction process.
            gamma (optional): A parameter that can be provided to the method (default: None).

        Returns:
            mindspore.Tensor: Returns a tensor representing the constructed hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitAttention(nn.Cell):

    """
    The `BeitAttention` class represents the attention mechanism for the Beit model, inheriting from `nn.Cell`. 
    It includes methods for initializing the attention mechanism, pruning attention heads, and constructing 
    the attention output.

    Attributes:
        attention: Instance of `BeitSelfAttention` representing the self-attention mechanism.
        output: Instance of `BeitSelfOutput` representing the output of the self-attention mechanism.
        pruned_heads: Set containing the indices of pruned attention heads.

    Methods:
        __init__:
            Initializes the `BeitAttention` instance with the provided configuration and optional window size.

        prune_heads(self, heads):
            Prunes the specified attention heads from the self-attention mechanism.

        construct:
            Constructs the attention output using the provided hidden states and optional masks or biases.
    """
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        """
        Initializes a new instance of the BeitAttention class.

        Args:
            self: The object itself.
            config (BeitConfig): The configuration object for the Beit model.
            window_size (Optional[tuple]): The size of the attention window. Defaults to None.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.attention = BeitSelfAttention(config, window_size=window_size)
        self.output = BeitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes the attention heads in the BeitAttention module.

        Args:
            self (BeitAttention): An instance of the BeitAttention module.
            heads (List[int]): A list of attention heads to be pruned.

        Returns:
            None.

        Raises:
            None.
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

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Constructs the attention mechanism for the BeitAttention class.

        Args:
            self: The instance of the BeitAttention class.
            hidden_states (mindspore.Tensor): The input hidden states for the attention mechanism.
            head_mask (Optional[mindspore.Tensor]): 
                Optional tensor for masking certain heads in the attention mechanism. Defaults to None.
            output_attentions (bool): 
                Flag to indicate whether to output attention scores. Defaults to False.
            relative_position_bias (Optional[BeitRelativePositionBias]): 
                Optional relative position bias for the attention mechanism. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:
                The output of the attention mechanism, which is a tuple containing attention output and other optional outputs.

        Raises:
            ValueError: If the input hidden_states tensor is not valid.
            TypeError: If the head_mask or relative_position_bias parameters have an invalid type.
            RuntimeError: If there is an issue during the attention mechanism processing.
        """
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Cell):

    """
    This class represents an intermediate layer of the Beit model. 
    It is a subclass of nn.Cell and is responsible for processing the hidden states of the model.

    Attributes:
        dense (nn.Dense): A fully connected layer that transforms the input hidden states.
        intermediate_act_fn (callable): The activation function applied to the transformed hidden states.

    Methods:
        __init__:
            Initializes a new instance of the BeitIntermediate class.

        construct:
            Processes the input hidden states by applying the dense layer and the intermediate activation function.

    """
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes an instance of the 'BeitIntermediate' class.

        Args:
            self: The current instance of the class.
            config (BeitConfig): An object containing the configuration settings for the 'BeitIntermediate' class.

        Returns:
            None

        Raises:
            None

        Description:
            This method is the constructor for the 'BeitIntermediate' class. 
            It initializes an instance of the class and sets up the necessary attributes.

            The 'config' parameter is an object of the 'BeitConfig' class and contains the configuration settings 
            for the 'BeitIntermediate' class. 
            It is required for proper initialization.

            The 'self.dense' attribute is an instance of the 'nn.Dense' class, which is used for dense layer operations. 
            It is initialized with the 'hidden_size' and 'intermediate_size' values from the 'config' object.

            The 'self.intermediate_act_fn' attribute is a callable object that represents the activation function for 
            the intermediate layer. It is determined by the 'hidden_act' attribute of the 'config' object. 
            If 'hidden_act' is a string, it is looked up in the 'ACT2FN' dictionary to obtain the corresponding 
            activation function. Otherwise, it is directly assigned to 'self.intermediate_act_fn'.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the intermediate layer of the Beit model.

        Args:
            self (BeitIntermediate): The instance of the BeitIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The tensor representing the intermediate states of the Beit model.
                It has the same shape as the input hidden states tensor.

        Raises:
            None: This method does not raise any exceptions.

        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BeitOutput(nn.Cell):

    """
    This class represents the output layer of the Beit model, inheriting from the nn.Cell class.

    The BeitOutput class applies a dense layer and a dropout layer to the input hidden states. These layers are configured based on the provided BeitConfig.

    Attributes:
        dense (nn.Dense): The dense layer used for transforming the hidden states.
        dropout (nn.Dropout): The dropout layer used for regularization.

    Methods:
        construct:
            Applies the dense and dropout layers to the input hidden states and returns the transformed hidden states.

    Example:
        >>> config = BeitConfig(...)
        >>> output_layer = BeitOutput(config)
        >>> hidden_states = mindspore.Tensor(...)
        >>> output = output_layer.construct(hidden_states)

    Note:
        This class assumes that BeitConfig has been properly initialized with the necessary configuration parameters.
    """
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes a new instance of the BeitOutput class.

        Args:
            self: The object instance.
            config (BeitConfig): The configuration object for the Beit model, 
                containing parameters such as intermediate_size and hidden_size.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the output tensor for the Beit model.

        Args:
            self (BeitOutput): The instance of the BeitOutput class.
            hidden_states (mindspore.Tensor): The input tensor containing the hidden states.

        Returns:
            mindspore.Tensor: The output tensor representing the constructed hidden states.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitLayer(nn.Cell):
    """This corresponds to the Block class in the timm implementation."""
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0) -> None:
        """
        Initializes a BeitLayer object.

        Args:
            self: The instance of the class.
            config (BeitConfig): The configuration object containing model parameters.
            window_size (Optional[tuple]): The window size for the attention mechanism. Default is None.
            drop_path_rate (float): The rate at which to apply drop path regularization. Default is 0.0.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type BeitConfig.
            ValueError: If the drop_path_rate is negative.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BeitAttention(config, window_size=window_size)
        self.intermediate = BeitIntermediate(config)
        self.output = BeitOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.drop_path = BeitDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = Parameter(init_values * ops.ones((config.hidden_size)), requires_grad=True)
            self.lambda_2 = Parameter(init_values * ops.ones((config.hidden_size)), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """Constructs the BeitLayer.

        Args:
            self (object): The instance of the BeitLayer class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            head_mask (Optional[mindspore.Tensor]): An optional tensor representing the head mask. Defaults to None.
            output_attentions (bool): A boolean flag indicating whether to output attentions. Defaults to False.
            relative_position_bias (Optional[BeitRelativePositionBias]): 
                An optional relative position bias. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:

                - A tuple containing the layer output tensor.
                - If output_attentions is True, a tuple containing the layer output tensor and attention tensors.

        Raises:
            TypeError: If the input types are incorrect.
            ValueError: If the head mask has an invalid shape.
            RuntimeError: If an error occurs during the computation.
        """
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in BEiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


class BeitRelativePositionBias(nn.Cell):

    """
    BeitRelativePositionBias

    Represents the relative position bias table used in the Beit model for attention mechanism.

    This class inherits from nn.Cell and provides functionality to construct the relative position bias table 
    by calculating relative positions and indexing.

    The relative position bias table is constructed based on the provided configuration and window size. 
    It contains the relative positional biases used in the attention mechanism of the Beit model.

    The relative position bias table is constructed using the provided configuration and window size to calculate 
    the relative position indices and biases. 
    The construct method returns the constructed relative position bias table as a mindspore.Tensor.

    """
    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        """
        Initializes an instance of the 'BeitRelativePositionBias' class.

        Args:
            self: The object instance.
            config (BeitConfig): The configuration object for the 'BeitRelativePositionBias' class.
            window_size (tuple): A tuple representing the size of the window.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = Parameter(
            ops.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = ops.arange(window_size[0])
        coords_w = ops.arange(window_size[1])
        coords = ops.stack(meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = ops.flatten(coords, start_dim=1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = ops.zeros(
            (window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.relative_position_index = relative_position_index

    def construct(self) -> mindspore.Tensor:
        """
        This method constructs relative position bias based on the given relative position index and window size.

        Args:
            self: An instance of the BeitRelativePositionBias class.

        Returns:
            A mindspore.Tensor representing the relative position bias matrix:
                The shape of the tensor is determined by the window size and the relative position index.

        Raises:
            None
        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww


class BeitEncoder(nn.Cell):

    """
    The BeitEncoder class represents an encoder for the Beit (Vision Transformer) model. 
    It is used to process input data and generate encoded representations. 
    The encoder contains layers and handles gradient checkpointing, if enabled.

    The constructor initializes the BeitEncoder with a configuration and an optional window size. 
    It sets up the relative position bias based on the configuration and initializes the layers.

    The construct method processes the input hidden states through the layers of the encoder. 
    It handles optional head masks, output configurations, and returns the encoded representations.

    Attributes:
        config: BeitConfig
            The configuration object for the BeitEncoder.
        relative_position_bias: Union[BeitRelativePositionBias, None]
            The relative position bias object used in the encoder if enabled.
        layer: nn.CellList
            List of BeitLayer instances representing the layers in the encoder.
        gradient_checkpointing: bool
            Indicates whether gradient checkpointing is enabled.

    Methods:
        __init__:
            Constructs a new BeitEncoder instance with the given configuration and window size.
        construct:
            Processes the input hidden states through the encoder layers and returns the encoded representations.

    """
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        """
        Initializes the BeitEncoder class.

        Args:
            self: The object itself.
            config (BeitConfig): An instance of BeitConfig containing the configuration settings.
            window_size (Optional[tuple]): A tuple representing the window size, defaults to None.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

        # stochastic depth decay rule
        dpr = [x.item() for x in ops.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layer = nn.CellList(
            [
                BeitLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        """
        Construct method in the BeitEncoder class.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states for the encoder.
            head_mask (Optional[mindspore.Tensor], optional): An optional head mask tensor. Defaults to None.
            output_attentions (bool, optional): Indicates whether to output attentions. Defaults to False.
            output_hidden_states (bool, optional): Indicates whether to output hidden states. Defaults to False.
            return_dict (bool, optional): Indicates whether to return a dictionary. Defaults to True.

        Returns:
            Union[tuple, BaseModelOutput]: The constructed output, which can be a tuple or BaseModelOutput object.

        Raises:
            None.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            relative_position_bias = (
                self.relative_position_bias() if self.relative_position_bias is not None else None
            )
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BeitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BeitConfig
    base_model_prefix = "beit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Conv2d, nn.Conv2dTranspose)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class BeitModel(BeitPreTrainedModel):

    """
    BeitModel
    =========

    Represents a BeiT (Vision Transformer) model that utilizes a combination of convolutional and transformer layers 
    for image recognition tasks.

    This class inherits from BeitPreTrainedModel and includes methods for initializing the model, 
    getting input embeddings, pruning heads, and constructing the model with optional arguments.

    Attributes:
        config (BeitConfig): The configuration for the model.
        embeddings (BeitEmbeddings): The embeddings for the model.
        encoder (BeitEncoder): The encoder component of the model.
        layernorm (nn.Identity or nn.LayerNorm): The layer normalization component of the model.
        pooler (BeitPooler): The pooling layer for the model, if included.

    Methods:
        __init__: Initializes the model with the given configuration and optional pooling layer.
        get_input_embeddings(self): Retrieves the input embeddings for the model.
        _prune_heads(self, heads_to_prune): 
            Prunes heads of the model based on the provided dictionary of layers and heads to prune.
        construct: 
            Constructs the model with optional arguments for pixel values, masked positions, head masks, and return types.

        Additional details and descriptions for each method can be found in the method docstrings.
    """
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        """
        Initializes a new instance of the BeitModel class.

        Args:
            self: The object itself.
            config (BeitConfig): The configuration object for the BeitModel.
                This object contains various hyperparameters and settings for the model.
            add_pooling_layer (bool, optional): Flag indicating whether to include a pooling layer.
                Defaults to True.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method sets up the BeitModel by initializing its attributes and components.
            It creates an instance of BeitEmbeddings, BeitEncoder, and BeitPooler based on the provided config.
            If add_pooling_layer is True, it also initializes a pooler for the model.
            Finally, it calls the post_init method to perform any additional initialization steps.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        )
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method 'get_input_embeddings' is part of the 'BeitModel' class and is used to retrieve the input embeddings.

        Args:
            self: An instance of the 'BeitModel' class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.patch_embeddings

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
        bool_masked_pos: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BeitModelOutputWithPooling]:
        r"""
        Args:
            bool_masked_pos (`mindspore.Tensor` of shape `(batch_size, num_patches)`, *optional*):
                Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values, bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class BeitPooler(nn.Cell):

    """
    This class represents a pooler module for the Beit model. It is responsible for performing pooling operations on the hidden states of the model.

    The BeitPooler class inherits from the nn.Cell class.

    Attributes:
        layernorm (nn.LayerNorm, optional): A LayerNorm module that applies layer normalization to the hidden states. 
            If the configuration allows mean pooling, this attribute is initialized with a LayerNorm module. 
            Otherwise, it is set to None.

    Methods:
        __init__:
            Initializes the BeitPooler module.

            Args:
                
            - config (BeitConfig): The configuration object for the Beit model.

            Returns:
            
            - None

        construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
            Constructs and applies the pooling operation on the hidden states.

            Args:
            
            - hidden_states (mindspore.Tensor): The input hidden states tensor.

            Returns:
            
            - mindspore.Tensor: The pooled output tensor.
    """
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes an instance of the BeitPooler class.

        Args:
            self: The object instance.
            config (BeitConfig): The configuration object that contains the settings for the pooler.
                It is expected to have the following attributes:
                
                - hidden_size (int): The size of the hidden state.
                - use_mean_pooling (bool): Whether to use mean pooling or not.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct the pooled output tensor from the given hidden states.

        Args:
            self (BeitPooler): An instance of the BeitPooler class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size),
                containing the hidden states.

        Returns:
            mindspore.Tensor: The pooled output tensor of shape (batch_size, hidden_size), representing the pooled
                representation of the input hidden states.

        Raises:
            None

        Note:
            - If the 'layernorm' attribute is not None, the 'hidden_states' tensor is sliced to exclude the first token
              (CLS token), resulting in a tensor of shape (batch_size, sequence_length - 1, hidden_size).
              The mean of the sliced tensor along the sequence_length dimension is computed using the 'mean' method of
              mindspore.Tensor, resulting in a tensor of shape (batch_size, hidden_size).
              The 'layernorm' method is then applied to the mean tensor to normalize the values across the hidden_size dimension.
              The resulting tensor is assigned to the 'pooled_output' variable.
            - If the 'layernorm' attribute is None, the 'hidden_states' tensor is sliced to extract only the first token
              (CLS token) of shape (batch_size, hidden_size), which is assigned to the 'pooled_output' variable.
            - The 'pooled_output' tensor is returned as the final result.

        Example:
            ```python
            >>> pooler = BeitPooler()
            >>> hidden_states = mindspore.Tensor(np.random.randn(2, 5, 768))
            >>> pooled_output = pooler.construct(hidden_states)
            ```
        """
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output


class BeitForMaskedImageModeling(BeitPreTrainedModel):

    """
    BeitForMaskedImageModeling is a class that represents a model for masked image modeling using the BEiT 
    (Vision Transformer) architecture. 
    It is designed for processing images with masked positions and generating predictions for masked language modeling tasks.

    This class inherits from BeitPreTrainedModel and includes methods for initialization and constructing the model. 
    The __init__ method initializes the model with the provided configuration, setting up various components such as 
    BEiT model, layer normalization, and the LM head.

    The construct method takes input pixel values, boolean masked positions, head mask, labels, 
    and other optional parameters to perform masked language modeling on the input image. 
    It processes the input through the BEiT model, applies layer normalization, computes prediction scores, 
    and calculates masked language modeling loss if labels are provided.

    The method returns a MaskedLMOutput object containing the masked language modeling loss, prediction scores, 
    hidden states, and attentions. 
    Additionally, the docstring includes examples demonstrating how to  use the BeitForMaskedImageModeling class 
    for masked image modeling tasks.
    """
    _keys_to_ignore_on_load_unexpectedecode_headd = [r"relative_position_index", r"num_batches_tracked"]
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes a new instance of the BeitForMaskedImageModeling class.

        Args:
            self: The instance of the class.
            config (BeitConfig): 
                The configuration object for the Beit model. It contains various hyperparameters and settings.
                
                - num_labels (int): The number of labels for classification.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # Classifier head
        self.layernorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        bool_masked_pos: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedLMOutput]:
        r"""
        Args:
            bool_masked_pos (`mindspore.Tensor` of shape `(batch_size, num_patches)`):
                Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[tuple, MaskedLMOutput]

        Example:
            ```python
            >>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
            >>> import torch
            >>> from PIL import Image
            >>> import requests
            ... 
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ... 
            >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            ... 
            >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
            >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
            >>> # create random boolean mask of shape (batch_size, num_patches)
            >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()
            ... 
            >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
            >>> loss, logits = outputs.loss, outputs.logits
            >>> list(logits.shape)
            [1, 196, 8192]
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores[bool_masked_pos], labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitForImageClassification(BeitPreTrainedModel):

    """
    This class is an implementation of the Beit model for image classification tasks. 
    It is designed to be used for both single-label and multi-label classification as well as regression tasks.

    The class inherits from the BeitPreTrainedModel, which provides the basic architecture and functionality of the Beit model.

    Attributes:
        num_labels (int): The number of labels in the classification task.
        beit (BeitModel): The Beit model for image classification.
        classifier (nn.Dense or nn.Identity): The classifier layer for predicting the labels.

    Methods:
        __init__:
            Initializes a new instance of the BeitForImageClassification class.

        construct:
            Constructs the forward pass of the model for image classification.
            This method takes input pixel values, head mask, labels, and other optional arguments, 
            and returns the output of the model.
    """
    _keys_to_ignore_on_load_unexpected = [r"relative_position_index", r"num_batches_tracked"]
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes a new instance of the `BeitForImageClassification` class.

        Args:
            self: The object instance.
            config (BeitConfig): The configuration object that holds various hyperparameters and settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)

        # Classifier head
        self.classifier = nn.Dense(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

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
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitConvModule(nn.Cell):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        """
        Initializes an instance of the 'BeitConvModule' class.

        Args:
            self: The object itself.
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (Union[int, Tuple[int, int]]): The size of the kernel for convolution.
                Can be either an integer or a tuple of integers for height and width.
            padding (Union[int, Tuple[int, int], str], optional): The amount of padding to be added to the input.
                Can be an integer, a tuple of integers for height and width, or a string.
                Defaults to 0, indicating no padding.
            bias (bool, optional): Whether to include a bias term in the convolution operation.
                Defaults to False.
            dilation (Union[int, Tuple[int, int]], optional): The dilation rate for the convolution operation.
                Can be either an integer or a tuple of integers for height and width.
                Defaults to 1, indicating no dilation.

        Returns:
            None.

        Raises:
            None.

        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding if isinstance(padding, (int, tuple)) else 0,
            pad_mode='pad' if padding != 0 else 'valid',
            has_bias=bias,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def construct(self, input: mindspore.Tensor) -> mindspore.Tensor:
        '''
        The 'construct' method constructs the BeitConvModule.

        Args:
            self: BeitConvModule object. The instance of the BeitConvModule class.
            input: mindspore.Tensor. The input tensor to be processed.

        Returns:
            mindspore.Tensor:
                The output tensor after passing through the convolutional layer, batch normalization, 
                and activation function.

        Raises:
            None.
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


class BeitPyramidPoolingBlock(nn.Cell):

    """
    BeitPyramidPoolingBlock is a class that represents a block for performing pyramid pooling operations in a neural network.
    It inherits from nn.Cell and contains methods for initialization and constructing the block.

    Attributes:
        pool_scale (int): The scale used for adaptive average pooling.
        in_channels (int): The number of input channels.
        channels (int): The number of output channels.

    Methods:
        __init__:
            Initializes the BeitPyramidPoolingBlock with the specified pool_scale, in_channels, and channels.

        construct:
            Constructs the pyramid pooling block by applying the layers to the input tensor.

    Usage:
        block = BeitPyramidPoolingBlock(pool_scale=2, in_channels=64, channels=128)
        output = block.construct(input_tensor)
    """
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        """
        Initializes a new instance of the BeitPyramidPoolingBlock class.

        Args:
            self: The object itself.
            pool_scale (int): The scale for adaptive average pooling.
            in_channels (int): The number of input channels.
            channels (int): The number of output channels.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            BeitConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            setattr(self, str(i), layer)
            # self(str(i), layer)

    def construct(self, input: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the BeitPyramidPoolingBlock by applying the specified layers to the input tensor.

        Args:
            self (BeitPyramidPoolingBlock): The instance of the BeitPyramidPoolingBlock class.
            input (mindspore.Tensor): The input tensor to be processed by the BeitPyramidPoolingBlock.

        Returns:
            mindspore.Tensor: A tensor representing the hidden state after applying the specified layers.

        Raises:
            None
        """
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class BeitPyramidPoolingModule(nn.Cell):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        """
        Initializes an instance of the BeitPyramidPoolingModule class.

        Args:
            pool_scales (Tuple[int, ...]): A tuple of integers representing the scales for pyramid pooling.
            in_channels (int): The number of input channels.
            channels (int): The number of output channels.
            align_corners (bool): A boolean indicating whether to align corners during pooling.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        for i, pool_scale in enumerate(pool_scales):
            block = BeitPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            setattr(self, str(i), block)
            # self._add_attr(str(i), block)

    def construct(self, x: mindspore.Tensor) -> List[mindspore.Tensor]:
        """
        Constructs the Beit Pyramid Pooling Module.

        Args:
            self (BeitPyramidPoolingModule): The instance of the BeitPyramidPoolingModule class.
            x (mindspore.Tensor): The input tensor for the pyramid pooling module.

        Returns:
            List[mindspore.Tensor]: A list of upsampled feature maps from the pyramid pooling module.

        Raises:
            ValueError: If the input tensor x is not of type mindspore.Tensor.
            RuntimeError: If an error occurs during the pyramid pooling module construction.
        """
        ppm_outs = []
        for ppm in self.blocks:
            ppm_out = ppm(x)
            upsampled_ppm_out = ops.interpolate(
                ppm_out, size=x.shape[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class BeitUperHead(nn.Cell):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(self, config: BeitConfig) -> None:
        """
        The __init__ method initializes an instance of the BeitUperHead class.

        Args:
            self: This parameter refers to the instance of the class itself.
            config (BeitConfig): An object of type BeitConfig containing the configuration settings for the UperHead. 
                It is used to initialize various attributes of the UperHead.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1, has_bias=True)

        # PSP Module
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = BeitConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.CellList()
        self.fpn_convs = nn.CellList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = BeitConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = BeitConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = BeitConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, inputs):
        """
        This method 'psp_forward' is defined in the 'BeitUperHead' class and is used to perform the forward pass of
        the Pyramid Scene Parsing network.

        Args:
            self (object): Reference to the current instance of the class BeitUperHead.
            inputs (list): List of input tensors where the last element is used for processing.

        Returns:
            None: This method does not return any value directly.
                It processes the input data and updates the internal state of the object.

        Raises:
            None: This method does not explicitly raise any exceptions.
                However, exceptions may be raised by the called functions within this method.
        """
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = ops.cat(psp_outs, axis=1)
        output = self.bottleneck(psp_outs)

        return output

    def construct(self, encoder_hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the Feature Pyramid Network (FPN) for the UperHead module in the Beit architecture.

        Args:
            self (BeitUperHead): An instance of the BeitUperHead class.
            encoder_hidden_states (mindspore.Tensor): The hidden states generated by the encoder.
                These hidden states are used as input to the FPN for feature extraction.

        Returns:
            mindspore.Tensor: The output tensor representing the final processed features obtained from the FPN.
                This tensor is the result of passing the encoder hidden states through the FPN layers.

        Raises:
            TypeError: If the input encoder_hidden_states is not of type mindspore.Tensor.
            ValueError: If the input encoder_hidden_states is empty or has incorrect dimensions.
            RuntimeError: If there is an issue during the FPN computation process.
        """
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + ops.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = ops.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = ops.cat(fpn_outs, axis=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class BeitFCNHead(nn.Cell):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (BeitConfig):
            Configuration.

            - in_channels
            - kernel_size (int): The kernel size for convs in the head. Default: 3.
            - dilation (int): The dilation rate for convs in the head. Default: 1.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    def __init__(
        self, config: BeitConfig, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        """
        Initializes a BeitFCNHead object.

        Args:
            self: The object instance.
            config (BeitConfig): An instance of BeitConfig containing configuration parameters.
            in_index (int): Index of the input tensor.
            kernel_size (int): Size of the convolutional kernel.
            dilation (Union[int, Tuple[int, int]]): Dilation factor for the convolution operation.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            BeitConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                BeitConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.SequentialCell(*convs)
        if self.concat_input:
            self.conv_cat = BeitConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1, has_bias=True)

    def construct(self, encoder_hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        '''
        This method constructs the FCN (Fully Convolutional Network) head for the Beit model.

        Args:
            self (BeitFCNHead): The instance of the BeitFCNHead class.
            encoder_hidden_states (mindspore.Tensor): The hidden states of the encoder. It is expected to be a Tensor.

        Returns:
            mindspore.Tensor: The output tensor of the FCN head.

        Raises:
            ValueError: If the input encoder_hidden_states is not a valid mindspore.Tensor.
            RuntimeError: If the method encounters any runtime errors during the construction of the FCN head.
        '''
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(ops.cat([hidden_states, output], axis=1))
        output = self.classifier(output)
        return output


class BeitForSemanticSegmentation(BeitPreTrainedModel):

    """
    A Python class representing the Beit model for semantic segmentation.

    This class inherits from the BeitPreTrainedModel and includes methods for initializing the model, computing loss,
    and constructing the model. T
    he `BeitForSemanticSegmentation` class is designed for semantic segmentation tasks, where it takes input pixel values
    and produces semantic segmentation maps.
    It utilizes the BeitModel for feature extraction, applies a series of operations to the extracted features,
    and outputs logits for semantic segmentation.

    The class's `__init__` method initializes the model with the specified configuration and sets up the necessary
    components for semantic segmentation, such as the BeitModel, decoder head, and auxiliary head.
    The `compute_loss` method calculates the loss based on the model's predictions and ground truth labels.
    The `construct` method processes input pixel values, applies the model, and returns the semantic segmentation logits
    along with optional outputs like hidden states and attentions.
    Additionally, it provides detailed information on the expected input format for labels and examples of how to use the class
    for semantic segmentation tasks.

    """
    _keys_to_ignore_on_load_unexpected = [r"relative_position_index", r"num_batches_tracked"]
    def __init__(self, config: BeitConfig) -> None:
        """
        Initializes a new instance of BeitForSemanticSegmentation.

        Args:
            self: The instance of the class.
            config (BeitConfig): The configuration object containing parameters for the model.
                It is used to set up the model architecture and define various hyperparameters.
                Must be an instance of BeitConfig.

        Returns:
            None.

        Raises:
            ValueError: Raised if the length of config.out_indices is not 4, indicating an incorrect specification.
                BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers,
                specifying which features to use from the backbone.
                Recommended values are [3, 5, 7, 11] for a base-sized architecture.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        if len(self.config.out_indices) != 4:
            raise ValueError(
                "BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        self.fpn1 = nn.SequentialCell(
            nn.Conv2dTranspose(config.hidden_size, config.hidden_size, kernel_size=2, stride=2, has_bias=True),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.Conv2dTranspose(config.hidden_size, config.hidden_size, kernel_size=2, stride=2, has_bias=True),
        )
        self.fpn2 = nn.SequentialCell(
            nn.Conv2dTranspose(config.hidden_size, config.hidden_size, kernel_size=2, stride=2, has_bias=True),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        self.decode_head = BeitUperHead(config)
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        '''
        This method computes the loss for semantic segmentation using the logits and auxiliary logits, and compares them with the provided labels.

        Args:
            self: (object) The instance of the class BeitForSemanticSegmentation.
            logits: (tensor) The main logits for semantic segmentation.
            auxiliary_logits: (tensor or None) The auxiliary logits for semantic segmentation. It can be None if not provided.
            labels: (tensor) The ground truth labels for semantic segmentation.

        Returns:
            None: The method computes the loss and updates the internal state of the class.

        Raises:
            ValueError: If the size of logits or auxiliary_logits does not match the size of the labels.
            ValueError: If labels contain values outside the range [0, num_classes-1], where num_classes is the number of classes for semantic segmentation.
            RuntimeError: If the mode specified for interpolation is not supported.
        '''
        # upsample logits to the images' original size
        upsampled_logits = ops.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = ops.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # compute weighted loss
        main_loss = ops.cross_entropy(upsampled_logits, labels, ignore_index=self.config.semantic_loss_ignore_index)
        loss = main_loss
        if auxiliary_logits is not None:
            auxiliary_loss = ops.cross_entropy(upsampled_auxiliary_logits, labels, ignore_index=self.config.semantic_loss_ignore_index)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
                Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:
            Union[tuple, SemanticSegmenterOutput]

        Example:
            ```python
            >>> from transformers import AutoImageProcessor, BeitForSemanticSegmentation
            >>> from PIL import Image
            >>> import requests
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
            >>> model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
            ...
            >>> inputs = image_processor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> # logits are of shape (batch_size, num_labels, height, width)
            >>> logits = outputs.logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features
        ]

        # apply FPNs
        operators = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = operators[i](features[i])

        logits = self.decode_head(features)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class BeitBackbone(BeitPreTrainedModel, BackboneMixin):

    """
    Represents the backbone of a BEiT (Bottleneck Enhanced Image Transformer) model for image recognition and classification tasks.
    This class inherits from BeitPreTrainedModel and BackboneMixin.

    The backbone consists of an image embedding module, an encoder module, and optionally,
    a feature pyramid network (FPN) for multi-scale feature extraction.

    The class provides methods for initializing the backbone, getting input embeddings, and constructing
    the backbone from input pixel values. It also supports the option to return hidden states and attentions.

    Example Usage:
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ...
        >>> processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/beit-base-patch16-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )
        ...
        >>> inputs = processor(image, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```

    Attributes:
        num_features (list): List of hidden sizes for each layer in the backbone.

    Methods:
        __init__: Initializes the backbone with the given configuration.
        get_input_embeddings: Returns the patch embeddings used as input to the backbone.
        construct: Constructs the backbone from input pixel values, optionally returning hidden states and attentions.

    Raises:
        ValueError: If the specified output indices are invalid.

    Returns:
        BackboneOutput: An object containing feature maps, hidden states, and attentions.

    Note:
        The class supports the use of the FPN for multi-scale feature extraction.
    """
    def __init__(self, config):
        """
        Initializes an instance of the 'BeitBackbone' class.

        Args:
            self: The instance of the 'BeitBackbone' class.
            config: An object containing the configuration settings for the 'BeitBackbone'.
                It should provide the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - num_hidden_layers (int): The number of hidden layers.
                - add_fpn (bool): Indicates whether to add a Feature Pyramid Network (FPN) to the backbone.
                - out_indices (list): A list of 4 integers specifying which features to use from the backbone if FPN is added.
                For example, [3, 5, 7, 11] can be used for a base-sized architecture.
                - batch_norm_eps (float): The value for epsilon in Batch Normalization.

                Note: Make sure 'config' provides the necessary attributes; otherwise, an exception will be raised.

        Returns:
            None.

        Raises:
            ValueError: If 'config.add_fpn' is True but 'len(config.out_indices)' is not equal to 4.
                        In this case, 'config.out_indices' should be a list of 4 integers specifying the features to use from the backbone.
        """
        super().__init__(config)
        super()._init_backbone(config)

        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        if config.add_fpn:
            if len(self.config.out_indices) != 4:
                raise ValueError(
                    "BeitBackbone requires config.out_indices to be a list of 4 integers, "
                    "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                    "a base-sized architecture."
                )
            hidden_size = config.hidden_size
            self.fpn1 = nn.SequentialCell(
                nn.Conv2dTranspose(hidden_size, hidden_size, kernel_size=2, stride=2, has_bias=True),
                nn.BatchNorm2d(hidden_size, eps=config.batch_norm_eps),
                nn.GELU(),
                nn.Conv2dTranspose(hidden_size, hidden_size, kernel_size=2, stride=2, has_bias=True),
            )

            self.fpn2 = nn.SequentialCell(nn.Conv2dTranspose(hidden_size, hidden_size, kernel_size=2, stride=2, has_bias=True))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the BeitBackbone class.

        Args:
            self (BeitBackbone): An instance of the BeitBackbone class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the BeitBackbone class.
        The input embeddings are obtained through the patch_embeddings attribute of the class
        and are used for further processing or analysis.

        Note:
            The input embeddings refer to the numerical representation of the input data,
            which can be used for tasks such as classification, regression, or other machine learning tasks.
        """
        return self.embeddings.patch_embeddings

    def construct(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
            >>> processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
            >>> model = AutoBackbone.from_pretrained(
            ...     "microsoft/beit-base-patch16-224", out_features=["stage1", "stage2", "stage3", "stage4"]
            ... )
            ...
            >>> inputs = processor(image, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> feature_maps = outputs.feature_maps
            >>> list(feature_maps[-1].shape)
            [1, 768, 14, 14]
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        batch_size = pixel_values.shape[0]
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values)

        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:, :]
                    hidden_state = hidden_state.permute(0, 2, 1)
                    hidden_state = hidden_state.reshape(batch_size, -1, patch_height, patch_width)

                feature_maps += (hidden_state,)

        if self.config.add_fpn:
            feature_maps = [
                self.fpn1(feature_maps[0]),
                self.fpn2(feature_maps[1]),
                self.fpn3(feature_maps[2]),
                self.fpn4(feature_maps[3]),
            ]
            feature_maps = tuple(feature_maps)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

__all__ = [
    "BEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "BeitForImageClassification",
    "BeitForMaskedImageModeling",
    "BeitForSemanticSegmentation",
    "BeitModel",
    "BeitPreTrainedModel",
    "BeitBackbone",
]
