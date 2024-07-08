# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
""" MindSpore CLIP model."""


from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig


logger = logging.get_logger(__name__)


CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/clip-vit-base-patch32",
    # See all CLIP models at https://hf-mirror.com/models?filter=clip
]


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/2021-03-07-clip.html
def contrastive_loss(logits: mindspore.Tensor) -> mindspore.Tensor:
    """
    Calculates the contrastive loss for the given logits.
    
    Args:
        logits (mindspore.Tensor): The logits tensor for the contrastive loss calculation.
    
    Returns:
        mindspore.Tensor: The calculated contrastive loss value as a tensor.
    
    Raises:
        None.
    
    """
    return ops.cross_entropy(logits, ops.arange(len(logits)))


def clip_loss(similarity: mindspore.Tensor) -> mindspore.Tensor:
    ''' 
    Calculate the average of caption loss and image loss obtained from contrastive loss calculation.
    
    Args:
        similarity (mindspore.Tensor): A tensor containing similarity values between caption and image features.
    
    Returns:
        mindspore.Tensor: A tensor representing the average of caption loss and image loss.
    
    Raises:
        None
    '''
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class CLIPVisionModelOutput(ModelOutput):
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
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    image_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class CLIPTextModelOutput(ModelOutput):
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
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class CLIPOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`CLIPVisionModel`].
    """
    loss: Optional[mindspore.Tensor] = None
    logits_per_image: mindspore.Tensor = None
    logits_per_text: mindspore.Tensor = None
    text_embeds: mindspore.Tensor = None
    image_embeds: mindspore.Tensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        """
        Converts the CLIPOutput object to a tuple representation.
        
        Args:
            self (CLIPOutput): The current instance of the CLIPOutput class.
        
        Returns:
            Tuple[Any]: A tuple representation of the CLIPOutput object, where each element corresponds to a key-value pair in the object.
                The values are preserved as is, except for the keys 'text_model_output' and 'vision_model_output',
                which are recursively transformed to their respective tuple representations.
        
        Raises:
            None.
        
        Note:
            The 'text_model_output' and 'vision_model_output' keys are skipped during the transformation to avoid
            potential circular references or infinite recursion.
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()         )


class CLIPVisionEmbeddings(nn.Cell):

    """
    CLIPVisionEmbeddings is a class that represents the embeddings used in the CLIP (Contrastive Language-Image Pretraining) model for vision. This class inherits from nn.Cell and is responsible for
    constructing the embeddings for input images.

    Attributes:
        config (CLIPVisionConfig): The configuration object that holds the parameters for the CLIPVisionEmbeddings.
        embed_dim (int): The dimension of the embeddings.
        image_size (int): The size of the input image.
        patch_size (int): The size of the patches used for creating embeddings.
        class_embedding (Parameter): The learnable parameter for the class embedding.
        patch_embedding (nn.Conv2d): The convolutional layer for creating patch embeddings.
        num_patches (int): The number of patches in the image.
        num_positions (int): The total number of positions, including the class embedding position.
        position_embedding (nn.Embedding): The embedding layer for positional embeddings.
        position_ids (Tensor): The tensor containing position IDs.

    Methods:
        construct(pixel_values: mindspore.Tensor) -> mindspore.Tensor:
            Constructs the embeddings for the input pixel values.

    Raises:
        NotImplementedError: If the construct method is not implemented in the subclass.
    """
    def __init__(self, config: CLIPVisionConfig):
        """
        Initialize the CLIPVisionEmbeddings class.

        Args:
            self (CLIPVisionEmbeddings): The instance of the CLIPVisionEmbeddings class.
            config (CLIPVisionConfig):
                An instance of CLIPVisionConfig containing configuration parameters.

                - config.hidden_size (int): The size of the hidden embedding dimension.
                - config.image_size (int): The size of the input image.
                - config.patch_size (int): The size of each patch in the image.
                - config.num_channels (int): The number of channels in the input image.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Parameter(ops.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            has_bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = ops.arange(self.num_positions).expand((1, -1))

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct embeddings for CLIP vision model.

        Args:
            self (CLIPVisionEmbeddings): The instance of the CLIPVisionEmbeddings class.
            pixel_values (mindspore.Tensor): A tensor containing pixel values of images.
                It should have a shape of (batch_size, channels, height, width).

        Returns:
            mindspore.Tensor: A tensor containing the constructed embeddings for the input images.
                The shape of the returned tensor is (batch_size, num_patches + 1, embedding_dim),
                where num_patches is the number of patches extracted from the image and
                embedding_dim is the dimension of the embedding space.

        Raises:
            ValueError: If the input pixel_values tensor is not in the expected shape.
            TypeError: If the dtype of the pixel_values tensor is not compatible with the patch_embedding weights.
            RuntimeError: If there is an issue during the computation of the embeddings.
        """
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPTextEmbeddings(nn.Cell):

    """
    This class represents the CLIPTextEmbeddings, which is a module for creating text embeddings in the
    CLIP (Contrastive Language-Image Pretraining) model. It inherits from the nn.Cell class.

    Attributes:
        token_embedding (nn.Embedding): Embedding layer for token inputs.
        position_embedding (nn.Embedding): Embedding layer for position inputs.
        position_ids (mindspore.Tensor): Tensor representing the position IDs.

    Methods:
        __init__: Initializes the CLIPTextEmbeddings module.
        construct: Constructs the text embeddings.

    """
    def __init__(self, config: CLIPTextConfig):
        """
        Initializes the CLIPTextEmbeddings class.

        Args:
            self: The instance of the class.
            config (CLIPTextConfig): An instance of CLIPTextConfig class representing the configuration parameters for the text embeddings.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not an instance of the CLIPTextConfig class.
            ValueError: If the 'embed_dim' calculated from the 'config' parameter is not valid.
            RuntimeError: If there is an issue with initializing the token_embedding or position_embedding layers.
        """
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
    ) -> mindspore.Tensor:
        """
        Constructs the text embeddings for the CLIP model.

        Args:
            self (CLIPTextEmbeddings): The instance of the CLIPTextEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input token IDs for the text. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position IDs for each token in the text. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The precomputed embeddings for the input tokens. Default is None.

        Returns:
            mindspore.Tensor: The constructed text embeddings combining input token embeddings and position embeddings.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None.
            ValueError: If seq_length is not valid based on input_ids and inputs_embeds shapes.
        """
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config):
        """
        __init__(self, config)

        Initializes a new instance of the CLIPAttention class.

        Args:
            self: The instance of the class.
            config: An object representing the configuration for the attention mechanism.
                It should contain the following attributes:

                - hidden_size: An integer representing the dimension of the hidden state.
                - num_attention_heads: An integer representing the number of attention heads.
                - attention_dropout: A floating-point number representing the dropout probability for attention weights.

        Returns:
            None.

        Raises:
            ValueError: If the embed_dim is not divisible by num_heads,
                a ValueError is raised with a message indicating the mismatched values.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor to match the required shape for attention calculation in the CLIPAttention class.

        Args:
            self (CLIPAttention): An instance of the CLIPAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None.

        Raises:
            None.

        This method reshapes the input tensor to have dimensions [bsz, seq_len, num_heads, head_dim].
        The original shape of the tensor is expected to be compatible with reshaping to the required shape.
        The reshaping operation involves rearranging the dimensions of the tensor and
        swapping the dimensions corresponding to seq_len and num_heads.

        Note:
            - The input tensor must have a compatible shape for reshaping.
            - The num_heads and head_dim attributes should be defined in the CLIPAttention class before calling this method.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        causal_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Cell):

    """
    The CLIPMLP class represents a multi-layer perceptron (MLP) neural network for the CLIP (Contrastive Language-Image Pre-training) model.
    This class inherits from nn.Cell and contains methods for initializing the network and performing forward propagation through the network.

    Attributes:
        config (dict): A dictionary containing configuration parameters for the network.
        activation_fn (function): Activation function used in the hidden layers of the network.
        fc1 (nn.Dense): Fully connected layer mapping input to intermediate size.
        fc2 (nn.Dense): Fully connected layer mapping intermediate size to output size.

    Methods:
        __init__: Initializes the CLIPMLP object with the provided configuration.
        construct: Performs forward propagation through the network
            given an input tensor of hidden states, returning the output tensor after passing through the MLP layers.
    """
    def __init__(self, config):
        """
        Initializes an instance of the CLIPMLP class.

        Args:
            self: The instance of the CLIPMLP class.
            config: A configuration object containing parameters for the CLIPMLP model.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not of the expected type.
            ValueError: If the config does not contain required parameters.
            RuntimeError: If there is an issue with initializing the neural network layers.
        """
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the forward pass of the CLIPMLP model.

        Args:
            self: An instance of the CLIPMLP class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape (batch_size, hidden_size).

        Returns:
            mindspore.Tensor: The output tensor after passing through the CLIPMLP model.
                The shape of the output tensor is (batch_size, hidden_size).

        Raises:
            None.

        This method applies linear transformations and activation functions to the input hidden states tensor to
        compute the forward pass of the CLIPMLP model. The forward pass consists of the following steps:

        1. Applies a linear transformation to the input hidden states tensor using the fully connected layer fc1.
        2. Applies an activation function to the output of fc1 using the activation function specified in the CLIPMLP instance.
        3. Applies another linear transformation to the output of the activation function using the fully connected layer fc2.
        4. Returns the final output tensor after passing through the fc2 layer.

        Note that the input hidden_states tensor must have a shape of (batch_size, hidden_size),
        where batch_size represents the number of input examples and hidden_size represents the size of the hidden
        states.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Cell):

    """
    This class represents a single layer of the CLIPEncoder, which is responsible for encoding input hidden states
    using self-attention and multi-layer perceptron (MLP) operations.

    Attributes:
        embed_dim (int): The dimensionality of the input embeddings.
        self_attn (CLIPAttention): The self-attention mechanism used for capturing relationships between different elements in the input.
        layer_norm1 (nn.LayerNorm): The layer normalization operation applied after the self-attention operation.
        mlp (CLIPMLP): The multi-layer perceptron used for non-linear transformations of the hidden states.
        layer_norm2 (nn.LayerNorm): The layer normalization operation applied after the MLP operation.

    Methods:
        construct(hidden_states, attention_mask, causal_attention_mask, output_attentions=False):
            Applies the CLIPEncoderLayer operations on the given input hidden states.

            Args:

                - hidden_states (mindspore.Tensor): The input hidden states of shape (batch, seq_len, embed_dim).
                - attention_mask (mindspore.Tensor): The attention mask of size (batch, 1, tgt_len, src_len) that indicates
                padding elements with very large negative values.
                - causal_attention_mask (mindspore.Tensor): The causal attention mask of size (batch, tgt_len, src_len)
                used for preventing information flow from future tokens to past tokens.
                - output_attentions (bool, optional): Whether or not to return the attention tensors of all attention layers. Default is False.

            Returns:

                - Tuple[mindspore.Tensor]: A tuple containing the encoded hidden states. If output_attentions is True,
                the tuple also includes the attention weights.
    """
    def __init__(self, config: CLIPConfig):
        """
        Initializes a new instance of the CLIPEncoderLayer class.

        Args:
            self: The current instance of the CLIPEncoderLayer.
            config (CLIPConfig): The configuration object for CLIP,
                which provides necessary settings for the encoder layer initialization.

                - hidden_size (int): The embedding dimension.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        causal_attention_mask: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = CLIPConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(cell, CLIPTextEmbeddings):
            cell.token_embedding.weight.set_data(initializer(Normal(factor * 0.02),
                                                 cell.token_embedding.weight.shape, cell.token_embedding.weight.dtype))
            cell.position_embedding.weight.set_data(initializer(Normal(factor * 0.02),
                                        cell.position_embedding.weight.shape, cell.position_embedding.weight.dtype))
        elif isinstance(cell, CLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            cell.class_embedding.set_data(initializer(Normal(cell.embed_dim**-0.5 * factor),
                                        cell.class_embedding.shape, cell.class_embedding.dtype))
            cell.patch_embedding.weight.set_data(initializer(Normal(cell.config.initializer_range * factor),
                                                 cell.patch_embedding.weight.shape, cell.patch_embedding.weight.dtype))
            cell.position_embedding.weight.set_data(initializer(Normal(cell.config.initializer_range * factor),
                                                 cell.position_embedding.weight.shape, cell.position_embedding.weight.dtype))

        elif isinstance(cell, CLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (cell.embed_dim**-0.5) * ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (cell.embed_dim**-0.5) * factor

            cell.q_proj.weight.set_data(initializer(Normal(in_proj_std),
                                        cell.q_proj.weight.shape, cell.q_proj.weight.dtype))
            cell.k_proj.weight.set_data(initializer(Normal(in_proj_std),
                                        cell.k_proj.weight.shape, cell.k_proj.weight.dtype))
            cell.v_proj.weight.set_data(initializer(Normal(in_proj_std),
                                        cell.v_proj.weight.shape, cell.v_proj.weight.dtype))
            cell.out_proj.weight.set_data(initializer(Normal(out_proj_std),
                                        cell.out_proj.weight.shape, cell.out_proj.weight.dtype))

        elif isinstance(cell, CLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (cell.config.hidden_size**-0.5) * ((2 * cell.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * cell.config.hidden_size) ** -0.5 * factor

            cell.fc1.weight.set_data(initializer(Normal(fc_std),
                                    cell.fc1.weight.shape, cell.fc1.weight.dtype))
            cell.fc2.weight.set_data(initializer(Normal(in_proj_std),
                                    cell.fc2.weight.shape, cell.fc2.weight.dtype))

        elif isinstance(cell, CLIPModel):
            cell.text_projection.weight.set_data(initializer(Normal(cell.text_embed_dim**-0.5 * self.config.initializer_factor),
                                    cell.text_projection.weight.shape, cell.text_projection.weight.dtype))

            cell.visual_projection.weight.set_data(initializer(Normal(cell.vision_embed_dim**-0.5 * self.config.initializer_factor),
                                    cell.visual_projection.weight.shape, cell.visual_projection.weight.dtype))
        elif isinstance(cell, CLIPVisionModelWithProjection):
            cell.visual_projection.weight.set_data(initializer(Normal(self.config.hidden_size**-0.5 * self.config.initializer_factor),
                                    cell.visual_projection.weight.shape, cell.visual_projection.weight.dtype))

        elif isinstance(cell, CLIPTextModelWithProjection):
            cell.text_projection.weight.set_data(initializer(Normal(self.config.hidden_size**-0.5 * self.config.initializer_factor),
                                    cell.text_projection.weight.shape, cell.text_projection.weight.dtype))
        if isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        if isinstance(cell, nn.Dense) and cell.bias is not None:
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class CLIPEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """
    def __init__(self, config: CLIPConfig):
        """
        Initializes an instance of the CLIPEncoder class.

        Args:
            self: The instance of the class.
            config (CLIPConfig): The configuration object for the encoder.
                It specifies the settings for the encoder's behavior.

                - Type: CLIPConfig
                - Purpose: To provide configuration options for the encoder.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None. The method does not raise any exceptions.
        """
        super().__init__()
        self.config = config
        self.layers = nn.CellList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[mindspore.Tensor] = None,
        causal_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class CLIPTextTransformer(nn.Cell):

    """
    The CLIPTextTransformer class represents a transformer model for processing text inputs in the Contextual
    Language-Image Pretraining (CLIP) framework. It includes methods for initializing the model and constructing
    the forward pass for text inputs.

    This class inherits from the nn.Cell module, and it contains an initialization method (__init__) for setting up
    the model configuration and a construct method for processing input text data through the transformer layers.

    The __init__ method initializes the CLIPTextTransformer instance with a provided CLIPTextConfig object,
    setting up the model's configuration and embedding layers.

    The construct method processes input text data through the transformer layers, including handling input_ids,
    attention_mask, position_ids, and other optional parameters. It applies the transformer encoder to the input
    embeddings and returns the encoded hidden states and pooled output.

    For additional details and usage examples, please refer to the code and method-specific docstrings.
    """
    def __init__(self, config: CLIPTextConfig):
        """
        Initializes an instance of the CLIPTextTransformer class.

        Args:
            self: The instance of the class.
            config (CLIPTextConfig):
                The configuration object containing parameters for the transformer.

                - `hidden_size` (int): The dimensionality of the embeddings and encoder layers.
                - `layer_norm_eps` (float): The epsilon value for layer normalization.
                - `eos_token_id` (int): The ID of the end-of-sentence token.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                ops.arange(last_hidden_state.shape[0]),
                input_ids.to(dtype=mindspore.int32).argmax(axis=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                ops.arange(last_hidden_state.shape[0]),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=mindspore.int32) == self.eos_token_id)
                .int()
                .argmax(axis=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextModel(CLIPPreTrainedModel):

    """
    The `CLIPTextModel` class represents a model for processing text inputs using the CLIP (Contrastive Language-Image Pretraining) framework.
    This class inherits from `CLIPPreTrainedModel` and provides methods for initializing the model, obtaining input embeddings,
    and constructing the model for inference.

    The `CLIPTextModel` class includes methods for initializing the model with a configuration, obtaining input embeddings,
    and constructing the model for inference. The `get_input_embeddings` method returns the token embeddings used as input
    to the model, while the `set_input_embeddings` method allows for updating the token embeddings.
    The `construct` method constructs the model for performing inference, with options for specifying input tensors,
    attention masks, position ids, and return settings.

    The `construct` method returns the model outputs based on the provided inputs and settings.
    Additionally, the docstring includes usage examples for initializing the `CLIPTextModel` and performing inference
    using the model.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel
        ...
        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        ...
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```
    """
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        """Initialize the CLIPTextModel object with the given configuration.

            Args:
                self (CLIPTextModel): The instance of the CLIPTextModel class.
                config (CLIPTextConfig): The configuration object for CLIPTextModel.

            Returns:
                None

            Raises:
                None
            """
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Method to retrieve the input embeddings from the CLIPTextModel.

        Args:
            self (CLIPTextModel): The instance of the CLIPTextModel class.
                This parameter refers to the current instance of the CLIPTextModel class
                from which the input embeddings are being retrieved.

        Returns:
            nn.Cell: An instance of the neural network Cell class representing the input embeddings.
                The return value is the token embedding from the text model, which serves as the input embeddings
                for further processing within the CLIPTextModel.

        Raises:
            None.
        """
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the CLIPTextModel.

        Args:
            self (CLIPTextModel): The instance of the CLIPTextModel.
            value: The new input embeddings to be set. It can be of any type.

        Returns:
            None

        Raises:
            None
        """
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, CLIPTextModel
            ...
            >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> last_hidden_state = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPVisionTransformer(nn.Cell):

    """
    This class represents a vision transformer model for the Contrastive Language-Image Pretraining (CLIP) framework.
    It inherits from the nn.Cell class and incorporates CLIPVisionConfig and CLIPVisionEmbeddings for configuration
    and embedding functionalities, respectively.
    The class includes methods for initialization and construction of the vision transformer.

    The __init__ method initializes the CLIPVisionTransformer class with the provided configuration. It sets up the required embeddings, layer normalization, and encoder components.

    The construct method processes the input pixel values and generates the outputs using the configured vision transformer. It handles optional arguments for controlling the output format and returns the
    resulting hidden states, pooled output, and other relevant information according to the specified return format.

    Note:
        This class is designed to be used within the MindSpore framework for vision-related tasks in the CLIP framework.
    """
    def __init__(self, config: CLIPVisionConfig):
        """
        Initializes an instance of the CLIPVisionTransformer class.

        Args:
            self: The instance of the class.
            config (CLIPVisionConfig):
                An object of the CLIPVisionConfig class containing the configuration parameters for the CLIPVisionTransformer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
            Union[Tuple, BaseModelOutputWithPooling]
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPVisionModel(CLIPPreTrainedModel):

    """
    The `CLIPVisionModel` class represents a model for vision tasks using the CLIP (Contrastive Language-Image Pre-training)
    framework. It is designed to process images and generate visual embeddings using the CLIPVisionTransformer.

    Args:
        config (CLIPVisionConfig): The configuration object that defines the model architecture and behavior.

    Attributes:
        vision_model (CLIPVisionTransformer): The CLIPVisionTransformer instance used for image processing.

    Methods:
        __init__: Initializes a new instance of the `CLIPVisionModel` class.
        get_input_embeddings: Returns the input embeddings of the vision model.
        construct: Constructs the vision model and performs image processing.

    Returns:
        The constructed `CLIPVisionModel` instance.

    Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel
        ...
        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPVisionConfig):
        """
        Initializes a new instance of the CLIPVisionModel class.

        Args:
            self: The instance of the class.
            config (CLIPVisionConfig): An instance of CLIPVisionConfig class representing the configuration settings.
                It is required to initialize the CLIPVisionModel.
                It must be of type CLIPVisionConfig.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type CLIPVisionConfig.
        """
        super().__init__(config)
        self.vision_model = CLIPVisionTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        This method returns the input embeddings from the CLIPVisionModel.

        Args:
            self (CLIPVisionModel): The instance of the CLIPVisionModel class.

        Returns:
            nn.Cell: The input embeddings from the vision model. This is of type nn.Cell.

        Raises:
            None
        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""

        Returns:
            `Union[Tuple, BaseModelOutputWithPooling]`

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, CLIPVisionModel
            ...
            >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class CLIPModel(CLIPPreTrainedModel):

    """
    A Python class representing a CLIP (Contrastive Language-Image Pre-training) model that combines text and vision
    inputs for image-text similarity scoring. This class inherits from CLIPPreTrainedModel and provides methods for
    extracting text and image features, as well as for constructing the final CLIP output.
    The class handles the initialization of model configurations, text and vision embeddings, projectionlayers,
    and scaling of logits for calculating similarity scores.
    It also includes examples on how to use the model for text and image inputs.
    """
    config_class = CLIPConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        """
        Initializes an instance of the CLIPModel class.

        Args:
            self: The instance of the class.
            config (CLIPConfig):
                An instance of the CLIPConfig class which holds the configuration parameters for the CLIPModel.

                - text_config (CLIPTextConfig): An instance of the CLIPTextConfig class which holds the configuration parameters for the text model.

                    - hidden_size (int): The dimension of the hidden state in the text model.

                - vision_config (CLIPVisionConfig): An instance of the CLIPVisionConfig class which holds the configuration parameters for the vision model.

                    - hidden_size (int): The dimension of the hidden state in the vision model.

                - projection_dim (int): The dimension of the projection output.

        Returns:
            None.

        Raises:
            ValueError: If the 'config.text_config' parameter is not of type CLIPTextConfig.
            ValueError: If the 'config.vision_config' parameter is not of type CLIPVisionConfig.
        """
        super().__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Dense(self.vision_embed_dim, self.projection_dim, has_bias=False)
        self.text_projection = nn.Dense(self.text_embed_dim, self.projection_dim, has_bias=False)
        self.logit_scale = mindspore.tensor([self.config.logit_scale_init_value])

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            text_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
                applying the projection layer to the pooled output of [`CLIPTextModel`].

        Example:
            ```python
            >>> from transformers import AutoTokenizer, CLIPModel
            ...
            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
            ```
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""

        Returns:
            image_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
                applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, CLIPModel
            ...
            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="pt")
            ...
            >>> image_features = model.get_image_features(**inputs)
            ```
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        pixel_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""

        Returns:
            Union[Tuple, CLIPOutput]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, CLIPModel
            ...
            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
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
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(ord=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(ord=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = ops.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class CLIPTextModelWithProjection(CLIPPreTrainedModel):

    """
    This class represents a CLIP text model with a projection layer for embedding text inputs.
    It inherits from the CLIPPreTrainedModel class.

    The CLIPTextModelWithProjection class is designed to process text inputs using the CLIP (Contrastive Language-Image Pretraining) model architecture.
    It incorporates a CLIPTextTransformer and a text projection layer to generate text embeddings.

    The class provides functionality for initializing the model with a CLIPTextConfig, accessing the input embeddings,
    setting the input embeddings, and constructing the model's outputs based on input text ids, attention masks, and position ids.

    The construct method takes optional input tensors representing text ids, attention masks, position ids,
    output attentions, output hidden states, and return dictionary flag.
    It returns a CLIPTextModelOutput object containing the text embeddings and other relevant information.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection
        ...
        >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        ...
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```

    """
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        """
        Initializes an instance of the CLIPTextModelWithProjection class.

        Args:
            self: The instance of the class.
            config (CLIPTextConfig):
                An instance of CLIPTextConfig class that contains the configuration parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.text_model = CLIPTextTransformer(config)

        self.text_projection = nn.Dense(config.hidden_size, config.projection_dim, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Method to get the input embeddings from the CLIPTextModelWithProjection instance.

        Args:
            self (object): Instance of the CLIPTextModelWithProjection class.
                Represents the current instance of the class.

        Returns:
            nn.Cell: Returns the input embeddings of type nn.Cell.
                Represents the token embeddings used by the text model.

        Raises:
            None.
        """
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the CLIPTextModelWithProjection class.

        Args:
            self (CLIPTextModelWithProjection): The instance of the CLIPTextModelWithProjection class.
            value: The input embeddings to be set for the text model.
                This should be a tensor or object that can be assigned to the `token_embedding` attribute of the text model.

        Returns:
            None: This method modifies the state of the text model by setting the input embeddings.

        Raises:
            None.
        """
        self.text_model.embeddings.token_embedding = value

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        r"""

        Returns:
            Union[Tuple, CLIPTextModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, CLIPTextModelWithProjection
            ...
            >>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            ...
            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> text_embeds = outputs.text_embeds
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


class CLIPVisionModelWithProjection(CLIPPreTrainedModel):

    '''
    Represents a vision model with projection for CLIP (Contrastive Language-Image Pre-training) framework.

    This class inherits from CLIPPreTrainedModel and includes methods for initializing the model,
    retrieving input embeddings, and constructing the model.

    The 'CLIPVisionModelWithProjection' class initializes with a configuration object of type 'CLIPVisionConfig'
    and sets up the vision model and visual projection.
    It provides a method to retrieve input embeddings and constructs the vision model with optional parameters for
    pixel values, attentions, hidden states, and return dictionary.
    The method returns image embeddings and other model outputs based on the input parameters.

    Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModelWithProjection
        ...
        >>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        ...
        >>> inputs = processor(images=image, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```
    '''
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        """
        Initializes a CLIPVisionModelWithProjection instance.

        Args:
            self: The instance itself.
            config (CLIPVisionConfig): The configuration object for the CLIPVisionModelWithProjection.
                It contains the necessary parameters for configuring the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.vision_model = CLIPVisionTransformer(config)

        self.visual_projection = nn.Dense(config.hidden_size, config.projection_dim, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Returns the input embeddings of the CLIPVisionModelWithProjection.

        Args:
            self (CLIPVisionModelWithProjection): An instance of CLIPVisionModelWithProjection class.

        Returns:
            nn.Cell: A neural network cell representing the input embeddings of the vision model.

        Raises:
            None.

        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionModelOutput]:
        r"""
        Returns:
            Union[Tuple, CLIPVisionModelOutput]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, CLIPVisionModelWithProjection
            ...
            >>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            >>> image_embeds = outputs.image_embeds
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )


class CLIPForImageClassification(CLIPPreTrainedModel):

    """
    The CLIPForImageClassification class represents a model for image classification using the Contrastive
    Language-Image Pretraining (CLIP) approach.
    It inherits from the CLIPPreTrainedModel class and implements the necessary methods for image classification tasks.

    Attributes:
        config (CLIPConfig):
            The configuration for the CLIP model, containing parameters such as num_labels, vision_model, and classifier.

    Methods:
        __init__:
            Initializes the CLIPForImageClassification model with the provided configuration.

        construct:
            Constructs the image classification model using the specified pixel values and labels.
                It returns the logits, loss, hidden states, and attentions if specified.

    Args:
        config (CLIPConfig): The configuration for the CLIP model.

    Returns:
        None

    Raises:
        None
    """
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPConfig) -> None:
        """
        Initializes an instance of the CLIPForImageClassification class.

        Args:
            self: The instance of the class.
            config (CLIPConfig): An instance of the CLIPConfig class containing configuration parameters for CLIP.
                It specifies the configuration settings needed for initializing the CLIP model.
                It must be of type CLIPConfig.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type CLIPConfig.
            ValueError: If the num_labels attribute in the config is invalid or missing.
            RuntimeError: If an error occurs during initialization of the vision model or classifier.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vision_model = CLIPVisionTransformer(config.vision_config)

        # Classifier head
        self.classifier = (
            nn.Dense(config.vision_config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # average pool the patch tokens
        sequence_output = ops.mean(sequence_output[:, 1:, :], axis=1)
        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
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

__all__ = [
    "CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
    "CLIPModel",
    "CLIPPreTrainedModel",
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "CLIPVisionModel",
    "CLIPVisionModelWithProjection",
    "CLIPForImageClassification",
]
