# coding=utf-8
# Copyright 2022 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
""" MindSpore BLIP model."""

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal, TruncatedNormal

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    logging,
)
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_blip_text import BlipTextLMHeadModel, BlipTextModel


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/blip-vqa-base"

def normalize(input, p=2.0, dim=1):
    """
    Normalize the input along a specified dimension using the specified p-norm.
    
    Args:
        input (Tensor): The input tensor to be normalized.
        p (float, optional): The p-norm to be used for normalization. Default is 2.0.
        dim (int, optional): The dimension along which the input tensor will be normalized. Default is 1.
    
    Returns:
        None
    
    Raises:
        TypeError: If input is not a tensor.
        ValueError: If p is not a positive float.
        ValueError: If dim is not a valid dimension for the input tensor.
    """
    return input / ops.norm(input, ord=p, dim=dim, keepdim=True)


# Copied from transformers.models.clip.modeling_clip.contrastive_loss
def contrastive_loss(logits: mindspore.Tensor) -> mindspore.Tensor:
    """
    Args:
        logits (mindspore.Tensor): The input logits for the contrastive loss function.
            It is a tensor containing the predicted values from the model.

    Returns:
        mindspore.Tensor: A tensor representing the contrastive loss value calculated based on the input logits.

    Raises:
        This function does not raise any exceptions.
    """
    return ops.cross_entropy(logits, ops.arange(len(logits)))


# Copied from transformers.models.clip.modeling_clip.clip_loss with clip->blip
def blip_loss(similarity: mindspore.Tensor) -> mindspore.Tensor:
    """
    Calculate Blip loss based on contrastive losses for caption and image similarities.
    
    Args:
        similarity (mindspore.Tensor): A tensor representing the similarity between captions and images.
            It is used to calculate both caption and image losses.
    
    Returns:
        mindspore.Tensor: A tensor representing the Blip loss value, which is the average of caption and image losses.
    
    Raises:
        None
    """
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


@dataclass
class BlipForConditionalGenerationModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`mindspore.Tensor`, *optional*, returned when `labels` is provided, `mindspore.Tensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional*):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`mindspore.Tensor` of shape `(batch_size, output_dim)`, *optional*):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[Tuple[mindspore.Tensor]] = None
    logits: Optional[Tuple[mindspore.Tensor]] = None
    image_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None

    @property
    def decoder_logits(self):
        """
        This method is part of the 'BlipForConditionalGenerationModelOutput' class and is used to retrieve the decoder logits attribute.
        
        Args:
            self: An instance of the 'BlipForConditionalGenerationModelOutput' class.
        
        Returns:
            None.
        
        Raises:
            FutureWarning: This method raises a 'FutureWarning' if the 'decoder_logits' attribute is used.
                This attribute is deprecated and will be removed in version 5 of Transformers. The 'logits' attribute
                should be used instead to retrieve the final output.

        Note:
            The 'decoder_logits' attribute is deprecated and will be removed in version 5 of Transformers.
            Please use the 'logits' attribute to retrieve the final output instead.
        """
        warnings.warn(
            "`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the `logits` attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.logits


@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
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
    loss: Optional[mindspore.Tensor] = None
    image_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class BlipImageTextMatchingModelOutput(ModelOutput):
    """
    Adapted from the base class for vision model's outputs that also contains image embeddings of the pooling of the
    last hidden states. This class also adds the loss term from the text decoder as well as the image-text similarity
    scores.

    Args:
        itm_score (`mindspore.Tensor`):
            The image-text similarity scores.
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Languge modeling loss from the text decoder.
        image_embeds (`mindspore.Tensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):

            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        vision_pooler_output (`mindspore.Tensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the vision of the vision-only branch of the model.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):

            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        question_embeds (`mindspore.Tensor`):
            The question embeddings obtained by the text projection layer.
    """
    itm_score: Optional[mindspore.Tensor] = None
    loss: Optional[mindspore.Tensor] = None
    image_embeds: Optional[mindspore.Tensor] = None
    last_hidden_state: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    vision_pooler_output: Optional[mindspore.Tensor] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None
    question_embeds: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class BlipOutput(ModelOutput):
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
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`mindspore.Tensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
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
        Converts the BlipOutput object to a tuple representation.

        Args:
            self (BlipOutput): The BlipOutput object to be converted to a tuple.

        Returns:
            Tuple[Any]: A tuple containing the values of the BlipOutput object.
            The 'text_model_output' and 'vision_model_output' keys are replaced with their respective tuple representations.

        Raises:
            AttributeError:
                If the 'text_model_output' or 'vision_model_output' attributes are not present
                or do not have a 'to_tuple' method.
        """
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class BlipVisionEmbeddings(nn.Cell):

    """
    The BlipVisionEmbeddings class represents the embeddings for vision data in the Blip framework.
    This class inherits from nn.Cell and provides methods for initializing and constructing vision embeddings.

    Attributes:
        config (BlipVisionConfig): The configuration object for the BlipVisionEmbeddings.
        embed_dim (int): The dimension of the embeddings.
        image_size (int): The size of the input image.
        patch_size (int): The size of the patches in the image.
        class_embedding (Parameter): Embedding for the class.
        patch_embedding (nn.Conv2d): Convolutional layer for patch embedding.
        num_patches (int): The number of patches in the image.
        num_positions (int): The total number of positions, including patches and class embedding.
        position_embedding (Parameter): Embedding for positional encoding.

    Methods:
        __init__: Initializes the BlipVisionEmbeddings with the given configuration.
        construct: Constructs the embeddings for the input pixel values.

    """
    def __init__(self, config: BlipVisionConfig):
        """
        Initializes an instance of the BlipVisionEmbeddings class.

        Args:
            self: The instance of the BlipVisionEmbeddings class.
            config (BlipVisionConfig):
                An object of type BlipVisionConfig containing configuration parameters.
                This object specifies the hidden size, image size, and patch size for the embeddings.

                Parameters:

                - hidden_size (int): The dimension of the embedding space.
                - image_size (int): The size of the input image.
                - patch_size (int): The size of each patch in the image.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = Parameter(ops.randn(1, 1, self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, has_bias=True
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = Parameter(ops.randn(1, self.num_positions, self.embed_dim))

    def construct(self, pixel_values: mindspore.Tensor) -> mindspore.Tensor:
        '''
        Constructs the embeddings for the BlipVisionEmbeddings class.

        Args:
            self (BlipVisionEmbeddings): The instance of the BlipVisionEmbeddings class.
            pixel_values (mindspore.Tensor): The input tensor containing pixel values. It should have a shape of (batch_size, channels, height, width).

        Returns:
            mindspore.Tensor: The constructed embeddings tensor.
                It has a shape of (batch_size, num_patches + 1, embedding_dim), where num_patches is the number of
                patches obtained from the input tensor and
                embedding_dim is the dimension of the embeddings.

        Raises:
            TypeError: If the pixel_values parameter is not of type mindspore.Tensor.
            ValueError: If the pixel_values tensor does not have the correct shape.
            TypeError: If the dtype of the pixel_values tensor is not compatible with the dtype of the patch_embedding weights.
            RuntimeError: If there is an error in the patch_embedding operation.
            RuntimeError: If there is an error in the flatten operation.
            RuntimeError: If there is an error in the swapaxes operation.
            RuntimeError: If there is an error in the class_embedding expansion operation.
            RuntimeError: If there is an error in the concatenation operation.
            RuntimeError: If there is an error in the addition operation.
        '''
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(start_dim=2).swapaxes(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = ops.cat([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding[:, : embeddings.shape[1], :].to(target_dtype)
        return embeddings


# Copied from transformers.models.clip.modeling_clip.CLIPTextEmbeddings with CLIP->Blip
class BlipTextEmbeddings(nn.Cell):

    """
    This class represents a text embeddings module for BlipText, providing functionality to construct embeddings for input tokens with position information.
    The BlipTextEmbeddings class inherits from nn.Cell and implements methods to initialize embeddings based on configuration settings and construct embeddings for input tokens with optional position
    information.
    """
    def __init__(self, config: BlipTextConfig):
        """
        Initializes an instance of the BlipTextEmbeddings class.

        Args:
            self: The current instance of the class.
            config (BlipTextConfig):
                An object of the BlipTextConfig class containing configuration parameters.

                - The 'config' object should have a 'hidden_size' attribute specifying the size of the embedding dimension.
                - The 'config' object should have a 'vocab_size' attribute specifying the size of the vocabulary.
                - The 'config' object should have a 'max_position_embeddings' attribute specifying the maximum number of position embeddings.

        Returns:
            None

        Raises:
            None
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
        '''
        Constructs BlipTextEmbeddings with given input_ids, position_ids, and inputs_embeds.

        Args:
            self (BlipTextEmbeddings): The object instance itself.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing token indices. Default is None.
            position_ids (Optional[mindspore.Tensor]): The input tensor containing position indices. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The input tensor containing pre-computed token embeddings. Default is None.

        Returns:
            mindspore.Tensor: The constructed embeddings tensor.

        Raises:
            ValueError: If input_ids and inputs_embeds are both None.
            ValueError: If input_ids and inputs_embeds have different sequence lengths.
        '''
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class BlipAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config):
        """
        Initializes the BlipAttention class with the provided configuration.

        Args:
            self (BlipAttention): The instance of the BlipAttention class.
            config (object):
                The configuration object containing the parameters for the BlipAttention module.

                - config.hidden_size (int): The size of the hidden layers.
                - config.num_attention_heads (int): The number of attention heads.
                - config.attention_dropout (float): The dropout rate for attention weights.

        Returns:
            None.

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
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
        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.qkv = nn.Dense(self.embed_dim, 3 * self.embed_dim)

        self.projection = nn.Dense(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' is a part of the 'BlipAttention' class and is used to reshape the input tensor for attention calculation.

        Args:
            self (BlipAttention): The instance of the BlipAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the input sequence.
            bsz (int): The batch size of the input data.

        Returns:
            None: This method returns None as the reshaped tensor is returned directly without assignment.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.shape

        mixed_qkv = (
            self.qkv(hidden_states)
            .reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_states, key_states.swapaxes(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.shape[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.projection(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Blip
class BlipMLP(nn.Cell):

    """
    The BlipMLP class represents a multi-layer perceptron (MLP) model for neural network computations.
    This class inherits from the nn.Cell module and provides functionality for constructing and applying a
    multi-layer perceptron with configurable activation functions and layer sizes.

    Attributes:
        config (object): The configuration object containing parameters for the MLP model.
        activation_fn (function): The activation function for hidden layers, derived from the ACT2FN dictionary in the configuration.
        fc1 (nn.Dense): The first fully connected layer with a size defined by the configuration.
        fc2 (nn.Dense): The second fully connected layer with a size defined by the configuration.

    Methods:
        construct:
            Constructs the multi-layer perceptron by applying the fully connected layers and activation functions to the input tensor.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BlipMLP class.

        Args:
            self: The instance of the BlipMLP class.
            config: An object containing configuration parameters for the BlipMLP model.
                It should have the following attributes:

                - hidden_act (str): The activation function to be used in the hidden layers.
                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a multi-layer perceptron (MLP) using the given hidden states.

        Args:
            self (BlipMLP): An instance of the BlipMLP class.
            hidden_states (mindspore.Tensor): The input hidden states to the MLP.

        Returns:
            mindspore.Tensor: The output tensor after passing through the MLP.

        Raises:
            None.

        This method takes in the hidden states and applies a series of linear transformations and non-linear
        activations to them. It first passes the hidden states through a fully connected layer (self.fc1),
        then applies an activation function (self.activation_fn) to the resulting tensor. It then passes the tensor
        through another fully connected layer (self.fc2) before returning the final output tensor.

        The purpose of this method is to construct the MLP and process the given hidden states to obtain a transformed
        tensor. The returned tensor can be used for further computations or as an output of the MLP.

        Note:
            The dimensions of the hidden_states tensor should be compatible with the dimensions of the MLP's layers
            in order for the method to execute successfully.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BlipEncoderLayer(nn.Cell):

    """
    This class represents a single layer of the Blip encoder. It consists of self-attention mechanism followed by
    a feedforward neural network layer with layer normalization and residual connections.

    This class initializes with a BlipConfig object to set up the layer configuration parameters.
    The `construct` method processes the input hidden_states through self-attention, layer normalization, and
    a feedforward neural network in sequence, with the option to return attention weights if specified.

    Args:
        config (BlipConfig): Configuration object containing parameters for the layer.

    Methods:
        __init__: Constructor method to initialize the encoder layer with the given configuration.
        construct:
            Process the input hidden_states through self-attention, layer normalization, and feedforward neural network,
            and return the output tensor(s).

    Attributes:
        embed_dim (int): Dimension of the hidden states in the layer.
        self_attn (BlipAttention): Self-attention mechanism.
        layer_norm1 (nn.LayerNorm): Layer normalization module for the first layer normalization.
        mlp (BlipMLP): Feedforward neural network module.
        layer_norm2 (nn.LayerNorm): Layer normalization module for the second layer normalization.

    Returns:
        Tuple[mindspore.Tensor]: Tuple containing the final hidden states of the layer.
            If output_attentions is True, the tuple also includes the attention weights.

    Note:
        - The attention_mask should have the shape `(batch, 1, tgt_len, src_len)` with padding elements indicated
        by very large negative values.
        - When output_attentions is True, the method returns the attention weights along with the hidden states.
    """
    def __init__(self, config: BlipConfig):
        """
        Args:
            self (BlipEncoderLayer): The instance of the BlipEncoderLayer class.
            config (BlipConfig):
                An instance of BlipConfig class containing configuration parameters for the BlipEncoderLayer.

                - hidden_size (int): The dimensionality of the input and output features.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type BlipConfig.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = BlipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = BlipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
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
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BlipConfig
    base_model_prefix = "blip"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(cell, (nn.Conv2d, nn.Dense, nn.Embedding)):
            cell.weight.set_data(initializer(Normal(factor), cell.weight.shape, cell.weight.dtype))
            if hasattr(cell, "bias") and cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        if isinstance(cell, BlipVisionEmbeddings):
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range

            cell.position_embedding.set_data(initializer(TruncatedNormal(factor), cell.position_embedding.shape, cell.position_embedding.dtype))
            cell.class_embedding.set_data(initializer(TruncatedNormal(factor), cell.class_embedding.shape, cell.class_embedding.dtype))

        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        elif isinstance(cell, nn.Dense) and cell.bias is not None:
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class BlipEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`BlipEncoderLayer`].

    Args:
        config (`BlipConfig`):
            The corresponding vision configuration for the `BlipEncoder`.
    """
    def __init__(self, config: BlipConfig):
        """
        Initializes a BlipEncoder object with the provided configuration.

        Args:
            self (BlipEncoder): The instance of the BlipEncoder class.
            config (BlipConfig): An object containing configuration settings for the BlipEncoder.
                The config parameter must be an instance of the BlipConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layers = nn.CellList([BlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

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
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
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


class BlipVisionModel(BlipPreTrainedModel):

    """
    A class representing the BlipVisionModel for vision tasks.

    This class inherits from the BlipPreTrainedModel and provides methods to construct the model, get input embeddings, and return the outputs.

    Attributes:
        config (BlipVisionConfig): The configuration for the BlipVisionModel.
        embeddings (BlipVisionEmbeddings): The embeddings layer for the BlipVisionModel.
        encoder (BlipEncoder): The encoder layer for the BlipVisionModel.
        post_layernorm (nn.LayerNorm): The post layer normalization layer for the BlipVisionModel.

    Methods:
        __init__: Initializes the BlipVisionModel with the given configuration.
        construct: Constructs the BlipVisionModel and returns the model outputs.
        get_input_embeddings: Returns the input embeddings for the BlipVisionModel.
    """
    main_input_name = "pixel_values"
    config_class = BlipVisionConfig

    def __init__(self, config: BlipVisionConfig):
        """
        Initializes a new instance of the BlipVisionModel class.

        Args:
            self: The object itself.
            config (BlipVisionConfig): The configuration object that holds all the necessary parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = BlipVisionEmbeddings(config)
        self.encoder = BlipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, epsilon=config.layer_norm_eps)

        self.post_init()

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

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

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

    def get_input_embeddings(self):
        """
        Returns the input embeddings from the BlipVisionModel.

        Args:
            self (BlipVisionModel): An instance of the BlipVisionModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings


class BlipModel(BlipPreTrainedModel):

    """
    BlipModel

    BlipModel is a class that represents a multimodal model for processing both text and images.
    It inherits from BlipPreTrainedModel and includes methods for obtaining text and image features, as well as for
    constructing the model output.

    Example:
        ```python
        >>> from transformers import AutoProcessor, BlipModel
        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ...
        >>> from PIL import Image
        >>> import requests
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pt")
        >>> image_features = model.get_image_features(**inputs)
        ...
        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```
    """
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        """
        Initializes an instance of the BlipModel class.

        Args:
            self: The current instance of the BlipModel class.
            config (BlipConfig): The configuration object for the BlipModel.
                It should contain the following attributes:

                - text_config (BlipTextConfig): The configuration object for the text model component of BlipModel.
                It should be of type BlipTextConfig and contain the necessary parameters for the text model.
                - vision_config (BlipVisionConfig): The configuration object for the vision model component of BlipModel.
                It should be of type BlipVisionConfig and contain the necessary parameters for the vision model.
                - projection_dim (int): The dimension of the projection space.
                - logit_scale_init_value (float): The initial value for the logit scale parameter.

        Returns:
            None

        Raises:
            ValueError: If config.text_config is not of type BlipTextConfig.
            ValueError: If config.vision_config is not of type BlipVisionConfig.

        Note:
            This method initializes the BlipModel instance by setting the projection dimension, text embedding dimension,
            vision embedding dimension, text model, vision model, visual projection, text projection, and logit scale
            attributes based on the provided configuration. It also calls the post_init method.
        """
        super().__init__(config)

        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = BlipTextModel(text_config)
        self.vision_model = BlipVisionModel(vision_config)

        self.visual_projection = nn.Dense(self.vision_embed_dim, self.projection_dim, has_bias=False)
        self.text_projection = nn.Dense(self.text_embed_dim, self.projection_dim, has_bias=False)
        self.logit_scale = Parameter(mindspore.tensor(self.config.logit_scale_init_value))

        # Initialize weights and apply final processing
        self.post_init()

    def get_text_features(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            text_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`BlipTextModel`].

        Example:
            ```python
            >>> from transformers import AutoProcessor, BlipModel
            ...
            >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            ...
            >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]
        text_features = self.text_projection(pooled_output)

        return text_features

    def get_image_features(
        self,
        pixel_values: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> mindspore.Tensor:
        r"""
        Returns:
            image_features (`mindspore.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
                applying the projection layer to the pooled output of [`BlipVisionModel`].

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipModel
            ...
            >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="pt")
            ...
            >>> image_features = model.get_image_features(**inputs)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)

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
    ) -> Union[Tuple, BlipOutput]:
        r"""
        Returns:
            `Union[Tuple, BlipOutput]`

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipModel
            ...
            >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
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
        # Use BLIP model's config for some fields (if specified) instead of those of vision & text components.
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
            loss = blip_loss(logits_per_text)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return BlipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


class BlipForConditionalGeneration(BlipPreTrainedModel):

    """
    A class representing the BlipForConditionalGeneration model for image captioning.

    This class extends the BlipPreTrainedModel class and provides methods for initializing the model,
    generating image captions, and constructing the model's architecture.

    Attributes:
        vision_model (BlipVisionModel): The vision model used for extracting image features.
        text_decoder (BlipTextLMHeadModel): The text decoder model used for generating captions.
        decoder_input_ids (int): The token ID to start the decoder input sequence.
        decoder_pad_token_id (int): The token ID used for padding the decoder input sequence.

    Methods:
        __init__: Initializes the BlipForConditionalGeneration model.
        get_input_embeddings: Returns the input embeddings of the vision model.
        construct: Constructs the model architecture and generates image captions.
        generate: Generates image captions based on the input image.

    Example:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration
        ...
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        ...
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"
        ...
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        ...
        >>> outputs = model(**inputs)
        ```

    Overrides:
        generate: Overrides the generate function to enable the model to be used as a conditional generator.

    """
    config_class = BlipConfig
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]
    main_input_name = "pixel_values"

    def __init__(self, config: BlipConfig):
        """
        Initializes an instance of the BlipForConditionalGeneration class.

        Args:
            self (BlipForConditionalGeneration): The instance of the BlipForConditionalGeneration class.
            config (BlipConfig): An object representing the configuration settings for the Blip model.
                It contains the necessary configurations for the vision model and text decoder.
                It is expected that the config parameter is of type BlipConfig.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type BlipConfig.
            ValueError: If the config parameter is missing required configuration settings.
        """
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        self.decoder_input_ids = config.text_config.bos_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        This method returns the input embeddings for the BlipForConditionalGeneration class.

        Args:
            self (BlipForConditionalGeneration): The instance of the BlipForConditionalGeneration class.

        Returns:
            nn.Cell: The input embeddings for the BlipForConditionalGeneration class. This is an instance of the nn.Cell class.

        Raises:
            None.

        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        pixel_values: mindspore.Tensor,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipForConditionalGenerationModelOutput]:
        r"""
        Returns:
            Union[Tuple, BlipForConditionalGenerationModelOutput]

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipForConditionalGeneration
            ...
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> text = "A picture of"
            ...
            >>> inputs = processor(images=image, text=text, return_tensors="pt")
            ...
            >>> outputs = model(**inputs)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return BlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def generate(
        self,
        pixel_values: mindspore.Tensor,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        **generate_kwargs,
    ) -> mindspore.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (*mindspore.Tensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            input_ids (*mindspore.Tensor* of shape *(batch_size, sequence_length)*, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (*mindspore.Tensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipForConditionalGeneration
            ...
            >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> inputs = processor(images=image, return_tensors="pt")
            ...
            >>> outputs = model.generate(**inputs)
            >>> print(processor.decode(outputs[0], skip_special_tokens=True))
            two cats sleeping on a couch
            ```
        """
        batch_size = pixel_values.shape[0]
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = ops.ones(image_embeds.shape[:-1], dtype=mindspore.int64)

        if isinstance(input_ids, list):
            input_ids = mindspore.Tensor(input_ids)
        elif input_ids is None:
            input_ids = (
                mindspore.Tensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
            )

        input_ids[:, 0] = self.config.text_config.bos_token_id
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        return outputs


class BlipForQuestionAnswering(BlipPreTrainedModel):

    """
    BlipForQuestionAnswering is a class that represents a model for question answering using both text and vision inputs. It is designed to be used with the BlipPreTrainedModel base class.

    This class has the following attributes:

    - vision_model: An instance of the BlipVisionModel class that handles the vision inputs.
    - text_encoder: An instance of the BlipTextModel class that encodes the text inputs.
    - text_decoder: An instance of the BlipTextLMHeadModel class that decodes the text inputs.
    - decoder_pad_token_id: The ID of the padding token used in the decoder.
    - decoder_start_token_id: The ID of the start token used in the decoder.

    The BlipForQuestionAnswering class provides the following methods:

    1. __init__:
    Initializes the BlipForQuestionAnswering instance with the given configuration.
    2. get_input_embeddings:
    Returns the input embeddings of the vision model.
    3. construct:
    Constructs the model and performs the forward pass. Returns the model outputs.
    4. generate:
    Generates text outputs based on the given input IDs and pixel values.

    Please refer to the code examples in the docstring for more information on how to use the BlipForQuestionAnswering
    class for training and inference.

    Note:
        This documentation is auto-generated and may not capture all the intricacies of the class implementation.
        For more details, please refer to the source code.
    """
    config_class = BlipConfig
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]

    def __init__(self, config: BlipConfig):
        """
        Initializes an instance of BlipForQuestionAnswering.

        Args:
            self: The instance of the class.
            config (BlipConfig): An instance of BlipConfig containing the configuration for the model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        This method returns the input embeddings from the vision model for question answering.

        Args:
            self (BlipForQuestionAnswering): The instance of the BlipForQuestionAnswering class.

        Returns:
            nn.Cell: The input embeddings from the vision model, which is of type nn.Cell.

        Raises:
            None
        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        input_ids: mindspore.Tensor,
        pixel_values: mindspore.Tensor,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:
            `Union[Tuple, BlipTextVisionModelOutput]`

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipForQuestionAnswering
            ...
            >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            ...
            >>> # training
            >>> text = "How many cats are in the picture?"
            >>> label = "2"
            >>> inputs = processor(images=image, text=text, return_tensors="pt")
            >>> labels = processor(text=label, return_tensors="pt").input_ids
            ...
            >>> inputs["labels"] = labels
            >>> outputs = model(**inputs)
            >>> loss = outputs.loss
            >>> loss.backward()
            ...
            >>> # inference
            >>> text = "How many cats are in the picture?"
            >>> inputs = processor(images=image, text=text, return_tensors="pt")
            >>> outputs = model.generate(**inputs)
            >>> print(processor.decode(outputs[0], skip_special_tokens=True))
            2
            ```
        """
        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either `decoder_input_ids` or `labels` should be passed when calling `forward` with"
                " `BlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you"
                " are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`"
            )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_attention_mask = ops.ones(image_embeds.shape[:-1], dtype=mindspore.int64)

        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=return_dict,
        )

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            decoder_input_ids = labels

        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

        answer_output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        if labels is not None:
            decoder_loss = answer_output.loss.mean() if return_dict else answer_output[0].mean()
        else:
            decoder_loss = None

        if not return_dict:
            outputs = (decoder_loss, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return BlipTextVisionModelOutput(
            loss=decoder_loss,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    def generate(
        self,
        input_ids: mindspore.Tensor,
        pixel_values: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        **generate_kwargs,
    ) -> mindspore.Tensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (*mindspore.Tensor* of shape *(batch_size, sequence_length)*):
                The sequence used as a prompt for the generation.
            pixel_values (*mindspore.Tensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            attention_mask (*mindspore.Tensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            **generate_kwargs:
                Additional arguments passed to the *generate* function of the decoder

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipForQuestionAnswering
            ...
            >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> text = "How many cats are in the picture?"
            ...
            >>> inputs = processor(images=image, text=text, return_tensors="pt")
            ...
            >>> outputs = model.generate(**inputs)
            >>> print(processor.decode(outputs[0], skip_special_tokens=True))
            2
            ```
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = ops.ones(image_embeds.shape[:-1], dtype=mindspore.int64)

        if isinstance(input_ids, list):
            input_ids = mindspore.Tensor(input_ids)

        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        question_embeds = question_outputs[0]

        question_attention_mask = ops.ones(question_embeds.shape[:-1], dtype=mindspore.int64)

        bos_ids = ops.full(
            (question_embeds.shape[0], 1), fill_value=self.decoder_start_token_id
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        return outputs


class BlipForImageTextRetrieval(BlipPreTrainedModel):

    """
    BlipForImageTextRetrieval is a class that implements a model for image-text retrieval tasks.
    It is designed to retrieve relevant text based on input images and vice versa. This class inherits from
    BlipPreTrainedModel.

    The class's constructor initializes the model with the provided configuration.
    It sets up the vision model, text encoder, projection layers, and other necessary components for image-text retrieval.

    The 'get_input_embeddings' method returns the patch embeddings from the vision model.

    The 'construct' method takes input image and text tensors and constructs the output based on the specified parameters.
    It utilizes the vision model to extract image features and the text encoder to process input text.
    Depending on the 'use_itm_head' parameter, the method either computes the similarity score between image and text
    features or uses the image and text projections for matching.

    The method also handles optional parameters for controlling the output format and behavior. It provides examples on
    how to use the BlipForImageTextRetrieval class for image-text retrieval tasks.

    Note:
        This docstring is a high-level overview and does not include method signatures or detailed implementation details.
    """
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        """
        Initializes an instance of the BlipForImageTextRetrieval class.

        Args:
            self: The instance of the class itself.
            config (BlipConfig):
                The configuration object containing various settings for the BlipForImageTextRetrieval model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # vision projection layer
        self.vision_proj = nn.Dense(config.vision_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_proj = nn.Dense(config.text_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Dense(config.text_config.hidden_size, 2)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Cell:
        """
        Method to get the input embeddings from the vision model for image-text retrieval.

        Args:
            self (BlipForImageTextRetrieval): The instance of the BlipForImageTextRetrieval class.
                This parameter is required to access the vision model and its embeddings.

        Returns:
            nn.Cell: A neural network cell representing the input embeddings obtained from the vision model.
                These embeddings are used for matching image features with text features in the retrieval process.

        Raises:
            None
        """
        return self.vision_model.embeddings.patch_embedding

    def construct(
        self,
        input_ids: mindspore.Tensor,
        pixel_values: mindspore.Tensor,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:
            `Union[Tuple, BlipTextVisionModelOutput]`

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor, BlipForImageTextRetrieval
            ...
            >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
            >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
            ...
            >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            >>> image = Image.open(requests.get(url, stream=True).raw)
            >>> text = "an image of a cat"
            ...
            >>> inputs = processor(images=image, text=text, return_tensors="pt")
            >>> outputs = model(**inputs)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        image_atts = ops.ones(image_embeds.shape[:-1], dtype=mindspore.int64)

        if use_itm_head:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            output = self.itm_head(question_embeds[:, 0, :])
        else:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            output = image_feat @ text_feat.t()

        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple(output for output in outputs if output is not None)

        return BlipImageTextMatchingModelOutput(
            itm_score=output,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            question_embeds=question_embeds,
        )

__all__ = [
    "BlipModel",
    "BlipPreTrainedModel",
    "BlipForConditionalGeneration",
    "BlipForQuestionAnswering",
    "BlipVisionModel",
    "BlipTextModel",
    "BlipForImageTextRetrieval",
]
