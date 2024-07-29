# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Musicgen model."""
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import mindspore
from mindnlp.core import nn, ops, get_default_dtype
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    ModelOutput,
    Seq2SeqLMOutput,
)
from ...modeling_utils import PreTrainedModel
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MusicgenConfig"
_CHECKPOINT_FOR_DOC = "facebook/musicgen-small"

MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/musicgen-small"]

@dataclass
class MusicgenUnconditionalInput(ModelOutput):
    """
    Args:
        encoder_outputs  (`Tuple[mindspore.Tensor]` of length 1, with tensor shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the text encoder model.
        attention_mask (`mindspore.Tensor`)  of shape `(batch_size, sequence_length)`, *optional*):
            Encoder attention mask to avoid performing attention on padding token indices. Mask values selected in `[0,
            1]`: 1 for tokens that are **not masked**, 0 for tokens that are **masked**.
        guidance_scale (`float`, *optional*):
            Guidance scale for classifier free guidance, setting the balance between the conditional logits (predicted
            from the prompts) and the unconditional logits (predicted without prompts).
    """
    encoder_outputs: Tuple[mindspore.Tensor] = None
    attention_mask: mindspore.Tensor = None
    guidance_scale: float = None


# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].copy()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class MusicgenSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int):
        """Initializes an instance of the MusicgenSinusoidalPositionalEmbedding class.
        
        Args:
            self: The object instance itself.
            num_positions (int): The number of positions to be embedded.
            embedding_dim (int): The dimensionality of the embedding.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        """
        Method to create and set the weights for sinusoidal positional embeddings in the
        MusicgenSinusoidalPositionalEmbedding class.
        
        Args:
            self (object): The instance of the MusicgenSinusoidalPositionalEmbedding class.
            num_embeddings (int): The number of embeddings to be created.
                Must be a positive integer representing the total number of embeddings to generate.
            embedding_dim (int): The dimensionality of each embedding vector.
                Must be a positive integer representing the size of each embedding vector.
        
        Returns:
            None.
        
        Raises:
            AttributeError: If the 'weights' attribute is not found in the instance.
            TypeError: If the data type of the embedding weights cannot be converted to the same data type as
                the existing weights.
        """
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype and device of the param
            emb_weights = emb_weights.to(dtype=self.weights.dtype) # pylint: disable=access-member-before-definition

        self.weights = Parameter(emb_weights)
        self.weights.requires_grad = False

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = ops.exp(ops.arange(half_dim, dtype=mindspore.int64).float() * -emb)
        emb = ops.arange(num_embeddings, dtype=mindspore.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = ops.cat([ops.cos(emb), ops.sin(emb)], axis=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = ops.cat([emb, ops.zeros(num_embeddings, 1)], axis=1)
        return emb.to(get_default_dtype())

    def forward(self, input_ids: mindspore.Tensor, past_key_values_length: int = 0):
        """
        Constructs sinusoidal positional embeddings for music generation.
        
        Args:
            self (MusicgenSinusoidalPositionalEmbedding): The instance of the MusicgenSinusoidalPositionalEmbedding class.
            input_ids (mindspore.Tensor): The input tensor representing the musical notes.
                It has the shape (batch size, codebooks, sequence length).
            past_key_values_length (int, optional): The length of past key values. Defaults to 0.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the sequence length is greater than the length of the weights.
            RuntimeError: If an error occurs during the index selection process.
        """
        bsz, codebooks, seq_len = input_ids.shape
        # Create the position ids from the input token ids.
        position_ids = (ops.arange(seq_len) + past_key_values_length)
        # expand embeddings if needed
        if seq_len > self.weights.shape[0]:
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1))


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Musicgen
class MusicgenAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MusicgenConfig] = None,
    ):
        """
        Initializes an instance of the MusicgenAttention class.
        
        Args:
            self: The instance of the class.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Whether the attention layer is used as a decoder. Defaults to False.
            bias (bool, optional): Whether to include bias in the linear projections. Defaults to True.
            is_causal (bool, optional): Whether the attention layer should operate in a causal manner.
                Defaults to False.
            config (Optional[MusicgenConfig], optional): The configuration object for the Musicgen model.
                Defaults to None.
        
        Returns:
            None
        
        Raises:
            ValueError: If the embed_dim is not divisible by num_heads.
        
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor for the attention mechanism in the MusicgenAttention class.
        
        Args:
            self (MusicgenAttention): An instance of the MusicgenAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of each sequence in the input tensor.
            bsz (int): The batch size of the input tensor.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        key_value_states: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MusicgenDecoderLayer(nn.Module):

    """
    MusicgenDecoderLayer represents a single layer of a music generation decoder model. 
    This class implements the decoding logic for generating music sequences based on the input hidden states and
    attention mechanisms.
    
    The `MusicgenDecoderLayer` class inherits from `nn.Module` and contains methods for initializing the layer and
    processing input tensors through the decoding pipeline.
    The layer consists of self-attention mechanisms, feedforward neural networks, and layer normalization operations.
    
    The `__init__` method initializes the decoder layer with configuration parameters such as hidden size, number of
    attention heads, dropout probabilities, activation functions, and layer normalization.
    
    The `forward` method processes the input hidden states along with optional arguments like attention masks,
    encoder hidden states, and past key-value states.
    It applies self-attention mechanisms, cross-attention if encoder hidden states are provided, and feedforward neural
    networks to generate the output hidden states.
    The method also supports outputting attention weights, caching key-value states, and returning intermediate values
    if specified.
    
    Please refer to the method signatures and argument descriptions for detailed information on the input and output
    tensors expected by each method.
    """
    def __init__(self, config: MusicgenDecoderConfig):
        """
        Initializes a MusicgenDecoderLayer.
        
        Args:
            self (object): The instance of the MusicgenDecoderLayer class.
            config (MusicgenDecoderConfig):
                An instance of MusicgenDecoderConfig containing configuration parameters for the decoder layer.

                - hidden_size (int): The size of the hidden layer.
                - num_attention_heads (int): The number of attention heads.
                - attention_dropout (float): The dropout rate for attention layers.
                - dropout (float): The dropout rate.
                - activation_function (str): The activation function for the layer.
                - activation_dropout (float): The dropout rate for activation function.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = MusicgenAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MusicgenAttention(
            self.embed_dim,
            config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer.forward
    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_layer_head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`mindspore.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`mindspore.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`mindspore.Tensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(mindspore.Tensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MusicgenPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MusicgenDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MusicgenDecoderLayer", "MusicgenAttention"]

    def _init_weights(self, cell):
        """
        Initializes the weights of the specified cell.

        Args:
            self (MusicgenPreTrainedModel): An instance of the MusicgenPreTrainedModel class.
            cell: The cell for which the weights need to be initialized.

        Returns:
            None: This method modifies the weights of the specified cell in-place.

        Raises:
            None.

        The method initializes the weights of the provided cell based on its type. The initialization process varies
        depending on the type of the cell. The supported cell types are Dense, Conv1d, Embedding, and LayerNorm.

        For Dense and Conv1d cells:

        - The weight tensor is initialized using the Normal initializer with the standard deviation specified in the
        'initializer_factor' configuration attribute.
        - If the cell has bias, the bias tensor is initialized with zeros.

        For Embedding cells:

        - The weight tensor is initialized using a normal distribution with mean 0.0 and standard deviation specified
        in the 'initializer_factor' configuration attribute.
        - If the cell has a padding_idx attribute, the corresponding element in the weight tensor is set to 0.

        For LayerNorm cells:

        - The weight tensor is initialized with ones.
        - The bias tensor is initialized with zeros.

        Note: This method modifies the weights of the cell in-place and does not return any value.
        """
        std = self.config.initializer_factor
        if isinstance(cell, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(std),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class MusicgenDecoder(MusicgenPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenDecoderLayer`]
    """
    def __init__(self, config: MusicgenDecoderConfig):
        """
        Initializes a MusicgenDecoder object with the provided configuration.

        Args:
            self: The instance of the MusicgenDecoder class.
            config (MusicgenDecoderConfig):
                An object containing the configuration parameters for the decoder.

                - dropout (float): The dropout probability for regularization.
                - layerdrop (float): The probability of dropping entire layers.
                - max_position_embeddings (int): The maximum number of positions in the input sequence.
                - hidden_size (int): The size of the hidden layers.
                - num_codebooks (int): The number of codebooks used for encoding.
                - scale_embedding (bool): A flag indicating whether to scale the embedding.
                - vocab_size (int): The size of the vocabulary.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.d_model = config.hidden_size
        self.num_codebooks = config.num_codebooks
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        embed_dim = config.vocab_size + 1
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)]
        )

        self.embed_positions = MusicgenSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.layers = nn.ModuleList([MusicgenDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings used by the MusicgenDecoder class.

        Args:
            self: An instance of the MusicgenDecoder class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the MusicgenDecoder class.

        Args:
            self (object): The instance of the MusicgenDecoder class.
            value (object): The input embeddings value to be set for the decoder.
                It should be of type 'None' or a tensor containing the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the MusicgenDecoder.

        This method is used to forward the MusicgenDecoder. It takes the following parameters:

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor, optional): The input tensor representing the IDs of the input sequences. Default is None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Default is None.
            encoder_hidden_states (mindspore.Tensor, optional): The hidden states of the encoder. Default is None.
            encoder_attention_mask (mindspore.Tensor, optional): The attention mask tensor for the encoder. Default is None.
            head_mask (mindspore.Tensor, optional): The mask tensor for the attention heads. Default is None.
            cross_attn_head_mask (mindspore.Tensor, optional): The mask tensor for the cross-attention heads. Default is None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values for caching. Default is None.
            inputs_embeds (mindspore.Tensor, optional): The embedded input tensor. Default is None.
            use_cache (bool, optional): Whether to use caching. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
                The return value is either a tuple containing the
                hidden states, next cache, all hidden states, all self attentions, and all cross
                attentions, or an instance of BaseModelOutputWithPastAndCrossAttentions.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are specified.
            ValueError: If neither `input_ids` nor `inputs_embeds` are specified.
            ValueError: If the length of `head_mask` or `cross_attn_head_mask` does not match the number of layers.
            Warning: If both `use_cache=True` and `gradient_checkpointing=True`.

        Note:
            - The `input_ids` and `inputs_embeds` parameters cannot be specified at the same time.
            - Either `input_ids` or `inputs_embeds` must be specified.
            - The length of `head_mask` and `cross_attn_head_mask` should match the number of layers.
            - If `use_cache=True` and `gradient_checkpointing=True`, `use_cache` will be set to `False`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # (bsz * codebooks, seq_len) -> (bsz, codebooks, seq_len)
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = sum(self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks))

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.shape[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MusicgenModel(MusicgenPreTrainedModel):

    """
    This class represents a Musicgen model, a subclass of MusicgenPreTrainedModel. It provides functionality for 
    generating music sequences based on given input.

    Please note that this class inherits from MusicgenPreTrainedModel and extends its functionality.

    For more details on the methods and their parameters, please refer to the method docstrings.
    """
    def __init__(self, config: MusicgenDecoderConfig):
        """
        Initializes a new instance of the MusicgenModel class.

        Args:
            self: The MusicgenModel instance.
            config (MusicgenDecoderConfig): The configuration object for the MusicgenDecoder.

        Returns:
            None.

        Raises:
            None.

        This method initializes the MusicgenModel by calling the superclass's __init__ method with the provided config 
        parameter. It then creates a new instance of the MusicgenDecoder class using the same config object and assigns 
        it to the 'decoder' attribute of the MusicgenModel instance. Finally, it calls the 'post_init' method to perform 
        any additional initialization tasks.
        """
        super().__init__(config)
        self.decoder = MusicgenDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method retrieves the input embeddings from the MusicgenModel class.

        Args:
            self (MusicgenModel): An instance of the MusicgenModel class.

        Returns:
            None

        Raises:
            None.
        """
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MusicgenModel.

        Args:
            self (MusicgenModel): The instance of the MusicgenModel class.
            value: The input embeddings to be set for the decoder in the MusicgenModel.
                It should be of type 'torch.nn.Embedding' and represent the embedding tokens.

        Returns:
            None.

        Raises:
            None.
        """
        self.decoder.embed_tokens = value

    def get_decoder(self):
        """
        Returns the decoder object used by the MusicgenModel.

        Args:
            self: An instance of the MusicgenModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the MusicgenModel.

        Args:
            self (MusicgenModel): The instance of the MusicgenModel class.
            input_ids (mindspore.Tensor, optional): The input tensor IDs. Default: None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Default: None.
            encoder_hidden_states (mindspore.Tensor, optional): The hidden states of the encoder. Default: None.
            encoder_attention_mask (mindspore.Tensor, optional): The attention mask tensor for the encoder. Default: None.
            head_mask (mindspore.Tensor, optional): The head mask tensor. Default: None.
            cross_attn_head_mask (mindspore.Tensor, optional): The cross-attention head mask tensor. Default: None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values tensor. Default: None.
            inputs_embeds (mindspore.Tensor, optional): The embedded inputs tensor. Default: None.
            use_cache (bool, optional): Whether to use cache. Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.
            return_dict (bool, optional): Whether to return a dictionary. Default: None.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: The output of the forward method.

        Raises:
            None.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


class MusicgenForCausalLM(MusicgenPreTrainedModel):

    """
    The MusicgenForCausalLM class represents a model for causal language modeling in the context of music generation.
    It is designed to handle decoding tasks using a MusicgenDecoderConfig configuration. The class includes methods
    for setting and getting input/output embeddings, setting the decoder, forwarding the model for generation,
    preparing inputs for generation, building delay pattern masks, applying delay pattern masks, and generating
    sequences of token ids.

    Inherits from:
        MusicgenPreTrainedModel

    The MusicgenForCausalLM class provides a comprehensive set of methods for handling causal language modeling
    tasks in the domain of music generation.
    """
    def __init__(self, config: MusicgenDecoderConfig):
        """
        Initializes an instance of the MusicgenForCausalLM class.

        Args:
            self: The instance of the class.
            config (MusicgenDecoderConfig): The configuration object for the Musicgen decoder.
                It contains various parameters required to initialize the model and its components.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.model = MusicgenModel(config)

        self.num_codebooks = config.num_codebooks
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the MusicgenForCausalLM model.

        Args:
            self: An instance of the MusicgenForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings used by the model's decoder. The input embeddings are obtained from
        the embed_tokens attribute of the model's decoder. The embed_tokens attribute represents the token embeddings
        used to transform input tokens into dense vectors.

        Note that the input embeddings are not modified or processed in any way by this method. They are simply
        retrieved and returned as they are.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MusicgenForCausalLM model.

        Args:
            self (MusicgenForCausalLM): An instance of the MusicgenForCausalLM class.
            value (Any): The new input embeddings to be set for the model. It should be an object of the desired
                input embeddings type.

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings of the decoder in the MusicgenForCausalLM model to the provided value.
        The input embeddings are responsible for converting input tokens into continuous representations that can
        be processed by the model.
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MusicgenForCausalLM model.

        Args:
            self (MusicgenForCausalLM): The instance of the MusicgenForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the MusicgenForCausalLM model.

        Args:
            self (MusicgenForCausalLM): The instance of the MusicgenForCausalLM class.
            new_embeddings: The new embeddings to be set as the output embeddings. This should be a tensor.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the MusicgenForCausalLM model.

        Args:
            self (object): The MusicgenForCausalLM instance itself.
            decoder (object): The decoder object to be set for the model.
                It should be an instance of the decoder class compatible with the MusicgenForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        Retrieve the decoder model.

        Args:
            self: The instance of the MusicgenForCausalLM class.

        Returns:
            None: The method returns the decoder model associated with the instance.

        Raises:
            None
        """
        return self.model.decoder

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""

        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        lm_logits = ops.stack([head(hidden_states) for head in self.lm_heads], axis=1)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented for Musicgen.")

        # (bsz, num_codebooks, seq_len, vocab_size) -> (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        """
        This method prepares inputs for generation in the MusicgenForCausalLM class.

        Args:
            self: The instance of the class.
            input_ids (torch.Tensor): The input tensor containing token ids.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): The hidden states from the encoder. Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): The attention mask for the encoder. Defaults to None.
            head_mask (torch.Tensor, optional): The mask for specific attention heads. Defaults to None.
            cross_attn_head_mask (torch.Tensor, optional): The mask for cross-attention heads. Defaults to None.
            past_key_values (torch.Tensor, optional): The cached key values from previous computations. Defaults to None.
            use_cache (bool): Flag indicating whether to use cache. Defaults to True.
            delay_pattern_mask (torch.Tensor, optional): The mask for introducing delays in the input. Defaults to None.
            guidance_scale (int, optional): The scale factor for guidance. Defaults to None.

        Returns:
            dict: A dictionary containing the prepared input tensors and masks.

        Raises:
            ValueError: If guidance_scale is provided but is not a positive integer.
            IndexError: If past_key_values are provided but not in the expected format.
        """
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # apply the delay pattern mask
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        if guidance_scale is not None and guidance_scale > 1:
            # for classifier free guidance we need to replicate the decoder args across the batch dim (we'll split these
            # before sampling)
            input_ids = input_ids.repeat((2, 1))
            if attention_mask is not None:
                attention_mask = attention_mask.repeat((2, 1))

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def build_delay_pattern_mask(self, input_ids: mindspore.Tensor, pad_token_id: int, max_length: int = None):
        """Build a delayed pattern mask to the input_ids. Each codebook is offset by the previous codebook by
        one, giving a delayed pattern mask at the start of sequence and end of sequence. Take the example where there
        are 4 codebooks and a max sequence length of 8, we have the delayed pattern mask of shape `(codebooks,
        seq_len)`:

        - [P, -1, -1, -1, -1, P, P, P]
        - [P, P, -1, -1, -1, -1, P, P]
        - [P, P, P, -1, -1, -1, -1, P]
        - [P, P, P, P, -1, -1, -1, -1]

        where P is the special padding token id and -1 indicates that the token is valid for prediction. If we include
        a prompt (decoder input ids), the -1 positions indicate where new tokens should be predicted. Otherwise, the
        mask is set to the value in the prompt:

        - [P, a, b, -1, -1, P, P, P]
        - [P, P, c, d, -1, -1, P, P]
        - [P, P, P, e, f, -1, -1, P]
        - [P, P, P, P, g, h, -1, -1]

        where a-h indicate the input prompt (decoder input ids) that are offset by 1. Now, we only override the -1
        tokens in our prediction.
        """
        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        input_ids = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
        bsz, num_codebooks, seq_len = input_ids.shape

        max_length = max_length if max_length is not None else self.generation_config.max_length
        input_ids_shifted = (
            ops.ones((bsz, num_codebooks, max_length), dtype=mindspore.int64) * -1
        )

        channel_codebooks = num_codebooks // 2 if self.config.audio_channels == 2 else num_codebooks
        # we only apply the mask if we have a large enough seq len - otherwise we return as is
        if max_length < 2 * channel_codebooks - 1:
            return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

        # fill the shifted ids with the prompt entries, offset by the codebook idx
        for codebook in range(channel_codebooks):
            if self.config.audio_channels == 1:
                # mono channel - loop over the codebooks one-by-one
                input_ids_shifted[:, codebook, codebook : seq_len + codebook] = input_ids[:, codebook]
            else:
                # left/right channels are interleaved in the generated codebooks, so handle one then the other
                input_ids_shifted[:, 2 * codebook, codebook : seq_len + codebook] = input_ids[:, 2 * codebook]
                input_ids_shifted[:, 2 * codebook + 1, codebook : seq_len + codebook] = input_ids[:, 2 * codebook + 1]

        # forward a pattern mask that indicates the positions of padding tokens for each codebook
        # first fill the upper triangular part (the EOS padding)
        delay_pattern = ops.triu(
            ops.ones((channel_codebooks, max_length), dtype=mindspore.int32), diagonal=max_length - channel_codebooks + 1
        )
        # then fill the lower triangular part (the BOS padding)
        delay_pattern = delay_pattern + ops.tril(ops.ones((channel_codebooks, max_length), dtype=mindspore.int32))

        if self.config.audio_channels == 2:
            # for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
            delay_pattern = delay_pattern.repeat_interleave(2, dim=0)

        delay_pattern = delay_pattern.to(mindspore.bool_)
        mask = ~delay_pattern
        input_ids = mask * input_ids_shifted + ~mask * pad_token_id

        # find the first position to start generating - this is the first place we have the -1 token
        # and will always be in the first codebook (since it has no codebook offset)
        first_codebook_ids = input_ids[:, 0, :]
        start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
        if len(start_ids) > 0:
            first_start_id = min(start_ids)
        else:
            # we have no tokens that need to be filled - return entire matrix of input ids
            first_start_id = seq_len

        # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
        pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
        input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
        return input_ids, pattern_mask

    @staticmethod
    def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
        """Apply a delay pattern mask to the decoder input ids, only preserving predictions where
        the mask is set to -1, and otherwise setting to the value detailed in the mask."""
        seq_len = input_ids.shape[-1]
        decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
        input_ids = ops.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
        return input_ids

    def generate(
        self,
        inputs: Optional[mindspore.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`mindspore.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Returns:
            [`~utils.ModelOutput`] or `mindspore.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        input_ids, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = input_ids.shape[0] // self.num_codebooks

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        requires_attention_mask = "encoder_outputs" not in model_kwargs
        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # 5. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            logger.warning(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length.  recommend setting `max_new_tokens` to control the maximum length of the generation."
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://hf-mirror.com/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(
                f"Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 6. Prepare `input_ids` which will be used for auto-regressive generation
        # Build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to MusicGen)
        input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
            input_ids,
            pad_token_id=generation_config.decoder_start_token_id,
            max_length=generation_config.max_length,
        )

        if streamer is not None:
            streamer.put(input_ids)

        # stash the delay mask so that we don't have to recompute it in each forward pass
        model_kwargs["delay_pattern_mask"] = delay_pattern_mask

        # 7. determine generation mode
        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
        )

        # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None

        # 9. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        # 10. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            outputs = self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                **model_kwargs,
            )

            # 12. run sample
            outputs = self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # apply the pattern mask to the final ids
        output_ids = self.apply_delay_pattern_mask(output_ids, model_kwargs["delay_pattern_mask"])

        # revert the pattern delay mask by filtering the pad token id
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(
            batch_size, self.num_codebooks, -1
        )

        if generation_config.return_dict_in_generate:
            outputs.sequences = output_ids
            return outputs
        else:
            return output_ids


class MusicgenForConditionalGeneration(PreTrainedModel):

    """
    Class representing a Music Generation model for Conditional Generation.

    This class provides methods for generating sequences of token ids for models with a language modeling head.
    It supports both encoder-decoder and decoder-only models for conditional generation tasks. The generation process
    can be customized using various generation configurations, logits processors, stopping criteria, and other
    parameters.

    The class includes methods for preparing inputs for generation, tying weights between encoder and decoder, getting
    audio and text encoders, getting the encoder, decoder, input embeddings, and output embeddings. It also provides
    methods for setting and getting output embeddings, loading models from pretrained checkpoints, and instantiating
    models from pretrained sub-models.

    Additionally, it includes methods for forwarding the model, tying encoder-decoder weights, and resizing token
    embeddings. The class supports generating audio samples unconditionally and provides methods for preparing inputs
    for generation, generating audio samples using greedy search or sampling, and handling guidance scales.

    For more detailed usage examples and information on the class methods and functionalities, please refer to the
    class documentation and the Hugging Face Transformers documentation.
    """
    config_class = MusicgenConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Optional[MusicgenConfig] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[MusicgenForCausalLM] = None,
    ):
        '''
        Initializes a new instance of the MusicgenForConditionalGeneration class.

        Args:
            self: The object itself.
            config (Optional[MusicgenConfig]): The configuration for the Musicgen model. If not provided, it will be
                created from the sub-models' configurations.
            text_encoder (Optional[PreTrainedModel]): The text encoder model. If not provided, it will be initialized
                using the default text encoder.
            audio_encoder (Optional[PreTrainedModel]): The audio encoder model. If not provided, it will be initialized
                using the default audio encoder.
            decoder (Optional[MusicgenForCausalLM]): The Musicgen decoder model. If not provided, it will be initialized
                using the default decoder.

        Returns:
            None.

        Raises:
            ValueError: Raised if either a configuration has to be provided, or all three of text encoder, audio encoder,
                and Musicgen decoder are not provided.
            ValueError: Raised if the provided config is not of type MusicgenConfig.
            ValueError: Raised if cross_attention_hidden_size is specified in the Musicgen decoder's configuration and
                is not equal to the text encoder's hidden size.
            ValueError: Raised if the encoder has an LM Head, which is not allowed.
            ValueError: Raised if the selected decoder is not prepared for the encoder hidden states to be passed.
        '''
        if config is None and (text_encoder is None or audio_encoder is None or decoder is None):
            raise ValueError(
                "Either a configuration has to be provided, or all three of text encoder, audio encoder and MusicGen decoder."
            )
        if config is None:
            config = MusicgenConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.text_encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the MusicGen decoder's configuration, it has to be equal"
                    f" to the text encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for"
                    f" `config.decoder.cross_attention_hidden_size` and {config.text_encoder.hidden_size} for"
                    " `config.text_encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if text_encoder is None:
            from ..auto.modeling_auto import AutoModelForTextEncoding

            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)

        if audio_encoder is None:
            from ..auto.modeling_auto import AutoModel

            audio_encoder = AutoModel.from_config(config.audio_encoder)

        if decoder is None:
            decoder = MusicgenForCausalLM(config.decoder)

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder

        if self.text_encoder.config.to_dict() != self.config.text_encoder.to_dict():
            logger.warning(
                f"Config of the text_encoder: {self.text_encoder.__class__} is overwritten by shared text_encoder config:"
                f" {self.config.text_encoder}"
            )
        if self.audio_encoder.config.to_dict() != self.config.audio_encoder.to_dict():
            logger.warning(
                f"Config of the audio_encoder: {self.audio_encoder.__class__} is overwritten by shared audio_encoder config:"
                f" {self.config.audio_encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder

        # text encoder outputs might need to be projected to different dimension for decoder
        if (
            self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head"
            )

        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if "encoder_hidden_states" not in decoder_signature:
            raise ValueError(
                "The selected decoder is not prepared for the encoder hidden states to be passed. Please see the "
                "following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350"
            )

        # tie text encoder, decoder weights if config set accordingly
        self.tie_weights()

    def tie_weights(self):
        """
        Method to tie weights between the text encoder and decoder components in the MusicgenForConditionalGeneration model.

        Args:
            self: MusicgenForConditionalGeneration object. The instance of the class invoking the method.

        Returns:
            None.

        Raises:
            None.
        """
        # tie text encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie text encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.text_encoder, self.decoder._cells[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_audio_encoder(self):
        """
        This method, get_audio_encoder, is a member of the MusicgenForConditionalGeneration class.
        It retrieves the audio encoder used for generating music.

        Args:
            self: An instance of the MusicgenForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.audio_encoder

    def get_text_encoder(self):
        """
        Returns the text encoder used by the MusicgenForConditionalGeneration class.

        Args:
            self: An instance of the MusicgenForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the text encoder object associated with the MusicgenForConditionalGeneration instance.
        The text encoder is responsible for encoding and decoding text data used in the music generation process.
        It is used internally by other methods in the class to perform various operations on the text data.
        """
        return self.text_encoder

    def get_encoder(self):
        """
        This method returns the text encoder used for music generation.

        Args:
            self: The instance of the MusicgenForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        # get the text encoder to compute the encoder hidden-states for generation
        return self.get_text_encoder()

    def get_decoder(self):
        """
        Returns the decoder model for the MusicgenForConditionalGeneration class.

        Args:
            self: An instance of the MusicgenForConditionalGeneration class.

        Returns:
            decoder: The method returns the decoder model, which is an instance of a specific class,
                or None if the decoder model is not available.

        Raises:
            None.

        Note:
            This method is a getter method used to retrieve the decoder model that is associated with the current
            instance of the MusicgenForConditionalGeneration class. The decoder model is responsible for
            decoding encoded data into a readable format.

        Example:
            ```python
            >>> # Create an instance of the MusicgenForConditionalGeneration class
            >>> music_generator = MusicgenForConditionalGeneration()
            ...
            >>> # Retrieve the decoder model
            >>> decoder_model = music_generator.get_decoder()
            ...
            >>> if decoder_model is not None:
            >>>     # Use the decoder model to decode the encoded data
            >>>     decoded_data = decoder_model.decode(encoded_data)
            >>> else:
            >>>     print("Decoder model is not available.")
            ```
        """
        return self.decoder

    def get_input_embeddings(self):
        """
        This method returns the input embeddings from the text encoder.

        Args:
            self: The instance of the class 'MusicgenForConditionalGeneration'.

        Returns:
            None: This method returns None as it directly calls the 'get_input_embeddings' method from the 'text_encoder'.

        Raises:
            None.
        """
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the decoder for conditional generation in the MusicgenForConditionalGeneration class.

        Args:
            self:
                An instance of the MusicgenForConditionalGeneration class.

                - Type: MusicgenForConditionalGeneration
                - Purpose: Represents the current instance of the class.
                - Restrictions: Must be a valid instance of the MusicgenForConditionalGeneration class.

        Returns:
            embeddings:
                - Purpose: The output embeddings from the decoder.

        Raises:
            None.
        """
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the MusicgenForConditionalGeneration decoder.

        Args:
            self (MusicgenForConditionalGeneration): The instance of the MusicgenForConditionalGeneration class.
            new_embeddings (object): The new output embeddings to be set for the decoder.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""

        Example:
            ```python
            >>> from transformers import MusicgenForConditionalGeneration
            ...
            >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            ```"""
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for MusicgenForConditionalGeneration. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_sub_models_pretrained(
        cls,
        *model_args,
        text_encoder_pretrained_model_name_or_path: str = None,
        audio_encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        **kwargs,
    ) -> PreTrainedModel:
        r"""
        Instantiate a text encoder, an audio encoder, and a MusicGen decoder from one, two or three base classes of the
        library from pretrained model checkpoints.
        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            text_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text encoder. Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on hf-mirror.com.
                - A path to a *directory* containing model weights saved using
                [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on hf-mirror.com.
                - A path to a *directory* containing model weights saved using
                [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on hf-mirror.com.
                - A path to a *directory* containing model weights saved using
                [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration
                parameter.
                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration
                parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:
            ```python
            >>> from transformers import MusicgenForConditionalGeneration
            ...
            >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder
            >>> model = MusicgenForConditionalGeneration.from_sub_models_pretrained(
            ...     text_encoder_pretrained_model_name_or_path="google-t5/t5-base",
            ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
            ...     decoder_pretrained_model_name_or_path="facebook/musicgen-small",
            ... )
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./musicgen-ft")
            >>> # load fine-tuned model
            >>> model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-ft")
            ```"""
        kwargs_text_encoder = {
            argument[len("text_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_encoder_")
        }

        kwargs_audio_encoder = {
            argument[len("audio_encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("audio_encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove text encoder, audio encoder and decoder kwargs from kwargs
        for key in kwargs_text_encoder.keys():
            del kwargs["text_encoder_" + key]
        for key in kwargs_audio_encoder.keys():
            del kwargs["audio_encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        text_encoder = kwargs_text_encoder.pop("model", None)
        if text_encoder is None:
            if text_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `text_encoder_model` is not defined as an argument, a `text_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_text_encoder:
                encoder_config, kwargs_text_encoder = AutoConfig.from_pretrained(
                    text_encoder_pretrained_model_name_or_path, **kwargs_text_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {text_encoder_pretrained_model_name_or_path} as a text_encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_text_encoder["config"] = encoder_config

            text_encoder = AutoModel.from_pretrained(
                text_encoder_pretrained_model_name_or_path, *model_args, **kwargs_text_encoder
            )

        audio_encoder = kwargs_audio_encoder.pop("model", None)
        if audio_encoder is None:
            if audio_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `audio_encoder_model` is not defined as an argument, an `audio_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_audio_encoder:
                encoder_config, kwargs_audio_encoder = AutoConfig.from_pretrained(
                    audio_encoder_pretrained_model_name_or_path, **kwargs_audio_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {audio_encoder_pretrained_model_name_or_path} as an audio_encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_audio_encoder["config"] = encoder_config

            audio_encoder = AutoModel.from_pretrained(
                audio_encoder_pretrained_model_name_or_path, *model_args, **kwargs_audio_encoder
            )

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(
                    decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True
                )

                if isinstance(decoder_config, MusicgenConfig):
                    decoder_config = decoder_config.decoder

                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention"
                        f" layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if"
                        f" {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. "
                    f"In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, "
                    "make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` "
                    "passed to `.from_sub_models_pretrained(...)` are set to `True` or do not pass a "
                    "`decoder_config` to `.from_sub_models_pretrained(...)`"
                )

            decoder = MusicgenForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = MusicgenConfig.from_sub_models_config(
            text_encoder.config, audio_encoder.config, decoder.config, **kwargs
        )
        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        input_values: Optional[mindspore.Tensor] = None,
        padding_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[mindspore.Tensor]] = None,
        past_key_values: Tuple[Tuple[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""

        Returns:
            `Union[Tuple, Seq2SeqLMOutput]`

        Example:
            ```python
            >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
            >>> import torch
            ...
            >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            ...
            >>> inputs = processor(
            ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
            ...     padding=True,
            ...     return_tensors="pt",
            ... )
            ...
            >>> pad_token_id = model.generation_config.pad_token_id
            >>> decoder_input_ids = (
            ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
            ...     * pad_token_id
            ... )
            ...
            >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
            >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)
            torch.Size([8, 1, 2048])
            ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_text_encoder = {
            argument[len("text_encoder_")]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_encoder_")
        }

        kwargs_audio_encoder = {
            argument[len("audio_encoder_")]: value
            for argument, value in kwargs.items()
            if argument.startswith("audio_encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_text_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        # optionally project encoder_hidden_states
        if (
            self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            audio_encoder_outputs = self.audio_encoder(
                input_values=input_values,
                padding_mask=padding_mask,
                **kwargs_audio_encoder,
            )
            audio_codes = audio_encoder_outputs.audio_codes
            frames, bsz, codebooks, seq_len = audio_codes.shape
            if frames != 1:
                raise ValueError(
                    f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                    "disabled by setting `chunk_length=None` in the audio encoder."
                )

            if self.config.decoder.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
                # mono input through encodec that we convert to stereo
                audio_codes = audio_codes.repeat_interleave(2, dim=2)

            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        This method prepares the inputs for the generation process in the MusicgenForConditionalGeneration class.

        Args:
            self (MusicgenForConditionalGeneration): The instance of the MusicgenForConditionalGeneration class.
            decoder_input_ids (torch.Tensor): The input tensor for the decoder. It represents the tokenized input sequence.
            past_key_values (Optional[List[torch.Tensor]]): The list of past key values for the decoder. Default is None.
            attention_mask (Optional[torch.Tensor]): The attention mask tensor. Default is None.
            head_mask (Optional[torch.Tensor]): The head mask tensor. Default is None.
            decoder_attention_mask (Optional[torch.Tensor]): The decoder attention mask tensor. Default is None.
            decoder_head_mask (Optional[torch.Tensor]): The decoder head mask tensor. Default is None.
            cross_attn_head_mask (Optional[torch.Tensor]): The cross-attention head mask tensor. Default is None.
            use_cache (Optional[bool]): Whether to use cache. Default is None.
            encoder_outputs (Optional[torch.Tensor]): The encoder outputs tensor. Default is None.
            decoder_delay_pattern_mask (Optional[torch.Tensor]): The decoder delay pattern mask tensor. Default is None.
            guidance_scale (Optional[int]): The guidance scale value. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared input tensors for generation.
                The keys are as follows:

                - 'input_ids': None (no input ids are returned)
                - 'encoder_outputs': The encoder outputs tensor
                - 'past_key_values': The list of past key values for the decoder
                - 'decoder_input_ids': The prepared decoder input ids tensor
                - 'attention_mask': The attention mask tensor
                - 'decoder_attention_mask': The decoder attention mask tensor
                - 'head_mask': The head mask tensor
                - 'decoder_head_mask': The decoder head mask tensor
                - 'cross_attn_head_mask': The cross-attention head mask tensor
                - 'use_cache': The value indicating whether to use cache

        Raises:
            None.
        """
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )

        # apply the delay pattern mask
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)

        if guidance_scale is not None and guidance_scale > 1:
            # for classifier free guidance we need to replicate the decoder args across the batch dim (we'll split these
            # before sampling)
            decoder_input_ids = decoder_input_ids.repeat(2, 1)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat(2, 1)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, mindspore.Tensor],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
    ) -> Tuple[mindspore.Tensor, Dict[str, mindspore.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids_start = (
            ops.ones((batch_size * self.decoder.num_codebooks, 1), dtype=mindspore.int64)
            * decoder_start_token_id
        )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = ops.cat([decoder_input_ids_start, decoder_input_ids.astype(mindspore.int64)], axis=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = ops.cat(
                    (ops.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    axis=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _prepare_text_encoder_kwargs_for_generation(
        self,
        inputs_tensor: mindspore.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prepare text encoder keyword arguments for generation.

        Args:
            self (MusicgenForConditionalGeneration): The instance of the MusicgenForConditionalGeneration class.
            inputs_tensor (mindspore.Tensor): The input tensor for the text encoder.
            model_kwargs (Dict[str, Any]): The keyword arguments for the model.
            model_input_name (Optional[str]): The name of the model input. Defaults to None.
            guidance_scale (Optional[float]): The scale for guidance. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the modified keyword arguments for the text encoder model.

        Raises:
            ValueError: If the guidance scale is not a positive value.
            TypeError: If the inputs_tensor or model_kwargs are not of the expected types.
            RuntimeError: If there is an issue with the text encoder forwardion or invocation.
        """
        # 1. get text encoder
        encoder = self.get_text_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        # if hasattr(encoder, "_hf_hook"):
        #     encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state

        # for classifier free guidance we need to add a 'null' input to our encoder hidden states
        if guidance_scale is not None and guidance_scale > 1:
            last_hidden_state = ops.cat([last_hidden_state, ops.zeros_like(last_hidden_state)], axis=0)
            if "attention_mask" in model_kwargs:
                model_kwargs["attention_mask"] = ops.cat(
                    [model_kwargs["attention_mask"], ops.zeros_like(model_kwargs["attention_mask"])], axis=0
                )

        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=last_hidden_state)

        return model_kwargs

    def _prepare_audio_encoder_kwargs_for_generation(
        self, input_values, model_kwargs, model_input_name: Optional[str] = None
    ):
        """
        Generate a detailed docstring for a method named '_prepare_audio_encoder_kwargs_for_generation'
        in the class named 'MusicgenForConditionalGeneration'.

        This method prepares the keyword arguments for the audio encoder before generating output based on
        the input values and model configuration.

        Args:
            self (object): The instance of the class invoking this method.
            input_values (numpy.ndarray): The input audio values to be encoded.
                Should have shape (frames, channels, seq_len).
            model_kwargs (dict): The keyword arguments for the model configuration.
            model_input_name (str, optional): The name of the main input for the audio encoder.
                If not provided, it defaults to the main input name of the audio encoder.

        Returns:
            None: The method modifies the 'model_kwargs' dictionary in place to include the necessary arguments
                for the audio encoder.

        Raises:
            ValueError:
                - If the audio encoder does not accept wildcard arguments and there are arguments in 'model_kwargs'
                that are not in the encoder's signature.
                - If the input audio is not stereo (2-channels) when the decoder configuration requires it.
                - If the number of frames in the audio code outputs is not 1, indicating an issue with chunking settings.
        """
        # 1. get audio encoder
        encoder = self.get_audio_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        # if hasattr(encoder, "_hf_hook"):
        #     encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
        encoder_kwargs["return_dict"] = True

        if self.decoder.config.audio_channels == 1:
            encoder_kwargs[model_input_name] = input_values
            audio_encoder_outputs = encoder.encode(**encoder_kwargs)
            audio_codes = audio_encoder_outputs.audio_codes
            audio_scales = audio_encoder_outputs.audio_scales

            frames, bsz, codebooks, seq_len = audio_codes.shape

        else:
            if input_values.shape[1] != 2:
                raise ValueError(
                    f"Expected stereo audio (2-channels) but example has {input_values.shape[1]} channel."
                )

            encoder_kwargs[model_input_name] = input_values[:, :1, :]
            audio_encoder_outputs_left = encoder.encode(**encoder_kwargs)
            audio_codes_left = audio_encoder_outputs_left.audio_codes
            audio_scales_left = audio_encoder_outputs_left.audio_scales

            encoder_kwargs[model_input_name] = input_values[:, 1:, :]
            audio_encoder_outputs_right = encoder.encode(**encoder_kwargs)
            audio_codes_right = audio_encoder_outputs_right.audio_codes
            audio_scales_right = audio_encoder_outputs_right.audio_scales

            frames, bsz, codebooks, seq_len = audio_codes_left.shape
            # copy alternating left/right channel codes into stereo codebook
            audio_codes = audio_codes_left.new_ones((frames, bsz, 2 * codebooks, seq_len))

            audio_codes[:, :, ::2, :] = audio_codes_left
            audio_codes[:, :, 1::2, :] = audio_codes_right

            if audio_scales_left != [None] or audio_scales_right != [None]:
                audio_scales = ops.stack([audio_scales_left, audio_scales_right], axis=1)
            else:
                audio_scales = [None] * bsz

        if frames != 1:
            raise ValueError(
                f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                "disabled by setting `chunk_length=None` in the audio encoder."
            )

        decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

        model_kwargs["decoder_input_ids"] = decoder_input_ids
        model_kwargs["audio_scales"] = audio_scales
        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare the decoder input IDs from the given labels.

        Args:
            self (MusicgenForConditionalGeneration): An instance of the MusicgenForConditionalGeneration class.
            labels (mindspore.Tensor): The labels tensor containing the target sequence.

        Returns:
            None: This method modifies the input labels tensor in-place.

        Raises:
            None.

        This method prepares the decoder input IDs from the given labels tensor. It shifts the tokens in the labels tensor
        to the right by one position, replacing the first token with the pad token ID and adding the decoder start token ID
        at the beginning of the tensor. The modified labels tensor is used as the input for the decoder during generation.
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        """
        Resize the token embeddings for the MusicgenForConditionalGeneration model.

        Args:
            self (object): The instance of the MusicgenForConditionalGeneration class.

        Returns:
            None.

        Raises:
            NotImplementedError: This exception is raised when attempting to resize the embedding layers directly via
            the EncoderDecoderModel. It is recommended to use the respective methods of the wrapped objects
            (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...)).
        """
        raise NotImplementedError(
            "Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the"
            " respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or"
            " model.decoder.resize_token_embeddings(...))"
        )

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[mindspore.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, mindspore.Tensor]] = None,
    ) -> mindspore.Tensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs[0].shape[:-1]
            return ops.ones(shape, dtype=mindspore.int64) * -100

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, mindspore.Tensor):
                batch_size = value.shape[0]
                break
        return ops.ones((batch_size, 1), dtype=mindspore.int64) * bos_token_id

    def generate(
        self,
        inputs: Optional[mindspore.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        streamer: Optional["BaseStreamer"] = None,
        **kwargs,
    ):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`mindspore.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Returns:
            [`~utils.ModelOutput`] or `mindspore.Tensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())

        if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) == tuple:
            # wrap the unconditional outputs as a BaseModelOutput for compatibility with the rest of generate
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if "encoder_outputs" not in model_kwargs:
            # encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(
                inputs_tensor,
                model_kwargs,
                model_input_name,
                guidance_scale=generation_config.guidance_scale,
            )

        if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(
                model_kwargs["input_values"],
                model_kwargs,
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            logger.warning(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation."
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://hf-mirror.com/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(
                f"Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to MusicGen)
        input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=generation_config.decoder_start_token_id,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        # input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(input_ids)

        # 7. determine generation mode
        is_greedy_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is False
        )
        is_sample_gen_mode = (
            (generation_config.num_beams == 1)
            and (generation_config.num_beam_groups == 1)
            and generation_config.do_sample is True
        )

        # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None

        # 9. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        # 10. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )

        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing greedy search, "
                    f"but is {generation_config.num_return_sequences}."
                )

            # 11. run greedy search
            outputs = self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            outputs = self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                output_scores=generation_config.output_scores,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        else:
            raise ValueError(
                "Got incompatible mode for generation, should be one of greedy or sampling. "
                "Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`."
            )

        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs

        # apply the pattern mask to the final ids
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # revert the pattern delay mask by filtering the pad token id
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(
            batch_size, self.decoder.num_codebooks, -1
        )

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...]

        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size

        if self.decoder.config.audio_channels == 1:
            output_values = self.audio_encoder.decode(
                output_ids,
                audio_scales=audio_scales,
            ).audio_values
        else:
            codec_outputs_left = self.audio_encoder.decode(output_ids[:, :, ::2, :], audio_scales=audio_scales)
            output_values_left = codec_outputs_left.audio_values

            codec_outputs_right = self.audio_encoder.decode(output_ids[:, :, 1::2, :], audio_scales=audio_scales)
            output_values_right = codec_outputs_right.audio_values

            output_values = ops.cat([output_values_left, output_values_right], axis=1)

        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values

    def get_unconditional_inputs(self, num_samples=1):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.
            max_new_tokens (int, *optional*):
                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
                longer inference (since more audio tokens need to be generated per sample).

        Example:
            ```python
            >>> from transformers import MusicgenForConditionalGeneration
            ...
            >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            ...
            >>> # get the unconditional (or 'null') inputs for the model
            >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
            >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
            ```"""
        last_hidden_state = ops.zeros(
            (num_samples, 1, self.config.text_encoder.hidden_size), dtype=self.dtype
        )

        attention_mask = ops.zeros((num_samples, 1), dtype=mindspore.int64)

        return MusicgenUnconditionalInput(
            encoder_outputs=(last_hidden_state,),
            attention_mask=attention_mask,
            guidance_scale=1.0,
        )

__all__ =  [
    "MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MusicgenForConditionalGeneration",
    "MusicgenForCausalLM",
    "MusicgenModel",
    "MusicgenPreTrainedModel",
]
