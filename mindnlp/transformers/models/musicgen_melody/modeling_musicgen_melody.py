# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore Musicgen Melody model."""
import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops, get_default_dtype
from mindnlp.core.nn import functional as F
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel, AutoModelForTextEncoding
from .configuration_musicgen_melody import MusicgenMelodyConfig, MusicgenMelodyDecoderConfig


if TYPE_CHECKING:
    from ...generation.streamers import BaseStreamer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MusicgenMelodyConfig"
_CHECKPOINT_FOR_DOC = "facebook/musicgen-melody"

MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/musicgen-melody",
    # See all Musicgen Melody models at https://huggingface.co/models?filter=musicgen_melody
]


@dataclass
class MusicgenMelodyOutputWithPast(ModelOutput):
    """
    Base class for Musicgen Melody autoregressive outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
            or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of conditional hidden-states representing the concatenation of the projeted text encoder output
            and the projeted audio encoder output. Used as a conditional signal.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    encoder_hidden_states: Optional[mindspore.Tensor] = None


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


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenSinusoidalPositionalEmbedding with Musicgen->MusicgenMelody
class MusicgenMelodySinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int):
        """
        Initializes an instance of the MusicgenMelodySinusoidalPositionalEmbedding class.
        
        Args:
            self: The instance of the class.
            num_positions (int): The number of positions in the embedding.
                This parameter specifies the total number of positions in the embedding.
                It should be a positive integer.
            embedding_dim (int): The dimension of the embedding.
                This parameter specifies the dimensionality of the embedding vector.
                It should be a positive integer.
        
        Returns:
            None.

        Raises:
            None.

        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.make_weights(num_positions, embedding_dim)

    def make_weights(self, num_embeddings: int, embedding_dim: int):
        """
        make_weights method in the MusicgenMelodySinusoidalPositionalEmbedding class.

        This method initializes the weights for the embeddings based on the specified number of embeddings and
        embedding dimension.

        Args:
            self: The instance of the class.
            num_embeddings (int): The number of unique embeddings to be used.
            embedding_dim (int): The dimension of the embedding.

        Returns:
            None.

        Raises:
            None
        """
        emb_weights = self.get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            # in forward put the weights on the correct dtype of the param
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
        emb = ops.cat([ops.cos(emb), ops.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = ops.cat([emb, ops.zeros(num_embeddings, 1)], dim=1)
        return emb.to(get_default_dtype())

    def forward(self, inputs_embeds: mindspore.Tensor, past_key_values_length: int = 0):
        """
        Constructs the sinusoidal positional embedding for the MusicgenMelodySinusoidalPositionalEmbedding model.

        Args:
            self (MusicgenMelodySinusoidalPositionalEmbedding): An instance of the
                MusicgenMelodySinusoidalPositionalEmbedding class.
            inputs_embeds (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size)
                containing the embedded inputs.
            past_key_values_length (int, optional): The length of the past key values. Defaults to 0.

        Returns:
            None

        Raises:
            None
        """
        bsz, seq_len, _ = inputs_embeds.shape
        # Create the position ids from the input token ids.
        position_ids = (ops.arange(seq_len) + past_key_values_length)
        # expand embeddings if needed
        if seq_len > self.weights.shape[0]:
            self.make_weights(seq_len + self.offset, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1))


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->MusicgenMelody
class MusicgenMelodyAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[MusicgenMelodyConfig] = None,
    ):
        """
        Initialize the MusicgenMelodyAttention class.

        Args:
            self: The object itself.
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Whether the attention layer is used as part of a decoder. Defaults to False.
            bias (bool, optional): Whether to include bias in the linear transformation. Defaults to True.
            is_causal (bool, optional): Whether the attention is causal, i.e., only attends to previous positions.
                Defaults to False.
            config (Optional[MusicgenMelodyConfig], optional): The configuration for the attention layer.
                Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
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
        Reshapes the input tensor to match the expected shape for the attention mechanism in the MusicgenMelodyAttention
        class.

        Args:
            self: An instance of the MusicgenMelodyAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped. It should have a shape of
                (batch_size * seq_len * num_heads * head_dim).
            seq_len (int): The length of the sequence in the input tensor.
            bsz (int): The batch size of the input tensor.

        Returns:
            None.

        Raises:
            None.

        This method reshapes the input tensor by rearranging its dimensions. It first reshapes the tensor to have a
        shape of (batch_size, seq_len, num_heads, head_dim) using the view function. Then, it swaps the second and third
        dimensions using the swapaxes function to match the expected shape for the attention mechanism in
        MusicgenMelodyAttention.
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
            key_states = ops.cat([past_key_value[0], key_states], dim=2)
            value_states = ops.cat([past_key_value[1], value_states], dim=2)
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

        attn_weights = ops.softmax(attn_weights, dim=-1)

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


class MusicgenMelodyDecoderLayer(nn.Module):

    """
    This class represents a layer in the Musicgen Melody Decoder model. It is a subclass of nn.Module and is responsible
    for performing the decoding operations on the input.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
        self_attn (MusicgenMelodyAttention): The self-attention layer used for capturing the dependencies between
            different elements of the input.
        dropout (float): The dropout probability applied to the output of the self-attention layer.
        activation_fn (function): The activation function used in the feed-forward neural network layers.
        activation_dropout (float): The dropout probability applied to the output of the activation function.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization applied to the output of the self-attention layer.
        fc1 (nn.Linear): The first fully connected layer of the feed-forward neural network.
        fc2 (nn.Linear): The second fully connected layer of the feed-forward neural network.
        final_layer_norm (nn.LayerNorm): The layer normalization applied to the final output of the layer.

    Methods:
        forward:
            Performs the decoding operations on the input hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input to the layer of shape `(batch, seq_len, embed_dim)`.
            - attention_mask (mindspore.Tensor): The attention mask of size `(batch, 1, tgt_len, src_len)`
            where padding elements are indicated by very large negative values. Defaults to None.
            - layer_head_mask (mindspore.Tensor): The mask for attention heads in a given layer of size `(attention_heads,)`.
            Defaults to None.
            - past_key_value (Tuple[mindspore.Tensor]): The cached past key and value projection states. Defaults to None.
            - output_attentions (bool): Whether or not to return the attentions tensors of all attention layers.
            Defaults to False.
            - use_cache (bool): Whether or not to cache the key and value projection states for future use.
            Defaults to True.

            Returns:

            - outputs (Tuple[mindspore.Tensor]): The outputs of the layer, which includes the hidden states and
            optionally the self-attention weights and present key value.
    """
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        """
        Initializes an instance of the MusicgenMelodyDecoderLayer class.

        Args:
            self: The instance of the class.
            config (MusicgenMelodyDecoderConfig):
                The configuration object that contains the settings for the decoder layer.

                - config.hidden_size (int): The embedding dimension.
                - config.num_attention_heads (int): The number of attention heads.
                - config.attention_dropout (float): The dropout rate for attention layers.
                - config.dropout (float): The dropout rate for the layer.
                - config.activation_function (str): The name of the activation function.
                - config.activation_dropout (float): The dropout rate for activation layers.
                - config.ffn_dim (int): The dimension of the feed-forward network.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = MusicgenMelodyAttention(
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

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size `(attention_heads,)`.
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
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenPreTrainedModel with Musicgen->MusicgenMelody
class MusicgenMelodyPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MusicgenMelodyDecoderConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MusicgenMelodyDecoderLayer", "MusicgenMelodyAttention"]

    def _init_weights(self, cell):
        """
        Initializes the weights of a given cell.

        Args:
            self (MusicgenMelodyPreTrainedModel): An instance of the MusicgenMelodyPreTrainedModel class.
            cell: The cell whose weights need to be initialized.

        Returns:
            None.

        Raises:
            None.
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

# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenDecoder with MUSICGEN->MUSICGEN_MELODY,Musicgen->MusicgenMelody
class MusicgenMelodyDecoder(MusicgenMelodyPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MusicgenMelodyDecoderLayer`]
    """
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        """
        Initializes the MusicgenMelodyDecoder class.

        Args:
            self: The instance of the class.
            config (MusicgenMelodyDecoderConfig): An instance of the MusicgenMelodyDecoderConfig class containing
                the configuration parameters for the decoder.

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

        self.embed_positions = MusicgenMelodySinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size,
        )

        self.layers = nn.ModuleList([MusicgenMelodyDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the MusicgenMelodyDecoder class.

        Args:
            self (MusicgenMelodyDecoder): An instance of the MusicgenMelodyDecoder class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the MusicgenMelodyDecoder class.

        Args:
            self (object): Instance of the MusicgenMelodyDecoder class.
            value (object): New input embeddings value to be set for the decoder.

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
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the MusicgenMelodyDecoder.

        Args:
            self (MusicgenMelodyDecoder): The instance of the MusicgenMelodyDecoder class.
            input_ids (mindspore.Tensor, optional): The input tensor containing the encoded input sequence.
                Default is None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor for the input sequence.
                Default is None.
            encoder_hidden_states (mindspore.Tensor, optional): The hidden states tensor from the encoder.
                Default is None.
            encoder_attention_mask (mindspore.Tensor, optional): The attention mask tensor for the encoder hidden states.
                Default is None.
            head_mask (mindspore.Tensor, optional): The head mask tensor for the decoder layers. Default is None.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The past key values tensor. Default is None.
            inputs_embeds (mindspore.Tensor, optional): The input tensor containing the embedded inputs. Default is None.
            use_cache (bool, optional): Whether to use caching. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                The output of the MusicgenMelodyDecoder.

                It can be either a tuple containing the hidden states, next cache, all hidden states, and all attentions,
                or an instance of the BaseModelOutputWithPast class.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If the head_mask shape does not match the number of layers.

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

        if encoder_hidden_states is not None:
            # take care of attention masks
            if encoder_attention_mask is not None and attention_mask is None:
                attention_mask = ops.ones(inputs_embeds.shape[:2])

            if attention_mask is not None:
                if encoder_attention_mask is None:
                    encoder_attention_mask = ops.ones(encoder_hidden_states.shape[:2])
                attention_mask = ops.cat([encoder_attention_mask.astype(attention_mask.dtype), attention_mask], dim=1)

            # fuse encoder_hidden_states and inputs_embeds
            inputs_embeds = ops.cat([encoder_hidden_states, inputs_embeds], dim=1)

        input_shape = inputs_embeds.shape[:-1]

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # embed positions
        positions = self.embed_positions(inputs_embeds, past_key_values_length)

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
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The `head_mask` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.forward,
                    hidden_states,
                    attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenModel with MUSICGEN->MUSICGEN_MELODY,Musicgen->MusicgenMelody
class MusicgenMelodyModel(MusicgenMelodyPreTrainedModel):

    """
    This class represents a music generation melody model that is used for decoding melodies.
    It inherits from the MusicgenMelodyPreTrainedModel class.

    The MusicgenMelodyModel class contains methods for initializing the model, getting and setting input embeddings,
    getting the decoder, and forwarding the model for decoding melodies.

    Methods:
        __init__: Initializes the MusicgenMelodyModel instance with the given configuration.
        get_input_embeddings: Retrieves the input embeddings used by the decoder.
        set_input_embeddings: Sets the input embeddings used by the decoder.
        get_decoder: Retrieves the decoder.
        forward: Constructs the model for decoding melodies using the provided input arguments.
            Returns the decoder outputs as a tuple or BaseModelOutputWithPast if return_dict is True.
    """
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        """
        Initializes a MusicgenMelodyModel instance.

        Args:
            self: The instance of the MusicgenMelodyModel class.
            config (MusicgenMelodyDecoderConfig): An instance of MusicgenMelodyDecoderConfig containing
                configuration parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MusicgenMelodyDecoderConfig.
        """
        super().__init__(config)
        self.decoder = MusicgenMelodyDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the decoder of the MusicgenMelodyModel.

        Args:
            self: MusicgenMelodyModel object. Represents the instance of the MusicgenMelodyModel class.

        Returns:
            embeddings: This method returns the input embeddings from the decoder of the MusicgenMelodyModel.

        Raises:
            None.
        """
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MusicgenMelodyModel.

        Args:
            self (MusicgenMelodyModel): The instance of the MusicgenMelodyModel class.
            value: The input embeddings to be set for the model.
                This should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings of the decoder in the MusicgenMelodyModel. The input embeddings are used
        to represent the input tokens in the model. By setting the input embeddings, you can customize the way the
        model represents the tokens.

        Note that the input embeddings should be a tensor of shape (vocab_size, embedding_dim), where vocab_size is the
        number of unique tokens in the vocabulary and embedding_dim is the dimensionality of the embedding space.
        The embedding_dim should match the hidden size of the model.

        Example:
            ```python
            >>> model = MusicgenMelodyModel()
            >>> embedding_tensor = torch.randn(vocab_size, embedding_dim)
            >>> model.set_input_embeddings(embedding_tensor)
            ```
        """
        self.decoder.embed_tokens = value

    def get_decoder(self):
        """
        Method to retrieve the decoder object associated with the MusicgenMelodyModel instance.

        Args:
            self (MusicgenMelodyModel): The instance of MusicgenMelodyModel class.
                This parameter is required to access the decoder object.

        Returns:
            None: This method returns the decoder object associated with the instance.
                The decoder object is used to decode data or perform specific operations related to decoding.

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
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the MusicgenMelodyModel.

        Args:
            self (MusicgenMelodyModel): The instance of the MusicgenMelodyModel class.
            input_ids (mindspore.Tensor, optional): The input tensor containing the indices of input sequence
                tokens in the vocabulary.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor indicating which tokens
                should be attended to.
            encoder_hidden_states (mindspore.Tensor, optional): The tensor containing the hidden states of the encoder.
            encoder_attention_mask (mindspore.Tensor, optional): The attention mask tensor for the encoder.
            head_mask (mindspore.Tensor, optional): The tensor indicating which heads should be masked.
            past_key_values (Tuple[Tuple[mindspore.Tensor]], optional): The tensor containing the cached key-value
                states of the past.
            inputs_embeds (mindspore.Tensor, optional): The tensor containing the embeddings of the input sequence tokens.
            use_cache (bool, optional): Whether to use cache for the model.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return as a dictionary.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                The output of the MusicgenMelodyModel.

                - If 'return_dict' is False, returns a tuple containing decoder outputs.
                - If 'return_dict' is True, returns an instance of BaseModelOutputWithPast
                which contains the last hidden state, past key values, hidden states, and attentions.

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
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


# Copied from transformers.models.musicgen.modeling_musicgen.MusicgenForCausalLM with MUSICGEN->MUSICGEN_MELODY,Musicgen->MusicgenMelody,MusicGen->Musicgen Melody
class MusicgenMelodyForCausalLM(MusicgenMelodyPreTrainedModel):

    """
    The `MusicgenMelodyForCausalLM` class represents a model for generating melodies using a causal language modeling
    head. This class inherits from the `MusicgenMelodyPreTrainedModel`.

    This class includes methods for initializing the model, setting input and output embeddings, forwarding the model,
    preparing inputs for generation, building a delay pattern mask, applying a delay pattern mask, and generating
    sequences of token ids.

    The `MusicgenMelodyForCausalLM` class provides detailed control over the generation process, including the ability
    to customize logits processors and stopping criteria. It also supports streaming generated sequences.

    For more information on the parameters and return types of the methods, please refer to the method docstrings or
    the official documentation.
    """
    def __init__(self, config: MusicgenMelodyDecoderConfig):
        """
        Initializes a MusicgenMelodyForCausalLM object.

        Args:
            self (MusicgenMelodyForCausalLM): An instance of the MusicgenMelodyForCausalLM class.
            config (MusicgenMelodyDecoderConfig): The configuration object containing the necessary parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.model = MusicgenMelodyModel(config)

        self.num_codebooks = config.num_codebooks
        self.lm_heads = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method is responsible for retrieving the input embeddings from the decoder model.

        Args:
            self:
                MusicgenMelodyForCausalLM object

                - Type: object
                - Purpose: Represents the instance of the MusicgenMelodyForCausalLM class.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the MusicgenMelodyForCausalLM class.

        Args:
            self (object): The instance of the MusicgenMelodyForCausalLM class.
            value (object): The input embeddings to be set for the model's decoder. It can be of any valid type.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the MusicgenMelodyForCausalLM model.

        Args:
            self (MusicgenMelodyForCausalLM): An instance of the MusicgenMelodyForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the MusicgenMelodyForCausalLM model.

        Args:
            self (object): The instance of the MusicgenMelodyForCausalLM class.
            new_embeddings (object): The new embeddings to be set for the output.
                It could be a tensor or any compatible object.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the MusicgenMelodyForCausalLM class.

        Args:
            self (MusicgenMelodyForCausalLM): An instance of the MusicgenMelodyForCausalLM class.
            decoder: The decoder to be set for the model, which should be an object of the appropriate decoder class.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        Returns the decoder model used in the MusicgenMelodyForCausalLM class.

        Args:
            self: An instance of the MusicgenMelodyForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.decoder

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, MusicgenMelodyOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:
            Union[Tuple, MusicgenMelodyOutputWithPast]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        lm_logits = ops.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented for MusicgenMelody.")

        # (bsz, num_codebooks, seq_len, vocab_size) -> (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MusicgenMelodyOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Ignore copy
    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        This method prepares the input data for the generation process in the `MusicgenMelodyForCausalLM` class.

        Args:
            self (MusicgenMelodyForCausalLM): The instance of the `MusicgenMelodyForCausalLM` class.
            input_ids (Tensor): The input tensor representing the tokenized input sequence.
            attention_mask (Tensor, optional): The attention mask tensor indicating which tokens should be attended to.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder.
            encoder_attention_mask (Tensor, optional): The attention mask tensor for the encoder.
            head_mask (Tensor, optional): The mask tensor for masking specific heads in the attention mechanism.
            past_key_values (Tuple, optional): The past key-value pairs of the model.
            use_cache (bool, optional): Whether to use the cache for faster generation.
            delay_pattern_mask (Tensor, optional): The delay pattern mask tensor indicating the pattern of delays in the input sequence.
            guidance_scale (int, optional): The scale factor for guidance.

        Returns:
            dict: A dictionary containing the prepared input data for generation.
                The dictionary has the following keys:

                - 'input_ids' (Tensor): The modified input tensor.
                - 'attention_mask' (Tensor, optional): The modified attention mask tensor.
                - 'encoder_hidden_states' (Tensor, optional): The modified encoder hidden states tensor.
                - 'encoder_attention_mask' (Tensor, optional): The modified encoder attention mask tensor.
                - 'head_mask' (Tensor, optional): The modified head mask tensor.
                - 'past_key_values' (Tuple, optional): The modified past key-value pairs.
                - 'use_cache' (bool): The value indicating whether to use the cache.

        Raises:
            None.
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
            input_ids = input_ids.tile((2, 1))
            if attention_mask is not None:
                attention_mask = attention_mask.tile((2, 1))

            if encoder_hidden_states is not None:
                encoder_hidden_states = ops.cat(
                    [encoder_hidden_states, ops.zeros_like(encoder_hidden_states)], dim=0
                )

            if encoder_attention_mask is not None:
                encoder_attention_mask = ops.cat(
                    encoder_attention_mask, ops.zeros_like(encoder_attention_mask), dim=0
                )

        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

            # we only want to use conditional signal in the 1st generation step but keeping the attention mask
            encoder_hidden_states = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "head_mask": head_mask,
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
            ops.ones((channel_codebooks, max_length)), diagonal=max_length - channel_codebooks + 1
        )
        # then fill the lower triangular part (the BOS padding)
        delay_pattern = delay_pattern + ops.tril(ops.ones((channel_codebooks, max_length)))

        if self.config.audio_channels == 2:
            # for left/right channel we need to duplicate every row of the pattern mask in an interleaved fashion
            delay_pattern = ops.repeat_interleave(delay_pattern, 2, dim=0)

        delay_pattern = delay_pattern.astype(mindspore.bool_)

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
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`:

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

        # Ignore copy
        if model_kwargs.get("attention_mask", None) is None:
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
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
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
            streamer.put(input_ids.cpu())

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


class MusicgenMelodyForConditionalGeneration(PreTrainedModel):

    """
    This class represents a model for generating sequences of token ids for music generation tasks.
    It is specifically designed for conditional generation of melodies. The model inherits from PreTrainedModel
    and includes methods for initializing the model, tying weights, getting various components of the model such as
    the text encoder, encoder, and decoder, as well as methods for preparing inputs for generation,
    forwarding sequences, and generating outputs based on given inputs and generation configurations.

    The class includes methods for handling input initialization, model configuration, token embeddings,
    and generation processes. It also provides functionalities for customizing logits processing, stopping
    criteria, and stream processing during generation. Additionally, the class offers methods for updating model
    keyword arguments for generation, handling past key values, states, token type ids, and decoder attention masks.

    The model is equipped with functionalities for greedy search, sampling, and audio decoding to generate sequences
    that adhere to specified constraints and configurations. It allows for fine-tuning and customization of generation
    parameters to control the length, style, and quality of the generated music sequences.

    For detailed information on how to use the model for conditional generation tasks, including examples, model
    instantiation, and generation strategies, refer to the official documentation and guidelines provided in the
    class's code.
    """
    config_class = MusicgenMelodyConfig
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: MusicgenMelodyConfig = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[MusicgenMelodyForCausalLM] = None,
    ):
        """
        Initializes a new instance of the MusicgenMelodyForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (MusicgenMelodyConfig, optional): The configuration for the model. Defaults to None.
            text_encoder (PreTrainedModel, optional): The pre-trained model for text encoding. Defaults to None.
            audio_encoder (PreTrainedModel, optional): The pre-trained model for audio encoding. Defaults to None.
            decoder (MusicgenMelodyForCausalLM, optional): The pre-trained model for music generation. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: Raised when either a configuration has to be provided or all three of text encoder,
                audio encoder, and Musicgen Melody decoder are missing.
            ValueError: Raised when the provided config parameter is not of type MusicgenMelodyConfig.
            ValueError: Raised when the encoder has a LM Head, which is not allowed.

        Note:
            This method initializes the model by setting the configuration and initializing the text_encoder,
            audio_encoder, and decoder. It also performs necessary checks and assignments based on the provided
            or default values.
        """
        if config is None and None in (text_encoder, audio_encoder, decoder):
            raise ValueError(
                "Either a configuration has to be provided, or all three of text encoder, audio encoder and Musicgen Melody decoder."
            )
        if config is None:
            config = MusicgenMelodyConfig.from_sub_models_config(
                text_encoder.config, audio_encoder.config, decoder.config
            )
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # initialize with config
        super().__init__(config)

        if text_encoder is None:
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)

        if audio_encoder is None:
            audio_encoder = AutoModel.from_config(config.audio_encoder)

        if decoder is None:
            decoder = MusicgenMelodyForCausalLM(config.decoder)

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder

        # text encoder outputs might need to be projected to different dimension for decoder
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)

        # audio encoder outputs after chroma extraction might need to be projected to different dimension for decoder
        if self.config.num_chroma != self.decoder.config.hidden_size:
            self.audio_enc_to_dec_proj = nn.Linear(self.config.num_chroma, self.decoder.config.hidden_size)

        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head"
            )

        # Initialize projection layers weights and tie text encoder and decoder weights if set accordingly
        self.post_init()

    def _init_weights(self, cell):
        """
        Initializes the weights of a given cell for the MusicgenMelodyForConditionalGeneration model.

        Args:
            self (MusicgenMelodyForConditionalGeneration):
                The instance of the MusicgenMelodyForConditionalGeneration class.
            cell (nn.Module): The cell for which the weights are to be initialized.

        Returns:
            None

        Raises:
            None

        """
        # MusicgenMelodyForConditionalGeneration is made of PreTrainedModels that have already been initialized
        # Projection layers still need to be initialized.
        std = self.decoder.config.initializer_factor
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(std),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def tie_weights(self):
        # tie text encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie text encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            tied_weights = self._tie_encoder_decoder_weights(
                self.text_encoder,
                self.decoder._modules[decoder_base_model_prefix],
                self.decoder.base_model_prefix,
                "text_encoder",
            )
            # Setting a dynamic variable instead of `_tied_weights_keys` because it's a class
            # attributed not an instance member, therefore modifying it will modify the entire class
            # Leading to issues on subsequent calls by different tests or subsequent calls.
            self._dynamic_tied_weights_keys = tied_weights

    def get_text_encoder(self):
        """
        This method returns the text encoder used for encoding text data.

        Args:
            self: The instance of the MusicgenMelodyForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.text_encoder

    def get_encoder(self):
        """
        Method to get the text encoder for MusicgenMelodyForConditionalGeneration.

        Args:
            self (object): The instance of the MusicgenMelodyForConditionalGeneration class.
                This parameter is required to access the methods and attributes of the class.

        Returns:
            None.

        Raises:
            None.
        """
        # get the text encoder to compute the conditionning hidden-states for generation
        return self.get_text_encoder()

    def get_decoder(self):
        """
        Returns the decoder used for generating music melody for conditional generation.

        Args:
            self (MusicgenMelodyForConditionalGeneration): The instance of the class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder

    def get_input_embeddings(self):
        """
        Method Name: get_input_embeddings

        Description:
            This method is used to retrieve the input embeddings from the text encoder in the
            MusicgenMelodyForConditionalGeneration class.

        Args:
            self: An instance of the MusicgenMelodyForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        """

        Description:
        Returns the output embeddings of the decoder for the conditional generation of music melodies.

        Args:
            self (MusicgenMelodyForConditionalGeneration): The instance of the MusicgenMelodyForConditionalGeneration class.
                This parameter is required to access the decoder's output embeddings.
                Expected to be an instance of the MusicgenMelodyForConditionalGeneration class.

        Returns:
            None: This method does not return any value explicitly.
                The output embeddings of the decoder for conditional generation can be accessed through the
                returned object.

        Raises:
            None.
        """
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the MusicgenMelodyForConditionalGeneration model.

        Args:
            self (MusicgenMelodyForConditionalGeneration): An instance of the
                MusicgenMelodyForConditionalGeneration class.
            new_embeddings (torch.nn.Embedding): The new embeddings to set for the decoder.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_sub_models_pretrained( # pylint: disable=keyword-arg-before-vararg
        cls,
        text_encoder_pretrained_model_name_or_path: str = None,
        audio_encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        *model_args,
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

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing model weights saved using
                [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing model weights saved using
                [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
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
            >>> from transformers import MusicgenMelodyForConditionalGeneration
            ...
            >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder
            >>> model = MusicgenMelodyForConditionalGeneration.from_sub_models_pretrained(
            ...     text_encoder_pretrained_model_name_or_path="google-t5/t5-base",
            ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
            ...     decoder_pretrained_model_name_or_path="facebook/musicgen-melody",
            ... )
            >>> # saving model after fine-tuning
            >>> model.save_pretrained("./musicgen-ft")
            >>> # load fine-tuned model
            >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("./musicgen-ft")
            ```
        """
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

                if isinstance(decoder_config, MusicgenMelodyConfig):
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

            decoder = MusicgenMelodyForCausalLM.from_pretrained(
                decoder_pretrained_model_name_or_path, **kwargs_decoder
            )

        # instantiate config with corresponding kwargs
        config = MusicgenMelodyConfig.from_sub_models_config(
            text_encoder.config, audio_encoder.config, decoder.config, **kwargs
        )
        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        input_features: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Tuple[Tuple[mindspore.Tensor]] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MusicgenMelodyOutputWithPast]:
        r"""

        Returns:
            Union[Tuple, MusicgenMelodyOutputWithPast]

        Example:
            ```python
            >>> from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
            >>> import torch
            ...
            >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
            >>> model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
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
            >>> logits.shape  # (bsz * num_codebooks, encoder_len + tgt_len, vocab_size)
            torch.Size([8, 249, 2048])
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_text_encoder = {
            argument[len("text_encoder_")]: value
            for argument, value in kwargs.items()
            if argument.startswith("text_encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_hidden_states is None:
            if inputs_embeds is not None or input_ids is not None:
                encoder_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs_text_encoder,
                )

                encoder_hidden_states = encoder_outputs[0]

                # optionally project encoder_hidden_states
                if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
                    encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            if attention_mask is not None and encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]

            # set a default audio conditional hidden states if text is not None
            if encoder_hidden_states is not None and input_features is None:
                input_features = ops.zeros(
                    (encoder_hidden_states.shape[0], 1, self.config.num_chroma),
                    dtype=self.dtype,
                )
                input_features[:, :, 0] = 1

            if input_features is not None:
                audio_hidden_states = input_features

                # optionally project audio_hidden_states ->
                # (batch_size, seq_len, num_chroma) -> (batch_size, seq_len, hidden_size)
                if self.config.num_chroma != self.decoder.config.hidden_size:
                    audio_hidden_states = self.audio_enc_to_dec_proj(audio_hidden_states)

                # pad or truncate to config.chroma_length
                if audio_hidden_states.shape[1] < self.config.chroma_length:
                    n_repeat = int(math.ceil(self.config.chroma_length / audio_hidden_states.shape[1]))
                    audio_hidden_states = audio_hidden_states.tile((1, n_repeat, 1))
                else:
                    logger.warning(
                        f"The conditional audio signal is of length {audio_hidden_states.shape[1]}, which exceeds"
                        f"the maximum chroma duration of {self.config.chroma_length}."
                        f"The audio will be truncated to {self.config.chroma_length} frames."
                    )
                audio_hidden_states = audio_hidden_states[:, : self.config.chroma_length]

                if encoder_hidden_states is not None:
                    encoder_hidden_states = ops.cat([audio_hidden_states, encoder_hidden_states], dim=1)
                else:
                    encoder_hidden_states = audio_hidden_states

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
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
                return (loss,) + decoder_outputs + (encoder_hidden_states,)
            else:
                return decoder_outputs + (encoder_hidden_states,)

        return MusicgenMelodyOutputWithPast(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            encoder_hidden_states=encoder_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        encoder_hidden_states=None,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        use_cache=None,
        decoder_delay_pattern_mask=None,
        guidance_scale=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        This method prepares input data for generation in the MusicgenMelodyForConditionalGeneration class.

        Args:
            self: The instance of the class.
            decoder_input_ids (torch.Tensor): The input tensor for the decoder. It contains tokenized input
                sequence for the decoder.
            encoder_hidden_states (torch.Tensor, optional): The hidden states of the encoder. Defaults to None.
            past_key_values (tuple, optional): Tuple containing past key values. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask for the input. Defaults to None.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder input. Defaults to None.
            decoder_head_mask (torch.Tensor, optional): The head mask for the decoder. Defaults to None.
            use_cache (bool, optional): Indicates whether to use cache for the input. Defaults to None.
            decoder_delay_pattern_mask (torch.Tensor, optional): The delay pattern mask for the decoder input.
                Defaults to None.
            guidance_scale (float, optional): The scale for guidance. Defaults to None.

        Returns:
            dict: A dictionary containing prepared input data including input_ids, encoder_hidden_states,
                past_key_values, decoder_input_ids, attention_mask, decoder_attention_mask, decoder_head_mask, and
                use_cache.

        Raises:
            None:
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
            decoder_input_ids = decoder_input_ids.tile((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.tile((2, 1))

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            # we only want to use conditional signal in the 1st generation step but keeping the attention mask
            encoder_hidden_states = None
            # we also have to update the attention mask

        return {
            "input_ids": None,  # encoder_hidden_states is defined. input_ids not needed
            "encoder_hidden_states": encoder_hidden_states,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_head_mask": decoder_head_mask,
            "use_cache": use_cache,
        }

    # Copied from transformers.models.musicgen.modeling_musicgen.MusicgenForConditionalGeneration._prepare_decoder_input_ids_for_generation
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
            decoder_input_ids = ops.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = ops.cat(
                    (ops.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _prepare_encoder_hidden_states_kwargs_for_generation(
        self,
        inputs_tensor: mindspore.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prepare encoder hidden states kwargs for generation.

        Args:
            self (MusicgenMelodyForConditionalGeneration): The instance of the class.
            inputs_tensor (mindspore.Tensor): The input tensor for the model.
            model_kwargs (Dict[str, Any]): Keyword arguments for the model.
            model_input_name (Optional[str], optional): The name of the model input. Defaults to None.
            guidance_scale (Optional[float], optional): The scale for guidance. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the prepared encoder hidden states kwargs for generation.

        Raises:
            KeyError: If 'attention_mask' key is not found in model_kwargs.
            ValueError: If guidance_scale is provided and is not a float.
            TypeError: If the inputs_tensor shape does not match the required shape.

        """
        encoder_hidden_states = None
        # attention mask is consumed once to produce text conditional hidden states through the text encoder
        encoder_attention_mask = model_kwargs.pop("attention_mask")

        # 1. condition on text
        if inputs_tensor is not None:
            encoder = self.get_text_encoder()
            # Prepare args and kwargs from model kwargs.
            irrelevant_prefix = ["decoder_", "use_cache"]
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

            # make sure that encoder returns `ModelOutput`
            model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
            encoder_kwargs["return_dict"] = True
            encoder_kwargs[model_input_name] = inputs_tensor
            if encoder_attention_mask is not None:
                encoder_kwargs["attention_mask"] = encoder_attention_mask
            encoder_hidden_states = encoder(**encoder_kwargs).last_hidden_state

            # optionally project encoder_hidden_states
            if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size:
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

            # for classifier free guidance we need to add a 'null' input to our encoder hidden states
            if guidance_scale is not None and guidance_scale > 1:
                encoder_hidden_states = ops.cat(
                    [encoder_hidden_states, ops.zeros_like(encoder_hidden_states)], dim=0
                )
                if encoder_attention_mask is not None:
                    encoder_attention_mask = ops.cat(
                        [encoder_attention_mask, ops.zeros_like(encoder_attention_mask)], dim=0
                    )
            if encoder_attention_mask is not None:
                encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[..., None]

        # 2. condition on audio
        audio_hidden_states = model_kwargs.get("input_features", None)

        if inputs_tensor is not None:
            if audio_hidden_states is not None:
                null_audio_hidden_states = ops.zeros_like(audio_hidden_states)
            else:
                null_audio_hidden_states = ops.zeros(
                    (inputs_tensor.shape[0], 1, self.config.num_chroma), dtype=self.dtype
                )
            null_audio_hidden_states[:, :, 0] = 1

            if audio_hidden_states is None:
                audio_hidden_states = null_audio_hidden_states

        if audio_hidden_states is not None:
            # for classifier free guidance we need to add a 'null' input to our audio hidden states
            if guidance_scale is not None and guidance_scale > 1:
                audio_hidden_states = ops.cat([audio_hidden_states, null_audio_hidden_states], dim=0)

            # optionally project audio_hidden_states ->
            # (batch_size, seq_len, num_chroma) -> (batch_size, seq_len, hidden_size)
            if self.config.num_chroma != self.decoder.config.hidden_size:
                audio_hidden_states = self.audio_enc_to_dec_proj(audio_hidden_states)

            # pad or truncate to config.chroma_length
            if audio_hidden_states.shape[1] < self.config.chroma_length:
                n_repeat = int(math.ceil(self.config.chroma_length / audio_hidden_states.shape[1]))
                audio_hidden_states = audio_hidden_states.tile((1, n_repeat, 1))
            audio_hidden_states = audio_hidden_states[:, : self.config.chroma_length]

            if encoder_hidden_states is not None:
                encoder_hidden_states = ops.cat([audio_hidden_states.type_as(encoder_hidden_states), encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = audio_hidden_states

        model_kwargs["encoder_hidden_states"] = encoder_hidden_states

        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare_decoder_input_ids_from_labels

        This method prepares decoder input IDs from the given labels for conditional generation in the
        MusicgenMelodyForConditionalGeneration class.

        Args:
            self: MusicgenMelodyForConditionalGeneration
                The instance of the MusicgenMelodyForConditionalGeneration class.
            labels: mindspore.Tensor
                The input labels representing the target sequence for decoding.

        Returns:
            None.

        Raises:
            ValueError: If the input labels are not of type mindspore.Tensor.
            RuntimeError: If the shift_tokens_right function encounters a runtime error during the token
                shifting process.
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        """
        Resize the token embeddings for the MusicgenMelodyForConditionalGeneration class.

        Args:
            self: The instance of the MusicgenMelodyForConditionalGeneration class.

        Returns:
            None.

        Raises:
            NotImplementedError: Resizing the embedding layers via the EncoderDecoderModel directly is not supported.
            Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...)
            or model.decoder.resize_token_embeddings(...)).
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
            synced_gpus (`bool`, *optional*):
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
                or when `config.return_dict_in_generate=True`) or a `mindspore.Tensor`:

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchDecoderOnlyOutput`],
                - [`~generation.SampleDecoderOnlyOutput`],
                - [`~generation.BeamSearchDecoderOnlyOutput`],
                - [`~generation.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                - [`~generation.GreedySearchEncoderDecoderOutput`],
                - [`~generation.SampleEncoderDecoderOutput`],
                - [`~generation.BeamSearchEncoderDecoderOutput`],
                - [`~generation.BeamSampleEncoderDecoderOutput`]
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
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache
        model_kwargs["guidance_scale"] = generation_config.guidance_scale

        if model_kwargs.get("attention_mask", None) is None:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        if "encoder_hidden_states" not in model_kwargs:
            # encoder_hidden_states are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_hidden_states_kwargs_for_generation(
                inputs_tensor,
                model_kwargs,
                model_input_name,
                guidance_scale=generation_config.guidance_scale,
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
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
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

        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Musicgen Melody)
        input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            pad_token_id=generation_config.decoder_start_token_id,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

        # input_ids are ready to be placed on the streamer (if used)
        if streamer is not None:
            streamer.put(input_ids.cpu())

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

            output_values = ops.cat([output_values_left, output_values_right], dim=1)

        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
        model_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        This method updates the model keyword arguments for generation based on the provided outputs and model inputs.
        
        Args:
            self (MusicgenMelodyForConditionalGeneration): The instance of the
                MusicgenMelodyForConditionalGeneration class.
            outputs (ModelOutput): The model outputs generated during the generation process.
            model_kwargs (Dict[str, Any]): A dictionary containing the model keyword arguments to be updated.
            is_encoder_decoder (bool, optional): A boolean indicating whether the model is an encoder-decoder model.
                Defaults to False.
            standardize_cache_format (bool, optional): A boolean indicating whether to standardize the cache format.
                Defaults to False.
            model_inputs (Optional[Dict[str, Any]]): Optional dictionary containing model inputs.
        
        Returns:
            Dict[str, Any]: A dictionary containing the updated model keyword arguments for generation.
        
        Raises:
            None.
        """
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = ops.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update decoder attention mask
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = ops.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

        return model_kwargs

__all__ = [
    "MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MusicgenMelodyForConditionalGeneration",
    "MusicgenMelodyForCausalLM",
    "MusicgenMelodyModel",
    "MusicgenMelodyPreTrainedModel",
]
