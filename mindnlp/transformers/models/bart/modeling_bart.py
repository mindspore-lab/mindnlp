# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore BART model."""
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_bart import BartConfig


logger = logging.get_logger(__name__)

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://hf-mirror.com/models?filter=bart
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    Args:
        attention_mask (Tensor): A 2D tensor representing the attention mask with values of 0 or 1. 
            Its shape is [batch_size, sequence_length], where batch_size is the number of input sequences 
            and sequence_length is the maximum length of the sequences in the batch.
    
    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A 1D tensor containing the indices of the non-zero elements in the flattened attention mask.
            - cu_seqlens (Tensor): A 1D tensor representing the cumulative sum of sequence lengths in the batch,
              padded with a zero at the beginning. Its shape is [batch_size + 1].
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

    Raises:
        None.
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].copy()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initializes a new instance of the BartLearnedPositionalEmbedding class.

        Args:
            self: The object instance.
            num_embeddings (int): The number of embeddings.
            embedding_dim (int): The dimension of the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def construct(self, input_ids: mindspore.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        ).expand(bsz, -1)

        return super().construct(positions + self.offset)


class BartAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BartConfig] = None,
    ):
        """
        Initializes an instance of the BartAttention class.

        Args:
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Whether the attention layer is used as a decoder. Defaults to False.
            bias (bool, optional): Whether to include bias terms in the linear projections. Defaults to True.
            is_causal (bool, optional): Whether the attention layer is causal. Defaults to False.
            config (Optional[BartConfig], optional): An optional BART configuration object. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
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

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method _shape is defined within the class BartAttention.

        Args:
            self: The instance of the BartAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped. It should be a multi-dimensional tensor.
            seq_len (int): The length of the sequence. It should be a positive integer.
            bsz (int): The batch size. It should be a positive integer.

        Returns:
            None.

        Raises:
            None
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
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

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

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


BART_ATTENTION_CLASSES = {
    "eager": BartAttention,
}


class BartEncoderLayer(nn.Cell):
    '''
    BartEncoderLayer represents a single layer of the BART (Bidirectional and Auto-Regressive Transformers) encoder.
    This layer consists of multi-head self-attention mechanism followed by feed-forward neural
    network (FFN) and layer normalization.

    Args:
        config (BartConfig):
            An instance of BartConfig containing the configuration for the BART model.

    Raises:
        ValueError:
            If the config provided is not of type BartConfig.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
        self_attn (BartAttention): The multi-head self-attention mechanism.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization applied after the self-attention mechanism.
        dropout (float): The dropout probability.
        activation_fn (function): The activation function used in the feed-forward neural network.
        activation_dropout (float): The dropout probability applied to the output of the activation function.
        fc1 (nn.Dense): The first fully connected layer in the feed-forward neural network.
        fc2 (nn.Dense): The second fully connected layer in the feed-forward neural network.
        final_layer_norm (nn.LayerNorm): The final layer normalization applied to the output of the feed-forward neural network.

    Methods:
        construct(hidden_states, attention_mask, layer_head_mask, output_attentions=False):
            Applies the BART encoder layer to the input hidden_states.

            Args:

            - hidden_states (mindspore.Tensor): Input to the layer of shape (batch, seq_len, embed_dim).
            - attention_mask (mindspore.Tensor): Attention mask of size (batch, 1, tgt_len, src_len)
            where padding elements are indicated by very large negative values.
            - layer_head_mask (mindspore.Tensor): Mask for attention heads in a given layer of size (encoder_attention_heads,).
            - output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.

            Returns:

            - Tuple[mindspore.Tensor, Optional[mindspore.Tensor]]: The output tensor and the attention weights
            if output_attentions is True.
    '''
    def __init__(self, config: BartConfig):
        """
        Initializes a new instance of BartEncoderLayer.

        Args:
            self: The instance of the class.
            config (BartConfig): The configuration object for the BART model, containing various parameters such as
                d_model, encoder_attention_heads, attention_dropout, dropout, activation_function, activation_dropout,
                and encoder_ffn_dim.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BART_ATTENTION_CLASSES["eager"](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_head_mask: mindspore.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == mindspore.float16 and (
            ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()
        ):
            clamp_value = finfo(hidden_states.dtype, 'max') - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class BartDecoderLayer(nn.Cell):

    """
    This class represents a BART decoder layer used in natural language processing tasks.
    The BARTDecoderLayer class implements the decoder layer architecture for the BART (Bidirectional and Auto-Regressive
    Transformers) model.

    Attributes:
        embed_dim (int): The dimension of the input embeddings.
        self_attn (BART_ATTENTION_CLASSES): Self-attention mechanism for the decoder layer.
        dropout (float): Dropout probability for regularization.
        activation_fn (ACT2FN): Activation function used in the decoder layer.
        activation_dropout (float): Dropout probability applied to the activation function output.
        self_attn_layer_norm (nn.LayerNorm): Layer normalization for the self-attention output.
        encoder_attn (BART_ATTENTION_CLASSES): Cross-attention mechanism with the encoder.
        encoder_attn_layer_norm (nn.LayerNorm): Layer normalization for the encoder attention output.
        fc1 (nn.Dense): Fully connected layer 1 in the decoder.
        fc2 (nn.Dense): Fully connected layer 2 in the decoder.
        final_layer_norm (nn.LayerNorm): Final layer normalization for the decoder output.

    Methods:
        construct(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, layer_head_mask,
                    cross_attn_layer_head_mask, past_key_value, output_attentions, use_cache):
            Constructs the forward pass of the BART decoder layer.

    Args:
        hidden_states (mindspore.Tensor): Input to the layer of shape (batch, seq_len, embed_dim).
        attention_mask (mindspore.Tensor): Attention mask of size (batch, 1, tgt_len, src_len).
        encoder_hidden_states (mindspore.Tensor): Cross-attention input to the layer of shape (batch, seq_len, embed_dim).
        encoder_attention_mask (mindspore.Tensor): Encoder attention mask of size (batch, 1, tgt_len, src_len).
        layer_head_mask (mindspore.Tensor): Mask for attention heads in a given layer of size (encoder_attention_heads).
        cross_attn_layer_head_mask (mindspore.Tensor): Mask for cross-attention heads in a given layer of size (decoder_attention_heads).
        past_key_value (Tuple(mindspore.Tensor)): Cached past key and value projection states.
        output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
        use_cache (bool, optional): Whether to use caching for past key and value states.

    Returns:
        Tuple containing the decoder layer outputs, optional attentions tensors, and cached key and value states if requested.
    """
    def __init__(self, config: BartConfig):
        """
        Args:
            self (BartDecoderLayer): The current instance of the BartDecoderLayer class.
            config (BartConfig):

                An instance of BartConfig containing the configuration parameters for the decoder layer.
                The config parameter should have the following attributes:

                - d_model (int): The dimension of the model.
                - decoder_attention_heads (int): The number of attention heads for the decoder.
                - attention_dropout (float): The dropout probability for attention layers.
                - activation_function (str): The name of the activation function to be used.
                - activation_dropout (float): The dropout probability for activation layers.
                - decoder_ffn_dim (int): The dimension of the feed-forward network in the decoder.

                This parameter is used to initialize the decoder layer with the specified configuration.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BART_ATTENTION_CLASSES["eager"](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.encoder_attn = BART_ATTENTION_CLASSES["eager"](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim])
        self.fc1 = nn.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Dense(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim])

    def construct(
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
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
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
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

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
            hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        """
        Initializes a new instance of the BartClassificationHead class.

        Args:
            self (BartClassificationHead): The current instance of the class.
            input_dim (int): The input dimension of the classifier.
            inner_dim (int): The inner dimension of the classifier.
            num_classes (int): The number of classes in the classification task.
            pooler_dropout (float): The dropout rate for the classifier's pooler layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Dense(inner_dim, num_classes)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the BartClassificationHead by processing the hidden states input.

        Args:
            self (BartClassificationHead): The instance of the BartClassificationHead class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed.
                It should have the shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The processed hidden states tensor after applying dropout, dense, tanh activation,
                and output projection operations. It has the shape (batch_size, sequence_length, hidden_size).

        Raises:
            None
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPreTrainedModel(PreTrainedModel):

    """
    BartPreTrainedModel class represents a pre-trained BART (Bidirectional and Auto-Regressive Transformers) model
    for natural language processing tasks.
    This class inherits from PreTrainedModel and includes methods for initializing weights and generating
    dummy inputs for the model.

    Attributes:
        config: The configuration instance for the BART model.

    Methods:
        _init_weights(self, cell):
            Initializes the weights for the BART model based on the specified cell type,
            using the provided configuration standard deviation.
        dummy_inputs(self): Generates dummy input data for the BART model, including input_ids and attention_mask.

    Example:
        ```python
        >>> # Initialize a BART pre-trained model
        >>> model = BartPreTrainedModel(config)
        ...
        >>> # Initialize the weights for the model
        >>> model._init_weights(cell)
        ...
        >>> # Generate dummy inputs for the model
        >>> inputs = model.dummy_inputs()
        ```
    """
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = ["encoder.version", "decoder.version"]
    _no_split_modules = [r"BartEncoderLayer", r"BartDecoderLayer"]

    def _init_weights(self, cell):
        """Initialize the weights"""
        std = self.config.init_std
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

    @property
    def dummy_inputs(self):
        """
        Method: dummy_inputs

        Description:
            This method generates dummy inputs for a BartPreTrainedModel.

        Args:
            self: BartPreTrainedModel
                The instance of BartPreTrainedModel class.

        Returns:
            dict:
                A dictionary containing dummy inputs for the model with the following keys:

                - 'attention_mask': A Tensor representing the attention mask for the input_ids.
                - 'input_ids': A Tensor representing the input token IDs.

        Raises:
        This method does not raise any exceptions.
        """
        pad_token = self.config.pad_token_id
        input_ids = mindspore.Tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class BartEncoder(BartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        """Initializes a new instance of the BartEncoder class.

        Args:
            self: The instance of the class.
            config (BartConfig): The configuration object for the Bart model.
            embed_tokens (Optional[nn.Embedding]): The optional embedding tensor for the tokens.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.CellList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([embed_dim])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the BartEncoder.

        Args:
            self: BartEncoder instance.
                The self parameter is required to access the instance variables and methods of BartEncoder.

        Returns:
            None: This method returns the embed_tokens attribute from BartEncoder, which represents the input embeddings.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the BartEncoder.

        Args:
            self (BartEncoder): The instance of BartEncoder.
                This parameter refers to the current instance of the BartEncoder class.
            value:
                The input embeddings value to be set.

                - Type: Any
                - Purpose: The input embeddings value to assign to the embed_tokens attribute of the BartEncoder instance.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None
        """
        self.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`mindspore.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = ops.rand(1)
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
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


class BartDecoder(BartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes a new instance of the BartDecoder class.

        Args:
            self: The BartDecoder instance.
            config (BartConfig):
                The configuration object for the Bart model.

                - config: BartConfig type.
                - Purpose: Specifies the configuration settings for the Bart model.
            embed_tokens (Optional[nn.Embedding]):
                Optional parameter. The embedding tokens for the Bart model.

                - embed_tokens: Optional[nn.Embedding] type.
                - Purpose: Represents the embedding tokens used in the Bart model.
                - Restrictions: Must be of type Optional[nn.Embedding]. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.CellList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layernorm_embedding = nn.LayerNorm([config.d_model])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings for the BartDecoder class.

        Args:
            self: An instance of the BartDecoder class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the BartDecoder.

        Args:
            self (BartDecoder): The instance of the BartDecoder class.
            value: The input embeddings to be set for the BartDecoder.
                Should be of the appropriate type.

        Returns:
            None.

        Raises:
            None
        """
        self.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        # 4d mask is passed through the layers
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
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand(1)
                if dropout_probability < self.layerdrop:
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


class BartModel(BartPreTrainedModel):

    """
    BartModel is a class that represents the BART (Bidirectional and Auto-Regressive Transformers) model
    for sequence-to-sequence tasks.
    It inherits from BartPreTrainedModel and encapsulates the architecture and functionality of the BART model.

    Attributes:
        shared (nn.Embedding): Shared embedding layer for both encoder and decoder parts of the model.
        encoder (BartEncoder): Encoder component of the BART model.
        decoder (BartDecoder): Decoder component of the BART model.

    Methods:
        __init__: Initializes the BART model with the provided configuration.
        _tie_weights: Ties the weights of word embeddings if specified in the configuration.
        get_input_embeddings: Retrieves the shared input embeddings.
        set_input_embeddings: Sets the shared input embeddings to the provided value.
        get_encoder: Retrieves the encoder component of the model.
        get_decoder: Retrieves the decoder component of the model.
        construct: Constructs the BART model for sequence-to-sequence tasks with the specified inputs and configurations.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        """
        Initialize the BartModel class with the provided configuration.

        Args:
            self: The instance of the BartModel class.
            config (BartConfig): The configuration object for the BartModel.
                It specifies the model's settings and hyperparameters.

                - config.pad_token_id (int): The index of the padding token in the vocabulary.
                - config.vocab_size (int): The size of the model's vocabulary.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type BartConfig.
            ValueError: If the provided pad_token_id is not a valid index in the vocabulary.
            ValueError: If the provided vocab_size is not a valid vocabulary size.
        """
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        """
        Ties the weights of the word embeddings in the encoder and decoder of the BartModel.

        Args:
            self (BartModel): An instance of the BartModel class.

        Returns:
            None

        Raises:
            None

        Description:
            This method is used to tie the weights of the word embeddings in the encoder and decoder of the BartModel.
            The tying of weights means that the same weight parameters are shared between the encoder and
            decoder embeddings.

            The method first checks if the 'tie_word_embeddings' flag in the model configuration is set to True.
            If it is, the method calls the '_tie_or_clone_weights' function to tie the weights of the
            'embed_tokens' in the encoder with the 'shared' weights.
            It then repeats the process for the 'embed_tokens' in the decoder.

            Note that tying the weights can help reduce the number of parameters in the model and improve efficiency,
            especially in scenarios where the encoder and decoder share the same vocabulary.

        Example:
            ```python
            >>> model = BartModel()
            >>> model._tie_weights()
            ```
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_input_embeddings(self):
        """
        Method to get the input embeddings for the BartModel.

        Args:
            self:
                The instance of the BartModel class.

                - Type: BartModel
                - Purpose: Represents the current instance of the BartModel class.
                - Restrictions: None.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, value):
        """
        Set input embeddings for the BartModel.

        Args:
            self (BartModel): The instance of the BartModel class.
            value (torch.Tensor): The input embeddings to be set. It should be a torch.Tensor.

        Returns:
            None.

        Raises:
            None.
        """
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        """
        Get the encoder associated with the BartModel.

        Args:
            self: BartModel instance. The current instance of BartModel.

        Returns:
            encoder: The encoder associated with the BartModel instance.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        Method get_decoder in class BartModel.

        Args:
            self: BartModel instance.
                Represents the current instance of the BartModel class.

        Returns:
            decoder:
                This method returns the decoder associated with the BartModel instance.

        Raises:
            None.
        """
        return self.decoder

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        """
        Constructs the BartModel.

        Args:
            self (BartModel): An instance of the BartModel class.
            input_ids (mindspore.Tensor, optional): The input sequence tensor. Default: None.
            attention_mask (Optional[mindspore.Tensor], optional): The attention mask tensor. Default: None.
            decoder_input_ids (Optional[mindspore.Tensor], optional): The decoder input sequence tensor. Default: None.
            decoder_attention_mask (Optional[mindspore.Tensor], optional): The decoder attention mask tensor. Default: None.
            head_mask (Optional[mindspore.Tensor], optional): The head mask tensor. Default: None.
            decoder_head_mask (Optional[mindspore.Tensor], optional): The decoder head mask tensor. Default: None.
            cross_attn_head_mask (Optional[mindspore.Tensor], optional): The cross attention head mask tensor. Default: None.
            encoder_outputs (Optional[List[mindspore.Tensor]], optional): The encoder outputs tensor. Default: None.
            past_key_values (Optional[List[mindspore.Tensor]], optional): The past key values tensor. Default: None.
            inputs_embeds (Optional[mindspore.Tensor], optional): The input embeddings tensor. Default: None.
            decoder_inputs_embeds (Optional[mindspore.Tensor], optional): The decoder input embeddings tensor. Default: None.
            use_cache (Optional[bool], optional): Whether to use cache. Default: None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Default: None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Default: None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Default: None.

        Returns:
            Union[Tuple, Seq2SeqModelOutput]: A tuple or a Seq2SeqModelOutput object containing the last hidden state,
                past key values, decoder hidden states, decoder attentions, cross attentions, encoder last
                hidden state, encoder hidden states, and encoder attentions.

        Raises:
            ValueError: If no `decoder_input_ids` or `decoder_inputs_embeds` are passed and `input_ids` is `None`.
        """
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartForConditionalGeneration(BartPreTrainedModel):

    """
    This class represents a BART model for conditional text generation.
    It inherits from BartPreTrainedModel and provides methods for model initialization, encoder and decoder retrieval,
    resizing token embeddings, output embeddings, model construction, preparing inputs for generation,
    preparing decoder input ids from labels, and reordering cache.
    The class includes methods for initializing the model, retrieving encoder and decoder, resizing token embeddings,
    constructing the model, preparing inputs for text generation, and reordering cache for efficient generation.
    Additionally, it provides methods for setting and getting output embeddings and resizing final logits bias.
    The class also includes a method for preparing decoder input ids from labels for masked language modeling.
    """
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BartConfig):
        """Initialize a BART model for conditional generation.

        Args:
            self (BartForConditionalGeneration): The object instance of the BartForConditionalGeneration class.
            config (BartConfig): The configuration object for the BART model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = BartModel(config)
        self.final_logits_bias = ops.zeros((1, self.model.shared.vocab_size))
        self.lm_head = nn.Dense(config.d_model, self.model.shared.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Method to retrieve the encoder from the BartForConditionalGeneration model.

        Args:
            self: Instance of the BartForConditionalGeneration class.
                This parameter refers to the current instance of the class.

        Returns:
            encoder:
                Returns the encoder obtained from the model.

        Raises:
            None
        """
        return self.model.get_encoder()

    def get_decoder(self):
        """
        Method to retrieve the decoder module from the BartForConditionalGeneration model.

        Args:
            self: An instance of the BartForConditionalGeneration class.

        Returns:
            decoder: Returns the decoder module from the BartForConditionalGeneration model.

        Raises:
            None.
        """
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        """
        Resize the token embeddings for the BartForConditionalGeneration model.

        Args:
            self: The instance of the BartForConditionalGeneration class.
            new_num_tokens (int): The new number of tokens to resize the embeddings to.
                Specifies the desired number of tokens for the embeddings.
            pad_to_multiple_of (Optional[int]): The optional value to pad the resize to a multiple of.
                If provided, the new embeddings size will be padded to the nearest multiple of this value.

        Returns:
            nn.Embedding: The resized token embeddings as an instance of nn.Embedding.
                Represents the updated embeddings after resizing.

        Raises:
            None.
        """
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        '''
        Resize the final logits bias to accommodate a different number of tokens.

        Args:
            self (BartForConditionalGeneration): The instance of the BartForConditionalGeneration class.
            new_num_tokens (int):
                The new number of tokens to resize the final logits bias to. It should be a positive integer.

        Returns:
            None: The method modifies the final_logits_bias attribute of the BartForConditionalGeneration instance in place.

        Raises:
            ValueError: If new_num_tokens is not a positive integer.
        '''
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.cat([self.final_logits_bias, extra_bias], axis=1)
        self.final_logits_bias = new_bias

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the BartForConditionalGeneration model.

        Args:
            self: An instance of the BartForConditionalGeneration class.
                This parameter is required to access the model's output embeddings.
                It should always be passed as the first argument when calling this method.

        Returns:
            lm_head: This method returns the lm_head attribute of the BartForConditionalGeneration instance.
                The lm_head attribute represents the output embeddings of the model.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Method:
            set_output_embeddings

        Description:
            Sets the output embeddings for the BartForConditionalGeneration model.

        Args:
            self (BartForConditionalGeneration): The instance of the BartForConditionalGeneration class.
            new_embeddings (Tensor): The new embeddings to be set as the output embeddings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of type Tensor.
            ValueError: If the new_embeddings parameter is empty or invalid.
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, Seq2SeqLMOutput]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare the inputs for generation in the BartForConditionalGeneration class.

        Args:
            self (BartForConditionalGeneration): The instance of the BartForConditionalGeneration class.
            decoder_input_ids (torch.Tensor): The input tensor representing the decoder input IDs.
            past_key_values (Tuple[torch.Tensor]): Optional. The past key values for generating the output. Default is None.
            attention_mask (torch.Tensor): Optional. The attention mask tensor for the encoder. Default is None.
            decoder_attention_mask (torch.Tensor): Optional. The attention mask tensor for the decoder. Default is None.
            head_mask (torch.Tensor): Optional. The mask tensor for the encoder's attention heads. Default is None.
            decoder_head_mask (torch.Tensor): Optional. The mask tensor for the decoder's attention heads. Default is None.
            cross_attn_head_mask (torch.Tensor): Optional. The mask tensor for the cross-attention heads. Default is None.
            use_cache (bool): Optional. Whether to use cache for faster decoding. Default is None.
            encoder_outputs (torch.Tensor): Optional. The tensor representing the encoder outputs. Default is None.

        Returns:
            dict: A dictionary containing the prepared inputs for generation.
                The dictionary has the following keys:

                - 'input_ids' (None): Represents the input IDs, which are set to None.
                - 'encoder_outputs' (torch.Tensor): Represents the encoder outputs.
                - 'past_key_values' (Tuple[torch.Tensor]): Represents the past key values for generating the output.
                - 'decoder_input_ids' (torch.Tensor): Represents the decoder input IDs after removing the prefix.
                - 'attention_mask' (torch.Tensor): Represents the attention mask tensor for the encoder.
                - 'decoder_attention_mask' (torch.Tensor): Represents the attention mask tensor for the decoder.
                - 'head_mask' (torch.Tensor): Represents the mask tensor for the encoder's attention heads.
                - 'decoder_head_mask' (torch.Tensor): Represents the mask tensor for the decoder's attention heads.
                - 'cross_attn_head_mask' (torch.Tensor): Represents the mask tensor for the cross-attention heads.
                - 'use_cache' (bool): Represents whether to use cache for faster decoding.

        Raises:
            None.
        """
        # cut decoder_input_ids if past_key_values is used
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
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare decoder input IDs from labels.

        This method takes in two parameters: self, labels.
        The 'self' parameter refers to the current instance of the 'BartForConditionalGeneration' class,
        while the 'labels' parameter is a tensor containing the input labels.

        Args:
            self (BartForConditionalGeneration): The current instance of the BartForConditionalGeneration class.
            labels (mindspore.Tensor): A tensor containing the input labels.

        Returns:
            None

        Raises:
            None
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the BartForConditionalGeneration model based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key and value states for each layer of the model.
                The states are used to generate the next token probabilities.
            beam_idx (torch.Tensor): A tensor representing the indices of the beams to reorder the cache.

        Returns:
            None: This method does not return a value but modifies the past_key_values in place
                by reordering the cache based on the beam index.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


class BartForSequenceClassification(BartPreTrainedModel):

    """
    The `BartForSequenceClassification` class represents a BART model fine-tuned for sequence classification tasks.
    It inherits from the `BartPreTrainedModel` class and includes methods for model initialization and sequence classification.

    This class includes an `__init__` method for initializing the BART model and a `construct` method for
    constructing the sequence classification outputs.
    The `construct` method accepts various input and output parameters, including input and output tensors,
    attention masks, labels, and cache usage.
    It processes the input data through the BART model, computes the classification logits, and calculates the loss
    based on the specified problem type.

    The class also includes additional methods for handling sequence classification tasks and managing model outputs.
    The `BartForSequenceClassification` class provides a comprehensive solution for utilizing
    BART models for sequence classification applications.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig, **kwargs):
        """
        Initializes a new instance of BartForSequenceClassification.

        Args:
            self: The instance of the class.
            config (BartConfig): The configuration for the BART model.
                It contains the model architecture and hyperparameters.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not an instance of BartConfig.
        """
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        # if len(ops.unique_consecutive(eos_mask.sum(1))) > 1:
        #     raise ValueError("All examples must have the same number of <eos> tokens.")

        sentence_representation = hidden_states[eos_mask].view(hidden_states.shape[0], -1, hidden_states.shape[-1])[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

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
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartForQuestionAnswering(BartPreTrainedModel):

    """
    This class represents a BART model for question answering tasks. It inherits from the BartPreTrainedModel class.

    BARTForQuestionAnswering is a fine-tuned version of the BART model, specifically designed for question answering tasks.
    It takes in input sequences and returns the predicted start and end positions of the answer span within the input sequence.

    The BARTForQuestionAnswering class contains the following methods:

    - __init__(self, config): Initializes the BARTForQuestionAnswering model with the provided configuration.
    - construct(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask, decoder_head_mask,
    cross_attn_head_mask, encoder_outputs, start_positions, end_positions, inputs_embeds,
    decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
    Constructs the BART model for question answering and returns the predicted start and end positions of the answer span.

    The construct method takes the following parameters:

    - input_ids (mindspore.Tensor): The input token IDs.
    - attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
    - decoder_input_ids (Optional[mindspore.Tensor]): The decoder input token IDs.
    - decoder_attention_mask (Optional[mindspore.Tensor]): The decoder attention mask tensor.
    - head_mask (Optional[mindspore.Tensor]): The attention head mask tensor.
    - decoder_head_mask (Optional[mindspore.Tensor]): The decoder attention head mask tensor.
    - cross_attn_head_mask (Optional[mindspore.Tensor]): The cross-attention head mask tensor.
    - encoder_outputs (Optional[List[mindspore.Tensor]]): The encoder outputs tensor.
    - start_positions (Optional[mindspore.Tensor]): The labels for the start positions of the answer span.
    - end_positions (Optional[mindspore.Tensor]): The labels for the end positions of the answer span.
    - inputs_embeds (Optional[mindspore.Tensor]): The embedded input tensor.
    - decoder_inputs_embeds (Optional[mindspore.Tensor]): The embedded decoder input tensor.
    - use_cache (Optional[bool]): Whether to use cache.
    - output_attentions (Optional[bool]): Whether to output attentions.
    - output_hidden_states (Optional[bool]): Whether to output hidden states.
    - return_dict (Optional[bool]): Whether to return a Seq2SeqQuestionAnsweringModelOutput object.

    The construct method returns a Seq2SeqQuestionAnsweringModelOutput object that contains the following attributes:

    - loss (Optional[mindspore.Tensor]): The total loss.
    - start_logits (mindspore.Tensor): The predicted start logits.
    - end_logits (mindspore.Tensor): The predicted end logits.
    - past_key_values (Optional[mindspore.Tensor]): The past key values.
    - decoder_hidden_states (Optional[mindspore.Tensor]): The decoder hidden states.
    - decoder_attentions (Optional[mindspore.Tensor]): The decoder attentions.
    - cross_attentions (Optional[mindspore.Tensor]): The cross attentions.
    - encoder_last_hidden_state (Optional[mindspore.Tensor]): The encoder last hidden state.
    - encoder_hidden_states (Optional[mindspore.Tensor]): The encoder hidden states.
    - encoder_attentions (Optional[mindspore.Tensor]): The encoder attentions.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the 'BartForQuestionAnswering' class.

        Args:
            self: The object instance.
            config: An instance of the 'BartConfig' class containing the model configuration.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BartModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BartDecoderWrapper(BartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config):
        """
        Initialize the BartDecoderWrapper class with the provided configuration.

        Args:
            self (object): The instance of the BartDecoderWrapper class.
            config (object): The configuration object containing initialization parameters.
                This config object is used to set up the BartDecoderWrapper instance.
                It should include all necessary parameters for configuring the BartDecoderWrapper.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.decoder = BartDecoder(config)

    def construct(self, *args, **kwargs):
        """
        Constructs a decoder wrapper instance.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None.

        Raises:
            Any exceptions raised by the self.decoder method may propagate to the caller.
        """
        return self.decoder(*args, **kwargs)


class BartForCausalLM(BartPreTrainedModel):

    """
    This class represents a Bart model for causal language modeling (LM). It is a subclass of BartPreTrainedModel.

    BartForCausalLM is designed for generating text in an autoregressive manner,
    where the model predicts the next word in a sequence given the previous words.
    It consists of a decoder component that takes input_ids and attention_mask as inputs,
    and produces a sequence of predicted logits.
    The decoder can be configured with various options such as encoder_hidden_states, encoder_attention_mask, head_mask,
    cross_attn_head_mask, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states,
    and return_dict.

    The class provides methods for getting and setting the input and output embeddings,
    as well as getting and setting the decoder component.
    The construct method is the main method for generating text. It akes input_ids, attention_mask, and other optional
    arguments, and returns the predicted logits, along with other optional outputs such as loss,
    past_key_values, hidden_states, attentions, and cross_attentions.

    The prepare_inputs_for_generation method is used to prepare inputs for text generation. It takes input_ids,
    past_key_values, attention_mask, use_cache, and other optional arguments, and returns a dictionary containing
    the prepared inputs.

    The _reorder_cache method is a static method that is used to reorder the past_key_values cache during beam search.

    Example:
        ```python
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        >>> model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        ```

    This example demonstrates how to use the BartForCausalLM class for text generation.
    The model takes input_ids as input and generates predicted logits as output.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of BartForCausalLM.

        Args:
            self: The instance of BartForCausalLM.
            config (dict): The configuration parameters for the model.
                Must contain the necessary settings for the model initialization.

        Returns:
            None.

        Raises:
            AttributeError: If the provided configuration is missing required attributes.
            TypeError: If the configuration is not in the expected format.
        """
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BartDecoderWrapper(config)

        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the decoder of a BartForCausalLM model.

        Args:
            self: BartForCausalLM - The instance of BartForCausalLM class.
                This parameter represents the current instance of the BartForCausalLM class.

        Returns:
            embed_tokens:
                This method returns the input embeddings from the decoder of the BartForCausalLM model.

        Raises:
            None.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the BartForCausalLM model.

        Args:
            self (BartForCausalLM): The instance of the BartForCausalLM class.
            value (torch.Tensor): The input embeddings to be set for the model.
                This should be a torch.Tensor representing the new input embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """Return the output embeddings of the BartForCausalLM model.

        Args:
            self (BartForCausalLM): The instance of the BartForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the `BartForCausalLM` model.

        Args:
            self: An instance of the `BartForCausalLM` class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as the output embeddings.
                This should be an instance of `torch.nn.Module` subclass, which represents the new embeddings
                to be used as output in the `BartForCausalLM` model. It is expected to have the same size as the
                existing embeddings.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method replaces the existing output embeddings in the `BartForCausalLM` model with the provided
            new embeddings. It is useful when fine-tuning the model's output layer or updating the embeddings with
            pre-trained weights.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the BartForCausalLM model.

        Args:
            self (BartForCausalLM): The instance of the BartForCausalLM class.
            decoder (nn.Module): The decoder module to be set for the model.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method allows the user to set the decoder module for the BartForCausalLM model.
            The decoder module is responsible for generating the output sequence during the model's forward pass.

            The `self` parameter refers to the instance of the BartForCausalLM class on which the method is called.

            The `decoder` parameter is of type `nn.Module` and represents the decoder module to be set for the model.
            The decoder module should be compatible with the BartForCausalLM model architecture.

            Note that setting the decoder module will overwrite any previously set decoder module for the model.

        Example:
            ```python
            >>> model = BartForCausalLM()
            >>> decoder = nn.Linear(768, 1024)
            >>> model.set_decoder(decoder)
            ```
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        This method returns the decoder component of the model.

        Args:
            self: An instance of the BartForCausalLM class.

        Returns:
            None

        Raises:
            No specific exceptions are raised by this method.
        """
        return self.model.decoder

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, BartForCausalLM
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            >>> model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> logits = outputs.logits
            >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
            >>> list(logits.shape) == expected_shape
            True
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
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

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        """
        Prepare the inputs for generation in the BartForCausalLM class.

        This method takes 5 parameters: self, input_ids, past_key_values, attention_mask, use_cache.

        Args:
            self: The instance of the BartForCausalLM class.
            input_ids (torch.Tensor): Tensor containing the input ids for the generation.
            past_key_values (tuple): Tuple of past key values for the generation. Default is None.
            attention_mask (torch.Tensor): Tensor containing the attention mask for the input ids.
                If None, a new attention mask with all ones will be created. Default is None.
            use_cache (bool): Whether or not to use past key values cache. Default is None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation.

                - input_ids (torch.Tensor): The modified input ids.
                - attention_mask (torch.Tensor): The attention mask.
                - past_key_values (tuple): The past key values.
                - use_cache (bool): The use_cache flag.
        
        Raises:
            None.
        
        """
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the BartForCausalLM model based on the specified beam index.
        
        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer of the model.
                Each element in the tuple represents the past key-value states for a layer.
            beam_idx (torch.Tensor): A 1D tensor containing the indices of the beams to reorder the past states.
        
        Returns:
            None: This method does not return any value but modifies the 'past_key_values' in place.
        
        Raises:
            IndexError: If the 'beam_idx' tensor contains indices that are out of range for the past states.
            TypeError: If the input types are not as expected, this method may raise a TypeError.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

__all__ = [
    "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
    "BartForCausalLM",
    "BartForConditionalGeneration",
    "BartForQuestionAnswering",
    "BartForSequenceClassification",
    "BartModel",
    "BartPreTrainedModel",
]
