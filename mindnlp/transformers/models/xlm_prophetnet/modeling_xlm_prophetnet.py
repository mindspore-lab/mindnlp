# coding=utf-8
# Copyright 2020 The Microsoft Authors and The HuggingFace Inc. team.
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
""" MindSpore XLM-ProphetNet model."""


import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import (
    ModelOutput,
    logging,
)

from ....common.activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel

from .configuration_xlm_prophetnet import XLMProphetNetConfig


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "XLMProphetNetConfig"

# Copied from transformers.models.prophetnet.modeling_prophetnet.softmax
def softmax(hidden_state, dim, onnx_trace=False):
    if onnx_trace:
        return ops.softmax(hidden_state.float(), dim=dim)
    else:
        return ops.softmax(hidden_state, dim=dim, dtype=mindspore.float32)


# Copied from transformers.models.prophetnet.modeling_prophetnet.ngram_attention_bias
def ngram_attention_bias(sequence_length, ngram, dtype):
    """
    This function computes the bias for the predict stream
    """
    left_block = (
        ops.ones((ngram, sequence_length, sequence_length), dtype=dtype) * np.finfo(mindspore.dtype_to_nptype(dtype)).min
    )
    #right_block = left_block.detach().clone()
    right_block = mindspore.Tensor(left_block.asnumpy(), dtype=left_block.dtype)
    # create bias
    for stream_idx in range(ngram):
        right_block[stream_idx].fill_diagonal(0.0, wrap=False)
        left_block[stream_idx].triu(-stream_idx + 1)

    left_block[:, :, 0] = 0
    return ops.cat([left_block, right_block], dim=2)


# Copied from transformers.models.prophetnet.modeling_prophetnet.compute_relative_buckets
def compute_relative_buckets(num_buckets, max_distance, relative_positions, is_bidirectional=False):
    """
    This function computes individual parts of the relative position buckets. For more detail, see paper.
    """
    inv_relative_positions = -relative_positions
    rel_positions_bucket = 0

    if is_bidirectional:
        num_buckets = num_buckets // 2
        rel_positions_bucket = (
            rel_positions_bucket
            + ops.lt(inv_relative_positions, ops.zeros_like(inv_relative_positions)).int() * num_buckets
        )
        inv_relative_positions = ops.abs(inv_relative_positions)
    else:
        inv_relative_positions = ops.maximum(inv_relative_positions, ops.zeros_like(inv_relative_positions))
    max_exact = num_buckets // 2
    is_small = ops.lt(inv_relative_positions, max_exact)
    val_if_large = max_exact + ops.log(inv_relative_positions.float() / max_exact) / math.log(
        max_distance / max_exact
    ) * (num_buckets - max_exact)
    val_if_large = ops.minimum(val_if_large, ops.ones_like(val_if_large) * (num_buckets - 1)).int()
    rel_positions_bucket = rel_positions_bucket + ops.where(is_small, inv_relative_positions.int(), val_if_large)
    return rel_positions_bucket


# Copied from transformers.models.prophetnet.modeling_prophetnet.compute_all_stream_relative_buckets
def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    # main stream
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1, position_ids.shape[-1], 1)
    main_stream_relative_positions = main_stream_relative_positions - position_ids.unsqueeze(-1)

    # predicting stream
    predicting_stream_relative_positions = ops.cat((position_ids - 1, position_ids), dim=-1).unsqueeze(1)
    predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, position_ids.shape[-1], 1)
    predicting_stream_relative_positions = predicting_stream_relative_positions - position_ids.unsqueeze(-1)

    # get both position buckets
    main_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, main_stream_relative_positions, is_bidirectional=False
    )
    predict_relative_position_buckets = compute_relative_buckets(
        num_buckets, max_distance, predicting_stream_relative_positions, is_bidirectional=False
    )
    return main_relative_position_buckets, predict_relative_position_buckets


@dataclass
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqLMOutput with ProphetNet->XLMProphetNet all-casing
class XLMProphetNetSeq2SeqLMOutput(ModelOutput):

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    logits_ngram: Optional[mindspore.Tensor] = None
    past_key_values: Optional[Tuple[mindspore.Tensor]] = None
    decoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    decoder_ngram_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    decoder_attentions: Optional[Tuple[mindspore.Tensor]] = None
    decoder_ngram_attentions: Optional[Tuple[mindspore.Tensor]] = None
    cross_attentions: Optional[Tuple[mindspore.Tensor]] = None
    encoder_last_hidden_state: Optional[mindspore.Tensor] = None
    encoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    encoder_attentions: Optional[Tuple[mindspore.Tensor]] = None

    @property
    def decoder_cross_attentions(self):
        warnings.warn(
            "`decoder_cross_attentions` is deprecated. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        return self.cross_attentions


@dataclass
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetSeq2SeqModelOutput with ProphetNet->XLMProphetNet all-casing
class XLMProphetNetSeq2SeqModelOutput(ModelOutput):

    last_hidden_state: mindspore.Tensor
    last_hidden_state_ngram: Optional[mindspore.Tensor] = None
    past_key_values: Optional[Tuple[mindspore.Tensor]] = None
    decoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    decoder_ngram_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    decoder_attentions: Optional[Tuple[mindspore.Tensor]] = None
    decoder_ngram_attentions: Optional[Tuple[mindspore.Tensor]] = None
    cross_attentions: Optional[Tuple[mindspore.Tensor]] = None
    encoder_last_hidden_state: Optional[mindspore.Tensor] = None
    encoder_hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    encoder_attentions: Optional[Tuple[mindspore.Tensor]] = None

    @property
    def decoder_cross_attentions(self):
        warnings.warn(
            "`decoder_cross_attentions` is deprecated. Please use `cross_attentions`"
            " instead.",
            FutureWarning,
        )
        return self.cross_attentions


@dataclass
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderModelOutput with ProphetNet->XLMProphetNet all-casing
class XLMProphetNetDecoderModelOutput(ModelOutput):

    last_hidden_state: mindspore.Tensor
    last_hidden_state_ngram: Optional[mindspore.Tensor] = None
    past_key_values: Optional[Tuple[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    hidden_states_ngram: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    ngram_attentions: Optional[Tuple[mindspore.Tensor]] = None
    cross_attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLMOutput with ProphetNet->XLMProphetNet all-casing
class XLMProphetNetDecoderLMOutput(ModelOutput):

    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    logits_ngram: Optional[mindspore.Tensor] = None
    past_key_values: Optional[Tuple[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    hidden_states_ngram: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    ngram_attentions: Optional[Tuple[mindspore.Tensor]] = None
    cross_attentions: Optional[Tuple[mindspore.Tensor]] = None


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetPreTrainedModel with ProphetNet->XLMProphetNet
class XLMProphetNetPreTrainedModel(PreTrainedModel):
    config_class = XLMProphetNetConfig
    base_model_prefix = "prophetnet"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        std = self.config.init_std
        if isinstance(cell, nn.Linear):
            cell.weight.assign_value(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.init_std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.assign_value(Tensor(weight, cell.weight.dtype))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In XLMProphetNet it is usually set to the"
            " pad_token_id. See XLMProphetNet docs for more information"
        )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = ops.masked_fill(shifted_input_ids, shifted_input_ids == -100, pad_token_id)

        assert ops.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetPositionalEmbeddings with ProphetNet->XLMProphetNet
class XLMProphetNetPositionalEmbeddings(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size. Padding ids are ignored by either offsetting
    based on padding_idx or by setting padding_idx to None and ensuring that the appropriate position ids are passed to
    the forward function.
    """

    def __init__(self, config: XLMProphetNetConfig) -> None:
        self.max_length = config.max_position_embeddings
        super().__init__(config.max_position_embeddings, config.hidden_size, config.pad_token_id)

    def forward(self, inputs_shape, attention_mask=None, past_key_values=None, position_ids=None):
        assert (position_ids is None) or (
            self.padding_idx is None
        ), "If position_ids is pre-computed then padding_idx should not be set."

        if position_ids is None:
            if past_key_values is not None:
                # position_ids is the same for every token when decoding a single step
                # Without the int() cast, it doesn't work in some cases when exporting to ONNX
                prev_num_input_ids = past_key_values[0][0].shape[2]
                num_input_ids = inputs_shape[1] + prev_num_input_ids
                position_ids = ops.ones((1, 1), dtype=mindspore.int64) * (
                    int(self.padding_idx + num_input_ids)
                )
            else:
                if attention_mask is None:
                    attention_mask = ops.ones(inputs_shape, dtype=mindspore.int64)

                # retrieve position_ids from input_ids / attention_mask
                position_ids = (
                    ops.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask
                ).long() + self.padding_idx

                # make sure position_ids are not bigger then max_length
                position_ids = position_ids.clamp(0, self.max_length - 1)

        return super().forward(position_ids), position_ids

    def _forward(self, position_ids):
        return super().forward(position_ids)


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetAttention with ProphetNet->XLMProphetNet
class XLMProphetNetAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: XLMProphetNetConfig,
        num_attn_heads: int,
    ):
        super().__init__()
        hidden_size = config.hidden_size

        self.attention_dropout = config.attention_dropout
        self.dropout = config.dropout
        self.num_attn_heads = num_attn_heads
        self.head_dim = hidden_size // num_attn_heads

        assert self.head_dim * num_attn_heads == hidden_size, (
            "`config.hidden_size` must be divisible by `config.num_encoder_attention_heads` and"
            " `config.num_decoder_attention_heads`"
        )

        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)

        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attn_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, tgt_len, hidden_size = hidden_states.shape

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        assert list(hidden_states.shape) == [
            batch_size,
            tgt_len,
            hidden_size,
        ], f"Size of hidden states should be {batch_size, tgt_len, hidden_size}, but is {hidden_states.shape}"

        # previous time steps are cached - no need to recompute key and value if they are static
        query_states = self.query_proj(hidden_states) / (self.head_dim**0.5)

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.key_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.value_proj(key_value_states), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.key_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.value_proj(hidden_states), -1, batch_size)

        if is_cross_attention:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        # project states into the correct shape
        proj_shape = (batch_size, self.num_attn_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.shape[2]
        attn_weights = ops.einsum("bsij,bsjk->bsik", query_states, key_states.swapaxes(2, 3))
        expected_shape = (batch_size, self.num_attn_heads, tgt_len, src_len)
        if attn_weights.shape != expected_shape:
            raise ValueError(f"Attention weights should have size {expected_shape}, but is {attn_weights.shape}")

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if attention_mask is not None and attention_mask.dim() == 0:
            attention_mask = None

        expected_shape = (batch_size, self.num_attn_heads, 1, src_len)
        if attention_mask is not None and attention_mask.shape != expected_shape:
            raise ValueError(f"Attention mask should have size {expected_shape}, but is {attention_mask.shape}")
        if attention_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights + attention_mask
        if output_attentions:
            attn_weights_reshaped = attn_weights
        else:
            attn_weights_reshaped = None

        attn_weights = ops.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.shape == (self.num_attn_heads,), (
                f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is"
                f" {layer_head_mask.shape}"
            )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                batch_size, self.num_attn_heads, tgt_len, src_len
            )

            # apply head_mask also on attn_weights_reshaped which is used for n-gram attention inside the model
            attn_weights_reshaped = layer_head_mask.view(1, -1, 1, 1) * attn_weights_reshaped

        attn_probs = F.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )
        attn_output = ops.einsum("bsij,bsjk->bsik", attn_probs, value_states)
        expected_shape = (batch_size, self.num_attn_heads, tgt_len, self.head_dim)
        if attn_output.shape != expected_shape:
            raise ValueError(f"`attn_output` should have shape {expected_shape}, but is of shape {attn_output.shape}")

        attn_output = attn_output.swapaxes(1, 2).reshape(batch_size, tgt_len, hidden_size)
        attn_output = self.out_proj(attn_output)

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        return attn_output, attn_weights_reshaped, past_key_value


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetFeedForward with ProphetNet->XLMProphetNet
class XLMProphetNetFeedForward(nn.Module):
    """
    This is the residual two feed-forward layer block based on the original Transformer implementation.
    """

    def __init__(self, config: XLMProphetNetConfig, ffn_dim: int):
        super().__init__()
        self.activation_fn = ACT2FN[config.activation_function]
        self.intermediate = nn.Linear(config.hidden_size, ffn_dim)
        self.output = nn.Linear(ffn_dim, config.hidden_size)
        self.activation_dropout = config.activation_dropout
        self.dropout = config.dropout

    def forward(self, hidden_states):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.output(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetNgramSelfAttention with ProphetNet->XLMProphetNet
class XLMProphetNetNgramSelfAttention(nn.Module):
    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.num_attn_heads = config.num_decoder_attention_heads
        self.dropout = config.dropout
        self.attention_dropout = config.attention_dropout
        self.head_dim = config.hidden_size // self.num_attn_heads
        self.ngram = config.ngram

        assert (
            self.head_dim * self.num_attn_heads == config.hidden_size
        ), "config.hidden_size must be divisible by num_attn_heads"
        # key, value, query projection
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # out projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # rel position embeddings
        self.relative_pos_embeddings = nn.Linear(config.hidden_size, self.num_buckets * self.num_attn_heads)

        # for onnx runtime
        self.onnx_trace = False

    def _shape(self, tensor, seq_len, batch_size):
        return tensor.view(batch_size, seq_len, self.num_attn_heads, self.head_dim).swapaxes(1, 2)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask=None,
        layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
    ):
        batch_size, ngram_sequence_length, hidden_size = hidden_states.shape
        assert list(hidden_states.shape) == [batch_size, ngram_sequence_length, hidden_size], (
            f"`hidden_states` should be of shape {batch_size, ngram_sequence_length, hidden_size}, but is of shape"
            f" {hidden_states.shape}"
        )

        # project
        query_states = self.query_proj(hidden_states)
        key_states = self.key_proj(hidden_states)
        value_states = self.value_proj(hidden_states)

        # normalize
        query_states = query_states / (self.head_dim**0.5)

        # reshape
        query_states = self._shape(query_states, ngram_sequence_length, batch_size)
        key_states = self._shape(key_states, -1, batch_size)
        value_states = self._shape(value_states, -1, batch_size)
        proj_shape = (batch_size, self.num_attn_heads, -1, self.head_dim)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        # chunk into main stream and predict stream
        hidden_states_list = hidden_states.chunk(1 + self.ngram, axis=1)
        query_states_list = query_states.chunk(1 + self.ngram, axis=2)
        key_states_list = key_states.chunk(1 + self.ngram, axis=2)
        value_states_list = value_states.chunk(1 + self.ngram, axis=2)

        main_hidden_states, hidden_states_predict_list = hidden_states_list[0], hidden_states_list[1:]
        main_query_states, predict_query_states_list = query_states_list[0], query_states_list[1:]
        main_key_states, predict_key_states_list = key_states_list[0], key_states_list[1:]
        main_value_states, predict_value_states_list = value_states_list[0], value_states_list[1:]

        # saved states are stored with shape (batch_size, num_attn_heads, seq_len, head_dim)
        if past_key_value is not None:
            prev_main_key_states = past_key_value[0]
            main_key_states = ops.cat((prev_main_key_states, main_key_states), dim=2)
            prev_main_value_states = past_key_value[1]
            main_value_states = ops.cat((prev_main_value_states, main_value_states), dim=2)

        # Update cache
        past_key_value = (main_key_states, main_value_states)

        # get seq_length of main stream only
        sequence_length = ngram_sequence_length // (1 + self.ngram)

        # MAIN-STREAM
        # main attn weights
        # [batch_size, number_heads, sequence_length, head_dimesion]
        # x [batch_size, number_heads, head_dimesion, sequence_length]
        # -> [batch_size, number_heads, sequence_length, sequence_length]
        main_attn_weights = ops.einsum("bntc,bncs->bnts", main_query_states, main_key_states.swapaxes(2, 3))

        # retrieve relative position embeddings for each layer -> see paper for more details
        main_relative_pos_embeddings = self.get_main_relative_pos_embeddings(
            main_hidden_states, main_attn_weights, position_ids, main_relative_position_buckets
        )

        main_attn_weights = main_attn_weights + main_relative_pos_embeddings

        if attention_mask is not None:
            main_attn_weights = main_attn_weights + attention_mask

        main_attn_probs = softmax(
            main_attn_weights,
            dim=-1,
            onnx_trace=self.onnx_trace,
        ).type_as(main_attn_weights)

        if layer_head_mask is not None:
            assert layer_head_mask.shape == (self.num_attn_heads,), (
                f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is"
                f" {layer_head_mask.shape}"
            )
            main_attn_probs = layer_head_mask.view(1, -1, 1, 1) * main_attn_probs.view(
                batch_size, self.num_attn_heads, -1, sequence_length
            )

        main_attn_probs = F.dropout(main_attn_probs, p=self.attention_dropout, training=self.training)
        # project to attn_output
        # [batch_size, number_heads, sequence_length, sequence_length]
        # x [batch_size, number_heads, sequence_length, head_dimesion]
        # -> [batch_size, number_heads, sequence_length, head_dimesion]
        main_attn_output = ops.einsum("bntc,bncs->bnts", main_attn_probs, main_value_states)
        # reshape so that num_heads dim is merged into last `head_dim` axis
        main_attn_output = main_attn_output.swapaxes(1, 2).reshape(batch_size, 1, sequence_length, hidden_size)
        main_attn_output = self.out_proj(main_attn_output)

        # PREDICT-STREAM
        # [batch_size, ngram, number_heads, sequence_length, head_dimesion]
        predict_query_states = ops.stack(predict_query_states_list, 1).view(
            batch_size, self.ngram, self.num_attn_heads, sequence_length, self.head_dim
        )

        # [batch_size, ngram, number_heads, 2*sequence_length, head_dimesion]
        predict_key_states = ops.stack([ops.cat([main_key_states, key], 2) for key in predict_key_states_list], 1)

        # [batch_size, sequence_length, ngram, hidden_size]
        predict_hidden_states = ops.stack(hidden_states_predict_list, dim=2)

        # [batch_size, number_heads, ngram, 2*sequence_length, head_dimesion]
        predict_value_states = ops.cat(
            [ops.cat([main_value_states, v_p], 2).unsqueeze(2) for v_p in predict_value_states_list], 2
        )

        # [batch_size, ngram, number_heads, sequence_length, head_dimesion]
        # x [batch_size, ngram, number_heads, 2*sequence_length, head_dimesion]
        # -> [batch_size, ngram, number_heads, sequence_length, 2*sequence_length]
        predict_attn_weights = ops.einsum("bnhtc,bnhsc->bnhts", (predict_query_states, predict_key_states))

        # retrieve relative position embeddings for each layer -> see paper for more details
        # [batch_size, ngram, number_heads, sequence_length, predict_relative_pos_embeddings]
        predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(
            predict_hidden_states, predict_attn_weights, position_ids, predict_relative_position_buckets
        )

        # [batch_size, ngram, number_heads, sequence_length, 2*sequence_length]
        predict_attn_weights = predict_attn_weights + predict_relative_pos_embeddings

        if extended_predict_attention_mask is not None:
            # Permuting Predict attention mask to [batch_size, ngram, number_heads, sequence_length, 2*sequence_length]
            extended_predict_attention_mask = extended_predict_attention_mask.permute(0, 2, 1, 3, 4)
            extended_predict_attention_mask = extended_predict_attention_mask.to(predict_attn_weights.dtype)
            predict_attn_weights = predict_attn_weights + extended_predict_attention_mask

        predict_attn_probs = softmax(
            predict_attn_weights,
            dim=-1,
            onnx_trace=self.onnx_trace,
        ).type_as(predict_attn_weights)

        if layer_head_mask is not None:
            assert layer_head_mask.shape == (self.num_attn_heads,), (
                f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is"
                f" {layer_head_mask.shape}"
            )
            predict_attn_probs = layer_head_mask.view(1, 1, -1, 1, 1) * predict_attn_probs

        predict_attn_probs = F.dropout(
            predict_attn_probs, p=self.attention_dropout, training=self.training
        )
        # project to attention output
        # [batch_size, ngram, number_heads, sequence_length, 2*sequence_length]
        # x [batch_size, ngram, number_heads, 2*sequence_length, head_dimesion]
        # -> [batch_size, ngram, number_heads, sequence_length, head_dimesion]
        predict_attn_output = ops.einsum(
            "bnhts,bnhsc->bnhtc", (predict_attn_probs, predict_value_states.swapaxes(1, 2))
        )

        # reshape so that num_heads dim is merged into last `head_dim` axis
        # [batch_size, ngram, number_heads, sequence_length, head_dimesion] -> [batch_size, ngram, sequence_length, hidden_size]
        predict_attn_output = predict_attn_output.swapaxes(2, 3)
        predict_attn_output = predict_attn_output.reshape(batch_size, self.ngram, sequence_length, hidden_size)
        predict_attn_output = self.out_proj(predict_attn_output)

        # concat to single attn output
        # [batch_size, (1+ngram)*sequence_length, hidden_size]
        attn_output = ops.cat([main_attn_output, predict_attn_output], 1).view(batch_size, -1, hidden_size)
        # reshape into better form for `config.output_attentions`
        main_attn_probs = main_attn_probs.view(batch_size, self.num_attn_heads, sequence_length, -1)

        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        return attn_output, main_attn_probs, predict_attn_probs, past_key_value

    def get_main_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
    ):
        # input hidden_states [batch_size, sequence_length, hidden_size]
        # input attn_weights [batch_size, num_heads, sequence_length, sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        batch_size, num_attn_heads, tgt_len, src_len = attn_weights.shape
        attn_weights = attn_weights.view(batch_size, num_attn_heads, tgt_len, src_len)
        if main_relative_position_buckets is None:
            batch_size, sequence_length = hidden_states.shape[:2]
            relative_positions = (
                ops.arange(1, attn_weights.shape[-1] + 1)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
            )
            # [batch_size, sequence_length, sequence_length+1]
            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            main_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # [batch_size, sequence_length, num_buckets * num_heads]
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
        rel_pos_embeddings = rel_pos_embeddings.view(
            rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
        )
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 3, 1, 2)
        # [batch_size, num_heads, sequence_length, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))

        main_relative_position_buckets = main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        # [batch_size * num_heads * sequence_length, sequence_length]
        main_relative_position_buckets = main_relative_position_buckets.view(
            -1, main_relative_position_buckets.shape[-1]
        )
        main_relative_position_buckets = main_relative_position_buckets.long()
        # [batch_size * num_heads * sequence_length, sequence_length]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.shape[-1])

        main_relative_pos_embeddings = ops.gather_elements(rel_pos_embeddings, 1, main_relative_position_buckets)
        main_relative_pos_embeddings = main_relative_pos_embeddings.view(batch_size, num_attn_heads, tgt_len, -1)
        return main_relative_pos_embeddings

    def get_predict_relative_pos_embeddings(
        self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
    ):
        # input hidden_states [batch_size, sequence_length, ngram, hidden_size]
        # input attn_weights [batch_size, ngram, num_heads, sequence_length, 2*sequence_length]
        # input position_ids [batch_size, sequence_length] or [1,1]
        # input predict_relative_position_buckets [batch_size, sequence_length, 2*sequence_length] or None
        batch_size, sequence_length = hidden_states.shape[0:2]

        if predict_relative_position_buckets is None:
            key_sequence_length = attn_weights.shape[-1]
            assert (
                position_ids[0][0] == key_sequence_length - 1
            ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
            relative_positions = (
                ops.arange(0, key_sequence_length)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, sequence_length, 1)
            )

            relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
            predict_relative_position_buckets = compute_relative_buckets(
                self.num_buckets, self.relative_max_distance, relative_positions, False
            )

        # [batch_size, ngram, sequence_length, hidden_size]
        hidden_states = hidden_states.swapaxes(1, 2)
        rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)

        # [batch_size, ngram, sequence_length, num_buckets, num_heads]
        rel_pos_embeddings = rel_pos_embeddings.view(
            hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
        )
        rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)
        # [batch_size * ngram * sequence_length * num_heads, num_buckets]
        rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)
        # [ngram, batch_size, num_heads * sequence_length, -1]
        predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
        predict_relative_position_buckets = predict_relative_position_buckets.repeat(
            self.ngram, 1, self.num_attn_heads, 1
        )
        # [ngram * batch_size * num_heads * sequence_length, -1]
        predict_relative_position_buckets = predict_relative_position_buckets.view(
            -1, predict_relative_position_buckets.shape[-1]
        ).long()

        predict_relative_pos_embeddings = ops.gather_elements(
            rel_pos_embeddings, 1, predict_relative_position_buckets
        )

        # [batch_size, gram, num_heads, sequence_length, -1]
        predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(
            batch_size, self.ngram, self.num_attn_heads, sequence_length, -1
        )

        return predict_relative_pos_embeddings


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetEncoderLayer with ProphetNet->XLMProphetNet, Prophetnet->XLMProphetnet
class XLMProphetNetEncoderLayer(nn.Module):
    """
    Encoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        # 1st residual block
        self.self_attn = XLMProphetNetAttention(config, config.num_encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 2nd residual block
        self.feed_forward = XLMProphetNetFeedForward(config, config.encoder_ffn_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        output_attentions: bool = False,
    ):
        # 1st residual block
        attention_output, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.self_attn_layer_norm(attention_output + hidden_states)

        # 2nd residual block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderLayer with Prophetnet->XLMProphetnet, ProphetNet->XLMProphetNet
class XLMProphetNetDecoderLayer(nn.Module):
    """
    Decoder block for XLMProphetnet
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__()
        # 1st residual block
        self.self_attn = XLMProphetNetNgramSelfAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 2nd residual block
        if config.add_cross_attention:
            self.cross_attn = XLMProphetNetAttention(config, config.num_decoder_attention_heads)
            self.cross_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 3rd residual block
        self.feed_forward = XLMProphetNetFeedForward(config, config.decoder_ffn_dim)
        self.feed_forward_layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attn_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        extended_predict_attention_mask=None,
        main_relative_position_buckets=None,
        predict_relative_position_buckets=None,
        position_ids=None,
        past_key_value=None,
        use_cache: bool = True,
        output_attentions: bool = False,
    ):
        # 1st residual block
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            extended_predict_attention_mask=extended_predict_attention_mask,
            main_relative_position_buckets=main_relative_position_buckets,
            predict_relative_position_buckets=predict_relative_position_buckets,
            position_ids=position_ids,
        )
        hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            # 2nd residual block
            attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attn_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # 3rd residual block
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class XLMProphetNetEncoder(XLMProphetNetPreTrainedModel):


    def __init__(self, config: XLMProphetNetConfig, word_embeddings: nn.Embedding = None):
        super().__init__(config)

        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size)

        self.layers = nn.ModuleList([XLMProphetNetEncoderLayer(config) for _ in range(config.num_encoder_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds has to be passed.")
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("Make sure to only pass input_ids or inputs_embeds.")
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # prepare attention mask
        if attention_mask is not None:
            extended_attention_mask = (
                1.0 - attention_mask[:, None, None, :].repeat(1, self.config.num_encoder_attention_heads, 1, 1)
            ) * np.finfo(mindspore.dtype_to_nptype(self.dtype)).min
            extended_attention_mask = extended_attention_mask.to(inputs_embeds.dtype)
        else:
            extended_attention_mask = None

        position_embeddings, position_ids = self.position_embeddings(inputs_embeds.shape[:2])

        hidden_states = inputs_embeds + position_embeddings
        hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)

        encoder_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.shape[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.shape[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_hidden_states = encoder_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    extended_attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_hidden_states = encoder_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_hidden_states, attentions=all_attentions
        )


class XLMProphetNetDecoder(XLMProphetNetPreTrainedModel):


    def __init__(self, config: XLMProphetNetConfig, word_embeddings: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.ngram = config.ngram
        self.num_buckets = config.num_buckets
        self.relative_max_distance = config.relative_max_distance
        self.dropout = config.dropout
        self.max_target_positions = config.max_position_embeddings

        self.word_embeddings = (
            word_embeddings
            if word_embeddings is not None
            else nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        )
        self.position_embeddings = XLMProphetNetPositionalEmbeddings(config)

        self.ngram_embeddings = nn.Embedding(self.ngram, config.hidden_size, None)
        self.layers = nn.ModuleList([XLMProphetNetDecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
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
    ) -> Union[Tuple, XLMProphetNetDecoderModelOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either `decoder_input_ids` or `decoder_inputs_embeds` has to be passed.")
        elif input_ids is not None and inputs_embeds is not None:
            raise ValueError("Make sure to only pass `decoder_input_ids` or `decoder_inputs_embeds`.")
        elif input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        batch_size, sequence_length = inputs_embeds.shape[:2]

        main_stream_pos_embed, position_ids = self.position_embeddings(
            (batch_size, sequence_length),
            past_key_values=past_key_values,
        )

        if past_key_values is not None:
            main_relative_position_buckets, predict_relative_position_buckets = None, None
        else:
            (
                main_relative_position_buckets,
                predict_relative_position_buckets,
            ) = self.compute_buffered_relative_buckets(position_ids)
        predicting_stream_pos_embed = self.position_embeddings._forward(position_ids + 1)

        # add position embeddings
        hidden_states = inputs_embeds + main_stream_pos_embed
        ngram_embeddings = self.ngram_embeddings.weight

        # prepare attention mask
        if past_key_values is not None:
            assert (
                hidden_states.shape[1] == 1
            ), "At the moment `use_cache` is only supported for `decoder_input_ids` of length 1"

            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed).repeat(batch_size, 1, 1)
                for ngram in range(self.ngram)
            ]
            extended_attention_mask = None
            extended_predict_attention_mask = None
        else:
            ngram_hidden_states = [
                (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed) for ngram in range(self.ngram)
            ]
            extended_attention_mask = self.prepare_attention_mask(hidden_states, attention_mask)
            extended_predict_attention_mask = self.prepare_predict_attention_mask(hidden_states, attention_mask)

        # prepare encoder attention mask
        if encoder_attention_mask is not None:
            extended_encoder_attention_mask = (
                1.0 - encoder_attention_mask[:, None, None, :].repeat(1, self.config.num_decoder_attention_heads, 1, 1)
            ) * np.finfo(mindspore.dtype_to_nptype(self.dtype)).min
            extended_encoder_attention_mask = extended_encoder_attention_mask.to(inputs_embeds.dtype)
        else:
            extended_encoder_attention_mask = None
        hidden_states = ops.cat([hidden_states] + ngram_hidden_states, 1)

        if self.embeddings_layer_norm:
            hidden_states = self.embeddings_layer_norm(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        # init attentions, hidden_states and cache with empty tuples
        all_main_stream_hidden_states = () if output_hidden_states else None
        all_ngram_stream_hidden_states = () if output_hidden_states and self.config.ngram > 0 else None
        all_main_stream_attns = () if output_attentions else None
        all_ngram_stream_attns = () if output_attentions else None
        all_cross_attns = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        present_key_values = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.shape[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                # grad cannot be kept because tensor is sliced
                all_main_stream_hidden_states += (hidden_states[:, :sequence_length],)
                if self.config.ngram > 0:
                    all_ngram_stream_hidden_states += (hidden_states[:, sequence_length:],)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    extended_attention_mask,
                    encoder_hidden_states,
                    extended_encoder_attention_mask,
                    (head_mask[idx] if head_mask is not None else None),
                    (cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    extended_predict_attention_mask,
                    main_relative_position_buckets,
                    predict_relative_position_buckets,
                    position_ids,
                    None,
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attn_mask=extended_encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    extended_predict_attention_mask=extended_predict_attention_mask,
                    main_relative_position_buckets=main_relative_position_buckets,
                    predict_relative_position_buckets=predict_relative_position_buckets,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                present_key_values += (layer_outputs[4 if output_attentions else 1],)

            if output_attentions:
                all_main_stream_attns += (layer_outputs[1],)
                all_ngram_stream_attns += (layer_outputs[2],)

                if self.config.add_cross_attention:
                    all_cross_attns += (layer_outputs[3],)

        if output_hidden_states:
            all_main_stream_hidden_states += (hidden_states[:, :sequence_length],)
            if self.config.ngram > 0:
                all_ngram_stream_hidden_states += (hidden_states[:, sequence_length:],)

        # split last_hidden_state for return
        last_hidden_state = hidden_states[:, :sequence_length]
        last_hidden_state_ngram = hidden_states[:, sequence_length:] if self.config.ngram > 0 else None

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    last_hidden_state_ngram,
                    present_key_values,
                    all_main_stream_hidden_states,
                    all_ngram_stream_hidden_states,
                    all_main_stream_attns,
                    all_ngram_stream_attns,
                    all_cross_attns,
                ]
                if v is not None
            )

        return XLMProphetNetDecoderModelOutput(
            last_hidden_state=last_hidden_state,
            last_hidden_state_ngram=last_hidden_state_ngram,
            past_key_values=present_key_values,
            hidden_states=all_main_stream_hidden_states,
            hidden_states_ngram=all_ngram_stream_hidden_states,
            attentions=all_main_stream_attns,
            ngram_attentions=all_ngram_stream_attns,
            cross_attentions=all_cross_attns,
        )

    def compute_buffered_relative_buckets(self, position_ids):
        batch_size, sequence_length = position_ids.shape

        position_ids = ops.arange(1, self.max_target_positions).repeat(1, 1)
        main_relative_buckets, predict_relative_buckets = compute_all_stream_relative_buckets(
            self.num_buckets, self.relative_max_distance, position_ids
        )

        # buffer relative buckets
        main_relative_buckets = main_relative_buckets[:, :sequence_length, :sequence_length].repeat(batch_size, 1, 1)
        predict_relative_buckets = ops.cat(
            [
                predict_relative_buckets[:, :sequence_length, :sequence_length],
                predict_relative_buckets[
                    :, :sequence_length, self.max_target_positions : self.max_target_positions + sequence_length
                ],
            ],
            2,
        ).repeat(batch_size, 1, 1)

        return main_relative_buckets, predict_relative_buckets

    def prepare_attention_mask(self, hidden_states, attention_mask):
        batch_size, seq_length = hidden_states.shape[:2]

        # get causal mask
        causal_mask = ops.full(
            (seq_length, seq_length),
            np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).min,
            dtype=hidden_states.dtype,
        )
        causal_mask = ops.triu(causal_mask, 1)

        extended_causal_mask = causal_mask[:seq_length, :seq_length][None, None, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + causal_mask.shape
        )

        # add usual attention mask
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * np.finfo(mindspore.dtype_to_nptype(self.dtype)).min
            extended_attention_mask = extended_causal_mask + extended_attention_mask
        else:
            extended_attention_mask = extended_causal_mask
        return extended_attention_mask.to(hidden_states.dtype)

    def prepare_predict_attention_mask(self, hidden_states, attention_mask):
        batch_size, seq_length = hidden_states.shape[:2]

        # get causal mask
        predict_causal_mask = ngram_attention_bias(
            self.max_target_positions, self.ngram, hidden_states.dtype
        )
        predict_causal_mask = ops.cat(
            [
                predict_causal_mask[:, :seq_length, :seq_length],
                predict_causal_mask[
                    :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
                ],
            ],
            dim=-1,
        )
        extended_predict_causal_mask = predict_causal_mask[None, None, :, :, :].expand(
            (batch_size, self.config.num_decoder_attention_heads) + predict_causal_mask.shape
        )

        # add usual attention mask
        if attention_mask is not None:
            extended_attention_mask = (1.0 - attention_mask[:, None, None, None, :]) * np.finfo(mindspore.dtype_to_nptype(self.dtype)).min
            extended_attention_mask = extended_attention_mask.expand(
                (batch_size, self.config.num_decoder_attention_heads, self.ngram, seq_length, seq_length)
            )
            # predicted stream attention_mask should always be 0
            extended_attention_mask = ops.cat(
                [extended_attention_mask, ops.zeros_like(extended_attention_mask)], dim=-1
            )
            extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
        else:
            extended_predict_attention_mask = extended_predict_causal_mask
        return extended_predict_attention_mask.to(hidden_states.dtype)

class XLMProphetNetModel(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight"]

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = XLMProphetNetEncoder(encoder_config, self.word_embeddings)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        self.decoder = XLMProphetNetDecoder(decoder_config, self.word_embeddings)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
        self.encoder.word_embeddings = self.word_embeddings
        self.decoder.word_embeddings = self.word_embeddings

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.word_embeddings, self.word_embeddings)
            self._tie_or_clone_weights(self.decoder.word_embeddings, self.word_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, XLMProphetNetSeq2SeqModelOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs
        return XLMProphetNetSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            last_hidden_state_ngram=decoder_outputs.last_hidden_state_ngram,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_ngram_hidden_states=decoder_outputs.hidden_states_ngram,
            decoder_attentions=decoder_outputs.attentions,
            decoder_ngram_attentions=decoder_outputs.ngram_attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class XLMProphetNetForConditionalGeneration(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = ["encoder.word_embeddings.weight", "decoder.word_embeddings.weight", "lm_head.weight"]

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)
        self.prophetnet = XLMProphetNetModel(config)
        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.word_embeddings, self.lm_head)

    def get_input_embeddings(self):
        return self.prophetnet.word_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, XLMProphetNetSeq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        outputs = self.prophetnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sequence_length = (
            decoder_input_ids.shape if decoder_input_ids is not None else decoder_inputs_embeds.shape[:2]
        )

        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None

        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)

        if not return_dict:
            all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
            return (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
        else:
            return XLMProphetNetSeq2SeqLMOutput(
                loss=loss,
                logits=logits,
                logits_ngram=logits_ngram,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_ngram_hidden_states=outputs.decoder_ngram_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                decoder_ngram_attentions=outputs.decoder_ngram_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

    def _compute_loss(self, logits, labels, ignore_index=-100):
        expend_targets = labels.new_zeros(self.config.ngram, labels.shape[0], labels.shape[1]).fill_(ignore_index)

        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        logits = logits.swapaxes(0, 1)
        lprobs = F.log_softmax(
            logits.view(-1, logits.shape[-1]),
            dim=-1,
        )

        loss = ops.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            eps_i = self.config.eps / lprobs.shape[-1]
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        return loss

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        assert encoder_outputs is not None, "`encoder_outputs` have to be passed for generation."

        if past_key_values:
            decoder_input_ids = decoder_input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        return self._shift_right(labels)

    @staticmethod
    # Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0,beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past

    def get_encoder(self):
        return self.prophetnet.encoder

    def get_decoder(self):
        return self.prophetnet.decoder

class XLMProphetNetForCausalLM(XLMProphetNetPreTrainedModel):
    _tied_weights_keys = [
        "prophetnet.word_embeddings.weight",
        "prophetnet.decoder.word_embeddings.weight",
        "lm_head.weight",
    ]

    def __init__(self, config: XLMProphetNetConfig):
        # set config for CLM
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.prophetnet = XLMProphetNetDecoderWrapper(config)

        self.padding_idx = config.pad_token_id
        self.disable_ngram_loss = config.disable_ngram_loss

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.prophetnet.decoder.word_embeddings

    def set_input_embeddings(self, value):
        self.prophetnet.decoder.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.prophetnet.decoder.word_embeddings, self.lm_head)

    def set_decoder(self, decoder):
        self.prophetnet.decoder = decoder

    def get_decoder(self):
        return self.prophetnet.decoder

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
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
    ) -> Union[Tuple, XLMProphetNetDecoderLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        outputs = self.prophetnet.decoder(
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

        batch_size, sequence_length = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]

        predicting_streams = outputs[1].view(batch_size, self.config.ngram, sequence_length, -1)
        predict_logits = self.lm_head(predicting_streams)

        logits = predict_logits[:, 0]
        logits_ngram = predict_logits[:, 1:] if self.config.ngram > 1 else None

        loss = None
        if labels is not None:
            loss = self._compute_loss(predict_logits, labels)

        if not return_dict:
            all_logits = tuple(v for v in [logits, logits_ngram] if v is not None)
            return (loss,) + all_logits + outputs[2:] if loss is not None else all_logits + outputs[2:]
        else:
            return XLMProphetNetDecoderLMOutput(
                loss=loss,
                logits=logits,
                logits_ngram=logits_ngram,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                hidden_states_ngram=outputs.hidden_states_ngram,
                attentions=outputs.attentions,
                ngram_attentions=outputs.ngram_attentions,
                cross_attentions=outputs.cross_attentions,
            )

    def _compute_loss(self, logits, labels, ignore_index=-100):
        expend_targets = labels.new_zeros(self.config.ngram, labels.shape[0], labels.shape[1]).fill_(ignore_index)

        for i in range(self.config.ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            expend_targets[i, :, :] = labels

        logits = logits.swapaxes(0, 1)
        lprobs = F.log_softmax(
            logits.view(-1, logits.shape[-1]),
            dim=-1,
        )

        loss = ops.nll_loss(lprobs, expend_targets.view(-1), reduction="mean")

        if self.config.eps > 0.0:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_masked_tokens = expend_targets.ne(ignore_index).view(-1)
            smooth_loss = smooth_loss[non_masked_tokens]
            smooth_loss = smooth_loss.mean()

            eps_i = self.config.eps / lprobs.shape[-1]
            loss = (1.0 - self.config.eps) * loss + eps_i * smooth_loss

        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    # Copied from transformers.models.bart.modeling_bart.BartForCausalLM._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0,beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.prophetnet.modeling_prophetnet.ProphetNetDecoderWrapper with ProphetNet->XLMProphetNet, prophetnet->XLMProphetNet
class XLMProphetNetDecoderWrapper(XLMProphetNetPreTrainedModel):
    """
    This is a wrapper class, so that [`XLMProphetNetForCausalLM`] can correctly be loaded from pretrained XLMProphetNet
    classes.
    """

    def __init__(self, config: XLMProphetNetConfig):
        super().__init__(config)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.decoder = XLMProphetNetDecoder(config, word_embeddings=self.word_embeddings)

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        self._tie_or_clone_weights(self.word_embeddings, self.decoder.get_input_embeddings())

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

__all__ = [
        "XLMProphetNetEncoder",
        "XLMProphetNetDecoder",
        "XLMProphetNetModel",
        "XLMProphetNetForConditionalGeneration",
        "XLMProphetNetForCausalLM",
        "XLMProphetNetDecoderWrapper",
    ]
