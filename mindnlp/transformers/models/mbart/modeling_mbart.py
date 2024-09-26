# coding=utf-8
# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
MindNLP MBART model.
"""
import copy
import math
import random
from typing import List, Optional, Tuple, Union
import numpy as np
from mindspore import log as logger
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
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
from .configuration_mbart import MBartConfig

MBART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mbart-large-cc25",
    # See all MBART models at https://hf-mirror.com/models?filter=mbart
]


def shift_tokens_right(input_ids: Tensor, pad_token_id: int):
    """
    Shift input ids one token to the right, and wrap the last non pad token (the <LID> token) Note that MBart does not
    have a single `decoder_start_token_id` in contrast to other Bart-like models.
    """
    prev_output_tokens = input_ids.copy()

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    prev_output_tokens.masked_fill(prev_output_tokens == -100, pad_token_id)

    index_of_eos = (prev_output_tokens.ne(pad_token_id).sum(axis=1) - 1).unsqueeze(-1)
    decoder_start_tokens = prev_output_tokens.gather_elements(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = prev_output_tokens[:, :-1].copy()
    prev_output_tokens[:, 0] = decoder_start_tokens

    return prev_output_tokens


class MBartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initializes a new instance of the MBartLearnedPositionalEmbedding class.
        
        Args:
            self (MBartLearnedPositionalEmbedding): The current instance of the MBartLearnedPositionalEmbedding class.
            num_embeddings (int): The number of embeddings.
            embedding_dim (int): The dimension of each embedding.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        # MBart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: Tensor, past_key_values_length: int = 0):
        """`ids' shape is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        ).broadcast_to((bsz, -1))

        return super().forward(positions + self.offset)


class MBartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        '''
        This method initializes an instance of the MBartAttention class.
        
        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads to use.
            dropout (float, optional): The dropout probability. Default is 0.0.
            is_decoder (bool, optional): Indicates if the attention mechanism is used in a decoder context.
                Default is False.
            bias (bool): Indicates whether bias is applied in linear transformations.

        Returns:
            None

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' is defined within the class 'MBartAttention' and is used to reshape the input tensor
        for multi-head self-attention computation.

        Args:
            self (object): The instance of the 'MBartAttention' class.
            tensor (mindspore.Tensor): The input tensor to be reshaped for multi-head self-attention computation.
            seq_len (int): The length of the sequence in the input tensor.
            bsz (int): The batch size of the input tensor.

        Returns:
            None: This method does not return any value. It performs an in-place operation on the input tensor.

        Raises:
            ValueError: If the input tensor or the batch size is invalid or incompatible with the reshaping operation.
            TypeError: If the input tensor or batch size is not of the expected type.
            RuntimeError: If an unexpected error occurs during the reshaping process.
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


class MBartEncoderLayer(nn.Module):
    """MBartEncoderLayer"""
    def __init__(self, config: MBartConfig):
        """
        Initializes an instance of the MBartEncoderLayer class.

        Args:
            self: The current instance of the class.
            config (MBartConfig):
                An instance of the MBartConfig class containing the configuration settings for the encoder layer.

                - The 'config' parameter is of type MBartConfig.
                - It specifies the configuration settings for the encoder layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim], eps=1e-5)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim], eps=1e-5)

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Tensor,
            layer_head_mask: Tensor,
            output_attentions: bool = False,
    ) -> Tensor:
        """
        Constructs the MBartEncoderLayer.

        Args:
            self: The instance of the MBartEncoderLayer class.
            hidden_states (Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor): The attention mask tensor of shape (batch_size, sequence_length, sequence_length).
            layer_head_mask (Tensor): The layer head mask tensor of shape
                (num_attention_heads, sequence_length, sequence_length).
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.

        Returns:
            Tensor: The output hidden states tensor of shape (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == mindspore.float16 and (
                ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()
        ):
            clamp_value = np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MBartDecoderLayer(nn.Module):
    """MBartDecoderLayer"""
    def __init__(self, config: MBartConfig):
        """
        Initializes an instance of the MBartDecoderLayer class.

        Args:
            self: The instance of the MBartDecoderLayer class.
            config (MBartConfig): The configuration object for the MBart model.
                It contains the model parameters and settings.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MBartConfig.
            ValueError: If the config parameter is missing or has invalid values.
            RuntimeError: If there are issues with initializing the model layers or norms.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = MBartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm([self.embed_dim], eps=1e-5)
        self.encoder_attn = MBartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm([self.embed_dim], eps=1e-5)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm([self.embed_dim], eps=1e-5)

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            layer_head_mask: Optional[Tensor] = None,
            cross_attn_layer_head_mask: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
    ) -> Tensor:
        """
        Constructs the MBartDecoderLayer.

        Args:
            self (MBartDecoderLayer): The instance of the MBartDecoderLayer class.
            hidden_states (Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[Tensor]): The attention mask tensor of shape (batch_size, sequence_length) or
                (batch_size, sequence_length, sequence_length), indicating which tokens should be attended to.
                Defaults to None.
            encoder_hidden_states (Optional[Tensor]): The hidden states tensor from the encoder of shape
                (batch_size, encoder_sequence_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[Tensor]): The attention mask tensor for the encoder of shape
                (batch_size, encoder_sequence_length) or (batch_size, encoder_sequence_length, encoder_sequence_length).
                Defaults to None.
            layer_head_mask (Optional[Tensor]): The mask tensor for attention heads of shape
                (batch_size, num_heads, sequence_length, sequence_length). Defaults to None.
            cross_attn_layer_head_mask (Optional[Tensor]): The mask tensor for attention heads in the cross-attention
                layer of shape (batch_size, num_heads, sequence_length, encoder_sequence_length). Defaults to None.
            past_key_value (Optional[Tuple[Tensor]]): The tuple of tensors containing the past key and value states.
                Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.
            use_cache (Optional[bool]): Whether to use the cache. Defaults to True.

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size).

        Raises:
            None
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


class MBartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        """
        Initializes an instance of the MBartClassificationHead class.

        Args:
            input_dim (int): The dimension of the input features.
            inner_dim (int): The dimension of the inner layer.
            num_classes (int): The number of output classes.
            pooler_dropout (float): The dropout probability for the pooler layer.

        Returns:
            None.

        Raises:
            ValueError: If input_dim, inner_dim, num_classes, or pooler_dropout is not a positive integer.
            TypeError: If input_dim, inner_dim, num_classes, or pooler_dropout is not of the correct type.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        This method forwards the MBartClassificationHead by processing the input hidden_states.

        Args:
            self (MBartClassificationHead): The instance of the MBartClassificationHead class.
            hidden_states (Tensor): A tensor representing the hidden states. It is the input to be processed.

        Returns:
            Tensor: A tensor representing the processed hidden states.

        Raises:
            None
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class MBartPreTrainedModel(PreTrainedModel):
    """MBartPreTrainedModel"""
    config_class = MBartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MBartDecoderLayer", "MBartAttention"]

    def _init_weights(self, cell):
        """Initializes the weights of a cell in the MBartPreTrainedModel.

        Args:
            self (MBartPreTrainedModel): The instance of MBartPreTrainedModel.
            cell (nn.Module): The cell whose weights are to be initialized.

        Returns:
            None: This method operates in-place and does not return any value.

        Raises:
            None.
        """
        std = self.config.init_std
        if isinstance(cell, nn.Linear):
            cell.weight.assign_value(initializer(Normal(self.config.init_std),
                                               cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.assign_value(initializer('zeros',
                                               cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.assign_value(Tensor(weight, cell.weight.dtype))

    @property
    def dummy_inputs(self):
        """dummy_inputs"""
        pad_token = self.config.pad_token_id
        input_ids = Tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class MBartEncoder(MBartPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MBartEncoderLayer`].

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes a new instance of the MBartEncoder class.

        Args:
            self: The object itself.
            config (MBartConfig): The configuration for the MBart model.
            embed_tokens (Optional[nn.Embedding]): An optional pre-trained embedding to be used for the tokens.
                If provided, it will be used instead of the default embedding in the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, padding_idx=self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([MBartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([embed_dim])
        self.layer_norm = nn.LayerNorm([config.d_model])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        '''
        Construct method in the MBartEncoder class.

        Args:
            self: MBartEncoder
                The instance of the MBartEncoder class.
            input_ids: Tensor, optional
                The input tensor containing the tokenized input sequence.
            attention_mask: Optional[Tensor], optional
                The attention mask tensor specifying which tokens should be attended to.
            head_mask: Optional[Tensor], optional
                The head mask tensor to mask heads in the self-attention layers.
            inputs_embeds: Optional[Tensor], optional
                The embedded input tensor if input_ids is not provided.
            output_attentions: Optional[bool], optional
                Whether to return the attentions.
            output_hidden_states: Optional[bool], optional
                Whether to return the hidden states.
            return_dict: Optional[bool], optional
                Whether to return a dictionary.

        Returns:
            Union[Tuple, BaseModelOutput]
                Returns a tuple or BaseModelOutput based on return_dict parameter.

        Raises:
            ValueError:
                - If both input_ids and inputs_embeds are provided simultaneously.
                - If neither input_ids nor inputs_embeds are provided.
                - If the head_mask is specified for an incorrect number of layers.

            TypeError:
                If the input_ids and inputs_embeds are not of type Tensor.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            _input = input_ids
            input_ids = input_ids.view(-1, input_ids.shape[-1])
        elif inputs_embeds is not None:
            _input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(_input)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and dropout_probability < self.layerdrop:  # skip the layer
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

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class MBartDecoder(MBartPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MBartDecoderLayer`]

    Args:
        config: MBartConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: MBartConfig, embed_tokens: Optional[nn.Embedding] = None):
        """Initialize the MBartDecoder class.

        Args:
            self: The object itself.
            config (MBartConfig): The configuration object for MBart.
                Contains various hyperparameters and settings for the model.
            embed_tokens (Optional[nn.Embedding]): An optional embedding layer.
                If provided, the weights of this layer will be used for the embed_tokens layer.
                Defaults to None.

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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, padding_idx=self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([MBartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm([config.d_model], eps=1e-5)
        self.layer_norm = nn.LayerNorm([config.d_model], eps=1e-5)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings used by the MBartDecoder.

        Args:
            self (MBartDecoder): An instance of the MBartDecoder class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        This method 'set_input_embeddings' is a member of the 'MBartDecoder' class and is used to set the
        input embeddings for the decoder.

        Args:
            self (object): The instance of the 'MBartDecoder' class.
            new_embeddings (object): The new input embeddings to be set for the decoder.
                It should be of the appropriate type and format compatible with the decoder's input requirements.

        Returns:
            None: This method does not return any value explicitly. It updates the input embeddings of the decoder in place.

        Raises:
            None.
        """
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        This method forwards the MBartDecoder model.

        Args:
            self: The object instance.
            input_ids (Tensor, optional): The input tensor containing token ids. Default is None.
            attention_mask (Optional[Tensor], optional): The attention mask tensor. Default is None.
            encoder_hidden_states (Optional[Tensor], optional): The hidden states of the encoder. Default is None.
            encoder_attention_mask (Optional[Tensor], optional): The attention mask for the encoder. Default is None.
            head_mask (Optional[Tensor], optional): The mask for attention heads. Default is None.
            cross_attn_head_mask (Optional[Tensor], optional): The mask for cross-attention heads. Default is None.
            past_key_values (Optional[Tuple[Tuple[Tensor]]], optional): The past key values. Default is None.
            inputs_embeds (Tensor, optional): The embedded input tensors. Default is None.
            use_cache (bool, optional): Flag to use cache. Default is None.
            output_attentions (bool, optional): Flag to output attentions. Default is None.
            output_hidden_states (bool, optional): Flag to output hidden states. Default is None.
            return_dict (bool, optional): Flag to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: A tuple or BaseModelOutputWithPastAndCrossAttentions
                object representing the output of the method.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If the specified head_mask or cross_attn_head_mask does not match the number of layers in the model.
            Warning: If `use_cache=True` is used with gradient checkpointing, as it is incompatible.
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
            _input = input_ids
            input_shape = _input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            _input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            # or _input
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

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
        positions = self.embed_positions(_input, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

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
            if self.training:
                dropout_probability = random.uniform(0, 1)
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


class MBartModel(MBartPreTrainedModel):
    """MBartModel"""
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MBartConfig):
        """Initialize an instance of the MBartModel class.

        Args:
            self: The instance of the MBartModel class.
            config (MBartConfig): The configuration object for the MBartModel.
                It specifies the parameters for the model, such as vocabulary size, model dimension, etc.
                The config parameter is of type MBartConfig and is required.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx=padding_idx)

        self.encoder = MBartEncoder(config, self.shared)
        self.decoder = MBartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the shared input embeddings for the MBartModel.

        Args:
            self (MBartModel): The instance of the MBartModel class.

        Returns:
            None: This method returns None as it directly returns the shared input embeddings.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the MBartModel.

        Args:
            self (MBartModel): The instance of the MBartModel class.
            new_embeddings (torch.Tensor): The new input embeddings to be set.
                Should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not a torch.Tensor.
            ValueError: If the shape of the new_embeddings tensor is invalid.
        """
        self.shared = new_embeddings
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def _tie_weights(self):
        """
        This method _tie_weights is a member of the class MBartModel and is used to tie the weights of word embeddings
        if the tie_word_embeddings configuration is set to True.

        Args:
            self: An instance of the MBartModel class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.get_input_embeddings())
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.get_input_embeddings())

    def get_encoder(self):
        """get_encoder"""
        return self.encoder

    def get_decoder(self):
        """get_decoder"""
        return self.decoder

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqModelOutput, Tuple[Tensor]]:
        """
        Constructs the MBartModel.

        Args:
            self (MBartModel): The instance of the MBartModel class.
            input_ids (Tensor, optional): The input token IDs tensor. Default: None.
            attention_mask (Optional[Tensor], optional): The attention mask tensor. Default: None.
            decoder_input_ids (Optional[Tensor], optional): The decoder input token IDs tensor. Default: None.
            decoder_attention_mask (Optional[Tensor], optional): The decoder attention mask tensor. Default: None.
            head_mask (Optional[Tensor], optional): The head mask tensor. Default: None.
            decoder_head_mask (Optional[Tensor], optional): The decoder head mask tensor. Default: None.
            cross_attn_head_mask (Optional[Tensor], optional): The cross attention head mask tensor. Default: None.
            encoder_outputs (Optional[Tuple[Tuple[Tensor]]], optional): The encoder outputs tensor. Default: None.
            past_key_values (Optional[Tuple[Tuple[Tensor]]], optional): The past key values tensor. Default: None.
            inputs_embeds (Optional[Tensor], optional): The input embeddings tensor. Default: None.
            decoder_inputs_embeds (Optional[Tensor], optional): The decoder input embeddings tensor. Default: None.
            use_cache (Optional[bool], optional): Whether to use cache. Default: None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Default: None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Default: None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Default: None.

        Returns:
            Union[Seq2SeqModelOutput, Tuple[Tensor]]: The output of the MBartModel.

        Raises:
            None.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # different to other models, MBart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id)

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


class MBartForConditionalGeneration(MBartPreTrainedModel):
    """MBartForConditionalGeneration"""
    base_model_prefix = "model"
    # _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: MBartConfig):
        """
        __init__

        Initializes an instance of the MBartForConditionalGeneration class.

        Args:
            self: The object instance itself.
            config (MBartConfig): An instance of MBartConfig class containing the configuration settings for the MBart model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = MBartModel(config)
        self.final_logits_bias = ops.zeros((1, self.model.shared.num_embeddings))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """get_encoder"""
        return self.model.get_encoder()

    def get_decoder(self):
        """get_decoder"""
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        """
        Resizes the token embeddings of the MBartForConditionalGeneration model.

        Args:
            self (MBartForConditionalGeneration): The instance of the MBartForConditionalGeneration class.
            new_num_tokens (int): The new number of tokens for the token embeddings.
            pad_to_multiple_of (Optional[int], optional): The value to pad the number of tokens to a multiple of.
                Defaults to None.

        Returns:
            nn.Embedding: The resized token embeddings.

        Raises:
            None.

        This method resizes the token embeddings of the MBartForConditionalGeneration model by calling the base class's
        'resize_token_embeddings' method with the specified 'new_num_tokens' and 'pad_to_multiple_of' values.
        The resulting resized token embeddings are then used to resize the final logits bias.
        The method returns the resized token embeddings.
        """
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        """_resize_final_logits_bias"""
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.concat([self.final_logits_bias, extra_bias], dim=1)
        self.final_logits_bias = new_bias

    def get_output_embeddings(self):
        """get_output_embeddings"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """set_output_embeddings"""
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Seq2SeqLMOutput, Tuple[Tensor]]:
        """
        Constructs the MBart model for conditional generation.

        Args:
            self (MBartForConditionalGeneration): The instance of the MBartForConditionalGeneration class.
            input_ids (Tensor, optional): The input sequence IDs. Defaults to None.
            attention_mask (Tensor, optional): The attention mask. Defaults to None.
            decoder_input_ids (Tensor, optional): The decoder input sequence IDs. Defaults to None.
            decoder_attention_mask (Tensor, optional): The decoder attention mask. Defaults to None.
            head_mask (Tensor, optional): The head mask. Defaults to None.
            decoder_head_mask (Tensor, optional): The decoder head mask. Defaults to None.
            cross_attn_head_mask (Tensor, optional): The cross-attention head mask. Defaults to None.
            encoder_outputs (Tuple[Tuple[Tensor]], optional): The encoder outputs. Defaults to None.
            past_key_values (Tuple[Tuple[Tensor]], optional): The past key values. Defaults to None.
            inputs_embeds (Tensor, optional): The input embeddings. Defaults to None.
            decoder_inputs_embeds (Tensor, optional): The decoder input embeddings. Defaults to None.
            labels (Tensor, optional): The labels tensor. Defaults to None.
            use_cache (bool, optional): Flag to indicate whether to use cache. Defaults to None.
            output_attentions (bool, optional): Flag to indicate whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Flag to indicate whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Flag to indicate whether to return a dictionary. Defaults to None.

        Returns:
            Union[Seq2SeqLMOutput, Tuple[Tensor]]: The output of the model. If `return_dict` is False, returns a tuple
                containing the masked language model logits and additional outputs. If `return_dict` is True, returns a
                Seq2SeqLMOutput object containing various model outputs.

        Raises:
            None.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

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
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        """
        Prepare inputs for generation.

        Args:
            self (MBartForConditionalGeneration): The instance of the MBartForConditionalGeneration class.
            decoder_input_ids (torch.Tensor): The input decoder sequence of token indices.
            past_key_values (Tuple[torch.Tensor]): The cached key-value pairs of the decoder's self-attention layers.
            attention_mask (torch.Tensor): The attention mask tensor indicating which tokens to attend to and
                which ones to ignore.
            head_mask (torch.Tensor): The mask tensor to nullify selected heads of the self-attention layers
                in the decoder.
            decoder_head_mask (torch.Tensor): The mask tensor to nullify selected heads of the self-attention layers in
                the decoder's cross-attention.
            cross_attn_head_mask (torch.Tensor): The mask tensor to nullify selected heads of the cross-attention layers
                in the decoder.
            use_cache (bool): Whether to use the cache for the decoder's self-attention layers.
            encoder_outputs (torch.Tensor): The output tensor from the encoder.

        Returns:
            dict: A dictionary containing the prepared inputs for generation,
                with the following key-value pairs:

                - 'input_ids' (None): The input token indices for generation (set to None).
                - 'encoder_outputs' (torch.Tensor): The output tensor from the encoder.
                - 'past_key_values' (Tuple[torch.Tensor]): The cached key-value pairs of the decoder's self-attention layers.
                - 'decoder_input_ids' (torch.Tensor): The modified decoder input sequence of token indices.
                - 'attention_mask' (torch.Tensor): The attention mask tensor indicating which tokens to attend to
                and which ones to ignore.
                - 'head_mask' (torch.Tensor): The mask tensor to nullify selected heads of the self-attention layers
                in the decoder.
                - 'decoder_head_mask' (torch.Tensor): The mask tensor to nullify selected heads of the self-attention
                layers in the decoder's cross-attention.
                - 'cross_attn_head_mask' (torch.Tensor): The mask tensor to nullify selected heads of the cross-attention
                layers in the decoder.
                - 'use_cache' (bool): Whether to use the cache for the decoder's self-attention layers.

        Raises:
            None.
        """
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            # 不一致
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: Tensor):
        """prepare_decoder_input_ids_from_labels"""
        return shift_tokens_right(labels, self.config.pad_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """reorder_cache"""
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


class MBartForSequenceClassification(MBartPreTrainedModel):
    """MBartForSequenceClassification"""
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]

    def __init__(self, config: MBartConfig):
        """
        Initializes an instance of the MBartForSequenceClassification class.

        Args:
            self (MBartForSequenceClassification): The instance of the MBartForSequenceClassification class.
            config (MBartConfig): The configuration object for the MBart model, specifying various model hyperparameters.
                It must be an instance of MBartConfig class.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not an instance of MBartConfig.
        """
        super().__init__(config)
        self.model = MBartModel(config)
        self.classification_head = MBartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        """
        Constructs the MBart model for sequence classification.

        Args:
            self (MBartForSequenceClassification): The instance of the MBartForSequenceClassification class.
            input_ids (Tensor, optional): The input sequence tokens. Default: None.
            attention_mask (Optional[Tensor], optional): The attention mask for the input sequence. Default: None.
            decoder_input_ids (Optional[Tensor], optional): The decoder input sequence tokens. Default: None.
            decoder_attention_mask (Optional[Tensor], optional): The attention mask for the decoder input sequence.
                Default: None.
            head_mask (Optional[Tensor], optional): The mask for hiding heads of the encoder layers. Default: None.
            decoder_head_mask (Optional[Tensor], optional): The mask for hiding heads of the decoder layers.
                Default: None.
            cross_attn_head_mask (Optional[Tensor], optional): The mask for hiding heads of the cross-attention layers.
                Default: None.
            encoder_outputs (Optional[List[Tensor]], optional): The outputs of the encoder layers. Default: None.
            inputs_embeds (Optional[Tensor], optional): The embedded input sequence tokens. Default: None.
            decoder_inputs_embeds (Optional[Tensor], optional): The embedded decoder input sequence tokens. Default: None.
            labels (Optional[Tensor], optional): The labels for the input sequence tokens. Default: None.
            use_cache (Optional[bool], optional): Whether to use cache. Default: None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Default: None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Default: None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Default: None.

        Returns:
            Union[Tuple, Seq2SeqSequenceClassifierOutput]: The output of the method, which can be a tuple of various
                values or an instance of Seq2SeqSequenceClassifierOutput class.

        Raises:
            NotImplementedError: If input embeddings are passed, which is currently not supported.
            ValueError: If all examples do not have the same number of <eos> tokens.

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

        eos_mask = ops.equal(input_ids, self.config.eos_token_id)

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
                elif self.config.num_labels > 1 and (labels.dtype in (mindspore.int64, mindspore.int32)):
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


class MBartForQuestionAnswering(MBartPreTrainedModel):
    """MBartForQuestionAnswering"""
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the MBartForQuestionAnswering class.

        Args:
            self: The instance of the class.
            config (MBartConfig): The configuration object for the MBart model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = MBartModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[List[Tensor]] = None,
            start_positions: Optional[Tensor] = None,
            end_positions: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqQuestionAnsweringModelOutput]:
        """
        Args:
            self: The object instance.
            input_ids (Tensor, optional): The input token IDs. Default is None.
            attention_mask (Optional[Tensor], optional): The attention mask for the input sequence. Default is None.
            decoder_input_ids (Optional[Tensor], optional): The decoder input token IDs. Default is None.
            decoder_attention_mask (Optional[Tensor], optional): The attention mask for the decoder input sequence.
                Default is None.
            head_mask (Optional[Tensor], optional): The head mask for the model. Default is None.
            decoder_head_mask (Optional[Tensor], optional): The decoder head mask for the model. Default is None.
            cross_attn_head_mask (Optional[Tensor], optional): The cross-attention head mask for the model.
                Default is None.
            encoder_outputs (Optional[List[Tensor]], optional): The outputs of the encoder. Default is None.
            start_positions (Optional[Tensor], optional): The start positions for training. Default is None.
            end_positions (Optional[Tensor], optional): The end positions for training. Default is None.
            inputs_embeds (Optional[Tensor], optional): The embedded inputs. Default is None.
            decoder_inputs_embeds (Optional[Tensor], optional): The embedded decoder inputs. Default is None.
            use_cache (Optional[bool], optional): Whether to use cached values. Default is None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Default is None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, Seq2SeqQuestionAnsweringModelOutput]: The model output, including the loss, start logits,
                end logits, past key values, decoder hidden states, decoder attentions, cross attentions,
                encoder last hidden state, encoder hidden states, and encoder attentions.

        Raises:
            None
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

            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
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


class MBartDecoderWrapper(MBartPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MBartDecoderWrapper class.

        Args:
            self: The instance of the class.
            config (object): The configuration object for the MBartDecoderWrapper.
                It contains the necessary settings and parameters for initializing the wrapper.
                The config object must be an instance of the MBartConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.decoder = MBartDecoder(config)

    def forward(self, *args, **kwargs):
        """
        Constructs a new instance of the MBartDecoderWrapper class.

        Args:
            self: The current instance of the MBartDecoderWrapper class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method is used to forward a new instance of the MBartDecoderWrapper class. It takes no additional
            parameters other than self, which is automatically passed to the method. The method initializes
            the instance by calling the decoder method with the provided arguments and keyword arguments.

        Note that this method does not return any value. It is used solely for initialization purposes.

        Example:
            ```python
            >>> wrapper = MBartDecoderWrapper()
            >>> wrapper.forward()
            ```
        """
        return self.decoder(*args, **kwargs)


class MBartForCausalLM(MBartPreTrainedModel):
    """MBartForCausalLM"""
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the 'MBartForCausalLM' class.

        Args:
            self: The current object instance.
            config (object): The configuration object for the model.
                It must have the following attributes:

                - is_decoder (bool): Specifies whether the model is a decoder or not. Set to True for decoder models.
                - is_encoder_decoder (bool): Specifies whether the model is an encoder-decoder or not.
                Set to False for decoder models.
                - hidden_size (int): The size of the hidden states.
                - vocab_size (int): The size of the vocabulary.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = MBartDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """get_input_embeddings"""
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """set_input_embeddings"""
        self.model.decoder.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        """get_output_embeddings"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """set_output_embeddings"""
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """set_decoder"""
        self.model.decoder = decoder

    def get_decoder(self):
        """get_decoder"""
        return self.model.decoder

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            encoder_hidden_states: Optional[Tensor] = None,
            encoder_attention_mask: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        '''
        Constructs the model for the MBartForCausalLM class.
        
        Args:
            self: The instance of the class.
            input_ids (Tensor, optional): The input token IDs. Default: None.
            attention_mask (Tensor, optional): The attention mask tensor. Default: None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor. Default: None.
            encoder_attention_mask (Tensor, optional): The encoder attention mask tensor. Default: None.
            head_mask (Tensor, optional): The head mask tensor. Default: None.
            cross_attn_head_mask (Tensor, optional): The cross attention head mask tensor. Default: None.
            past_key_values (List[Tensor], optional): The past key values tensor. Default: None.
            inputs_embeds (Tensor, optional): The embedded inputs tensor. Default: None.
            labels (Tensor, optional): The labels tensor. Default: None.
            use_cache (bool, optional): Whether to use cache. Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.
            return_dict (bool, optional): Whether to return a dictionary. Default: None.
        
        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]: The output of the model, which can be a tuple or an
                instance of CausalLMOutputWithCrossAttentions.
            
        Raises:
            None.
        '''
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
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

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
        This method prepares inputs for generation in the MBartForCausalLM class.
        
        Args:
            self (object): The instance of the class.
            input_ids (tensor): The input tensor containing the token ids.
            past_key_values (tuple, optional): A tuple of past key values for faster decoding.
            attention_mask (tensor, optional): A tensor specifying which elements in the input_ids should be attended to.
            use_cache (bool, optional): A boolean indicating whether to use cache for faster decoding.
        
        Returns:
            dict: A dictionary containing the prepared inputs for generation including 'input_ids', 'attention_mask',
                'past_key_values', and 'use_cache'.
        
        Raises:
            ValueError: If the input_ids and past_key_values are not compatible.
            IndexError: If the input_ids shape is invalid.
        """
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = ops.ones_like(input_ids)

        if past_key_values:
            # input_ids = input_ids[:, -1:]
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
        Reorders the cache for the specified beam index.
        
        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer.
                Each element in the tuple is a tensor representing the past states for a particular layer.
            beam_idx (tensor): A tensor containing the indices of the selected beams.
        
        Returns:
            tuple: A tuple containing the reordered past key-value states for each layer.
                Each element in the tuple is a tensor representing the reordered past states for a particular layer.
        
        Raises:
            None.
        
        Note:
            This method is a static method and should be accessed using the class name 'MBartForCausalLM'.
        
        Example:
            ```python
            >>> past_key_values = (tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]]))
            >>> beam_idx = tensor([2, 0, 1])
            >>> reordered_past = MBartForCausalLM._reorder_cache(past_key_values, beam_idx)
            >>> # Output: (tensor([[7, 8, 9], [1, 2, 3], [4, 5, 6]]), tensor([[16, 17, 18], [10, 11, 12], [13, 14, 15]]))
            ```
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


__all__ = [
    "MBART_PRETRAINED_MODEL_ARCHIVE_LIST",
    'MBartForCausalLM',
    'MBartForConditionalGeneration',
    'MBartForQuestionAnswering',
    'MBartForSequenceClassification',
    'MBartModel',
    'MBartPreTrainedModel',
]
