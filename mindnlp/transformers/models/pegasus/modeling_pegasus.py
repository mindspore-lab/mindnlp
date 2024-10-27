# coding=utf-8
# Copyright 2021, Google and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 Huawei Technologies Co., Ltd
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
""" MindSpore PEGASUS model."""

import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F, Parameter
from mindnlp.utils import logging

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_pegasus import PegasusConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/pegasus-large"
_CONFIG_FOR_DOC = "PegasusConfig"


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->Pegasus
class PegasusSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""
    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        """
        Initialize the PegasusSinusoidalPositionalEmbedding object.
        
        Args:
            self (PegasusSinusoidalPositionalEmbedding): The instance of the class itself.
            num_positions (int): The number of positions to encode.
            embedding_dim (int): The dimension of the embedding vector.
            padding_idx (Optional[int], optional): The index used for padding. Default is None.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: Parameter) -> Parameter:
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        out_np = out.asnumpy()
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out_np[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
        out_np[:, sentinel:] = np.cos(position_enc[:, 1::2])
        out.assign_value(Tensor(out_np, out.dtype))
        return out

    def forward(self, input_ids_shape, past_key_values_length: int = 0) -> mindspore.Tensor:
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        )
        return super().forward(positions)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Pegasus
class PegasusAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[PegasusConfig] = None,
    ):
        """
        Initializes the PegasusAttention class.
        
        Args:
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Default is 0.0.
            is_decoder (bool, optional): Whether the attention is used in a decoder setting. Default is False.
            bias (bool, optional): Whether to use bias in linear projections. Default is True.
            is_causal (bool, optional): Whether the attention is causal. Default is False.
            config (Optional[PegasusConfig], optional): An optional Pegasus configuration object. Default is None.
        
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
        Method to reshape a tensor for Pegasus attention mechanism.
        
        Args:
            self (PegasusAttention): An instance of the PegasusAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.
        
        Returns:
            None: The method modifies the shape of the input tensor and returns None.
        
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


PEGASUS_ATTENTION_CLASSES = {"eager": PegasusAttention}


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Pegasus, MBART->PEGASUS
class PegasusEncoderLayer(nn.Module):

    '''
    The PegasusEncoderLayer class represents a single layer of the Pegasus encoder.
    This layer includes self-attention, feed-forward neural network (FFN) processing, and layer normalization.
    
    This class inherits from nn.Module and has the following attributes:

    - embed_dim: The dimension of the input embeddings
    - self_attn: The self-attention mechanism used in the layer
    - self_attn_layer_norm: The layer normalization applied after self-attention
    - dropout: The dropout rate applied during processing
    - activation_fn: The activation function used in the feed-forward neural network
    - activation_dropout: The dropout rate applied after the activation function
    - fc1: The first fully connected layer in the feed-forward neural network
    - fc2: The second fully connected layer in the feed-forward neural network
    - final_layer_norm: The layer normalization applied after the feed-forward neural network processing

    The PegasusEncoderLayer class has a forward method that takes the following arguments:

    - hidden_states: Input to the layer of shape `(batch, seq_len, embed_dim)`
    - attention_mask: Attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated
    by very large negative values
    - layer_head_mask: Mask for attention heads in a given layer of size `(encoder_attention_heads,)`
    - output_attentions: Whether or not to return the attentions tensors of all attention layers

    The forward method returns the following outputs:

    - hidden_states: The processed hidden states
    - attn_weights: The attention weights if output_attentions is set to True
    '''
    def __init__(self, config: PegasusConfig):
        """
        Initialize a PegasusEncoderLayer object.

        Args:
            self (PegasusEncoderLayer): The instance of the PegasusEncoderLayer class.
            config (PegasusConfig):
                The configuration object containing parameters for initializing the encoder layer.

                - Type: PegasusConfig
                - Purpose: Specifies the configuration settings for the encoder layer.
                - Restrictions: Must be an instance of the PegasusConfig class.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_head_mask: mindspore.Tensor,
        output_attentions: bool = False,
    ) -> mindspore.Tensor:
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
            clamp_value = ops.finfo(hidden_states.dtype).max
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Pegasus, MBART->PEGASUS
class PegasusDecoderLayer(nn.Module):

    """
    The PegasusDecoderLayer class represents a single layer of the Pegasus decoder model.
    It includes self-attention and encoder-decoder cross-attention mechanisms followed by feedforward
    neural network layers. This class inherits from nn.Module and implements the decoding logic for the Pegasus model.

    Attributes:
        embed_dim (int): The dimension of the embeddings used in the layer.
        self_attn (PegasusAttention): The self-attention mechanism used in the layer.
        dropout (float): The dropout probability applied in the layer.
        activation_fn (function): The activation function used in the feedforward neural network layers.
        activation_dropout (float): The dropout probability applied after the activation function.
        self_attn_layer_norm (LayerNorm): Layer normalization applied after self-attention.
        encoder_attn (PegasusAttention): The encoder-decoder cross-attention mechanism used in the layer.
        encoder_attn_layer_norm (LayerNorm): Layer normalization applied after encoder-decoder cross-attention.
        fc1 (Dense): The first feedforward neural network layer.
        fc2 (Dense): The second feedforward neural network layer.
        final_layer_norm (LayerNorm): Layer normalization applied at the end of the layer.

    Methods:
        forward:
            Constructs the output of the layer based on the input hidden states and optional arguments.
            Returns the output tensor.

    Args:
        hidden_states (Tensor): Input to the layer of shape (batch, seq_len, embed_dim).
        attention_mask (Tensor): Attention mask of size (batch, 1, tgt_len, src_len) with padding indicated by
            large negative values.
        encoder_hidden_states (Tensor): Encoder input to the layer of shape (batch, seq_len, embed_dim).
        encoder_attention_mask (Tensor): Encoder attention mask of size (batch, 1, tgt_len, src_len) with padding
            indicated by large negative values.
        layer_head_mask (Tensor): Mask for attention heads in a given layer.
        cross_attn_layer_head_mask (Tensor): Mask for cross-attention heads in a given layer.
        past_key_value (Tuple(Tensor)): Cached past key and value projection states.
        output_attentions (bool): Flag to determine whether to return attention tensors.
        use_cache (bool): Flag to determine whether to use caching mechanism for key-value states.

    Returns:
        outputs (Tuple): Tuple containing the output tensor and optionally self-attention and cross-attention weights
            if output_attentions is True, and present key-value states if use_cache is True.
    """
    def __init__(self, config: PegasusConfig):
        """
        Initializes an instance of the PegasusDecoderLayer class.

        Args:
            self (PegasusDecoderLayer): The current instance of the class.
            config (PegasusConfig): The configuration object containing various settings for the decoder layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
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

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.encoder_attn = PEGASUS_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

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


class PegasusPreTrainedModel(PreTrainedModel):

    """
    PegasusPreTrainedModel represents a pre-trained model for Pegasus, inheriting from PreTrainedModel.

    This class provides methods for initializing weights, including handling different types of cells such as Dense,
    PegasusSinusoidalPositionalEmbedding, and Embedding. The _init_weights method sets the data for weights based on
    the specified standard deviation and initializes biases or padding indices as needed.

    For further details on the implementation and usage of this class, please refer to the corresponding code documentation.
    """
    config_class = PegasusConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, PegasusSinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0


class PegasusEncoder(PegasusPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`PegasusEncoderLayer`].

    Args:
        config: PegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        '''
        Initializes a PegasusEncoder object.

        Args:
            self: The PegasusEncoder object itself.
            config (PegasusConfig): An instance of PegasusConfig containing the configuration settings for
                the Pegasus model.
            embed_tokens (Optional[nn.Embedding]): An optional instance of nn.Embedding representing the
                token embeddings.

        Returns:
            None.

        Raises:
            None.
        '''
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([PegasusEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size
                will add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
            self.padding_idx,
        )

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos

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
            to_drop = False
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
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


class PegasusDecoder(PegasusPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`PegasusDecoderLayer`]

    Args:
        config: PegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: PegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes a PegasusDecoder instance.

        Args:
            self: The object itself.
            config (PegasusConfig): An instance of PegasusConfig containing configuration parameters.
            embed_tokens (Optional[nn.Embedding]): An optional instance of nn.Embedding representing embeddings.
                Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([PegasusDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the PegasusDecoder.

        Args:
            self (PegasusDecoder): The instance of the PegasusDecoder class.

        Returns:
            embed_tokens: This method returns the input embeddings stored in the 'embed_tokens' attribute of
                the PegasusDecoder instance.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the PegasusDecoder.

        Args:
            self (PegasusDecoder): The instance of the PegasusDecoder class.
            value: The input embeddings to be set for the decoder.
                It should be of type torch.Tensor and represent the embeddings for the input tokens.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        logger.info(f"Setting `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        self.embed_positions = PegasusSinusoidalPositionalEmbedding(
            self.config.max_position_embeddings,
            self.config.d_model,
            self.padding_idx,
        )

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.embed_positions

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
                Mask to nullify selected heads of the cross-attention modules in decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
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
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
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
        positions = self.embed_positions(input_shape, past_key_values_length)
        hidden_states = inputs_embeds + positions

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
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
                        f" {head_mask.shape[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
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


class PegasusModel(PegasusPreTrainedModel):

    """
    The `PegasusModel` class is a PyTorch-based model implementation for Pegasus, a pre-trained model for abstractive
    text summarization. It is a subclass of `PegasusPreTrainedModel`, which provides common methods and attributes
    for all Pegasus model variants.

    The `PegasusModel` class has the following methods:

    - `__init__`: Initializes a `PegasusModel` instance with the given configuration.
    - `get_input_embeddings`: Returns the shared input embeddings.
    - `set_input_embeddings`: Sets the shared input embeddings to the given value.
    - `get_encoder`: Returns the Pegasus encoder module.
    - `get_decoder`: Returns the Pegasus decoder module.
    - `resize_position_embeddings`: Resizes the position embeddings matrix of  the model if the new number of position
    embeddings is different from the maximum position embeddings defined in the configuration.
    - `get_position_embeddings(self) -> Tuple[nn.Embedding]`: Returns the position embeddings matrix used by the encoder
    and decoder.
    - `forward: Constructs the Pegasus model by encoding the input and decoding it with the provided decoder inputs
    and attention masks.

    The `PegasusModel` class provides an example in its docstring to demonstrate how to use the model for text
    summarization.

    Please refer to the Pegasus documentation for more details on the model architecture and usage examples.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: PegasusConfig):
        """
        Initialize the PegasusModel with the provided configuration.

        Args:
            self: The instance of the PegasusModel class.
            config (PegasusConfig): The configuration object containing model hyperparameters and settings.
                It should be an instance of PegasusConfig class.
                This parameter is required to initialize the PegasusModel.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = PegasusEncoder(config, self.shared)
        self.decoder = PegasusDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the PegasusModel instance.

        Args:
            self: The PegasusModel instance.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the PegasusModel.

        Args:
            self (PegasusModel): The instance of the PegasusModel class.
            value (torch.Tensor): The input embeddings to be set.
                It should be a torch.Tensor representing the shared input embeddings.

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
        This method returns the encoder associated with the PegasusModel.

        Args:
            self (PegasusModel): An instance of the PegasusModel class.
                Represents the current PegasusModel object.

        Returns:
            encoder:
                The method returns the encoder associated with the PegasusModel.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        Method to retrieve the decoder attribute of a PegasusModel instance.

        Args:
            self (PegasusModel): The instance of the PegasusModel class from which to retrieve the decoder.
                This parameter is required for accessing the decoder attribute.
                It must be an instance of the PegasusModel class.

        Returns:
            decoder: This method returns the decoder attribute of the PegasusModel instance.
                The decoder attribute is of type None and represents the decoder component of the model.

        Raises:
            None
        """
        return self.decoder

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.encoder.get_position_embeddings(), self.decoder.get_position_embeddings())

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[mindspore.Tensor]] = None,
        past_key_values: Optional[Tuple[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        r"""

        Returns:
            `Union[Tuple, Seq2SeqModelOutput]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, PegasusModel
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
            >>> model = PegasusModel.from_pretrained("google/pegasus-large")
            ...
            >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="ms")
            >>> decoder_inputs = tokenizer("Studies show that", return_tensors="ms")
            >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)
            ...
            >>> last_hidden_states = outputs.last_hidden_state
            >>> list(last_hidden_states.shape)
            [1, 4, 1024]
            ```
        """
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


class PegasusForConditionalGeneration(PegasusPreTrainedModel):

    """
    This class represents a Pegasus model for conditional generation. It is a subclass of the PegasusPreTrainedModel.

    PegasusForConditionalGeneration provides methods to initialize the model, resize the token and position embeddings,
    retrieve the encoder and decoder, get the output embeddings, set the output embeddings, and prepare inputs for
    generation.

    Methods:
        __init__: Initializes the PegasusForConditionalGeneration instance.
        get_encoder: Retrieves the encoder of the Pegasus model.
        get_decoder: Retrieves the decoder of the Pegasus model.
        resize_token_embeddings: Resizes the token embeddings.
        _resize_final_logits_bias: Resizes the final logits bias.
        get_output_embeddings: Retrieves the output embeddings of the Pegasus model.
        set_output_embeddings: Sets the output embeddings of the Pegasus model.
        resize_position_embeddings: Resizes the position embeddings matrix of the model.
        get_position_embeddings: Retrieves the position embeddings matrix.
        forward: Constructs the Pegasus model.
        prepare_inputs_for_generation: Prepares inputs for generation.
        prepare_decoder_input_ids_from_labels: Prepares decoder input ids from labels.
        _reorder_cache: Reorders the past key values for beam search.

    Please refer to the individual methods for more detailed information.
    """
    base_model_prefix = "model"
    # _keys_to_ignore_on_load_unexpected = ["final_logits_bias"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: PegasusConfig):
        """
        Initializes an instance of the 'PegasusForConditionalGeneration' class.

        Args:
            self: The instance of the class.
            config (PegasusConfig): The configuration object for Pegasus model.
                It contains various hyperparameters and settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model = PegasusModel(config)
        self.final_logits_bias = ops.zeros((1, self.model.shared.num_embeddings))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        This method returns the encoder from the Pegasus model for conditional generation.

        Args:
            self: An instance of the PegasusForConditionalGeneration class.

        Returns:
            encoder: The method returns the encoder from the Pegasus model for conditional generation.

        Raises:
            This method does not raise any exceptions.
        """
        return self.model.get_encoder()

    def get_decoder(self):
        """
        Returns the decoder of the PegasusForConditionalGeneration model.

        Args:
            self: An instance of the PegasusForConditionalGeneration class.

        Returns:
            decoder: The method returns the decoder of the PegasusForConditionalGeneration model.

        Raises:
            None.
        """
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        """
        Resize the token embeddings of the Pegasus model for conditional generation to accommodate a new number of tokens.

        Args:
            self (PegasusForConditionalGeneration): The instance of the Pegasus model class.
            new_num_tokens (int): The desired new number of tokens for the token embeddings.
            pad_to_multiple_of (Optional[int], optional): The optional value to pad the number of tokens to
                a multiple of. Defaults to None.

        Returns:
            nn.Embedding: The resized token embeddings as an instance of the nn.Embedding class.

        Raises:
            None: This method does not raise any exceptions.

        This method resizes the token embeddings of the Pegasus model for conditional generation to the specified
        new number of tokens. It uses the super() function to call the parent class's resize_token_embeddings() method
        and obtains the new embeddings. Then, the method calls the _resize_final_logits_bias() method to adjust the
        final logits bias based on the new embeddings' weight shape.
        Finally, it returns the resized token embeddings as an instance of the nn.Embedding class.
        """
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        """
        Resizes the final logits bias tensor in the PegasusForConditionalGeneration class.

        Args:
            self (PegasusForConditionalGeneration): An instance of the PegasusForConditionalGeneration class.
            new_num_tokens (int): The desired number of tokens for the resized bias tensor.

        Returns:
            None: This method modifies the self.final_logits_bias attribute directly.

        Raises:
            None.

        This method resizes the final_logits_bias tensor in the PegasusForConditionalGeneration class based on
        the specified new_num_tokens. If new_num_tokens is less than or equal to the current number of tokens in
        self.final_logits_bias, a new_bias tensor is created by slicing the original tensor. Otherwise, extra_bias
        tensor is created with additional columns (new_num_tokens - old_num_tokens) and concatenated with the original
        tensor along the column axis. Finally, the self.final_logits_bias attribute is updated with the new_bias tensor.
        """
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.cat([self.final_logits_bias, extra_bias], dim=1)
        self.final_logits_bias = new_bias

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the PegasusForConditionalGeneration model.

        Args:
            self (PegasusForConditionalGeneration): The instance of the PegasusForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the Pegasus model.

        Args:
            self (PegasusForConditionalGeneration): The instance of the PegasusForConditionalGeneration class.
            new_embeddings (tensor): The new output embeddings to be set for the model.
                It should be a tensor of the same shape as the current output embeddings.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of type tensor.
            ValueError: If the shape of the new_embeddings tensor does not match the shape of the current
                output embeddings.
        """
        self.lm_head = new_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.model.encoder.resize_position_embeddings(new_num_position_embeddings)
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    def get_position_embeddings(self) -> Tuple[nn.Embedding]:
        """
        Returns the position embeddings matrix
        """
        return (self.model.encoder.get_position_embeddings(), self.model.decoder.get_position_embeddings())

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Tuple[mindspore.Tensor]] = None,
        past_key_values: Optional[Tuple[mindspore.Tensor]] = None,
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = F.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1).astype(mindspore.int32))

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
        '''
        Prepare inputs for generation.

        This method prepares the inputs for the generation process in the PegasusForConditionalGeneration class.

        Args:
            self: The instance of the class.
            decoder_input_ids (Tensor): The input tensor containing the decoder input IDs.
            past_key_values (tuple, optional): The tuple of past key values. Default is None.
            attention_mask (Tensor, optional): The attention mask tensor. Default is None.
            head_mask (Tensor, optional): The head mask tensor. Default is None.
            decoder_head_mask (Tensor, optional): The decoder head mask tensor. Default is None.
            cross_attn_head_mask (Tensor, optional): The cross-attention head mask tensor. Default is None.
            use_cache (bool, optional): Indicates whether to use cache. Default is None.
            encoder_outputs (tuple, optional): The tuple of encoder outputs. Default is None.

        Returns:
            dict: A dictionary containing the prepared inputs for generation.
                The dictionary contains the following keys:

                - 'input_ids': None
                - 'encoder_outputs': The encoder outputs
                - 'past_key_values': The past key values
                - 'decoder_input_ids': The decoder input IDs
                - 'attention_mask': The attention mask
                - 'head_mask': The head mask
                - 'decoder_head_mask': The decoder head mask
                - 'cross_attn_head_mask': The cross-attention head mask
                - 'use_cache': The flag indicating whether to use cache

        Raises:
            None.
        '''
        # cut decoder_input_ids if past is used
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
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare_decoder_input_ids_from_labels method in the PegasusForConditionalGeneration class.

        Args:
            self (PegasusForConditionalGeneration): The instance of the PegasusForConditionalGeneration class.
            labels (mindspore.Tensor): The input labels for the decoder.
                It is of type mindspore.Tensor and contains the token ids for the decoder input.

        Returns:
            None.

        Raises:
            None.
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the PegasusForConditionalGeneration class.

        Args:
            past_key_values (tuple): A tuple containing the past key values to be reordered.
            beam_idx (Tensor): The tensor representing the beam index.

        Returns:
            tuple: The reordered past key values.

        Raises:
            None.

        This static method reorders the cache for the PegasusForConditionalGeneration class.

        Args:
            past_key_values: A tuple containing the past key values.
                It is used to store the previous key-value states for each layer.
            The method reorders the past key values based on the beam index provided.
            beam_idx: A tensor representing the beam index.
                It is used to specify the index of the beam to be reordered.

        The method returns the reordered past key values as a tuple.
        The reordered past key values maintain the same structure as the input tuple, but the values are reordered
        based on the beam index.

        This method does not raise any exceptions.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


# Copied from transformers.models.bart.modeling_bart.BartDecoderWrapper with Bart->Pegasus
class PegasusDecoderWrapper(PegasusPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config):
        """
        Initializes an instance of the PegasusDecoderWrapper class.

        Args:
            self (PegasusDecoderWrapper): The instance of the class itself.
            config: The configuration object containing the necessary parameters for initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.decoder = PegasusDecoder(config)

    def forward(self, *args, **kwargs):
        """
        Method 'forward' in the class 'PegasusDecoderWrapper'.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None: This method returns None.

        Raises:
            None.
        """
        return self.decoder(*args, **kwargs)


class PegasusForCausalLM(PegasusPreTrainedModel):

    """
    This class represents a Pegasus model for causal language modeling (LM). It is a subclass of PegasusPreTrainedModel,
    which provides the basic infrastructure for loading and saving pre-trained models.

    The PegasusForCausalLM class is designed for generating text in a causal manner, where each token is generated
    based on the previously generated tokens. It takes as input a sequence of tokens and predicts the probability
    distribution over the next token in the sequence.

    The PegasusForCausalLM class provides various methods for interacting with the model. These include initializing
    the model with a configuration, getting and setting input and output embeddings, getting and setting the decoder,
    getting the position embeddings, resizing the position embeddings, and forwarding the model for generation.

    The `__init__` method initializes the PegasusForCausalLM object with a configuration.
    It sets the decoder configuration and initializes the model and the LM head.

    The `get_input_embeddings` method returns the input embeddings of the model.

    The `set_input_embeddings` method sets the input embeddings of the model to a new value.

    The `get_output_embeddings` method returns the output embeddings (LM head) of the model.

    The `set_output_embeddings` method sets the output embeddings (LM head) of the model to a new value.

    The `set_decoder` method sets the decoder of the model to a new decoder.

    The `get_decoder` method returns the decoder of the model.

    The `get_position_embeddings` method returns the position embeddings matrix of the model.

    The `resize_position_embeddings` method resizes the position embeddings matrix of the model if the new number of
    position embeddings is different from the maximum number of position embeddings specified in the configuration.

    The `forward` method forwards the model for generation. It takes input tensors such as input_ids, attention_mask,
    encoder_hidden_states, and labels, and returns the model outputs, including the logits, loss, past key values,
    hidden states, attentions, and cross attentions.

    The `prepare_inputs_for_generation` method prepares the inputs for generation. It takes input tensors such as
    input_ids, past_key_values, and attention_mask, and returns a dictionary of prepared inputs.

    The `_reorder_cache` method reorders the past key values for generation based on the beam index.

    Note:
        This class inherits from PegasusPreTrainedModel and provides additional methods specific to causal LM tasks.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the PegasusForCausalLM class.

        Args:
            self (PegasusForCausalLM): The instance of the PegasusForCausalLM class.
            config (object): The configuration object containing settings for the model.
                This object is deep copied to avoid modification of the original configuration.
                It must have the following attributes:

                - is_decoder (bool): Set to True.
                - is_encoder_decoder (bool): Set to False.

        Returns:
            None.

        Raises:
            None.
        """
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = PegasusDecoderWrapper(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method retrieves the input embeddings from the PegasusForCausalLM model.

        Args:
            self: PegasusForCausalLM instance. Represents the current instance of the PegasusForCausalLM class.

        Returns:
            None

        Raises:
            None
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        set_input_embeddings method in the PegasusForCausalLM class sets the input embeddings for the model.

        Args:
            self (PegasusForCausalLM): The instance of the PegasusForCausalLM class.
            value (torch.Tensor): The input embeddings to be set for the model.
                It should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the PegasusForCausalLM model.

        Args:
            self (PegasusForCausalLM): The instance of the PegasusForCausalLM class.
                This parameter is a reference to the current instance of the class.

        Returns:
            None: This method returns None, as it retrieves the output embeddings from the model
                and does not return any specific value.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for PegasusForCausalLM model.

        Args:
            self (PegasusForCausalLM): The instance of the PegasusForCausalLM class.
            new_embeddings (object): The new embeddings to be set as output embeddings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder of the PegasusForCausalLM model.

        Args:
            self (PegasusForCausalLM): The instance of the PegasusForCausalLM class.
            decoder (object): The decoder object to be set for the model.

        Returns:
            None: This method modifies the decoder attribute of the PegasusForCausalLM instance.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        This method retrieves the decoder component of the PegasusForCausalLM model.

        Args:
            self: An instance of the PegasusForCausalLM class.

        Returns:
            decoder: The method returns the decoder component of the model.

        Raises:
            None.
        """
        return self.model.decoder

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings matrix
        """
        return self.model.decoder.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings matrix of the model if `new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embeddings.

                - If position embeddings are learned, increasing the size will add newly initialized vectors at the end,
                whereas reducing the size will remove vectors from the end.
                - If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size
                will add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.config.max_position_embeddings = new_num_position_embeddings
        self.model.decoder.resize_position_embeddings(new_num_position_embeddings)

    # Copied from transformers.models.bart.modeling_bart.BartForCausalLM.forward with Bart->Pegasus, facebook/bart-base->google/pegasus-large
    def forward(
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
            >>> from transformers import AutoTokenizer, PegasusForCausalLM
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
            >>> model = PegasusForCausalLM.from_pretrained("google/pegasus-large", add_cross_attention=False)
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="ms")
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
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1).astype(mindspore.int32))

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
        Prepare inputs for generation in the PegasusForCausalLM class.

        This method prepares inputs for generating text by adjusting input_ids and attention_mask based on
        past_key_values if provided.

        Args:
            self (PegasusForCausalLM): The instance of the PegasusForCausalLM class.
            input_ids (torch.Tensor): The input tensor containing token ids for the model.
            past_key_values (tuple, optional): Tuple of past key values for faster generation, if available.
            attention_mask (torch.Tensor, optional): Tensor indicating which tokens should be attended to.
            use_cache (bool, optional): Flag indicating whether to use cache for faster decoding.

        Returns:
            dict:
                A dictionary containing the following keys:

                - input_ids (torch.Tensor): The adjusted input tensor after processing.
                - attention_mask (torch.Tensor): The attention mask for the input tensor.
                - past_key_values (tuple): Past key values if provided, else None.
                - use_cache (bool): Flag indicating whether to use cache for faster decoding.

        Raises:
            ValueError: If the input_ids and past_key_values shapes are incompatible.
            IndexError: If the input_ids shape is invalid for processing.
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
        Reorders the cache for beam search in the PegasusForCausalLM class.

        Args:
            past_key_values (tuple): A tuple of past key-values containing cached states for each layer.
            beam_idx (torch.Tensor): A tensor representing the indices of the selected beams.

        Returns:
            tuple: A tuple of reordered past key-values for each layer.

        Raises:
            None.

        This static method reorders the cache for beam search in the PegasusForCausalLM class. It takes two parameters:

        - `past_key_values`: A tuple of past key-values which contains the cached states for each layer.
        This is used to keep track of the previous states.
        - `beam_idx`: A tensor representing the indices of the selected beams. This tensor is used to select the
        states corresponding to the selected beams.
        
        The method returns a tuple of reordered past key-values for each layer.
        This reordering is done by selecting the states in each layer's past key-values tensor based on the beam
        indices provided.
        
        The method does not raise any exceptions.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


__all__ = [
    "PegasusModel",
    "PegasusPreTrainedModel",
    "PegasusForConditionalGeneration",
    "PegasusForCausalLM",
]
