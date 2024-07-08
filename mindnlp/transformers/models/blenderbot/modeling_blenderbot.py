# coding=utf-8
# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore Blenderbot model."""
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.common.initializer import initializer, Normal

from ....modules.functional import finfo
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ....utils import logging
from ...modeling_utils import PreTrainedModel
from .configuration_blenderbot import BlenderbotConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BlenderbotConfig"
_CHECKPOINT_FOR_DOC = "facebook/blenderbot-400M-distill"


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
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


class BlenderbotLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def construct(self, input_ids_shape, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        )
        return super().construct(positions)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Blenderbot
class BlenderbotAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BlenderbotConfig] = None,
    ):
        """
        Initializes the BlenderbotAttention class.
        
        Args:
            self: The instance of the class.
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Default is 0.0.
            is_decoder (bool, optional): Indicates if the attention mechanism is used as a decoder. Default is False.
            bias (bool, optional): Indicates whether the linear layers have bias. Default is True.
            is_causal (bool, optional): Indicates if the attention is causal. Default is False.
            config (Optional[BlenderbotConfig], optional): The configuration for the Blenderbot model. Default is None.
        
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

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor according to the specified dimensions.
        
        Args:
            self (BlenderbotAttention): An instance of the BlenderbotAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence in the input tensor.
            bsz (int): The batch size of the input tensor.
        
        Returns:
            None: The method modifies the input tensor in-place.
        
        Raises:
            None.
        
        This method reshapes the input tensor using the provided dimensions.
        The tensor is reshaped into a new shape where the batch size is `bsz`, the sequence length is `seq_len`,
        and the number of heads is `self.num_heads`.
        The reshaping operation is performed by calling the `view` method of the input tensor, which returns a new
        view of the tensor with the specified dimensions.
        Additionally, the `swapaxes` method is called on the reshaped tensor to swap the dimensions at index 1 and index 2.

        Note that this method modifies the input tensor in-place and does not return any value.
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


BLENDERBOT_ATTENTION_CLASSES = {"eager": BlenderbotAttention}


# Copied from transformers.models.mbart.modeling_mbart.MBartEncoderLayer with MBart->Blenderbot, MBART->BLENDERBOT
class BlenderbotEncoderLayer(nn.Cell):

    """
    This class represents a single layer of the BlenderbotEncoder.
    It is responsible for processing the input hidden states and applying self-attention,
    feed-forward neural network (FFN) layers, and layer normalization.

    The BlenderbotEncoderLayer class inherits from the nn.Cell class.

    Attributes:
        embed_dim (int): The dimension of the input hidden states.
        self_attn (nn.Layer): The self-attention layer used to capture dependencies between different positions
            within the input hidden states.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization applied to the output of the self-attention layer.
        dropout (float): The dropout probability applied to the output of the self-attention layer.
        activation_fn (function): The activation function applied to the output of the FFN layers.
        activation_dropout (float): The dropout probability applied to the output of the activation function.
        fc1 (nn.Dense): The first linear transformation layer in the FFN.
        fc2 (nn.Dense): The second linear transformation layer in the FFN.
        final_layer_norm (nn.LayerNorm): The layer normalization applied to the output of the FFN layers.
    """
    def __init__(self, config: BlenderbotConfig):
        """
        Initializes a new instance of the BlenderbotEncoderLayer class.

        Args:
            self: The object itself.
            config (BlenderbotConfig):
                The configuration object for Blenderbot, which contains various settings for the encoder layer.

                - config.d_model (int): The embedding dimension.
                - config.encoder_attention_heads (int): The number of attention heads in the self-attention mechanism.
                - config.attention_dropout (float): The dropout probability for the attention weights.
                - config.dropout (float): The dropout probability for the output of each sub-layer.
                - config.activation_function (str): The type of activation function used in the feed-forward neural network.
                - config.activation_dropout (float): The dropout probability for the activation output.
                - config.encoder_ffn_dim (int): The dimension of the feed-forward neural network intermediate layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BLENDERBOT_ATTENTION_CLASSES["eager"](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def construct(
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
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == mindspore.float16 and (
            ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()
        ):
            clamp_value = finfo(hidden_states.dtype, 'max') - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer with MBart->Blenderbot, MBART->BLENDERBOT
class BlenderbotDecoderLayer(nn.Cell):

    """
    A BlenderbotDecoderLayer represents a single layer of the Blenderbot decoder model.
    It is used to decode the input sequence and generate the output sequence. This class inherits from nn.Cell and contains
    various components such as self-attention, encoder attention, feed-forward networks, and layer normalization.

    Attributes:
        embed_dim (int): The embedding dimension of the layer.
        self_attn (nn.Layer): The self-attention mechanism used in the layer.
        dropout (float): The dropout rate used in the layer.
        activation_fn (function): The activation function used in the feed-forward networks.
        activation_dropout (float): The dropout rate used in the activation function.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization applied after the self-attention.
        encoder_attn (nn.Layer): The encoder attention mechanism used in the layer.
        encoder_attn_layer_norm (nn.LayerNorm): The layer normalization applied after the encoder attention.
        fc1 (nn.Dense): The first feed-forward network.
        fc2 (nn.Dense): The second feed-forward network.
        final_layer_norm (nn.LayerNorm): The final layer normalization applied after the feed-forward networks.
    """
    def __init__(self, config: BlenderbotConfig):
        """
        Initialize a BlenderbotDecoderLayer object.

        Args:
            self (BlenderbotDecoderLayer): The instance of the BlenderbotDecoderLayer class.
            config (BlenderbotConfig):
                An object containing configuration settings for the decoder layer.

                - config.d_model (int): The embedding dimension to be used.
                - config.decoder_attention_heads (int): The number of attention heads for decoder self-attention.
                - config.attention_dropout (float): The dropout rate for attention weights.
                - config.activation_function (str): The name of the activation function to be used.
                - config.activation_dropout (float): The dropout rate for activation functions.
                - config.decoder_ffn_dim (int): The dimensionality of the feedforward network.

        Returns:
            None.

        Raises:
            ValueError: If any of the input parameters are invalid or missing.
            KeyError: If the provided activation function is not found in the predefined mapping.
            NotImplementedError: If the attention mechanism specified is not implemented in the BLENDERBOT_ATTENTION_CLASSES dictionary.
        """
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BLENDERBOT_ATTENTION_CLASSES["eager"](
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

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BLENDERBOT_ATTENTION_CLASSES["eager"](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Dense(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

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
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
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
            hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BlenderbotPreTrainedModel(PreTrainedModel):

    """
    BlenderbotPreTrainedModel is a Python class representing a pre-trained model for Blenderbot.
    This class inherits from PreTrainedModel and includes methods for initializing weights and providing dummy inputs.

    The _init_weights method initializes the weights of the model based on the specified standard deviation and cell type,
    ensuring proper initialization for both Dense and Embedding cells.

    The dummy_inputs method generates a set of dummy inputs for the model, including attention mask, input IDs,
    and decoder input IDs, with consideration for padding tokens.

    This class provides essential functionality for initializing model weights and generating dummy inputs,
    making it a crucial component for working with pre-trained Blenderbot models.
    """
    config_class = BlenderbotConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        std = self.config.init_std
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(std),
                                                    cell.weight.shape, cell.weight.dtype))
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
        This method generates dummy inputs for the BlenderbotPreTrainedModel.

        Args:
            self: The instance of the BlenderbotPreTrainedModel class.

        Returns:
            A dictionary containing dummy inputs in the following format:
                {
                    'attention_mask': A tensor representing the attention mask where pad tokens are masked,
                    'input_ids': A tensor representing the input IDs,
                    'decoder_input_ids': A tensor representing the decoder input IDs
                }

        Raises:
            None
        """
        pad_token = self.config.pad_token_id
        input_ids = mindspore.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_input_ids": input_ids,
        }
        return dummy_inputs


class BlenderbotEncoder(BlenderbotPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BlenderbotEncoderLayer`].

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes a BlenderbotEncoder instance.

        Args:
            self (BlenderbotEncoder): The instance of the BlenderbotEncoder class.
            config (BlenderbotConfig): An instance of BlenderbotConfig containing configuration settings.
            embed_tokens (Optional[nn.Embedding]): Optional parameter representing embedding tokens. Defaults to None.

        Returns:
            None.

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

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.CellList([BlenderbotEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
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
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

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

        # add final layer norm
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BlenderbotDecoder(BlenderbotPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BlenderbotDecoderLayer`]

    Args:
        config: BlenderbotConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BlenderbotConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes a new instance of the BlenderbotDecoder class.

        Args:
            self: The instance of the class.
            config (BlenderbotConfig): An object containing configuration settings for the BlenderbotDecoder.
            embed_tokens (Optional[nn.Embedding]): An optional embedding tensor. If provided, it will be used as the embedding tokens.
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

        self.embed_positions = BlenderbotLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.CellList([BlenderbotDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the BlenderbotDecoder.

        Args:
            self (BlenderbotDecoder): The instance of the BlenderbotDecoder class.
                This parameter is used to access the embed_tokens attribute.

        Returns:
            embed_tokens: This method returns the embed_tokens attribute, which represents the input embeddings.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set input embeddings for the BlenderbotDecoder class.

        Args:
            self (BlenderbotDecoder): An instance of the BlenderbotDecoder class.
                This parameter refers to the current instance of the class.
            value: The input embeddings value to be set for the instance.
                It can be of any valid data type and is used to set the embed_tokens attribute.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def construct(
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
            head_mask (`mindspore.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

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

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
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

        # add final layer norm
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


class BlenderbotModel(BlenderbotPreTrainedModel):

    """
        The `BlenderbotModel` class represents a model for generating responses in conversational AI systems.
        It is a subclass of `BlenderbotPreTrainedModel` and inherits its functionality.

        Args:
            config (BlenderbotConfig): The configuration class that contains the model's hyperparameters.

        Attributes:
            shared (nn.Embedding): The shared embedding layer used for both the encoder and decoder.
            encoder (BlenderbotEncoder): The encoder module of the model.
            decoder (BlenderbotDecoder): The decoder module of the model.

        Methods:
            __init__: Initializes the `BlenderbotModel` instance.
            get_input_embeddings: Retrieves the shared embedding layer.
            set_input_embeddings: Sets the shared embedding layer to a new value.
            get_encoder: Retrieves the encoder module.
            get_decoder: Retrieves the decoder module.
            construct: Constructs the model and performs the forward pass.

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]: The output of the forward pass,
                including the last hidden state, past key values, decoder hidden states, decoder attentions, cross attentions,
                encoder last hidden state, encoder hidden states, and encoder attentions.

        Example:
            ```python
            >>> from transformers import AutoTokenizer, BlenderbotModel
            ...
            >>> model = BlenderbotModel.from_pretrained("facebook/blenderbot-400M-distill")
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
            ...
            >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)
            ...
            >>> last_hidden_states = outputs.last_hidden_state
            >>> list(last_hidden_states.shape)
            [1, 6, 1280]
            ```
        """
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: BlenderbotConfig):
        """
        This method initializes a new instance of the BlenderbotModel class.

        Args:
            self: The instance of the BlenderbotModel class.
            config (BlenderbotConfig):
                An instance of the BlenderbotConfig class containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BlenderbotEncoder(config, self.shared)
        self.decoder = BlenderbotDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the BlenderbotModel.

        Args:
            self: BlenderbotModel instance. The instance of the BlenderbotModel class.

        Returns:
            None: This method returns the shared input embeddings.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BlenderbotModel.

        Args:
            self (BlenderbotModel): The instance of the BlenderbotModel class.
            value: The input embeddings to be set. It should be a tensor of shape (vocab_size, embeddings_dim).

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
        Returns the encoder used in the BlenderbotModel.

        Args:
            self (BlenderbotModel): An instance of the BlenderbotModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        This method returns the decoder used in the BlenderbotModel.

        Args:
            self: The instance of the BlenderbotModel class.

        Returns:
            None: This method returns the decoder used in the BlenderbotModel. It returns None if the decoder is not set.

        Raises:
            None.
        """
        return self.decoder

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqModelOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, BlenderbotModel
            ...
            >>> model = BlenderbotModel.from_pretrained("facebook/blenderbot-400M-distill")
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
            ...
            >>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_input_ids)
            ...
            >>> last_hidden_states = outputs.last_hidden_state
            >>> list(last_hidden_states.shape)
            [1, 6, 1280]
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


class BlenderbotForConditionalGeneration(BlenderbotPreTrainedModel):

    """
    A class for generating text using the Blenderbot model with conditional generation. This class inherits from BlenderbotPreTrainedModel and provides methods for preparing inputs for generation and
    reordering cache.

    Attributes:
        model (BlenderbotModel): A model instance of the BlenderbotModel class.
        final_logits_bias (mindspore.Tensor): A tensor representing the final logits bias.
        lm_head (mindspore.nn.Dense): A fully connected linear layer for the language modeling head.

    Methods:
        __init__: Initializes the class with a BlenderbotConfig instance.
        get_encoder: Returns the encoder from the model.
        get_decoder: Returns the decoder from the model.
        resize_token_embeddings: Resizes the token embeddings.
        _resize_final_logits_bias: Resizes the final logits bias.
        get_output_embeddings: Returns the output embeddings.
        set_output_embeddings: Sets the output embeddings.
        construct: Constructs the model for generation.
        prepare_inputs_for_generation: Prepares the inputs for generation.
        _reorder_cache: Reorders the cache.

    """
    base_model_prefix = "model"
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: BlenderbotConfig):
        """
        Initializes a new instance of the BlenderbotForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (BlenderbotConfig): An instance of the BlenderbotConfig class containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.model = BlenderbotModel(config)
        self.final_logits_bias = ops.zeros((1, self.model.shared.vocab_size))
        self.lm_head = nn.Dense(config.d_model, self.model.shared.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        This method returns the encoder of the BlenderbotForConditionalGeneration model.

        Args:
            self: The instance of the BlenderbotForConditionalGeneration class.

        Returns:
            None: This method returns the encoder of the model as an object of type 'None'.

        Raises:
            None
        """
        return self.model.get_encoder()

    def get_decoder(self):
        """
        Returns the decoder of the BlenderbotForConditionalGeneration model.

        Args:
            self: An instance of the BlenderbotForConditionalGeneration class.

        Returns:
            None: The method returns the decoder of the model, which is of type None.

        Raises:
            None.

        Note:
            The decoder is a component of the BlenderbotForConditionalGeneration model
            that is responsible for generating responses based on the input.

        Example:
            ```python
            >>> blenderbot = BlenderbotForConditionalGeneration()
            >>> decoder = blenderbot.get_decoder()
            >>> print(decoder)
            None
            ```
        """
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        """
        Resize the token embeddings of the Blenderbot model.

        Args:
            self (BlenderbotForConditionalGeneration): The instance of the BlenderbotForConditionalGeneration class.
            new_num_tokens (int): The desired number of tokens for the resized embeddings.
            pad_to_multiple_of (Optional[int], optional): If provided, the number of tokens will be padded to a multiple of this value. Defaults to None.

        Returns:
            nn.Embedding: The new resized token embeddings.

        Raises:
            None.
        """
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        """
        Resizes the final logits bias of the BlenderbotForConditionalGeneration model.

        Args:
            self (BlenderbotForConditionalGeneration): The instance of the BlenderbotForConditionalGeneration class.
            new_num_tokens (int): The desired number of tokens for the resized final logits bias.

        Returns:
            None: This method modifies the 'final_logits_bias' attribute of the BlenderbotForConditionalGeneration instance.

        Raises:
            None.

        Description:
            This method resizes the 'final_logits_bias' attribute of the BlenderbotForConditionalGeneration model.
            The 'final_logits_bias' is a tensor that represents the bias to be added to the final logits of the model.

            If the desired number of tokens, given by 'new_num_tokens', is less than or equal to the current number
            of tokens in the 'final_logits_bias', no resizing is performed.
            In this case, the 'final_logits_bias' is sliced to retain the desired number of tokens.

            If the desired number of tokens is greater than the current number of tokens,
            the 'final_logits_bias' is extended by appending zero-valued bias columns.
            The number of extra tokens is calculated as 'new_num_tokens - old_num_tokens',
            where 'old_num_tokens' is the current number of tokens in the 'final_logits_bias'.
            The extra bias columns are created using ops.zeros() function and then concatenated with the existing
            'final_logits_bias' tensor using ops.cat() function along the last axis.

            The 'final_logits_bias' attribute is updated with the resized tensor.

        Note:
            This method does not perform any validation on the inputs or check for any specific restrictions.

        Example:
            ```python
            >>> # Create an instance of the BlenderbotForConditionalGeneration model
            >>> model = BlenderbotForConditionalGeneration()
            ...
            >>> # Resize the final_logits_bias to have 100 tokens
            >>> model._resize_final_logits_bias(100)
            ```
        """
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.cat([self.final_logits_bias, extra_bias], axis=1)
        self.final_logits_bias = new_bias

    def get_output_embeddings(self):
        """
        This method retrieves the output embeddings from the BlenderbotForConditionalGeneration model.

        Args:
            self (BlenderbotForConditionalGeneration): The instance of the BlenderbotForConditionalGeneration class.
                It is used to access the lm_head attribute, which contains the output embeddings.

        Returns:
            None.

        Raises:
            None
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the Blenderbot model.

        Args:
            self (BlenderbotForConditionalGeneration): The instance of the BlenderbotForConditionalGeneration class.
            new_embeddings: The new embeddings to be set as the output embeddings. This parameter can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[Union[Tuple, BaseModelOutput]] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple[mindspore.Tensor], Seq2SeqLMOutput]
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
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        This method prepares inputs for generation in the BlenderbotForConditionalGeneration class.

        Args:
            self: The instance of the class.
            decoder_input_ids (Tensor): The input tensor for the decoder.
            past_key_values (Tuple): A tuple of past key values for attention mechanism.
            attention_mask (Tensor, optional): An optional tensor for attention mask.
            head_mask (Tensor, optional): An optional tensor for head mask.
            decoder_head_mask (Tensor, optional): An optional tensor for decoder head mask.
            cross_attn_head_mask (Tensor, optional): An optional tensor for cross-attention head mask.
            use_cache (bool, optional): A flag indicating whether to use cache.
            encoder_outputs (Dict, optional): A dictionary containing encoder outputs.

        Returns:
            Dict: A dictionary containing the prepared inputs for generation
                including 'input_ids', 'encoder_outputs', 'past_key_values', 'decoder_input_ids', 'attention_mask',
                'head_mask', 'decoder_head_mask', 'cross_attn_head_mask', and 'use_cache'.

        Raises:
            None
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

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the past key values according to the beam index.

        Args:
            past_key_values (Tuple): A tuple of past key values for each layer of the model.
                Each past key value is a tuple with three elements:

                - A tensor of shape (batch_size * beam_size, sequence_length, hidden_size)
                - A tensor of shape (batch_size * beam_size, hidden_size)
                - A tensor of shape (batch_size * beam_size, sequence_length)

            beam_idx (Tensor): A tensor of shape (batch_size, beam_size) containing the indices to reorder the past key values.

        Returns:
            reordered_past: This method returns the reordered past key values as a tuple.

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


# Copied from transformers.models.bart.modeling_bart.BartDecoderWrapper with Bart->Blenderbot
class BlenderbotDecoderWrapper(BlenderbotPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BlenderbotDecoderWrapper class.

        Args:
            self: The instance of the class.
            config:
                The configuration object for initializing the BlenderbotDecoderWrapper.

                - Type: object
                - Purpose: Specifies the configuration settings for the BlenderbotDecoderWrapper.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.decoder = BlenderbotDecoder(config)

    def construct(self, *args, **kwargs):
        """
        Method 'construct' in the class 'BlenderbotDecoderWrapper'.

        Args:
            self (BlenderbotDecoderWrapper): The instance of BlenderbotDecoderWrapper.
                It is used to access the methods and attributes of the class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder(*args, **kwargs)


# Copied from transformers.models.bart.modeling_bart.BartForCausalLM with Bart->Blenderbot, facebook/bart-base->facebook/blenderbot-400M-distill
class BlenderbotForCausalLM(BlenderbotPreTrainedModel):

    """
    Represents the Blenderbot model for causal language modeling.

    This class provides the functionality to initialize the model, set input and output embeddings, set the decoder,
    and construct the model. It also includes methods for preparing inputs for generation and reordering cache.

    The `construct` method takes various input arguments and returns the model outputs.
    The `prepare_inputs_for_generation` method prepares inputs for generation, and the `_reorder_cache` method
    reorders the cache.

    The class inherits from `BlenderbotPreTrainedModel` and includes detailed explanations of the input arguments,
    return values, and examples for usage.

    For consistency, the docstring follows the triple double quotes format.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the BlenderbotForCausalLM class.

        Args:
            self: The object instance.
            config (obj): The configuration object containing various settings for the model.
                It must have the following attributes:

                - is_decoder (bool): Specifies whether the model is a decoder. Must be set to True.
                - is_encoder_decoder (bool): Specifies whether the model is an encoder-decoder. Must be set to False.
                - hidden_size (int): The size of the hidden states.
                - vocab_size (int): The size of the vocabulary.

        Returns:
            None

        Raises:
            None
        """
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BlenderbotDecoderWrapper(config)

        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the BlenderbotForCausalLM model.

        Args:
            self (BlenderbotForCausalLM): The instance of the BlenderbotForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings from the decoder of the BlenderbotForCausalLM model.
        The input embeddings are used to convert the input tokens into continuous vector representations.
        These embeddings capture the semantic meaning of the input tokens and are essential for
        the model's understanding and generation of text.

        Note:
            The input embeddings are accessed using the 'embed_tokens' attribute of the model's decoder.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the BlenderbotForCausalLM model.

        Args:
            self (BlenderbotForCausalLM): The instance of BlenderbotForCausalLM class.
                This parameter is always implicitly passed and refers to the current instance of the class.
            value (torch.Tensor): The input embeddings to be set for the model.
                This parameter should be a torch.Tensor containing the input embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the BlenderbotForCausalLM model.

        Args:
            self (BlenderbotForCausalLM): The instance of the BlenderbotForCausalLM class.
                This parameter refers to the current instance of the model.

        Returns:
            None: This method returns the output embeddings represented by the lm_head attribute.
                The output embeddings are used for generating the model's output.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the BlenderbotForCausalLM model.

        Args:
            self (BlenderbotForCausalLM): The instance of the BlenderbotForCausalLM class.
            new_embeddings (torch.nn.Embedding): The new embeddings to be set as the output embeddings.
                It should be an instance of `torch.nn.Embedding` class.

        Returns:
            None.

        Raises:
            None.

        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Method to set the decoder for the BlenderbotForCausalLM model.

        Args:
            self (BlenderbotForCausalLM): The instance of the BlenderbotForCausalLM class.
                This parameter refers to the current instance of the class.
            decoder: The decoder object to be set for the model.
                It should be a valid decoder object compatible with the model.

        Returns:
            None: This method does not return any value. It updates the decoder for the model in-place.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        Returns the decoder of the BlenderbotForCausalLM model.

        Args:
            self: An instance of the BlenderbotForCausalLM class.

        Returns:
            None: This method returns the decoder of the BlenderbotForCausalLM model.
                The decoder is responsible for decoding the input sequence into a generated response.

        Raises:
            None.
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
            >>> from transformers import AutoTokenizer, BlenderbotForCausalLM
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
            >>> model = BlenderbotForCausalLM.from_pretrained("facebook/blenderbot-400M-distill", add_cross_attention=False)
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
        This method prepares inputs for generation in the BlenderbotForCausalLM class.

        Args:
            self: The instance of the class.
            input_ids (torch.Tensor): The input tensor containing token ids for the input sequence.
            past_key_values (Tuple[torch.Tensor]): Optional past key values for caching attention weights.
            attention_mask (torch.Tensor): Optional tensor specifying which elements of the input sequence should be attended to.
            use_cache (bool): Flag indicating whether to use caching for efficient generation.

        Returns:
            dict: A dictionary containing the updated input_ids, attention_mask, past_key_values, and use_cache.

        Raises:
            ValueError: If input_ids or attention_mask is not provided.
            IndexError: If the input_ids shape does not match the past key values.
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
        Method to reorder cache for beam search in the BlenderbotForCausalLM class.
        
        Args:
            past_key_values (tuple): Tuple containing past key-value states for each layer.
            beam_idx (Tensor): Index tensor specifying the order for reordering the past states.
        
        Returns:
            None: This method modifies the past_key_values in-place to reorder the cache according to the beam_idx.
        
        Raises:
            IndexError: If the provided beam_idx is out of bounds or not compatible with past_key_values.
            ValueError: If the input parameters are not in the expected format or do not meet the requirements.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

__all__ = [
    "BlenderbotForCausalLM",
    "BlenderbotForConditionalGeneration",
    "BlenderbotModel",
    "BlenderbotPreTrainedModel",
]
