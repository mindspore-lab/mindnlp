# coding=utf-8
# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science All rights reserved.
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
""" MindSpore BioGPT model."""

import math
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_biogpt import BioGptConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/biogpt"
_CONFIG_FOR_DOC = "BioGptConfig"


BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/biogpt",
    "microsoft/BioGPT-Large",
    # See all BioGPT models at https://hf-mirror.com/models?filter=biogpt
]


# Copied from transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding with OPT->BioGpt
class BioGptLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Initializes a new instance of BioGptLearnedPositionalEmbedding.
        
        Args:
            self: The instance of the class.
            num_embeddings (int): The number of embeddings to be created.
            embedding_dim (int): The dimension of the embeddings.
        
        Returns:
            None.
        
        Raises:
            None
        """
        # BioGpt is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def construct(self, attention_mask: mindspore.Tensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (ops.cumsum(attention_mask, axis=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().construct(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->BioGpt
class BioGptAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BioGptConfig] = None,
    ):
        """
        Initializes an instance of the BioGptAttention class.
        
        Args:
            self: The instance of the class.
            embed_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability (default: 0.0).
            is_decoder (bool, optional): Specifies if the attention is used in a decoder (default: False).
            bias (bool, optional): Specifies whether to include bias in the linear projections (default: True).
            is_causal (bool, optional): Specifies if the attention is causal or not (default: False).
            config (Optional[BioGptConfig], optional): The configuration object (default: None).
        
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
        This method '_shape' in the class 'BioGptAttention' reshapes the input tensor to match the desired shape for multi-head attention computation.
        
        Args:
            self (BioGptAttention): The instance of the BioGptAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence in the tensor.
            bsz (int): The batch size of the tensor.
        
        Returns:
            None: The method reshapes the input tensor and returns None.
        
        Raises:
            None.
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


class BioGptDecoderLayer(nn.Cell):

    """
    This class represents a BioGptDecoderLayer, which is a component of a BioGptDecoder in a Transformer-based model.
    It performs the decoding operation on the input hidden states.
    
    Attributes:
        embed_dim (int): The dimension of the hidden states.
        self_attn (BioGptAttention): The self-attention mechanism.
        dropout (float): The dropout rate.
        activation_fn (function): The activation function.
        activation_dropout (float): The dropout rate for the activation function.
        self_attn_layer_norm (nn.LayerNorm): The layer normalization for the self-attention output.
        fc1 (nn.Dense): The first fully connected layer.
        fc2 (nn.Dense): The second fully connected layer.
        final_layer_norm (nn.LayerNorm): The final layer normalization.

    Methods:
        construct(hidden_states, attention_mask, layer_head_mask, past_key_value, output_attentions, use_cache):
            Performs the decoding operation on the input hidden states.

    """
    def __init__(self, config: BioGptConfig):
        """
        Initializes a BioGptDecoderLayer object.

        Args:
            self (BioGptDecoderLayer): The instance of the BioGptDecoderLayer class.
            config (BioGptConfig):
                An instance of BioGptConfig containing configuration parameters.

                - config.hidden_size (int): The size of the hidden layer.
                - config.num_attention_heads (int): The number of attention heads.
                - config.attention_probs_dropout_prob (float): Dropout probability for attention weights.
                - config.hidden_dropout_prob (float): Dropout probability for hidden layers.
                - config.hidden_act (str): The activation function for hidden layers.
                - config.activation_dropout (float): Dropout probability for activation functions.

        Returns:
            None.

        Raises:
            ValueError: If config.hidden_size is not an integer.
            ValueError: If config.num_attention_heads is not an integer.
            ValueError: If config.attention_probs_dropout_prob is not a float.
            ValueError: If config.hidden_dropout_prob is not a float.
            KeyError: If config.hidden_act is not a valid activation function.
            ValueError: If config.activation_dropout is not a float.
        """
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = BioGptAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )
        self.dropout = config.hidden_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Dense(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            past_key_value (`Tuple(mindspore.Tensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BioGptPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BioGptConfig
    base_model_prefix = "biogpt"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
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


class BioGptModel(BioGptPreTrainedModel):

    """
    BioGptModel represents a GPT (Generative Pre-trained Transformer) model customized for bioinformatics tasks.
    This class inherits from BioGptPreTrainedModel and implements methods for initializing the model,
    setting input embeddings, and constructing the model for inference or training.
    The model includes parameters for layer dropout, hidden dropout probability, embedding dimensions,
    padding index, and layer normalization.
    The construct method processes input data, applies attention masks, computes positional embeddings,
    and iterates through decoder layers to generate model output.
    Additionally, the class supports gradient checkpointing and caching for efficient training.
    """
    def __init__(self, config: BioGptConfig):
        """
        Initializes a BioGptModel instance with the provided configuration.

        Args:
            self: The instance of the BioGptModel class.
            config (BioGptConfig):
                An instance of BioGptConfig containing the configuration parameters for the model.

                - BioGptConfig is a data class that holds various settings for the BioGptModel.
                - It must be provided to properly configure the model.
                - The config parameter is required and should not be None.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.config = config
        self.layerdrop = config.layerdrop
        self.dropout = config.hidden_dropout_prob
        self.embed_dim = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, self.embed_dim, self.padding_idx)
        self.embed_positions = BioGptLearnedPositionalEmbedding(config.max_position_embeddings, self.embed_dim)

        self.layers = nn.CellList([BioGptDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method retrieves the input embeddings from the BioGptModel instance.

        Args:
            self (BioGptModel): The current instance of the BioGptModel class.

        Returns:
            embed_tokens: This method returns the input embeddings associated with the BioGptModel instance.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the BioGptModel.

        Args:
            self (BioGptModel): The instance of the BioGptModel class.
            value (Any): The input embeddings to be set for the model.

        Returns:
            None:
                This method updates the 'embed_tokens' attribute of the BioGptModel instance with the provided
                input embeddings.

        Raises:
            None.
        """
        self.embed_tokens = value

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the BioGptModel.

        Args:
            self (BioGptModel): The instance of the BioGptModel class.
            input_ids (Optional[mindspore.Tensor]): The input token IDs. Defaults to None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask. Defaults to None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded inputs. Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values. Defaults to None.
            use_cache (Optional[bool]): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool]): Whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: The output of the BioGptModel.
                It can either be a tuple or an instance of BaseModelOutputWithPastAndCrossAttentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If the length of the provided attention mask is incorrect.
            Warning: If use_cache is set to True and gradient checkpointing is enabled.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input = input_ids
            input_shape = input.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input) * self.embed_scale

        if attention_mask is None:
            attention_mask = ops.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1] + past_key_values_length),
                dtype=mindspore.bool_,
            )
        elif attention_mask.shape[1] != past_key_values_length + input_shape[1]:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{past_key_values_length + input_shape[1]} (sum of the lengths of current and past inputs)"
            )

        # embed positions
        positions = self.embed_positions(attention_mask, past_key_values_length)

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds + positions

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = None
        next_decoder_cache = () if use_cache else None

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
                all_self_attns += (layer_outputs[1],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.layer_norm(hidden_states)

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


class BioGptForCausalLM(BioGptPreTrainedModel):

    """
    BioGptForCausalLM represents a BioGpt model for Causal Language Modeling.
    This class inherits from BioGptPreTrainedModel and provides methods for initializing the model,
    constructing the model, and preparing inputs for generation.

    Attributes:
        config: The configuration for the BioGptForCausalLM model.

    Methods:
        __init__(config): Initializes the BioGptForCausalLM model with the given configuration.
        get_output_embeddings(): Returns the output projection layer for the model.
        set_output_embeddings(new_embeddings): Sets the output projection layer to the new embeddings.
        construct(input_ids, attention_mask, head_mask, inputs_embeds, past_key_values, labels, use_cache,
            output_attentions, output_hidden_states, return_dict):
            Constructs the BioGptForCausalLM model with the given input parameters and returns the model output.
        prepare_inputs_for_generation(input_ids, attention_mask, inputs_embeds, past_key_values, **kwargs):
            Prepares the inputs for generation based on the given parameters.
        _reorder_cache(past_key_values, beam_idx): Reorders the past key values based on the given beam index.

    Note:
        Labels for language modeling are shifted inside the model,
        and the loss is only computed for valid labels within the vocabulary size.
    """
    _tied_weights_keys = ["output_projection.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the BioGptForCausalLM class.

        Args:
            self: The instance of the BioGptForCausalLM class.
            config: An object containing configuration settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If the config parameter does not contain the required settings for initializing the model.
        """
        super().__init__(config)

        self.biogpt = BioGptModel(config)
        self.output_projection = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method retrieves the output embeddings from the BioGptForCausalLM model.

        Args:
            self: An instance of the BioGptForCausalLM class.

        Returns:
            None: This method returns the output projection of the model as a value of type 'None'.

        Raises:
            None.
        """
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the BioGptForCausalLM model.

        Args:
            self (BioGptForCausalLM): The instance of the BioGptForCausalLM class.
                This parameter is automatically passed and refers to the current instance.
            new_embeddings (object): New embeddings to be set as the output projections.
                This parameter should be an object representing the new embeddings to be used.

        Returns:
            None.

        Raises:
            None.
        """
        self.output_projection = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
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
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.output_projection(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = ops.cross_entropy(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask, inputs_embeds=None, past_key_values=None, **kwargs
    ):
        '''
        This method prepares inputs for generation in the BioGptForCausalLM class.

        Args:
            self (object): The instance of the class.
            input_ids (torch.Tensor): The input tensor containing the tokenized input IDs.
            attention_mask (torch.Tensor): The tensor containing the attention mask to avoid attending to padding tokens.
            inputs_embeds (torch.Tensor, optional): The tensor containing the embeddings for the input tokens. Defaults to None.
            past_key_values (tuple, optional): The tuple containing past key values for fast decoding. Defaults to None.

        Returns:
            dict: A dictionary containing the model inputs,
                including 'input_ids', 'attention_mask', 'past_key_values', and 'use_cache' (if provided).
                Returns None if past_key_values is provided and the input_ids shape is greater than the past_length.

        Raises:
            ValueError: If the input_ids shape is less than or equal to zero.
            TypeError: If the input_ids, attention_mask, or inputs_embeds are not of type torch.Tensor.
            IndexError: If the past_key_values tuple does not have the expected shape.
        '''
        # only last tokens for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache of past key values based on the given beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
                Each element in the tuple is a tensor representing the past states for the corresponding layer.
            beam_idx (tensor): A tensor containing the indices of the beams to reorder the past key values.

        Returns:
            None: This method does not return any value, as it modifies the 'past_key_values' in place.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class BioGptForTokenClassification(BioGptPreTrainedModel):

    """
    This class represents a BioGpt model for token classification, inheriting from BioGptPreTrainedModel.
    It includes methods for initializing the model and constructing token classification outputs based on input data.
    The model utilizes a transformer architecture for processing input sequences and generating classification predictions.
    The class provides functionality for computing loss based on predicted logits and actual labels, as well as
    handling optional parameters for caching, attention masks, and return dictionary configurations.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BioGptForTokenClassification class.

        Args:
            self: The instance of the BioGptForTokenClassification class.
            config:
                An object containing configuration parameters for the model.

                - Type: object
                - Purpose: Configuration object that specifies model settings.
                - Restrictions: Must contain at least the 'num_labels' attribute.

        Returns:
            None

        Raises:
            TypeError: If the 'config' parameter is not provided or is invalid.
            AttributeError: If the 'config' object does not have the required 'num_labels' attribute.
            ValueError: If the 'classifier_dropout' or 'hidden_dropout_prob' attributes are invalid in the 'config' object.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.biogpt = BioGptModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        else:
            classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.biogpt(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = ops.where(
                    active_loss, labels.view(-1), mindspore.tensor(-100).type_as(labels)
                )
                loss = ops.cross_entropy(active_logits, active_labels)
            else:
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BioGptForSequenceClassification(BioGptPreTrainedModel):

    '''
    This class represents a BioGpt model for sequence classification tasks.
    It inherits from the BioGptPreTrainedModel and includes methods for initialization,
    constructing the model, getting input embeddings, and setting input embeddings.

    The __init__ method initializes the BioGptForSequenceClassification instance with a BioGptConfig
    and sets the number of labels, BioGptModel, and score.

    The construct method takes input tensors for the model and returns the sequence classifier output with past key values.
    It also handles labels for computing the sequence classification/regression loss and handles different problem types
    such as regression, single-label classification, and multi-label classification.

    The get_input_embeddings method returns the input embeddings of the BioGpt model.

    The set_input_embeddings method sets the input embeddings of the BioGpt model.

    Note:
        The class inherits from BioGptPreTrainedModel and includes additional methods not provided in the given code snippet.
    '''
    def __init__(self, config: BioGptConfig):
        """
        Initializes a BioGptForSequenceClassification instance.

        Args:
            self: The instance of the BioGptForSequenceClassification class.
            config (BioGptConfig): An instance of BioGptConfig containing configuration settings for the model.
                This parameter is required for initializing the BioGptForSequenceClassification instance.
                It specifies the configuration details such as the number of labels and hidden size.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type BioGptConfig.
            ValueError: If the num_labels attribute is not present in the config parameter.
            ValueError: If an error occurs during the initialization of the BioGptModel or Dense layers.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.biogpt = BioGptModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.biogpt(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_length = -1
        else:
            if input_ids is not None:
                sequence_length = ops.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_length = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[ops.arange(batch_size), sequence_length]

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
                    loss = ops.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from BioGptForSequenceClassification.
        
        Args:
            self (BioGptForSequenceClassification): The instance of the BioGptForSequenceClassification class.
                Represents the object itself.
        
        Returns:
            None: This method returns the embeddings obtained from the BioGpt model.
        
        Raises:
            None.
        """
        return self.biogpt.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the BioGptForSequenceClassification model.
        
        Args:
            self (BioGptForSequenceClassification): The instance of the BioGptForSequenceClassification class.
            value (Tensor): The input embeddings to be set for the model. It should be a 2D tensor.
        
        Returns:
            None.
        
        Raises:
            None
        """
        self.biogpt.embed_tokens = value

__all__ = [
    "BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "BioGptForCausalLM",
    "BioGptForTokenClassification",
    "BioGptForSequenceClassification",
    "BioGptModel",
    "BioGptPreTrainedModel",
]
