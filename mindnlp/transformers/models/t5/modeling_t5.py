# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
# Copyright 2022 Huawei Technologies Co., Ltd
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
""" MindSpore T5 model."""

import copy
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Constant, Normal

from mindnlp.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_t5 import T5Config

logger = logging.get_logger(__name__)

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://hf-mirror.com/models?filter=t5
]

class T5LayerNorm(nn.Cell):
    """T5LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = Parameter(initializer('zeros', (hidden_size,), mindspore.float32), 'weight')
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        This method 'construct' is a part of the class 'T5LayerNorm' and is used to perform layer normalization
        on the input hidden states.
        
        Args:
            self (T5LayerNorm): The instance of the T5LayerNorm class.
            hidden_states (numpy.ndarray): The input hidden states to be normalized.
                It is expected to be an array of numerical values.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the input hidden_states is not a valid numerical array.
            TypeError: If the input hidden_states or self.weight is not of the expected data type.
            RuntimeError: If there is an issue with the normalization process.
        """
        variance = hidden_states.astype(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states / ops.sqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.astype(self.weight.dtype)
        return self.weight * hidden_states

ALL_LAYERNORM_LAYERS.append(T5LayerNorm)

class T5DenseActDense(nn.Cell):
    """T5DenseActDense"""
    def __init__(self, config: T5Config):
        """
        Initializes an instance of the T5DenseActDense class.
        
        Args:
            self: The instance of the class.
            config (T5Config):
                The configuration object containing the model's settings.

                - The 'config' parameter is of type T5Config, which specifies the configuration for the T5 model.
                - It is used to set up the parameters for the dense layers and the dropout rate.
                - This parameter is required and has no default value.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.wi = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        """
        This method constructs the hidden states by applying a series of transformations including linear mapping,
        activation function, dropout, and additional conversion based on weight data types.

        Args:
            self (T5DenseActDense): The instance of the T5DenseActDense class.
            hidden_states (Tensor): The input hidden states to be processed by the method.

        Returns:
            None.

        Raises:
            TypeError:
                If the data type of weights in self.wo does not match the data type of hidden_states or mindspore.int8.
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.wo.weight.dtype not in (hidden_states.dtype, mindspore.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Cell):
    """T5DenseGatedActDense"""
    def __init__(self, config: T5Config):
        """
        Initializes an instance of the T5DenseGatedActDense class.

        Args:
            self: An instance of the T5DenseGatedActDense class.
            config (T5Config): The configuration object for the T5 model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.wi_0 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wi_1 = nn.Dense(config.d_model, config.d_ff, has_bias=False)
        self.wo = nn.Dense(config.d_ff, config.d_model, has_bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def construct(self, hidden_states):
        """
        Constructs the hidden states of the T5DenseGatedActDense model.

        Args:
            self: The instance of the T5DenseGatedActDense class.
            hidden_states (Tensor): The input hidden states.
                It should have the shape (batch_size, sequence_length, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        if self.wo.weight.dtype not in (hidden_states.dtype, mindspore.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Cell):
    """T5LayerFF"""
    def __init__(self, config: T5Config):
        """
        Initializes an instance of the T5LayerFF class.

        Args:
            self: The instance of the T5LayerFF class.
            config (T5Config): The configuration object for the T5 model.
                It contains various parameters and settings for the model.

        Returns:
            None

        Raises:
            None.
        """
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(self, hidden_states):
        """
        Constructs the forward pass of the T5LayerFF class.

        Args:
            self (T5LayerFF): An instance of the T5LayerFF class.
            hidden_states (Tensor): The hidden states input tensor.
                Shape (batch_size, sequence_length, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Cell):
    """T5Attention"""
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        """
        Initializes an instance of the T5Attention class.

        Args:
            self: The object itself.
            config (T5Config):
                An instance of the T5Config class that holds the configuration parameters for the attention mechanism.
            has_relative_attention_bias (bool):
                A boolean value indicating whether the attention mechanism has relative attention bias.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.k = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.v = nn.Dense(self.d_model, self.inner_dim, has_bias=False)
        self.o = nn.Dense(self.inner_dim, self.d_model, has_bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes the attention heads in the T5Attention class.

        Args:
            self (T5Attention): An instance of the T5Attention class.
            heads (list): A list of attention heads to be pruned.

        Returns:
            None.

        Raises:
            None.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, axis=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, ops.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += ops.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = ops.arange(query_length, dtype=mindspore.int64)[:, None]
        memory_position = ops.arange(key_length, dtype=mindspore.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def construct(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).swapaxes(1, 2)

        def unshape(states):
            """reshape"""
            return states.swapaxes(1, 2).view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = ops.cat([past_key_value, hidden_states], axis=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, key_states.swapaxes(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = ops.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = ops.softmax(scores.float() + 1e-10, axis=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = ops.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ops.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Cell):
    """T5LayerSelfAttention"""
    def __init__(self, config, has_relative_attention_bias=False):
        """Initialize the T5LayerSelfAttention.

        Args:
            self (T5LayerSelfAttention): An instance of the T5LayerSelfAttention class.
            config (Config): An object containing the configuration parameters.
            has_relative_attention_bias (bool, optional): A flag indicating whether the attention bias is relative or not.
                Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        This method 'construct' in the class 'T5LayerSelfAttention' constructs the output of a T5 self-attention layer.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The hidden states of the input sequence.
            attention_mask (Optional[Tensor]): An optional tensor for masking out certain positions in the input
                sequence during attention calculation.
            position_bias (Optional[Tensor]): An optional tensor providing additional bias to attention scores
                based on position.
            layer_head_mask (Optional[Tensor]): An optional tensor for masking out certain heads in the attention
                calculation.
            past_key_value (Optional[Tuple[Tensor]]): An optional tuple of key and value tensors from the previous
                time steps for faster decoding.
            use_cache (bool): A flag indicating whether to use caching for faster decoding.
            output_attentions (bool): A flag indicating whether to output attention weights.

        Returns:
            Tuple[Tensor]: A tuple containing the updated hidden states after self-attention and any additional outputs
                from the attention mechanism.

        Raises:
            None
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerCrossAttention(nn.Cell):
    """T5LayerCrossAttention"""
    def __init__(self, config):
        """
        Initializes an instance of the T5LayerCrossAttention class.

        Args:
            self: The object instance.
            config: An instance of the configuration class that contains the model's hyperparameters and settings.
                It is of type 'Any' and is used to configure the behavior of the cross-attention layer.
                The configuration object must have the following attributes:

                - d_model: An integer representing the dimensionality of the model's hidden states.
                - layer_norm_epsilon: A small float value used to stabilize the layer normalization process.
                - dropout_rate: A float value between 0 and 1, denoting the dropout rate for the layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def construct(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        """
        This method constructs the T5 layer cross-attention mechanism.

        Args:
            self: Reference to the current instance of the class.
            hidden_states: Tensor representing the input hidden states.
            key_value_states: Tensor representing the key-value states for the attention mechanism.
            attention_mask: Optional tensor specifying the attention mask. Defaults to None.
            position_bias: Optional tensor providing positional bias information. Defaults to None.
            layer_head_mask: Optional tensor masking specific attention heads. Defaults to None.
            past_key_value: Optional tensor containing cached key-value states from previous steps. Defaults to None.
            use_cache: Boolean indicating whether to use cache for key-value states. Defaults to False.
            query_length: Optional integer specifying the length of the query. Defaults to None.
            output_attentions: Boolean indicating whether to output attentions. Defaults to False.

        Returns:
            Tuple containing the layer output and additional attention outputs.

        Raises:
            None
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Cell):
    """T5Block"""
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initializes a new instance of the T5Block class.

        Args:
            self: The object itself.
            config (object): The configuration object containing the settings for the T5Block.
            has_relative_attention_bias (bool, optional): Specifies whether the attention bias is relative or not.
                Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.CellList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        # return_dict=True,
    ):
        """
        Constructs a T5Block.

        Args:
            self (T5Block): The T5Block instance.
            hidden_states (Tensor): The input hidden states.
            attention_mask (Tensor, optional): The attention mask tensor. Defaults to None.
            position_bias (Tensor, optional): The position bias tensor. Defaults to None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor. Defaults to None.
            encoder_attention_mask (Tensor, optional): The encoder attention mask tensor. Defaults to None.
            encoder_decoder_position_bias (Tensor, optional): The encoder-decoder position bias tensor. Defaults to None.
            layer_head_mask (Tensor, optional): The layer head mask tensor. Defaults to None.
            cross_attn_layer_head_mask (Tensor, optional): The cross-attention layer head mask tensor. Defaults to None.
            past_key_value (Tuple[Tensor], optional): The past key-value states. Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to False.
            output_attentions (bool, optional): Whether to output attentions. Defaults to False.

        Returns:
            Tuple:
                A tuple containing the following elements:

                - hidden_states (Tensor): The output hidden states.
                - present_key_value_state (Tuple[Tensor], optional): The present key-value state. None if not available.
                - attention_outputs (Tuple[Tensor], optional): The attention outputs. None if not available.

        Raises:
            ValueError: If the number of past states is not as expected.
            Warning: If `past_key_values` is passed to the encoder.
        """
        if past_key_value is not None:
            if not self.is_decoder:
                logging.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
                clamp_value = mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs
        # hidden-states, present_key_value_states, (self-attention position bias),
        # (self-attention weights), (cross-attention position bias),(cross-attention weights)


class T5ClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(self, config: T5Config):
        """
        Initializes a T5ClassificationHead instance.

        Args:
            self: The T5ClassificationHead instance.
            config (T5Config): The configuration for the T5 model. It specifies the model's architecture and parameters.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type T5Config.
            ValueError: If the config parameters are not valid or if there are any issues during initialization.
        """
        super().__init__()
        self.dense = nn.Dense(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Dense(config.d_model, config.num_labels)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the T5 classification head.

        Args:
            self: The T5ClassificationHead object.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
                This tensor contains the hidden states from the T5 model.
                Shape of the tensor should be (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The output tensor after passing through the T5 classification head.
                Shape of the tensor is (batch_size, sequence_length, num_labels).

        Raises:
            None.
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = T5Config
    base_model_prefix = "transformer"

    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        """
        Method: dummy_inputs

        Description:
            This method generates dummy input data for the T5PreTrainedModel.

        Args:
            self: An instance of the T5PreTrainedModel class.

        Returns:
            `dict`:

                - Type: None
                - Purpose: This method returns a dictionary containing dummy input data for the model.

                The dictionary includes the following keys:

                - 'decoder_input_ids': Tensor containing dummy input IDs.
                - 'input_ids': Tensor containing dummy input IDs.
                - 'decoder_attention_mask': Tensor containing dummy mask data.

        Raises:
            This method does not raise any exceptions.
        """
        input_ids = mindspore.tensor(DUMMY_INPUTS)
        input_mask = mindspore.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(cell, T5LayerNorm):
            cell.weight.set_data(initializer(Constant(factor * 1.0), cell.weight.shape, cell.weight.dtype))
        elif isinstance(
            cell,
            (T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            cell.shared.weight.set_data(initializer(Normal(factor * 1.0),
                                                cell.shared.weight.shape, cell.shared.weight.dtype))
            if hasattr(cell, "lm_head") and not self.config.tie_word_embeddings:
                cell.lm_head.weight.set_data(initializer(Normal(factor * 1.0), cell.lm_head.weight.shape, cell.lm_head.weight.dtype))
            if hasattr(cell, "qa_outputs"):
                cell.qa_outputs.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                            cell.qa_outputs.weight.shape, cell.qa_outputs.weight.dtype))
                cell.qa_outputs.bias.set_data(initializer('zeros', cell.qa_outputs.bias.shape, cell.qa_outputs.bias.dtype))
        elif isinstance(cell, T5ClassificationHead):
            cell.dense.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.dense.weight.shape, cell.dense.weight.dtype))

            if hasattr(cell.dense, "bias") and cell.dense.bias is not None:
                cell.dense.bias.set_data(initializer('zeros', cell.dense.bias.shape, cell.dense.bias.dtype))
            cell.out_proj.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.out_proj.weight.shape, cell.out_proj.weight.dtype))

            if hasattr(cell.out_proj, "bias") and cell.out_proj.bias is not None:
                cell.out_proj.bias.set_data(initializer('zeros', cell.out_proj.bias.shape, cell.out_proj.bias.dtype))
        elif isinstance(cell, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            cell.wi.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi.weight.shape, cell.wi.weight.dtype))
            if hasattr(cell.wi, "bias") and cell.wi.bias is not None:
                cell.wi.bias.set_data(initializer('zeros', cell.wi.bias.shape, cell.wi.bias.dtype))

            cell.wo.weight.set_data(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)),
                                                cell.wo.weight.shape, cell.wo.weight.dtype))

            if hasattr(cell.wo, "bias") and cell.wo.bias is not None:
                cell.wo.bias.set_data(initializer('zeros', cell.wo.bias.shape, cell.wo.bias.dtype))
        elif isinstance(cell, T5DenseGatedActDense):
            cell.wi_0.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi_0.weight.shape, cell.wi_0.weight.dtype))
            if hasattr(cell.wi_0, "bias") and cell.wi_0.bias is not None:
                cell.wi_0.bias.set_data(initializer('zeros', cell.wi_0.bias.shape, cell.wi_0.bias.dtype))

            cell.wi_1.weight.set_data(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi_1.weight.shape, cell.wi_1.weight.dtype))
            if hasattr(cell.wi_1, "bias") and cell.wi_1.bias is not None:
                cell.wi_1.bias.set_data(initializer('zeros', cell.wi_1.bias.shape, cell.wi_1.bias.dtype))

            cell.wo.weight.set_data(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)),
                                                cell.wo.weight.shape, cell.wo.weight.dtype))

            if hasattr(cell.wo, "bias") and cell.wo.bias is not None:
                cell.wo.bias.set_data(initializer('zeros', cell.wo.bias.shape, cell.wo.bias.dtype))
        elif isinstance(cell, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads

            cell.q.weight.set_data(initializer(Normal(factor * ((d_model * key_value_proj_dim) ** -0.5)),
                                                cell.q.weight.shape, cell.q.weight.dtype))
            cell.k.weight.set_data(initializer(Normal(factor * (d_model**-0.5)),
                                                cell.k.weight.shape, cell.k.weight.dtype))
            cell.v.weight.set_data(initializer(Normal(factor * (d_model**-0.5)),
                                                cell.v.weight.shape, cell.v.weight.dtype))
            cell.o.weight.set_data(initializer(Normal(factor * ((n_heads * key_value_proj_dim) ** -0.5)),
                                                cell.o.weight.shape, cell.o.weight.dtype))
            if cell.has_relative_attention_bias:
                cell.relative_attention_bias.weight.set_data(initializer(Normal(factor * (d_model**-0.5)),
                                                    cell.relative_attention_bias.weight.shape, cell.relative_attention_bias.weight.dtype))

    def _shift_right(self, input_ids):
        """
        Shifts the input IDs to the right by one position, inserting the decoder start token ID at the beginning.

        Args:
            self (T5PreTrainedModel): An instance of the T5PreTrainedModel class.
            input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the input IDs.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length) representing the shifted input IDs.

        Raises:
            ValueError: If `self.model.config.decoder_start_token_id` is not defined
                or if `self.model.config.pad_token_id` is not defined.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    """T5Stack"""
    def __init__(self, config):
        """
        Initializes an instance of the T5Stack class.

        Args:
            self: The instance of the T5Stack class.
            config: An object containing the configuration parameters for the T5Stack.
                It should have the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - d_model (int): The dimensionality of the model.
                - is_decoder (bool): Indicates whether the T5Stack is used as a decoder.
                num_layers (int): The number of layers in the T5Stack.
                - layer_norm_epsilon (float): The epsilon value for layer normalization.
                - dropout_rate (float): The dropout rate.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.is_decoder = config.is_decoder

        self.block = nn.CellList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.post_init()

    def get_input_embeddings(self):
        """Return the input embeddings of the T5Stack.

        Args:
            self: An instance of the T5Stack class.

        Returns:
            embed_tokens: This method returns the input embeddings of the T5Stack.
                The input embeddings are the embedded tokens used as input for the T5 model.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set new input embeddings for the T5Stack model.

        Args:
            self (T5Stack): The instance of the T5Stack class.
            new_embeddings (object): The new embeddings to set for the input.
                It should be compatible with the model's input format.

        Returns:
            None: This method updates the input embeddings of the T5Stack model in place.

        Raises:
            None.
        """
        self.embed_tokens = new_embeddings

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Constructs the T5Stack model.

        Args:
            self (T5Stack): The instance of the T5Stack class.
            input_ids (Tensor, optional): The input token IDs. Default: None.
            attention_mask (Tensor, optional): The attention mask tensor. Default: None.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder. Default: None.
            encoder_attention_mask (Tensor, optional): The attention mask for encoder hidden states. Default: None.
            inputs_embeds (Tensor, optional): The embedded inputs. Default: None.
            head_mask (list, optional): The mask for attention heads. Default: None.
            cross_attn_head_mask (list, optional): The mask for cross-attention heads. Default: None.
            past_key_values (list, optional): The past key values for caching. Default: None.
            use_cache (bool, optional): Whether to use caching. Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.
            return_dict (bool, optional): Whether to return a dictionary. Default: None.

        Returns:
            None

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified at the same time.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            AssertionError: If the model is not initialized with valid token embeddings.
            AssertionError: If use_cache is set to True and the model is not used as a decoder.

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )

        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids.astype(mindspore.int64))

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, mask_seq_length), mindspore.float32)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = ops.ones(
                (batch_size, encoder_seq_length), mindspore.int64
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), \
            # (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5Model(T5PreTrainedModel):
    """T5Model"""
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        """
        __init__ method in the T5Model class initializes a new instance of the class.

        Args:
            self: A reference to the instance of the class.
            config (T5Config): An instance of T5Config class containing configuration parameters for the T5 model.
                It includes parameters such as vocab_size, d_model, is_decoder, use_cache, is_encoder_decoder,
                and num_decoder_layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings for the T5Model.

        Args:
            self: The instance of the T5Model class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the T5Model.

        Args:
            self (T5Model): The instance of the T5Model class.
            new_embeddings: The new input embeddings to be set for the model.
                This should be a tensor of shape (vocab_size, hidden_size).

        Returns:
            None.

        Raises:
            None.
        """
        self.shared = new_embeddings
        # self.encoder.set_input_embeddings(new_embeddings)
        # self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        Tie the weights of the T5Model if specified in the configuration.

        Args:
            self (T5Model):
                The instance of the T5Model class.

                - This parameter represents the T5Model object on which the method is called.

        Returns:
            None.

        Raises:
            None
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        """
        This method returns the encoder for the T5Model.

        Args:
            self: The instance of the T5Model class.

        Returns:
            encoder:
                Returns the encoder associated with the T5Model.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        Method to retrieve the decoder of the T5Model.

        Args:
            self (T5Model): The T5Model instance on which the method is called.

        Returns:
            decoder: The method returns the decoder attribute of the T5Model instance.

        Raises:
            This method does not raise any exceptions.
        """
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        Constructs the T5 model for sequence-to-sequence tasks.

        Args:
            self (T5Model): The instance of the T5Model class.
            input_ids (torch.Tensor, optional): The input sequence tensor IDs. Default: None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Default: None.
            decoder_input_ids (torch.Tensor, optional): The decoder input sequence tensor IDs. Default: None.
            decoder_attention_mask (torch.Tensor, optional): The decoder attention mask tensor. Default: None.
            head_mask (torch.Tensor, optional): The head mask tensor. Default: None.
            decoder_head_mask (torch.Tensor, optional): The decoder head mask tensor. Default: None.
            cross_attn_head_mask (torch.Tensor, optional): The cross-attention head mask tensor. Default: None.
            encoder_outputs (tuple, optional): The encoder outputs. Default: None.
            past_key_values (tuple, optional): The past key values. Default: None.
            inputs_embeds (torch.Tensor, optional): The input embeddings tensor. Default: None.
            decoder_inputs_embeds (torch.Tensor, optional): The decoder input embeddings tensor. Default: None.
            use_cache (bool, optional): Whether to use cache. Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.
            return_dict (bool, optional): Whether to return a dictionary. Default: None.

        Returns:
            None

        Raises:
            None
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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


class T5ForConditionalGeneration(T5PreTrainedModel):
    """T5ForConditionalGeneration"""
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        """
        Initializes an instance of the T5ForConditionalGeneration class.

        Args:
            self: The object instance.
            config (T5Config): The configuration object for the T5 model.
                It contains various parameters to customize the model's behavior, such as the model dimension,
                vocabulary size, and number of decoder layers.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False)

        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the T5 model.

        Args:
            self (T5ForConditionalGeneration): The instance of the T5ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Set input embeddings for the T5 model.

        Args:
            self (T5ForConditionalGeneration): The instance of the T5ForConditionalGeneration class.
            new_embeddings (tensor): The new input embeddings to be set for the model.
                It should be a tensor of shape (vocab_size, hidden_size).

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not a tensor.
            ValueError: If the shape of the new_embeddings tensor does not match the required shape
                (vocab_size, hidden_size).
        """
        self.shared = new_embeddings
        # self.encoder.set_input_embeddings(new_embeddings)
        # self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        Method _tie_weights in the class T5ForConditionalGeneration ties or clones weights for word embeddings.

        Args:
            self (T5ForConditionalGeneration): The instance of the T5ForConditionalGeneration class.
                It represents the current object and is used to access attributes and methods within the class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the T5 model.

        Args:
            self (T5ForConditionalGeneration): The T5 model instance.
            new_embeddings (torch.Tensor): The new embeddings to set as the output embeddings for the model.

        Returns:
            None: This method updates the output embeddings of the T5 model in place.

        Raises:
            TypeError: If the new_embeddings parameter is not a torch.Tensor.
            ValueError: If the shape of the new_embeddings does not match the expected shape for model output embeddings.
        """
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the T5 model.

        Args:
            self (T5ForConditionalGeneration): An instance of the T5ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def get_encoder(self):
        """
        This method is part of the 'T5ForConditionalGeneration' class and is used to retrieve the encoder.

        Args:
            self (T5ForConditionalGeneration): An instance of the 'T5ForConditionalGeneration' class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder used by the T5 model for conditional generation.

        Args:
            self (T5ForConditionalGeneration): The current instance of the T5ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.decoder

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """Constructs the T5 model for conditional generation.

        Args:
            self (T5ForConditionalGeneration): The instance of the T5ForConditionalGeneration class.
            input_ids (torch.Tensor, optional): The input sequence tensor of shape (batch_size, sequence_length).
                Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                Defaults to None.
            decoder_input_ids (torch.Tensor, optional): The decoder input sequence tensor of shape
                (batch_size, decoder_sequence_length).  Defaults to None.
            decoder_attention_mask (torch.Tensor, optional): The decoder attention mask tensor of shape
                (batch_size, decoder_sequence_length). Defaults to None.
            head_mask (torch.Tensor, optional): The head mask tensor of shape (num_layers, num_heads).
                Defaults to None.
            decoder_head_mask (torch.Tensor, optional): The decoder head mask tensor of shape (num_layers, num_heads).
                Defaults to None.
            cross_attn_head_mask (torch.Tensor, optional): The cross-attention head mask tensor of shape
                (num_layers, num_heads). Defaults to None.
            encoder_outputs (tuple, optional): The encoder outputs returned by the encoder model.
                Defaults to None.
            past_key_values (tuple, optional): The past key values returned by the decoder model.
                Defaults to None.
            inputs_embeds (torch.Tensor, optional): The input embeddings tensor of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            decoder_inputs_embeds (torch.Tensor, optional): The decoder input embeddings tensor of shape
                (batch_size, decoder_sequence_length, hidden_size). Defaults to None.
            labels (torch.Tensor, optional): The labels tensor of shape (batch_size, sequence_length).
                Defaults to None.
            use_cache (bool, optional): Whether to use cache for the model.
                Defaults to None.
            output_attentions (bool, optional): Whether to output attentions.
                Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states.
                Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary as the output.
                Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1), ignore_index=-100)
            # TODO(thom): Add z_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
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
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        Args:
            self (T5ForConditionalGeneration): The instance of the T5ForConditionalGeneration class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length) containing input IDs.
            past_key_values (tuple, optional): The tuple of past key values for the transformer decoder. Default is None.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length)
                indicating which tokens to attend to. Default is None.
            head_mask (torch.Tensor, optional): The head mask tensor of shape (num_layers, num_heads) indicating
                which heads to mask. Default is None.
            decoder_head_mask (torch.Tensor, optional): The decoder head mask tensor of shape (num_layers, num_heads)
                indicating which decoder heads to mask. Default is None.
            decoder_attention_mask (torch.Tensor, optional): The decoder attention mask tensor of shape
                (batch_size, sequence_length) indicating which tokens to attend to in the decoder. Default is None.
            cross_attn_head_mask (torch.Tensor, optional): The cross-attention head mask tensor of shape
                (num_layers, num_heads) indicating which cross-attention heads to mask. Default is None.
            use_cache (bool, optional): Whether to use cache. Default is None.
            encoder_outputs (torch.Tensor, optional): The encoder outputs tensor of shape
                (batch_size, sequence_length, hidden_size) containing the hidden states of the encoder. Default is None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'decoder_input_ids' (torch.Tensor): The decoder input tensor of shape (batch_size, sequence_length)
                containing input IDs.
                - 'past_key_values' (tuple): The tuple of past key values for the transformer decoder.
                - 'encoder_outputs' (torch.Tensor): The encoder outputs tensor of shape
                (batch_size, sequence_length, hidden_size) containing the hidden states of the encoder.
                - 'attention_mask' (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length)
                indicating which tokens to attend to.
                - 'head_mask' (torch.Tensor): The head mask tensor of shape (num_layers, num_heads) indicating
                which heads to mask.
                - 'decoder_head_mask' (torch.Tensor): The decoder head mask tensor of shape (num_layers, num_heads)
                indicating which decoder heads to mask.
                - 'decoder_attention_mask' (torch.Tensor): The decoder attention mask tensor of shape
                (batch_size, sequence_length) indicating which tokens to attend to in the decoder.
                - 'cross_attn_head_mask' (torch.Tensor): The cross-attention head mask tensor of shape
                (num_layers, num_heads) indicating which cross-attention heads to mask.
                - 'use_cache' (bool): Whether to use cache.

        Raises:
            None.
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare decoder input ids from labels.

        This method is used to prepare the input ids for the decoder by shifting the given labels sequence to the right.

        Args:
            self (T5ForConditionalGeneration): An instance of the T5ForConditionalGeneration class.
            labels (mindspore.Tensor): The labels tensor containing the sequence of labels.

        Returns:
            None: This method modifies the decoder input ids in-place.

        Raises:
            None.

        """
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        This method '_reorder_cache' is defined within the class 'T5ForConditionalGeneration' and is used to reorder
        the cache for decoding during the T5 model's conditional generation.

        Args:
            self (object): The instance of the class.
            past_key_values (tuple): The past key value states generated during the model's previous decoding steps.
                If set to None, a warning is logged to consider setting `use_cache=True` to speed up decoding.
            beam_idx (tensor): The indices of the beam to reorder the cache.

        Returns:
            tuple: The reordered past key value states for the decoder. If 'past_key_values' is None, it returns None.

        Raises:
            ValueError: If the shape of the reordered layer past states and the original layer past states mismatch.
            ValueError: If the length of the reordered layer past states and the original layer past states mismatch.
        """
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class T5EncoderModel(T5PreTrainedModel):
    """T5EncoderModel"""
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config):
        """
        Initializes a T5EncoderModel instance.

        Args:
            self: The T5EncoderModel instance itself.
            config (T5Config): An instance of T5Config containing the configuration parameters for the model.
                It specifies the configuration settings such as vocab_size and d_model.
                This parameter is required for configuring the T5EncoderModel.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings.

        This method is used to obtain the input embeddings for the T5EncoderModel class.

        Args:
            self: An instance of the T5EncoderModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the T5EncoderModel.

        Args:
            self (T5EncoderModel): The instance of the T5EncoderModel class.
            new_embeddings (torch.Tensor): The new input embeddings to be set.

        Returns:
            None

        Raises:
            None
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        Ties the weights of the word embeddings in the T5EncoderModel.

        Args:
            self (T5EncoderModel): An instance of the T5EncoderModel class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        """
        Get the encoder of the T5EncoderModel.

        Args:
            self (T5EncoderModel): An instance of the T5EncoderModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def construct(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        Constructs the T5EncoderModel.

        Args:
            self: The T5EncoderModel object.
            input_ids (optional): A tensor of shape (batch_size, sequence_length) containing the input token IDs.
                Defaults to None.
            attention_mask (optional): A tensor of shape (batch_size, sequence_length) containing the attention mask.
                Defaults to None.
            head_mask (optional): A tensor of shape (num_heads,) containing the head mask. Defaults to None.
            inputs_embeds (optional): A tensor of shape (batch_size, sequence_length, embedding_size)
                containing the input embeddings. Defaults to None.
            output_attentions (optional): A boolean indicating whether to return the attentions. Defaults to None.
            output_hidden_states (optional): A boolean indicating whether to return the hidden states. Defaults to None.
            return_dict (optional): A boolean indicating whether to return a dictionary. If not provided,
                it is determined by self.config.use_return_dict. Defaults to None.

        Returns:
            encoder_outputs: A tuple containing the encoder outputs.
                It typically consists of the following elements:

                - last_hidden_state: A tensor of shape (batch_size, sequence_length, hidden_size) containing the last
                hidden state of the encoder.
                - hidden_states: A tuple of tensors containing all the hidden states of the encoder. Each tensor has
                a shape of (batch_size, sequence_length, hidden_size).
                - attentions: A tuple of tensors containing the attentions of the encoder. Each tensor has a shape of
                (batch_size, num_heads, sequence_length, sequence_length).

        Raises:
            None.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class T5ForSequenceClassification(T5PreTrainedModel):

    """
    T5ForSequenceClassification class implements a T5 model for sequence classification tasks.
    It inherits from the T5PreTrainedModel class.

    This class includes methods for initializing the model with a T5 configuration, constructing the model for
    sequence classification tasks, and computing the loss based on the provided labels.

    The __init__ method initializes the T5ForSequenceClassification instance with a T5 configuration.
    The construct method constructs the model for sequence classification tasks and returns the computed loss and logits.

    The construct method takes various input arguments such as input_ids, attention_mask, decoder_input_ids, labels,
    and other optional parameters to customize the behavior of the model during inference.

    If labels are provided, the model computes the loss based on the problem type specified in the T5 configuration.
    The loss can be computed for regression, single-label classification, or multi-label classification tasks.

    This class provides flexibility in handling different types of sequence classification tasks and supports
    customization through the T5 configuration settings.

    """
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        """
        Initializes an instance of the T5ForSequenceClassification class.

        Args:
            self: An instance of the T5ForSequenceClassification class.
            config (T5Config): The configuration object that contains the model's hyperparameters and settings.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes an instance of the T5ForSequenceClassification class by setting up the necessary
            components for sequence classification tasks. It takes in the self parameter, which refers to the instance
            of the class itself, and the config parameter, which is an instance of the T5Config class.

            The config parameter is of type T5Config and represents the configuration object that contains various
            hyperparameters and settings for the T5 model. It is used to initialize the transformer and
            classification_head attributes of the T5ForSequenceClassification instance.

            The transformer attribute is of type T5Model and is responsible for the main transformer model used for
            sequence classification. It is initialized with the provided config object.

            The classification_head attribute is of type T5ClassificationHead and represents the classification head
            that is added on top of the transformer model. It is also initialized with the provided config object.

            After initializing the transformer and classification_head attributes, the post_init method is called to
            perform any additional setup or customization required.

        Note:
            This method is automatically called when creating a new instance of the T5ForSequenceClassification class.
        """
        super().__init__(config)
        self.transformer = T5Model(config)
        self.classification_head = T5ClassificationHead(config)
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

        Returns:
            Union[Tuple, Seq2SeqSequenceClassifierOutput]
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
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

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(ops.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask].view(batch_size, -1, hidden_size)[:, -1, :]
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


class T5ForQuestionAnswering(T5PreTrainedModel):

    """
    This class represents a T5 model for question answering tasks. It is designed specifically for question answering
    applications where the model takes input text and outputs answers to questions posed about the input.
    The model architecture includes an encoder and a decoder, both based on the T5Stack structure.
    The T5ForQuestionAnswering class provides methods for setting input embeddings, tying weights, accessing the encoder
    and decoder components, and constructing the model for inference or training.

    The constructor initializes the T5ForQuestionAnswering model with a T5Config object, setting up the model dimensions,
    shared embeddings, encoder, decoder, and other necessary components. The model can be fine-tuned for specific
    question answering tasks by adjusting configurations and utilizing the provided methods.

    The construct method executes the forward pass of the model, taking input tensors and generating outputs for
    question answering. It handles input embeddings, attention masks, decoder inputs, and various optional arguments
    to control the model's behavior during inference or training. The method returns the model's output,
    including predicted start and end positions for answering questions, loss values, and other relevant information.

    Overall, the T5ForQuestionAnswering class encapsulates a T5 model tailored for question answering tasks,
    providing a convenient interface for utilizing and fine-tuning the model for specific applications.
    """
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        """
        Initializes an instance of the T5ForQuestionAnswering class.

        Args:
            self: The instance of the class.
            config (T5Config):
                The configuration object that defines the model's parameters.

                - The config parameter must be an instance of the T5Config class.
                - It is used to set up the model's architecture and hyperparameters.
                - This parameter is required.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)

        self.num_labels = config.num_labels
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        '''
        Description:
            This method returns the shared input embeddings of the T5 model for question answering.
        
        Args:
            self: The instance of the T5ForQuestionAnswering class.
        
        Returns:
            None
        
        Raises:
            None
        '''
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set new input embeddings for the T5 model used for Question Answering.
        
        Args:
            self (T5ForQuestionAnswering): The instance of the T5ForQuestionAnswering class.
                This parameter is automatically passed and refers to the current instance of the class.
            new_embeddings (object): The new input embeddings to be set for the model.
                This parameter represents the embeddings that will replace the existing ones in the model.
                
        Returns:
            None.
        
        Raises:
            None.
        """
        self.shared = new_embeddings
        # self.encoder.set_input_embeddings(new_embeddings)
        # self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        Ties the weights of the word embeddings in the T5ForQuestionAnswering model.
        
        Args:
            self: An instance of the T5ForQuestionAnswering class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        """
        Returns the encoder used for T5 question answering.
        
        Args:
            self: An instance of the T5ForQuestionAnswering class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder for the T5 model used for question answering.
        
        Args:
            self: An instance of the T5ForQuestionAnswering class.
        
        Returns:
            None.
        
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
        encoder_outputs: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], Seq2SeqQuestionAnsweringModelOutput]:
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

        Returns:
            `Union[Tuple[mindspore.Tensor], Seq2SeqQuestionAnsweringModelOutput]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if start_positions is not None and end_positions is not None:
            use_cache = False

        # Copied from models.bart.modeling_bart.BartModel.forward
        #   different to other models, T5 automatically creates decoder_input_ids from
        #   input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn("""
                The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
                `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
                If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = ops.ones(num_layers,
                num_heads)`.
                """, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

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
            output = (start_logits, end_logits) + decoder_outputs[1:] + encoder_outputs
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

__all__ = [
    "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
    "T5EncoderModel",
    "T5ForConditionalGeneration",
    "T5Model",
    "T5PreTrainedModel",
    "T5ForQuestionAnswering",
    "T5ForSequenceClassification",
]
