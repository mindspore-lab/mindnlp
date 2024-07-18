# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" MindSpore ChatGLM model on Graph mode. """

import copy
import warnings
import re
from typing import Optional, Tuple, List, Callable, Dict, Any

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor

from mindnlp.utils import logging
from mindnlp.modules.functional import embedding
from ...modeling_utils import PreTrainedModel
from ...generation.logits_process import LogitsProcessor
from ...generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from .configuration_chatglm import ChatGLMConfig

logger = logging.get_logger(__name__)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    """Invalid Score Processer."""
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method processes invalid score logits in the InvalidScoreLogitsProcessor class.
        
        Args:
            self (object): The instance of the class.
            input_ids (mindspore.Tensor): A tensor containing input IDs.
            scores (mindspore.Tensor): A tensor containing scores to be processed.
                It is expected to be a valid tensor.
        
        Returns:
            mindspore.Tensor: A tensor representing the processed scores.
                If any values in the input 'scores' tensor are NaN or Inf,
            they are replaced with zeros, and the value at index 5 is set to 50000.0.
        
        Raises:
            None
        """
        if ops.isnan(scores).any() or ops.isinf(scores).any():
            scores = ops.zeros_like(scores)
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(nn.Cell):
    """
    The model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self, config):
        """
        Initializes the PrefixEncoder class.
        
        Args:
            self: The instance of the class.
            config: An object containing configuration parameters.
                It should have the following attributes:

                - prefix_projection (bool): A flag indicating whether to use prefix projection.
                - pre_seq_len (int): The length of the input sequence.
                - hidden_size (int): The size of the hidden layers.
                - num_layers (int): The number of layers.

        Returns:
            None.

        Raises:
            AttributeError: If the config object is missing any of the required attributes.
            ValueError: If the config attributes are not of the expected types or do not meet the specified restrictions.
            TypeError: If the config object is not of the expected type.
        """
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = nn.SequentialCell(
                nn.Dense(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Dense(config.hidden_size, config.num_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = nn.Embedding(config.pre_seq_len, config.num_layers * config.hidden_size * 2)

    def construct(self, prefix: mindspore.Tensor):
        """
        Constructs past key values for the PrefixEncoder.

        Args:
            self (PrefixEncoder): An instance of the PrefixEncoder class.
            prefix (mindspore.Tensor): The input prefix tensor.

        Returns:
            The past key values for the PrefixEncoder.

        Raises:
            None.
        """
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


class RotaryEmbedding(nn.Cell):
    """Rotary Embedding."""
    def __init__(self, dim, base=10000, precision=mindspore.float16, max_seq_len=2048):
        """
        Initializes an instance of the RotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimensionality of the embeddings.
            base (int, optional): The base value used for calculating inverse frequencies. Defaults to 10000.
            precision (mindspore.dtype, optional): The data type precision of the embeddings. Defaults to mindspore.float16.
            max_seq_len (int, optional): The maximum sequence length. Defaults to 2048.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        inv_freq = 1. / (base ** (np.arange(0, dim, 2) / dim))
        t = np.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = np.expand_dims(np.cos(emb), 1)
        self.sin_cached = np.expand_dims(np.sin(emb), 1)
        self.cos_cached = Tensor(self.cos_cached, precision)
        self.sin_cached = Tensor(self.sin_cached, precision)

    def construct(self, seq_len):
        """
        Constructs and returns the cached cosine and sine arrays of the specified length for the RotaryEmbedding class.

        Args:
            self (RotaryEmbedding): An instance of the RotaryEmbedding class.
            seq_len (int): The length of the sequence for which the cosine and sine arrays should be constructed.

        Returns:
            None

        Raises:
            None.

        """
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x):
    """rotate half tensor."""
    x1, x2 = x.chunk(2, x.ndim - 1)
    return ops.cat((-x2, x1), axis=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    """apply rotary pos"""
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        embedding(position_id, sin.squeeze(1)).unsqueeze(2)

    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


def default_init(cls, *args, **kwargs):
    """default init"""
    return cls(*args, **kwargs)


class SelfAttention(nn.Cell):
    """Self Attention."""
    def __init__(self, config, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=mindspore.float32, position_encoding_2d=True):
        """
        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the maximum sequence length.
            hidden_size (int): The size of the hidden state.
            num_attention_heads (int): The number of attention heads.
            layer_id (int): The ID of the layer.
            hidden_size_per_attention_head (int, optional): The size of the hidden state per attention head. Defaults to None.
            bias (bool): A flag indicating whether to use bias in the dense layers.
            params_dtype (object): The data type of the parameters. Defaults to mindspore.float32.
            position_encoding_2d (bool): A flag indicating whether to use 2D position encoding.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size_per_attention_head is provided and is not compatible with the hidden_size
                and num_attention_heads.
            ValueError: If config.max_sequence_length is not provided or is invalid.
            TypeError: If the data type of the parameters is not supported.
        """
        super().__init__()

        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hidden_size_per_partition = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads
        self.position_encoding_2d = position_encoding_2d
        self.rotary_emb = RotaryEmbedding(
            self.hidden_size // (self.num_attention_heads * 2)
            if position_encoding_2d
            else self.hidden_size // self.num_attention_heads,
            base=10000,
            precision=mindspore.float16,
            max_seq_len=config.max_sequence_length
        )

        self.scale_mask_softmax = None

        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head

        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head

        # Strided linear layer.
        self.query_key_value = nn.Dense(
            hidden_size,
            3 * self.inner_hidden_size,
            has_bias=bias,
            dtype=params_dtype,
        )

        self.dense = nn.Dense(
            self.inner_hidden_size,
            hidden_size,
            has_bias=bias,
            dtype=params_dtype,
        )

    @staticmethod
    def attention_mask_func(attention_scores, attention_mask):
        """attention mask function"""
        return attention_scores.masked_fill(attention_mask, -10000.0)

    def split_tensor_along_last_dim(self, tensor, num_partitions):
        """Split a tensor along its last dimension.
        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous in memory.
        """
        # Get the size and dimension.
        last_dim = tensor.ndim - 1
        last_dim_size = tensor.shape[last_dim] // num_partitions
        # Split.
        tensor_list = ops.split(tensor, last_dim_size, axis=last_dim)
        # Note: torch.split does not create contiguous tensors by default.

        return tensor_list

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            position_ids,
            attention_mask: mindspore.Tensor,
            layer_id,
            layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states: [seq_len, batch, hidden_size]
            attention_mask: [(1, 1), seq_len, seq_len]
        """
        # [seq_len, batch, 3 * hidden_size]
        mixed_raw_layer = self.query_key_value(hidden_states)
        # [seq_len, batch, 3 * hidden_size] --> [seq_len, batch, num_attention_heads, 3 * hidden_size_per_attention_head]
        new_tensor_shape = mixed_raw_layer.shape[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(new_tensor_shape)
        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, axis=query_layer.ndim - 1)
            k1, k2 = key_layer.chunk(2, axis=key_layer.ndim - 1)
            # return (k1,)
            cos, sin = self.rotary_emb(position_ids.max() + 1)
            position_ids, block_position_ids = position_ids[:, 0, :].swapaxes(0, 1), \
                position_ids[:, 1, :].swapaxes(0, 1)

            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)

            query_layer = ops.concat([q1, q2], axis=3)
            key_layer = ops.concat([k1, k2], axis=3)
        else:
            position_ids = position_ids.swapaxes(0, 1)
            cos, sin = self.rotary_emb(position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)
        # [seq_len, batch, hidden_size]

        context_layer, present, attention_probs = self.attention_fn(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask=attention_mask,
            hidden_size_per_partition=self.hidden_size_per_partition,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache
        )

        output = self.dense(context_layer)

        outputs = (output, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs  # output, present, attention_probs

    def attention_fn(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            hidden_size_per_partition,
            layer_id,
            layer_past=None,
            scaling_attention_score=True,
            use_cache=False,
    ):
        """attention function."""
        seq_len = query_layer.shape[0]
        if layer_past is not None and seq_len == 1:
            # layer_past = layer_past.chunk(2, 0)
            past_key, past_value = layer_past[0], layer_past[1]
            key_layer = ops.cat((past_key, key_layer), axis=0)
            value_layer = ops.cat((past_value, value_layer), axis=0)

        # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
        hidden_size = key_layer.shape[-1]

        if use_cache:
            present = (key_layer, value_layer)
            present = ops.stack(present)
        else:
            present = None

        query_key_layer_scaling_coeff = ops.cast(layer_id + 1, query_layer.dtype)
        if scaling_attention_score:
            query_layer = query_layer / (ops.sqrt(ops.cast(hidden_size, query_layer.dtype)) * query_key_layer_scaling_coeff)

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view((output_size[2], output_size[0] * output_size[1], -1))
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view((output_size[3], output_size[0] * output_size[1], -1))

        matmul_result = ops.bmm(
            query_layer.swapaxes(0, 1),  # [b * np, sq, hn]
            key_layer.permute(1, 2, 0),  # [b * np, hn, sk]
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size)

        # if self.scale_mask_softmax:
        #     self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
        #     attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        # else:
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores = attention_scores.masked_fill(attention_mask, -60000.0 / query_key_layer_scaling_coeff)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff

        attention_probs = ops.softmax(attention_scores, axis=-1)

        attention_probs = attention_probs.astype(dtype)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.shape[0], output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = ops.bmm(attention_probs, value_layer.swapaxes(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3)

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.shape[:-2] + (hidden_size_per_partition,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, present, attention_probs)

        return outputs

def gelu(x):
    """gelu"""
    return ops.gelu(x, approximate='tanh')


class GEGLU(nn.Cell):
    """GEGLU"""
    def __init__(self):
        """
        Initializes an instance of the GEGLU class.

        Args:
            self: The instance of the GEGLU class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.activation_fn = ops.gelu

    def construct(self, x):
        """
        Constructs a GEGLU object.

        Args:
            self (GEGLU): The instance of the GEGLU class.
            x (Tensor): The input tensor to be processed. It should have a dimension greater than or equal to 1.

        Returns:
            None: The method modifies the instance's internal state.

        Raises:
            ValueError: If the input tensor `x` has a dimension less than 1.

        This method takes an input tensor `x` and performs the GEGLU operation on it.
        The GEGLU operation splits the input tensor into two parts, `x1` and `x2`, along the last axis, and multiplies `x1` with
        the activation function applied to `x2`. The resulting tensor is stored internally in the GEGLU object.

        Note that the activation function used for the GEGLU operation is defined during the instantiation of the GEGLU object.

        Example:
            ```python
            >>> g = GEGLU()
            >>> x = torch.tensor([1, 2, 3, 4])
            >>> g.construct(x)
            ```
        """
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, axis=x.ndim - 1)
        return x1 * self.activation_fn(x2)


class GLU(nn.Cell):
    """GLU"""
    def __init__(self, hidden_size, inner_hidden_size=None,
                 layer_id=None, bias=True, activation_func=gelu, params_dtype=mindspore.float32):
        """
        Initializes an instance of the GLU class.

        Args:
            hidden_size (int): The size of the hidden layer.
            inner_hidden_size (int, optional): The size of the inner hidden layer.
                If not provided, it defaults to 4 times the hidden size.
            layer_id (int, optional): The ID of the layer. Defaults to None.
            bias (bool, optional): Indicates whether bias should be included in the dense layers. Defaults to True.
            activation_func (function, optional): The activation function to be used. Defaults to gelu.
            params_dtype (mindspore.dtype, optional): The data type of the parameters. Defaults to mindspore.float32.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func

        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = nn.Dense(
            self.hidden_size,
            self.inner_hidden_size,
            has_bias=bias,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = nn.Dense(
            self.inner_hidden_size,
            self.hidden_size,
            has_bias=bias,
            dtype=params_dtype,
        )

    def construct(self, hidden_states):
        """
        Args:
            hidden_states: [seq_len, batch, hidden_size]
        """
        # [seq_len, batch, inner_hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLMBlock(nn.Cell):
    """GLM Block."""
    def __init__(
            self,
            config,
            hidden_size,
            num_attention_heads,
            layernorm_epsilon,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            use_bias=True,
            params_dtype=mindspore.float32,
            num_layers=28,
            position_encoding_2d=True,
    ):
        """
        Initializes a GLMBlock object.

        Args:
            self: The object itself.
            config: Configuration object.
            hidden_size (int): The size of the hidden layer.
            num_attention_heads (int): The number of attention heads.
            layernorm_epsilon (float): The epsilon value for layer normalization.
            layer_id (int): The id of the layer.
            inner_hidden_size (int, optional): The size of the inner hidden layer. Defaults to None.
            hidden_size_per_attention_head (int, optional): The size of the hidden layer per attention head. Defaults to None.
            use_bias (bool, optional): Whether to use bias in the layers. Defaults to True.
            params_dtype (mindspore.dtype, optional): The data type of the parameters. Defaults to mindspore.float32.
            num_layers (int, optional): The number of layers. Defaults to 28.
            position_encoding_2d (bool, optional): Whether to use 2D position encoding. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        # Set output layer initialization if not provided.

        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = nn.LayerNorm([hidden_size], epsilon=layernorm_epsilon)

        self.position_encoding_2d = position_encoding_2d

        # Self attention.
        self.attention = SelfAttention(
            config,
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
            params_dtype=params_dtype,
            position_encoding_2d=self.position_encoding_2d,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = nn.LayerNorm([hidden_size], epsilon=layernorm_epsilon)

        self.num_layers = num_layers

        # GLU
        self.mlp = GLU(
            hidden_size,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            params_dtype=params_dtype,
        )

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            position_ids,
            attention_mask: mindspore.Tensor,
            layer_id,
            layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states: [seq_len, batch, hidden_size]
            attention_mask: [(1, 1), seq_len, seq_len]
        """
        # Layer norm at the begining of the transformer layer.
        # [seq_len, batch, hidden_size]
        attention_input = self.input_layernorm(hidden_states)

        # Self attention.
        attention_outputs = self.attention(
            attention_input,
            position_ids,
            attention_mask=attention_mask,
            layer_id=layer_id,
            layer_past=layer_past,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        # return attention_outputs

        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = attention_input * alpha + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class MSChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    is_parallelizable = False
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]
    _keys_to_ignore_on_load_unexpected = [r'inv_freq']

    def _init_weights(self, cell: nn.Cell):
        """Initialize the weights."""
    def get_masks(self, input_ids):
        """get masks"""
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.asnumpy().tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = ops.ones((batch_size, seq_length, seq_length))
        attention_mask = attention_mask.tril()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, use_gmasks=None):
        """get position ids"""
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.asnumpy().tolist().index(self.config.bos_token_id) for seq in input_ids]
        if self.position_encoding_2d:
            position_ids = ops.arange(seq_length, dtype=mindspore.int64).unsqueeze(0).tile((batch_size, 1))
            for i, context_length in enumerate(context_lengths):
                position_ids[i, context_length:] = mask_positions[i]
            block_position_ids = [ops.cat((
                ops.zeros(context_length, dtype=mindspore.int64),
                ops.arange(seq_length - context_length, dtype=mindspore.int64) + 1
            )) for context_length in context_lengths]
            block_position_ids = ops.stack(block_position_ids, axis=0)
            position_ids = ops.stack((position_ids, block_position_ids), axis=1)
        else:
            position_ids = ops.arange(seq_length, dtype=mindspore.int64).unsqueeze(0).tile((batch_size, 1))
            for i, context_length in enumerate(context_lengths):
                if not use_gmasks[i]:
                    position_ids[i, context_length:] = mask_positions[i]

        return position_ids


class MSChatGLMModel(MSChatGLMPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    def __init__(self, config: ChatGLMConfig):
        """
        Initializes an instance of the MSChatGLMModel class with the provided configuration.

        Args:
            self: The instance of the MSChatGLMModel class.
            config (ChatGLMConfig):
                The configuration for the model.

                - max_sequence_length (int): The maximum sequence length for the input.
                - hidden_size (int): The size of the hidden layer.
                - num_attention_heads (int): The number of attention heads.
                - vocab_size (int): The size of the vocabulary.
                - num_layers (int): The number of layers for the model.
                - layernorm_epsilon (float): The epsilon value for the layer normalization.
                - inner_hidden_size (int): The size of the inner hidden layer.
                - position_encoding_2d (bool): Whether to use 2D position encoding.
                - pre_seq_len (int): The length of the prefix sequence.
                - prefix_projection (bool): Whether to use prefix projection.
                - use_cache (bool): Whether to use cache.
                - output_hidden_states (bool): Whether to output hidden states.
                - output_attentions (bool): Whether to output attentions.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        # recording parameters
        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.params_dtype = mindspore.float16
        self.num_attention_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.layernorm_epsilon = config.layernorm_epsilon
        self.inner_hidden_size = config.inner_hidden_size
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.position_encoding_2d = config.position_encoding_2d
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection

        self.use_cache = config.use_cache
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.hidden_size,
            dtype=self.params_dtype
        )

        def get_layer(layer_id):
            return GLMBlock(
                config,
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                use_bias=True,
                params_dtype=self.params_dtype,
                position_encoding_2d=self.position_encoding_2d,
            )

        self.layers = nn.CellList(
            [get_layer(layer_id) for layer_id in range(self.num_layers)]
        )
        # Final layer norm before output.
        self.final_layernorm = nn.LayerNorm([self.hidden_size], epsilon=self.layernorm_epsilon)

        if self.pre_seq_len is not None:
            # for param in self.parameters():
            #     param.requires_grad = False
            self.prefix_tokens = Tensor(np.arange(self.pre_seq_len))
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = nn.Dropout(p=0.1)

            # total_params = sum(p.numel() for p in self.parameters())
            # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            # print("Using p-tuning v2: # trainable_params = {} / {}".format(trainable_params, total_params))

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings for the MSChatGLMModel.

        Args:
            self (MSChatGLMModel): An instance of the MSChatGLMModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        """
        Sets the input embeddings for the MSChatGLMModel.

        Args:
            self (MSChatGLMModel): The instance of the MSChatGLMModel class.
            new_embeddings (mindspore.Tensor): The new embeddings to be set as input.
                It should be a tensor object representing the word embeddings.

        Returns:
            None.

        Raises:
            None.

        Note:
            The input embeddings are used for representing words in the MSChatGLMModel.
            By setting new embeddings, the model can be fine-tuned or customized to use different word representations.
        """
        self.word_embeddings = new_embeddings

    def get_prompt(self, batch_size, dtype=mindspore.float16):
        """get prompt."""
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        # past_key_values = [(v[0], v[1]) for v in past_key_values]
        return past_key_values

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor, ...]:
        """Constructs the MSChatGLMModel.

        This method is used to construct the MSChatGLMModel. It takes in several parameters and returns a tuple of tensors.

        Args:
            self (MSChatGLMModel): The instance of the MSChatGLMModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor representing the tokenized input sequences. Default is None.
            position_ids (Optional[mindspore.Tensor]): The input tensor representing the position ids of the tokens. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The input tensor representing the attention mask. Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]]):
                The input tensor representing the past key values. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The input tensor representing the embedded input sequences. Default is None.

        Returns:
            Tuple[mindspore.Tensor, ...]: A tuple containing the hidden states, presents, all hidden states, and all self attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, _ = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if past_key_values is None:
            if self.pre_seq_len is not None:
                past_key_values = self.get_prompt(batch_size=input_ids.shape[0],
                                                  dtype=inputs_embeds.dtype)
            else:
                past_key_values = tuple([None] * len(self.layers))

            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                )

            if position_ids is None:
                MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
                seqs = input_ids.asnumpy().tolist()

                mask_positions, use_gmasks = [], []
                for seq in seqs:
                    mask_token = gMASK if gMASK in seq else MASK
                    use_gmask = mask_token == gMASK
                    mask_positions.append(seq.index(mask_token))
                    use_gmasks.append(use_gmask)

                position_ids = self.get_position_ids(
                    input_ids,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks
                )

        if self.pre_seq_len is not None and attention_mask is not None:
            prefix_attention_mask = ops.ones((batch_size, 1, input_ids.shape[-1], self.pre_seq_len))
            prefix_attention_mask = (prefix_attention_mask < 0.5).bool()
            attention_mask = ops.cat((prefix_attention_mask, attention_mask), axis=3)

        # [seq_len, batch, hidden_size]
        hidden_states = inputs_embeds.swapaxes(0, 1)

        presents = ()
        all_self_attentions = ()
        all_hidden_states = ()

        if attention_mask is None:
            attention_mask = ops.zeros((1, 1)).bool()

        # past_key_values = past_key_values.chunk(self.num_layers, 0)
        for i, layer in enumerate(self.layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values[i]

            layer_ret = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                layer_id=mindspore.Tensor(i),
                layer_past=layer_past,
                use_cache=self.use_cache,
                output_attentions=self.output_attentions
            )
            hidden_states = layer_ret[0]

            if self.use_cache:
                presents = presents + (layer_ret[1],)

            if self.output_attentions:
                idx = 2 if self.use_cache else 1
                all_self_attentions = all_self_attentions + (layer_ret[idx],)

        # Final layer norm.
        # return (hidden_states,)
        hidden_states = self.final_layernorm(hidden_states)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.use_cache:
            presents = ops.stack(presents)

        return (hidden_states, presents, all_hidden_states, all_self_attentions)


class MSChatGLMForConditionalGeneration(MSChatGLMPreTrainedModel):
    """MSChatGLMForConditionalGeneration"""
    def __init__(self, config: ChatGLMConfig):
        """
        Initializes an instance of the MSChatGLMForConditionalGeneration class.

        Args:
            self: The instance of the MSChatGLMForConditionalGeneration class.
            config (ChatGLMConfig):
                An object of type ChatGLMConfig containing configuration parameters for the model.

                - max_sequence_length (int): The maximum length of input sequences.
                - position_encoding_2d (bool): Flag indicating whether to use 2D position encoding.
                - quantization_bit (int): Number of bits to use for quantization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.max_sequence_length = config.max_sequence_length
        self.position_encoding_2d = config.position_encoding_2d
        self.transformer = MSChatGLMModel(config)
        self.lm_head = nn.Dense(
            config.hidden_size,
            config.vocab_size,
            has_bias=False,
            dtype=mindspore.float16
        )
        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MSChatGLMForConditionalGeneration model.

        Args:
            self: The instance of the MSChatGLMForConditionalGeneration class.

        Returns:
            returns the output embeddings of the model as a tensor.

        Raises:
            None.

        This method retrieves the output embeddings of the MSChatGLMForConditionalGeneration model.
        The output embeddings are the final representations of the input tokens after being processed by the model's
        language model head. The embeddings are returned as a tensor.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the MSChatGLMForConditionalGeneration model.

        Args:
            self (MSChatGLMForConditionalGeneration): The instance of the MSChatGLMForConditionalGeneration class.
            new_embeddings (object): The new embeddings to be set as the output embeddings for the model.
                It can be of any valid type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """
        This method '_update_model_kwargs_for_generation' in the class 'MSChatGLMForConditionalGeneration' updates
        the model_kwargs for generation based on the provided outputs and other parameters.

        Args:
            self: The instance of the class.
            outputs: The model outputs that are used to update the model_kwargs.
            model_kwargs (Dict[str, Any]): A dictionary containing keyword arguments for the model.
            is_encoder_decoder (bool): A boolean indicating whether the model is an encoder-decoder model. Default is False.
            standardize_cache_format (bool): A boolean indicating whether to standardize the cache format. Default is False.

        Returns:
            Dict[str, Any]: A dictionary containing updated keyword arguments for the model.

        Raises:
            ValueError: If the provided attention_mask has an unsupported data type.
            IndexError: If there are issues with indexing while updating position_ids.
        """
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if attention_mask is not None and attention_mask.dtype == mindspore.bool_:
                attention_mask = ops.cat(
                    [attention_mask, attention_mask.new_ones((*attention_mask.shape[:3], 1))], axis=3)
                new_attention_mask = attention_mask[:, :, -1:].copy()
                new_attention_mask[..., -1] = False
                model_kwargs["attention_mask"] = ops.cat(
                    [attention_mask, new_attention_mask], axis=2
                )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].copy()
            new_position_id[:, 1, :] += 1
            model_kwargs["position_ids"] = ops.cat(
                [position_ids, new_position_id], axis=-1
            )

        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: mindspore.Tensor,
            past: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            **kwargs
    ) -> dict:
        """
        This method prepares inputs for generation in the MSChatGLMForConditionalGeneration class.

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor): The input tensor containing token ids.
            past (Optional[mindspore.Tensor]): The past state tensor (default is None).
            past_key_values (Optional[mindspore.Tensor]): The past key values tensor (default is None).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor (default is None).
            position_ids (Optional[mindspore.Tensor]): The position ids tensor (default is None).
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the prepared inputs for generation including 'input_ids', 'past_key_values',
                'position_ids', and 'attention_mask'.

        Raises:
            TypeError: If the input arguments are not of the expected types.
            ValueError: If there are issues with the input data or configuration.
            IndexError: If there are index out of bounds errors during processing.
            Warning: If there are issues with the dtype of attention mask.
        """
        batch_size, seq_length = input_ids.shape

        if self.get_inputs() is None:
            self.set_inputs(
                Tensor(shape=[batch_size, None], dtype=mindspore.int64), # input_ids
                Tensor(shape=[batch_size, 2, None], dtype=mindspore.int64), # position_ids
                Tensor(shape=[batch_size, 1, None, None], dtype=mindspore.bool_), # attention_mask
                Tensor(shape=[self.config.num_layers, 2, None, batch_size, 32, 128], dtype=mindspore.float16) # past_key_values
            )
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.asnumpy().tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            # past_key_values = ops.stack([ops.stack(past_key_values[i]) for i in range(self.config.num_layers)])
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == mindspore.bool_:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None

            if attention_mask is None:
                attention_mask = ops.zeros((1, 1, 1, 1)).bool()

            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:
                    position_ids = mindspore.Tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in
                         zip(mask_positions, context_lengths)], dtype=mindspore.int64).unsqueeze(-1)
                else:
                    position_ids = mindspore.Tensor(mask_positions, dtype=mindspore.int64).unsqueeze(-1)

            if past is None:
                past = past_key_values
            return {
                "input_ids": last_token,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        else:
            if attention_mask is not None and attention_mask.dtype != mindspore.bool_:
                logger.warning_once(f"The dtype of attention mask ({attention_mask.dtype}) is not bool")
                attention_mask = None
            if attention_mask is None:
                attention_mask = self.get_masks(input_ids)
            if position_ids is None:
                position_ids = self.get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)

            past_key_values = ops.zeros((28, 2, input_ids.shape[1], 1, 32, 128), dtype=mindspore.float16)
            return {
                "input_ids": input_ids,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }

    def construct(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[mindspore.Tensor]] = None,
            **kwargs
    ):
        """
        Constructs the MSChatGLMForConditionalGeneration model.

        Args:
            self (MSChatGLMForConditionalGeneration): The instance of the MSChatGLMForConditionalGeneration class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor containing the tokenized input sequence. Default is None.
            position_ids (Optional[mindspore.Tensor]):
                The tensor containing the position indices for each token in the input sequence. Default is None.
            attention_mask (Optional[mindspore.Tensor]):
                The mask tensor indicating which elements in the input sequence should be attended to. Default is None.
            past_key_values (Optional[Tuple[mindspore.Tensor]]):
                The tuple of tensors containing the key-value pairs from the previous attention pass. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the following keys:

                - 'loss' (None): The loss value. Always None.
                - 'logits' (mindspore.Tensor): The output logits tensor of shape (batch_size, sequence_length, vocab_size).
                - 'past_key_values' (Tuple[mindspore.Tensor]): The tuple of tensors containing the key-value pairs from the current attention pass.
                - 'hidden_states' (mindspore.Tensor): The hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                - 'attentions' (mindspore.Tensor): The attention tensor of shape (batch_size, num_heads, sequence_length, sequence_length).
        
        Raises:
            None.
        """
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=None,
        )

        hidden_states = transformer_outputs[0]

        # return (hidden_states,)
        lm_logits = self.lm_head(hidden_states).permute(1, 0, 2)

        loss = None

        return {'loss': loss, 'logits': lm_logits,
                'past_key_values': transformer_outputs[1],
                'hidden_states': transformer_outputs[2],
                'attentions': transformer_outputs[3]
            }

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...], beam_idx: mindspore.Tensor
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx),
                layer_past[1].index_select(1, beam_idx),
            )
            for layer_past in past
        )

    def process_response(self, response):
        """process_response"""
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            [r"\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        """chat."""
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="ms")
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048,
                    do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        """stream chat"""
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="ms")
        for outputs in self.stream_generate(**inputs, **gen_kwargs):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            response = self.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, mindspore.Tensor], List[int]]] = None,
            **kwargs,
    ):
        """stream generate"""
        _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        _, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://hf-mirror.com/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill(1)
        scores = None

        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = ops.softmax(next_token_scores, axis=-1)
            if generation_config.do_sample:
                next_tokens = ops.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = ops.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = ops.cat([input_ids, next_tokens[:, None]], axis=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break
            yield input_ids

    def quantize(self, bits: int, empty_init=False, **kwargs):
        """TODO: support quantize"""
__all__ = [
    'MSChatGLMModel',
    'MSChatGLMPreTrainedModel',
    'MSChatGLMForConditionalGeneration'
]
