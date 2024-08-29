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
""" MindSpore ChatGLM2 model. """

import math
import copy
import warnings
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import mindspore
from mindspore import Parameter
from mindspore.common.api import _no_grad

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...ms_utils import zero_init
from ...modeling_utils import PreTrainedModel
from ...generation.logits_process import LogitsProcessor
from ...generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_chatglm2 import ChatGLM2Config

logger = logging.get_logger(__name__)


CHATGLM2_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://hf-mirror.com/models?filter=chatglm
]


def default_init(cls, *args, **kwargs):
    """
    Args:
        cls (class): The class to be instantiated.
        
    Returns:
        None: Returns an instance of the input class with the provided arguments and keyword arguments.
    
    Raises:
        None.
    """
    return cls(*args, **kwargs)


class InvalidScoreLogitsProcessor(LogitsProcessor):

    """
    This class represents an invalid score logits processor that handles invalid scores in the input logits.
    
    The InvalidScoreLogitsProcessor is a subclass of the LogitsProcessor class and provides functionality to process invalid score values. 
    Invalid score values include NaN (Not a Number) and infinity. This processor replaces all invalid scores in the input logits with zeros,
    except for the score at index 5 which is set to a high value of 50000.0.

    Example:
        ```python
        >>> processor = InvalidScoreLogitsProcessor()
        >>> processed_scores = processor(input_ids, scores)
        ```

    Attributes:
        None.

    Methods:
        __call__(input_ids, scores):
            Process the input logits and replace any invalid scores with zeros,
            except for the score at index 5 which is set to a high value of 50000.0.

    Example:
        ```python
        >>> # Create an instance of the InvalidScoreLogitsProcessor
        >>> processor = InvalidScoreLogitsProcessor()
        ...
        >>> # Process the input logits using the processor
        >>> input_ids = mindspore.Tensor(...)
        >>> scores = mindspore.Tensor(...)
        >>> processed_scores = processor(input_ids, scores)
        ...
        >>> # Use the processed scores for further computations
        >>> ...
        ```
    """
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        Applies the InvalidScoreLogitsProcessor to the input scores.

        Args:
            self: An instance of the InvalidScoreLogitsProcessor class.
            input_ids (mindspore.Tensor): The input tensor representing the IDs of the model inputs.
            scores (mindspore.Tensor): The input tensor representing the scores of the model outputs.

        Returns:
            mindspore.Tensor: The processed scores tensor after applying the InvalidScoreLogitsProcessor.

        Raises:
            None.

        Description:
            The '__call__' method of the InvalidScoreLogitsProcessor class applies a series of operations to the input scores tensor.
            If any of the scores are NaN (not-a-number) or infinite, the tensor is modified as follows:

                - A new tensor with the same shape and data type as the scores tensor is created, filled with zeros.
                - The element at index 5 of the last dimension of the new tensor is set to 50000.0.
            The processed scores tensor is then returned.

        Example:
            ```python
            >>> input_ids = mindspore.Tensor(...)
            >>> scores = mindspore.Tensor(...)
            >>> processor = InvalidScoreLogitsProcessor()
            >>> processed_scores = processor(input_ids, scores)
            >>> print(processed_scores)
            tensor([...])
            ```
        """
        if ops.isnan(scores).any() or ops.isinf(scores).any():
            scores = ops.zeros_like(scores)
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(nn.Module):
    """
    The nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self, config: ChatGLM2Config):
        """
        Initialize the PrefixEncoder object.

        Args:
            config (ChatGLM2Config):
                The configuration object that holds the parameters for the PrefixEncoder.

                - `prefix_projection` (bool): Flag indicating whether to use the prefix projection.
                - `num_layers` (int): The number of layers.
                - `kv_channels` (int): The number of channels in key-value projection.
                - `multi_query_group_num` (int): The number of multi-query groups.
                - `pre_seq_len` (int): The length of the prefix sequence.
                - `hidden_size` (int): The size of the hidden layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = nn.Sequential([
                nn.Linear(kv_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, kv_size)
            ])
        else:
            self.embedding = nn.Embedding(config.pre_seq_len,
                                          config.num_layers * config.kv_channels * config.multi_query_group_num * 2)

    def forward(self, prefix: mindspore.Tensor):
        """
        Construct the past key values for the PrefixEncoder.

        This method takes two parameters: self and prefix. It forwards the past key values based on the given prefix.

        Args:
            self (PrefixEncoder): An instance of the PrefixEncoder class.
            prefix (mindspore.Tensor): The prefix tensor used to forward the past key values.
                It can be either an embedding tensor or a token tensor.

                - If self.prefix_projection is True, prefix should be an embedding tensor.
                - If self.prefix_projection is False, prefix should be a token tensor.

        Returns:
            None.

        Raises:
            None: This method does not raise any exceptions.
        """
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def split_tensor_along_last_dim(
        tensor: mindspore.Tensor,
        num_partitions: int,
) -> List[mindspore.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.ndim - 1
    last_dim_size = tensor.shape[last_dim] // num_partitions
    # Split.
    tensor_list = ops.split(tensor, last_dim_size, dim=last_dim)
    return tensor_list


class RotaryEmbedding(nn.Module):

    """
    This class represents a Rotary Position Embedding module that enhances the Transformer model.
    It provides a mechanism to incorporate positional information into the model's input embeddings, improving its
    ability to understand the order and relationships between elements in a sequence.

    The RotaryEmbedding class inherits from the nn.Module class, a base class for all neural network modules in the MindSpore framework.

    The RotaryEmbedding class contains the following methods:

    - __init__(self, dim, original_impl=False, dtype=None): Initializes a RotaryEmbedding object.
    - forward_impl(self, seq_len: int, n_elem: int, dtype: mindspore.dtype, base: int = 10000): Constructs the rotary position embeddings.
    - forward(self, max_seq_len, offset=0): Constructs the rotary position embeddings with the given maximum sequence length.

    For more information on the RotaryEmbedding class and its usage, refer to the original implementation at:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/rope/__init__.py

    This implementation is licensed under the MIT License. For details, please refer to the license file at:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license
    """
    def __init__(self, dim, original_impl=False, dtype=None):
        """
        Initializes a RotaryEmbedding object.

        Args:
            self (RotaryEmbedding): The instance of the RotaryEmbedding class.
            dim (int): The dimensionality of the embedding space.
            original_impl (bool): Flag indicating whether to use the original implementation.
            dtype (str, optional): The data type to be used. Defaults to None.

        Returns:
            None.

        Raises:
            TypeError: If dim is not an integer or if original_impl is not a boolean.
        """
        super().__init__()
        inv_freq = 1.0 / (10000 ** (ops.arange(0, dim, 2).to(dtype=dtype) / dim))
        self.inv_freq = inv_freq
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: mindspore.dtype, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (ops.arange(0, n_elem, 2, dtype=dtype) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = ops.arange(seq_len, dtype=dtype)

        # Calculate the product of position index and $\theta_i$
        idx_theta = ops.outer(seq_idx, theta).float()

        cache = ops.stack([ops.cos(idx_theta), ops.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (mindspore.float16, mindspore.bfloat16, mindspore.int8):
            cache = cache.bfloat16() if dtype == mindspore.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        """
        Constructs a rotary embedding for a given maximum sequence length.

        Args:
            self (RotaryEmbedding): An instance of the RotaryEmbedding class.
            max_seq_len (int): The maximum length of the sequence to be embedded.
            offset (int, optional): The offset value to be used in the embedding forwardion. Defaults to 0.

        Returns:
            None.

        Raises:
            None.
        """
        return self.forward_impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype)


def apply_rotary_pos_emb(x: mindspore.Tensor, rope_cache: mindspore.Tensor) -> mindspore.Tensor:
    """
    Apply rotary positional embedding to the input tensor.

    Args:
        x (mindspore.Tensor): The input tensor of shape (sq, _, np, _).
        rope_cache (mindspore.Tensor): The cache tensor of shape (_, _, _, _).

    Returns:
        mindspore.Tensor: The transformed tensor after applying rotary positional embedding.

    Raises:
        None
    """
    # x: [sq, b, np, hn]
    sq, _, np, _ = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.shape[3], 2)
    x_out2 = ops.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(start_dim=3)
    return ops.cat((x_out2, x_pass), dim=-1)

LayerNorm = nn.LayerNorm

class RMSNorm(nn.Module):

    """
    RMSNorm is a normalization technique implemented as a Python class that inherits from nn.Module.
    It is designed to normalize the hidden states of a neural network using the Root Mean Square (RMS) method.

    The RMSNorm class has the following attributes:

    Attributes:
        weight (mindspore.Parameter): A trainable parameter representing the weight matrix used for scaling the normalized hidden states.
        eps (float): A small value added to the denominator to avoid division by zero.

    Methods:
        __init__(self, normalized_shape, eps=1e-05, dtype=None, **kwargs): Initializes the RMSNorm instance with the provided parameters.
        forward(self, hidden_states: mindspore.Tensor): Normalizes the given hidden states using the RMS method.

    Usage Example:
        ```python
        >>> import mindspore.ops as ops
        >>> import mindspore.nn as nn
        >>> import mindspore
        ...
        >>> class RMSNorm(nn.Module):
        >>>     def __init__(self, normalized_shape, eps=1e-05, dtype=None, **kwargs):
        >>>         super().__init__()
        >>>         self.weight = mindspore.Parameter(ops.zeros(normalized_shape, dtype=dtype))
        >>>         self.eps = eps
        ...
        >>>     def forward(self, hidden_states: mindspore.Tensor):
        >>>         input_dtype = hidden_states.dtype
        >>>         variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        >>>         hidden_states = hidden_states * ops.rsqrt(variance + self.eps)
        >>>         return (self.weight * hidden_states).to(input_dtype)
        ```
    """
    def __init__(self, normalized_shape, eps=1e-5, dtype=None, **kwargs):
        '''
        Initializes an instance of the RMSNorm class.

        Args:
            self (RMSNorm): The instance of the RMSNorm class.
            normalized_shape (tuple): The shape of the input tensor, excluding the batch dimension.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-05.
            dtype (torch.dtype, optional): The desired data type of the weight parameter. Defaults to None.

        Returns:
            None

        Raises:
            None
        '''
        super().__init__()
        self.weight = Parameter(ops.zeros(normalized_shape, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: mindspore.Tensor):
        """
        Constructs the RMSNorm of the given hidden states.

        Args:
            self (RMSNorm): The instance of the RMSNorm class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states. It should have a shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method modifies the hidden_states tensor in-place.

        Raises:
            TypeError: If the hidden_states parameter is not of type mindspore.Tensor.
            ValueError: If the hidden_states tensor does not have the expected shape.

        Notes:
            - This method calculates the RMSNorm of the hidden states using the following steps:

                1. Calculate the variance of the hidden states by squaring each element, taking the mean along the last dimension, and keeping the dimensions intact.
                2. Normalize the hidden states by dividing them element-wise by the square root of the variance plus a small constant epsilon.
                3. Multiply the normalized hidden states by the weight parameter of the RMSNorm instance.
                4. Convert the modified hidden_states tensor back to the same data type as the input hidden_states tensor.

            - The hidden_states parameter should have a shape of (batch_size, sequence_length, hidden_size).
            - The hidden_states tensor is modified in-place, meaning the changes are made directly to the input tensor.

        Example:
            ```python
            >>> rms_norm = RMSNorm()
            >>> hidden_states = mindspore.Tensor(np.random.randn(3, 5, 10), dtype=mindspore.float32)
            >>> rms_norm.forward(hidden_states)
            ```
        """
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(nn.Module):

    """
    The CoreAttention class represents a core component of attention mechanism for neural network models.
    This class is used to perform attention operations on query, key, and value layers.
    It inherits from the nn.Module class.

    Attributes:
        config (ChatGLM2Config): The configuration for the attention mechanism.
        layer_number (int): The layer number for the attention mechanism.

    Methods:
        __init__: Initializes the CoreAttention instance with the provided configuration and layer number.
        forward: Constructs the attention mechanism by performing attention operations on the input query, key,
            and value layers with the optional attention mask.

    The __init__ method initializes the CoreAttention instance with the given configuration and layer number.
    The forward method performs attention operations on the query, key, and value layers, and optionally applies
    the attention mask.

    Note:
        This docstring is generated based on the provided information. Please add any additional information as needed.
    """
    def __init__(self, config: ChatGLM2Config, layer_number):
        """
        Initializes a CoreAttention object with the provided configuration and layer number.

        Args:
            self (CoreAttention): The CoreAttention object itself.
            config (ChatGLM2Config): An instance of ChatGLM2Config containing the configuration settings.
            layer_number (int): The layer number to be assigned to the CoreAttention object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = nn.Dropout(p=config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        """
        Constructs the attention layer of the CoreAttention class.

        Args:
            self (CoreAttention): An instance of the CoreAttention class.
            query_layer (Tensor): The input query layer tensor of shape (batch_size, seq_length, hidden_size).
            key_layer (Tensor): The input key layer tensor of shape (batch_size, seq_length, hidden_size).
            value_layer (Tensor): The input value layer tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (Tensor): The attention mask tensor of shape (batch_size, seq_length, seq_length) or None.
                It is used to mask certain positions of the attention scores.

        Returns:
            Tensor: The output context layer tensor of shape (batch_size, seq_length, hidden_size).

        Raises:
            ValueError: If the shape of query_layer, key_layer, or value_layer is not appropriate.
            TypeError: If the data type of attention_mask is not appropriate.
            ValueError: If the shape of attention_mask is not compatible with the shape of attention_scores.
            ValueError: If the shape of attention_mask is not compatible with the shape of attention_probs.

        Note:
            - The attention mask tensor is used to prevent attention to certain positions, such as padding positions in the input sequence.
            - The attention scores are computed as the matrix multiplication of the query and key layers, followed by scaling with a normalization factor.
            - The attention probabilities are obtained by applying the softmax function to the attention scores.
            - The attention probabilities are then multiplied with the value layer to obtain the context layer.
            - The context layer is reshaped and returned as the output.

        Example:
            ```python
            >>> query = torch.randn(2, 5, 10)
            >>> key = torch.randn(2, 5, 10)
            >>> value = torch.randn(2, 5, 10)
            >>> attention = CoreAttention()
            >>> attention.forward(query, key, value, None)
            ```
        """
        # wait for flash attention
        # query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
        # if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
        #     context_layer = nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
        #                                                                      is_causal=True)
        # else:
        #     if attention_mask is not None:
        #         attention_mask = ~attention_mask
        #     context_layer = nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
        #                                                                      attention_mask)
        # context_layer = context_layer.permute(2, 0, 1, 3)
        # new_context_layer_shape = context_layer.shape[:-2] + (self.hidden_size_per_partition,)
        # context_layer = context_layer.reshape(*new_context_layer_shape)
        # Raw attention scores

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = ops.bmm(
            query_layer.swapaxes(0, 1),  # [b * np, sq, hn]
            key_layer.permute(1, 2, 0),  # [b * np, hn, sk]
        ) * (1.0 / self.norm_factor)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = ops.ones(output_size[0], 1, output_size[2], output_size[3], dtype=mindspore.int32)
            attention_mask = attention_mask.tril().bool()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
        attention_probs = ops.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.astype(value_layer.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)
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
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3)
        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.shape[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
    def __init__(self, config: ChatGLM2Config, layer_number):
        """
        Initializes a new instance of the SelfAttention class.

        Args:
            self (SelfAttention): The current instance of the SelfAttention class.
            config (ChatGLM2Config): An instance of the ChatGLM2Config class containing configuration parameters.
            layer_number (int): The layer number for the self-attention operation.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.layer_number = max(1, layer_number)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                        bias=config.add_bias_linear or config.add_qkv_bias,
                                        **_config_to_kwargs(config))

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                              **_config_to_kwargs(config))

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        """
        Constructs the self-attention mechanism for the SelfAttention class.

        Args:
            self (object): The instance of the SelfAttention class.
            hidden_states (tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the hidden states of the input sequence.
            attention_mask (tensor): The attention mask tensor of shape (batch_size, sequence_length, sequence_length).
                This tensor is used to mask certain positions in the input sequence for the attention computation.
            rotary_pos_emb (tensor): The rotary position embedding tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor is used to apply rotary position embeddings to the query and key layers.
            kv_cache (tuple): The key-value cache tuple containing two tensors, cache_k and cache_v.
                cache_k (tensor): The cached key tensor of shape (num_cached_tokens, num_attention_heads, sequence_length, hidden_size_per_attention_head).
                    This tensor stores the cached key values from previous attention computations.
                cache_v (tensor): The cached value tensor of shape (num_cached_tokens, num_attention_heads, sequence_length, hidden_size_per_attention_head).
                    This tensor stores the cached value values from previous attention computations.
            use_cache (bool): Whether to use the key-value cache or not. Defaults to True.

        Returns:
            output (tensor): The output tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the output of the self-attention mechanism.
            kv_cache (tuple): The updated key-value cache tuple containing two tensors, key_layer and value_layer.
                key_layer (tensor): The updated key tensor of shape (batch_size, sequence_length, num_attention_heads, hidden_size_per_attention_head).
                    This tensor contains the updated key values after concatenating with the cache_k tensor.
                value_layer (tensor): The updated value tensor of shape (batch_size, sequence_length, num_attention_heads, hidden_size_per_attention_head).
                    This tensor contains the updated value values after concatenating with the cache_v tensor.

        Raises:
            None.
        """
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                axis=-1,
            )
            query_layer = query_layer.view(
                query_layer.shape[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.shape[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.shape[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.shape[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = ops.cat((cache_k, key_layer), dim=0)
            value_layer = ops.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            key_layer = key_layer.view(
                key_layer.shape[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.view(
                value_layer.shape[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


def _config_to_kwargs(args):
    """
    This function converts configuration arguments to keyword arguments for a function.

    Args:
        args (dict): A dictionary containing configuration arguments.

    Returns:
        dict: A dictionary of common keyword arguments with the 'dtype' key set to the value of 'ms_dtype' from the input args.

    Raises:
        None.
    """
    common_kwargs = {
        "dtype": args.ms_dtype,
    }
    return common_kwargs


class MLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """
    def __init__(self, config: ChatGLM2Config):
        """
        Initializes an instance of the MLP class.

        Args:
            self: The instance of the MLP class.
            config (ChatGLM2Config): The configuration object for the MLP model. It contains various settings and hyperparameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = ops.chunk(x, 2, dim=-1)
            return ops.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        """
        Constructs the output of the MLP.

        Args:
            self (MLP): An instance of the MLP class.
            hidden_states (Tensor): The hidden states input to the MLP.
                This should be a tensor of shape (batch_size, hidden_dim), where batch_size is the number of samples
                in the batch and hidden_dim is the dimension of the hidden states.
                The hidden states serve as the input to the MLP for forwarding the output.

        Returns:
            output (Tensor): The output tensor forwarded by the MLP. This is a tensor of shape (batch_size, hidden_dim),
                where batch_size is the number of samples in the batch and hidden_dim is the dimension of the hidden
                states. The output tensor represents the result of applying the MLP layers to the input hidden states.

        Raises:
            None.
        """
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """
    def __init__(self, config: ChatGLM2Config, layer_number):
        """
        Initializes a new instance of the GLMBlock class.

        Args:
            self: The current instance of the GLMBlock class.
            config (ChatGLM2Config):
                The configuration object for the GLMBlock.

                - This parameter specifies the configuration settings for the GLMBlock.
                - It should be an instance of the ChatGLM2Config class.
            layer_number:
                The layer number of the GLMBlock.

                - This parameter indicates the position of the GLMBlock within the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                             dtype=config.ms_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                                      dtype=config.ms_dtype)

        # MLP
        self.mlp = MLP(config)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        """
        Constructs a GLMBlock by performing a series of operations on the input data.

        Args:
            self: The GLMBlock instance.
            hidden_states (Tensor): The input hidden states. Shape [batch_size, sequence_length, hidden_size].
            attention_mask (Tensor): The attention mask. Shape [batch_size, sequence_length].
            rotary_pos_emb (Tensor): The rotary positional embeddings. Shape [sequence_length, hidden_size].
            kv_cache (Tensor, optional): The key-value cache. Shape [batch_size, num_heads, sequence_length, hidden_size].
            use_cache (bool, optional): Whether to use the key-value cache. Default is True.

        Returns:
            output (Tensor): The output hidden states after performing the operations. Shape [batch_size, sequence_length, hidden_size].
            kv_cache (Tensor): The updated key-value cache. Shape [batch_size, num_heads, sequence_length, hidden_size].

        Raises:
            TypeError: If the input parameters are of incorrect types.
            ValueError: If the input parameters have invalid shapes.

        Note:
            - The `hidden_states` should have the same hidden size as the `rotary_pos_emb`.
            - The `attention_mask` should have shape [batch_size, sequence_length] and contains 0s for padding tokens and 1s for non-padding tokens.
            - The `rotary_pos_emb` should have shape [sequence_length, hidden_size] and contains positional embeddings for each token.
            - The `kv_cache` should have shape [batch_size, num_heads, sequence_length, hidden_size] and is optional.
            - The `use_cache` is a boolean flag indicating whether to use the key-value cache. It is optional and defaults to True.
            - The `output` is computed by applying a series of operations including self-attention, residual connection, layer normalization, and multi-layer perceptron (MLP).
            - The `kv_cache` is updated during the self-attention operation if `use_cache` is True.
        """
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = F.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = F.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(nn.Module):
    """Transformer class."""
    def __init__(self, config: ChatGLM2Config):
        '''
        Initializes a GLMTransformer object.

        Args:
            self: The instance of the GLMTransformer class.
            config (ChatGLM2Config): An instance of ChatGLM2Config representing the configuration for the GLMTransformer.
                It should contain the following attributes:

                - fp32_residual_connection (bool): Indicates whether to use FP32 residual connection.
                - post_layer_norm (bool): Indicates whether to apply layer normalization after the block.
                - num_layers (int): The number of layers in the GLMTransformer.

        Returns:
            None.

        Raises:
            None.
        '''
        super().__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number)

        self.layers = nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon,
                                                 dtype=config.ms_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        """
        This method retrieves a specific layer from the GLMTransformer.

        Args:
            self (GLMTransformer): The instance of the GLMTransformer class.
            layer_number (int): The index of the layer to be retrieved.
                It should be a non-negative integer within the range of the available layers.

        Returns:
            None: This method returns None if the layer is not found.

        Raises:
            IndexError: If the layer_number is out of range or if the layers list is empty.
            TypeError: If the layer_number is not an integer.
        """
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        """
        Constructs the hidden states of the GLMTransformer model.

        Args:
            self (GLMTransformer): The instance of the GLMTransformer class.
            hidden_states (Tensor): The input hidden states.
            attention_mask (Tensor): The attention mask tensor.
            rotary_pos_emb (Tensor): The rotary position embedding tensor.
            kv_caches (Optional[List[Optional[Tensor]]]): The list of key-value caches for each layer. Defaults to None.
            use_cache (Optional[bool]): Whether to use the key-value caches. Defaults to True.
            output_hidden_states (Optional[bool]): Whether to output hidden states for each layer. Defaults to False.

        Returns:
            Tuple[Tensor, Tuple[Optional[Tensor], ...], Optional[Tuple[Tensor, ...]], Optional[Tensor]]:

                - hidden_states (Tensor): The output hidden states.
                - presents (Tuple[Optional[Tensor], ...]): The key-value caches for each layer, or an empty tuple if `use_cache` is False.
                - all_hidden_states (Optional[Tuple[Tensor, ...]]): The hidden states for each layer, or None if `output_hidden_states` is False.
                - all_self_attentions (Optional[Tensor]): The self-attention matrices for each layer, or None.

        Raises:
            None.
        """
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self._get_layer(index)

            layer_ret = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches[index],
                use_cache=use_cache
            )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLM2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    is_parallelizable = False
    config_class = ChatGLM2Config
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, cell):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        '''
            This method calculates the attention masks for the input sequence in the context of the ChatGLM2PreTrainedModel class.

            Args:
                self (ChatGLM2PreTrainedModel): The instance of the ChatGLM2PreTrainedModel class.
                input_ids (torch.Tensor): The input sequence tensor of shape (batch_size, seq_length).
                past_key_values (tuple of torch.Tensor): The past key-value pairs for attention weights of shape
                    (past_length, batch_size, num_heads, past_seq_length, embed_dim).
                padding_mask (torch.Tensor, optional): The tensor indicating the positions of padding tokens in the input sequence.
                    It has the shape (batch_size, seq_length) and contains 0's for non-padding tokens and 1's for padding tokens.
                    Defaults to None.

            Returns:
                torch.Tensor: The attention mask tensor of shape (batch_size, 1, seq_length, seq_length).

            Raises:
                None.
        '''
        batch_size, seq_length = input_ids.shape
        full_attention_mask = ops.ones(batch_size, seq_length, seq_length)
        full_attention_mask = full_attention_mask.tril()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = ops.cat((ops.ones(batch_size, seq_length, past_length), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask = full_attention_mask.unsqueeze(1)
        return full_attention_mask

    def get_position_ids(self, input_ids):
        """
        Returns the position IDs corresponding to input IDs.

        Args:
            self (ChatGLM2PreTrainedModel): The instance of the ChatGLM2PreTrainedModel class.
            input_ids (ndarray): A 2-dimensional array of shape (batch_size, seq_length) containing input IDs.

        Returns:
            ndarray: A 2-dimensional array of shape (batch_size, seq_length) containing position IDs corresponding to input IDs.

        Raises:
            None.

        """
        batch_size, seq_length = input_ids.shape
        position_ids = ops.arange(seq_length, dtype=mindspore.int64).unsqueeze(0).repeat(batch_size, 1)
        return position_ids


class Embedding(nn.Module):
    """Language model embeddings."""
    def __init__(self, config: ChatGLM2Config):
        """
        Initializes an Embedding object.

        Args:
            self: The instance of the Embedding class.
            config (ChatGLM2Config):
                An instance of the ChatGLM2Config class containing the configuration parameters for the Embedding object.

               - type: ChatGLM2Config
               - purpose: Specifies the configuration parameters for the Embedding object.
               - restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.ms_dtype,
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        """
        Construct word embeddings from input_ids.

        Args:
            self (Embedding): An instance of the Embedding class.
            input_ids (Tensor): A tensor of shape (batch_size, sequence_length) containing the input token ids.

        Returns:
            embeddings (Tensor): A tensor of shape (sequence_length, batch_size, embedding_size) containing the word embeddings.

        Raises:
            None.
        """
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.swapaxes(0, 1)
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLM2Model(ChatGLM2PreTrainedModel):

    """
    This class represents the ChatGLM2Model, which is used for natural language processing tasks.
    It inherits from the ChatGLM2PreTrainedModel and contains methods for initializing the model, getting input
    embeddings, getting prompts, forwarding the model, and quantizing the model's weights.
    The class contains attributes for embedding, number of layers, multi-query group number, key-value channels,
    sequence length, rotary position embedding, encoder, output layer, prefix sequence length, prefix projection,
    prefix tokens, prefix encoder, and dropout.
    The methods included are __init__, get_input_embeddings, get_prompt, forward, and quantize.
    """
    def __init__(self, config: ChatGLM2Config, empty_init=True):
        """
        This method initializes an instance of the ChatGLM2Model class.

        Args:
            self: The instance of the ChatGLM2Model class.
            config (ChatGLM2Config): An instance of the ChatGLM2Config class containing configuration parameters for the model.
            empty_init (bool): A flag indicating whether to perform an empty initialization.
                If True, the initialization method is set to zero_init; otherwise, it is set to default_init.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        if empty_init:
            init_method = zero_init
        else:
            init_method = default_init
        init_kwargs = {}
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope,
                                              dtype=config.ms_dtype)
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.ms_dtype, **init_kwargs)
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = ops.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = nn.Dropout(p=0.1)

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the ChatGLM2Model.

        Args:
            self (ChatGLM2Model): The instance of the ChatGLM2Model class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, dtype=mindspore.float16):
        """
        Retrieves the prompt for the ChatGLM2Model.

        Args:
            self (ChatGLM2Model): The instance of the ChatGLM2Model class.
            batch_size (int): The number of sequences in a batch.
            dtype (mindspore.dtype, optional): The data type of the returned prompt. Defaults to mindspore.float16.

        Returns:
            None.

        Raises:
            None.
        """
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        past_key_values = self.prefix_encoder(prefix_tokens).astype(dtype)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids,
            position_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            full_attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the ChatGLM2Model.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor): The input token IDs of shape (batch_size, seq_length).
            position_ids (Optional[mindspore.Tensor]): The position IDs tensor. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor. Default is None.
            full_attention_mask (Optional[mindspore.Tensor]): The full attention mask tensor. Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]]):
                The past key values. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded inputs tensor. Default is None.
            use_cache (Optional[bool]): Flag to use cache. Default is None.
            output_hidden_states (Optional[bool]): Flag to output hidden states. Default is None.
            return_dict (Optional[bool]): Flag to return a dictionary. Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = ops.cat([attention_mask.new_ones((batch_size, self.pre_seq_len), dtype=attention_mask.dtype),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.swapaxes(0, 1)

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def quantize(self, weight_bit_width: int):
        """Quantize the weights of the ChatGLM2Model.

        This method quantizes the weights of the ChatGLM2Model object according to the specified weight bit width.

        Args:
            self (ChatGLM2Model): The ChatGLM2Model object to be quantized.
            weight_bit_width (int): The number of bits to be used for quantizing the weights.
                This value determines the precision of the quantization. Valid values are positive integers.

        Returns:
            None.

        Raises:
            None.
        """


class ChatGLM2ForConditionalGeneration(ChatGLM2PreTrainedModel):

    """A Python class representing a conditional generation model for chat-based tasks using ChatGLM2.

    This class inherits from ChatGLM2PreTrainedModel and includes methods to initialize the model, update model keyword
    arguments for generation, prepare inputs for generation, forward the model, reorder cache, process response,
    build inputs, build stream inputs, chat, stream chat, stream generate, and quantize the model.

    The methods in this class enable the generation of responses for chat-based queries, handling of input data, and
    model quantization for improved efficiency.

    For detailed information on the methods and their parameters, please refer to the method docstrings within the class implementation.
    """
    def __init__(self, config: ChatGLM2Config, empty_init=True):
        """
        Initializes an instance of the ChatGLM2ForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (ChatGLM2Config): An object of type ChatGLM2Config which provides configuration settings for the model.
            empty_init (bool, optional): Indicates whether to initialize the ChatGLM2Model with empty weights. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = ChatGLM2Model(config, empty_init=empty_init)
        self.config = config
        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        '''
        Updates the model keyword arguments for generation in the `ChatGLM2ForConditionalGeneration` class.

        Args:
            self (ChatGLM2ForConditionalGeneration): The instance of the ChatGLM2ForConditionalGeneration class.
            outputs (ModelOutput): The output of the model.
            model_kwargs (Dict[str, Any]): The dictionary containing the model keyword arguments.
            is_encoder_decoder (bool, optional): Indicates if the model is an encoder-decoder model. Defaults to False.
            standardize_cache_format (bool, optional): Indicates if the cache format should be standardized. Defaults to False.

        Returns:
            Dict[str, Any]: The updated model keyword arguments.

        Raises:
            None.
        '''
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = ops.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].copy()
            new_position_id += 1
            model_kwargs["position_ids"] = ops.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: mindspore.Tensor,
            past_key_values: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        """
        Prepares input tensors for generation during ChatGLM2ForConditionalGeneration model training.

        Args:
            self (ChatGLM2ForConditionalGeneration): The instance of the ChatGLM2ForConditionalGeneration class.
            input_ids (mindspore.Tensor): The input tensor of shape (batch_size, seq_length) containing the input sequence indices.
            past_key_values (Optional[mindspore.Tensor]): Optional past key values tensor of shape
                (batch_size, num_heads, past_seq_length, hidden_size_per_head) used for generation in accordance with GPT-2.
            attention_mask (Optional[mindspore.Tensor]): Optional attention mask tensor of shape
                (batch_size, seq_length) used for masking out padded tokens.
            position_ids (Optional[mindspore.Tensor]): Optional position ids tensor of shape
                (batch_size, seq_length) used for generation in accordance with GPT-2.
            use_cache (Optional[bool]): Optional flag indicating whether to use cache during generation.
            is_first_forward (bool): Flag indicating whether it is the first forward pass.

        Returns:
            dict:
                A dictionary containing input tensors for generation:

                - input_ids (mindspore.Tensor): The input tensor of shape (batch_size, seq_length) containing the input sequence indices.
                - past_key_values (Optional[mindspore.Tensor]): Optional past key values tensor of shape
                (batch_size, num_heads, past_seq_length, hidden_size_per_head) used for generation in accordance with  GPT-2.
                - position_ids (mindspore.Tensor): The position ids tensor of shape (batch_size, seq_length) used for generation in accordance with GPT-2.
                - attention_mask (Optional[mindspore.Tensor]): Optional attention mask tensor of shape (batch_size, seq_length) used for masking out padded tokens.
                - return_last_logit (bool): Flag indicating whether to return the last logit during generation.
                - use_cache (Optional[bool]): Optional flag indicating whether to use cache during generation.

        Raises:
            None.
        """
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        '''
        Constructs a ChatGLM2ForConditionalGeneration object.

        Args:
            self (ChatGLM2ForConditionalGeneration): The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, sequence_length] representing the tokenized input sequences.
                Default is None.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, sequence_length] representing the position indices of the input tokens.
                Default is None.
            attention_mask (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, sequence_length] representing the attention mask to avoid
                performing attention on padding tokens. Default is None.
            past_key_values (Optional[Tuple[mindspore.Tensor]]):
                The optional tuple of tensors that contains pre-computed key and value tensors for fast decoding.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, sequence_length, hidden_size] representing the embedded inputs.
                Default is None.
            labels (Optional[mindspore.Tensor]):
                The input tensor of shape [batch_size, sequence_length] representing the labels. Default is None.
            use_cache (Optional[bool]): Whether to use caching mechanism for faster decoding.
                If not provided, it takes the value from self.config.use_cache. Default is None.
            output_attentions (Optional[bool]): Whether to output attention weights. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return outputs as a dictionary instead of a tuple.
                If not provided, it takes the value from self.config.use_return_dict. Default is None.
            return_last_logit (Optional[bool]): Whether to return the last logit. Default is False.

        Returns:
            None

        Raises:
            None
        '''
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.swapaxes(0, 1)

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(mindspore.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1),
                                     ignore_index=-100)

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

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
        """
        Process the response received from the chat model.

        Args:
            self: An instance of the ChatGLM2ForConditionalGeneration class.
            response (str): The response received from the chat model.

        Returns:
            None.

        Raises:
            None.
        """
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        """
        Builds the input tensors for the ChatGLM2ForConditionalGeneration model.

        Args:
            self (ChatGLM2ForConditionalGeneration): An instance of the ChatGLM2ForConditionalGeneration class.
            tokenizer (PreTrainedTokenizer): An instance of PreTrainedTokenizer used for tokenizing the input.
            query (str): A string containing the user query.
            history (List[Tuple[str, str]], optional): A list of tuples containing previous queries and their respective responses.
                Defaults to None.

        Returns:
            None.

        Raises:
            None.

        The method takes in a tokenizer instance, a user query, and optionally a list of previous queries and their
        respective responses. It then builds the input tensors using the provided tokenizer by calling the build_prompt
        method on the tokenizer instance. The input tensors are then returned as a dictionary with a single key and value
        pair. The key is 'input_ids' and the value is a tensor containing the tokenized input.
        """
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="ms")
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        """
        This method builds stream inputs for the ChatGLM2ForConditionalGeneration class.

        Args:
            self: The instance of the class.
            tokenizer: An object of the tokenizer used to encode the input prompt. It should be compatible with the model being used.
            query (str): The query string for which the stream inputs are being generated.
            history (List[Tuple[str, str]], optional): A list of historical tuples containing the previous queries and responses.
                Defaults to None.

        Returns:
            None: This method does not return any value, but it populates the 'inputs' variable with the encoded input prompt
                and returns it.

        Raises:
            None.
        """
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:]
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="ms", add_special_tokens=False)
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="ms")
        return inputs

    @_no_grad()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        """
        This method 'chat' is defined in the class 'ChatGLM2ForConditionalGeneration' and is used for generating a response to a given query in a chat scenario.

        Args:
            self: Represents the instance of the class.
            tokenizer: An object used for tokenizing the input query and decoding the generated response.
            query (str): The input query for which a response needs to be generated.
            history (List[Tuple[str, str]]): A list of previous query-response pairs. Defaults to an empty list.
            max_length (int): The maximum length of the generated response. Defaults to 8192.
            num_beams (int): The number of beams to be used in beam search. Defaults to 1.
            do_sample (bool): A flag indicating whether sampling should be used during generation. Defaults to True.
            top_p (float): The nucleus sampling parameter. Defaults to 0.8.
            temperature (float): The temperature parameter for sampling. Defaults to 0.8.
            logits_processor: An object for processing the logits during generation. Defaults to None.

        Returns:
            response (str): The generated response to the input query.
            history (List[Tuple[str, str]]): The updated history including the input query and generated response.

        Raises:
            None

        Note:
            The method appends the input query and generated response to the history and returns the generated response along with the updated history.
        """
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @_no_grad()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                    max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                    return_past_key_values=False, **kwargs):
        """
        Method to perform streaming chat using the ChatGLM2ForConditionalGeneration model.

        Args:
            self: The instance of the ChatGLM2ForConditionalGeneration class.
            tokenizer: An instance of the tokenizer to encode/decode the input/output sequences.
            query (str): The input query for the chat conversation.
            history (List[Tuple[str, str]], optional): List of previous chat history tuples,
                where each tuple contains the input query and the corresponding response. Defaults to None.
            past_key_values: The past key values for the model's autoregressive generation. Defaults to None.
            max_length (int): The maximum length of the output sequence. Defaults to 8192.
            do_sample (bool): Flag to enable sampling of the output sequence. Defaults to True.
            top_p (float): The nucleus sampling parameter for the output sequence generation. Defaults to 0.8.
            temperature (float): The temperature parameter for the output sequence generation. Defaults to 0.8.
            logits_processor: The logits processor to modify model's output distribution. Defaults to None.
            return_past_key_values (bool): Flag to return the past key values along with the response. Defaults to False.
            **kwargs: Additional keyword arguments for generating the output sequence.

        Returns:
            None: However, yields a tuple containing the response, updated chat history,
                and past key values if return_past_key_values is True.

        Raises:
            None.
        """
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None and not return_past_key_values:
            inputs = self.build_inputs(tokenizer, query, history=history)
        else:
            inputs = self.build_stream_inputs(tokenizer, query, history=history)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0]
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len
            inputs['position_ids'] = inputs.position_ids + past_length # mindspore do not support `x += 1`
            attention_mask = inputs.attention_mask
            attention_mask = ops.cat((attention_mask.new_ones((1, past_length), dtype=attention_mask.dtype), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
                response = self.process_response(response)
                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    @_no_grad()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, mindspore.Tensor], List[int]]] = None,
            return_past_key_values=False,
            **kwargs,
    ):
        """
        Generates a stream of conditional text based on the given input_ids using the ChatGLM2 model.

        Args:
            self (ChatGLM2ForConditionalGeneration): The instance of the ChatGLM2ForConditionalGeneration class.
            input_ids (mindspore.Tensor): The input token ids for text generation.
            generation_config (Optional[GenerationConfig]): The configuration for text generation. Default is None.
            logits_processor (Optional[LogitsProcessorList]):
                The list of logits processors to be applied on the generated logits. Default is None.
            stopping_criteria (Optional[StoppingCriteriaList]):
                The list of stopping criteria to determine when to stop text generation. Default is None.
            prefix_allowed_tokens_fn (Optional[Callable[[int, mindspore.Tensor], List[int]]]):
                The function that returns a list of allowed tokens for each prefix. Default is None.
            return_past_key_values (bool): Whether to return the past key values during generation. Default is False.

        Returns:
            None.

        Raises:
            UserWarning: If using `max_length`'s default value to control generation length.
                This behavior is deprecated.
                It is recommended to use `max_new_tokens` instead.
            UserWarning: If both `max_new_tokens` and `max_length` are set. `max_new_tokens` takes precedence.
            UserWarning: If the input length exceeds `max_length` and may lead to unexpected behavior.

        Note:
            This method yields generated text in a streaming fashion.
        """
        _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        model_kwargs["use_cache"] = generation_config.use_cache
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

        unfinished_sequences = ops.ones(input_ids.shape[0], dtype=input_ids.dtype)
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
            probs = ops.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = ops.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = ops.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = ops.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    def quantize(self, bits: int, empty_init=False, **kwargs):
        """
        This method quantizes the input data to a specified number of bits.

        Args:
            self: The instance of the ChatGLM2ForConditionalGeneration class.
            bits (int): The number of bits to quantize the input data to.
                Must be a positive integer.
            empty_init (bool): Optional. If True, the initialization process is skipped.
                Defaults to False.

        Returns:
            None.

        Raises:
            ValueError: If the bits parameter is not a positive integer.
            TypeError: If the bits parameter is not an integer.
        """

class ChatGLM2ForSequenceClassification(ChatGLM2PreTrainedModel):
    """
    ChatGLM2ForSequenceClassification is a class representing a pre-trained model for sequence classification based on
    the ChatGLM2 architecture. It inherits from the ChatGLM2PreTrainedModel and provides methods for initializing the model
    and generating classification outputs.

    The class contains an initializer method that takes in a ChatGLM2Config object and an optional boolean parameter for
    empty initialization. It initializes the model with the provided configuration and sets up the transformer and
    classifier head layers.

    The forward method takes various input tensors and parameters for generating the sequence classification output.
    It returns a sequence classifier output with past states if the return_dict parameter is set, or a tuple of tensors
    including logits and transformer outputs. The method also handles the calculation of loss based on the provided labels and problem type.
    
    This class provides a comprehensive interface for utilizing the ChatGLM2 model for sequence classification tasks,
    including handling transformer outputs, dropout, and classification head operations.
    
    """
    def __init__(self, config: ChatGLM2Config, empty_init=True):
        """
        Initializes an instance of the ChatGLM2ForSequenceClassification class.
        
        Args:
            self: The object itself.
            config (ChatGLM2Config): An instance of the ChatGLM2Config class containing the configuration settings for the model.
            empty_init (bool): A flag indicating whether to initialize the transformer with empty values. Defaults to True.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = ChatGLM2Model(config, empty_init=empty_init)

        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=mindspore.float16)
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(p=config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            full_attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, ...], SequenceClassifierOutputWithPast]:
        '''
        Constructs the ChatGLM2ForSequenceClassification model.
        
        Args:
            self: The object instance.
            input_ids (Optional[mindspore.Tensor]): The input token IDs. Default: None.
            position_ids (Optional[mindspore.Tensor]): The position IDs. Default: None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask. Default: None.
            full_attention_mask (Optional[mindspore.Tensor]): The full attention mask. Default: None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]]): The past key values. Default: None.
            inputs_embeds (Optional[mindspore.Tensor]): The input embeddings. Default: None.
            labels (Optional[mindspore.Tensor]): The labels. Default: None.
            use_cache (Optional[bool]): Whether to use cache. Default: None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default: None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default: None.
        
        Returns:
            Union[Tuple[mindspore.Tensor, ...], SequenceClassifierOutputWithPast]: The model outputs.
        
        Raises:
            None.
        '''
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        pooled_hidden_states = hidden_states[-1]
        if self.dropout is not None:
            pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier_head(pooled_hidden_states)

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
                    loss = F.mse_loss(logits.squeeze().float(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels).float(), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits.float(), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

__all__ = [
    'CHATGLM2_6B_PRETRAINED_MODEL_ARCHIVE_LIST',
    'ChatGLM2Model',
    'ChatGLM2PreTrainedModel',
    'ChatGLM2ForConditionalGeneration',
    'ChatGLM2ForSequenceClassification'
]
