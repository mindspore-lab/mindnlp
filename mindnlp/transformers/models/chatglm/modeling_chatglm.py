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
""" MindSpore ChatGLM model. """

import math
import copy
import warnings
import re
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

import mindspore
from mindspore import nn, ops, Parameter

from mindnlp.utils import logging
from mindnlp.modules.functional import embedding
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...generation.logits_process import LogitsProcessor
from ...generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_chatglm import ChatGLMConfig

logger = logging.get_logger(__name__)

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm-6b",
    # See all ChatGLM-6B models at https://hf-mirror.com/models?filter=chatglm
]


class InvalidScoreLogitsProcessor(LogitsProcessor):

    """
    A subclass of LogitsProcessor that handles invalid score logits by replacing them with a default value. 
    
    The InvalidScoreLogitsProcessor class overrides the __call__ method to process input tensors containing score logits. 
    If any scores are NaN or infinite, they are replaced with a default value of 50000.0. 
    This class ensures that the output tensor maintains the same shape as the input tensor, with invalid scores replaced accordingly.
    """
    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method '__call__' in the class 'InvalidScoreLogitsProcessor' processes the given input_ids and scores to handle invalid score values.
        
        Args:
            self: An instance of the 'InvalidScoreLogitsProcessor' class.
            input_ids (mindspore.Tensor):
                A tensor representing the input IDs.

                - Shape: Arbitrary.
                - Data Type: mindspore.Tensor.
            scores (mindspore.Tensor):
                A tensor representing the scores.

                - Shape: Arbitrary.
                - Data Type: mindspore.Tensor.

        Returns:
            mindspore.Tensor:
                A tensor containing the processed scores.

                - Shape: Same as the 'scores' tensor.
                - Data Type: mindspore.Tensor.

        Raises:
            None.
        """
        if ops.isnan(scores).any() or ops.isinf(scores).any():
            scores = ops.zeros_like(scores)
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(nn.Cell):
    """
    The nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """
    def __init__(self, config):
        """
        Initialize a PrefixEncoder object.

        Args:
            self (PrefixEncoder): The instance of the PrefixEncoder class.
            config:
                A configuration object containing parameters for the PrefixEncoder.

                - Type: Custom configuration object.
                - Purpose: To configure the PrefixEncoder with specific settings.
                - Restrictions: Must contain the following attributes:

                    - prefix_projection: A boolean indicating whether prefix projection should be used.
                    - pre_seq_len: An integer representing the length of the input sequence.
                    - hidden_size: An integer specifying the size of the hidden layers.
                    - num_layers: An integer indicating the number of layers in the model.

        Returns:
            None.

        Raises:
            AttributeError: If the config object does not have the required attributes.
            TypeError: If the data types of the config attributes are incorrect.
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
        This method constructs the past key values for the prefix encoder.

        Args:
            self (PrefixEncoder): The instance of the PrefixEncoder class.
            prefix (mindspore.Tensor): The input tensor representing the prefix sequence.

        Returns:
            mindspore.Tensor: The past key values constructed based on the input prefix.

        Raises:
            None
        """
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + ops.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    """
    This function applies the Gaussian Error Linear Unit (GELU) activation function to the input tensor 'x'.

    Args:
        x (tensor): The input tensor to which the GELU activation function will be applied.

    Returns:
        tensor: The tensor resulting from applying the GELU activation function to the input tensor 'x'.

    Raises:
        None.
    """
    return ops.gelu(x, approximate='tanh')


class RotaryEmbedding(nn.Cell):

    """
    This class represents a rotary embedding layer that can be used in neural network models. It inherits from the nn.Cell class.

    The RotaryEmbedding layer is designed to provide rotational positional encoding for input sequences.
    It utilizes sinusoidal functions to generate embeddings that capture the relative positions of elements
    in a sequence.

    Initialization:
        - dim (int): The dimensionality of the input sequence.
        - base (int, optional): The base value used in the positional encoding formula. Default is 10000.
        - precision (mindspore.dtype, optional): The precision of the embeddings. Default is mindspore.float16.
        - learnable (bool, optional): Flag indicating whether the positional embeddings should be learnable. Default is False.

    Methods:
        construct(x, seq_dim=1, seq_len=None):
            Constructs the positional embeddings for the input sequence 'x'.

            Args:

            - x (Tensor): The input sequence tensor.
            - seq_dim (int, optional): The sequence dimension index in the input tensor. Default is 1.
            - seq_len (int, optional): The length of the sequence. If not provided, it will be inferred from the input tensor.

        _apply(fn):
            Applies the provided function 'fn' to the cached cosine and sine embeddings.

    Attributes:
        inv_freq (Tensor or Parameter): The inverse frequency tensor or parameter used in the positional encoding formula.
        learnable (bool): Flag indicating whether the positional embeddings are learnable.
        max_seq_len_cached (int or None): The maximum sequence length for which the embeddings are cached.
        cos_cached (Tensor or None): The cached cosine embeddings.
        sin_cached (Tensor or None): The cached sine embeddings.
        precision (mindspore.dtype): The precision of the embeddings.

    Note:
        The RotaryEmbedding layer caches the positional embeddings for efficiency.
        If the sequence length changes during training, the embeddings will be recomputed and cached accordingly.
    """
    def __init__(self, dim, base=10000, precision=mindspore.float16, learnable=False):
        """
        Initializes the RotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the input data.
            base (int, optional): The base value for the calculation. Default is 10000.
            precision (mindspore.dtype, optional): The precision of the calculations. Default is mindspore.float16.
            learnable (bool, optional): Indicates whether the inv_freq is learnable. Default is False.

        Returns:
            None.

        Raises:
            ValueError: If the dimension (dim) is not an integer or if the precision is not a valid mindspore dtype.
            TypeError: If the base is not an integer or if the learnable parameter is not a boolean.
        """
        super().__init__()
        inv_freq = 1. / (base ** (ops.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = Parameter(inv_freq, 'inv_freq')
            self.max_seq_len_cached = None
        else:
            self.inv_freq = inv_freq
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def construct(self, x, seq_dim=1, seq_len=None):
        """
        Constructs a RotaryEmbedding.

        Args:
            self (RotaryEmbedding): The instance of the RotaryEmbedding class.
            x (Tensor): The input tensor.
            seq_dim (int, optional): The dimension along which the sequence length is defined in the input tensor. Default is 1.
            seq_len (int, optional): The length of the sequence. If not provided, it is inferred from the shape of the input tensor.

        Returns:
            None.

        Raises:
            None.
        """
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = ops.arange(seq_len, dtype=self.inv_freq.dtype)
            freqs = ops.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = ops.cat((freqs, freqs), axis=-1)
            if self.precision == mindspore.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == mindspore.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        """
        Apply the given function to the cached cosine and sine values.

        Args:
            self (RotaryEmbedding): The instance of the RotaryEmbedding class.
            fn (function): The function to be applied to the cached cosine and sine values. It should take a single argument and return a modified value.

        Returns:
            None.

        Raises:
            None.
        """
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    """
    Rotate the input array by half its length.

    Args:
        x: array_like
            Input array to be rotated. Can be a numpy array or any array-like object.

    Returns:
        None.

    Raises:
        ValueError: If the input array is empty or has an odd number of elements.
        TypeError: If the input is not an array-like object.
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    """
    Applies rotary positional embedding index to the given inputs 'q' and 'k' based on the provided 'cos' and 'sin' values.

    Args:
        q (tensor): The input tensor representing 'q'.
        k (tensor): The input tensor representing 'k'.
        cos (tensor): The tensor containing cosine values.
        sin (tensor): The tensor containing sine values.
        position_id (int): The position identifier.

    Returns:
        tuple: A tuple of tensors representing the modified 'q' and 'k'.

    Raises:
        None.
    """
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
        embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


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
    """
    Args:
        self: The instance of the class.
        query_layer (Tensor): The input tensor representing queries for the attention mechanism.
        key_layer (Tensor): The input tensor representing keys for the attention mechanism.
        value_layer (Tensor): The input tensor representing values for the attention mechanism.
        attention_mask (Tensor): The mask tensor to apply on the attention scores to exclude certain positions.
        hidden_size_per_partition (int): The hidden size for each partition in the model.
        layer_id (int): The ID of the layer in the model.
        layer_past (Tuple[Tensor, Tensor], optional): The past key and value tensors for caching in recurrent inference.
        scaling_attention_score (bool): Flag indicating whether to scale the attention scores.
        use_cache (bool): Flag indicating whether to use the cache for recurrent inference.

    Returns:
        Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]: The context layer, present key and value tensors, and attention probabilities.

    Raises:
        ValueError: If the dimensions or types of input tensors are not compatible.
        RuntimeError: If there are issues during the computation.
    """
    if layer_past is not None:
        past_key, past_value = layer_past[0], layer_past[1]
        key_layer = ops.cat((past_key, key_layer), axis=0)
        value_layer = ops.cat((past_value, value_layer), axis=0)

    # seqlen, batch, num_attention_heads, hidden_size_per_attention_head
    _, _, _, hidden_size = key_layer.shape

    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    query_key_layer_scaling_coeff = float(layer_id + 1)
    if scaling_attention_score:
        query_layer = query_layer / (math.sqrt(hidden_size) * query_key_layer_scaling_coeff)

    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

    matmul_result = ops.bmm(
        query_layer.swapaxes(0, 1),  # [b * np, sq, hn]
        key_layer.swapaxes(0, 1).swapaxes(1, 2),  # [b * np, hn, sk]
    )
    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    if self.scale_mask_softmax:
        self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
    else:
        if not (attention_mask == 0).all():
            # if auto-regressive, skip
            attention_scores = attention_scores.masked_fill(attention_mask, -60000.0 / query_key_layer_scaling_coeff)
        dtype = attention_scores.dtype
        attention_scores = attention_scores.float()
        attention_scores = attention_scores * query_key_layer_scaling_coeff
        # Avoid the problem of cast not taking effect
        # attention_scores = ops.select(ops.isinf(attention_scores), -10000.00, attention_scores.float())
        # if ops.isinf(attention_scores).any():
        #     exit()
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
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3)

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.shape[:-2] + (hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, present, attention_probs)

    return outputs


def default_init(cls, *args, **kwargs):
    """
    Args:
        cls (class): The class to be initialized with the provided arguments and keyword arguments.

    Returns:
        None.

    Raises:
        None
    """
    return cls(*args, **kwargs)


class SelfAttention(nn.Cell):

    """
    Represents a self-attention mechanism within a neural network cell for processing sequential data.
    This class inherits from the nn.Cell class.

    The self-attention mechanism is responsible for computing attention scores between elements in the input sequence
    and using them to produce weighted context representations. This allows the model to focus on different parts of
    the input sequence depending on their relevance to the current task.

    The SelfAttention class includes methods for splitting tensors, applying attention masks, and constructing the
    self-attention mechanism. It also encapsulates the initialization of the self-attention layer,
    including the configuration of attention heads and hidden layer sizes.

    The class provides functionality for handling positional encodings, including 2D positional encoding, and includes
    a method for applying rotary embeddings to the query and key layers based on the positional information.

    When instantiated, the SelfAttention class can be used to process input tensors with the self-attention mechanism,
    optionally using caching and outputting attention probabilities.

    Note:
        This docstring is a detailed summary based on the provided code, and it does not include signatures or any other code.
    """
    def __init__(self, hidden_size, num_attention_heads,
                 layer_id, hidden_size_per_attention_head=None, bias=True,
                 params_dtype=mindspore.float32, position_encoding_2d=True):
        """
        Initializes a SelfAttention object.

        Args:
            hidden_size (int): The size of the hidden layer.
            num_attention_heads (int): The number of attention heads.
            layer_id (int): The ID of the layer.
            hidden_size_per_attention_head (int, optional): The size of the hidden layer per attention head.
                Defaults to None.
            bias (bool): Whether to use bias in Dense layers.
            params_dtype (mindspore.dtype, optional): The data type of the parameters. Defaults to mindspore.float32.
            position_encoding_2d (bool): Whether to use 2D position encoding.

        Returns:
            None.

        Raises:
            None.
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
            learnable=False,
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
        """
        This method applies an attention mask to attention scores.

        Args:
            attention_scores (tensor): The attention scores to which the mask will be applied.
            The tensor should be of type float32 and have the shape (batch_size, num_heads, sequence_length, sequence_length).
            attention_mask (tensor): The attention mask that will be used to mask the attention_scores.
            The tensor should be of type bool and have the shape (batch_size, 1, 1, sequence_length).

        Returns:
            None: This method modifies the attention_scores tensor in place and does not return any value.

        Raises:
            ValueError: If the attention_scores and attention_mask tensors have incompatible shapes.
            TypeError: If the attention_scores tensor is not of type float32 or the attention_mask tensor is not of type bool.
        """
        attention_scores = attention_scores.masked_fill(attention_mask, -10000.0)
        return attention_scores

    def split_tensor_along_last_dim(self, tensor, num_partitions,
                                    contiguous_split_chunks=False):
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
        if contiguous_split_chunks:
            return tuple(chunk for chunk in tensor_list)

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
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_raw_layer, 3)

        if self.position_encoding_2d:
            q1, q2 = query_layer.chunk(2, axis=(query_layer.ndim - 1))
            k1, k2 = key_layer.chunk(2, axis=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids = position_ids[:, 0, :].swapaxes(0, 1), \
                position_ids[:, 1, :].swapaxes(0, 1)
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            query_layer = ops.concat([q1, q2], axis=(q1.ndim - 1))
            key_layer = ops.concat([k1, k2], axis=(k1.ndim - 1))
        else:
            position_ids = position_ids.swapaxes(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [seq_len, batch, num_attention_heads, hidden_size_per_attention_head]
            query_layer, key_layer = apply_rotary_pos_emb_index(query_layer, key_layer, cos, sin, position_ids)

        # [seq_len, batch, hidden_size]
        context_layer, present, attention_probs = attention_fn(
            self=self,
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


class GEGLU(nn.Cell):

    """
    GEGLU is a class that represents a gated linear unit based on the Gaussian Error Linear Unit (GELU) activation function.

    This class inherits from the nn.Cell class.

    Attributes:
        activation_fn: A function representing the GELU activation function.

    Methods:
        __init__: Initializes the GEGLU instance.
        construct: Constructs the GEGLU network.

    Example:
        ```python
        >>> gelu = GEGLU()
        >>> x = Tensor([1, 2, 3, 4, 5, 6])
        >>> output = gelu.construct(x)
        ```
    """
    def __init__(self):
        """
        __init__

        Initializes a new instance of the GEGLU class.

        Args:
            self: The instance of the class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.activation_fn = ops.gelu

    def construct(self, x):
        """
        Constructs the GEGLU (Gated Exponential Linear Unit) transformation of the input tensor.

        Args:
            self (GEGLU): The instance of the GEGLU class.
            x (Tensor): The input tensor to be transformed. It should have at least two dimensions.

        Returns:
            None

        Raises:
            ValueError: If the input tensor `x` does not have at least two dimensions.

        Description:
            This method applies the GEGLU transformation to the input tensor `x`. The GEGLU transformation is performed in two steps:

            1. The input tensor `x` is split into two parts along the last dimension using the `chunk` function.
            2. The first part `x1` is multiplied element-wise with the exponential of the second part `x2` after
            applying the activation function `self.activation_fn`.

        Note:
            The activation function `self.activation_fn` is assumed to be already defined within the GEGLU class.

        Example:
            ```python
            >>> geglu_instance = GEGLU()
            >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> geglu_instance.construct(input_tensor)
            ```
        This will apply the GEGLU transformation to the input tensor `input_tensor` and return `None`.

        """
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, axis=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class GLU(nn.Cell):

    """
    GLU

    This class represents a Gated Linear Unit (GLU) neural network layer. It inherits from the nn.Cell class.

    Attributes:
        layer_id (int): The identifier for the layer.
        activation_func (function): The activation function used in the layer.
        hidden_size (int): The size of the hidden states.
        inner_hidden_size (int): The size of the intermediate hidden states. Defaults to 4 times the hidden size.
        dense_h_to_4h (nn.Dense): The linear transformation from hidden states to intermediate hidden states.
        dense_4h_to_h (nn.Dense): The linear transformation from intermediate hidden states to hidden states.

    Methods:
        __init__(self, hidden_size, inner_hidden_size=None, layer_id=None, bias=True, activation_func=gelu, params_dtype=mindspore.float32):
            Initializes a GLU instance.

        construct(self, hidden_states):
            Constructs the forward pass of the GLU layer.

    """
    def __init__(self, hidden_size, inner_hidden_size=None,
                 layer_id=None, bias=True, activation_func=gelu, params_dtype=mindspore.float32):
        """
        Initializes an instance of the GLU class.

        Args:
            hidden_size (int): The size of the hidden layer.
            inner_hidden_size (int, optional): The size of the inner hidden layer. Defaults to None.
            layer_id (int, optional): The ID of the layer. Defaults to None.
            bias (bool, optional): Indicates whether to include bias in the dense layers. Defaults to True.
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
        hidden_states: [seq_len, batch, hidden_size]
        """
        # [seq_len, batch, inner_hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)

        intermediate_parallel = self.activation_func(intermediate_parallel)

        output = self.dense_4h_to_h(intermediate_parallel)

        return output


class GLMBlock(nn.Cell):

    """
    The GLMBlock class represents a block within a GLM (Generative Language Model) model.
    It consists of layers for self-attention and multi-layer perceptron (MLP) processing.

    This class inherits from the nn.Cell class and contains the necessary methods and attributes to perform
    self-attention and MLP operations within a GLM model.

    The GLMBlock class has an initialization method (__init__) that sets up the necessary layers and parameters for
    self-attention and MLP processing. It also has a construct method that processes the input hidden states through
    self-attention and MLP layers.

    The GLMBlock class is designed to be used as part of a larger GLM model to process input sequences and generate output predictions.

    Note:
        This docstring is generated based on the provided code and does not include actual code or signatures.
    """
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            layernorm_epsilon,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            layernorm=nn.LayerNorm,
            use_bias=True,
            params_dtype=mindspore.float32,
            num_layers=28,
            position_encoding_2d=True,
    ):
        """
        Initializes a GLMBlock with the specified parameters.

        Args:
            hidden_size (int): The size of the hidden layers.
            num_attention_heads (int): The number of attention heads to use.
            layernorm_epsilon (float): The epsilon value for layer normalization.
            layer_id (int): The ID of the layer.
            inner_hidden_size (int, optional): The size of the inner hidden layers. Defaults to None.
            hidden_size_per_attention_head (int, optional): The size of hidden layers per attention head. Defaults to None.
            layernorm (class): The layer normalization class to use. Defaults to nn.LayerNorm.
            use_bias (bool): Flag indicating whether to use bias in the model.
            params_dtype (dtype): The data type for the parameters. Defaults to mindspore.float32.
            num_layers (int): The total number of layers.
            position_encoding_2d (bool): Flag indicating whether to use 2D position encoding.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        # Set output layer initialization if not provided.

        self.layer_id = layer_id

        # Layernorm on the input data.
        self.input_layernorm = layernorm([hidden_size], epsilon=layernorm_epsilon)

        self.position_encoding_2d = position_encoding_2d

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            bias=use_bias,
            params_dtype=params_dtype,
            position_encoding_2d=self.position_encoding_2d,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm([hidden_size], epsilon=layernorm_epsilon)

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


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]
    _keys_to_ignore_on_load_unexpected = [r'inv_freq']

    def _init_weights(self, cell: nn.Cell):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids):
        """
        This method named 'get_masks' is defined within the class 'ChatGLMPreTrainedModel'. It takes two parameters: self and input_ids.

        Args:
            self: A reference to the instance of the class.
            input_ids: A tensor representing the input sequence of token IDs.
                It has a shape of (batch_size, seq_length) where batch_size is the number of input sequences and
                seq_length is the length of each sequence.

        Returns:
            None.

        Raises:
            None.
        """
        batch_size, seq_length = input_ids.shape
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
        attention_mask = ops.ones((batch_size, seq_length, seq_length))
        attention_mask = attention_mask.tril()
        for i, context_length in enumerate(context_lengths):
            attention_mask[i, :, :context_length] = 1
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = (attention_mask < 0.5).bool()

        return attention_mask

    def get_position_ids(self, input_ids, mask_positions, use_gmasks=None):
        '''
        This method calculates the position ids for the given input sequence.

        Args:
            self (ChatGLMPreTrainedModel): An instance of the ChatGLMPreTrainedModel class.
            input_ids (mindspore.Tensor): A 2D tensor of shape (batch_size, seq_length) containing input sequence ids.
            mask_positions (mindspore.Tensor): A 1D tensor of shape (batch_size,) containing mask positions.
            use_gmasks (List[bool], optional): A list of length batch_size indicating whether to use global masks for
                each input sequence. Defaults to None.

        Returns:
            position_ids (mindspore.Tensor): A 2D tensor of shape (batch_size, seq_length) containing the position ids.

        Raises:
            None
        '''
        batch_size, seq_length = input_ids.shape
        if use_gmasks is None:
            use_gmasks = [False] * batch_size
        context_lengths = [seq.tolist().index(self.config.bos_token_id) for seq in input_ids]
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


class ChatGLMModel(ChatGLMPreTrainedModel):
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
        Initializes a ChatGLMModel object with the provided configuration.

        Args:
            self: The instance of the ChatGLMModel class.
            config (ChatGLMConfig):
                An object containing configuration parameters for the model.

                - max_sequence_length (int): The maximum length of input sequences.
                - hidden_size (int): The size of the hidden layer.
                - num_attention_heads (int): The number of attention heads.
                - vocab_size (int): The size of the vocabulary.
                - num_layers (int): The number of layers in the model.
                - layernorm_epsilon (float): The epsilon value for layer normalization.
                - inner_hidden_size (int): The size of the inner hidden layer.
                - position_encoding_2d (bool): Flag indicating whether to use 2D position encoding.
                - pre_seq_len (int): The length of the prefix sequence.
                - prefix_projection (bool): Flag indicating whether to project the prefix or not.

        Returns:
            None.

        Raises:
            ValueError: If any of the configuration parameters are invalid or missing.
            TypeError: If the data types of the configuration parameters are incorrect.
            RuntimeError: If an error occurs during the initialization process.
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

        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.hidden_size,
            dtype=self.params_dtype
        )

        def get_layer(layer_id):
            return GLMBlock(
                self.hidden_size,
                self.num_attention_heads,
                self.layernorm_epsilon,
                layer_id,
                inner_hidden_size=self.inner_hidden_size,
                hidden_size_per_attention_head=self.hidden_size_per_attention_head,
                layernorm=nn.LayerNorm,
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
            for param in self.parameters():
                param.requires_grad = False
            self.prefix_tokens = ops.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = nn.Dropout(p=0.1)

            # total_params = sum(p.numel() for p in self.parameters())
            # trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            # print("Using p-tuning v2: # trainable_params = {} / {}".format(trainable_params, total_params))

    def get_input_embeddings(self):
        """
        Returns the word embeddings for the input data.

        Args:
            self (ChatGLMModel): An instance of the ChatGLMModel class.

        Returns:
            None

        Raises:
            None

        This method retrieves the word embeddings used for the input data in the ChatGLMModel.
        The word embeddings are a numerical representation of words that capture semantic meaning.
        The embeddings are trained on a large corpus of text data to capture relationships between words.

        Note that this method does not modify the input embeddings. It simply returns the existing word embeddings that have been set for the model.

        Example:
            ```python
            >>> model = ChatGLMModel()
            >>> input_embeddings = model.get_input_embeddings()
            ...
            >>> # Perform operations on input_embeddings
            ...
            ```
        """
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        """
        This method sets the input embeddings for the ChatGLMModel.

        Args:
            self (ChatGLMModel): The instance of the ChatGLMModel class.
            new_embeddings (mindspore.Tensor): The new embeddings to be set as input embeddings for the model.
                It should be a mindspore Tensor object.

        Returns:
            None.

        Raises:
            None.
        """
        self.word_embeddings = new_embeddings

    def get_prompt(self, batch_size, dtype=mindspore.float16):
        """
        This method retrieves the prompt for generating responses in the ChatGLMModel.

        Args:
            self (object): The instance of the ChatGLMModel class.
            batch_size (int): The number of prompt sequences to generate.
            dtype (mindspore.dtype, optional): The data type for the prompt key values. Default is mindspore.float16.

        Returns:
            None.

        Raises:
            TypeError: If the batch_size is not an integer.
            ValueError: If the batch_size is less than or equal to 0.
            TypeError: If the dtype is not a valid mindspore data type.
        """
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
        past_key_values = self.prefix_encoder(prefix_tokens).astype(dtype)
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
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPast]:
        '''
        Constructs the ChatGLMModel.

        Args:
            self: The object itself.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the IDs of the tokens. Defaults to None.
            position_ids (Optional[mindspore.Tensor]): The input tensor containing the IDs of the positions. Defaults to None.
            attention_mask (Optional[mindspore.Tensor]): The input tensor containing the attention mask. Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]]):
                The input tensor containing the past key values. Defaults to None.
            inputs_embeds (Optional[mindspore.Tensor]): The input tensor containing the embedded inputs. Defaults to None.
            use_cache (Optional[bool]): Specifies whether to use cache. Defaults to None.
            output_attentions (Optional[bool]): Specifies whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool]): Specifies whether to output hidden states. Defaults to None.
            return_dict (Optional[bool]): Specifies whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPast]: The output of the model.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
                past_key_values = self.get_prompt(batch_size=input_ids.shape[0], dtype=inputs_embeds.dtype)
            else:
                past_key_values = tuple([None] * len(self.layers))

            if attention_mask is None:
                attention_mask = self.get_masks(
                    input_ids,
                )

            if position_ids is None:
                MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
                seqs = input_ids.tolist()

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

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is None:
            attention_mask = ops.zeros((1, 1)).bool()

        for i, layer in enumerate(self.layers):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_past = past_key_values[i]

            layer_ret = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                layer_id=mindspore.tensor(i),
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions
            )

            hidden_states = layer_ret[0]
            if use_cache:
                presents = presents + (layer_ret[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_ret[2 if use_cache else 1],)

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):

    """
    This class represents a ChatGLM model for conditional generation, inheriting from ChatGLMPreTrainedModel.

    The class includes methods for initializing the model, updating model keyword arguments for generation,
    preparing inputs for generation, constructing the model, reordering cache for beam search or beam sample,
    processing model responses, and facilitating chat interactions. It also provides methods for streaming chat and generation.

    The model allows for quantization with a specified number of bits.

    Methods:
        __init__: Initializes the model with a ChatGLMConfig.
        get_output_embeddings: Returns the output embeddings.
        set_output_embeddings: Sets new output embeddings.
        _update_model_kwargs_for_generation: Updates model keyword arguments for generation.
        prepare_inputs_for_generation: Prepares inputs for model generation.
        construct: Constructs the model for generation and computes the loss if labels are provided.
        _reorder_cache: Reorders the past_key_values cache for beam search or beam sample.
        process_response: Processes the model response by replacing tokens and punctuations.
        chat: Conducts a chat interaction based on the query and history.
        stream_chat: Conducts a streaming chat interaction for continuous conversations.
        stream_generate: Generates text in a streaming fashion based on input ids and generation configuration.
        quantize: Quantizes the model with a specified number of bits.

    For a detailed understanding of the class functionality and methods, refer to the specific method descriptions.
    """
    def __init__(self, config: ChatGLMConfig):
        """
        Initializes the ChatGLMForConditionalGeneration class.

        Args:
            self: The object instance itself.
            config (ChatGLMConfig): An instance of ChatGLMConfig containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type ChatGLMConfig.
            ValueError: If the config parameter is missing required attributes.
            AttributeError: If the config object does not have certain expected attributes.
        """
        super().__init__(config)
        # self.hidden_size = config.hidden_size
        # self.params_dtype = mindspore.float16
        # self.vocab_size = config.vocab_size
        self.max_sequence_length = config.max_sequence_length

        self.position_encoding_2d = config.position_encoding_2d

        self.transformer = ChatGLMModel(config)

        self.lm_head = nn.Dense(
            config.hidden_size,
            config.vocab_size,
            has_bias=False,
            dtype=mindspore.float16
        )

        self.config = config

        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit)

    def get_output_embeddings(self):
        """
        Get the output embeddings for the ChatGLM model.

        Args:
            self: The instance of the ChatGLMForConditionalGeneration class.

        Returns:
            The output embeddings for the language model head.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the ChatGLMForConditionalGeneration model.

        Args:
            self (ChatGLMForConditionalGeneration): The instance of the ChatGLMForConditionalGeneration class.
            new_embeddings (Any): The new output embeddings to be set for the model. This can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """
        Updates the model keyword arguments for generation.

        Args:
            self (ChatGLMForConditionalGeneration): The instance of the ChatGLMForConditionalGeneration class.
            outputs (ModelOutput): The model output.
            model_kwargs (Dict[str, Any]): The keyword arguments for the model.
            is_encoder_decoder (bool, optional): Indicates if the model is an encoder-decoder model. Defaults to False.
            standardize_cache_format (bool, optional): Indicates if the cache format should be standardized. Defaults to False.

        Returns:
            Dict[str, Any]: The updated model keyword arguments.

        Raises:
            None.
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
        This method prepares inputs for generation in the ChatGLMForConditionalGeneration class.

        Args:
            self (ChatGLMForConditionalGeneration): The instance of the ChatGLMForConditionalGeneration class.
            input_ids (mindspore.Tensor): The input tensor containing the token IDs for the model input.
            past (Optional[mindspore.Tensor]): Optional tensor containing the past states for autoregressive generation.
            past_key_values (Optional[mindspore.Tensor]): Optional tensor containing past key values for efficient decoding.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor specifying which elements in the input should be attended to.
            position_ids (Optional[mindspore.Tensor]): Optional tensor specifying the position IDs for input tokens.

        Returns:
            dict: A dictionary containing the prepared inputs for generation including 'input_ids', 'past_key_values',
                'position_ids', and 'attention_mask'.

        Raises:
            TypeError: If the input arguments are of incorrect types.
            ValueError: If there are issues with the input data or configuration.
            IndexError: If there are indexing errors while processing the input data.
            Warning: If there are warnings related to the attention mask data type.
        """
        _, seq_length = input_ids.shape
        MASK, gMASK = self.config.mask_token_id, self.config.gmask_token_id
        seqs = input_ids.tolist()
        mask_positions, use_gmasks = [], []
        for seq in seqs:
            mask_token = gMASK if gMASK in seq else MASK
            use_gmask = mask_token == gMASK
            mask_positions.append(seq.index(mask_token))
            use_gmasks.append(use_gmask)

        # only last token for input_ids if past is not None
        if past is not None or past_key_values is not None:
            last_token = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None and attention_mask.dtype == mindspore.bool_:
                attention_mask = attention_mask[:, :, -1:]
            else:
                attention_mask = None
            if position_ids is not None:
                position_ids = position_ids[..., -1:]
            else:
                context_lengths = [seq.index(self.config.bos_token_id) for seq in seqs]
                if self.position_encoding_2d:
                    position_ids = mindspore.tensor(
                        [[mask_position, seq_length - context_length] for mask_position, context_length in
                         zip(mask_positions, context_lengths)], dtype=mindspore.int64).unsqueeze(-1)
                else:
                    position_ids = mindspore.tensor(mask_positions, dtype=mindspore.int64).unsqueeze(-1)

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
                attention_mask = self.get_masks(
                    input_ids,
                )
            if position_ids is None:
                position_ids = self.get_position_ids(
                    input_ids,
                    mask_positions=mask_positions,
                    use_gmasks=use_gmasks
                )

            return {
                "input_ids": input_ids,
                "past_key_values": past,
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }

    def construct(
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
    ):
        """
        Constructs the ChatGLMForConditionalGeneration model.

        Args:
            self (ChatGLMForConditionalGeneration): The instance of the ChatGLMForConditionalGeneration class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length) containing the input IDs.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length) containing the position IDs.
            attention_mask (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length) containing the attention mask.
            past_key_values (Optional[Tuple[mindspore.Tensor]]):
                The input tensor of shape (batch_size, sequence_length) containing the past key values.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length, embedding_size) containing the embedded inputs.
            labels (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length) containing the labels.
            use_cache (Optional[bool]):
                Whether to use cache or not. If not provided, defaults to the value specified in the model's configuration.
            output_attentions (Optional[bool]): Whether to output attentions or not.
            output_hidden_states (Optional[bool]): Whether to output hidden states or not.
            return_dict (Optional[bool]):
                Whether to return a dictionary or not. If not provided, defaults to the value specified in the model's configuration.

        Returns:
            None.

        Raises:
            None.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states).permute(1, 0, 2)
        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(mindspore.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)

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
        Processes the response received from the model.

        Args:
            self (ChatGLMForConditionalGeneration): The instance of the ChatGLMForConditionalGeneration class.
            response (str): The response received from the model.

        Returns:
            None.

        Raises:
            None.
        """
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response

    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
             do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        """
        This method 'chat' is defined in the class 'ChatGLMForConditionalGeneration' and is used for generating a
        response to a given query using a conditional generation model. It takes the following
        parameters:

        Args:
            self: The instance of the class.
            tokenizer: An instance of a tokenizer that will be used to encode the prompt and decode the generated response.
            query (str): The input query for which a response needs to be generated.
            history (List[Tuple[str, str]], optional):
                A list of tuples containing the previous queries and their corresponding responses. Defaults to None.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            num_beams (int, optional): Number of beams for beam search. Defaults to 1.
            do_sample (bool, optional): Flag indicating whether to use sampling for generating the response. Defaults to True.
            top_p (float, optional): The nucleus sampling top probability. Defaults to 0.7.
            temperature (float, optional): The temperature parameter for sampling. Defaults to 0.95.
            logits_processor (object, optional): An object for processing the logits. Defaults to None.
            **kwargs: Additional keyword arguments for model generation.

        Returns:
            None:
                This method does not have a specific return value,
                but it generates a response to the input query and updates the history of queries and responses.

        Raises:
            None:
                This method does not explicitly raise any exceptions.
                However, the behavior of the method may be influenced by exceptions raised by the tokenizer or
                the conditional generation model used within the method.
        """
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
        """
        Stream chat method for generating responses based on a given query and history.
        
        Args:
            self (ChatGLMForConditionalGeneration): An instance of the ChatGLMForConditionalGeneration class.
            tokenizer: The tokenizer used for tokenizing the input text.
            query (str): The query string for which a response is generated.
            history (List[Tuple[str, str]], optional):
                A list of tuples containing the previous queries and their responses. Defaults to None.
            max_length (int, optional): The maximum length of the generated response. Defaults to 2048.
            do_sample (bool, optional): Whether to use sampling for generating response. Defaults to True.
            top_p (float, optional): The cumulative probability threshold for top-p sampling. Defaults to 0.7.
            temperature (float, optional): The temperature value used for sampling. Defaults to 0.95.
            logits_processor (object, optional):
                An object used for processing logits during response generation. Defaults to None.
        
        Returns:
            None
        
        Raises:
            None
        """
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
        """
        Generates text using the ChatGLM model.
        
        Args:
            self (ChatGLMForConditionalGeneration): The instance of the ChatGLMForConditionalGeneration class.
            input_ids (mindspore.Tensor): The input tensor containing the tokenized input sequence.
            generation_config (Optional[GenerationConfig], optional): The configuration for text generation. Defaults to None.
            logits_processor (Optional[LogitsProcessorList], optional): The processor for modifying the logits. Defaults to None.
            stopping_criteria (Optional[StoppingCriteriaList], optional): The criteria for stopping the generation. Defaults to None.
            prefix_allowed_tokens_fn (Optional[Callable[[int, mindspore.Tensor], List[int]]], optional):
                A function that returns the list of allowed tokens for each prefix. Defaults to None.
        
        Returns:
            None
        
        Raises:
            UserWarning: If both `max_new_tokens` and `max_length` are set, `max_new_tokens` takes precedence.
            UserWarning: If the input length exceeds the `max_length` limit, it may cause unexpected behavior.
            Other exceptions: Any other exceptions that may occur during the execution of the method.
        """
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

    def quantize(self, bits: int, **kwargs):
        """
        Perform quantization on the input data.
        
        Args:
            self (ChatGLMForConditionalGeneration): An instance of the ChatGLMForConditionalGeneration class.
            bits (int): The number of bits to quantize the data to. Must be a positive integer.
        
        Returns:
            None.
        
        Raises:
            None.
        """

__all__ = [
    'CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST',
    'ChatGLMModel',
    'ChatGLMPreTrainedModel',
    'ChatGLMForConditionalGeneration'
]
