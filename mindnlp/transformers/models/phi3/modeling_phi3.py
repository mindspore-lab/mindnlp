# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

""" MindSpore Phi-3 model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import Normal

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from .configuration_phi3 import Phi3Config


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "microsoft/Phi-3-mini-4k-instruct"
_CONFIG_FOR_DOC = "Phi3Config"

PHI3_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    # See all Phi-3 models at https://huggingface.co/models?filter=Phi-3
]


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Phi3
class Phi3RMSNorm(nn.Cell):

    """ 
        Phi3RMSNorm is a custom normalization layer that performs the Phi3 RMS normalization, equivalent to T5LayerNorm.
    
        This class inherits from the nn.Cell class in the MindSpore framework.
    
        Attributes:
            weight (Parameter): The weight parameter for the normalization layer.
            variance_epsilon (float): A small value added to the variance to avoid division by zero.
    
        Methods:
            __init__:
                Initializes a new instance of the Phi3RMSNorm class.
                
            construct:
                Applies Phi3 RMS normalization to the input hidden_states.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        This method constructs Phi3RMSNorm by performing normalization on the hidden_states.
        
        Args:
            self: Instance of the Phi3RMSNorm class.
            hidden_states: A tensor containing the hidden states to be normalized.
                It should be of type 'Tensor' and compatible with the operations performed in the method.
        
        Returns:
            None.
        
        Raises:
            None
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    This function takes an attention_mask as input and returns three values: indices, cu_seqlens, and max_seqlen_in_batch.
    
    Args:
        attention_mask (tensor): A tensor representing the attention mask. It should have a shape of
            [batch_size, sequence_length], where batch_size is the number of sequences in the batch and sequence_length
            is the maximum length of any sequence. The attention mask is used to indicate which tokens should be attended
            to and which should be ignored.

    Returns:
        indices (tensor): A tensor containing the indices of non-zero values in the flattened attention_mask tensor.
            This tensor has a shape of [num_non_zero_values], where num_non_zero_values is the total
            number of non-zero values in the attention_mask.

        cu_seqlens (tensor): A tensor representing the cumulative sum of sequence lengths in the batch.
            It has a shape of [batch_size + 1] and is padded with a zero at the beginning. The cumulative sum is
            computed along the 0th axis of the seqlens_in_batch tensor, which is obtained by summing the attention_mask
            tensor along the -1th axis.

        max_seqlen_in_batch (int): The maximum sequence length in the batch. This is a scalar value indicating the
            length of the longest sequence in the batch.

    Raises:
        None.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with gemma->phi3, Gemma->Phi3
class Phi3RotaryEmbedding(nn.Cell):

    """
    This class represents the Phi3RotaryEmbedding, a rotary positional embedding layer used in neural network models.
    It is a subclass of nn.Cell.

    The Phi3RotaryEmbedding class provides methods for constructing rotary embeddings based on input tensors and
    position IDs. It utilizes cosine and sine functions to generate embeddings with rotational properties.

    Attributes:
        dim (int): The dimension of the embeddings.
        max_position_embeddings (int): The maximum number of position embeddings.
        base (int): The base value for calculating inverse frequencies.
        inv_freq (ndarray): The inverse frequencies calculated based on the dimension and base values.

    Methods:
        construct(x, position_ids, seq_len=None):
            Constructs rotary embeddings based on the input tensor and position IDs.

            Args:

            - x (Tensor): The input tensor.
            - position_ids (Tensor): The position IDs.
            - seq_len (int, optional): The length of the sequence. Defaults to None.

            Returns:

            - Tensor: The cosine and sine embeddings, converted to the same data type as the input tensor.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes a new instance of the Phi3RotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int): The maximum number of position embeddings allowed (default is 2048).
            base (int): The base value for calculations (default is 10000).

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = None

    def construct(self, x, position_ids, seq_len=None):
        '''
        This method constructs the rotary embedding for the Phi3RotaryEmbedding class.

        Args:
            self (Phi3RotaryEmbedding): The instance of the Phi3RotaryEmbedding class.
            x (Tensor): The input tensor for which the rotary embedding is being constructed.
            position_ids (Tensor): The tensor containing the position IDs.
            seq_len (int, optional): The length of the sequence. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Returns a tuple containing the cosine and sine values of the constructed
                rotary embedding. Both tensors have the same shape as the input tensor x.

        Raises:
            ValueError: If the length of the position_ids tensor does not match the sequence length.
            TypeError: If the input parameters are not of the expected types.
        '''
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3SuScaledRotaryEmbedding(Phi3RotaryEmbedding):

    """
    The `Phi3SuScaledRotaryEmbedding` class represents a specialized implementation of the `Phi3RotaryEmbedding` class,
    which provides functionalities for constructing a scaled rotary embedding for a given input tensor.

    Attributes:
        `dim` (int): The dimensionality of the input tensor.
        `config` (object): The configuration object containing various parameters.
        `short_factor` (float): The scaling factor for short sequences.
        `long_factor` (float): The scaling factor for long sequences.
        `original_max_position_embeddings` (int): The maximum number of positions in the original input tensor.

    Methods:
        `__init__`: Initializes the `Phi3SuScaledRotaryEmbedding` object.

            Args:

            - `dim` (int): The dimensionality of the input tensor.
            - `config` (object): The configuration object containing various parameters.

        `construct`: Constructs the scaled rotary embedding.

            Args:

            - `x` (tensor): The input tensor.
            - `position_ids` (tensor): The tensor containing position indices.
            - `seq_len` (int, optional): The length of the sequence. Defaults to None.

            Returns:

            - `cos` (tensor): The cosine component of the scaled rotary embedding.
            - `sin` (tensor): The sine component of the scaled rotary embedding.
    """
    def __init__(self, dim, config):
        """
        Initializes an instance of the Phi3SuScaledRotaryEmbedding class.

        Args:
            self (Phi3SuScaledRotaryEmbedding): The instance of the class.
            dim (int): The dimension of the embedding.
            config (object): The configuration object containing various settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    def construct(self, x, position_ids, seq_len=None):
        """
        Constructs the scaled rotary embedding for the Phi3SuScaledRotaryEmbedding.

        Args:
            self: The object instance.
            x (Tensor): The input tensor for which the scaled rotary embedding is constructed.
            position_ids (Tensor): The position indices for the input tensor.
            seq_len (int, optional): The length of the sequence. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of tensors containing the cosine and sine of the scaled rotary embedding.

        Raises:
            ValueError: If the sequence length exceeds the original maximum position embeddings.
            TypeError: If the input tensors are not of the expected data type.
        """
        seq_len = ops.max(position_ids)[0] + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = mindspore.Tensor(self.long_factor, dtype=mindspore.float32)
        else:
            ext_factors = mindspore.Tensor(self.short_factor, dtype=mindspore.float32)

        inv_freq_shape = ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

        cos = emb.cos() * scaling_factor
        sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3YarnScaledRotaryEmbedding(Phi3RotaryEmbedding):

    """
    This class represents the Phi3YarnScaledRotaryEmbedding, a subclass of Phi3RotaryEmbedding.
    It provides methods for constructing scaled rotary embeddings for Phi3Yarn models.

    Attributes:
        dim (int): The dimension of the embeddings.
        config (object): The configuration object containing various parameters.
        short_factor (float): The scaling factor for short sequences.
        long_factor (float): The scaling factor for long sequences.
        original_max_position_embeddings (int): The original maximum position embeddings.

    Methods:
        __init__: Initializes a Phi3YarnScaledRotaryEmbedding instance.
        construct: Constructs the scaled rotary embeddings.

    """
    def __init__(self, dim, config):
        """
        Initializes a Phi3YarnScaledRotaryEmbedding object with the specified dimension and configuration.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embedding space.
            config (object): An object containing configuration parameters including max_position_embeddings,
                rope_theta, rope_scaling, and original_max_position_embeddings.

        Returns:
            None.

        Raises:
            KeyError: If the 'short_factor' or 'long_factor' keys are missing in the 'rope_scaling' dictionary within
                the config object.
            TypeError: If the 'max_position_embeddings' or 'original_max_position_embeddings' attributes are not present
                in the config object.
        """
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    def construct(self, x, position_ids, seq_len=None):
        """
        Constructs the Phi3YarnScaledRotaryEmbedding.

        Args:
            self: The instance of the Phi3YarnScaledRotaryEmbedding class.
            x: A tensor representing the input data.
            position_ids: A tensor containing the position IDs for each element in the input tensor.
            seq_len: An optional integer representing the length of the input sequence. If not provided, it is calculated as
                the maximum value in the position_ids tensor plus one.

        Returns:
            None

        Raises:
            None
        """
        seq_len = ops.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = mindspore.Tensor(self.long_factor, dtype=mindspore.float32)
        else:
            ext_factors = mindspore.Tensor(self.short_factor, dtype=mindspore.float32)

        inv_freq_shape = ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = 0.1 * math.log(scale) + 1.0

        cos = emb.cos() * scaling_factor
        sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = x.tensor_split(2, -1)
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mindspore.Tensor`): The query tensor.
        k (`mindspore.Tensor`): The key tensor.
        cos (`mindspore.Tensor`): The cosine part of the rotary embedding.
        sin (`mindspore.Tensor`): The sine part of the rotary embedding.
        position_ids (`mindspore.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(mindspore.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Phi3MLP(nn.Cell):

    """
    This class represents a multi-layer perceptron (MLP) module with a Phi3 activation function.
    It inherits from the nn.Cell class.

    The Phi3MLP module is used for processing hidden states in a neural network. It consists of an up projection layer,
    a gate activation function, and a down projection layer.

    Attributes:
        config (object): An object containing configuration settings for the module.

    Methods:
        __init__:
            Initializes a Phi3MLP instance.

            Args:

            - config (object): An object containing configuration settings for the module.

        construct:
            Constructs the Phi3MLP module.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states tensor.

            Returns:

           - mindspore.Tensor: The output tensor after applying the Phi3MLP module.
    """
    def __init__(self, config):
        """
        Initializes an instance of the Phi3MLP class.

        Args:
            self: The instance of the Phi3MLP class.
            config:
                An object containing configuration settings for the Phi3MLP model.

                - Type: Any
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Dense(config.hidden_size, 2 * config.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(config.intermediate_size, config.hidden_size, has_bias=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs and processes the hidden states using the Phi3MLP class.

        Args:
            self (Phi3MLP): The instance of the Phi3MLP class.
            hidden_states (mindspore.Tensor): The input tensor containing the hidden states to be processed.

        Returns:
            mindspore.Tensor: The processed tensor representing the output of the method.

        Raises:
            None
        """
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, axis=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


# Copied from transformers.models.llama.modeling_llama.repeat_kv with llama->phi
def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Phi3Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
        """
        Initializes an instance of the `Phi3Attention` class.

        Args:
            self: The instance of the `Phi3Attention` class.
            config (Phi3Config): An instance of the `Phi3Config` class containing the configuration settings
                for the attention layer.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_heads`.

        Notes:
            - Instantiating `Phi3Attention` without passing a `layer_idx` is not recommended and may lead to
            errors during the forward call if caching is used. It is advised to provide a `layer_idx` when
            creating this class.
            - The `Phi3Attention` class expects `hidden_size` to be divisible by `num_heads`.

            - The following attributes are initialized within the `__init__` method:

                - `self.config`: An instance of the `Phi3Config` class containing the configuration settings.
                - `self.layer_idx`: The index of the layer.
                - `self.attention_dropout`: The dropout rate for attention.
                - `self.hidden_size`: The hidden size of the layer.
                - `self.num_heads`: The number of attention heads.
                - `self.head_dim`: The dimension of each attention head.
                - `self.num_key_value_heads`: The number of key-value attention heads.
                - `self.num_key_value_groups`: The number of groups formed by key-value attention heads.
                - `self.max_position_embeddings`: The maximum number of position embeddings.
                - `self.original_max_position_embeddings`: The original maximum number of position embeddings.
                - `self.rope_theta`: The theta value for relative position encoding.
                - `self.rope_scaling`: The scaling factor for relative position encoding.
                - `self.is_causal`: A boolean indicating if the attention is causal.
                - `self.o_proj`: A fully connected layer for projecting the output.
                - `self.qkv_proj`: A fully connected layer for projecting the queries, keys, and values.
                - `self._init_rope()`: A private method for initializing the relative position encoding.

        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)
        self.qkv_proj = nn.Dense(self.hidden_size, op_size, has_bias=False)
        self._init_rope()

    def _init_rope(self):
        """
        Initializes the RoPE (Rotary Positional Encoding) for the Phi3Attention class.

        Args:
            self: The instance of the Phi3Attention class.

        Returns:
            None

        Raises:
            ValueError: If the RoPE scaling type is unknown.

        This method initializes the RoPE based on the provided configurations. If the 'rope_scaling' attribute is None,
        it creates a Phi3RotaryEmbedding object with the specified parameters. Otherwise, it checks the type of scaling
        specified in the 'rope_scaling' attribute and creates the appropriate Phi3ScaledRotaryEmbedding object
        accordingly. The Phi3ScaledRotaryEmbedding objects provide additional scaling options for the Rotary Positional
        Encoding.

        The available scaling types are as follows:

        - 'su': Creates a Phi3SuScaledRotaryEmbedding object.
        - 'yarn': Creates a Phi3YarnScaledRotaryEmbedding object.

        Note:
            The Phi3SuScaledRotaryEmbedding and Phi3YarnScaledRotaryEmbedding classes are specific implementations of
            the Phi3RotaryEmbedding class with additional scaling capabilities.

        Example:
            ```python()
            >>> # Initialize RoPE without scaling
            >>> _init_rope()
            ...
            >>> # Initialize RoPE with 'su' scaling
            >>> self.rope_scaling = {'type': 'su'}
            >>> _init_rope()
            ...
            >>> # Initialize RoPE with 'yarn' scaling
            >>> self.rope_scaling = {'type': 'yarn'}
            >>> _init_rope()
            ```
        """
        if self.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "su":
                self.rotary_emb = Phi3SuScaledRotaryEmbedding(self.head_dim, self.config)
            elif scaling_type == "yarn":
                self.rotary_emb = Phi3YarnScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        This method constructs the Phi3Attention mechanism.

        Args:
            self: The instance of the Phi3Attention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional mask tensor of shape
                (batch_size, 1, sequence_length, sequence_length) to mask some positions in the input.
            position_ids (Optional[mindspore.Tensor]): An optional tensor of shape (batch_size, sequence_length)
                containing the position indices.
            past_key_value (Optional[Cache]): An optional cache storing the past key and value states for efficient
                auto-regressive decoding.
            output_attentions (bool): A flag indicating whether to output the attention weights.
            use_cache (bool): A flag indicating whether to use the cache for storing key and value states.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]: A tuple containing
                the attention output tensor of shape (batch_size, sequence_length, hidden_size),
                optionally the attention weights tensor, and optionally the updated past key and value states.

        Raises:
            ValueError: Raised if the cache structure has changed, if the attention weights or mask tensors have
                incorrect shapes, or if the output tensors have unexpected shapes.
        """
        bsz, q_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(value_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


PHI3_ATTENTION_CLASSES = {
    "eager": Phi3Attention,
}


class Phi3DecoderLayer(nn.Cell):

    '''
    Phi3DecoderLayer represents a single layer of the Phi3 decoder. This layer includes self-attention, residual
    connections, layer normalization, and a multi-layer perceptron (MLP) sublayer.

    This class inherits from the nn.Cell class and is designed to be used as a building block for constructing Phi3
    decoder models.

    The __init__ method initializes the Phi3DecoderLayer with the provided configuration and layer index.
    It sets up the self-attention mechanism, MLP, input layer normalization, and dropout layers.

    The construct method processes the input hidden states through the layer. It applies input layer normalization,
    self-attention, residual connections, post-attention layer normalization, and the MLP sublayer. The method also
    handles optional arguments such as attention_mask, position_ids, past_key_value, output_attentions, and use_cache,
    and returns the resulting hidden states along with optional outputs based on the provided arguments.

    Note:
        The construct method also issues a warning if the 'padding_mask' argument is used, as it is deprecated and
        will be removed in a future version in favor of 'attention_mask'.

    Args:
        hidden_states (mindspore.Tensor):
            Input to the layer of shape `(batch, seq_len, embed_dim)`.
        attention_mask (mindspore.Tensor, *optional*):
            Attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large
            negative values.
        position_ids (mindspore.Tensor of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
            `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        use_cache (bool, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (Tuple(mindspore.Tensor), *optional*):
            Cached past key and value projection states

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
            The resulting hidden states, and optionally, the self-attention weights and present key-value states
            if requested.
    '''
    def __init__(self, config: Phi3Config, layer_idx: int):
        """
        Initializes a new instance of the Phi3DecoderLayer class.

        Args:
            self (Phi3DecoderLayer): The current instance of the Phi3DecoderLayer class.
            config (Phi3Config): The configuration object containing parameters for the decoder layer.
            layer_idx (int): The index of the decoder layer.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes the Phi3DecoderLayer object with the provided configuration and layer index.
            It sets up the self-attention mechanism, multi-layer perceptron, input layer normalization, and other
            components required for the decoder layer.

            - config: The Phi3Config object that contains the configuration parameters for the decoder layer.
            This includes parameters such as hidden size, dropout rate, and RMS normalization epsilon.

            - layer_idx: An integer representing the index of the decoder layer.
            This index is used to identify the layer and is required for initializing the self-attention mechanism.

            The method does not return any value.
        """
        super().__init__()

        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES['eager'](config, layer_idx=layer_idx)

        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        '''
        Constructs a Phi3DecoderLayer object.

        Args:
            self: The object itself.
            hidden_states (mindspore.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (mindspore.Tensor, optional): Attention mask of size `(batch, 1, tgt_len, src_len)`,
                where padding elements are indicated by very large negative values.
            position_ids (mindspore.Tensor, optional): Indices of positions of each input sequence tokens in the
                position embeddings. Selected in the range `[0, config.n_positions - 1]`. (default: None)
            past_key_value (Tuple[mindspore.Tensor], optional): Cached past key and value projection states.
                (default: None)
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail. (default: False)
            use_cache (bool, optional): If set to True, `past_key_values` key value states are returned and can be used
                to speed up decoding (see `past_key_values`). (default: False)

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]: A tuple containing the
                hidden states of shape `(batch, seq_len, embed_dim)`. Optionally, the tuple may also contain the
                attentions tensors of all attention layers and the cached past key and value projection states.

        Raises:
            None.
        '''
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`mindspore.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Phi3PreTrainedModel(PreTrainedModel):

    """
    This class represents a Phi3PreTrainedModel, which is a subclass of PreTrainedModel.

    Phi3PreTrainedModel inherits the following methods from PreTrainedModel:

    - forward(input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
    This method performs the forward pass for the Phi3PreTrainedModel. It takes input_ids as input and
    returns the model's output.

    - save_pretrained(save_directory):
    This method saves the model's weights and configuration to the specified directory.

    - from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
    This method loads the pretrained model from the specified path or model name. Additional
    arguments can be passed to customize the loading process.

    - config_class:
    This attribute holds the configuration class of the model.

    - base_model_prefix:
    This attribute holds the prefix used by the model's modules.

    The Phi3PreTrainedModel class introduces the following methods:

    - _init_weights:
    This method initializes the weights for the given module. If the module is of type nn.Dense,
    the weight is initialized using the Normal distribution with a standard deviation of
    self.config.initializer_range. If the module has a bias, it is initialized with zeros.
    If the module is of type nn.Embedding, the weight is randomly initialized using the Normal
    distribution with a standard deviation of self.config.initializer_range. If the module has
    a padding index, the weight at the padding index is set to zero.

    Note:
        This class does not provide an implementation for the forward method. The implementation
        should be provided by subclasses that inherit from Phi3PreTrainedModel.
    """
    config_class = Phi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, module):
        """
        Initializes the weights of a given module.

        Args:
            self (Phi3PreTrainedModel): The instance of the Phi3PreTrainedModel class.
            module: The module to initialize the weights for.

        Returns:
            None: This method modifies the weights of the given module in-place.

        Raises:
            None.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Dense):
            module.weight.initialize(Normal(std))
            if module.bias is not None:
                module.bias.initialize('zeros')
        elif isinstance(module, nn.Embedding):
            weight = np.random.normal(0.0, std, module.weight.shape)
            if module.padding_idx:
                weight[module.padding_idx] = 0

            module.weight.set_data(Tensor(weight, module.weight.dtype))

class Phi3Model(Phi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]

    Args:
        config: Phi3Config
    """
    def __init__(self, config: Phi3Config):
        """
        Initializes a new instance of the Phi3Model class.

        Args:
            self: The object instance.
            config (Phi3Config): The configuration object for Phi3Model.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes a new instance of the Phi3Model class. It takes in a configuration object,
            'config', which is of type Phi3Config. The 'config' parameter contains various settings and
            hyperparameters for the model.

        The method performs the following steps:

        1. Calls the __init__ method of the parent class (super().__init__(config)) to initialize the parent class
        with the provided configuration.
        2. Sets the 'padding_idx' attribute to the 'pad_token_id' value from the 'config' object.
        This value represents the padding token index in the vocabulary.
        3. Sets the 'vocab_size' attribute to the 'vocab_size' value from the 'config' object.
        This value represents the size of the vocabulary.
        4. Initializes the 'embed_tokens' attribute as an instance of the nn.Embedding class.
        It takes the 'vocab_size', 'hidden_size', and 'padding_idx' values from the 'config' object as parameters.
        This embedding layer is responsible for converting input tokens to their corresponding embeddings.
        5. Initializes the 'embed_dropout' attribute as an instance of the nn.Dropout class.
        It takes the 'embd_pdrop' value from the 'config' object as a parameter.
        This dropout layer is applied to the embeddings.
        6. Initializes the 'layers' attribute as an instance of the nn.CellList class.
        It contains Phi3DecoderLayer instances, one for each layer index from 0 to 'num_hidden_layers' - 1 (inclusive).
        Each Phi3DecoderLayer is initialized with the 'config' object and the corresponding layer index.
        7. Sets the '_attn_implementation' attribute to the '_attn_implementation' value from the 'config' object.
        This value represents the implementation type of the attention mechanism.
        8. Initializes the 'norm' attribute as an instance of the Phi3RMSNorm class. It takes the 'hidden_size'
        and 'eps' values from the 'config' object as parameters. This layer applies root mean square normalization to
        the hidden states.
        9. Sets the 'gradient_checkpointing' attribute to False. This attribute determines whether gradient
        checkpointing is enabled during training.
        10. Calls the 'post_init' method, which can be overridden by subclasses to perform additional initialization
        steps.

        Note:
            This method is called automatically when creating a new instance of the Phi3Model class.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.CellList(
            [Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the Phi3Model.

        Args:
            self: The instance of the Phi3Model class.

        Returns:
            embed_tokens: The method returns the input embeddings stored in the 'embed_tokens' attribute of the
                Phi3Model instance.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the Phi3Model.

        Args:
            self: The instance of the Phi3Model class.
            value: A tensor representing the input embeddings. It should have a shape of
                (batch_size, sequence_length, embedding_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the Phi3Model.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor, optional): The input tensor of shape (batch_size, seq_length). Defaults to None.
            attention_mask (Optional[mindspore.Tensor], optional): The attention mask tensor of shape
                (batch_size, seq_length). Defaults to None.
            position_ids (Optional[mindspore.Tensor], optional): The position ids tensor of shape
                (batch_size, seq_length). Defaults to None.
            past_key_values (Optional[List[mindspore.Tensor]], optional): List of past key value tensors.
                Defaults to None.
            inputs_embeds (Optional[mindspore.Tensor], optional): The input embeddings tensor of shape
                (batch_size, seq_length). Defaults to None.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Defaults to None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The output of the Phi3Model.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If attempting to perform batched generation with padding_side='right' in flash_attention_2
                implementation.
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
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3ForCausalLM(Phi3PreTrainedModel):
    r"""
    A class representing the Phi3 model for causal language modeling.

    This class extends the Phi3PreTrainedModel class and provides methods for initializing the model, setting and
    getting input and output embeddings, setting and getting the decoder, constructing the model, and preparing inputs
    for generation.

    Attributes:
        model (Phi3Model): The Phi3 model.
        vocab_size (int): The size of the vocabulary.
        lm_head (nn.Dense): The language model head.

    Methods:
        __init__: Initializes the Phi3ForCausalLM instance.
        get_input_embeddings: Returns the input embeddings.
        set_input_embeddings: Sets the input embeddings.
        get_output_embeddings: Returns the output embeddings.
        set_output_embeddings: Sets the output embeddings.
        set_decoder: Sets the decoder.
        get_decoder: Returns the decoder.
        construct: Constructs the model and returns the output.
        prepare_inputs_for_generation: Prepares inputs for generation.
        _reorder_cache: Reorders the cache based on the beam index.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, Phi3ForCausalLM
        ...
        >>> model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        ...
        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        ...
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```
    """
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi3
    def __init__(self, config):
        """
        Initializes a new instance of the Phi3ForCausalLM class.

        Args:
            self: The object itself.
            config:
                A configuration object of type Config, containing the necessary parameters for model initialization.

                - Type: Config
                - Purpose: To provide the required parameters for model initialization.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model = Phi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the Phi3ForCausalLM model.

        Args:
            self: An instance of the Phi3ForCausalLM class.
                This parameter represents the current instance of the Phi3ForCausalLM class.
                It is used to access the model's embed_tokens attribute.

        Returns:
            None:
                This method returns None as it directly returns the embed_tokens attribute of the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Method to set input embeddings for the Phi3ForCausalLM model.

        Args:
            self (Phi3ForCausalLM): The instance of the Phi3ForCausalLM class.
            value (Any): The input embeddings to be set for the model.
                Should be compatible with the model's embed_tokens attribute.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings of the Phi3ForCausalLM model.

        This method takes no additional parameters.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the Phi3ForCausalLM model.

        Args:
            self (Phi3ForCausalLM): The instance of the Phi3ForCausalLM class.
            new_embeddings: The new embeddings to be set for the model's output. It can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        """
        Sets the decoder for the Phi3ForCausalLM class.

        Args:
            self (Phi3ForCausalLM): The instance of Phi3ForCausalLM class.
            decoder: The decoder object to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        """
        This method returns the decoder model used in the Phi3ForCausalLM class.

        Args:
            self: The instance of the Phi3ForCausalLM class.

        Returns:
            model: The decoder model associated with the Phi3ForCausalLM instance.

        Raises:
            None.
        """
        return self.model

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, Phi3ForCausalLM
            ...
            >>> model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
            ...
            >>> prompt = "This is an example script ."
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = ops.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        This method prepares inputs for generation in the Phi3ForCausalLM class.

        Args:
            self (object): The instance of Phi3ForCausalLM.
            input_ids (torch.Tensor): The input tensor containing token indices for the input sequence.
            past_key_values (Union[None, Cache, Tuple[Tensor, Tensor]]): A cache of past key values or the tuple
                of past key and value tensors. Defaults to None.
            attention_mask (Optional[torch.Tensor]): An optional tensor containing attention mask values
                for the input sequence.
            inputs_embeds (Optional[torch.Tensor]): An optional tensor containing the embedded inputs.

        Returns:
            model_inputs (Dict[str, Union[torch.Tensor, Cache]]):
                A dictionary containing the model inputs with the following keys:

                - 'inputs_embeds': The embedded inputs if 'inputs_embeds' is not None.
                - 'input_ids': The input tensor containing token indices if 'inputs_embeds' is None.
                - 'position_ids': The position indices tensor.
                - 'past_key_values': The cache of past key values.
                - 'use_cache': A boolean indicating whether to use cache.
                - 'attention_mask': The attention mask tensor.

        Raises:
            ValueError: If the dimensions of the input tensors are incompatible.
            TypeError: If the input types are invalid or incompatible.
        """
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache based on the beam index for the Phi3ForCausalLM class.

        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer. Each layer's past
                state is represented by a tensor.
            beam_idx (tensor): A tensor containing the beam index.

        Returns:
            tuple: A tuple containing the reordered past key-value states for each layer. Each layer's reordered past
                state is represented by a tensor.

        Raises:
            None.

        This static method reorders the cache based on the provided beam index. It iterates through each layer's
        past key-value states and selects the corresponding states from the past based on the beam index.
        The reordered past key-value states are then returned as a tuple.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Phi3, LLAMA->PHI3, self.transformer->self.model, transformer_outputs->model_outputs
class Phi3ForSequenceClassification(Phi3PreTrainedModel):

    """
    This class represents a Phi3 model for sequence classification. It is a subclass of the Phi3PreTrainedModel class.

    Attributes:
        num_labels (int): The number of labels for sequence classification.
        model (Phi3Model): The Phi3 model for sequence classification.
        score (nn.Dense): The dense layer for scoring the hidden states.

    Methods:
        __init__: Initializes a new instance of the Phi3ForSequenceClassification class.
        get_input_embeddings: Retrieves the input embeddings from the Phi3 model.
        set_input_embeddings: Sets the input embeddings for the Phi3 model.
        construct: Constructs the Phi3 model for sequence classification.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the Phi3ForSequenceClassification class.

        Args:
            self (Phi3ForSequenceClassification): The current instance of the Phi3ForSequenceClassification class.
            config (object): An object containing configuration settings for the model.
                It should have the following attributes:

                - num_labels (int): The number of labels/classes for classification.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Phi3Model(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the Phi3ForSequenceClassification model.

        Args:
            self: An instance of the Phi3ForSequenceClassification class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method is used to extract the input embeddings from the Phi3ForSequenceClassification model.
            The input embeddings represent the learned representations of the input tokens in the model.

            This method takes one parameter 'self', which refers to the current instance of the
            Phi3ForSequenceClassification class. It is required to access the model and its embedded tokens.

        Example:
            ```python
            >>> model = Phi3ForSequenceClassification()
            >>> input_embeddings = model.get_input_embeddings()
            ```

        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Phi3ForSequenceClassification model.

        Args:
            self (Phi3ForSequenceClassification): The instance of Phi3ForSequenceClassification.
            value (Tensor): The input embeddings to be set for the model. Should be a tensor of shape
                (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
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

        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = model_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

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
            output = (pooled_logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


# Copied from transformers.models.mpt.modeling_mpt.MptForTokenClassification with Mpt->Phi3,MPT->PHI3,self.transformer->self.model,transformer_outputs->model_outputs
class Phi3ForTokenClassification(Phi3PreTrainedModel):

    """
    Phi3ForTokenClassification is a class that represents a token classification model for Phi3, inheriting from
    Phi3PreTrainedModel.

    The class includes an __init__ method for initializing the model with a Phi3Config object, setting up the
    necessary components such as the model architecture, dropout layers, and classifier for token classification.

    It also contains a construct method for performing the token classification task, taking input tensors,
    past key values, attention masks, and other optional arguments. It computes the classification loss using
    cross-entropy and returns the loss along with logits and hidden states if specified in the return_dict.

    Attributes:
        num_labels: The number of labels for token classification.
        model: The Phi3Model instance for processing inputs.
        dropout: Dropout layer for regularization.
        classifier: Dense layer for classification.

    Methods:
        __init__: Constructor method to initialize the Phi3ForTokenClassification instance.
        construct: Method for performing token classification task using the model.

    Note:
        Ensure to set the appropriate labels for computing the loss, and handle the return_dict parameter for
        controlling the output format.
    """
    def __init__(self, config: Phi3Config):
        """
        Initializes an instance of Phi3ForTokenClassification with the provided configuration.

        Args:
            self: The instance of the Phi3ForTokenClassification class.
            config (Phi3Config): An instance of Phi3Config containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type Phi3Config.
            AttributeError: If the config object does not have the required attributes.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = Phi3Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = model_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            batch_size, seq_length = labels.shape
            loss = ops.cross_entropy(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

__all__ = [
    'Phi3ForTokenClassification',
    'Phi3ForSequenceClassification',
    'Phi3ForCausalLM',
    'Phi3Model',
    'Phi3PreTrainedModel',
]
