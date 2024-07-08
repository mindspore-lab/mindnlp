# coding=utf-8
# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
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
# pylint: disable=unexpected-keyword-arg
""" MindSpore Phi model."""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Normal, initializer

from mindnlp.utils import logging, get_default_dtype
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
from .configuration_phi import PhiConfig

logger = logging.get_logger(__name__)


PHI_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    # See all Phi models at https://hf-mirror.com/models?filter=phi
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    This function extracts necessary data from the attention_mask tensor for further processing.
    
    Args:
        attention_mask (Tensor): A 2D tensor representing the attention mask for the input data. 
            Each element indicates whether the corresponding position in the input sequence is a valid token (1)
            or a padding token (0).
    
    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A 1D tensor of indices indicating the positions of valid tokens in the
            flattened attention_mask.
            - cu_seqlens (Tensor): A 1D tensor of cumulative sequence lengths up to each batch element.
            - max_seqlen_in_batch (int): The maximum sequence length among the batch elements.

    Raises:
        ValueError: If the attention_mask is not a valid tensor or if there are issues with calculating
            the required data.
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


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Phi
class PhiRotaryEmbedding(nn.Cell):

    """
    The PhiRotaryEmbedding class represents a rotational positional embedding for neural network models.
    It inherits from nn.Cell and provides functionality for constructing rotational embeddings based on
    input sequences and sequence lengths.

    Attributes:
        dim (int): The dimension of the rotational positional embedding.
        max_position_embeddings (int): The maximum position embeddings allowed.
        base (int): The base value used in the rotational embedding calculation.
        inv_freq (Tensor): The inverse frequency used in the rotational embedding calculation.
        max_seq_len_cached (int): The maximum sequence length for which the cosine and sine cache is precomputed.
        cos_cached (Tensor): Precomputed cosine values for positional embeddings.
        sin_cached (Tensor): Precomputed sine values for positional embeddings.

    Methods:
        _set_cos_sin_cache:
            Precomputes and caches cosine and sine values for positional embeddings based on the specified sequence
            length and data type.

        construct:
            Constructs the rotational positional embedding for the input sequence based on the specified sequence
            length or the maximum cached sequence length.

    Note:
        This docstring is based on the provided code snippet and may need additional details or context to fully
        describe the class and its functionality.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes an instance of the PhiRotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimensionality of the embeddings.
            max_position_embeddings (int, optional): The maximum number of position embeddings to generate. 
                Default is 2048.
            base (int, optional): The base value used in the calculation. Default is 10000.

        Returns:
            None.

        Raises:
            ValueError: If dim is not a positive integer.
            ValueError: If max_position_embeddings is not a positive integer.
            ValueError: If base is not a positive integer.
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        """Sets the cosine and sine cache for PhiRotaryEmbedding.

        This method sets the cosine and sine cache for PhiRotaryEmbedding based on the given sequence 
        length and data type.

        Args:
            self (PhiRotaryEmbedding): The PhiRotaryEmbedding instance.
            seq_len (int): The length of the sequence.
            dtype (torch.dtype): The desired data type for the cache.

        Returns:
            None.

        Raises:
            None.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def construct(self, x, seq_len=None):
        """
        Constructs a PhiRotaryEmbedding.

        Args:
            self (PhiRotaryEmbedding): The instance of the PhiRotaryEmbedding class.
            x: The input tensor.
            seq_len (int, optional): The length of the sequence. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If `seq_len` is greater than `max_seq_len_cached`.

        This method constructs a PhiRotaryEmbedding by calculating and returning the cosine and sine cached values 
        based on the input tensor `x` and the provided sequence length `seq_len`. If `seq_len` is not specified, 
        the method returns the cosine and sine cached values for the entire sequence. 
        The returned values are converted to the same data type as `x`.

        If the specified `seq_len` is greater than the `max_seq_len_cached` value, the method internally updates the 
        cached values by calling the `_set_cos_sin_cache` method. This method should be called before accessing the 
        cached values to ensure they are up to date.

        Note that this method does not modify the instance's state and only returns the calculated cached values.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->Phi
class PhiLinearScalingRotaryEmbedding(PhiRotaryEmbedding):
    """PhiRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes the PhiLinearScalingRotaryEmbedding object.

        Args:
            self: The object itself.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value for calculations. Defaults to 10000.
            scaling_factor (float, optional): The scaling factor applied to the embeddings. Defaults to 1.0.

        Returns:
            None.

        Raises:
            None.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        """Sets the cosine and sine cache for the PhiLinearScalingRotaryEmbedding layer.

        Args:
            self: The PhiLinearScalingRotaryEmbedding instance.
            seq_len (int): The maximum sequence length to cache.
            dtype: The data type of the cache.

        Returns:
            None.

        Raises:
            None.

        This method sets the cosine and sine cache for the PhiLinearScalingRotaryEmbedding layer.
        It creates an array of range values from 0 to the maximum sequence length and divides it by the scaling factor.
        It then creates an array of frequencies by taking the outer product of the range values and the inverse
        frequency values. The cosine and sine of the frequencies are then computed and stored in the cache. The
        maximum sequence length cached is stored in the instance variable max_seq_len_cached."""
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->Phi
class PhiDynamicNTKScalingRotaryEmbedding(PhiRotaryEmbedding):
    """PhiRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        """
        Initializes an instance of PhiDynamicNTKScalingRotaryEmbedding.

        Args:
            self: The instance of the class.
            dim (int): The dimensionality of the embedding.
            max_position_embeddings (int): The maximum number of position embeddings.
            base (int): The base value used in calculations.
            scaling_factor (float): The scaling factor applied to the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        '''
        _set_cos_sin_cache method in the PhiDynamicNTKScalingRotaryEmbedding class.

        This method is used to set the cosine and sine cache for the rotary position embeddings based on the
        given sequence length and data type.

        Args:
            self (PhiDynamicNTKScalingRotaryEmbedding): The instance of the PhiDynamicNTKScalingRotaryEmbedding class.
            seq_len (int): The length of the sequence for which the cosine and sine cache is to be set.
            dtype: The data type for the cache values.

        Returns:
            None.

        Raises:
            ValueError: If the sequence length is less than or equal to 0.
            TypeError: If the input data type is not compatible with the operations performed within the method.
        '''
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2).float() / self.dim))
            self.inv_freq = inv_freq

        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.tensor_split(2, -1)
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`mindspore.Tensor`): The query tensor.
        k (`mindspore.Tensor`): The key tensor.
        cos (`mindspore.Tensor`): The cosine part of the rotary embedding.
        sin (`mindspore.Tensor`): The sine part of the rotary embedding.
        position_ids (`mindspore.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Phi
class PhiMLP(nn.Cell):

    """
    PhiMLP represents a Multi-Layer Perceptron (MLP) neural network with configurable hidden layer sizes and
    activation functions.

    This class inherits from nn.Cell and implements the forward pass of the MLP by defining the layers and
    activation functions.

    Attributes:
        config (object): A configuration object that specifies the MLP architecture parameters.
        activation_fn (function): The activation function used in the hidden layers of the MLP.
        fc1 (nn.Dense): The first fully connected layer of the MLP.
        fc2 (nn.Dense): The second fully connected layer of the MLP.

    Methods:
        __init__:
            Initializes the PhiMLP instance with the provided configuration.

        construct:
            Constructs the forward pass of the MLP using the provided input tensor.

    Returns:
        mindspore.Tensor: The output tensor of the forward pass through the MLP.
    """
    def __init__(self, config):
        """
        Initializes an instance of the PhiMLP class.

        Args:
            self: The instance of the class.
            config (object):
                An object containing configuration parameters for the PhiMLP model.

                - Type: Custom object
                - Purpose: Stores various configuration parameters for the model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the forward pass of the PhiMLP model.

        Args:
            self (PhiMLP): The instance of the PhiMLP class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed.
                The shape of the hidden_states tensor should be compatible with the model's architecture.

        Returns:
            mindspore.Tensor: The tensor resulting from the forward pass through the PhiMLP model.

        Raises:
            TypeError: If the input hidden_states is not of type mindspore.Tensor.
            ValueError: If the shape of the hidden_states tensor is incompatible with the model's architecture.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


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


class PhiAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: PhiConfig, layer_idx: Optional[int] = None):
        """
        Initializes an instance of the PhiAttention class.

        Args:
            self: The instance of the class.
            config (PhiConfig): An instance of the PhiConfig class containing configuration parameters.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the hidden_size is not divisible by num_heads.
            TypeError: If config is not an instance of PhiConfig.
            TypeError: If layer_idx is not an integer or None.
            Warning: If layer_idx is None, it is not recommended and may lead to errors during forward call
                if caching is used.

        Note:
            This method initializes the PhiAttention class with the given configuration and layer index.
            It sets the various properties and performs necessary checks.

        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=True)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.dense = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=True)

        self.qk_layernorm = config.qk_layernorm
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(
                [config.hidden_size // self.num_heads], epsilon=config.layer_norm_eps, elementwise_affine=True
            )
            self.k_layernorm = nn.LayerNorm(
                [config.hidden_size // self.num_heads], epsilon=config.layer_norm_eps, elementwise_affine=True
            )

        self._init_rope()

    def _init_rope(self):
        """
        Initializes the RoPE (Rotary Position Embedding) for PhiAttention.

        Args:
            self: The instance of the PhiAttention class.

        Returns:
            None: This method modifies the rotary_emb attribute of the instance.

        Raises:
            ValueError: If the RoPE scaling type is unknown.

        The method initializes the RoPE based on the configuration provided. If the rope_scaling is not specified,
        the method initializes a PhiRotaryEmbedding object with the given partial_rotary_factor and
        max_position_embeddings.

        If rope_scaling is specified, the method checks the scaling_type. If the scaling_type is 'linear',
        it initializes a PhiLinearScalingRotaryEmbedding object with the given partial_rotary_factor,
        max_position_embeddings, scaling_factor, and base. If the scaling_type is 'dynamic',
        it initializes a PhiDynamicNTKScalingRotaryEmbedding object with the given partial_rotary_factor,
        max_position_embeddings, scaling_factor, and base.

        Note:
            RoPE stands for Rotary Position Embedding and is used to incorporate positional information in the
            attention mechanism.

        """
        if self.config.rope_scaling is None:
            self.rotary_emb = PhiRotaryEmbedding(
                int(self.partial_rotary_factor * self.head_dim),
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = PhiLinearScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = PhiDynamicNTKScalingRotaryEmbedding(
                    int(self.partial_rotary_factor * self.head_dim),
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
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
        '''
        This method, named 'construct', is defined in the class 'PhiAttention'.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states with shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor with shape
                (batch_size, 1, sequence_length, sequence_length) to mask the attention scores.
            position_ids (Optional[mindspore.Tensor]): An optional tensor representing the position indices of
                input tokens with shape (batch_size, sequence_length).
            past_key_value (Optional[Cache]): An optional cache structure for storing previous key and value states
                during auto-regressive decoding.
            output_attentions (bool): A boolean flag indicating whether to return the attention weights.
            use_cache (bool): A boolean flag indicating whether to use caching for key and value states.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output tensor with shape (batch_size, sequence_length, hidden_size),
                optional attention weights tensor, and optional updated cache structure.

        Raises:
            ValueError: If the cache structure has changed since version v4.36 and the layer index is not initialized
                when using the cache for auto-regressive decoding.
            ValueError: If the shape of attention weights does not match
                (batch_size, num_heads, sequence_length, sequence_length).
            ValueError: If the shape of attention mask does not match (batch_size, 1, sequence_length, sequence_length).
            ValueError: If the shape of attn_output does not match (batch_size, num_heads, sequence_length, hidden_size).
        '''
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

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
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = ops.cat((query_rot, query_pass), axis=-1)
        key_states = ops.cat((key_rot, key_pass), axis=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        attn_weights = ops.matmul(
            query_states.to(mindspore.float32), key_states.to(mindspore.float32).swapaxes(2, 3)
        ) / math.sqrt(self.head_dim)
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

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


PHI_ATTENTION_CLASSES = {
    "eager": PhiAttention,
}


class PhiDecoderLayer(nn.Cell):

    """
    PhiDecoderLayer represents a single layer of the Phi decoder model.

    This class inherits from nn.Cell and contains methods for initializing the layer and constructing the
    layer's computations.

    The __init__ method initializes the PhiDecoderLayer with the provided configuration and layer index.
    It sets up the self-attention mechanism, multi-layer perceptron, layer normalization, and residual dropout.

    The construct method takes hidden_states as input and applies layer normalization. It then computes the
    self-attention outputs, optionally returning attention weights and caching key-value states. The method also
    computes the feed-forward hidden states and returns the final layer outputs, optionally including attention weights
    and key-value states in the output tuple.
    """
    def __init__(self, config: PhiConfig, layer_idx: int):
        """
        This method initializes a PhiDecoderLayer object.

        Args:
            self (PhiDecoderLayer): The current instance of PhiDecoderLayer.
            config (PhiConfig): An object containing configuration settings for the PhiDecoderLayer.
            layer_idx (int): An integer representing the index of the layer within the PhiDecoderLayer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type PhiConfig.
            ValueError: If the layer_idx parameter is not an integer.
        """
        super().__init__()
        self.self_attn = PHI_ATTENTION_CLASSES["eager"](config, layer_idx=layer_idx)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
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
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PhiPreTrainedModel(PreTrainedModel):

    """
    This class represents a PhiPreTrainedModel, which is a subclass of PreTrainedModel.
    It is designed for pre-training models using the Phi framework.

    The class includes a method called _init_weights which initializes the weights of the model's cells.
    The method takes a cell object as an argument and sets the weights and biases for the cell based on the
    configuration settings.

    If the cell is an instance of nn.Dense, the method sets the weight data using the initializer function with a
    normal distribution and the specified standard deviation. It also sets the bias data to zeros if the cell has a bias.

    If the cell is an instance of nn.Embedding, the method generates random weight values from a normal distribution
    with a mean of 0 and the specified standard deviation. If the cell has a padding index, the weight value at that
    index is set to 0. The weight data is then set for the cell.

    Note:
        This docstring does not include signatures or any other code. Please refer to the actual code implementation
        for more details.
    """
    config_class = PhiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["PhiDecoderLayer"]
    _supports_cache_class = True

    def _init_weights(self, cell):
        """
        Initializes the weights and biases of a neural network cell based on the specified configuration.

        Args:
            self (PhiPreTrainedModel): The instance of the PhiPreTrainedModel class.
            cell (nn.Module): The neural network cell for which the weights and biases are initialized.

        Returns:
            None.

        Raises:
            ValueError: If the cell type is neither nn.Dense nor nn.Embedding.
            TypeError: If the cell type is not recognized or if there are issues with setting the data for
                weights and biases.
            IndexError: If there are issues with indexing while setting the weight values.
        """
        std = self.config.initializer_range
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

class PhiModel(PhiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """
    def __init__(self, config: PhiConfig):
        """
        Initializes an instance of the PhiModel class.

        Args:
            self: The instance of the PhiModel class.
            config (PhiConfig):
                The configuration object containing the model's hyperparameters and settings.

                `config` is of type PhiConfig.
                It specifies the configuration for the PhiModel.
                This object is used to set various attributes of the PhiModel instance.
                The attributes set are:

                - `padding_idx`: The index to use for padding in the input sequence.
                It is initialized with the value of `config.pad_token_id`.
                - `vocab_size`: The size of the vocabulary, i.e., the total number of unique tokens.
                It is initialized with the value of `config.vocab_size`.
                - `embed_tokens`: The embedding layer for the input tokens.
                It is an instance of `nn.Embedding` and is initialized with the values:

                    - `config.vocab_size`: The size of the vocabulary.
                    - `config.hidden_size`: The dimensionality of the hidden state.
                    - `self.padding_idx`: The index used for padding.

                - `embed_dropout`: The dropout layer applied to the input embeddings.
                It is an instance of `nn.Dropout` and is initialized with the dropout probability `config.embd_pdrop`.
                - `layers`: A list of PhiDecoderLayer instances representing the decoder layers of the model.
                It is initialized as a `nn.CellList` containing `config.num_hidden_layers` PhiDecoderLayer instances.
                Each PhiDecoderLayer instance is created using the `PhiDecoderLayer` constructor with `config` and `layer_idx`.
                - `final_layernorm`: The layer normalization applied to the final hidden state.
                It is an instance of `nn.LayerNorm` and is initialized with the following attributes:

                    - `[config.hidden_size]`: The normalized shape of the input tensor.
                    - `epsilon=config.layer_norm_eps`: The epsilon value added to the denominator for numerical stability.

                - `gradient_checkpointing`: A boolean flag indicating whether gradient checkpointing is enabled.
                It is initialized as `False`.

        Raises:
            None.

        Returns:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(p=config.embd_pdrop)
        self.layers = nn.CellList(
            [PhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the PhiModel.

        Args:
            self (PhiModel): The instance of the PhiModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the PhiModel.

        Args:
            self (PhiModel): The instance of the PhiModel class.
            value: The input embeddings to be set for the PhiModel.
                It should be a tensor or an object that can be assigned to self.embed_tokens.

        Returns:
            None.

        Raises:
            TypeError: If the provided value is not compatible with the expected input embeddings format.
            ValueError: If the provided value is empty or invalid.
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
        This method constructs the PhiModel using the specified input parameters and returns the output as a tuple or
        a BaseModelOutputWithPast object.

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor, optional): The input tensor containing the token ids for the model input.
                Defaults to None.
            attention_mask (mindspore.Tensor, optional): An optional tensor providing the attention mask for the input.
                Defaults to None.
            position_ids (mindspore.Tensor, optional): An optional tensor representing the position ids for the input.
                Defaults to None.
            past_key_values (List[mindspore.Tensor], optional): An optional list of tensors containing the past key values.
                Defaults to None.
            inputs_embeds (mindspore.Tensor, optional): An optional tensor representing the embedded inputs.
                Defaults to None.
            use_cache (bool, optional): An optional boolean flag indicating whether to use caching. Defaults to None.
            output_attentions (bool, optional): An optional boolean flag indicating whether to output attentions.
                Defaults to None.
            output_hidden_states (bool, optional): An optional boolean flag indicating whether to output hidden states.
                Defaults to None.
            return_dict (bool, optional): An optional boolean flag indicating whether to return a dictionary.
                Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The output is either a tuple containing the hidden states,
                next_cache, all_hidden_states, and all_self_attns or a BaseModelOutputWithPast object containing
                the last hidden state, past key values, hidden states, and attentions.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified simultaneously or if neither input_ids
                nor inputs_embeds are specified.
            Warning: If `use_cache=True` is incompatible with gradient checkpointing, a warning is raised to indicate
                that `use_cache` will be set to False.
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
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.embed_dropout(inputs_embeds)

        # Attention mask.
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

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

        hidden_states = self.final_layernorm(hidden_states)

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


class PhiForCausalLM(PhiPreTrainedModel):

    """
    The `PhiForCausalLM` class represents a Phi model for causal language modeling. It inherits from `PhiPreTrainedModel`
    and provides methods for initializing the model, getting and setting input and output embeddings, setting the
    decoder, constructing the model, preparing inputs for generation, and reordering cache.
    The `PhiForCausalLM` class also includes detailed type annotations and example usage.

    The class includes the following methods:

    - `__init__`: Initializes the PhiForCausalLM model with the provided configuration.
    - `get_input_embeddings`: Returns the input embeddings of the model.
    - `set_input_embeddings`: Sets the input embeddings of the model to the provided value.
    - `get_output_embeddings`: Returns the output embeddings of the model.
    - `set_output_embeddings`: Sets the output embeddings of the model to the provided new_embeddings.
    - `set_decoder`: Sets the decoder of the model to the provided decoder.
    - `get_decoder`: Returns the decoder of the model.
    - `construct`: Constructs the model for causal language modeling with the specified inputs and returns the outputs.
    - `prepare_inputs_for_generation`: Prepares the inputs for generation based on the provided input_ids,
    past_key_values, attention_mask, and inputs_embeds.
    - `_reorder_cache`: Reorders the past_key_values based on the specified beam index.

    The class docstring includes detailed descriptions of the methods, their arguments, and return values, as well as
    an example usage demonstrating how to use the `PhiForCausalLM` class for generating text using the model.

    """
    _tied_weights_keys = ["lm_head.weight"]

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi,bias=False->has_bias=True
    def __init__(self, config):
        """
        Initializes an instance of the 'PhiForCausalLM' class.

        Args:
            self: The instance of the class.
            config (object):
                The configuration object containing the necessary parameters for the Phi model.

                - config.vocab_size (int): The size of the vocabulary.
                - config.hidden_size (int): The size of the hidden state of the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = PhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=True)
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the PhiForCausalLM model.

        Args:
            self (PhiForCausalLM): An instance of the PhiForCausalLM class.
                Represents the current object instance.

        Returns:
            embed_tokens: This method returns the input embeddings as obtained from the model's embed_tokens attribute.

        Raises:
            None.
        """
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the PhiForCausalLM model.

        Args:
            self (PhiForCausalLM): The instance of the PhiForCausalLM class.
            value (Tensor): The input embeddings to be set for the model.
                It should be a tensor of appropriate shape and type.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings for the PhiForCausalLM model.

        Args:
            self: An instance of the PhiForCausalLM class.

        Returns:
            None: The method returns the output embeddings for the PhiForCausalLM model.
                These embeddings are used to map the output tokens to a continuous representation.

        Raises:
            None.
        """
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the PhiForCausalLM model.

        Args:
            self (PhiForCausalLM): The instance of the PhiForCausalLM class.
            new_embeddings: The new embeddings to be set as the model's output embeddings.
                It should be a tensor of shape (vocab_size, hidden_size) where 'vocab_size'
                represents the size of the vocabulary and 'hidden_size' represents the size
                of the hidden layer. The new embeddings should be compatible with the model's
                existing architecture.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        """
        Args:
            self (PhiForCausalLM): The instance of the PhiForCausalLM class.
            decoder: The decoder object to be set for the model. It should be an instance of the decoder class.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        """
        Returns the decoder model used for PhiForCausalLM.

        Args:
            self: An instance of the PhiForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the decoder model that is used for PhiForCausalLM. The decoder model is an essential component
        of the PhiForCausalLM class and is responsible for generating output based on the input data. The decoder model
        contains the learned weights and biases that enable the PhiForCausalLM class to perform its tasks effectively.
        The returned decoder model is of type 'None' as it is used internally within the PhiForCausalLM class and is not
        intended to be directly accessed or modified by the user.
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
            >>> from transformers import AutoTokenizer, PhiForCausalLM
            ...
            >>> model = PhiForCausalLM.from_pretrained("microsoft/phi-1")
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
            ...
            >>> prompt = "This is an example script ."
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            'This is an example script .\n\n\n\nfrom typing import List\n\ndef find_most_common_letter(words: List[str'
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

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepares inputs for generating output sequences using PhiForCausalLM model.

        Args:
            self (PhiForCausalLM): An instance of PhiForCausalLM class.
            input_ids (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing input sequence tokens.
            past_key_values (Cache or tuple or None): A cache object or tuple of two tensors containing previously
                computed key and value pairs for the attention mechanism. If None, no caching is performed.
            attention_mask (torch.Tensor or None): An optional tensor of shape (batch_size, sequence_length)
                containing a mask to avoid performing attention on padding tokens.
            inputs_embeds (torch.Tensor or None): An optional tensor of shape (batch_size, sequence_length, hidden_size)
                containing precomputed embeddings for the input sequence.

        Returns:
            model_inputs (dict):
                A dictionary containing the following keys:

                - 'input_ids': The input sequence tokens tensor.
                - 'position_ids': The tensor of positional encoding for the input sequence.
                - 'past_key_values': The cache object or tuple of two tensors containing previously computed key and
                value pairs for the attention mechanism.
                - 'use_cache': A boolean indicating whether to use caching.
                - 'attention_mask': The tensor containing the attention mask for the input sequence.

        Raises:
            None.
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
        Reorders the cache of past key values based on the given beam index.

        Args:
            past_key_values (tuple): A tuple containing past key values for each layer. Each element in the tuple is
                a tensor representing the past key values.
            beam_idx (Tensor): A tensor containing the indices of the beams to reorder the cache based on.

        Returns:
            None: The method modifies the cache in place.

        Raises:
            ValueError: If the input past_key_values or beam_idx are not in the expected format.
            IndexError: If the beam index is out of bounds.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with LLAMA->PHI,Llama->Phi with self.transformer->self.model, transformer_outputs->model_outputs
class PhiForSequenceClassification(PhiPreTrainedModel):

    """PhiForSequenceClassification

    This class is a sequence classification model that uses the PHI algorithm for natural language processing tasks.
    It inherits from the PhiPreTrainedModel class.

    Attributes:
        config (PhiConfig): The model configuration class instance.
        num_labels (int): The number of labels for the classification task.
        model (PhiModel): The PHI model for token embeddings.
        score (nn.Dense): The dense layer for scoring hidden states.

    Methods:
        __init__: Initializes a new PhiForSequenceClassification instance.
        get_input_embeddings: Retrieves the input embeddings from the model.
        set_input_embeddings: Sets the input embeddings for the model.
        construct: Constructs the model for sequence classification.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the PhiForSequenceClassification class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the model.

                - Type: object
                - Purpose: Configuration object specifying the model's settings.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PhiModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the PhiForSequenceClassification model.

        Args:
            self (PhiForSequenceClassification): The instance of the PhiForSequenceClassification class.

        Returns:
            None: This method does not return any value.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the PhiForSequenceClassification model.

        Args:
            self (PhiForSequenceClassification): The instance of the PhiForSequenceClassification class.
            value (Tensor): The new input embeddings tensor to be set for the model.

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


# Copied from transformers.models.mpt.modeling_mpt.MptForTokenClassification with MPT->PHI,Mpt->Phi,self.transformer->self.model,transformer_outputs->model_outputs
class PhiForTokenClassification(PhiPreTrainedModel):

    """
    This class represents a PhiForTokenClassification model, which is used for token classification tasks such as
    Named Entity Recognition (NER) or Part-of-Speech (POS) tagging. It is a subclass of the PhiPreTrainedModel.

    The PhiForTokenClassification class initializes with a PhiConfig object, which contains the configuration parameters
    for the model. It sets the number of labels for the classification task and creates an instance of the PhiModel
    based on the provided configuration.

    The class also handles the initialization of the classifier dropout, which can be set either through the
    'classifier_dropout' parameter in the config or the 'hidden_dropout' parameter. If neither is provided, a default
    dropout rate of 0.1 is used.

    The 'construct' method is used to perform the forward pass of the model. It takes several input tensors such as
    'input_ids', 'past_key_values', 'attention_mask', 'inputs_embeds', and 'labels'. It also supports various optional
    arguments such as 'use_cache', 'output_attentions', 'output_hidden_states', and 'return_dict'.

    The 'labels' tensor is optional and represents the ground truth labels for computing the sequence
    classification/regression loss. The indices in 'labels' should be in the range of [0, config.num_labels - 1].
    If 'config.num_labels == 1', a regression loss (Mean-Square loss) is computed. If 'config.num_labels > 1',
    a classification loss (Cross-Entropy) is computed.

    The 'construct' method returns either a tuple of logits and other model outputs or a TokenClassifierOutput object
    depending on the 'return_dict' parameter. If 'labels' are provided, the method also computes the loss using the
    logits and the ground truth labels.

    Please note that the class inherits additional functionality and attributes from the PhiPreTrainedModel superclass.

    """
    def __init__(self, config: PhiConfig):
        """
        Initializes a new instance of the PhiForTokenClassification class.

        Args:
            self: The object itself.
            config (PhiConfig): The configuration object for PhiForTokenClassification.
                This object contains various parameters for configuring the model.
                The config parameter is required and cannot be None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = PhiModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(p=classifier_dropout)
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
    'PHI_PRETRAINED_MODEL_ARCHIVE_LIST',
    'PhiForTokenClassification',
    'PhiForSequenceClassification',
    'PhiForCausalLM',
    'PhiModel',
    'PhiPreTrainedModel',
]
