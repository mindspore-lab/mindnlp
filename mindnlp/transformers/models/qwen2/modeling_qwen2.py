# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" MindSpore Qwen2 model."""
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging, get_default_dtype
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from .configuration_qwen2 import Qwen2Config


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"

QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen2-7B-beta",
    # See all Qwen2 models at https://hf-mirror.com/models?filter=qwen2
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """

    Args:
        attention_mask (Tensor): A tensor representing the attention mask for the input sequences.
            Its purpose is to indicate which tokens in the input sequences should be attended to and which should be ignored.
            It should be a 2D tensor with a shape of (batch_size, sequence_length) and contain binary values (0 or 1).
    
    Returns:
        Tuple of Tensors:
            The function returns a tuple containing the following:

            - indices (Tensor): A 1D tensor containing the indices of the non-zero elements in the flattened
            attention_mask tensor.
            - cu_seqlens (Tensor): A 1D tensor representing the cumulative sum of the sequence lengths in the batch,
            padded with a zero at the beginning.
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

    Raises:
        None
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


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Cell):

    """
    Qwen2RMSNorm is a custom normalization layer that inherits from nn.Cell. It is equivalent to T5LayerNorm and is
    designed to normalize the input hidden states.

    This class initializes with the specified hidden_size and an optional epsilon value for variance smoothing.
    The normalization process involves scaling the hidden states based on the calculated variance and the provided
    weight parameter.

    The construct method takes hidden_states as input and performs the normalization operation, ensuring that the
    output matches the input data type. The normalized hidden_states are then multiplied by the weight parameter to
    produce the final output.

    Note:
        This docstring is based on the provided information and does not include actual code or signatures.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        Constructs the RMS normalization of hidden states.

        Args:
            self (Qwen2RMSNorm): The instance of the Qwen2RMSNorm class.
            hidden_states (Tensor): The input hidden states to be normalized.
                Should be a tensor of any shape with dtype compatible with float32.

        Returns:
            None: The method modifies the hidden_states tensor in-place.

        Raises:
            ValueError: If hidden_states is not a valid tensor.
            TypeError: If hidden_states dtype is not compatible with float32.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Qwen2
class Qwen2RotaryEmbedding(nn.Cell):

    """
    Represents a Qwen2RotaryEmbedding module that inherits from nn.Cell. This module implements the Qwen2Rotary
    embedding as described in the code.

    Attributes:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int): The maximum position embeddings.
        base (int): The base value used in the embedding calculation.

    Methods:
        _set_cos_sin_cache: Sets the cosine and sine cache for the given sequence length and data type.
        construct(: Constructs the Qwen2Rotary embedding for the input with optional sequence length.

    Note:
        The Qwen2RotaryEmbedding module provides functionality for Qwen2Rotary embedding calculation, including setting
        cosine and sine cache and constructing the embedding.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes a new instance of the Qwen2RotaryEmbedding class.

        Args:
            self: The object itself.
            dim (int): The dimensionality of the embedding vectors.
            max_position_embeddings (int, optional): The maximum number of position embeddings to generate.
                Defaults to 2048.
            base (int, optional): The base value used in the calculation of inverse frequency. Defaults to 10000.

        Returns:
            None.

        Raises:
            None.

        This method initializes the Qwen2RotaryEmbedding object with the specified dimensionality,
        maximum position embeddings, and base value. It calculates the inverse frequency based on the dimensionality
        and stores it in the 'inv_freq' attribute. Additionally, it sets the cosine and sine cache based on the
        maximum position embeddings.
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim))
        self.inv_freq = inv_freq

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        """
        Sets the cosine and sine cache for the Qwen2RotaryEmbedding class.

        Args:
            self (Qwen2RotaryEmbedding): The instance of the Qwen2RotaryEmbedding class.
            seq_len (int): The length of the sequence.
            dtype (dtype): The desired data type for the cache.

        Returns:
            None: This method updates the 'cos_cached' and 'sin_cached' attributes of the Qwen2RotaryEmbedding instance.

        Raises:
            None.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(self.inv_freq)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def construct(self, x, seq_len=None):
        """
        Constructs the Qwen2RotaryEmbedding for the given input tensor 'x' and sequence length 'seq_len'.

        Args:
            self: The instance of the Qwen2RotaryEmbedding class.
            x: A tensor representing the input data.
            seq_len: An optional integer representing the length of the sequence. Defaults to None.

        Returns:
            None: This method modifies the internal state of the Qwen2RotaryEmbedding instance.

        Raises:
            ValueError: If 'seq_len' is not a positive integer.
            TypeError: If the data type of 'x' is not supported for the internal calculations.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = x.tensor_split(2, -1)
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.

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


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Cell):

    """
    Qwen2MLP is a Python class that represents a multi-layer perceptron (MLP) with specific configurations for gate, up,
    and down projections. This class inherits from nn.Cell and is designed to be used in neural network models for
    deep learning applications.

    Attributes:
        config: A configuration object containing settings for the hidden size and intermediate size of the MLP.
        hidden_size: An integer representing the size of the hidden layer in the MLP.
        intermediate_size: An integer representing the size of the intermediate layer in the MLP.
        gate_proj: An instance of nn.Dense for projecting input data to the intermediate size with no bias.
        up_proj: An instance of nn.Dense for projecting input data to the intermediate size with no bias.
        down_proj: An instance of nn.Dense for projecting data from the intermediate size back to the hidden size with no bias.
        act_fn: An activation function determined by the configuration settings.

    Methods:
        construct: A method that takes input data x and performs the forward pass through the MLP using the
            defined projections and activation function.

    Note:
        The Qwen2MLP class is intended to be used as part of a larger neural network model and provides a configurable
        multi-layer perceptron with specific projection and activation settings.
    """
    def __init__(self, config):
        """
        Initializes an instance of the Qwen2MLP class.

        Args:
            self: The instance of the class.
            config: An object containing configuration parameters for the MLP.
                It should have the following attributes:

                - hidden_size: An integer specifying the size of the hidden layer.
                - intermediate_size: An integer specifying the size of the intermediate layer.
                - hidden_act: A string specifying the activation function to be used in the hidden layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        """
        Constructs a new object using the Qwen2MLP class.

        Args:
            self: An instance of the Qwen2MLP class.
            x: The input parameter of type 'Any', representing the data to be processed.

        Returns:
            This method returns None.

        Raises:
            None.

        This method constructs a new object by performing a series of operations on the input data 'x'.
        It first applies the 'gate_proj' function to 'x' and then applies the 'act_fn' function to the result.
        The output of 'act_fn' is multiplied element-wise with the result of applying the 'down_proj' function to 'x'.
        Finally, the result is multiplied with the output of applying the 'up_proj' function to 'x'.
        The constructed object is returned as None.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


class Qwen2Attention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        """
        Initializes an instance of the Qwen2Attention class.

        Args:
            self: The instance of the class.
            config (Qwen2Config): An instance of the Qwen2Config class containing configuration parameters for
                the attention mechanism.
            layer_idx (Optional[int]): The index of the layer. Defaults to None. If None, a warning is logged
                as it may lead to errors during forward call if caching is used. It is recommended to provide a
                valid layer index when creating the class.

        Returns:
            None.

        Raises:
            ValueError: If the `hidden_size` is not divisible by `num_heads`.
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

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=True)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        '''
        This method constructs the Qwen2Attention layer.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, 1, sequence_length, key_value_sequence_length) containing indices to be masked.
            position_ids (Optional[mindspore.Tensor]): An optional tensor of shape (batch_size, sequence_length)
                containing the position indices of each token in the input sequence.
            past_key_value (Optional[Cache]): An optional object representing the cached key and value tensors
                from previous time steps.
            output_attentions (bool): A flag indicating whether to return the attention weights.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing:

                - attn_output (mindspore.Tensor): The output tensor of shape (batch_size, sequence_length, hidden_size).
                - attn_weights (Optional[mindspore.Tensor]): The attention weights tensor of shape
                (batch_size, num_heads, sequence_length, key_value_sequence_length),
                if output_attentions is True, else None.
                - past_key_value (Optional[Tuple[mindspore.Tensor]]): The updated key and value tensors,
                if past_key_value is not None and caching is enabled, else None.

        Raises:
            ValueError: If the cache structure has changed and the layer index is not provided,
                if the shape of attention weights or attention mask is incorrect, or if the shape of the
                output tensor is not as expected.
        '''
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

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
        attn_weights = ops.softmax(attn_weights, axis=-1, dtype=mindspore.float32).to(query_states.dtype)
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


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
}


class Qwen2DecoderLayer(nn.Cell):

    """
    Qwen2DecoderLayer is a class representing a single layer of the Qwen2 decoder. It inherits from nn.Cell and
    contains methods for initializing the layer and constructing the layer's operations.

    Attributes:
        hidden_size (int): The size of the hidden state.
        self_attn (QWEN2_ATTENTION_CLASSES): The self-attention mechanism used in the layer.
        mlp (Qwen2MLP): The multi-layer perceptron used in the layer.
        input_layernorm (Qwen2RMSNorm): The layer normalization applied to the input.
        post_attention_layernorm (Qwen2RMSNorm): The layer normalization applied after the attention mechanism.

    Methods:
        __init__: Initializes the Qwen2DecoderLayer with the given configuration and layer index.
        construct:
            Applies the layer operations to the input hidden_states and returns the resulting output tensor along with
            optional additional tensors, such as attention weights and present key value.

    Args:
        hidden_states (mindspore.Tensor): Input to the layer of shape (batch, seq_len, embed_dim).
        attention_mask (mindspore.Tensor, optional): Attention mask of size (batch, sequence_length)
            where padding elements are indicated by 0.
        output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
        use_cache (bool, optional): If set to True, past_key_values key value states are returned and can be used to
            speed up decoding.
        past_key_value (Tuple(mindspore.Tensor), optional): Cached past key and value projection states.

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]: The output tensor and optional
            additional tensors based on the input arguments.
    """
    def __init__(self, config: Qwen2Config, layer_idx: int):
        """
        Initializes a Qwen2DecoderLayer object.

        Args:
            self (Qwen2DecoderLayer): The instance of the Qwen2DecoderLayer class.
            config (Qwen2Config): An object containing configuration settings for the decoder layer.
            layer_idx (int): An integer representing the index of the layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QWEN2_ATTENTION_CLASSES["eager"](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
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
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2PreTrainedModel(PreTrainedModel):

    """
    This class represents a Qwen2PreTrainedModel, which is a subclass of PreTrainedModel.
    It provides methods for initializing the weights of the model's cells.

    Methods:
        _init_weights:
            Initializes the weights of a given cell.

            Parameters:

            - cell: The cell to initialize the weights for.

            Returns:
                None

    Details:
        The _init_weights method initializes the weights of the specified cell. It first checks the type of the cell.
        If it is of type nn.Dense, it sets the weight data using the initializer function.
        The initializer function takes the following parameters:

        - Normal(self.config.initializer_range): A normal distribution initializer with the specified range.
        - cell.weight.shape: The shape of the weight tensor.
        - cell.weight.dtype: The data type of the weight tensor.

        If the cell has a bias, it also sets the bias data using the initializer function with the following parameters:

        - 'zeros': A zero initializer.
        - cell.bias.shape: The shape of the bias tensor.
        - cell.bias.dtype: The data type of the bias tensor.

        If the cell is of type nn.Embedding, it generates random weights using the numpy random.normal function.
        The parameters for the random.normal function are:

        - 0.0: The mean of the normal distribution.
        - self.config.initializer_range: The standard deviation of the normal distribution.
        - cell.weight.shape: The shape of the weight tensor.

        If the cell has a padding_idx, it sets the value at that index to 0.

        Finally, the initialized weights are set to the cell using the Tensor function with the following parameters:

        - weight: The initialized weight tensor.
        - cell.weight.dtype: The data type of the weight tensor.
    """
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """
    def __init__(self, config: Qwen2Config):
        """
        Initializes a Qwen2Model instance.

        Args:
            self (Qwen2Model): The instance of the Qwen2Model class.
            config (Qwen2Config):
                An instance of Qwen2Config containing configuration parameters for the model.
                It specifies the model configuration including the vocabulary size, hidden size, number of
                hidden layers, padding token id, and RMS normalization epsilon.

                The config object should have the following attributes:

                - pad_token_id (int): The token id for padding.
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - num_hidden_layers (int): The number of hidden layers in the model.
                - rms_norm_eps (float): Epsilon value for RMS normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the Qwen2Model class.

        Args:
            self: An instance of the Qwen2Model class.
                This parameter refers to the current instance of the Qwen2Model class.
                It is used to access the embed tokens for input embeddings.

        Returns:
            None:
                This method returns None as it simply provides access to the input embeddings.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the Qwen2Model.

        Args:
            self: An instance of the Qwen2Model class.
            value: The input embeddings to be set for the model. This should be of type torch.Tensor.

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings for the Qwen2Model by assigning the provided 'value' to the
        'embed_tokens' attribute of the model instance.
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
        Construct method in the Qwen2Model class.

        Args:
            self (Qwen2Model): The instance of the Qwen2Model class.
            input_ids (mindspore.Tensor, optional): The input tensor containing token IDs. Default is None.
            attention_mask (mindspore.Tensor, optional): An optional tensor specifying the attention mask.
                Default is None.
            position_ids (mindspore.Tensor, optional): An optional tensor specifying the position IDs. Default is None.
            past_key_values (List[mindspore.Tensor], optional): An optional list of tensors for past key values.
                Default is None.
            inputs_embeds (mindspore.Tensor, optional): An optional tensor containing input embeddings. Default is None.
            use_cache (bool, optional): A flag indicating whether to use caching. Default is None.
            output_attentions (bool, optional): A flag indicating whether to output attentions. Default is None.
            output_hidden_states (bool, optional): A flag indicating whether to output hidden states. Default is None.
            return_dict (bool, optional): A flag indicating whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                Returns a tuple or BaseModelOutputWithPast object containing model outputs.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified, or if neither is specified.
            Warning: Raised if `use_cache=True` is incompatible with gradient checkpointing.
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
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

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


class Qwen2ForCausalLM(Qwen2PreTrainedModel):

    """
    This class represents a Qwen2 model for causal language modeling (LM). It is a subclass of Qwen2PreTrainedModel.
    The Qwen2ForCausalLM class provides methods for initializing the model, setting and getting input and output
    embeddings, setting and getting the decoder, constructing the model, and preparing inputs for generation.

    To initialize an instance of the Qwen2ForCausalLM class, a configuration object should be passed as a parameter
    to the constructor. The model's architecture and settings are defined by this configuration.

    The Qwen2ForCausalLM class has the following methods:

    - `__init__`: Initializes the Qwen2ForCausalLM instance with the given configuration.
    - `get_input_embeddings`: Returns the input embeddings of the model.
    - `set_input_embeddings`: Sets the input embeddings of the model to the given value.
    - `get_output_embeddings`: Returns the output embeddings of the model.
    - `set_output_embeddings`: Sets the output embeddings of the model to the given new_embeddings.
    - `set_decoder`: Sets the decoder of the model to the given decoder.
    - `get_decoder`: Returns the decoder of the model.
    - `construct`: Constructs the model using the provided input arguments.
    This method returns a tuple of outputs, including the logits and optionally the loss, past key values,
    hidden states, and attentions.
    - `prepare_inputs_for_generation`: Prepares the inputs for generation. This method takes input_ids, past_key_values,
    attention_mask, inputs_embeds, and additional keyword arguments as input and returns a dictionary of model inputs.
    - `_reorder_cache(past_key_values, beam_idx)`: Reorders the past key values according to the given beam indices.
    This method is static and is used internally in the class.

    Example:
        ```python
        >>> from transformers import Qwen2ForCausalLM, Qwen2Config
        ...
        >>> # Create a configuration object
        >>> config = Qwen2Config(vocab_size=100, hidden_size=512)
        ...
        >>> # Initialize a Qwen2ForCausalLM instance
        >>> model = Qwen2ForCausalLM(config)
        ...
        >>> # Set the input embeddings
        >>> embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        >>> model.set_input_embeddings(embeddings)
        ...
        >>> # Get the output embeddings
        >>> output_embeddings = model.get_output_embeddings()
        ...
        >>> # Set the decoder
        >>> decoder = Qwen2Model(config)
        >>> model.set_decoder(decoder)
        ...
        >>> # Get the decoder
        >>> decoder = model.get_decoder()
        ...
        >>> # Construct the model
        >>> input_ids = [1, 2, 3]
        >>> attention_mask = [1, 1, 1]
        >>> outputs = model.construct(input_ids=input_ids, attention_mask=attention_mask)
        ...
        >>> # Prepare inputs for generation
        >>> input_ids = [4, 5, 6]
        >>> past_key_values = [tensor1, tensor2]
        >>> attention_mask = [1, 1, 1]
        >>> inputs_embeds = [embedding1, embedding2]
        >>> model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values, attention_mask, inputs_embeds)
        ... 
        >>> # Reorder cache
        >>> past_key_values = [tensor1, tensor2]
        >>> beam_idx = [0, 1, 2]
        >>> reordered_past = Qwen2ForCausalLM._reorder_cache(past_key_values, beam_idx)
        ```
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the Qwen2ForCausalLM class.

        Args:
            self: The object itself.
            config: An instance of the Qwen2Config class containing the configuration settings for the model.
                This parameter is required and must not be None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings from the model.

        Args:
            self (Qwen2ForCausalLM): The object instance of the Qwen2ForCausalLM class.
                This parameter represents the instance of the Qwen2ForCausalLM class, which contains the model 
                for which input embeddings are to be retrieved.

        Returns:
            None: This method returns None, as it directly accesses and returns the input embeddings from the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the Qwen2ForCausalLM model.

        Args:
            self (Qwen2ForCausalLM): The instance of Qwen2ForCausalLM.
            value (object): The input embeddings to be set for the model.
                It can be an instance of a custom embedding class or any other object with
                the required attributes and methods.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Method: get_output_embeddings

        Description:
            This method returns the output embeddings from the Qwen2ForCausalLM model.

        Args:
            self: Qwen2ForCausalLM object.
                Represents the instance of the Qwen2ForCausalLM class.

        Returns:
            None
                This method returns None.

        Raises:
            None
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Sets the output embeddings for the Qwen2ForCausalLM model.

        Args:
            self (Qwen2ForCausalLM): The instance of the Qwen2ForCausalLM class.
            new_embeddings: The new embeddings to be set for the output layer.
                This can be a tensor or any other object that can be assigned to the 'lm_head' attribute of the
                Qwen2ForCausalLM instance.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the Qwen2ForCausalLM object.

        Args:
            self (Qwen2ForCausalLM): An instance of the Qwen2ForCausalLM class.
            decoder: The decoder object to be set as the model for Qwen2ForCausalLM.
                The decoder should implement the necessary methods and functionality required by Qwen2ForCausalLM.

        Returns:
            None.

        Raises:
            None.

        Note:
            The decoder object should be compatible with the Qwen2ForCausalLM class and fulfill the requirements
            necessary for generating predictions or processing inputs.

        Example:
            ```python
            >>> qwen2 = Qwen2ForCausalLM()
            >>> decoder = Decoder()
            >>> qwen2.set_decoder(decoder)
            ```
        """
        self.model = decoder

    def get_decoder(self):
        """
        Method to retrieve the decoder model from the Qwen2ForCausalLM class.

        Args:
            self (object): An instance of the Qwen2ForCausalLM class.
                This parameter is required for accessing the decoder model.

        Returns:
            model:
                The method returns the decoder model associated with the Qwen2ForCausalLM class.

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
            >>> from transformers import AutoTokenizer, Qwen2ForCausalLM
            ...
            >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
            >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
            ...
            >>> prompt = "Hey, are you conscious? Can you talk to me?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self (Qwen2ForCausalLM): An instance of the Qwen2ForCausalLM class.
            input_ids (torch.Tensor): The input token IDs of shape (batch_size, sequence_length).
            past_key_values (Union[Cache, tuple, None]): The cached key-value states from previous generations.
                If past_key_values is an instance of Cache, it contains information about the sequence length,
                past length, and maximum cache length. If past_key_values is a tuple, it contains the past length.
                If past_key_values is None, no cached key-value states are provided.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                It helps to mask out tokens that should not be attended to, such as padding tokens.
            inputs_embeds (torch.Tensor, optional): The input embeddings tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the embedded representation of the input tokens.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the model inputs for generation:

                - If inputs_embeds is not None and past_key_values is None, the dictionary contains
                {'inputs_embeds': inputs_embeds}.
                - Otherwise, the dictionary contains {'input_ids': input_ids}.
                - The dictionary also includes 'position_ids', 'past_key_values', 'use_cache', and 'attention_mask'.

        Raises:
            None.
        """
        # Omit tokens covered by past_key_values
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
    def _reorder_cache(past_key_values, beam_idx):
        """
        Method to reorder the cache based on the beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer. Each element in the tuple
                should be a tensor representing the past state for a layer.
            beam_idx (torch.Tensor): A tensor containing the beam indices used for reordering the past states.

        Returns:
            None: This method modifies the input past_key_values in place and does not return any explicit value.

        Raises:
            ValueError: If the past_key_values or beam_idx are not in the expected format or shape.
            IndexError: If the beam indices provided in beam_idx are out of bounds or not applicable to the
                past_key_values.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class Qwen2ForSequenceClassification(Qwen2PreTrainedModel):

    """
    Qwen2ForSequenceClassification is a class representing a sequence classification model that inherits from
    Qwen2PreTrainedModel. It includes methods for initializing the model with a configuration, getting
    and setting input embeddings, and constructing the model for sequence classification.

    Attributes:
        num_labels (int): The number of labels for sequence classification.

    Methods:
        __init__: Initializes the sequence classification model with the given configuration.
        get_input_embeddings: Retrieves the input embeddings from the model.
        set_input_embeddings: Sets the input embeddings for the model.
        construct: Constructs the sequence classification
            model with the specified inputs and returns the sequence classifier output with past values.

    Args:
        input_ids (Tensor, optional): The input tensor of shape `(batch_size, sequence_length)`
            representing the input sequence.
        attention_mask (Tensor, optional): The attention mask tensor of shape `(batch_size, sequence_length)`
            indicating which tokens should be attended to.
        position_ids (Tensor, optional): The position IDs tensor of shape `(batch_size, sequence_length)`
            representing the position of each token in the input sequence.
        past_key_values (List[Tensor], optional): The list of past key values tensors for handling incremental decoding.
        inputs_embeds (Tensor, optional): The input embeddings tensor of shape
            `(batch_size, sequence_length, hidden_size)` representing the embedded input sequence.
        labels (Tensor, optional): The tensor of shape `(batch_size,)` representing the labels for computing
            the sequence classification/regression loss.
        use_cache (bool, optional): Indicates whether to use the cache for handling incremental decoding.
        output_attentions (bool, optional): Indicates whether to output attentions.
        output_hidden_states (bool, optional): Indicates whether to output hidden states.
        return_dict (bool, optional): Indicates whether to return a dictionary of outputs.

    Returns:
        Union[Tuple, SequenceClassifierOutputWithPast]: The sequence classifier output with past values.

    Raises:
        ValueError: If batch sizes > 1 and no padding token is defined.

    Note:
        This docstring is generated based on the provided code and is intended to provide a comprehensive understanding
        of the Qwen2ForSequenceClassification class and its methods. Additional details and
        specific usage instructions may be available in the official documentation or source code.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the Qwen2ForSequenceClassification class.

        Args:
            self: The instance of the class.
            config: An object of the Qwen2Config class containing the configuration settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the Qwen2ForSequenceClassification model.

        Args:
            self: An instance of the Qwen2ForSequenceClassification class.

        Returns:
            embed_tokens: The method returns the input embeddings from the model.

        Raises:
            This method does not raise any exceptions.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Qwen2ForSequenceClassification model.

        Args:
            self (Qwen2ForSequenceClassification): The instance of the Qwen2ForSequenceClassification class.
            value (object): The input embeddings to be set for the model.
                Should be of type torch.Tensor or any compatible object.

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

        transformer_outputs = self.model(
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
        hidden_states = transformer_outputs[0]
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
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

__all__ = [
    "Qwen2ForCausalLM",
    "Qwen2Model",
    "Qwen2PreTrainedModel",
    "Qwen2ForSequenceClassification",
]
