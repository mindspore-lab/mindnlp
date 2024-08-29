# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
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
""" MindSpore Mistral model."""
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops, get_default_dtype
from mindnlp.core.nn import functional as F
from mindnlp.core.nn import CrossEntropyLoss
from mindnlp.utils import logging
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
from .configuration_mistral import MistralConfig


logger = logging.get_logger(__name__)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    Args:
        attention_mask (Tensor): A tensor representing the attention mask for the input data.
            It is used to mask the padding tokens in the input data. The tensor should have the same shape
            as the input data and have values of 0 for padding tokens and 1 for non-padding tokens.
    
    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A tensor containing the indices of non-padding tokens in the attention mask.
            - cu_seqlens (Tensor): A tensor representing the cumulative sum of sequence lengths in the batch.
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

    Raises:
        None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):

    """
    MistralRMSNorm is a normalization layer equivalent to T5LayerNorm, designed to normalize hidden states in
    deep learning models. It inherits from nn.Module and provides methods for normalizing and scaling input hidden states
    based on the given parameters.

    Attributes:
        hidden_size (int): The size of the hidden states.
        eps (float): The epsilon value used for numerical stability in variance calculation.

    Methods:
        __init__: Initializes the MistralRMSNorm layer with the specified hidden_size and epsilon value.
        forward: Normalizes the input hidden_states by calculating the variance and applying scaling.

    Example:
        ```python
        >>> # Initialize MistralRMSNorm layer
        >>> norm_layer = MistralRMSNorm(hidden_size=768, eps=1e-06)
        ...
        >>> # Normalize hidden states
        >>> normalized_states = norm_layer.forward(input_hidden_states)
        ```

    Note:
        - This implementation assumes the use of the MindSpore deep learning framework.
        - The class utilizes the Parameter and ops modules for efficient computation.
        - Make sure to convert input hidden states to the appropriate data type before passing to the forward method.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size), 'weight')
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Constructs the RMS normalization of the hidden states.

        Args:
            self (MistralRMSNorm): The instance of the MistralRMSNorm class.
            hidden_states (Tensor): The input tensor containing the hidden states.

        Returns:
            None: This method does not return any value.
                The normalization is applied in-place to the hidden_states tensor.

        Raises:
            TypeError: If the input_dtype of hidden_states is not supported.
            ValueError: If variance_epsilon is not a valid value.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):

    """
    The MistralRotaryEmbedding class represents a rotary positional embedding for sequences.
    It inherits from the nn.Module class and provides methods for setting up the rotary embedding and forwarding the
    embeddings for input sequences.

    Attributes:
        dim (int): The dimension of the embedding.
        max_position_embeddings (int): The maximum position for which embeddings are cached.
        base (int): The base value for calculating the inverse frequency.
        inv_freq (Tensor): The inverse frequency values used in the embedding calculation.
        max_seq_len_cached (int): The maximum sequence length for which embeddings are cached.
        cos_cached (Tensor): Cached cosine embeddings for positional sequences.
        sin_cached (Tensor): Cached sine embeddings for positional sequences.

    Methods:
        _set_cos_sin_cache: Sets up the cosine and sine cache for a given sequence length and data type.
        forward: Constructs the embeddings for the input sequence, optionally updating the cache if the sequence
            length exceeds the cached values.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes a new instance of the MistralRotaryEmbedding class.

        Args:
            self: The object itself.
            dim (int): The dimensionality of the embedding vectors.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value used for calculating inverse frequencies. Defaults to 10000.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        '''
        Set the cosine and sine cache for MistralRotaryEmbedding.

        Args:
            self (MistralRotaryEmbedding): The instance of MistralRotaryEmbedding.
            seq_len (int): The length of the input sequence. Must be a positive integer.
            dtype (str): The data type for the cache. Must be a valid data type supported by the system.

        Returns:
            None.

        Raises:
            ValueError: If seq_len is not a positive integer.
            TypeError: If dtype is not a valid data type.
        '''
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        """
        Constructs the MistralRotaryEmbedding.

        Args:
            self (MistralRotaryEmbedding): The instance of the MistralRotaryEmbedding class.
            x: The input tensor of shape (batch_size, input_dim).
            seq_len (int, optional): The length of the sequence. If not provided, the default value is None.

        Returns:
            None.

        Raises:
            ValueError: If seq_len is greater than the maximum sequence length cached.
            TypeError: If the data type of the input tensor is not supported for cosine and sine cache.
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
    x1, x2 = x.chunk(2, -1)
    return ops.cat((-x2, x1), dim=-1)


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


class MistralMLP(nn.Module):

    """
    MistralMLP

    This class represents a multi-layer perceptron (MLP) model for Mistral, a deep learning framework.
    It inherits from the nn.Module class and is designed for processing sequential data.

    Attributes:
        config (object): The configuration object containing various parameters for the MLP.
        hidden_size (int): The size of the hidden layer in the MLP.
        intermediate_size (int): The size of the intermediate layer in the MLP.
        gate_proj (nn.Linear): The dense layer used for projecting the input data to the intermediate size in the MLP.
        up_proj (nn.Linear): The dense layer used for projecting the input data to the hidden size in the MLP.
        down_proj (nn.Linear): The dense layer used for projecting the intermediate data back to the hidden size in the MLP.
        act_fn (function): The activation function used in the MLP.

    Methods:
        forward(x):
            Constructs the forward pass of the MistralMLP model.

            Args:

            - x (Tensor): The input data to be processed by the MLP.

            Returns:

            - Tensor: The output of the MLP after processing the input data.
    """
    def __init__(self, config):
        """
        Initializes a MistralMLP object with the provided configuration.

        Args:
            self (MistralMLP): The MistralMLP object itself.
            config (object): The configuration object containing parameters for the model.
                It should include the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - intermediate_size (int): The size of the intermediate layers.
                - hidden_act (str): The activation function to be used in the hidden layers.

        Returns:
            None.

        Raises:
            KeyError: If the 'hidden_act' attribute in the config object does not match any predefined
                activation function.
            AttributeError: If the config object is missing any of the required attributes
                (hidden_size, intermediate_size, hidden_act).
            ValueError: If any of the provided attributes have invalid values or types.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        Constructs a new instance of the MistralMLP class.

        Args:
            self (MistralMLP): The current instance of the MistralMLP class.
            x: The input data to be processed. It can be of any type.

        Returns:
            None.

        Raises:
            None.
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
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        """
        Initializes an instance of the MistralAttention class.

        Args:
            self: The MistralAttention instance.
            config (MistralConfig): The configuration object for the MistralAttention model.
            layer_idx (Optional[int], default=None): The index of the layer. If not provided,
                it will issue a warning and may cause errors during the forward call if caching is used.
                It is recommended to always provide a `layer_idx` when creating this class.

        Returns:
            None

        Raises:
            ValueError: If `hidden_size` is not divisible by `num_heads`.

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
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' in the class 'MistralAttention' reshapes the input tensor based on the provided parameters.

        Args:
            self (MistralAttention): The instance of the MistralAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None.

        Raises:
            ValueError: If the provided sequence length or batch size is not a positive integer.
            TypeError: If the input tensor is not of type mindspore.Tensor.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Constructs the MistralAttention.

        Args:
            self (MistralAttention): An instance of the MistralAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask of shape
                (batch_size, 1, sequence_length, key_value_sequence_length). If provided, it masks the attention scores.
            position_ids (Optional[mindspore.Tensor]): The position IDs of shape (batch_size, sequence_length).
            past_key_value (Optional): The cached key and value states for auto-regressive decoding.
                This is used to speed up decoding by reusing the key and value states from previous time steps.
            output_attentions (bool): Whether to return attention weights. Default is False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output of shape (batch_size, sequence_length, hidden_size),
                the attention weights of shape (batch_size, num_heads, sequence_length, key_value_sequence_length)
                if output_attentions is True, and the updated past key and value states if past_key_value is not None.

        Raises:
            ValueError: If the attention weights are not of shape (batch_size, num_heads, sequence_length, key_value_sequence_length).
            ValueError: If the attention mask is not of shape (batch_size, 1, sequence_length, key_value_sequence_length).
            ValueError: If the attention output is not of shape (batch_size, num_heads, sequence_length, hidden_size).
            ValueError: If the cache structure has changed since version v4.36 and past_key_value is not None.
        """
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
        attn_weights = ops.softmax(attn_weights, dim=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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


class MistralDecoderLayer(nn.Module):

    """
    MistralDecoderLayer represents a single layer of the Mistral decoder model. This class implements the logic for
    processing input hidden states through self-attention mechanism and multi-layer perceptron (MLP) in a decoder layer.

    Inherits From:
        nn.Module

    Attributes:
        hidden_size (int): The size of the hidden states.
        self_attn (MistralAttention): The self-attention mechanism used in the layer.
        mlp (MistralMLP): The multi-layer perceptron used in the layer.
        input_layernorm (MistralRMSNorm): Layer normalization applied to the input hidden states.
        post_attention_layernorm (MistralRMSNorm): Layer normalization applied after the self-attention mechanism.

    Methods:
        __init__:
            Initializes the MistralDecoderLayer with the given configuration and layer index.

        forward:
            Processes the input hidden states through self-attention and MLP mechanisms in the decoder layer,
            optionally returning additional tensors based on the arguments provided.

    Args:
        config (MistralConfig): Configuration object containing model hyperparameters.
        layer_idx (int): Index of the layer within the decoder model.

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]: Tuple containing the output
            hidden states and additional tensors based on the method arguments.

    Raises:
        NotImplementedError: If any specific requirements are not met.

    Example:
        Instantiate MistralDecoderLayer:
        ```python
        >>> config = MistralConfig(hidden_size=512)
        >>> layer = MistralDecoderLayer(config, layer_idx=1)
        ```
    """
    def __init__(self, config: MistralConfig, layer_idx: int):
        """
        Initializes a MistralDecoderLayer object.

        Args:
            self (MistralDecoderLayer): The instance of the MistralDecoderLayer class.
            config (MistralConfig): An object containing configuration parameters for the layer.
            layer_idx (int): The index of the layer within the decoder.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MistralConfig.
            ValueError: If the layer_idx parameter is not an integer.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MistralAttention(config, layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
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


class MistralPreTrainedModel(PreTrainedModel):

    """
    This class represents the Mistral pre-trained model for natural language processing tasks.
    It is a subclass of the PreTrainedModel class.

    The MistralPreTrainedModel class provides methods for initializing the weights of the model's cells.
    The _init_weights method is used to initialize the weights of the cells in the model.

    Parameters:
        cell: The cell for which the weights need to be initialized.

    The _init_weights method initializes the weights of the given cell based on its type.
    If the cell is of type nn.Linear, the weights are set using a normal distribution with a range specified by the
    'initializer_range' attribute in the configuration. If the cell has a bias, it is initialized with zeros.
    If the cell is of type nn.Embedding, the weights are initialized using a normal distribution with a range specified
    by the 'initializer_range' attribute in the configuration. If the cell has a padding index, the weight corresponding
    to the padding index is set to zero.

    Note:
        The MistralPreTrainedModel class assumes that the cell's weight and bias attributes are accessible using the
        'weight' and 'bias' properties, respectively.

    Example:
        ```python
        >>> model = MistralPreTrainedModel()
        >>> model._init_weights(cell)
        ```
    """
    config_class = MistralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MistralDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """
    def __init__(self, config: MistralConfig):
        """
        __init__

        Initialize MistralModel with the specified configuration.

        Args:
            self: The instance of the MistralModel class.
            config (MistralConfig):
                An instance of MistralConfig containing the configuration settings for the MistralModel.
                It includes the following attributes:

                - pad_token_id (int): The index of the padding token in the vocabulary.
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers in the model.
                - num_hidden_layers (int): The number of hidden layers in the model.
                - rms_norm_eps (float): The epsilon value for stability in the RMS normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method retrieves the input embeddings from the MistralModel instance.

        Args:
            self: MistralModel instance
                The self parameter refers to the current MistralModel instance.

        Returns:
            None:
                This method returns None as it simply retrieves the input embeddings from the MistralModel instance.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the MistralModel.

        Args:
            self (MistralModel): The instance of the MistralModel class.
            value (object): The input embeddings value to be set for the model. Should be of type 'object'.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def forward(
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
        This method forwards the MistralModel. It takes the following parameters:

        Args:
            self: The object itself.
            input_ids (mindspore.Tensor, optional): The input tensor of shape (batch_size, seq_length) representing input IDs.
            attention_mask (mindspore.Tensor, optional): An optional tensor of shape (batch_size, seq_length)
                representing attention mask.
            position_ids (mindspore.Tensor, optional): An optional tensor representing the position IDs.
            past_key_values (List[mindspore.Tensor], optional): An optional list of tensors representing past key values.
            inputs_embeds (mindspore.Tensor, optional): An optional tensor representing input embeddings.
            use_cache (bool, optional): An optional boolean flag indicating whether to use cache.
            output_attentions (bool, optional): An optional boolean flag indicating whether to output attentions.
            output_hidden_states (bool, optional): An optional boolean flag indicating whether to output hidden states.
            return_dict (bool, optional): An optional boolean flag indicating whether to return a dictionary.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                A tuple or BaseModelOutputWithPast object containing the last hidden state, past key values,
                hidden states, and attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified, if neither input_ids nor inputs_embeds are
                specified, or if an invalid argument combination is provided.
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


class MistralForCausalLM(MistralPreTrainedModel):

    """
    The MistralForCausalLM class represents a causal language model for Mistral. It inherits from MistralPreTrainedModel
    and includes methods for initializing the model, setting and getting input and output embeddings, setting the decoder,
    forwarding the model, and preparing inputs for generation. The class also includes a method for reordering cache
    during generation. The forward method handles the model's forward pass, while the prepare_inputs_for_generation
    method prepares inputs for generation. The class provides functionality for generating text based on input prompts.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of MistralForCausalLM.

        Args:
            self: The instance of the MistralForCausalLM class.
            config:
                A dictionary containing configuration parameters for the model.

                - Type: dict
                - Purpose: The configuration settings for the model, including hyperparameters, data paths,
                and model architecture.
                - Restrictions: Must contain required keys for model initialization.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type dict.
            ValueError: If the config parameter does not contain the required keys for model initialization.
        """
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the MistralForCausalLM model.

        Args:
            self (MistralForCausalLM): An instance of the MistralForCausalLM class.
                Represents the current MistralForCausalLM model object.

        Returns:
            None: This method returns None as it simply retrieves and returns the input embeddings
                from the model.

        Raises:
            None
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MistralForCausalLM model.

        Args:
            self (MistralForCausalLM): The object instance of MistralForCausalLM.
            value (Tensor): The input embeddings to be set for the model.
                It should be a tensor of shape (vocab_size, embed_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        This method returns the output embeddings for the MistralForCausalLM model.

        Args:
            self (MistralForCausalLM): The instance of MistralForCausalLM class.

        Returns:
            None: This method returns the output embeddings (lm_head) for the MistralForCausalLM model.

        Raises:
            This method does not raise any exceptions.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the MistralForCausalLM model.

        Args:
            self (MistralForCausalLM): The instance of the MistralForCausalLM class.
            new_embeddings (Tensor): The new output embeddings to be set for the model.
                It should be a tensor of the same shape as the existing output embeddings.

        Returns:
            None.

        Raises:
            ValueError: If the shape of the new_embeddings tensor does not match the shape of the
                existing output embeddings.
            TypeError: If the new_embeddings parameter is not of type Tensor.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the MistralForCausalLM model.

        Args:
            self (MistralForCausalLM): The MistralForCausalLM instance.
            decoder: The decoder that will be set for the model.
                It should be an object that implements the decoding logic for the MistralForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        '''
        Method: get_decoder

        Description:
        This method returns the decoder model for MistralForCausalLM.

        Args:
            self:
                MistralForCausalLM

                - Type: class instance
                - Purpose: Represents the current instance of the MistralForCausalLM class.

        Returns:
            model:

                - Type: None
                - Purpose: The method returns the decoder model associated with the MistralForCausalLM instance.

        Raises:
            None.
        '''
        return self.model

    def forward(
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
            >>> from transformers import AutoTokenizer, MistralForCausalLM
            ...
            >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
            # Enable model parallelism
            loss = F.cross_entropy(shift_logits, shift_labels)

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
            self (MistralForCausalLM): The instance of the MistralForCausalLM class.
            input_ids (torch.Tensor): The input tensor of token IDs with shape (batch_size, sequence_length).
            past_key_values (tuple, optional): The tuple of past key values for efficient generation. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor with shape (batch_size, sequence_length).
                It indicates which tokens should be attended to (1 for tokens to attend, 0 for tokens to ignore).
                Defaults to None.
            inputs_embeds (torch.Tensor, optional): The tensor of input embeddings with shape (batch_size, sequence_length,
                hidden_size). Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing model inputs for generation.
                It includes the following keys:

                - 'inputs_embeds' (torch.Tensor): The tensor of input embeddings with shape (batch_size, sequence_length,
                hidden_size). It is used when 'inputs_embeds' is not None and 'past_key_values' is None.
                - 'input_ids' (torch.Tensor): The input tensor of token IDs with shape (batch_size, sequence_length).
                It is used when 'inputs_embeds' is None or 'past_key_values' is not None.
                - 'position_ids' (torch.Tensor): The tensor of position IDs with shape (batch_size, sequence_length).
                It is computed based on 'attention_mask' and used for positional embeddings.
                - 'past_key_values' (tuple, optional): The tuple of past key values for efficient generation.
                - 'use_cache' (bool): Whether to use cache for generation.
                - 'attention_mask' (torch.Tensor, optional): The attention mask tensor with shape (batch_size, sequence_length).
                It is truncated or padded if necessary.

        Raises:
            None.
        """
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
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
            position_ids = attention_mask.int().cumsum(-1) - 1
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
        Reorders the cache for a causal language model.

        Args:
            past_key_values (tuple): A tuple containing past key values for each layer of the model.
            beam_idx (Tensor): A tensor specifying the indices to reorder the past key values by.

        Returns:
            None: The method modifies the input past_key_values in place and does not return any value.

        Raises:
            ValueError: If the input past_key_values or beam_idx are not in the expected format.
            IndexError: If the beam_idx contains out-of-bounds indices.
            TypeError: If the input types are not as expected.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Mistral, LLAMA->MISTRAL
class MistralForSequenceClassification(MistralPreTrainedModel):

    """
    This class represents a Mistral model for sequence classification. It inherits from MistralPreTrainedModel
    and provides functionality for sequence classification tasks.

    Attributes:
        num_labels (int): The number of labels in the classification task.
        model (MistralModel): The Mistral model used for the sequence classification task.
        score (nn.Linear): The dense layer for scoring the classification logits.

    Methods:
        __init__: Initializes the MistralForSequenceClassification instance with the given configuration.
        get_input_embeddings: Retrieves the input embeddings from the Mistral model.
        set_input_embeddings: Sets the input embeddings for the Mistral model.
        forward: Performs the sequence classification task and returns the classification output.

            Args:

            - input_ids (mindspore.Tensor, optional): The input token IDs.
            - attention_mask (mindspore.Tensor, optional): The attention mask for the input.
            - position_ids (mindspore.Tensor, optional): The position IDs for the input.
            - past_key_values (List[mindspore.Tensor], optional): The past key values for the input.
            - inputs_embeds (mindspore.Tensor, optional): The embedded inputs.
            - labels (mindspore.Tensor, optional): The labels for computing the sequence classification/regression loss.
            - use_cache (bool, optional): Indicates whether to use cache for the computation.
            - output_attentions (bool, optional): Indicates whether to output attentions.
            - output_hidden_states (bool, optional): Indicates whether to output hidden states.
            - return_dict (bool, optional): Indicates whether to return a dictionary.

            Returns:

            - Union[Tuple, SequenceClassifierOutputWithPast]: The classification output or a tuple with loss and
            output if loss is available.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MistralForSequenceClassification class.

        Args:
            self: The object instance.
            config:
                A configuration object that holds various parameters for the model.

                - Type: Config
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be an instance of the Config class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the MistralForSequenceClassification model.

        Args:
            self: The instance of the MistralForSequenceClassification class.

        Returns:
            None: This method returns None as it directly retrieves the input embeddings from the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MistralForSequenceClassification model.

        Args:
            self (MistralForSequenceClassification): An instance of the MistralForSequenceClassification class.
            value: The input embeddings to be set for the model. This should be a tensor of shape
                (vocab_size, embedding_dim).

        Returns:
            None. The method modifies the 'embed_tokens' attribute of the model in-place.

        Raises:
            None.

        Note:
            The 'embed_tokens' attribute of the MistralForSequenceClassification model is used to store the
            input embeddings.
            By setting this attribute, the user can customize the input embeddings used by the model.

        Example:
            ```python
            >>> model = MistralForSequenceClassification()
            >>> embeddings = torch.randn((vocab_size, embedding_dim))
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.model.embed_tokens = value

    def forward(
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
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
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
                    loss = F.mse_loss(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(pooled_logits, labels)
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

class MistralForTokenClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MistralModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
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
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "MistralForCausalLM",
    "MistralModel",
    "MistralPreTrainedModel",
    "MistralForSequenceClassification",
    "MistralForTokenClassification"
]
