# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
""" MindSpore Gemma model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS
from .configuration_gemma import GemmaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GemmaConfig"


def _get_unpad_data(attention_mask):
    """
    This function retrieves un-padded data from the attention_mask.
    
    Args:
        attention_mask (ndarray): An array representing the attention mask.
            Each element indicates whether a token is a valid input (1) or a padding token (0).
    
    Returns:
        indices (ndarray): An array of indices corresponding to the non-padding tokens in the attention_mask.
        cu_seqlens (ndarray): An array of cumulative sequence lengths, including the padding tokens.
        max_seqlen_in_batch (int): The maximum sequence length in the batch.
    
    Raises:
        None
    
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, axis=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class GemmaRMSNorm(nn.Cell):

    """
    This class represents a custom implementation of Root Mean Square Normalization (RMSNorm) called GemmaRMSNorm,
    which is designed for neural network operations.
    It inherits from the nn.Cell class. The GemmaRMSNorm class initializes with parameters for dimension and epsilon
    value, and includes methods for calculating the normalized output based on the input data and weight parameters.
    The _norm method calculates the normalized output based on the input data and epsilon value.
    The construct method applies the normalization and weight parameters to the input data to generate the final output.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initializes a GemmaRMSNorm instance.
        
        Args:
            self: The object instance itself.
            dim (int): The dimension of the GemmaRMSNorm.
            eps (float, optional): The epsilon value for numerical stability. Defaults to 1e-06.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.zeros(dim))

    def _norm(self, x):
        """
        Calculates the normalized value of a given input tensor 'x' using the root mean square (RMS) normalization method.
        
        Args:
            self (GemmaRMSNorm): An instance of the GemmaRMSNorm class.
            x (Tensor):
                The input tensor to be normalized.

                - Shape: (batch_size, ..., features)
                - dtype: torch.float32 or torch.float64

        Returns:
            None

        Raises:
            ValueError: If the input tensor 'x' is not a valid tensor.
            RuntimeError: If an error occurs during the calculation.

        Notes:
            - The RMS normalization method divides each element of the input tensor 'x' by the root mean square of the tensor.
            - The root mean square of 'x' is calculated as follows:

                - square each element of 'x'
                - calculate the mean across the last dimension of the tensor (features)
                - take the square root of the mean

            - The resulting normalized tensor has the same shape as the input tensor 'x'.
            - The 'keep_dims' argument in the mean operation ensures that the mean is calculated along the last
            dimension and the resulting tensor has the same number of dimensions as the input tensor.

        Example:
            ```python
            >>> norm = GemmaRMSNorm()
            >>> x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            >>> norm._norm(x)
            >>> # x is now normalized using the RMS normalization method.
            ```
        """
        return x * ops.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x):
        """
        Constructs a normalized tensor using the GemmaRMSNorm algorithm.

        Args:
            self (GemmaRMSNorm): An instance of the GemmaRMSNorm class.
            x (Tensor): The input tensor to be normalized. It should have a numeric data type.

        Returns:
            None: This method does not return any value.
                The normalized tensor is stored internally within the GemmaRMSNorm instance.

        Raises:
            TypeError: If the input tensor `x` is not of numeric data type.
        """
        output = self._norm(x.float()).astype(x.dtype)
        return output * (1 + self.weight)


ALL_LAYERNORM_LAYERS.append(GemmaRMSNorm)


class GemmaRotaryEmbedding(nn.Cell):

    """
    This class represents a GemmaRotaryEmbedding module, which is a custom embedding layer used in neural networks.
    It inherits from the nn.Cell class.

    The GemmaRotaryEmbedding module is designed to construct rotary embeddings for input data sequences.
    It creates embeddings based on the positions in the input sequence, using a sinusoidal function.
    The embeddings are computed as the cosine and sine of the frequency values derived from the positions.

    Attributes:
        dim (int): The dimension of the embeddings.
        max_position_embeddings (int): The maximum number of positions in the input sequence. Defaults to 2048.
        base (int): The base value used in the frequency calculation. Defaults to 10000.
        inv_freq (ndarray or None): An array storing the precomputed inverse frequencies. Defaults to None.

    Methods:
        __init__(self, dim, max_position_embeddings=2048, base=10000):
            Initializes the GemmaRotaryEmbedding module with the given parameters.

            Args:

            - dim (int): The dimension of the embeddings.
            - max_position_embeddings (int, optional): The maximum number of positions in the input sequence.
            Defaults to 2048.
            - base (int, optional): The base value used in the frequency calculation. Defaults to 10000.

        construct(self, x, position_ids, seq_len=None):
            Constructs the rotary embeddings based on the input data and position IDs.

            Args:

            - x (Tensor): The input data tensor.
            - position_ids (Tensor): The tensor containing the position IDs corresponding to each element in the
            input sequence.
            - seq_len (int, optional): The length of the input sequence. Defaults to None.

            Returns:

            - Tensor: The constructed rotary embeddings as the cosine and sine of the frequency values,
            casted to the same data type as the input tensor.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initialize GemmaRotaryEmbedding object with specified parameters.

        Args:
            self (object): The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): Maximum number of positions for the embeddings. Default is 2048.
            base (int, optional): Base value used for calculations. Default is 10000.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = None

    def construct(self, x, position_ids, seq_len=None):
        """
        Constructs GemmaRotaryEmbedding for positional encoding.

        Args:
            self (GemmaRotaryEmbedding): The instance of the GemmaRotaryEmbedding class.
            x (Tensor): The input tensor.
            position_ids (Tensor): The tensor containing positional IDs.
            seq_len (int): The length of the input sequence.

        Returns:
            The concatenated cosine and sine embeddings of the positional encoding.

        Raises:
            ValueError: If self.inv_freq is not initialized.
            TypeError: If the input tensors x and position_ids are not of the correct data type.
            IndexError: If the dimensions of the input tensors are incompatible for matrix multiplication.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim)
            )

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).swapaxes(1, 2)
        emb = ops.cat((freqs, freqs), axis=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x1 = x[..., : x.shape[-1] // 2]
    # x2 = x[..., x.shape[-1] // 2 :]
    x1, x2 = x.tensor_split(2, axis=-1)
    return ops.cat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

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


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Gemma
class GemmaMLP(nn.Cell):

    """
    GemmaMLP is a class representing a multi-layer perceptron (MLP) model for neural network operations.
    It inherits from nn.Cell and implements functionality for constructing the MLP.

    Attributes:
        config: A configuration object containing parameters for the MLP.
        hidden_size: The size of the hidden layers in the MLP.
        intermediate_size: The size of the intermediate layers in the MLP.
        gate_proj: A dense layer for projecting input to the intermediate size with no bias.
        up_proj: A dense layer for projecting input to the intermediate size with no bias.
        down_proj: A dense layer for projecting from intermediate size to hidden size with no bias.
        act_fn: The activation function to be used in the hidden layers.

    Methods:
        construct(x): Constructs the multi-layer perceptron using the given input x by applying the specified operations.
    """
    def __init__(self, config):
        """
        Initializes a GemmaMLP instance with the provided configuration.

        Args:
            self (GemmaMLP): The GemmaMLP instance to be initialized.
            config (Config):
                An object containing configuration parameters for the GemmaMLP model.

                - hidden_size (int): The size of the hidden layers in the model.
                - intermediate_size (int): The size of the intermediate layers in the model.

        Returns:
            None.

        Raises:
            TypeError: If config is not provided or is not of type Config.
            ValueError: If hidden_size or intermediate_size are not valid integer values.
            RuntimeError: If there is an issue initializing the gate_proj, up_proj, down_proj, or act_fn attributes.
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
        Constructs a multi-layer perceptron using the GemmaMLP class.

        Args:
            self (object): The instance of the GemmaMLP class.
            x (object): Input tensor or data to be processed by the MLP.

        Returns:
            None: The method modifies the internal state of the GemmaMLP instance.

        Raises:
            TypeError: If any of the input parameters are of incorrect types.
            ValueError: If there are issues during the execution of the method.
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


class GemmaAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # Ignore copy
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        """
        Initializes a new instance of the GemmaAttention class.

        Args:
            self: The object itself.
            config (GemmaConfig): The configuration object for the attention layer.
            layer_idx (Optional[int]): The index of the layer. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the `hidden_size` is not divisible by `num_heads`.

        Note:
            - If `layer_idx` is not provided, a warning message will be logged to indicate potential errors during the
            forward call if caching is used.
            - The GemmaAttention class performs attention calculations in the transformer model. It takes in a
            configuration object and initializes various attributes based on the provided configuration.
            - The attention_dropout attribute determines the dropout rate for attention weights.
            - The hidden_size attribute specifies the dimensionality of the hidden state.
            - The num_heads attribute specifies the number of attention heads.
            - The head_dim attribute specifies the dimensionality of each attention head.
            - The num_key_value_heads attribute specifies the number of key-value attention heads.
            - The num_key_value_groups attribute specifies the number of groups for key-value attention heads.
            - The max_position_embeddings attribute specifies the maximum number of position embeddings.
            - The rope_theta attribute specifies the base value for the rotary position encoding.
            - The is_causal attribute is set to True to indicate causal attention.
            - The q_proj attribute is a linear projection layer for the query values.
            - The k_proj attribute is a linear projection layer for the key values.
            - The v_proj attribute is a linear projection layer for the value values.
            - The o_proj attribute is a linear projection layer for the output values.
            - The rotary_emb attribute is a GemmaRotaryEmbedding object for rotary position encoding.
            - If the hidden_size is not divisible by num_heads, a ValueError will be raised.

        Example:
            ```python
            >>> config = GemmaConfig(hidden_size=768, num_attention_heads=12)
            >>> attention = GemmaAttention(config)
            ```
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
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=config.attention_bias)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=config.attention_bias)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
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
        use_cache: bool = False,
        cache_position: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        '''
        This method constructs attention output using the given hidden states and optional attention mask, position ids,
        past key value, and other parameters.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                An optional attention mask of shape (batch_size, sequence_length, sequence_length) to mask the
                attention scores. Default is None.
            position_ids (Optional[mindspore.Tensor]):
                An optional tensor of shape (batch_size, sequence_length) containing the position indices of the input tokens.
            past_key_value (Optional[Cache]):
                An optional cache of previous key and value states. Default is None.
            output_attentions (bool): A flag indicating whether to output the attention weights. Default is False.
            use_cache (bool): A flag indicating whether to use cache for previous key and value states. Default is False.
            cache_position (Optional[mindspore.Tensor]): An optional cache position tensor. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output tensor of shape (batch_size, sequence_length, hidden_size),
                optional attention weights tensor, and optional tuple of key and value cache states.

        Raises:
            ValueError: If the shape of `attn_output` does not match the expected shape
                (batch_size, num_heads, sequence_length, head_dim).
        '''
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, None)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            else:
                causal_mask = attention_mask
            attn_weights = attn_weights + causal_mask

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

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


GEMMA_ATTENTION_CLASSES = {
    "eager": GemmaAttention,
}


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with LLAMA->GEMMA,Llama->Gemma
class GemmaDecoderLayer(nn.Cell):

    """
    The GemmaDecoderLayer class represents a single layer of the Gemma decoder.
    It inherits from the nn.Cell class and provides methods for constructing the decoder layer.

    Attributes:
        hidden_size (int): The size of the hidden states in the layer.
        self_attn (GemmaAttention): The attention mechanism used in the layer.
        mlp (GemmaMLP): The multi-layer perceptron used in the layer.
        input_layernorm (GemmaRMSNorm): The layer normalization applied to the input.
        post_attention_layernorm (GemmaRMSNorm): The layer normalization applied after the attention mechanism.

    Methods:
        construct:
            Constructs the decoder layer using the given input and optional arguments.
            Returns the resulting hidden states and optionally the attention weights and present key value.

    Args:
        hidden_states (mindspore.Tensor): Input to the layer of shape (batch, seq_len, embed_dim).
        attention_mask (mindspore.Tensor, optional): Attention mask of size (batch_size, sequence_length)
            if flash attention is used or (batch_size, 1, query_sequence_length, key_sequence_length) if default
            attention is used.
        output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
        use_cache (bool, optional): If set to True, past key value states are returned and can be used to speed up decoding.
        past_key_value (Tuple(mindspore.Tensor), optional): Cached past key and value projection states.
        cache_position (mindspore.Tensor, optional): Position of the cache.
        **kwargs: Additional keyword arguments.

    Raises:
        DeprecationWarning: If 'padding_mask' is passed, a warning is issued indicating that it is deprecated and will
            be removed in a future version.

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
            The resulting hidden states and optionally the attention weights and present key value.
    """
    def __init__(self, config: GemmaConfig, layer_idx: int):
        """
        Initializes a new instance of the GemmaDecoderLayer class.

        Args:
            self: The object itself.
            config (GemmaConfig): The configuration object containing various settings.
            layer_idx (int): The index of the decoder layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GEMMA_ATTENTION_CLASSES["eager"](config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
            cache_position=cache_position,
            **kwargs,
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


class GemmaPreTrainedModel(PreTrainedModel):

    """
    The `GemmaPreTrainedModel` class is a subclass of `PreTrainedModel` that represents a pre-trained model for
    natural language processing tasks. It provides methods for initializing weights, setting up cache, and
    resetting cache.

    Methods:
        `_init_weights`: Initializes the weights of the given `cell`, which can be either a dense layer or an embedding layer.
        `_setup_cache`: Sets up the cache for the model using the specified cache class, maximum batch size,
            and maximum cache length.
        `_reset_cache`: Resets the cache for the model.

    Example:
        ```python
        >>> model = GemmaPreTrainedModel()
        >>> model._init_weights(cell)
        >>> model._setup_cache(cache_cls, max_batch_size, max_cache_len)
        >>> model._reset_cache()
        ```

    Note:
        The `GemmaPreTrainedModel` class inherits from `PreTrainedModel`. Refer to the documentation of `PreTrainedModel`
        for more information.
    """
    config_class = GemmaConfig
    base_model_prefix = "model"
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]
    _no_split_modules = ["GemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
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

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        """
        This method initializes the cache for the GemmaPreTrainedModel.

        Args:
            self (object): The instance of the GemmaPreTrainedModel class.
            cache_cls (class): The class representing the cache implementation.
            max_batch_size (int): The maximum batch size for caching.
            max_cache_len (int, Optional): The maximum length of the cache. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the attention implementation is 'flash_attention_2' and the cache_cls is StaticCache,
                as these are not compatible. It advises to use 'sdpa' as an alternative and to open an issue at
                https://github.com/huggingface/transformers.
            ValueError: If the max_cache_len exceeds the length of the model's causal mask.
                This ensures that the cache length does not exceed the model's capabilities.
        """
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        if max_cache_len > self.model.causal_mask.shape[-1]:
            causal_mask = ops.full((max_cache_len, max_cache_len), fill_value=1)
            self.causal_mask = ops.triu(causal_mask, diagonal=1)

        for layer in self.model.layers:
            weights = layer.self_attn.o_proj.weight
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len, dtype=weights.dtype
            )

    def _reset_cache(self):
        """
        Resets the cache for the GemmaPreTrainedModel.

        Args:
            self: GemmaPreTrainedModel instance. The instance of the GemmaPreTrainedModel for which the cache is to be reset.

        Returns:
            None.

        Raises:
            None.
        """
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->GEMMA,Llama->Gemma
class GemmaModel(GemmaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GemmaDecoderLayer`]

    Args:
        config: GemmaConfig
    """
    def __init__(self, config: GemmaConfig):
        """
        Initializes a GemmaModel instance.

        Args:
            self: The instance of the GemmaModel class.
            config (GemmaConfig): An instance of GemmaConfig containing the configuration parameters for the GemmaModel.
                This includes information such as the vocabulary size, hidden size, number of hidden layers, pad token id,
                maximum position embeddings, and RMS normalization epsilon.

                - config.pad_token_id (int): The padding token ID.
                - config.vocab_size (int): The size of the vocabulary.
                - config.hidden_size (int): The size of the hidden layers.
                - config.num_hidden_layers (int): The number of hidden layers.
                - config.max_position_embeddings (int): The maximum number of position embeddings.
                - config.rms_norm_eps (float): The epsilon value for RMS normalization.

        Returns:
            None: The method initializes various attributes of the GemmaModel instance, such as padding_idx, vocab_size,
                embed_tokens, layers, norm, gradient_checkpointing, causal_mask, and invokes the post_init method.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # register a causal mask to separate causal and padding mask creation. Merging happends in the attention class
        causal_mask = ops.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
        self.causal_mask = ops.triu(causal_mask, diagonal=1)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings for the GemmaModel.

        Args:
            self (GemmaModel): An instance of the GemmaModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the GemmaModel.

        Args:
            self (GemmaModel): The instance of the GemmaModel class.
            value: The input embeddings to set for the model. This should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    # Ignore copy
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
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs GemmaModel.

        This method constructs the GemmaModel and performs the forward pass of the model.
        It takes various input parameters and returns the output hidden states, cache values, and attention values.

        Args:
            self (GemmaModel): The instance of the GemmaModel class.
            input_ids (mindspore.Tensor, optional):
                The input tensor containing the tokenized input sequence. Default is None.
            attention_mask (mindspore.Tensor, optional):
                The attention mask tensor to avoid attending to padding tokens. Default is None.
            position_ids (mindspore.Tensor, optional):
                The position indices tensor to specify the position of each token. Default is None.
            past_key_values (List[mindspore.Tensor], optional):
                The list of tensors containing the cached key-value pairs of the previous attention mechanism.
                Default is None.
            inputs_embeds (mindspore.Tensor, optional): The input embedding tensor. Default is None.
            use_cache (bool, optional): Whether to use cache mechanism. Default is None.
            output_attentions (bool, optional): Whether to output the attention values. Default is None.
            output_hidden_states (bool, optional): Whether to output the hidden states. Default is None.
            return_dict (bool, optional): Whether to return the output as a dictionary. Default is None.
            cache_position (mindspore.Tensor, optional): The tensor representing the position of each token in the cache. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The output of the model.
                It can be a tuple containing hidden states, cache values, hidden states from all layers,
                and attention values from all layers; or an instance of BaseModelOutputWithPast containing the
                last hidden state, cache values, hidden states from all layers, and attention values from all layers.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified or neither of them is specified.
            Warning: If use_cache is set to True while using gradient checkpointing, it will be set to False as it is not compatible.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = ops.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        hidden_states = hidden_states * (self.config.hidden_size**0.5)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
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
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
    # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
    # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
    # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
    def _update_causal_mask(self, attention_mask, input_tensor):
        '''
        Updates the causal mask used for self-attention in the GemmaModel class.

        Args:
            self (GemmaModel): The instance of the GemmaModel class.
            attention_mask (Tensor, optional): The attention mask tensor. Default is None.
            input_tensor (Tensor): The input tensor used to determine the shape of the causal mask.

        Returns:
            None

        Raises:
            None
        '''
        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = ops.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
            self.causal_mask = ops.triu(causal_mask, diagonal=1)

        # We use the current dtype to avoid any overflows
        causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * finfo(dtype, 'min')

        causal_mask = causal_mask.to(dtype=dtype)
        if attention_mask is not None and attention_mask.dim() == 2:
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[..., :mask_length].eq(0.0).astype(mindspore.int32) * attention_mask[:, None, None, :].eq(0.0).astype(mindspore.int32)
            causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                padding_mask.astype(mindspore.bool_), finfo(dtype, 'min')
            )

        return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->GEMMA,Llama->Gemma,llama->gemma
class GemmaForCausalLM(GemmaPreTrainedModel):

    """
    This class represents a model for Causal Language Modeling using the Gemma architecture.
    It provides methods for setting and getting input and output embeddings, setting the decoder, and generating text
    based on input sequences. The class also includes methods for preparing inputs for text generation
    and reordering past key values.

    The class inherits from GemmaPreTrainedModel and includes the following methods:

    - __init__: Initializes the model with the given configuration.
    - get_input_embeddings): Returns the input embeddings.
    - set_input_embeddings: Sets the input embeddings to the given value.
    - get_output_embeddings: Returns the output embeddings.
    - set_output_embeddings: Sets the output embeddings to the new embeddings.
    - set_decoder: Sets the decoder model.
    - get_decoder: Returns the decoder model.
    - construct: Constructs the model for
    text generation.
    - prepare_inputs_for_generation: Prepares inputs for text generation.
    - _reorder_cache(past_key_values, beam_idx): Reorders the cache based on the beam index.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM
        ...
        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        ...
        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        ...
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> "What is your favorite condiment?"
        ```
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the GemmaForCausalLM class.

        Args:
            self: The object itself.
            config: An instance of the configuration class that holds the model configuration settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the GemmaForCausalLM model.

        Args:
            self (GemmaForCausalLM): An instance of the GemmaForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the GemmaForCausalLM model.

        Args:
            self (GemmaForCausalLM): The instance of the GemmaForCausalLM class.
            value: The input embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method sets the input embeddings for the GemmaForCausalLM model. The input embeddings are used to map
            input tokens to their corresponding embedding vectors. The `value` parameter should be an object containing
            the desired input embeddings. The input embeddings are assigned to the `embed_tokens` attribute of the model.

        Example:
            ```python
            >>> model = GemmaForCausalLM()
            >>> embeddings = Embeddings()
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from a GemmaForCausalLM model.

        Args:
            self (GemmaForCausalLM): The instance of GemmaForCausalLM class.
                Represents the model object for which the output embeddings are to be retrieved.

        Returns:
            None: This method returns None as it directly provides access to the 'lm_head' attribute
                containing the output embeddings.

        Raises:
            None
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the GemmaForCausalLM model.

        Args:
            self (GemmaForCausalLM): The GemmaForCausalLM instance.
            new_embeddings (torch.Tensor): The new embeddings to be set as the model's output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the GemmaForCausalLM model.

        Args:
            self (GemmaForCausalLM): The instance of the GemmaForCausalLM class.
            decoder: The decoder object to be set for the model. It should be compatible with the GemmaForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Returns the decoder model used for causal language modeling in the GemmaForCausalLM class.

        Args:
            self: An instance of the GemmaForCausalLM class.

        Returns:
            The decoder model:
                which is an instance of the model used for causal language modeling.

        Raises:
            None.
        """
        return self.model

    # Ignore copy
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
        cache_position: Optional[mindspore.Tensor] = None,
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
            >>> from transformers import AutoTokenizer, GemmaForCausalLM
            ...
            >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
            >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
            ...
            >>> prompt = "What is your favorite condiment?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")
            ...
            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "What is your favorite condiment?"
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
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self (object): The instance of the GemmaForCausalLM class.
            input_ids (tensor): The input tensor containing token indices for the input sequence.
            past_key_values (Cache or tuple or None): The past key values used in the generation process.
                If past_key_values is a Cache object, it contains the cached key value states.
                If past_key_values is a tuple, it represents the cached key value states as a tuple of tensors.
                If past_key_values is None, no cached key value states are used.
            attention_mask (tensor or None): The attention mask tensor used to mask the input sequence.
                If provided, it should have the same shape as input_ids.
                If None, no attention mask is applied.
            inputs_embeds (tensor or None): The tensor containing the embedded input embeddings.
                If provided, it should have the same shape as input_ids.
                If None, input_ids is used for token embeddings.

        Returns:
            dict or None: A dictionary containing the model inputs including input_ids, position_ids, cache_position,
                past_key_values, use_cache, and attention_mask. Returns None if no inputs are provided.

        Raises:
            TypeError: If input_ids, attention_mask, or inputs_embeds have invalid types.
            ValueError: If input_ids and attention_mask have incompatible shapes.
            ValueError: If cache_position is not None and is not a valid cache position tensor.
            ValueError: If past_key_values is not of type Cache or tuple.
        """
        past_length = 0
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

        if getattr(self.model.layers[0].self_attn, "past_key_value", None) is not None:
            # generation with static cache
            cache_position = kwargs.get("cache_position", None)
            if cache_position is None:
                past_length = 0
            else:
                past_length = cache_position[-1] + 1
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        # TODO @gante we should only keep a `cache_position` in generate, and do +=1.
        # same goes for position ids. Could also help with continued generation.
        cache_position = ops.arange(past_length, past_length + position_ids.shape[-1])

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the given beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key-value states for each layer of the model.
                Each layer's past key-value state is a tuple of tensors with shape (batch_size, sequence_length, hidden_size).
            beam_idx (torch.Tensor): A tensor of shape (batch_size,) representing the beam index to reorder the cache for.

        Returns:
            None: This method does not return any value. The cache is reordered in-place.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with LLAMA->GEMMA,Llama->Gemma
class GemmaForSequenceClassification(GemmaPreTrainedModel):

    """
    A Python class that represents a Gemma model for sequence classification tasks.
    This class inherits from the GemmaPreTrainedModel class.

    This class provides methods for initializing the model, getting and setting input embeddings, and constructing
    the model for sequence classification. It also includes methods for computing the loss and returning the model outputs.

    Attributes:
        num_labels (int): The number of labels for the sequence classification task.
        model (GemmaModel): The underlying Gemma model.
        score (nn.Dense): The dense layer for computing the logits.

    Methods:
        __init__: Initializes the GemmaForSequenceClassification instance with the given configuration.
        get_input_embeddings: Returns the input embeddings of the model.
        set_input_embeddings: Sets the input embeddings of the model.
        construct: Constructs the model for sequence classification and returns the model outputs.

    Example:
        ```python
        >>> # Initialize the GemmaForSequenceClassification instance
        >>> model = GemmaForSequenceClassification(config)
        ...
        >>> # Get the input embeddings
        >>> embeddings = model.get_input_embeddings()
        ...
        >>> # Set new input embeddings
        >>> model.set_input_embeddings(embeddings)
        ...
        >>> # Construct the model for sequence classification
        >>> outputs = model.construct(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
        ...
        >>> # Get the logits and past key values
        >>> logits = outputs.logits
        >>> past_key_values = outputs.past_key_values
        ...
        >>> # Compute the loss
        >>> loss = outputs.loss
        ...
        >>> # Return the model outputs
        >>> return_dict = True
        >>> output = model.construct(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict)
        ```

    Note:
        This class assumes that the GemmaPreTrainedModel class is already defined and imported.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the GemmaForSequenceClassification class.

        Args:
            self: The object itself.
            config (class): A configuration class that contains the necessary parameters for initializing the model.
                This includes the number of labels for classification.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GemmaModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the GemmaForSequenceClassification model.

        Args:
            self: The instance of the GemmaForSequenceClassification class.

        Returns:
            embed_tokens: This method returns the input embeddings from the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the GemmaForSequenceClassification model.

        Args:
            self (GemmaForSequenceClassification): The instance of the GemmaForSequenceClassification class.
            value (object): The input embeddings to be set for the model. This should be an object that represents the
                embeddings, such as a tensor or a list of tensors.

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
    "GemmaForCausalLM",
    "GemmaModel",
    "GemmaPreTrainedModel",
    "GemmaForSequenceClassification",
]
