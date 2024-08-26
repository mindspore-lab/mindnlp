# coding=utf-8
# Copyright 2024 BigCode and the HuggingFace Inc. team. All rights reserved.
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
# ===========================================================================
""" MindSpore Starcoder2 model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
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
from .configuration_starcoder2 import Starcoder2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Starcoder2Config"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    '''
    This function retrieves unpad data from the attention mask.
    
    Args:
        attention_mask (Tensor): A tensor representing the attention mask of the input data.
    
    Returns:
        indices (Tensor): A tensor containing the indices of the flattened attention mask.
        cu_seqlens (Tensor): A tensor representing the cumulative sequence lengths of the attention mask.
        max_seqlen_in_batch (int): The maximum sequence length in the batch.
    
    Raises:
        None
    '''
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten()).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Starcoder2
class Starcoder2RotaryEmbedding(nn.Module):

    """
    The Starcoder2RotaryEmbedding class represents a rotary embedding module used for positional encoding in neural
    network models. This class inherits from the nn.Module class.
    
    The class's forwardor method initializes the Starcoder2RotaryEmbedding instance with the specified dimensions,
    maximum position embeddings, and base value for the rotary embedding. It computes the inverse frequency and sets
    the cosine and sine cache for positional encoding.

    The _set_cos_sin_cache method sets the cosine and sine cache based on the maximum sequence length and data type.

    The forward method applies the positional encoding to the input tensor based on the sequence length and returns
    the cosine and sine embeddings.

    Note:
        This docstring is a detailed summary of the Starcoder2RotaryEmbedding class and its methods, providing an
        overview of its functionality and purpose within the context of neural network modeling.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initialize the Starcoder2RotaryEmbedding object.

        Args:
            self (Starcoder2RotaryEmbedding): The current instance of the Starcoder2RotaryEmbedding class.
            dim (int): The dimensionality of the embedding.
            max_position_embeddings (int, optional): The maximum number of positions to embed. Default is 2048.
            base (int, optional): The base value used in the calculation. Default is 10000.

        Returns:
            None.

        Raises:
            ValueError: If dim is not an integer.
            ValueError: If max_position_embeddings is not an integer.
            ValueError: If base is not an integer.
            ValueError: If any of the provided values are invalid or out of range.
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
        Sets the cosine and sine cache for the Starcoder2RotaryEmbedding class.

        Args:
            self: The instance of the Starcoder2RotaryEmbedding class.
            seq_len (int): The length of the sequence.
            dtype: The data type for the cache.

        Returns:
            None: This method modifies the cos_cached and sin_cached attributes of the Starcoder2RotaryEmbedding instance.

        Raises:
            None.
        """
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=mindspore.int64).type_as(self.inv_freq)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len=None):
        """
        Construct and return the cosine and sine embeddings for the given sequence length.

        Args:
            self (Starcoder2RotaryEmbedding): An instance of the Starcoder2RotaryEmbedding class.
            x: The input tensor.
            seq_len (Optional[int]): The length of the sequence. If not provided, the default value is None.

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor]:
                A tuple containing the cosine and sine embeddings for the given sequence length.

                - The cosine embedding is obtained by taking the first 'seq_len' elements from the cached cosine values.
                - The sine embedding is obtained by taking the first 'seq_len' elements from the cached sine values.

                Both embeddings are converted to the same data type as the input tensor 'x'.

        Raises:
            TypeError: If the input 'seq_len' is not an integer.
            ValueError: If the input 'seq_len' is negative.
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
    return ops.cat((-x2, x1), dim=-1)


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


class Starcoder2MLP(nn.Module):

    '''
    A class representing a multi-layer perceptron (MLP) for Starcoder2 model.

    This class inherits from nn.Module and implements the forwardion of the MLP for the Starcoder2 model.
    The MLP consists of fully connected layers with activation functions and residual dropout.

    Attributes:
        config (Starcoder2Config): The configuration for the Starcoder2 model.

    Methods:
        __init__:
            Initializes the Starcoder2MLP with the given configuration.

        forward:
            Constructs the multi-layer perceptron using the provided hidden states.

    '''
    def __init__(self, config: Starcoder2Config):
        """
        Initializes a Starcoder2MLP instance.

        Args:
            self (Starcoder2MLP): The current instance of the Starcoder2MLP class.
            config (Starcoder2Config): An instance of Starcoder2Config containing the configuration parameters.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of type Starcoder2Config.
            ValueError: If the provided config parameter does not contain valid configuration values.
        """
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, config.intermediate_size, bias=config.use_bias)
        self.c_proj = nn.Linear(config.intermediate_size, embed_dim, bias=config.use_bias)
        self.act = ACT2FN[config.hidden_act]
        self.residual_dropout = config.residual_dropout

    def forward(self, hidden_states: Optional[Tuple[mindspore.Tensor]]) -> mindspore.Tensor:
        """
        This method forwards the forward pass of the Starcoder2MLP model.

        Args:
            self (Starcoder2MLP): The instance of the Starcoder2MLP class.
            hidden_states (Optional[Tuple[mindspore.Tensor]]): The input hidden states to be processed.
                It can be a tuple of mindspore.Tensor objects or None.

        Returns:
            mindspore.Tensor: The processed hidden states after passing through the MLP layers.

        Raises:
            None.
        """
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.residual_dropout, training=self.training)
        return hidden_states


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


class Starcoder2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: Starcoder2Config, layer_idx: Optional[int] = None):
        """
        Initializes a new instance of the Starcoder2Attention class.

        Args:
            self: The object instance.
            config (Starcoder2Config): The configuration object containing various model hyperparameters.
            layer_idx (Optional[int], default=None): The index of the layer. If None, a warning will be logged and
                it is not recommended to omit this parameter as it may cause errors during the forward call if
                caching is used.

        Returns:
            None

        Raises:
            ValueError: If the `hidden_size` is not divisible by `num_heads`.

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

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.use_bias = config.use_bias
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.residual_dropout = config.residual_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.use_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.use_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.use_bias)

        self.rotary_emb = Starcoder2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        '''
        This method forwards the Starcoder2Attention layer.

        Args:
            self (Starcoder2Attention): The instance of the Starcoder2Attention layer.
            hidden_states (mindspore.Tensor): The input hidden states tensor with shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor for masking the attention scores with
                shape (batch_size, 1, sequence_length, key_value_sequence_length).
            position_ids (Optional[mindspore.Tensor]): An optional tensor to specify the position ids with shape
                (batch_size, sequence_length).
            past_key_value (Optional[Cache]): An optional cache for previous key and value states.
            output_attentions (bool): A flag to indicate whether to return the attention weights.
            use_cache (bool): A flag to indicate whether to use the cache for key and value states.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]: A tuple containing
                the attention output tensor with shape (batch_size, sequence_length, hidden_size),
                attention weights tensor (optional), and past key value tuple (optional).

        Raises:
            ValueError: If the attention weights or attention mask shape does not match the expected shape.
            ValueError: If the output size of `attn_output` does not match the expected shape.
            ValueError: If the cache structure has changed and the layer index is not initialized for
                auto-regressive decoding.
            '''
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated.37. Please make sure use `attention_mask` instead.`"
            )
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
        attn_output = F.dropout(attn_output, p=self.residual_dropout, training=self.training)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


STARCODER2_ATTENTION_CLASSES = {
    "eager": Starcoder2Attention,
}


class Starcoder2DecoderLayer(nn.Module):

    """
    The Starcoder2DecoderLayer class represents a single layer of the Starcoder2 decoder model.
    This class inherits from nn.Module and implements the operations required for the decoder layer.

    Attributes:
        hidden_size (int): The size of the hidden state in the layer.
        self_attn (STARCODER2_ATTENTION_CLASSES): The self-attention mechanism used in the layer.
        mlp (Starcoder2MLP): The multi-layer perceptron used in the layer.
        input_layernorm (nn.LayerNorm): The layer normalization applied to the input.
        post_attention_layernorm (nn.LayerNorm): The layer normalization applied after the attention mechanism.

    Methods:
        forward:
            Applies the operations of the decoder layer to the input hidden states and returns the output along with
            optional values based on the provided arguments.

    Args:
        config (Starcoder2Config): The configuration for the Starcoder2 model.
        layer_idx (int): The index of the layer within the model.

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]: The output tensor and optionally,
            attention weights and/or present key value states.

    Raises:
        ValueError: If the input dimensions are incompatible.

    Note:
        - The attention_mask should indicate padding elements with 0.
        - If output_attentions is True, the attention weights for all attention layers will be returned.
        - If use_cache is True, the present_key_value states can be used to speed up decoding.
        - The input hidden_states should be of shape (batch, seq_len, embed_dim).
    """
    def __init__(self, config: Starcoder2Config, layer_idx: int):
        """
        Initializes a new instance of the Starcoder2DecoderLayer class.

        Args:
            self: The object itself.
            config (Starcoder2Config): An instance of the Starcoder2Config class containing the configuration settings
                for the decoder layer.
            layer_idx (int): The index of the decoder layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = STARCODER2_ATTENTION_CLASSES["eager"](config, layer_idx)

        self.mlp = Starcoder2MLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

    # Copied from transformers.models.mistral.modeling_mistral.MistralDecoderLayer.forward
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


# Copied from transformers.models.mistral.modeling_mistral.MistralPreTrainedModel with Mistral->Starcoder2
class Starcoder2PreTrainedModel(PreTrainedModel):

    """
    This class represents a Starcoder2PreTrainedModel, which is a subclass of PreTrainedModel.

    The Starcoder2PreTrainedModel class provides methods for initializing the weights of different types of cells.
    The initialization process depends on the type of the cell. If the cell is an instance of nn.Linear, the weights are
    initialized using a normal distribution with a mean of 0 and a standard deviation defined by the `initializer_range`
    attribute of the configuration. If the cell has a bias, the bias is initialized with zeros.

    If the cell is an instance of nn.Embedding, the weights are initialized using a normal distribution with a mean
    of 0 and a standard deviation defined by the `initializer_range` attribute of the configuration.
    If the cell has a padding index, the weight corresponding to the padding index is set to 0.

    Note:
        It is assumed that the `cell` parameter passed to the `_init_weights` method is an instance of either nn.Linear
        or nn.Embedding.

    Please refer to the source code for more details on the implementation.

    """
    config_class = Starcoder2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Starcoder2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class Starcoder2Model(Starcoder2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Starcoder2DecoderLayer`]

    Args:
        config: Starcoder2Config
    """
    def __init__(self, config: Starcoder2Config):
        """
        Initializes a new instance of the Starcoder2Model class.

        Args:
            self (Starcoder2Model): The current instance of the Starcoder2Model class.
            config (Starcoder2Config):
                An instance of Starcoder2Config containing the configuration parameters for the model.

                - config.pad_token_id (int): The index of the padding token in the vocabulary.
                - config.vocab_size (int): The size of the vocabulary.
                - config.hidden_size (int): The size of the hidden layers.
                - config.embedding_dropout (float): The dropout probability for the embedding layer.
                - config.num_hidden_layers (int): The number of hidden layers in the model.
                - config.norm_epsilon (float): The epsilon value for normalization.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not an instance of Starcoder2Config.
            ValueError: If the config parameters are invalid or out of range.
            RuntimeError: If there is an issue with initializing the model attributes.
            """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embedding_dropout = config.embedding_dropout
        self.layers = nn.ModuleList(
            [Starcoder2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # self._attn_implementation = config._attn_implementation
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the Starcoder2Model.

        Args:
            self: The instance of the Starcoder2Model class.

        Returns:
            None: This method returns the input embeddings as stored in the 'embed_tokens' attribute of the
                Starcoder2Model instance.

        Raises:
            None
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Starcoder2Model.

        Args:
            self (Starcoder2Model): The instance of the Starcoder2Model class.
            value (any): The input embeddings to be set for the model.

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
        Constructs the Starcoder2Model.

        Args:
            self (Starcoder2Model): The instance of the Starcoder2Model class.
            input_ids (mindspore.Tensor, optional): The input tensor containing the indices of input tokens.
                Defaults to None.
            attention_mask (mindspore.Tensor, optional): The attention mask tensor. Defaults to None.
            position_ids (mindspore.Tensor, optional): The position ids tensor. Defaults to None.
            past_key_values (List[mindspore.Tensor], optional): The list of tensors containing past key values.
                Defaults to None.
            inputs_embeds (mindspore.Tensor, optional): The embedded input tensors. Defaults to None.
            use_cache (bool, optional): Flag to indicate whether to use cache. Defaults to None.
            output_attentions (bool, optional): Flag to indicate whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Flag to indicate whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Flag to indicate whether to return a dictionary. Defaults to None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: The forwarded model output.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified, or if neither of them is specified.
            Warning: If use_cache is set to True and gradient checkpointing is enabled,
                the use_cache flag will be overridden.
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

        # if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        #     is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        #     if is_padding_right:
        #         raise ValueError(
        #             "You are attempting to perform batched generation with padding_side='right'"
        #             " this may lead to unexpected behaviour for Flash Attention version of Starcoder2. Make sure to "
        #             " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
        #         )

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

        hidden_states = inputs_embeds
        hidden_states = F.dropout(hidden_states, p=self.embedding_dropout, training=self.training)

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


# Copied from transformers.models.mistral.modeling_mistral.MistralForCausalLM with MISTRAL->STARCODER2,Mistral-7B-v0.1->starcoder2-7b_16k,Mistral->Starcoder2,mistralai->bigcode
class Starcoder2ForCausalLM(Starcoder2PreTrainedModel):
    r"""
    The Starcoder2ForCausalLM class represents a Starcoder2 model for causal language modeling.
    It inherits from the Starcoder2PreTrainedModel.

    This class provides methods to initialize the model, get and set input embeddings, get and set output embeddings,
    set the decoder, get the decoder, forward the model with various optional inputs, prepare inputs for generation,
    and reorder the cache.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, Starcoder2ForCausalLM
        ...
        >>> model = Starcoder2ForCausalLM.from_pretrained("bigcode/starcoder2-7b_16k")
        >>> tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b_16k")
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
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        This method initializes an instance of the Starcoder2ForCausalLM class.

        Args:
            self: The instance of the class.
            config: A dictionary containing configuration parameters for the model.
                It is used to initialize the Starcoder2Model, determine the vocabulary size, and configure the lm_head.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = Starcoder2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings from the model.

        Args:
            self: The instance of Starcoder2ForCausalLM class.
                This parameter is required to access the model's embedded tokens.

        Returns:
            None: This method returns None as it simply retrieves the input embeddings from the model.

        Raises:
            None
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the Starcoder2ForCausalLM model.

        Args:
            self (Starcoder2ForCausalLM): The instance of the Starcoder2ForCausalLM class.
            value: The input embeddings to be set for the model. It should be compatible with the model's
                embed_tokens attribute.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the Starcoder2ForCausalLM class.

        Args:
            self:
                Instance of the Starcoder2ForCausalLM class.

                - Type: Starcoder2ForCausalLM
                - Purpose: Represents the current instance of the class.
                - Restrictions: None

        Returns:
            lm_head:
                The output embeddings.

                - Type: None
                - Purpose: The method returns the output embeddings.

        Raises:
            None
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for Starcoder2ForCausalLM.

        This method allows the user to set the output embeddings for the Starcoder2ForCausalLM model.
        The output embeddings are responsible for generating the predicted output sequence.

        Args:
            self (Starcoder2ForCausalLM): The current instance of the Starcoder2ForCausalLM class.
            new_embeddings (Any): The new embeddings to set as the output embeddings.
                This can be of any type, as long as it is compatible with the model architecture.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the Starcoder2ForCausalLM class.

        Args:
            self (Starcoder2ForCausalLM): The instance of the Starcoder2ForCausalLM class.
            decoder: The decoder object to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        This method returns the decoder model associated with the Starcoder2ForCausalLM instance.

        Args:
            self: The instance of the Starcoder2ForCausalLM class.

        Returns:
            model: This method returns the decoder model associated with the instance.

        Raises:
            None.
        """
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
            >>> from transformers import AutoTokenizer, Starcoder2ForCausalLM
            ...
            >>> model = Starcoder2ForCausalLM.from_pretrained("bigcode/starcoder2-7b_16k")
            >>> tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b_16k")
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
            # Ensure tensors are on the same device
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
            self: An instance of the Starcoder2ForCausalLM class.
            input_ids (mindspore.Tensor): Tensor of shape (batch_size, sequence_length) containing the input IDs.
            past_key_values (Cache, tuple, or None): Cache object or tuple of tensors containing the past key values.
                If Cache object is provided, the cache_length, past_length, and max_cache_length are
                extracted. If tuple is provided, cache_length and past_length are extracted from the first element.
                If None, cache_length and past_length are calculated based on input_ids.
            attention_mask (mindspore.Tensor or None): Tensor of shape (batch_size, sequence_length)
                containing the attention mask. If not None and attention_mask.shape[1] is greater than
                input_ids.shape[1], the input_ids are truncated accordingly. If attention_mask is not None and
                past_length is less than input_ids.shape[1], the input_ids are sliced accordingly.
                If max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] is
                greater than max_cache_length, the attention_mask is truncated accordingly.
            inputs_embeds (mindspore.Tensor or None): Tensor of shape (batch_size, sequence_length, embedding_size)
                containing the input embeddings. If not None and past_key_values is None, the model_inputs
                dictionary is updated with 'inputs_embeds' key.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the model inputs. The dictionary includes the following keys:

                - 'input_ids': Tensor of shape (batch_size, sequence_length) containing the input IDs.
                - 'position_ids': Tensor of shape (batch_size, sequence_length) containing the position IDs.
                If attention_mask is not None and position_ids is None, the position_ids are calculated based on the
                attention_mask.
                - 'past_key_values': Cache object or tuple of tensors containing the past key values.
                - 'use_cache': Boolean indicating whether to use cache or not.
                - 'attention_mask': Tensor of shape (batch_size, sequence_length) containing the attention mask.

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
        Reorders the cache of past key values based on the given beam index.

        Args:
            past_key_values (tuple): A tuple containing the cache of past key values.
                Each element in the tuple represents the past key values for a specific layer.
            beam_idx (mindspore.Tensor): A tensor containing the beam indices for reordering the cache.

        Returns:
            None.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Starcoder2, LLAMA->STARCODER2
class Starcoder2ForSequenceClassification(Starcoder2PreTrainedModel):
    """
    This class represents a sequence classification model based on Starcoder2 architecture.
    It inherits functionality from Starcoder2PreTrainedModel.
    The class includes methods for initializing the model, getting and setting input embeddings, and forwarding
    the model for sequence classification.
    The 'forward' method takes various input parameters like input_ids, attention_mask, position_ids, etc.,
    and returns the sequence classifier output.
    It supports computing loss based on different problem types such as regression, single-label classification,
    and multi-label classification.
    The class provides flexibility to handle different problem types and batch sizes, ensuring efficient training
    and inference for sequence classification tasks.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the Starcoder2ForSequenceClassification class.

        Args:
            self (Starcoder2ForSequenceClassification): The object instance.
            config: An instance of the Starcoder2Config class containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Starcoder2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the 'Starcoder2ForSequenceClassification' model.

        Args:
            self: An instance of the 'Starcoder2ForSequenceClassification' class.

        Returns:
            None: The method retrieves the input embeddings from the model and does not return any value.

        Raises:
            None.

        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings of the Starcoder2ForSequenceClassification model.

        Args:
            self (Starcoder2ForSequenceClassification): The instance of the Starcoder2ForSequenceClassification class.
            value: The input embeddings to be set for the model.
                It should be compatible with the model's embedding layer.

        Returns:
            None.

        Raises:
            TypeError: If the value provided is not compatible with the model's embedding layer.
            AttributeError: If the model instance does not have the 'embed_tokens' attribute.
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

class Starcoder2ForTokenClassification(Starcoder2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Starcoder2Model(config)
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
    "Starcoder2ForCausalLM",
    "Starcoder2Model",
    "Starcoder2PreTrainedModel",
    "Starcoder2ForSequenceClassification",
    "Starcoder2ForTokenClassification"
]
