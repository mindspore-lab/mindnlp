# coding=utf-8
# Copyright 2024 JetMoE AI and the HuggingFace Inc. team. All rights reserved.
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
"""MindSpore JetMoE model."""

import math
import warnings
from typing import List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from...modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    dataclass,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging, get_default_dtype
from .configuration_jetmoe import JetMoEConfig
from .utils import MoE, ParallelExperts


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "jetmoe"
_CONFIG_FOR_DOC = "JetMoEConfig"


@dataclass
class JetMoEBaseModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True`
            is passed or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    aux_loss: Optional[mindspore.Tensor] = None


@dataclass
class JetMoECausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
            or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    aux_loss: Optional[mindspore.Tensor] = None


@dataclass
class JetMoESequenceClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`mindspore.Tensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
            or when `config.use_cache=True`):
            Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    aux_loss: Optional[mindspore.Tensor] = None


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    This function retrieves unpadded data from the input attention_mask.
    
    Args:
        attention_mask (Tensor): A 2D tensor representing the attention mask for the batch.
            It is used to determine the sequence lengths in the batch.
    
    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): 1D tensor containing the indices of non-zero elements in the flattened attention_mask.
            - cu_seqlens (Tensor): 1D tensor representing the cumulative sum of sequence lengths in the batch.
            - max_seqlen_in_batch (int): The maximum sequence length in the batch.

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


class JetMoERMSNorm(nn.Cell):

    """
    The 'JetMoERMSNorm' class is a custom implementation of the root mean square normalization (RMSNorm) module,
    specifically designed for the JetMoE model. It inherits from the 'nn.Cell' class, which is a base class for
    all neural network modules in MindSpore.

    This class provides a trainable normalization layer that performs RMS normalization on the input hidden states.
    The normalization is applied along the last dimension of the input tensor, reducing the variance across
    that dimension.

    The constructor '__init__' initializes the 'JetMoERMSNorm' module.
    It takes two parameters: 'hidden_size' specifies the size of the hidden states, and 'eps' (default value 1e-06)
    is the epsilon value used for numerical stability in the normalization calculation.

    The 'construct' method is the main functionality of the 'JetMoERMSNorm' module. It performs the RMS normalization
    on the input 'hidden_states' tensor. The method first converts the input tensor to 'mindspore.float32' to ensure
    consistent data type for the calculations. It then computes the variance along the last dimension of the tensor
    using the 'pow' and 'mean' operations. Afterward, the input tensor is multiplied element-wise by the reciprocal
    square root of the variance plus epsilon, using the 'rsqrt' and 'ops' operations. Finally, the normalized tensor
    is multiplied element-wise by the weight tensor and converted back to the original input data type.

    Note that the 'JetMoERMSNorm' module is intended to be used as a part of the JetMoE model and can be applied to the
    hidden states of the model's components.

    Please refer to the MindSpore documentation for more information on the 'nn.Cell' class and the 'mindspore.float32'
    data type.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        JetMoERMSNorm module
        """
        super().__init__()
        self.weight = Parameter(initializer('ones', (hidden_size,)))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        Constructs the JetMoERMSNorm.

        This method takes in a tensor of hidden states and performs normalization using the RMSNorm technique.
        The normalized tensor is then multiplied by a weight parameter.

        Args:
            self (JetMoERMSNorm): An instance of the JetMoERMSNorm class.
            hidden_states (Tensor): A tensor containing the hidden states.
                The dtype of the tensor should be compatible with the operations performed within the method.

        Returns:
            None: This method does not return any value.
                The normalization is performed in-place on the hidden_states tensor.

        Raises:
            None.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
class JetMoERotaryEmbedding(nn.Cell):

    """
    The JetMoERotaryEmbedding class represents a rotary position embedding module that can be used in
    neural network models. It inherits from the nn.Cell class and provides functionality for generating rotary
    position embeddings based on the input sequence length.

    Attributes:
        dim (int): The dimension of the position embeddings.
        max_position_embeddings (int): The maximum position embeddings allowed.
        base (int): The base value used in the calculation of position embeddings.
        inv_freq (Tensor): The inverse frequency values used in the calculation of position embeddings.
        max_seq_len_cached (int): The maximum sequence length for which cosine and sine embeddings are cached.
        cos_cached (Tensor): Cached cosine embeddings for the given sequence length.
        sin_cached (Tensor): Cached sine embeddings for the given sequence length.

    Methods:
        _set_cos_sin_cache: Sets the cosine and sine embeddings cache for a given sequence length and data type.
        construct: Constructs the cosine and sine embeddings for the input sequence, updating the cache if necessary.

    Note:
        This class is designed to be used as part of neural network models,
        particularly in scenarios where rotary position embeddings are required.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes the JetMoERotaryEmbedding object with the specified parameters.

        Args:
            self: The object itself.
            dim (int): The dimensionality of the embeddings.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value used in the calculation. Defaults to 10000.

        Returns:
            None.

        Raises:
            ValueError: If the dimensionality 'dim' is not a positive integer.
            ValueError: If 'max_position_embeddings' is not a positive integer.
            ValueError: If 'base' is not a positive integer.
            TypeError: If the data type of 'dim', 'max_position_embeddings', or 'base' is not an integer.
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
        Sets the cosine and sine cache for the JetMoERotaryEmbedding class.

        Args:
            self (JetMoERotaryEmbedding): The instance of the JetMoERotaryEmbedding class.
            seq_len (int): The length of the sequence for which the cosine and sine cache is being set.
            dtype (dtype): The data type for the cache, e.g., float32, float64, etc.

        Returns:
            None.

        Raises:
            TypeError: If seq_len is not an integer or dtype is not a valid data type.
            ValueError: If seq_len is less than 1.
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
        Construct the JetMoERotaryEmbedding.

        Args:
            self (JetMoERotaryEmbedding): The instance of the JetMoERotaryEmbedding class.
            x:
                The input tensor.

                - Type: Any
                - Purpose: The input tensor for which the cos and sin cached values need to be constructed.

                It is expected to be a tensor.
            seq_len:
                The length of the sequence for which the cached values need to be constructed.

                - Type: int
                - Purpose: Determines the length of the sequence for which the cos and sin cached values
                need to be constructed.
                - Restrictions: Should be a positive integer.

        Returns:
            None.

        Raises:
            ValueError: If seq_len is not a positive integer.
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


# copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2):
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


class JetMoEAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    """
    def __init__(self, config: JetMoEConfig, layer_idx: Optional[int] = None):
        """
        Initialize the JetMoEAttention module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.top_k = config.moe_top_k

        self.kv_projection_size = config.kv_channels * config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        assert self.num_heads == self.num_key_value_heads * config.moe_top_k
        self.hidden_size_per_attention_head = config.kv_channels

        self.experts = MoE(
            input_size=config.hidden_size,
            hidden_size=self.kv_projection_size,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            glu=False,
        )

        self.kv_proj = nn.Dense(config.hidden_size, self.kv_projection_size * 2, has_bias=False)

        self.rotary_emb = JetMoERotaryEmbedding(
            config.kv_channels,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        Constructs the JetMoEAttention.

        Args:
            self (JetMoEAttention): The object itself.
            hidden_states (mindspore.Tensor): The input hidden states with shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor], optional): The attention mask tensor with shape
                (batch_size, 1, sequence_length, key_value_sequence_length). Defaults to None.
            position_ids (Optional[mindspore.Tensor], optional): The position ids tensor with shape
                (batch_size, sequence_length). Defaults to None.
            past_key_value (Optional[Cache], optional): The past key-value cache. Defaults to None.
            output_attentions (bool, optional): Whether to return the attention weights. Defaults to False.
            use_cache (bool, optional): Whether to use cache for the key-value pairs. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing the attention output tensor with shape (batch_size, sequence_length, hidden_size),
                the attention weights tensor (if output_attentions is True), and the updated past key-value cache.

        Raises:
            ValueError: If the attention weights or mask have invalid shapes.
            ValueError: If the cache structure has changed and the layer index is not initialized for auto-regressive decoding.
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.shape

        query_states, aux_loss = self.experts.map(hidden_states)
        key_states, value_states = self.kv_proj(hidden_states).chunk(2, axis=-1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.hidden_size_per_attention_head).swapaxes(
            1, 2
        )
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.hidden_size_per_attention_head
        ).swapaxes(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.hidden_size_per_attention_head
        ).swapaxes(1, 2)

        kv_seq_len = key_states.shape[2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids, unsqueeze_dim=1
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = key_states.repeat(1, self.top_k, 1, 1)
        value_states = value_states.repeat(1, self.top_k, 1, 1)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(
            self.hidden_size_per_attention_head
        )

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

        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.hidden_size_per_attention_head):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.hidden_size_per_attention_head)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.top_k, self.kv_projection_size)

        attn_output = self.experts.reduce(attn_output)
        attn_output = attn_output.view(bsz, q_len, -1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, aux_loss

JETMOE_ATTENTION_CLASSES = {
    "eager": JetMoEAttention,
}


class JetMoEBlock(nn.Cell):

    """
    The 'JetMoEBlock' class represents a module that implements a JetMoE block for a neural network model.
    This block consists of components such as self-attention mechanism, layer normalization,
    and a multi-layer perceptron (MLP) with Mixture of Experts (MoE) architecture.
    The block is designed to be used within a larger neural network model for various natural language processing tasks.

    The class provides methods for initialization and forward pass computation.
    During initialization, it sets up the necessary components including input layer normalization,
    self-attention mechanism, post-attention layer normalization, and the MLP with MoE architecture based on the
    provided configuration.

    The 'construct' method performs the forward pass computation of the JetMoEBlock module.
    It takes input hidden states, optional position IDs, past key-value states, attention mask,
    and other optional arguments.
    The method computes the self-attention output, updates the hidden states, applies the MLP operation,
    and returns the final outputs.
    Optional outputs such as attention weights and cached states can also be returned based on the method arguments.

    Overall, the 'JetMoEBlock' class encapsulates the functionality of a JetMoE block within a neural network model,
    providing the necessary components for attention-based computations and expert-based transformations.
    """
    def __init__(self, config: JetMoEConfig, layer_idx: Optional[int] = None):
        """
        Initialize the JetMoEBlock module.

        Args:
            config: Configuration object with model hyperparameters.
        """
        super().__init__()
        self.input_layernorm = JetMoERMSNorm(config.hidden_size)
        self.self_attention = JETMOE_ATTENTION_CLASSES["eager"](config, layer_idx)
        self.post_attention_layernorm = JetMoERMSNorm(config.hidden_size)

        self.mlp = MoE(
            input_size=config.hidden_size,
            hidden_size=config.ffn_hidden_size,
            num_experts=config.moe_num_experts,
            activation=ACT2FN[config.activation_function],
            top_k=config.moe_top_k,
            bias=config.bias,
            glu=config.glu,
        )

    def construct(
        self,
        hidden_states: Optional[mindspore.Tensor],
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
        """
        Forward pass of the JetMoEBlock module.

        Args:
            hidden_states (Optional[mindspore.Tensor]): Input hidden states.
            layer_past (Optional[Tuple[mindspore.Tensor]]): Past layer state.
            attention_mask (Optional[mindspore.Tensor]): Attention mask.
            head_mask (Optional[mindspore.Tensor]): Head mask.
            use_cache (Optional[bool]): Whether to use cached states.
            output_attentions (Optional[bool]): Whether to output attention weights.

        Returns:
            Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
                Tuple containing outputs or optional attention weights.
        """
        # Self Attention
        attn_output, self_attn_weights, present_key_value, att_aux_loss = self.self_attention(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = hidden_states + attn_output
        x_mlp, mlp_aux_loss = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + x_mlp

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (att_aux_loss + mlp_aux_loss,)

        return outputs


class JetMoEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = JetMoEConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["JetMoEBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        """
        Initialize the JetMoEPreTrainedModel.

        Args:
            *inputs: Variable length input arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*inputs, **kwargs)

        self.gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Dense,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.initialize(Normal(self.config.initializer_range))
            if module.bias is not None:
                module.bias.initialize('zeros')
        elif isinstance(module, nn.Embedding):
            module.weight.initialize(Normal(self.config.initializer_range))
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0
        elif isinstance(module, nn.LayerNorm):
            module.bias.initialize('zeros')
            module.weight.initialize('ones')
        elif isinstance(module, ParallelExperts):
            module.weight.initialize(Normal(self.config.initializer_range))


class JetMoEModel(JetMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JetMoEBlock`]

    Args:
        config: JetMoEConfig
    """
    def __init__(self, config: JetMoEConfig):
        """
        Initializes a new instance of the JetMoEModel class.

        Args:
            self: The object itself.
            config (JetMoEConfig):
                The configuration object that contains various settings for the model.

                - 'config' must be an instance of JetMoEConfig.
                - It specifies the configuration parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList([JetMoEBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = JetMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the JetMoEModel.

        Args:
            self : JetMoEModel
                The instance of the JetMoEModel class.

        Returns:
            None
                Returns the input embeddings represented by embed_tokens.

        Raises:
            None
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the JetMoEModel.

        Args:
            self (JetMoEModel): The instance of the JetMoEModel class.
            value (Any): The input embeddings to be set for the model.
                Should be a tensor or an object that can be assigned to self.embed_tokens.

        Returns:
            None: This method does not return any value.

        Raises:
            None
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
        This method constructs the JetMoEModel by processing input data and generating the model output.

        Args:
            self: The instance of the JetMoEModel class.
            input_ids (mindspore.Tensor): The input tensor containing token IDs. Default is None.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor representing the attention mask.
                Default is None.
            position_ids (Optional[mindspore.Tensor]): Optional tensor containing position IDs. Default is None.
            past_key_values (Optional[List[mindspore.Tensor]]): Optional list of tensors representing past key values.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): Optional tensor containing input embeddings. Default is None.
            use_cache (Optional[bool]): Optional flag indicating whether to use cache. Default is None.
            output_attentions (Optional[bool]): Optional flag indicating whether to output attentions
                Default is None.
            output_hidden_states (Optional[bool]): Optional flag indicating whether to output hidden states.
                Default is None.
            return_dict (Optional[bool]): Optional flag indicating whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                The return value is a tuple or an instance of BaseModelOutputWithPast, which contains the model output.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified, or if neither is specified,
                or if incompatible combinations are provided.
            Warning: Raised if `use_cache=True` is incompatible with gradient checkpointing.
            ValueError: Raised if attempting to perform batched generation with certain settings
                that may lead to unexpected behavior.
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
        elif input_ids is not None:
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

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of JetMoE. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        aux_loss = 0
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # hidden_states: Optional[mindspore.Tensor],
            # position_ids: Optional[mindspore.Tensor] = None,
            # past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            # attention_mask: Optional[mindspore.Tensor] = None,
            # output_attentions: Optional[bool] = False,
            # use_cache: Optional[bool] = False,

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    # decoder_layer.__call__,
                    decoder_layer,
                    hidden_states,
                    position_ids,
                    past_key_values,
                    attention_mask,
                    output_attentions,
                    use_cache,
                    use_reentrant=False,
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

            aux_loss += layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return JetMoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            aux_loss=aux_loss,
        )


class JetMoEForCausalLM(JetMoEPreTrainedModel):

    '''
    The JetMoEForCausalLM class represents a JetMoE model for causal language modeling. 
    It inherits from the JetMoEPreTrainedModel.

    This class includes methods for initializing the model, getting and setting input and output embeddings, 
    setting and getting the decoder, constructing the model, preparing inputs for generation, and reordering cache. 
    The construct method handles the generation of outputs based on input and model configuration, 
    while the prepare_inputs_for_generation method prepares inputs for the generation process.
    Additionally, the _reorder_cache method is a static method for reordering past key values based on beam index.

    The class also includes attributes for model configuration, vocabulary size, auxiliary loss coefficient, LM head, 
    and tie_word_embeddings.

    The class provides flexibility for customizing and utilizing the JetMoE model for causal language modeling tasks.
    '''
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes an instance of JetMoEForCausalLM.

        Args:
            self: Instance of the JetMoEForCausalLM class.
            config:
                An object containing configuration parameters for the model.

                - Type: Any
                - Purpose: Contains settings and hyperparameters for the model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None.
        """
        super().__init__(config)
        self.model = JetMoEModel(config)
        self.vocab_size = config.vocab_size
        self.aux_loss_coef = getattr(config, "aux_loss_coef", 0.01)
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.tie_word_embeddings = config.tie_word_embeddings

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the model.

        Args:
            self:
                The instance of the JetMoEForCausalLM class.

                - Type: JetMoEForCausalLM
                - Purpose: Represents the current instance of the JetMoEForCausalLM class.
                - Restrictions: None

        Returns:
            embed_tokens:
                The input embeddings from the model.

                - Type: None
                - Purpose: Represents the embedding tokens used as input for the model.

        Raises:
            None
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        This method sets the input embeddings for the JetMoEForCausalLM model.

        Args:
            self (JetMoEForCausalLM): The instance of the JetMoEForCausalLM class.
            value (torch.Tensor): The input embeddings to be set for the model.
                It should be a tensor of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the JetMoE model for causal language modeling.

        Args:
            self: An instance of the JetMoEForCausalLM class.

        Returns:
            The output embeddings of the JetMoE model for causal language modeling.

        Raises:
            None.

        Note:
            This method is a part of the JetMoEForCausalLM class and can be used to retrieve
            the output embeddings of the model.
            The output embeddings represent the contextualized representations of
            the input tokens generated by the model.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the JetMoEForCausalLM model.

        Args:
            self (JetMoEForCausalLM): The instance of the JetMoEForCausalLM model.
            new_embeddings (Tensor): The new output embeddings to be set for the model.
                Should be a tensor of shape (vocab_size, hidden_size).

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not a valid tensor.
            ValueError: If the new_embeddings tensor shape does not match the expected (vocab_size, hidden_size) shape.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the JetMoEForCausalLM model.

        Args:
            self (JetMoEForCausalLM): An instance of the JetMoEForCausalLM class.
            decoder: The decoder to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Method to retrieve the decoder model for the JetMoEForCausalLM class.

        Args:
            self: JetMoEForCausalLM instance.
                The instance of the JetMoEForCausalLM class.

        Returns:
            None:
                The decoder model associated with the JetMoEForCausalLM instance.

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
            loss = ops.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if labels is not None and self.model.training:
            loss += self.aux_loss_coef * outputs.aux_loss

        return JetMoECausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            aux_loss=outputs.aux_loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self (JetMoEForCausalLM): The instance of the JetMoEForCausalLM class.
            input_ids (torch.Tensor): The input tensor with token IDs.
            past_key_values (Union[Cache, Tuple[torch.Tensor]]):
                The past key values for caching.

                - If Cache instance is provided, information about cache length, past length,
                and max cache length are extracted.
                - If Tuple is provided, the past length is determined as the shape of the first
                dimension of the first element.
            attention_mask (torch.Tensor): The attention mask tensor to mask certain tokens.
            inputs_embeds (torch.Tensor): The embeddings tensor for input tokens.

        Returns:
            model_inputs (Dict[str, Any]): A dictionary containing model inputs for generation.
              It includes 'inputs_embeds' if inputs_embeds is provided, otherwise 'input_ids'.
              Additionally, 'position_ids', 'past_key_values', 'use_cache', and 'attention_mask' are included.

        Raises:
            TypeError: If past_key_values is not of type Cache or Tuple.
            IndexError: If attention_mask shape is inconsistent with input_ids shape.
            ValueError: If cache_length + input_ids length exceeds max_cache_length.
            AttributeError: If position_ids calculation encounters errors.
            RuntimeError: If there are issues with masked_fill operation.
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
        Reorders the cache for the JetMoEForCausalLM model based on the specified beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for the model's cache.
            beam_idx (int): The index of the beam to use for reordering the cache.
                It represents the position of the beam in the cache.

        Returns:
            None: This method modifies the cache in place.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->JetMoE, LLAMA->JETMOE
class JetMoEForSequenceClassification(JetMoEPreTrainedModel):

    """
    JetMoEForSequenceClassification is a class that represents a sequence classification model based on the JetMoE
    architecture. It is designed to handle tasks such as sentiment analysis, text classification,
    and natural language inference.

    This class inherits from the JetMoEPreTrainedModel class, which provides a set of pre-trained parameters
    and methods for fine-tuning the model on specific downstream tasks.

    The JetMoEForSequenceClassification class provides the following methods:

    - __init__: Initializes the JetMoEForSequenceClassification instance with the given configuration.
    - get_input_embeddings: Returns the input embeddings used by the model.
    - set_input_embeddings: Sets the input embeddings of the model to the given value.
    - construct: Constructs the sequence classification model and computes the output logits.
    It takes several optional arguments such as input_ids, attention_mask, and labels,
    and returns a tuple containing the loss, logits, and other outputs.

    The JetMoEForSequenceClassification class follows the configuration provided to initialize the model,
    including the number of labels for the classification task. It utilizes the JetMoEModel for the main
    transformer architecture and applies a score layer to compute the logits.
    The construct method handles the computation of the model's output based on the given inputs and labels,
    including handling different problem types (regression, single-label classification, or multi-label classification)
    and computing the loss.

    Note:
        This docstring does not include the method signatures or any other code for clarity and readability.
    """
    def __init__(self, config):
        """
        Initializes a JetMoEForSequenceClassification instance.

        Args:
            self: The object instance itself.
            config (object): An object containing configuration settings for the model.
                It should include the following attributes:

                - num_labels (int): The number of labels for classification.
                - hidden_size (int): The size of the hidden layers in the model.
                This parameter is used to configure the model and its components.

        Returns:
            None.

        Raises:
            NotImplementedError: If the method 'post_init()' is not implemented.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = JetMoEModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the JetMoEForSequenceClassification model.

        Args:
            self: An instance of the JetMoEForSequenceClassification class.

        Returns:
            None: This method returns the input embeddings which are of type 'None'.

        Raises:
            None
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the JetMoEForSequenceClassification model.

        Args:
            self (JetMoEForSequenceClassification): The instance of the JetMoEForSequenceClassification class.
            value: The input embeddings to be set for the model.
                This should be an object that provides the embedding functionality.

        Returns:
            None.

        Raises:
            None.

        This method allows you to set the input embeddings for the JetMoEForSequenceClassification model.
        The input embeddings should be provided as an object that provides the embedding functionality.
        By setting the input embeddings, you can customize the way the model represents the input data.

        Note:
            The 'embed_tokens' attribute of the 'model' instance is updated with the provided 'value' to
            set the input embeddings.
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
    "JetMoEForCausalLM",
    "JetMoEModel",
    "JetMoEPreTrainedModel",
    "JetMoEForSequenceClassification",
]
