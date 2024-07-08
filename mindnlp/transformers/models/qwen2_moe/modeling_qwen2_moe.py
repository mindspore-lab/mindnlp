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
""" MindSpore Qwen2MoE model."""
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging, get_default_dtype
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from .configuration_qwen2_moe import Qwen2MoeConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen1.5-MoE-A2.7B"
_CONFIG_FOR_DOC = "Qwen2MoeConfig"

QWEN2MOE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen1.5-MoE-A2.7B",
    # See all Qwen2 models at https://huggingface.co/models?filter=qwen2
]


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: mindspore.Tensor, num_experts: mindspore.Tensor = None, top_k=2, attention_mask: Optional[mindspore.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in MindSpore.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`mindspore.Tensor`, Tuple[mindspore.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`mindspore.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        concatenated_gate_logits = ops.cat(list(gate_logits), axis=0)

    routing_weights = ops.softmax(concatenated_gate_logits, axis=-1)

    _, selected_experts = ops.topk(routing_weights, top_k, dim=-1)

    expert_mask = ops.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = ops.mean(expert_mask.float(), axis=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = ops.mean(routing_weights, axis=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = ops.sum(expert_mask.float() * expert_attention_mask, dim=0) / ops.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = ops.sum(routing_weights * router_per_expert_attention_mask, dim=0) / ops.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = ops.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    This function takes an attention_mask as input and performs several operations to obtain unpad data. 
    
    Args:
        attention_mask (Tensor): A tensor representing the attention mask.
            This tensor should have dimensions [batch_size, sequence_length].
    
    Returns:
        tuple:
            A tuple containing the following elements:

            - indices (Tensor): A tensor containing the indices of non-zero elements in the attention_mask tensor.
            - cu_seqlens (Tensor): A tensor representing the cumulative sum of sequence lengths in the batch,
            padded with a zero at the beginning. It has dimensions [batch_size + 1].
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


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2Moe
class Qwen2MoeRMSNorm(nn.Cell):

    """
    Qwen2MoeRMSNorm is a custom normalization layer that is equivalent to T5LayerNorm. It inherits from the nn.Cell class.
    
    This normalization layer performs root mean square normalization (RMSNorm) on the input hidden states.
    It is commonly used in neural network architectures, such as T5 models, to improve the training
    efficiency and convergence.

    Parameters:
        hidden_size (int): The size of the hidden states.
        eps (float, optional): A small value added to the variance for numerical stability. Defaults to 1e-06.

    Methods:
        __init__:
            Initializes a new instance of the Qwen2MoeRMSNorm class.

        construct:
            Applies RMSNorm normalization to the input hidden_states.

            Parameters:

            - hidden_states (Tensor): The input hidden states to be normalized.

            Returns:

            - Tensor: The normalized hidden states after applying RMSNorm.

    Example:
        ```python
        >>> # Create a Qwen2MoeRMSNorm instance
        >>> norm_layer = Qwen2MoeRMSNorm(hidden_size=512)
        ...
        >>> # Apply RMSNorm normalization to the input tensor
        >>> input_tensor = ops.randn((batch_size, sequence_length, hidden_size))
        >>> normalized_tensor = norm_layer.construct(input_tensor)
        ...
        >>> # The normalized_tensor now contains the input tensor after applying RMSNorm normalization.
        ```
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        Constructs the Qwen2MoeRMSNorm for the given hidden states.

        Args:
            self: An instance of the Qwen2MoeRMSNorm class.
            hidden_states (Tensor): The input hidden states to normalize. It should be of type 'mindspore.dtype'.

        Returns:
            None.

        Raises:
            None.

        Note:
            - The hidden_states parameter is expected to be a tensor of shape (batch_size, sequence_length, hidden_size).
            - The hidden_states tensor is converted to 'mindspore.float32' type.
            - The variance of the hidden_states tensor is calculated by squaring each element and then taking the mean
            along the last dimension.
            - The hidden_states tensor is then multiplied by the reciprocal square root of the variance plus
            'self.variance_epsilon'.
            - The final result is the element-wise multiplication of the hidden_states tensor with the weight tensor,
            which is then casted back to the input_dtype.

        Example:
            ```python
            >>> qwen = Qwen2MoeRMSNorm()
            >>> hidden_states = mindspore.Tensor(np.random.rand(2, 3, 4), dtype=mindspore.float16)
            >>> qwen.construct(hidden_states)
            ```
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Qwen2Moe
class Qwen2MoeRotaryEmbedding(nn.Cell):

    """
    This class represents a Qwen2MoeRotaryEmbedding, which is a rotary positional embedding used in natural language
    processing tasks. It is a subclass of the nn.Cell class.

    The Qwen2MoeRotaryEmbedding class initializes with the following parameters:

    - dim (int): The dimension of the embedding.
    - max_position_embeddings (int): The maximum number of position embeddings.
    - base (int): The base used in the exponential calculation.

    The class provides the following methods:

    - __init__:
    Initializes the Qwen2MoeRotaryEmbedding instance.

    - _set_cos_sin_cache:
    Sets the cosine and sine cache for the given sequence length and data type.

    - construct:
    Constructs the rotary embedding for the given input tensor and sequence length.

    Note:
        The methods above are inherited from the nn.Cell class.

    Example:
        ```python
        >>> # Create a Qwen2MoeRotaryEmbedding instance
        >>> embedding = Qwen2MoeRotaryEmbedding(dim=512)
        ...
        >>> # Generate rotary embedding for input tensor x
        >>> x = ...  # Input tensor
        >>> seq_len = ...  # Sequence length
        >>> cos_embedding, sin_embedding = embedding.construct(x, seq_len)
        ```
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        Initializes an instance of the Qwen2MoeRotaryEmbedding class.

        Args:
            self: The instance of the class.
            dim (int): The dimension of the embedding.
            max_position_embeddings (int, optional): The maximum number of position embeddings. Defaults to 2048.
            base (int, optional): The base value used in the calculation. Defaults to 10000.

        Returns:
            None.

        Raises:
            None.

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
        This method '_set_cos_sin_cache' is defined within the class 'Qwen2MoeRotaryEmbedding' and is responsible for
        setting up the cosine and sine cache based on the input sequence length and data type.

        Args:
            self: The instance of the class.
            seq_len (int): The length of the sequence for which the cosine and sine cache is to be computed.
            dtype: The data type for the cache values.

        Returns:
            None.

        Raises:
            ValueError: If the sequence length is not a positive integer.
            TypeError: If the data type is not valid or compatible with the expected operations.
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
        Constructs a rotary embedding for the given input sequence.

        Args:
            self (Qwen2MoeRotaryEmbedding): An instance of the Qwen2MoeRotaryEmbedding class.
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, input_size).
            seq_len (int, optional): The length of the input sequence. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If seq_len is greater than the maximum sequence length that is cached.

        This method constructs a rotary embedding for the input sequence. It first checks if the provided seq_len
        is greater than the maximum sequence length that is currently cached. If so, it updates the cosine and sine
        caches by calling the _set_cos_sin_cache method. The cached cosine and sine values are then returned for the
        specified sequence length.

        Note that the returned cosine and sine tensors are converted to the same dtype as the input tensor x.
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


# Modified from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2Moe
class Qwen2MoeMLP(nn.Cell):

    """
    Qwen2MoeMLP represents a multi-layer perceptron (MLP) model with customized projection layers for gating and
    feature transformation.

    The Qwen2MoeMLP class inherits from nn.Cell and is initialized with a configuration and an optional intermediate size.
    The class provides methods to construct and manipulate the MLP model.

    Attributes:
        config: The configuration object used for initializing the MLP.
        hidden_size: The size of the hidden layers in the MLP.
        intermediate_size: The optional intermediate size for the projection layers.
        gate_proj: The projection layer for gating, implemented as a Dense layer with the hidden size and
            intermediate size.
        up_proj: The projection layer for feature transformation, implemented as a Dense layer with the hidden size
            and intermediate size.
        down_proj: The inverse projection layer for feature transformation, implemented as a Dense layer with the
            intermediate size and hidden size.
        act_fn: The activation function used in the MLP model, derived from the configuration's
            hidden activation function.

    Methods:
        construct(x): Constructs the multi-layer perceptron model using the provided input x.
            This method applies the gating, feature transformation, and activation function to the input data.

    Note:
        The Qwen2MoeMLP class assumes the availability of the nn module for neural network operations.
    """
    def __init__(self, config, intermediate_size=None):
        """
        Initializes an instance of the Qwen2MoeMLP class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing various settings and parameters.
            intermediate_size (int, optional): The size of the intermediate layer. Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):
        """
        Constructs a modified multi-layer perceptron in the Qwen2MoeMLP class.

        Args:
            self (Qwen2MoeMLP): An instance of the Qwen2MoeMLP class.
                Represents the object itself.
            x:
                Input data for constructing the modified MLP.

                - Type: Any
                - Purpose: The input data to be processed by the MLP.
                - Restrictions: None

        Returns:
            None:

                - Type: None
                - Purpose: The method modifies the MLP structure within the class instance.

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
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2Attention with Qwen2->Qwen2Moe
class Qwen2MoeAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: Qwen2MoeConfig, layer_idx: Optional[int] = None):
        """
        Initialize a Qwen2MoeAttention instance.

        Args:
            self: The instance of the class.
            config (Qwen2MoeConfig): The configuration object containing model hyperparameters.
            layer_idx (Optional[int]): The index of the layer within the model. Defaults to None if not provided.
                If layer_idx is None, a warning is issued indicating potential issues during forward call
                if caching is used.
                It is recommended to always provide a layer index when creating an instance of this class.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not divisible by num_heads.
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

        self.rotary_emb = Qwen2MoeRotaryEmbedding(
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
        **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """
        This method constructs Qwen2MoeAttention and performs attention mechanism.

        Args:
            self: The instance of the Qwen2MoeAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor], optional): An optional tensor specifying the attention mask of
                shape (batch_size, 1, sequence_length, key_value_sequence_length). Defaults to None.
            position_ids (Optional[mindspore.Tensor], optional): An optional tensor specifying the position ids of
                shape (batch_size, sequence_length). Defaults to None.
            past_key_value (Optional[Cache], optional): An optional cache object for storing key and value states
                from previous steps. Defaults to None.
            output_attentions (bool): A flag indicating whether to output attention weights. Defaults to False.
            use_cache (bool): A flag indicating whether to use cache for storing key and value states. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
            A tuple containing the attention output tensor of shape (batch_size, sequence_length, hidden_size),
            optional attention weights tensor, and optional updated past key value tuple.

        Raises:
            ValueError: If the size of attention weights or attention mask does not match the expected shape.
            ValueError: If the size of the final attention output tensor does not match the expected shape.
            ValueError: If the cache structure has changed since version v4.36 and a layer index is not provided
                for auto-regressive decoding with k/v caching.
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
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


QWEN2MOE_ATTENTION_CLASSES = {
    "eager": Qwen2MoeAttention,
}


class Qwen2MoeSparseMoeBlock(nn.Cell):

    """
    This class represents a sparse mixture-of-experts (MoE) block for the Qwen2 model. It is a subclass of nn.Cell.

    Attributes:
        num_experts (int): The number of experts in the MoE block.
        top_k (int): The number of top experts to select per token.
        norm_topk_prob (bool): Flag indicating whether to normalize the probabilities of the top experts.
        gate (nn.Dense): The gate layer that computes the routing probabilities for the experts.
        experts (nn.CellList): List of expert layers in the MoE block.
        shared_expert (Qwen2MoeMLP): The shared expert layer in the MoE block.
        shared_expert_gate (nn.Dense): The gate layer for the shared expert.

    Methods:
        construct:
            Constructs the MoE block by processing the given hidden states.

    """
    def __init__(self, config):
        """
        Args:
            self (Qwen2MoeSparseMoeBlock): The instance of the Qwen2MoeSparseMoeBlock class.
            config (Config): A configuration object containing various parameters for the Qwen2MoeSparseMoeBlock.

        Returns:
            None.

        Raises:
            ValueError: If the number of experts (config.num_experts) is not a positive integer.
            ValueError: If the top k value (config.num_experts_per_tok) is not a positive integer.
            ValueError: If the normalized top k probability (config.norm_topk_prob) is not in the range [0, 1].
            ValueError: If the hidden size for the gate (config.hidden_size) is not a positive integer.
            ValueError: If the intermediate size for the experts (config.moe_intermediate_size) or shared expert
                (config.shared_expert_intermediate_size) is not a positive integer.
            ValueError: If the number of shared expert gates (1) is not a positive integer.
            TypeError: If the provided configuration object is not of type Config.
            RuntimeError: If there is an issue with initializing the gate or expert models.
        """
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Dense(config.hidden_size, config.num_experts, has_bias=False)
        self.experts = nn.CellList(
            [Qwen2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

        self.shared_expert = Qwen2MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Dense(config.hidden_size, 1, has_bias=False)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs a Qwen2MoeSparseMoeBlock by processing the input hidden_states.

        Args:
            self: Qwen2MoeSparseMoeBlock
                The instance of the Qwen2MoeSparseMoeBlock class.
            hidden_states: mindspore.Tensor
                A tensor representing the hidden states with the shape (batch_size, sequence_length, hidden_dim).

        Returns:
            mindspore.Tensor
                A tensor representing the final hidden states after processing, with the shape
                (batch_size, sequence_length, hidden_dim).

        Raises:
            None
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = ops.softmax(router_logits, axis=1, dtype=mindspore.float32)
        routing_weights, selected_experts = ops.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = ops.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = ops.one_hot(selected_experts, self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            nonezero = ops.nonzero(expert_mask[expert_idx])
            idx, top_x = nonezero.tensor_split(2, -1)
            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states = final_hidden_states.index_add(0, top_x.astype(mindspore.int32).reshape(-1), current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = ops.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class Qwen2MoeDecoderLayer(nn.Cell):

    """
    The `Qwen2MoeDecoderLayer` class represents a single layer of the Qwen2Moe decoder model.
    It is designed to be used in the Qwen2MoeDecoder model to process the input hidden states and generate output
    representations.

    This class inherits from the `nn.Cell` class.

    Attributes:
        hidden_size (int): The size of the hidden state.
        self_attn (Qwen2MoeAttention): The self-attention mechanism used in the layer.
        mlp (Union[Qwen2MoeSparseMoeBlock, Qwen2MoeMLP]): The multi-layer perceptron used in the layer.
        input_layernorm (Qwen2MoeRMSNorm): The layer normalization applied to the input hidden states.
        post_attention_layernorm (Qwen2MoeRMSNorm): The layer normalization applied after the attention mechanism.

    Note:
        - The `hidden_states` argument represents the input to the layer.
        - The `attention_mask` argument is an optional tensor that masks certain positions in the input sequence.
        - The `position_ids` argument is an optional tensor that represents the position IDs of the input hidden states.
        - The `past_key_value` argument is an optional tuple of tensors that caches the past key and value projection states.
        - The `output_attentions` argument is an optional boolean flag indicating whether to return the attention tensors.
        - The `output_router_logits` argument is an optional boolean flag indicating whether to return the logits of the routers.
        - The `use_cache` argument is an optional boolean flag indicating whether to use the cached key value states for decoding.

    Please refer to the source code for more information on the specific implementation details.
    """
    def __init__(self, config: Qwen2MoeConfig, layer_idx: int):
        """
        Initializes a Qwen2MoeDecoderLayer object.

        Args:
            self: The instance of the Qwen2MoeDecoderLayer class.
            config (Qwen2MoeConfig): An object containing configuration settings for the decoder layer.
                It specifies the hidden size, number of experts, decoder sparse step, and intermediate size.
            layer_idx (int): An integer indicating the index of the layer within the decoder.
                It is used to determine the behavior of the layer based on the configuration.

        Returns:
            None.

        Raises:
            KeyError: If the attention class specified in the configuration is not found in QWEN2MOE_ATTENTION_CLASSES.
            ValueError: If the number of experts specified in the configuration is less than or equal to 0.
            TypeError: If the configuration parameters are not of the expected types.
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2MOE_ATTENTION_CLASSES["eager"](config, layer_idx)

        if config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0:
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
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
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class Qwen2MoePreTrainedModel(PreTrainedModel):

    """
    Qwen2MoePreTrainedModel is a Python class that represents a pre-trained model for Qwen2Moe.
    This class inherits from PreTrainedModel and contains methods for initializing weights for different types
    of cells such as Dense and Embedding.

    Methods:
        _init_weights: Initializes the weights for the given cell. If the cell is a Dense type,
            it initializes the weight using a normal distribution with a specified range and initializes the bias to
            zeros if present. If the cell is an Embedding type, it initializes the weight with random values
            within the specified range and handles padding if necessary.

    Parameters:
        cell: The cell for which weights need to be initialized. It can be a nn.Dense or nn.Embedding type.

    Returns:
        None.
    """
    config_class = Qwen2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
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


class Qwen2MoeModel(Qwen2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2MoeDecoderLayer`]

    Args:
        config: Qwen2MoeConfig
    """
    def __init__(self, config: Qwen2MoeConfig):
        """
        Initializes a new instance of the Qwen2MoeModel class.

        Args:
            self: The current object instance.
            config (Qwen2MoeConfig): The configuration object for the model.

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
            [Qwen2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the input embeddings.

        This method takes the 'self' parameter, which refers to the instance of the Qwen2MoeModel class.

        Args:
            self (Qwen2MoeModel): The instance of the Qwen2MoeModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the Qwen2MoeModel.

        Args:
            self (Qwen2MoeModel): The instance of the Qwen2MoeModel class.
            value (Any): The input embeddings to be set.
                This should be a tensor or an object that can be assigned to the `embed_tokens` attribute.

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings for the Qwen2MoeModel by assigning the given value to the
        `embed_tokens` attribute of the instance.
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        """
        Constructs the Qwen2MoeModel.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor, optional): The input tensor representing the token ids. Defaults to None.
            attention_mask (mindspore.Tensor, optional): The tensor representing the attention mask. Defaults to None.
            position_ids (mindspore.Tensor, optional): The tensor representing the position ids. Defaults to None.
            past_key_values (List[mindspore.Tensor], optional): The list of tensors representing past key values.
                Defaults to None.
            inputs_embeds (mindspore.Tensor, optional): The tensor representing the embedded inputs. Defaults to None.
            use_cache (bool, optional): Whether to use cache or not. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions or not. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states or not. Defaults to None.
            output_router_logits (bool, optional): Whether to output router logits or not. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary or not. Defaults to None.

        Returns:
            Union[Tuple, MoeModelOutputWithPast]: The constructed model output.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified at the same time.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            Warning: If use_cache=True is incompatible with gradient checkpointing.

            """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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
        all_router_logits = () if output_router_logits else None
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
                    output_router_logits,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class Qwen2MoeForCausalLM(Qwen2MoePreTrainedModel):

    """
    This class represents a Qwen2Moe model for causal language modeling.
    It is used for generating text based on a given input. The model is initialized with a configuration and consists of
    a Qwen2MoeModel for encoding and a linear layer (lm_head) for decoding. It also includes methods for getting and
    setting the input and output embeddings, setting and getting the decoder, and generating text.

    Attributes:
        `model` (Qwen2MoeModel): The Qwen2MoeModel used for encoding.
        `vocab_size` (int): The size of the vocabulary.
        `lm_head` (nn.Dense): The linear layer used for decoding.
        `router_aux_loss_coef` (float): The coefficient for the auxiliary loss.
        `num_experts` (int): The number of experts.
        `num_experts_per_tok` (int): The number of experts per token.

    Methods:
        `get_input_embeddings`: Returns the input embeddings.
        `set_input_embeddings`: Sets the input embeddings.
        `get_output_embeddings`: Returns the output embeddings.
        `set_output_embeddings`: Sets the output embeddings.
        `set_decoder`: Sets the decoder.
        `get_decoder`: Returns the decoder.
        `construct`: Constructs the model with the given inputs and returns the output logits.
            Optionally computes the masked language modeling loss and the auxiliary loss.
        `prepare_inputs_for_generation`: Prepares the inputs for text generation, taking into account past key values
            and attention mask.
        `_reorder_cache`: Reorders the cache based on the beam index.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes a Qwen2MoeForCausalLM object.

        Args:
            self (Qwen2MoeForCausalLM): The instance of the class.
            config (dict):
                A dictionary containing configuration parameters.

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layer.
                - router_aux_loss_coef (float): Coefficient for router auxiliary loss.
                - num_experts (int): The total number of experts.
                - num_experts_per_tok (int): Number of experts per token.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = Qwen2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings for the Qwen2MoeForCausalLM model.

        Args:
            self (Qwen2MoeForCausalLM): The instance of the Qwen2MoeForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method: set_input_embeddings

        Description:
            This method sets the input embeddings for the Qwen2MoeForCausalLM model.

        Args:
            self (Qwen2MoeForCausalLM): The instance of the Qwen2MoeForCausalLM class.
                This parameter refers to the current instance of the model where the input embeddings will be set.

            value:
                The input embeddings to be set for the model.

                - Type: Any
                - Purpose: The value representing the input embeddings that will be assigned to the model's 
                embed_tokens attribute.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        """Return the output embeddings of the Qwen2MoeForCausalLM model.

        Args:
            self (Qwen2MoeForCausalLM): An instance of the Qwen2MoeForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets new output embeddings for the Qwen2MoeForCausalLM model.

        Args:
            self (Qwen2MoeForCausalLM): The instance of the Qwen2MoeForCausalLM class.
            new_embeddings (object): The new output embeddings to be set for the model. 
                Should be of the desired embedding type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the Qwen2MoeForCausalLM class.

        Args:
            self (Qwen2MoeForCausalLM): An instance of the Qwen2MoeForCausalLM class.
            decoder: The decoder to be set for the Qwen2MoeForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):
        """
        Returns the decoder model used in the Qwen2MoeForCausalLM class.

        Args:
            self: An instance of the Qwen2MoeForCausalLM class.

        Returns:
            None

        Raises:
            None
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""

        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            `Union[Tuple, MoeCausalLMOutputWithPast]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, Qwen2MoeForCausalLM
            ...
            >>> model = Qwen2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
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
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
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
            output_router_logits=output_router_logits,
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

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepares inputs for generation in the Qwen2MoeForCausalLM class.

        Args:
            self: The instance of the Qwen2MoeForCausalLM class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length) containing the input IDs.
            past_key_values (Union[Cache, Tuple[torch.Tensor]]): Optional. The past key values used for caching during generation.
                If past_key_values is an instance of Cache, it represents the cached key values with attributes:

                - cache_length (int): The length of the cache.
                - past_length (int): The length of the past tokens.
                - max_cache_length (Optional[int]): The maximum cache length, if applicable.
                If past_key_values is a tuple, it represents the shape of the past key values tensor.
            attention_mask (torch.Tensor): Optional. The attention mask tensor of shape (batch_size, sequence_length) containing
                the attention mask for the input IDs.
            inputs_embeds (torch.Tensor): Optional. The input embeddings tensor of shape (batch_size, sequence_length, hidden_size)
                containing the input embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            dict:
                A dictionary containing the model inputs for generation with the following keys:

                - 'inputs_embeds' (torch.Tensor): The input embeddings tensor.
                - 'input_ids' (torch.Tensor): The input IDs tensor.
                - 'position_ids' (torch.Tensor): The position IDs tensor.
                - 'past_key_values' (Union[Cache, Tuple[torch.Tensor]]): The past key values tensor.
                - 'use_cache' (Optional[bool]): Indicates whether to use cache during generation.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.

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
        Reorders the cache based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
                Each element in the tuple represents the past key values for a specific layer.
            beam_idx (Tensor): A tensor containing the indices to reorder the cache based on the beam search results.

        Returns:
            None.

        Raises:
            TypeError: If the input past_key_values is not a tuple or if beam_idx is not a tensor.
            ValueError: If the dimensions of the input tensors are incompatible for reordering.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->Qwen2Moe, LLAMA->QWEN2MOE
class Qwen2MoeForSequenceClassification(Qwen2MoePreTrainedModel):

    """
    Qwen2MoeForSequenceClassification is a class that implements a sequence classification model based on the
    Qwen2Moe architecture.
    It inherits from the Qwen2MoePreTrainedModel class and provides methods for initializing the model,
    getting and setting input embeddings, and constructing the model for sequence classification tasks.

    Attributes:
        num_labels (int): Number of labels for classification.
        model (Qwen2MoeModel): The Qwen2MoeModel instance used in the classification model.
        score (nn.Dense): Dense layer for computing the classification scores.

    Methods:
        __init__: Initializes the Qwen2MoeForSequenceClassification instance with the provided configuration.
        get_input_embeddings: Retrieves the input embeddings from the model.
        set_input_embeddings: Sets the input embeddings of the model to the given value.
        construct:
            Constructs the model for sequence classification based on the input parameters.
            Computes the classification loss based on the provided labels and problem type.
            Returns a tuple of loss and output if loss is computed, otherwise returns the model outputs.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the Qwen2MoeForSequenceClassification class.

        Args:
            self: A reference to the current instance of the class.
            config: An instance of the Qwen2MoeConfig class containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Qwen2MoeModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
            This method retrieves the input embeddings from the 'Qwen2MoeForSequenceClassification' model.

        Args:
            self: An instance of the 'Qwen2MoeForSequenceClassification' class.

        Returns:
            None

        Raises:
            None

        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the Qwen2MoeForSequenceClassification model.

        Args:
            self (Qwen2MoeForSequenceClassification): The instance of the Qwen2MoeForSequenceClassification class.
            value: The input embeddings to be set for the model. It should be an object of type torch.nn.Embedding.

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
    "Qwen2MoeForCausalLM",
    "Qwen2MoeModel",
    "Qwen2MoePreTrainedModel",
    "Qwen2MoeForSequenceClassification",
]
