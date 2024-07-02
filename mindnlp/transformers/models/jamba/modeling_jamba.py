# coding=utf-8
# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
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
""" MindSpore Jamba model."""
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging

from .configuration_jamba import JambaConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "JambaConfig"


# Adapted from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
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
            Logits from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
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
        concatenated_gate_logits = ops.cat(
            [layer_gate for layer_gate in gate_logits if layer_gate.shape[1] > 1], axis=0
        )

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
    This function takes in an attention_mask tensor and performs several operations to retrieve un-padded data.
    
    Args:
        attention_mask (Tensor):
            The attention_mask tensor with shape (batch_size, sequence_length) that indicates the padding elements.
    
    Returns:
        indices (Tensor): The indices tensor with shape (num_nonpad_elements,)
            that represents the flattened indices of non-padding elements in the attention_mask.
        cu_seqlens (Tensor): The cu_seqlens tensor with shape (batch_size + 1,)
            that contains the cumulative sum of sequence lengths in the batch, padded with a zero at the beginning.
        max_seqlen_in_batch (int): The maximum sequence length in the batch.
    
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


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Jamba
class JambaRMSNorm(nn.Cell):

    """
    The 'JambaRMSNorm' class represents a layer normalization module equivalent to T5LayerNorm.
    It inherits from nn.Cell and includes methods for initialization and construction.
    The class provides functionality for normalizing input hidden states using the RMS normalization technique,
    with the ability to specify the hidden size and epsilon value for variance stabilization.

    Attributes:
        weight (Parameter): A parameter representing the weight used for normalization.
        variance_epsilon (float): The epsilon value for stabilizing the variance during normalization.

    Methods:
        __init__: Initializes the 'JambaRMSNorm' instance with the specified hidden size and epsilon value.
        construct: Applies RMS normalization to the input hidden states and returns the normalized output.

    Note:
        This class is designed for use in neural network models for natural language processing and other
        deep learning tasks.
    """
    def __init__(self, hidden_size, eps=1e-6):
        """
        JambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        """
        Constructs the JambaRMSNorm layer.

        Args:
            self (JambaRMSNorm): The instance of the JambaRMSNorm class.
            hidden_states (Tensor): The input hidden states. It should be a tensor of any numerical data type.

        Returns:
            Tensor: A tensor representing the output of the JambaRMSNorm layer.

        Raises:
            ValueError: If the input hidden_states tensor is empty or invalid.
            TypeError: If the input hidden_states tensor is not of a numerical data type.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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


# Adapted from transformers.models.mistral.modeling_mistral.MistralAttention with Mistral->Jamba
class JambaAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: JambaConfig, layer_idx: Optional[int] = None):
        '''
        Initializes a new instance of the JambaAttention class.

        Args:
            self: The instance of the JambaAttention class.
            config (JambaConfig): An instance of JambaConfig containing the configuration parameters for the attention layer.
            layer_idx (Optional[int]): The index of the layer. If not provided, it may lead to errors during the
                forward call if caching is used. It is recommended to always provide a layer index when creating
                this class.

        Returns:
            None.

        Raises:
            ValueError: If the `hidden_size` is not divisible by `num_heads`.
        '''
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
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Args:
            self (JambaAttention): The instance of the JambaAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
                It should have a shape compatible with the reshaping operation.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None: This method does not explicitly return a value.
                The reshaped tensor is directly manipulated within the method.

        Raises:
            ValueError: If the dimensions of the input tensor are not compatible with the reshaping operation.
            TypeError: If the input tensor is not of type mindspore.Tensor.
            ValueError: If seq_len or bsz are not positive integers.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

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
        This method constructs the JambaAttention mechanism for processing hidden states.

        Args:
            self: The instance of the JambaAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask with
                shape (batch_size, 1, sequence_length, sequence_length).
            position_ids (Optional[mindspore.Tensor]): An optional tensor containing positional ids with shape
                (batch_size, sequence_length).
            past_key_value (Optional[Cache]):
                An optional cache object for storing key and value states from previous steps.
            output_attentions (bool): A flag indicating whether to output attention weights.
            use_cache (bool): A flag indicating whether to use caching for key and value states.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing:

                - attn_output (mindspore.Tensor): The output tensor after applying the attention mechanism of shape
                (batch_size, sequence_length, hidden_size).
                - attn_weights (Optional[mindspore.Tensor]): The attention weights tensor of shape
                (batch_size, num_heads, sequence_length, sequence_length).
                - past_key_value (Optional[Tuple[mindspore.Tensor]]): The updated key and value states from the current step.

        Raises:
            ValueError: If the attention weights shape or attention mask shape does not match the expected dimensions.
            ValueError: If the shape of the output tensor 'attn_output' does not match the expected shape.
            ValueError: If the cache structure has changed and requires a layer index for auto-regressive decoding.
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

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

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


JAMBA_ATTENTION_CLASSES = {
    "eager": JambaAttention,
}


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    It stores the Key and Value states as a list of tensors, one for each layer.
    The expected shape for each tensor for attention layers is `[batch_size, num_heads, seq_len, head_dim]`.
    For the mamba layers, the `key_cache` represents the convolution state and has a shape of `[batch_size, d_inner, 1, d_conv]`,
    and the `value_cache` represents the ssm state and has a shape of `[batch_size, d_inner, 1, d_state]`. Mamba cache
    shape[2] is a dummy "seqlen" dimension to match the number of attention cache dimensions. For mamba, the cache
    doesn't grow with seqlen so this dimension is always 1.
    """
    def __init__(self) -> None:
        """
        Initializes an instance of the HybridMambaAttentionDynamicCache class.

        Args:
            self: The instance of the class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes an instance of the HybridMambaAttentionDynamicCache class.
            It is called automatically when a new object of this class is created.
            The method sets the attention_layer_idx attribute of the instance to None.

        Example:
            ```python
            >>> cache = HybridMambaAttentionDynamicCache()
            ```
        """
        super().__init__()
        self.attention_layer_idx = None  # used to know which layer has data on seqlen in the cache shape

    def update(
            self,
            key_states: mindspore.Tensor,
            value_states: mindspore.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`mindspore.Tensor`):
                The new key states to cache.
            value_states (`mindspore.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.
                No additional arguments are used in `HybridMambaAttentionDynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if self.attention_layer_idx is None and self._is_attn_layer(key_states, value_states):
            self.attention_layer_idx = layer_idx
        if self.attention_layer_idx is not None and layer_idx == self.attention_layer_idx:
            if hasattr(self, "_seen_tokens"):
                self._seen_tokens += key_states.shape[-2]
            else:
                self.seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                # attention layer - append the new states to the existing cache on the seqlen dimension
                self.key_cache[layer_idx] = ops.cat([self.key_cache[layer_idx], key_states], axis=-2)
                self.value_cache[layer_idx] = ops.cat([self.value_cache[layer_idx], value_states], axis=-2)
            else:
                # mamba layer - replace the cache with the new states
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is not None:
            if len(self.key_cache) <= layer_idx:
                return 0
            if self._is_attn_layer(self.key_cache[layer_idx], self.value_cache[layer_idx]):
                return self.key_cache[layer_idx].shape[-2]
            else:
                warnings.warn(
                    f"Asked to get the sequence length from cache of layer {layer_idx} which is not an attention layer. "
                    f"Ignoring that and using an attention layer cache"
                )
        if self.attention_layer_idx is None or len(self.key_cache) <= self.attention_layer_idx:
            return 0
        return self.key_cache[self.attention_layer_idx].shape[-2]

    @staticmethod
    def _is_attn_layer(key_states: mindspore.Tensor, value_states: mindspore.Tensor):
        """
        This method checks if the key and value states have the same last dimension size, which is crucial for
        attention layers.

        Args:
            key_states (mindspore.Tensor): A tensor representing the key states in the attention mechanism.
                It is expected to have a specific shape for compatibility with the attention layer.
                The last dimension size of the key_states tensor must match the last dimension size of the
                value_states tensor.

            value_states (mindspore.Tensor): A tensor representing the value states in the attention mechanism.
                It is expected to have a specific shape for compatibility with the attention layer.
                The last dimension size of the value_states tensor must match the last dimension size of the
                key_states tensor.

        Returns:
            None: This method does not return any value
                but performs a check on the compatibility of key and value states.

        Raises:
            None.
        """
        return key_states.shape[-1] == value_states.shape[-1]


@dataclass
class MambaCacheParams:

    """
    Represents a set of parameters for configuring the Mamba Cache system.

    This class provides a structure for storing and managing various parameters that are used to customize the
    behavior of the Mamba Cache system.
    """
    seqlen_offset: int = 0
    conv_states: Dict[int, mindspore.Tensor] = field(default_factory=dict)
    ssm_states: Dict[int, mindspore.Tensor] = field(default_factory=dict)


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class JambaMambaMixer(nn.Cell):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """
    def __init__(self, config: JambaConfig, layer_idx):
        """
        Initializes an instance of the JambaMambaMixer class.

        Args:
            self: The instance of the class.
            config (JambaConfig): An instance of the JambaConfig class, containing the configuration settings.
            layer_idx (int): The index of the layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            has_bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            group=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
            pad_mode='pad'
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.apply_inner_layernorms = config.mamba_inner_layernorms

        self.use_fast_kernels = config.use_mamba_kernels

        # projection of the input hidden states
        self.in_proj = nn.Dense(self.hidden_size, self.intermediate_size * 2, has_bias=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Dense(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, has_bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Dense(self.time_step_rank, self.intermediate_size, has_bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = ops.arange(1, self.ssm_state_size + 1, dtype=mindspore.float32)[None, :]
        A = A.expand(self.intermediate_size, -1)

        self.A_log = Parameter(ops.log(A))
        self.D = Parameter(ops.ones(self.intermediate_size))
        self.out_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=self.use_bias)

        if self.apply_inner_layernorms:
            self.dt_layernorm = JambaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
            self.B_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
            self.C_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

    def _apply_layernorms(self, dt, B, C):
        """
        Applies layer normalization to the given inputs.

        Args:
            self (JambaMambaMixer): The instance of JambaMambaMixer class.
            dt (type): The input value representing dt.
            B (type): The input value representing B.
            C (type): The input value representing C.

        Returns:
            None.

        Raises:
            None.
        """
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    # fmt: off
    def slow_forward(self, input_states, cache_params: MambaCacheParams = None):

        """
        Method 'slow_forward' in the class 'JambaMambaMixer'.

        This method performs a forward pass through the JambaMambaMixer model.

        Args:
            self: The instance of the JambaMambaMixer class.
            input_states (torch.Tensor): The input states to be processed by the model.
                Expected shape is (batch_size, seq_len, _).
            cache_params (MambaCacheParams, optional): Parameters used for caching intermediate states.
                Default is None.

        Returns:
            torch.Tensor: The contextualized states generated by the model.

        Raises:
            ValueError: If the input_states shape is incorrect or if cache_params are provided but not in
                the expected format.
            RuntimeError: If there is an issue with caching states during training or inference.
            TypeError: If the data types are not as expected during the computations.
            AssertionError: If there is a logical inconsistency detected during the execution.
        """
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).swapaxes(1, 2) # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, axis=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            if self.training:
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backwards pass
                ssm_state = cache_params.ssm_states[self.layer_idx].copy()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = ops.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = ops.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = ops.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = ops.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size), dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.swapaxes(1, 2))
        time_step, B, C = ops.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], axis=-1
        )
        time_step, B, C = self._apply_layernorms(time_step, B, C)
        discrete_time_step = self.dt_proj(time_step)                                    # [batch, seq_len, intermediate_size]
        discrete_time_step = ops.softplus(discrete_time_step).swapaxes(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -ops.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = ops.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            scan_output = ops.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = ops.stack(scan_outputs, axis=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.swapaxes(1, 2))             # [batch, seq_len, hidden_size]
        return contextualized_states

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
            **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor]]]:

        """Construct method in the JambaMambaMixer class.

        This method constructs the model based on the hidden states and past key value.

        Args:
            self: Instance of the JambaMambaMixer class.
            hidden_states (mindspore.Tensor): Hidden states used as input for the model construction.
            past_key_value (Optional[HybridMambaAttentionDynamicCache]): Optional past key value used for caching.
                Default is None. If provided, cache_params are initialized based on past_key_value, else set to None.

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor]]]:

                - The resulting tensor after processing the hidden states.
                - Updated past key value containing the newly calculated convolution and ssm states.

        Raises:
            None.
        """
        if past_key_value is not None:
            cache_params = MambaCacheParams(
                seqlen_offset=0 if hidden_states.shape[1] > 1 else past_key_value.seen_tokens,
            )
            if len(past_key_value.key_cache) > self.layer_idx:
                # we already have cache for this layer, use it
                # remove the dummy seqlen dim (dim=2)
                cache_params.conv_states[self.layer_idx] = past_key_value.key_cache[self.layer_idx].squeeze(2)
                cache_params.ssm_states[self.layer_idx] = past_key_value.value_cache[self.layer_idx].squeeze(2)
            else:
                # we don't have cache for this layer, initialize it with zeros
                batch_size = hidden_states.shape[0]
                cache_params.conv_states[self.layer_idx] = ops.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.conv_kernel_size,
                    dtype=hidden_states.dtype,
                )
                cache_params.ssm_states[self.layer_idx] = ops.zeros(
                    batch_size,
                    self.intermediate_size,
                    self.ssm_state_size,
                    dtype=hidden_states.dtype,
                )
        else:
            cache_params = None

        res = self.slow_forward(hidden_states, cache_params)

        if past_key_value is not None:
            past_key_value.update(
                # add dummy seqlen dim (dim=2) to match the number of dimensions of the attention cache
                cache_params.conv_states[self.layer_idx].unsqueeze(2),
                cache_params.ssm_states[self.layer_idx].unsqueeze(2),
                self.layer_idx,
            )

        return res, past_key_value


class JambaMLP(nn.Cell):

    """
    JambaMLP represents a multi-layer perceptron (MLP) model used in the Jamba project. It inherits from nn.Cell.

    This class implements the construction and initialization of the JambaMLP model.
    The model consists of three linear layers: gate_proj, down_proj, and up_proj.
    The activation function used in the hidden layer is determined by the hidden_act parameter in the JambaConfig object.

    Attributes:
        ffn_dim (int): The size of the intermediate layer in the MLP.
        hidden_dim (int): The size of the hidden layer in the MLP.
        gate_proj (nn.Dense): The linear layer for the gate projection.
        down_proj (nn.Dense): The linear layer for the down projection.
        up_proj (nn.Dense): The linear layer for the up projection.
        act_fn (function): The activation function used in the hidden layer.

    Methods:
        __init__: Initializes the JambaMLP object with the provided configuration.
        construct: Constructs the MLP model using the provided input.

    Example:
        ```python
        >>> config = JambaConfig(intermediate_size=512, hidden_size=256, hidden_act='relu')
        >>> model = JambaMLP(config)
        >>> output = model.construct(input_data)
        ```
    """
    def __init__(self, config: JambaConfig):

        """
        Initializes an instance of the JambaMLP class.

        Args:
            self: The instance of the class.
            config (JambaConfig):
                The configuration object containing the parameters for the model.

                - config.intermediate_size (int): The dimensionality of the intermediate layer.
                - config.hidden_size (int): The dimensionality of the hidden layer.
                - config.hidden_act (str): The activation function for the hidden layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.gate_proj = nn.Dense(self.hidden_dim, self.ffn_dim, has_bias=False)
        self.down_proj = nn.Dense(self.ffn_dim, self.hidden_dim, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_dim, self.ffn_dim, has_bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x):

        """
        Constructs a new feature representation using the JambaMLP model.

        Args:
            self (JambaMLP): An instance of the JambaMLP class.
                This parameter represents the current instance of the JambaMLP model.
            x (Tensor): The input tensor to be processed.
                This tensor serves as the input to the construction process.

        Returns:
            None: This method does not return any value explicitly but modifies the internal state of the model.

        Raises:
            None.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock with Mistral->Jamba
class JambaSparseMoeBlock(nn.Cell):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """
    def __init__(self, config: JambaConfig, num_experts: int, num_experts_per_tok: int):

        """
        Initializes a JambaSparseMoeBlock object.

        Args:
            self: The object itself.
            config (JambaConfig): An instance of JambaConfig that contains the configuration parameters.
            num_experts (int): The total number of experts in the MoE (Mixture of Experts) block.
            num_experts_per_tok (int): The number of experts to assign per token.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size

        #   these values are decided on runtime depending on the layer index
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            # expert routing
            self.router = nn.Dense(self.hidden_dim, self.num_experts, has_bias=False)
        else:
            self.router = None

        self.experts = nn.CellList([JambaMLP(config) for _ in range(self.num_experts)])

    def construct(self, hidden_states: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:

        '''
        Constructs a JambaSparseMoeBlock.

        Args:
            self: An instance of the JambaSparseMoeBlock class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                It should have a shape of (batch_size, sequence_length, hidden_dim).

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor]: A tuple containing the final hidden states and the router logits.
                The final hidden states have a shape of (batch_size, sequence_length, hidden_dim), and
                the router logits have a shape of (batch_size * sequence_length, 1).

        Raises:
            None.
        '''
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.num_experts == 1:
            # in this case we have a single MLP block and don't need to do any routing
            final_hidden_states = self.experts[0](hidden_states)
            router_logits = ops.ones(
                (batch_size * sequence_length, 1),
                dtype=hidden_states.dtype,
            )
            return final_hidden_states, router_logits

        # in this case we have multiple experts and need to do routing
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = ops.softmax(router_logits, axis=1, dtype=mindspore.float32)
        routing_weights, selected_experts = ops.topk(routing_weights, self.top_k, dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = ops.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = ops.one_hot(selected_experts, self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            nonzero = ops.nonzero(expert_mask[expert_idx])
            idx, top_x = nonzero.tensor_split(2, -1)

            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states = final_hidden_states.index_add(0, top_x.astype(mindspore.int32).reshape(-1), current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class JambaAttentionDecoderLayer(nn.Cell):

    """
    This class represents an attention decoder layer in the Jamba model for natural language processing tasks.
    The layer consists of self-attention mechanism and a mixture of experts block for handling complex patterns in
    the input data. It also includes layer normalization for input and pre-mixture-of-experts processing.
    This class is designed to be used as part of a neural network architecture for sequence-to-sequence tasks.
    """
    def __init__(self, config: JambaConfig, num_experts: int, layer_idx: int):

        """
        Initializes a new instance of the JambaAttentionDecoderLayer class.

        Args:
            self (JambaAttentionDecoderLayer): The current instance of the class.
            config (JambaConfig): The configuration object containing various settings.
            num_experts (int): The number of experts for the attention layer.
            layer_idx (int): The index of the current layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.self_attn = JAMBA_ATTENTION_CLASSES["eager"](config, layer_idx)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = JambaSparseMoeBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:

        '''
        Construct method in the JambaAttentionDecoderLayer class.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (mindspore.Tensor, optional): Attention mask of size `(batch, sequence_length)`.
                Padding elements are indicated by 0.
            position_ids (mindspore.Tensor, optional): Tensor containing the position ids of the input.
            past_key_value (Tuple[mindspore.Tensor], optional): Cached past key and value projection states.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
            output_router_logits (bool, optional): Whether or not to return the logits of all the routers.
                They are useful for computing the router loss and should not be returned during inference.
            use_cache (bool, optional): If set to `True`, `past_key_values` key value states are returned
                and can be used to speed up decoding (see `past_key_values`).

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
                A tuple containing the following:

                - hidden_states (mindspore.Tensor): The output hidden states.
                - self_attn_weights (Optional[mindspore.Tensor]): Attention weights of the self-attention layer.
                Returned if `output_attentions` is set to `True`.
                - present_key_value (Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]):
                Cached key and value projection states. Returned if `use_cache` is set to `True`.
                - router_logits (Optional[mindspore.Tensor]): Logits of all the routers.
                Returned if `output_router_logits` is set to `True`.

            Raises:
                None.
        '''
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # residual connection after attention
        hidden_states = residual + hidden_states

        # Experts
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class JambaMambaDecoderLayer(nn.Cell):

    """
    This class represents a decoder layer for Jamba Mamba model, implementing the logic for processing input sequences
    in a transformer architecture.

    Inherits from the nn.Cell class, this decoder layer consists of components such as JambaMambaMixer,
    JambaSparseMoeBlock, JambaRMSNorm, and implements methods for processing hidden states, attention masks,
    and past key-value states.

    Attributes:
        mamba (JambaMambaMixer): A mixer module for Jamba Mamba processing.
        moe (JambaSparseMoeBlock): A sparse MoE block for handling expert computations.
        input_layernorm (JambaRMSNorm): Layer normalization module for input data.
        pre_moe_layernorm (JambaRMSNorm): Layer normalization module before MoE processing.

    Methods:
        construct:
            Processes the input hidden states through the decoder layer, applying layer normalization, mixer, MoE block,
            and returns the output along with optional tensors like attentions, router logits, and cache values.

        _get_past_seqlen:
            Helper method to calculate the past sequence length based on past key-value states and current sequence length.

    Note:
        The 'construct' method supports various optional arguments for controlling output behavior such as attentions,
        router logits, and cache usage.
        The 'padding_mask' argument is deprecated and will be removed in version 4.37.

    Please refer to the method docstrings for detailed information on parameters and return values.
    """
    def __init__(self, config: JambaConfig, num_experts: int, layer_idx: int):

        """
        Initializes a new instance of the JambaMambaDecoderLayer class.

        Args:
            self: The instance of the class.
            config (JambaConfig): The configuration object containing various settings.
            num_experts (int): The number of experts to be used.
            layer_idx (int): The index of the decoder layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.mamba = JambaMambaMixer(config=config, layer_idx=layer_idx)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = JambaSparseMoeBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            position_ids: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
            output_attentions: Optional[bool] = False,
            output_router_logits: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:

        """
        Construct the JambaMambaDecoderLayer.

        Args:
            self: The instance of the JambaMambaDecoderLayer class.
            hidden_states (mindspore.Tensor): Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (mindspore.Tensor, optional): Attention mask of size `(batch, sequence_length)`
                where padding elements are indicated by 0.
            position_ids (mindspore.Tensor, optional): Not used in this method.
            past_key_value (HybridMambaAttentionDynamicCache, optional): Cached past key and value projection states.
            output_attentions (bool, optional): Whether to return the attentions tensors of all attention layers.
            output_router_logits (bool, optional): Whether to return the logits of all the routers.
            use_cache (bool, optional):
                If set to True, past key value states are returned and can be used to speed up decoding.

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
                The output tensor and optional present key value tuple.

        Raises:
            None
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(mindspore.Tensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value = self.mamba(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
        )
        bs, seqlen, _ = hidden_states.shape
        past_seqlen = self._get_past_seqlen(past_key_value, seqlen)
        num_attention_heads = self.mamba.config.num_attention_heads
        self_attn_weights = ops.zeros(bs, num_attention_heads, seqlen, past_seqlen)

        # residual connection after mamba
        hidden_states = residual + hidden_states

        # Experts
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    def _get_past_seqlen(self, past_key_value, seqlen):

        """
        This method '_get_past_seqlen' is defined in the class 'JambaMambaDecoderLayer' and is used to determine
        the past sequence length based on certain conditions.

        Args:
            self: The instance of the class.
            past_key_value: An object representing past key value. It may be None initially.
                If not None, it should have a method 'get_seq_length()' and an attribute 'attention_layer_idx'.
            seqlen: An integer representing the current sequence length.

        Returns:
            None: The method does not return any value explicitly but calculates and handles past sequence length
                based on the conditions specified in the code.

        Raises:
            None: This method does not raise any exceptions.
        """
        if past_key_value is None:
            return seqlen
        past_seqlen = past_key_value.get_seq_length()
        if past_seqlen == 0:
            return seqlen
        if past_key_value.attention_layer_idx is None:
            return seqlen
        if self.mamba.layer_idx < past_key_value.attention_layer_idx:
            return past_seqlen + 1
        return past_seqlen


# Adapted from transformers.models.mistral.modeling_mistral.MistralPreTrainedModel with Mistral->Jamba
class JambaPreTrainedModel(PreTrainedModel):

    """
    The 'JambaPreTrainedModel' class is a subclass of 'PreTrainedModel' and represents a model that has been pre-trained
    for various tasks in natural language processing. This class provides additional methods for converting cache
    formats between standard and Jamba formats.

    Methods:
        _convert_to_standard_cache:
            Standardizes the format of the cache to match most implementations. This method ensures that the cache has
            the sequence length as the third dimension, even for mamba layers.

        _convert_to_jamba_cache:
            Converts the cache to the format expected by Jamba. This method adds a dummy sequence length dimension with
            size 1 for mamba layers.

    Note:
        - The 'JambaPreTrainedModel' class assumes that the 'PreTrainedModel' class has already been defined and imported.

    Example:
        ```python
        >>> model = JambaPreTrainedModel()
        >>> standard_cache = model._convert_to_standard_cache(past_key_value, batch_size)
        >>> jamba_cache = model._convert_to_jamba_cache(past_key_value)
        ```
    """
    config_class = JambaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):

        """
        Initialize weights for the specified cell in the JambaPreTrainedModel.

        Args:
            self: The instance of the JambaPreTrainedModel class.
            cell: The cell for which weights are to be initialized.
                It can be an instance of nn.Dense, nn.Conv1d, or nn.Embedding.

        Returns:
            None.

        Raises:
            TypeError: If the cell parameter is not an instance of nn.Dense, nn.Conv1d, or nn.Embedding.
            ValueError: If the cell has an unsupported type or configuration.
            AttributeError: If the cell does not have required attributes for weight initialization.
            IndexError: If there is an index error during padding of weights for nn.Embedding.
            ValueError: If there is an error in setting the data for weights or bias.
        """
        std = self.config.initializer_factor
        if isinstance(cell, (nn.Dense, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(std),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


    @staticmethod
    def _convert_to_standard_cache(
            past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]], batch_size: int
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. have the seqlen as the third dim
        also for mamba layers
        """
        attn_layer_index = [k.shape == v.shape for k, v in past_key_value].index(True)
        seqlen = past_key_value[attn_layer_index][0].shape[2]
        standard_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                # expand doesn't use more memory, so it's fine to do it here
                standard_past_key_value += ((k.expand(-1, -1, seqlen, -1), v.expand(-1, -1, seqlen, -1)),)
            else:
                standard_past_key_value += ((k, v),)
        return standard_past_key_value

    @staticmethod
    def _convert_to_jamba_cache(
            past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]],
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Converts the cache to the format expected by Jamba, i.e. dummy seqlen dimesion with size 1 for mamba layers
        """
        jamba_past_key_value = ()
        for k, v in past_key_value:
            if k.shape != v.shape:
                # mamba layer
                jamba_past_key_value += ((k[:, :, :1, :], v[:, :, :1, :]),)
            else:
                jamba_past_key_value += ((k, v),)
        return jamba_past_key_value


# Adapted from transformers.models.mistral.modeling_mistral.MistralModel with MISTRAL->JAMBA, Mistral->Jamba
class JambaModel(JambaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JambaDecoderLayer`]

    Args:
        config: JambaConfig
    """
    def __init__(self, config: JambaConfig):

        """
        Initializes a JambaModel instance with the provided configuration.

        Args:
            self (JambaModel): The instance of the JambaModel class.
            config (JambaConfig): An instance of JambaConfig containing configuration parameters for the model.
                The configuration should include:

                - pad_token_id (int): The index of the padding token.
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layer.
                - num_hidden_layers (int): The total number of hidden layers in the model.
                - attn_layer_offset (int): The offset for the attention layer.
                - attn_layer_period (int): The period for the attention layer.
                - expert_layer_offset (int): The offset for the expert layer.
                - expert_layer_period (int): The period for the expert layer.
                - num_experts (int): The number of experts in the model.

        Returns:
            None.

        Raises:
            ValueError:
                - If at least one layer in the decoder is not an attention layer.
                - If at least one layer in the decoder is not a Mamba layer.
                - If the Mamba state size is equal to the convolution size in the Mamba layer.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # init each model layer, decide if it's mamba/attention and has experts or not
        decoder_layers = []
        for i in range(config.num_hidden_layers):
            is_attn = (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0
            is_expert = (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0

            num_experts = self.config.num_experts if is_expert else 1
            if is_attn:
                decoder_layers.append(JambaAttentionDecoderLayer(config, num_experts=num_experts, layer_idx=i))
            else:
                decoder_layers.append(JambaMambaDecoderLayer(config, num_experts=num_experts, layer_idx=i))

        if not any(isinstance(layer, JambaAttentionDecoderLayer) for layer in decoder_layers):
            raise ValueError("At least one layer in the decoder must be an attention layer")
        self._attn_layer_index = [isinstance(layer, JambaAttentionDecoderLayer) for layer in decoder_layers].index(
            True
        )

        if not any(isinstance(layer, JambaMambaDecoderLayer) for layer in decoder_layers):
            raise ValueError("At least one layer in the decoder must be a Mamba layer")
        self._mamba_layer_index = [isinstance(layer, JambaMambaDecoderLayer) for layer in decoder_layers].index(True)

        if (
                decoder_layers[self._mamba_layer_index].mamba.ssm_state_size
                == decoder_layers[self._mamba_layer_index].mamba.conv_kernel_size
        ):
            raise ValueError("Mamba state size and convolution size must be different")

        self.layers = nn.CellList(decoder_layers)

        self.final_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):

        """
        Method to retrieve the input embeddings from the JambaModel instance.

        Args:
            self (JambaModel): The current instance of the JambaModel class.
                This parameter is required to access the embed_tokens attribute.

        Returns:
            None: This method returns the embed_tokens attribute of the JambaModel instance.
                The embed_tokens attribute represents the input embeddings used by the model.

        Raises:
            This method does not raise any exceptions.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):

        """
        Set the input embeddings for the JambaModel.

        Args:
            self (JambaModel): The instance of the JambaModel class.
            value: The input embeddings to be set for the model.
               This can be a tensor, array, or any compatible data structure.
               It represents the embeddings to be assigned to the model for input processing.

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
            past_key_values: Optional[Union[List[mindspore.Tensor], HybridMambaAttentionDynamicCache]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:

        """
        This method 'construct' in the class 'JambaModel' constructs the model by processing input data through
        the layers of the model.

        Args:
            self: The instance of the class.
            input_ids (mindspore.Tensor): The input tensor containing token ids for the input sequence.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor specifying the attention mask to be applied.
            position_ids (Optional[mindspore.Tensor]): An optional tensor containing positional ids for the input sequence.
            past_key_values (Optional[Union[List[mindspore.Tensor], HybridMambaAttentionDynamicCache]]):
                Optional past key values for attention mechanism.
            inputs_embeds (Optional[mindspore.Tensor]): Optional tensor containing input embeddings.
            use_cache (Optional[bool]): Optional boolean specifying whether to use cache for the model.
            output_attentions (Optional[bool]): Optional boolean specifying whether to output attentions.
            output_hidden_states (Optional[bool]): Optional boolean specifying whether to output hidden states.
            output_router_logits (Optional[bool]): Optional boolean specifying whether to output router logits.
            return_dict (Optional[bool]): Optional boolean specifying whether to return the output as a dictionary.

        Returns:
            Union[Tuple, MoeModelOutputWithPast]: The return value can either be a tuple containing relevant outputs
                or an instance of MoeModelOutputWithPast class.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified at the same time,
                or if neither input_ids nor inputs_embeds are specified.
            Warning: Raised as a warning if 'use_cache=True' is incompatible with gradient checkpointing,
                and it automatically sets 'use_cache=False'.
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
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
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
            if isinstance(past_key_values, Cache) and not isinstance(
                    past_key_values, HybridMambaAttentionDynamicCache
            ):
                past_key_values = HybridMambaAttentionDynamicCache.from_legacy_cache(past_key_values.to_legacy_cache())
            use_legacy_cache = not isinstance(past_key_values, HybridMambaAttentionDynamicCache)
            if use_legacy_cache:
                past_key_values = HybridMambaAttentionDynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length, self._attn_layer_index)

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

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.final_layernorm(hidden_states)

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


# Adapted from transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM with MIXTRAL->JAMBA, Mixtral->Jamba
class JambaForCausalLM(JambaPreTrainedModel):

    """
    This class represents a Jamba model for causal language modeling tasks. It is a subclass of JambaPreTrainedModel.

    The JambaForCausalLM class encapsulates the architecture and functionality of the Jamba model for generating text.
    It includes methods for initializing the model, getting and setting input and output
    embeddings, setting the decoder, and constructing the model.

    Attributes:
        model (JambaModel): The Jamba model used for text generation.
        vocab_size (int): The size of the vocabulary.
        lm_head (nn.Dense): The linear layer for generating the next token in the sequence.
        router_aux_loss_coef (float): The coefficient for the auxiliary loss used in load balancing.
        num_experts (int): The number of experts used in load balancing.
        num_experts_per_tok (int): The number of experts per token used in load balancing.

    Methods:
        __init__: Initializes the JambaForCausalLM instance with the given configuration.
        get_input_embeddings: Returns the input embeddings of the model.
        set_input_embeddings: Sets the input embeddings of the model.
        get_output_embeddings: Returns the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model.
        set_decoder: Sets the decoder of the model.
        get_decoder: Returns the decoder of the model.
        construct: Constructs the model for generating text and returns the outputs.
        prepare_inputs_for_generation: Prepares the inputs for text generation by reordering the cache and
            updating the position ids.

    Please refer to the source code for more details on the implementation of each method.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: JambaConfig):

        """
        Initializes an instance of the JambaForCausalLM class.

        Args:
            self: The instance of the class.
            config (JambaConfig): An instance of JambaConfig containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model = JambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):

        """
        This method retrieves the input embeddings from the JambaForCausalLM model.

        Args:
            self: An instance of the JambaForCausalLM class.

        Returns:
            embed_tokens: The method returns the embed_tokens attribute of the model.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):

        """
            Set the input embeddings for the JambaForCausalLM model.

            Args:
                self (JambaForCausalLM): The instance of the JambaForCausalLM class.
                value (object): The input embeddings to be set.

            Returns:
                None.

            Raises:
                None.
            """
        self.model.embed_tokens = value

    def get_output_embeddings(self):

        """
        Returns the output embeddings of the JambaForCausalLM model.

        Args:
            self: An instance of the JambaForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):

        """
        Method to set new output embeddings for the JambaForCausalLM model.

        Args:
            self (JambaForCausalLM): The instance of the JambaForCausalLM class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings for the model.
              This can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):

        """
        Sets the decoder for the JambaForCausalLM model.

        Args:
            self (JambaForCausalLM): The instance of the JambaForCausalLM class.
            decoder: The decoder object to be set for the JambaForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        self.model = decoder

    def get_decoder(self):

        """
        This method returns the decoder model for the JambaForCausalLM class.

        Args:
            self: The instance of the JambaForCausalLM class.

        Returns:
            The decoder model associated with the instance of the JambaForCausalLM class.

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
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            calc_logits_for_entire_prompt: Optional[bool] = True,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            calc_logits_for_entire_prompt (`bool`, *optional*):
                Whether or not to calculate the logits for the entire prompt, or just the last token. Only last token
                logits are needed for generation, and calculating them only for that token can save memory,
                which becomes pretty significant for long sequences.

        Returns:
            Union[Tuple, MoeCausalLMOutputWithPast]
        ```"""
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
        if calc_logits_for_entire_prompt:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -1:, :])
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
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

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
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            output_router_logits=False,
            **kwargs,
    ):

        """
        Prepare inputs for generation in the JambaForCausalLM class.

        Args:
            self (object): The instance of the JambaForCausalLM class.
            input_ids (torch.Tensor): The input token IDs for the generation process.
            past_key_values (Union[Tuple, Cache]): The past key values from previous generations,
                which can be a Tuple or a Cache object.
            attention_mask (torch.Tensor): Optional tensor indicating the attention mask for the input.
            inputs_embeds (torch.Tensor): Optional tensor containing the embeddings for the input tokens.
            output_router_logits (bool): Flag indicating whether to calculate output router logits.

        Returns:
            dict: A dictionary containing the prepared model inputs for generation, including input_ids, position_ids,
                past_key_values, use_cache, attention_mask, output_router_logits, and calc_logits_for_entire_prompt.

        Raises:
            ValueError: If the shape of past_key_values is not as expected.
            AttributeError: If an attribute error occurs during the method execution.
            RuntimeError: If a runtime error occurs during the method execution.
        """
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            # the cache may be in the stardard format (e.g. in contrastive search), convert to Jamba's format if needed
            if isinstance(past_key_values, Tuple):
                if past_key_values[self.model._mamba_layer_index][0].shape[2] > 1:
                    past_key_values = self._convert_to_jamba_cache(past_key_values)

            if isinstance(past_key_values, Cache):
                if not isinstance(past_key_values, HybridMambaAttentionDynamicCache):
                    past_key_values = HybridMambaAttentionDynamicCache.from_legacy_cache(
                        past_key_values.to_legacy_cache()
                    )
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[self.model._attn_layer_index][0].shape[2]
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
            position_ids.masked_fill_(attention_mask == 0, 1)
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
                "output_router_logits": output_router_logits,
                "calc_logits_for_entire_prompt": self.config.calc_logits_for_entire_prompt,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):

        """
        Reorders the cache based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple of past key-value states for each layer in the model.
            beam_idx (torch.Tensor): A tensor containing indices to reorder the past key-value states.

        Returns:
            tuple: A tuple of reordered past key-value states for each layer in the model.

        Raises:
            None
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralForSequenceClassification with Mixtral->Jamba, MIXTRAL->JAMBA
class JambaForSequenceClassification(JambaPreTrainedModel):

    """
    JambaForSequenceClassification is a class that represents a sequence classification model based on the
    Jamba architecture.

    This class extends JambaPreTrainedModel and includes methods for initializing the model,
    getting and setting input embeddings, and constructing the sequence classification output.

    The construct method takes input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels,
    and various optional arguments to generate the sequence classifier output.

    It calculates the loss based on the labels provided and handles different types of classification problems
    such as regression, single-label classification, and multi-label classification.

    The class provides flexibility in handling return types and outputs a SequenceClassifierOutputWithPast object
    if return_dict is set to True.
    """
    def __init__(self, config):

        """
        Initializes a new instance of the JambaForSequenceClassification class.

        Args:
            self: The object itself.
            config: An instance of the JambaConfig class that contains the configuration settings for the Jamba model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = JambaModel(config)
        self.score = nn.Dense(config.hidden_size, self.num_labels, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):

        """
        Method to retrieve input embeddings from the model for JambaForSequenceClassification.

        Args:
            self: The instance of the JambaForSequenceClassification class.

        Returns:
            The input embeddings obtained from the model's embed_tokens attribute.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value):

        """
        Set the input embeddings for the JambaForSequenceClassification model.

        Args:
            self (JambaForSequenceClassification): The instance of the JambaForSequenceClassification class.
            value (torch.Tensor): The input embeddings to be set for the model.
                Should be a torch.Tensor representing the embeddings to be used.

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
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
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
    "JambaModel",
    "JambaForCausalLM",
    "JambaForSequenceClassification"
]
