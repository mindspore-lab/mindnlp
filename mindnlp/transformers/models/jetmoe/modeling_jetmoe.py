# coding=utf-8
# Copyright 2024 JetMoe AI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch JetMoe model."""

import math
from typing import List, Optional, Tuple, Union

import mindspore
from mindnlp.core import nn, ops, no_grad
from mindnlp.core.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindnlp.core.nn import functional as F

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ....utils import logging
from .configuration_jetmoe import JetMoeConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "jetmoe"
_CONFIG_FOR_DOC = "JetMoeConfig"


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: mindspore.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: mindspore.dtype,
    min_dtype: float,
    cache_position: mindspore.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`mindspore.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`mindspore.dtype`):
            The dtype to use for the 4D attention mask.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`mindspore.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`mindspore.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.ndim == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = ops.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = ops.triu(causal_mask, diagonal=1)
        causal_mask *= ops.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].broadcast_to((batch_size, 1, -1, -1))
        if attention_mask is not None:
            causal_mask = causal_mask.copy()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: mindspore.Tensor, num_experts: mindspore.Tensor = None, top_k=2, attention_mask: Optional[mindspore.Tensor] = None
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`mindspore.Tensor`, Tuple[mindspore.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`mindspore.Tensor`, *optional*):
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
        concatenated_gate_logits = ops.cat(list(gate_logits), dim=0)

    routing_weights = nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = ops.topk(routing_weights, top_k, dim=-1)

    expert_mask = nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = ops.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = ops.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .broadcast_to((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = ops.sum(expert_mask.float() * expert_attention_mask, dim=0) / ops.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .broadcast_to((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = ops.sum(routing_weights * router_per_expert_attention_mask, dim=0) / ops.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = ops.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class JetMoeParallelExperts(nn.Module):
    def __init__(self, num_experts: int, input_size: int, output_size: int) -> None:
        """
        Initialize the JetMoeParallelExperts module.
        The experts weights are stored in [num_experts, output_size, input_size] format. Such that it's comptible with
        many MoE libraries, such as [Megablock](https://github.com/databricks/megablocks) and
        [ScatterMoE](https://github.com/shawntan/scattermoe), as well as the
        [MoE kernel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py)
        used in vllm.

        Args:
            num_experts (int):
                Number of experts.
            input_size (int):
                Size of the input.
            output_size (int):
                Size of the output.
        """
        super().__init__()
        self.weight = nn.Parameter(ops.empty(num_experts, output_size, input_size))
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, inputs, expert_size):
        """
        Forward pass of the JetMoeParallelExperts module.

        Args:
            inputs (Tensor):
                Input tensor.
            expert_size:
                Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = ops.split(inputs, expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        results = ops.cat(output_list, dim=0)
        return results


class JetMoeTopKGating(nn.Module):
    def __init__(self, input_size: int, num_experts: int, top_k: int):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (`int`):
                Size of the input.
            num_experts (`int`):
                Number of experts.
            top_k (`int`):
                Number of top experts to select.
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def forward(self, hidden_states):
        # compute the top_k routing decision
        logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
        top_k_gates = ops.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

        # compute number of input given to each expert
        zeros = ops.zeros(
            [top_k_gates.shape[0], self.num_experts], dtype=top_k_gates.dtype
        )  # [num_tokens, num_experts]
        gates = ops.scatter(zeros, 1, top_k_indices, ops.ones(top_k_indices.shape, zeros.dtype))  # [num_tokens, num_experts]
        expert_size = gates.long().sum(0)  # [num_experts,]
        expert_size = expert_size.tolist()

        # sort and group input tokens according to expert assignment
        top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
        _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
        batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

        # gather the gate values for grouped input tokens
        top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
        batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

        return index_sorted_experts, batch_index, batch_gates, expert_size, logits


class JetMoeMoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: JetMoeConfig):
        super(JetMoeMoE, self).__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.intermediate_size
        self.activation = ACT2FN[config.activation_function]
        self.bias = nn.Parameter(ops.empty(self.input_size))
        self.input_linear = JetMoeParallelExperts(config.num_local_experts, self.input_size, self.hidden_size * 2)
        self.output_linear = JetMoeParallelExperts(config.num_local_experts, self.hidden_size, self.input_size)

        self.router = JetMoeTopKGating(
            input_size=self.input_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(self, layer_input):
        """
        Forward pass of the mixture of experts layer.

        Args:
            layer_input (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Output tensor.
            Tensor:
                Router logits.
        """
        bsz, length, emb_size = layer_input.shape
        layer_input = layer_input.reshape(-1, emb_size)
        _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)

        expert_inputs = layer_input[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        chunked_hidden_states = ops.chunk(hidden_states, 2, dim=-1)
        hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        expert_outputs = self.output_linear(hidden_states, expert_size)

        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = ops.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        layer_output = layer_output + self.bias
        return layer_output, router_logits


class JetMoeMoA(nn.Module):
    """
    A Sparsely gated mixture of attention layer with pairs of query- and output-projections as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: JetMoeConfig):
        super(JetMoeMoA, self).__init__()

        self.num_experts = config.num_local_experts
        self.input_size = config.hidden_size
        self.hidden_size = config.kv_channels * config.num_key_value_heads
        self.top_k = config.num_experts_per_tok
        self.bias = nn.Parameter(ops.empty(self.input_size))

        self.input_linear = JetMoeParallelExperts(self.num_experts, self.input_size, self.hidden_size)
        self.output_linear = JetMoeParallelExperts(self.num_experts, self.hidden_size, self.input_size)

        self.router = JetMoeTopKGating(
            input_size=self.input_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
        )

    def map(self, layer_input):
        """
        Map inputs to attention experts according to routing decision and compute query projection inside each experts.
        """

        # Compute gating topology
        bsz, length, emb_size = layer_input.shape
        layer_input = layer_input.reshape(-1, emb_size)  # [bsz * length, emb_size]
        index_sorted_experts, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)
        topo_info = (index_sorted_experts, batch_index, batch_gates, expert_size)

        # Group inputs according to topology and compute query projection
        expert_inputs = layer_input[batch_index]  # [bsz * length * top_k, emb_size]
        expert_outputs = self.input_linear(expert_inputs, expert_size)  # [bsz * length * top_k, hidden_size]

        # Ungroup queries back to original order
        zeros = ops.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype
        )
        layer_output = zeros.index_add(0, index_sorted_experts, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.top_k, -1)  # [bsz, length, top_k, hidden_size]
        return layer_output, router_logits, topo_info

    def reduce(self, layer_input, topo_info):
        """
        Compute output projection inside each attention experts and merge the outputs of different experts.
        """
        bsz, length, k, hidden_size = layer_input.shape
        layer_input = layer_input.reshape(-1, hidden_size)  # [bsz * length * k, hidden_size]
        index_sorted_experts, batch_index, batch_gates, expert_size = topo_info

        # Group inputs according to topology and compute output projection
        expert_inputs = layer_input[index_sorted_experts]  # [bsz * length * top_k, hidden_size]
        expert_outputs = self.output_linear(expert_inputs, expert_size)  # [bsz * length * top_k, emb_size]

        # Apply gates to attention expert outputs
        expert_outputs = expert_outputs * batch_gates[:, None]

        # Ungroup and merge outputs to original order
        zeros = ops.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        layer_output = layer_output + self.bias
        return layer_output

    def forward(self, layer_input):
        raise NotImplementedError("This module doesn't support call and forward.")


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->JetMoe
class JetMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JetMoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with Gemma->JetMoe
class JetMoeRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2, dtype=mindspore.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().broadcast_to((position_ids.shape[0], -1, 1))
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285

        freqs = ops.transpose((inv_freq_expanded.float() @ position_ids_expanded.float()), 1, 2)
        emb = ops.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), dim=-1)


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


class JetMoeAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    """

    def __init__(self, config: JetMoeConfig, layer_idx: Optional[int] = None):
        """
        Initialize the JetMoeAttention module.

        Args:
            config:
                Configuration object with model hyperparameters.
            layer_idx:
                Index of the layer in the model.
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

        self.top_k = config.num_experts_per_tok
        self.attention_dropout = config.attention_dropout
        self.kv_projection_size = config.kv_channels * config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads
        self.head_dim = config.kv_channels

        self.experts = JetMoeMoA(config)

        self.kv_proj = nn.Linear(config.hidden_size, self.kv_projection_size * 2, bias=False)

        self.rotary_emb = JetMoeRotaryEmbedding(
            config.kv_channels,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states, router_logits, topo_info = self.experts.map(hidden_states)
        key_states, value_states = ops.chunk(self.kv_proj(hidden_states), 2, dim=-1)

        query_states = ops.transpose(query_states.view(bsz, q_len, self.num_heads, self.head_dim), 1, 2)
        key_states = ops.transpose(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)
        value_states = ops.transpose(value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads for top-k attention experts
        key_states = key_states.tile((1, self.top_k, 1, 1))
        value_states = value_states.tile((1, self.top_k, 1, 1))

        attn_weights = ops.matmul(query_states, ops.transpose(key_states, 2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=mindspore.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = ops.transpose(attn_output, 1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.top_k, self.kv_projection_size)

        attn_output = self.experts.reduce(attn_output, topo_info)
        attn_output = attn_output.view(bsz, q_len, -1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, router_logits



JETMOE_ATTENTION_CLASSES = {
    "eager": JetMoeAttention,
}


class JetMoeBlock(nn.Module):
    def __init__(self, config: JetMoeConfig, layer_idx: Optional[int] = None):
        """
        Initialize the JetMoeBlock module.

        Args:
            config:
                Configuration object with model hyperparameters.
        """
        super().__init__()
        self.input_layernorm = JetMoeRMSNorm(config.hidden_size)
        self.self_attention = JETMOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.post_attention_layernorm = JetMoeRMSNorm(config.hidden_size)

        self.mlp = JetMoeMoE(config)

    def forward(
        self,
        hidden_states: Optional[mindspore.Tensor],
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple[mindspore.Tensor], Optional[Tuple[mindspore.Tensor, Tuple[mindspore.Tensor, ...]]]]:
        # Self Attention
        attn_output, self_attn_weights, present_key_value, attn_router_logits = self.self_attention(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = hidden_states + attn_output
        x_mlp, mlp_router_logits = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + x_mlp

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += attn_router_logits, mlp_router_logits

        return outputs


class JetMoePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = JetMoeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["JetMoeBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear,)):
            # Slightly different from Mesh Transformer JAX which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, JetMoeParallelExperts):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, JetMoeMoA):
            nn.init.zeros_(module.bias)
        elif isinstance(module, JetMoeMoE):
            nn.init.zeros_(module.bias)


class JetMoeModel(JetMoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`JetMoeBlock`]

    Args:
        config:
            JetMoeConfig
    """

    def __init__(self, config: JetMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([JetMoeBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        self.norm = JetMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
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

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = ops.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            batch_size = inputs_embeds.shape[0]
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of JetMoe. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
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
                    position_ids,
                    past_key_values,
                    causal_mask,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
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
                all_router_logits += (layer_outputs[-2], layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: mindspore.Tensor,
        input_tensor: mindspore.Tensor,
        cache_position: mindspore.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):


        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        min_dtype = float(ops.finfo(dtype).min)
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, mindspore.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        return causal_mask


class JetMoeForCausalLM(JetMoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = JetMoeModel(config)
        self.vocab_size = config.vocab_size
        self.aux_loss_coef = config.aux_loss_coef
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = config.tie_word_embeddings

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
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
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
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
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.aux_loss_coef * aux_loss

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

    # Copied from transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        output_router_logits=False,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                if 0 not in input_ids.shape:
                    input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.int().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
            }
        )
        return model_inputs


# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->JetMoe, LLAMA->JETMOE
class JetMoeForSequenceClassification(JetMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = JetMoeModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

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
        past_key_values: Optional[Union[Cache, List[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
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
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
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
    "JetMoeForCausalLM",
    "JetMoeModel",
    "JetMoePreTrainedModel",
    "JetMoeForSequenceClassification",
]
