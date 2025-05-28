# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
""" MindNLP DeepSeek model."""
import math
import warnings
from typing import List, Optional, Tuple, Union
import mindspore
from mindnlp.core import nn, ops, no_grad
from mindnlp.core.nn import functional as F
from mindnlp.common.activations import ACT2FN
from mindnlp.transformers.cache_utils import Cache, DynamicCache
from mindnlp.core.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from mindnlp.transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    # _prepare_4d_causal_attention_mask_for_sdpa
)
from mindnlp.transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from mindnlp.transformers.modeling_utils import PreTrainedModel
from mindnlp.transformers.ms_utils import (
    ALL_LAYERNORM_LAYERS,
)

from mindnlp.utils import (
    logging,
)

from .configuration_deepseek import DeepseekConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DeepseekConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def _make_causal_mask(
    input_ids_shape, dtype: mindspore.dtype, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.Deepseek.modeling_Deepseek._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.Deepseek.modeling_Deepseek.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, past_key_values_length=past_key_values_length
    )


class DeepseekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(ops.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = hidden_states.pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(DeepseekRMSNorm)


class DeepseekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (ops.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            dtype=ops.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(
            self.max_seq_len_cached, dtype=self.inv_freq.dtype
        )

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->Deepseek
class DeepseekLinearScalingRotaryEmbedding(DeepseekRotaryEmbedding):
    """DeepseekRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(
            self.max_seq_len_cached, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->Deepseek
class DeepseekDynamicNTKScalingRotaryEmbedding(DeepseekRotaryEmbedding):
    """DeepseekRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (ops.arange(0, self.dim, 2).float() / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = ops.arange(
            self.max_seq_len_cached, dtype=self.inv_freq.dtype
        )

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
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


class DeepseekMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = ops.split(self.gate_proj.weight, slice, dim=0)
            up_proj_slices = ops.split(self.up_proj.weight, slice, dim=0)
            down_proj_slices = ops.split(self.down_proj.weight, slice, dim=1)

            gate_proj = ops.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = ops.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = ops.split((self.act_fn(gate_proj) * up_proj), slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            ops.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states, self.weight, None
        )
        if self.scoring_func == "softmax":
            scores = ops.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        topk_weight, topk_idx = ops.topk(
            scores, k=self.top_k, dim=-1, sorted=False
        )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = ops.zeros(
                    bsz, self.n_routed_experts)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    ops.ones(bsz, seq_len * aux_topk),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class DeepseekMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([DeepseekMLP(config, intermediate_size=config.moe_intermediate_size) for i in
                                      range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekMLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            raise NotImplementedError("Training is not supported yet.")
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # expert_cache = torch.zeros_like(x)
        # idxs = flat_expert_indices.argsort()
        # tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # token_idxs = idxs // self.num_experts_per_tok
        # for i, end_idx in enumerate(tokens_per_expert):
        #     start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
        #     if start_idx == end_idx:
        #         continue
        #     expert = self.experts[i]
        #     exp_token_idx = token_idxs[start_idx:end_idx]
        #     expert_tokens = x[exp_token_idx]
        #     expert_out = expert(expert_tokens)
        #     expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
        #     expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        # return expert_cache
        expert_cache = ops.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out = expert_out.mul(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = mindspore.mint.scatter_add(expert_cache, 0, exp_token_idx.view(-1, 1).tile((1, x.shape[-1])), expert_out)
        return expert_cache

# class AddAuxiliaryLoss(mindnlp.core.autograd.Function):
#     """
#     The trick function of adding auxiliary (aux) loss, 
#     which includes the gradient of the aux loss during backpropagation.
#     """
#     @staticmethod
#     def forward(ctx, x, loss):
#         assert loss.numel() == 1
#         ctx.dtype = loss.dtype
#         ctx.required_aux_loss = loss.requires_grad
#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_loss = None
#         if ctx.required_aux_loss:
#             grad_loss = ops.ones(1, dtype=ctx.dtype)
#         return grad_output, grad_loss


# class DeepseekMoE(nn.Module):
#     '''
#     A mixed expert module containing shared experts.
#     '''
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.num_experts_per_tok = config.num_experts_per_tok
#         if hasattr(config, "ep_size") and config.ep_size > 1:
#             assert config.ep_size == mindspore.mint.distributed.get_world_size()
#             self.ep_size = config.ep_size
#             self.experts_per_rank = config.n_routed_experts // config.ep_size
#             self.ep_rank = mindspore.mint.distributed.get_rank()
#             self.experts = nn.ModuleList(
#                 [
#                     (
#                         DeepseekMLP(
#                             config, intermediate_size=config.moe_intermediate_size
#                         )
#                         if i >= self.ep_rank * self.experts_per_rank
#                            and i < (self.ep_rank + 1) * self.experts_per_rank
#                         else None
#                     )
#                     for i in range(config.n_routed_experts)
#                 ]
#             )

#         else:
#             self.ep_size = 1
#             self.experts_per_rank = config.n_routed_experts
#             self.ep_rank = 0
#             self.experts = nn.ModuleList(
#                 [
#                     DeepseekMLP(
#                         config, intermediate_size=config.moe_intermediate_size
#                     )
#                     for i in range(config.n_routed_experts)
#                 ]
#             )
#         self.gate = MoEGate(config)
#         if config.n_shared_experts is not None:
#             intermediate_size = config.moe_intermediate_size * config.n_shared_experts
#             self.shared_experts = DeepseekMLP(
#                 config=config, intermediate_size=intermediate_size
#             )

#     def forward(self, hidden_states):
#         identity = hidden_states
#         orig_shape = hidden_states.shape
#         topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
#         hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
#         flat_topk_idx = topk_idx.view(-1)
#         if self.training:
#             hidden_states = hidden_states.repeat_interleave(
#             self.num_experts_per_tok, dim=0
#             )
#             y = ops.empty(hidden_states.shape)
#             for i, expert in enumerate(self.experts):
#                 y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
#             y = ops.sum(y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1), dim=1)
#             y = y.to(hidden_states.dtype).view(*orig_shape)
#             # y = AddAuxiliaryLoss.apply(y, aux_loss)
#         else:
#             # y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
#             y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
#         if self.config.n_shared_experts is not None:
#             y = y + self.shared_experts(identity)
#         return y
    
#     # # @mindnlp.core.no_grad()
#     # def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
#     #     expert_cache = ops.zeros_like(x)
#     #     idxs = flat_expert_indices.argsort()
#     #     tokens_per_expert = flat_expert_indices.bincount().cumsum(0)
#     #     token_idxs = idxs // self.num_experts_per_tok
#     #     for i, end_idx in enumerate(tokens_per_expert):
#     #         start_idx = 0 if i == 0 else tokens_per_expert[i-1]
#     #         if start_idx == end_idx:
#     #             continue
#     #         expert = self.experts[i]
#     #         exp_token_idx = token_idxs[start_idx:end_idx]
#     #         expert_tokens = x[exp_token_idx]
#     #         expert_out = expert(expert_tokens)
#     #         expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
#     #         # expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).tile((1, x.shape[-1])), expert_out, reduce='sum')
#     #     return expert_out # expert_cache
#     def moe_infer(self, x, topk_ids, topk_weight):
#         cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
#         cnts.scatter_(1, topk_ids, 1)
#         tokens_per_expert = cnts.sum(dim=0)
#         idxs = topk_ids.view(-1).argsort()
#         sorted_tokens = x[idxs // topk_ids.shape[1]]
#         sorted_tokens_shape = sorted_tokens.shape
#         if self.ep_size > 1:
#             tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
#             tokens_per_expert_group = tokens_per_expert.new_empty(
#                 tokens_per_expert.shape[0]
#             )
#             mindspore.mint.distributed.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
#             output_splits = (
#                 tokens_per_expert_group.view(self.ep_size, -1)
#                 .sum(1)
#                 .cpu()
#                 .numpy()
#                 .tolist()
#             )
#             gathered_tokens = sorted_tokens.new_empty(
#                 tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
#             )
#             input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
#             mindspore.mint.distributed.all_to_all(
#                 list(gathered_tokens.split(output_splits)),
#                 list(sorted_tokens.split(input_split_sizes)),
#             )
#             tokens_per_expert_post_gather = tokens_per_expert_group.view(
#                 self.ep_size, self.experts_per_rank
#             ).sum(dim=0)
#             gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
#             s = 0
#             for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
#                 gatherd_idxs[s : s + k] = i % self.experts_per_rank
#                 s += k
#             gatherd_idxs = gatherd_idxs.argsort()
#             sorted_tokens = gathered_tokens[gatherd_idxs]
#             tokens_per_expert = tokens_per_expert_post_gather
#         tokens_per_expert = tokens_per_expert.cpu().numpy()
#         outputs = []
#         start_idx = 0
#         for i, num_tokens in enumerate(tokens_per_expert):
#             end_idx = start_idx + num_tokens
#             if num_tokens == 0:
#                 continue
#             expert = self.experts[i + self.ep_rank * self.experts_per_rank]
#             tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
#             expert_out = expert(tokens_for_this_expert)
#             outputs.append(expert_out)
#             start_idx = end_idx

#         outs = ops.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
#         if self.ep_size > 1:
#             new_x = ops.empty_like(outs)
#             new_x[gatherd_idxs] = outs
#             gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
#             mindspore.mint.distributed.all_to_all(
#                 list(gathered_tokens.split(input_split_sizes)),
#                 list(new_x.split(output_splits)),
#             )
#             outs = gathered_tokens

#         new_x = ops.empty_like(outs)
#         new_x[idxs] = outs
#         final_out = (
#             new_x.view(*topk_ids.shape, -1)
#             .type(topk_weight.dtype)
#             .mul_(topk_weight.unsqueeze(dim=-1))
#             .sum(dim=1)
#             .type(new_x.dtype)
#         )
#         return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].broadcast_to((
        batch, num_key_value_heads, n_rep, slen, head_dim
    ))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->Deepseek
class DeepseekAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
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
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):

        return ops.transpose(tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim), 1, 2)


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
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.shape

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, axis=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, axis=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, axis=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = ops.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = ops.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = ops.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = ops.transpose(query_states.view(bsz, q_len, self.num_heads, self.head_dim), 1, 2)
        key_states = ops.transpose(key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)
        value_states = ops.transpose(value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim), 1, 2)

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

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, ops.transpose(key_states, 2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )
        assert attention_mask is not None
        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

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

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = ops.split(attn_output, self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = ops.split(self.o_proj.weight, self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


Deepseek_ATTENTION_CLASSES = {
    "eager": DeepseekAttention,
}


class DeepseekDecoderLayer(nn.Module):
    def __init__(self, config: DeepseekConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Deepseek_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = (
            DeepseekMoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekMLP(config)
        )
        self.input_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]
    ]:
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



class DeepseekPreTrainedModel(PreTrainedModel):
    config_class = DeepseekConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0


class DeepseekModel(DeepseekPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekDecoderLayer`]

    Args:
        config: DeepseekConfig
    """

    def __init__(self, config: DeepseekConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                DeepseekDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
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
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=mindspore.int64,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
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
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekForCausalLM(DeepseekPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from mindnlp.transformers import AutoTokenizer, DeepseekForCausalLM

        >>> model = DeepseekForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        if self.config.pretraining_tp > 1:
            lm_head_slices = ops.split(self.lm_head.weight, self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits)
            loss = loss_fct(shift_logits, shift_labels)

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
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
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
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
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
            position_ids = attention_mask.to(mindspore.int32).cumsum(-1) - 1
            # position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = ops.masked_fill(position_ids, attention_mask == 0, 1)
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
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            )
        return reordered_past


class DeepseekForSequenceClassification(DeepseekPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
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
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            ops.arange(batch_size), sequence_lengths
        ]

        loss = None
        if labels is not None:
            # labels = labels
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype in (mindspore.int32, mindspore.int64)
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
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
    "DeepseekPreTrainedModel",
    "DeepseekModel",
    "DeepseekForCausalLM",
    "DeepseekForSequenceClassification"
]

if __name__ == "__main__":
    import numpy as np
    mindspore.set_context(pynative_synchronize=True)
    from configuration_deepseek import DeepseekConfig
    from mindnlp.transformers.models.auto.configuration_auto import AutoConfig
    config, kwargs = AutoConfig.from_pretrained(
                "/home/zqh/deepseek-moe-16b-chat",
                return_unused_kwargs=True
            )
    print(config)
    config.num_hidden_layers = 2
    config.n_routed_experts = 2
    model = DeepseekForCausalLM(config)

    print('init model')
    input_ids = mindspore.Tensor(np.random.randint(0, 10000, (1, 11)), mindspore.int32)
    attention_mask = mindspore.Tensor(np.ones((1,11)), mindspore.int32)
    position_ids = mindspore.Tensor(np.array([np.arange(0, 11)]), mindspore.int32)
    output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    print(output)
    
    
    
    