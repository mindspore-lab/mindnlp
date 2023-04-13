# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" MindNLP LLaMA model."""

import math
from typing import Tuple, Optional

import mindspore
import numpy as np
from mindspore import nn, ops, Parameter, numpy, Tensor
from .llama_hf_config import LlamaConfig
from ..utils.activations import ACT2FN


def _make_causal_mask(
    input_ids_shape, dtype: mindspore.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = numpy.full((tgt_len, tgt_len), Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min, dtype))
    mask_cond = numpy.arange(mask.shape[-1])
    mask = ops.masked_fill(mask, mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
    mask = mask.astype(dtype)
    if past_key_values_length > 0:
        mask = ops.cat([numpy.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)
    return ops.expand(mask[None, None, :, :], Tensor([bsz, 1, tgt_len, tgt_len + past_key_values_length]))

def _expand_mask(mask: Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = ops.expand(mask[:, None, None, :], Tensor([bsz, 1, tgt_len, src_len])).astype(dtype)

    inverted_mask = 1.0 - expanded_mask

    return ops.masked_fill(
        inverted_mask, inverted_mask.astype(mindspore.bool_),
        np.finfo(mindspore.dtype_to_nptype(dtype)).min
    )

class LlamaRMSNorm(nn.Cell):
    '''
    LlamaRMSNorm
    '''
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size, mindspore.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        variance = ops.mean(ops.pow(hidden_states.astype(mindspore.float32), 2), -1, True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16]:
            hidden_states = hidden_states.astype(self.weight.dtype)

        return self.weight * hidden_states

class LlamaRotaryEmbedding(nn.Cell):
    '''
    LlamaRotaryEmbedding
    '''
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2).float() / dim))

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        _t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        freqs = ops.outer(_t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb: Tensor = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            _t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
            freqs = ops.outer(_t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = ops.cat((freqs, freqs), axis=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        print("in construct", seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].astype(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].astype(dtype=x.dtype),
        )

def rotate_half(_x):
    """Rotates half the hidden dims of the input."""
    x_1 = _x[..., : _x.shape[-1] // 2]
    x_2 = _x[..., _x.shape[-1] // 2 :]
    return ops.cat((-x_2, x_1), axis=-1)

def apply_rotary_pos_emb(_q, _k, cos, sin, position_ids):
    '''
    apply_rotary_pos_emb
    '''
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.tile((1, cos.shape[1], 1, cos.shape[3]))
    cos = ops.gather_elements(input=cos.tile((gather_indices.shape[0], 1, 1, 1)), dim=2, index=gather_indices)
    sin = ops.gather_elements(input=sin.tile((gather_indices.shape[0], 1, 1, 1)), dim=2, index=gather_indices)
    q_embed = (_q * cos) + (rotate_half(_q) * sin)
    k_embed = (_k * cos) + (rotate_half(_k) * sin)
    return q_embed, k_embed

class LlamaMLP(nn.Cell):
    '''
    LlamaMLP
    '''
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(intermediate_size, hidden_size, has_bias=False)
        self.up_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def construct(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False)
        self.k_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False)
        self.v_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None, # TODO: LongTensor
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        print(cos.shape, sin.shape)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = ops.maximum(attn_weights,
                                       Tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1).astype(query_states.dtype)
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
    