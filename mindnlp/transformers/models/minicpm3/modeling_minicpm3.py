# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" MindSpore MiniCPM model."""
import math
import warnings
from typing import List, Optional, Tuple, Union, Dict

import mindspore
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from mindnlp.core.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....common.activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask
)
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS

from .configuration_minicpm3 import MiniCPM3Config



logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniCPM3Config"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: mindspore.Tensor, dtype: mindspore.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.minicpm3.modeling_minicpm3._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: Tuple, dtype: mindspore.dtype, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.minicpm3.modeling_minicpm3._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.minicpm3.modeling_minicpm3.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, past_key_values_length=past_key_values_length
    )

def rms_layernorm(hidden: mindspore.Tensor, weight: mindspore.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(mindspore.float32).pow(2).mean(axis=-1, keep_dims=True)
    hidden = (hidden * ops.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPM3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPM3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(ops.ones((hidden_size,)))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPM3RMSNorm)


class MiniCPM3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (ops.arange(0, self.dim, 2).float() / self.dim))
        self.inv_freq = inv_freq

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, dtype=mindspore.float32
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPM3LinearScalingRotaryEmbedding(MiniCPM3RotaryEmbedding):
    """MiniCPM3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class MiniCPM3DynamicNTKScalingRotaryEmbedding(MiniCPM3RotaryEmbedding):
    """MiniCPM3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (ops.arange(0, self.dim, 2).float() / self.dim))
            self.inv_freq = inv_freq

        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = ops.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)


class MiniCPM3LongRoPE(MiniCPM3RotaryEmbedding):
    """MiniCPM3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, short_factor=None, long_factor=None, original_max_position_embeddings=None):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = (max_position_embeddings /
                 self.original_max_position_embeddings)
        self.scaling_factor = math.sqrt(
                1 + math.log(scale) /
                math.log(self.original_max_position_embeddings))
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        if seq_len > self.original_max_position_embeddings:
            ext_factors = mindspore.Tensor(self.long_factor, dtype=mindspore.float32)
        else:
            ext_factors = mindspore.Tensor(self.short_factor, dtype=mindspore.float32)

        freqs = ops.mul(
            ops.outer(t, 1.0 / ext_factors),
            self.inv_freq.to(dtype)
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), dim=-1)


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
    # cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=mindspore.float32)
    k_fp32 = k.to(dtype=mindspore.float32)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)

class MiniCPM3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = mindspore.ops.silu

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = ops.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = ops.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """
    This is the equivalent of ops.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class MiniCPM3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPM3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.hidden_size // config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        self.q_a_proj = nn.Linear(
            self.hidden_size, config.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = MiniCPM3RMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = MiniCPM3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads
            * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPM3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "linear":
                self.rotary_emb = MiniCPM3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor = self.config.rope_scaling["factor"],
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPM3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor = self.config.rope_scaling["factor"],
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPM3LongRoPE(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor = self.config.rope_scaling["short_factor"],
                    long_factor = self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling["original_max_position_embeddings"]
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).swapaxes(1, 2)

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

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).swapaxes(1, 2)
        q_nope, q_pe = ops.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = ops.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).swapaxes(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .swapaxes(1, 2)
        )

        k_nope, value_states = ops.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = ops.zeros((bsz, self.num_heads, q_len, self.q_head_dim), dtype=q_pe.dtype)
        # query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        # query_states[:, :, :, self.qk_nope_head_dim :] = q_pe
        query_states = ops.cat([q_nope, q_pe], dim=-1)

        key_states = ops.zeros((bsz, self.num_heads, q_len, self.q_head_dim), dtype=k_pe.dtype)
        # key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        # key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        k_pe = ops.broadcast_to(k_pe, (bsz, self.num_heads, q_len, self.qk_rope_head_dim))
        key_states = ops.cat([k_nope, k_pe], dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_weights = (
            ops.matmul(query_states, key_states.swapaxes(2, 3)) * self.softmax_scale
        )

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
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=mindspore.float32
        ).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value




MiniCPM3_ATTENTION_CLASSES = {
    "eager": MiniCPM3Attention,
}


class MiniCPM3DecoderLayer(nn.Module):
    def __init__(self, config: MiniCPM3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniCPM3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MiniCPM3MLP(config)
        self.input_layernorm = MiniCPM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniCPM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

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

        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MiniCPM3PreTrainedModel(PreTrainedModel):
    """
    Represents a pre-trained mini version of CPM (Code-PM) model for various NLP tasks.
    This class inherits from PreTrainedModel and provides functionality to initialize weights for different types
    of cells.

    The _init_weights method initializes the weights of the given cell based on the specified configuration.
    It sets the weights using either a normal distribution with the specified standard deviation or zeros for bias,
    depending on the type of the cell. For Dense cells, it initializes both weights and biases, while for Embedding cells,
    it initializes weights with random values and sets a specific padding index to zero if provided.

    Parameters:
        cell: The cell for which weights need to be initialized.

    Returns:
        None
    """
    config_class = MiniCPM3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPM3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, cell):
        """
        Initializes the weights of the given cell.

        Args:
            self (MiniCPM3PreTrainedModel): The instance of the MiniCPM3PreTrainedModel class.
            cell: The cell whose weights need to be initialized.

        Returns:
            None. This method initializes the weights of the cell in-place.

        Raises:
            None.
        """
        std = self.config.initializer_range
        if isinstance(cell, nn.Linear):
            cell.weight.set_data(initializer(Normal(std), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            nn.init.normal_(cell.weight, mean=0.0, std=std)
            if cell.padding_idx is not None:
                cell.weight[cell.padding_idx] = 0


class MiniCPM3Model(MiniCPM3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMDecoderLayer`]
    Args:
        config: MiniCPM3Config
    """

    def __init__(self, config: MiniCPM3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MiniCPM3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = MiniCPM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

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
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb

        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
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
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPM3ForCausalLM(MiniCPM3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPM3Model(config)
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
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, MiniCPM3ForCausalLM
        >>> model = MiniCPM3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="ms")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = ops.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states / (self.config.hidden_size / self.config.dim_model_base))
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
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
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
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
            # position_ids = attention_mask.int().cumsum(-1) - 1
            position_ids = ops.cumsum(attention_mask.int(), -1) - 1
            position_ids.masked_fill(attention_mask == 0, 1)
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
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 4096, num_beams=1, do_sample=True, top_p=0.8, temperature=0.3, logits_processor=None,
             **kwargs):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        else:
            gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                        "temperature": temperature, "logits_processor": logits_processor, **kwargs}

        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(history_str, return_tensors='ms')
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        history.append({"role": "assistant", "content": response})
        return response, history



class MiniCPM3ForSequenceClassification(MiniCPM3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPM3Model(config)
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
                sequence_lengths = (ops.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1)
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int32, mindspore.int64):
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
    "MiniCPM3Model",
    "MiniCPM3ForCausalLM",
    "MiniCPM3ForSequenceClassification",
    "MiniCPM3PreTrainedModel",
]
