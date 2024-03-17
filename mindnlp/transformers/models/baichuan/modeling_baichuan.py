# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=arguments-renamed
# pylint: disable=attribute-defined-outside-init
# pylint: disable=redefined-builtin
"""
MindSpore BaiChuan Model
"""

import math
from typing import List, Optional, Tuple, Union
from queue import Queue
from threading import Thread


import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindspore import dtype as mstype
from mindnlp.utils import logging

from .configuration_baichuan import BaiChuanConfig
from ...generation.utils import GenerationConfig
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

logger = logging.get_logger(__name__)


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return mindspore.tensor([input_tokens])


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(
                self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        return value

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: Union[tuple, list], dtype: mstype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full(
        (tgt_len, tgt_len),
        Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min, dtype),
    )
    mask_cond = ops.arange(mask.shape[-1])
    mask = ops.masked_fill(mask, Tensor(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1)), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.concat(
            [ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1
        )
    return ops.broadcast_to(
        mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length)
    )


def _expand_mask(mask: Tensor, dtype: mstype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = ops.broadcast_to(
        mask[:, None, None, :], (bsz, 1, tgt_len, src_len)
    ).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(mindspore.bool_),
        mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min),
    )

def _get_interleave(n):
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return _get_interleave_power_of_2(closest_power_of_2) + \
            _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill(float("-inf")).astype(t.dtype)

def _gen_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = mindspore.Tensor(_get_interleave(n_head))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * ops.arange(max_pos).unsqueeze(0).unsqueeze(0).broadcast_to(
        (n_head, -1, -1))
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = ops.triu(
        _fill_with_neg_inf(ops.zeros((max_pos, max_pos))), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    """used in training only"""
    _future_mask = ops.triu(
        _fill_with_neg_inf(ops.zeros([maxpos, maxpos])), 1
    )
    _future_mask = _future_mask.unsqueeze(0) + alibi
    _future_mask = _future_mask.to(tensor)
    return _future_mask[:tensor.shape[0] * attn_heads, :maxpos, :maxpos]


class RMSNorm(nn.Cell):
    """
    RMSNorm
    """

    def __init__(self, hidden_size, epsilon=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size), 'weight')
        self.variance_epsilon = epsilon

    def construct(self, hidden_states):
        variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RotaryEmbedding(nn.Cell):
    """
    RotaryEmbedding
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2).float() / dim))

        self.max_seq_len_cached = max_position_embeddings
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = ops.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = ops.cat((freqs, freqs), axis=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary positional embeddings to input queries (q) and keys (k).
    """
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Cell):
    """
    MLP
    """

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


class Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: BaiChuanConfig):
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
        self.W_pack = nn.Dense(self.hidden_size, 3 * self.hidden_size, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        proj = self.W_pack(hidden_states)
        m = nn.Unflatten(-1, (3, self.hidden_size))
        proj = m(proj).unsqueeze(0).swapaxes(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)  # batch_size x source_len x hidden_size
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)  # batch_size x target_len x head_size
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)  # batch_size x source_len x hidden_size

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
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
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
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


class BaiChuanAttention(nn.Cell):
    def __init__(self, config: BaiChuanConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = nn.Dense(self.hidden_size, 3 * self.hidden_size, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:

        bsz, q_len, _ = hidden_states.shape

        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).swapaxes(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1: # inference with cache
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask.astype(attn_weights.dtype)
            attn_weights = ops.maximum(attn_weights, mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))

        attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_output = ops.matmul(attn_weights, value_states)

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DecoderLayer(nn.Cell):
    """
    DecoderLayer
    """

    def __init__(self, config: BaiChuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            hidden_states (`Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(Tensor)`, *optional*): cached past key and value projection states
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

class BaiChuanLayer(nn.Cell):
    def __init__(self, config: BaiChuanConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BaiChuanAttention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BaiChuanPreTrainedModel(PreTrainedModel):
    """
    BaiChuanPreTrainedModel
    """
    config_class = BaiChuanConfig
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer", "BaiChuanLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, cell):
        std = self.config.initializer_range
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(
                sigma=std, mean=0.0), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class BaiChuan7bModel(BaiChuanPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DecoderLayer`]
    Args:
        config: BaiChuanConfig
    """

    def __init__(self, config: BaiChuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
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
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = ops.ones(
                (batch_size, seq_length_with_past), dtype=mindspore.bool_
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # TODO: how checkpoint
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BaiChuan13bModel(BaiChuanPreTrainedModel):
    def __init__(self, config: BaiChuanConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([BaiChuanLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.post_init()
        self.max_cache_pos = config.model_max_length
        self.first_run = True
        self.alibi_mask = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_alibi_mask(self, tensor, seq_length_with_past):
        if self.training:
            slopes = mindspore.Tensor(_get_interleave(self.n_head))
            alibi = slopes.unsqueeze(1).unsqueeze(1) * ops.arange(seq_length_with_past).unsqueeze(0).unsqueeze(0).broadcast_to(
                (self.n_head, -1, -1))
            alibi = alibi.view(self.n_head, 1, seq_length_with_past)
            mask = _buffered_future_mask(tensor, seq_length_with_past, alibi, self.n_head)
        else:
            if self.first_run:
                self.first_run = False
                self.future_mask = _gen_alibi_mask(self.n_head, self.max_cache_pos)
            if seq_length_with_past > self.max_cache_pos:
                self.max_cache_pos = seq_length_with_past
                self.future_mask = _gen_alibi_mask(self.n_head, self.max_cache_pos)
            mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past]
        return mask

    def construct(
            self,
            input_ids: mindspore.Tensor = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds simultaneously")
        if input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.training:
            if self.alibi_mask is None or self.alibi_mask.shape[-1] != seq_length_with_past:
                self.alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
            alibi_mask = self.alibi_mask
        else:
            alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                expanded_mask = attention_mask.to(alibi_mask.dtype)
                expanded_mask = ops.tril(ops.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                                ) * ops.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
            else:
                expanded_mask = attention_mask
            bsz = inputs_embeds.size(0)
            src_len, tgt_len = alibi_mask.shape[-2:]
            expanded_mask = expanded_mask.unsqueeze(1).broadcast_to((bsz, 1, src_len, tgt_len)).to(alibi_mask.dtype)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = inverted_mask.masked_fill(inverted_mask.to(mindspore.bool_), np.finfo(mindspore.dtype_to_nptype(alibi_mask.dtype)).min)
            attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
        else:
            attention_mask = alibi_mask

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BaiChuanForCausalLM(BaiChuanPreTrainedModel):
    """
    BaiChuanForCausalLM
    """

    def __init__(self, config, size=None):
        super().__init__(config)
        if size == '7b':
            self.model = BaiChuan7bModel(config)
        elif size == '13b':
            self.model = BaiChuan13bModel(config)
        else:
            self.model = BaiChuan7bModel(config)
            raise ValueError('BaiChuan model only support 7b and 13b, please check your config.')

        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        set_decoder
        """
        self.model = decoder

    def get_decoder(self):
        """
        get_decoder
        """
        return self.model

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if isinstance(self.model, BaiChuan7bModel):
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
        elif isinstance(self.model, BaiChuan13bModel):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            raise ValueError('BaiChuan model only support 7b and 13b, please check your config.')

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
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

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
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None):
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs={
                                                    "inputs": input_ids,
                                                    "streamer": streamer,
                                                    "generation_config": generation_config
                                                }
            ).start()
            return streamer

        outputs = self.generate(input_ids, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response


__all__ = [
    "BaiChuanPreTrainedModel",
    "BaiChuan7bModel",
    "BaiChuan13bModel",
    "BaiChuanForCausalLM"
]
