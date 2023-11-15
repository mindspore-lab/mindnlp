# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import math
import warnings
import numpy as np
from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import log as logger

from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal

from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_utils import PreTrainedModel

from .config_falcon import FalconConfig

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# logger = logging.get_logger(__name__)

FALCON_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tiiuae/falcon-40b",
    "tiiuae/falcon-40b-instruct",
    "tiiuae/falcon-7b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-rw-7b",
    "tiiuae/falcon-rw-1b",
]
_CHECKPOINT_FOR_DOC = "Rocketknight1/falcon-rw-1b"
_CONFIG_FOR_DOC = "FalconConfig"


# NOTE(Hesslow): Unfortunately we did not fuse matmul and bias during training, this means that there's one additional quantization to bfloat16 between the operations.
# In order not to degrade the quality of our HF-port, we keep these characteristics in the final model.
class FalconLinear(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True):
        super(FalconLinear, self).__init__(in_channels, out_channels, has_bias=has_bias)

    def construct(self, input):
        hidden_states = ops.matmul(input, self.weight.T)
        if self.has_bias:
            hidden_states = hidden_states + self.bias
        return hidden_states


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return ops.Concat(-1)((-x2, x1))


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(padding_mask.flatten()).flatten()
    max_seqlen_in_batch = ops.ReduceMax()(seqlens_in_batch).item()
    cu_seqlens = ops.Pad(((1, 0),))(
        ops.CumSum()(seqlens_in_batch, axis=0, exclusive=True)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


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


# TODO (joao): Is this the same implementation as in Llama? If so, let's make them the same and add the copy facilities
class FalconRotaryEmbedding(nn.Cell):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, dim: int, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (self.base ** (ops.arange(0, dim, 2).float() / dim))
        # here inv_freq is a fixed parameter of the model and does not participate in the training update process
        self.inv_freq = mindspore.Parameter(
            inv_freq, name="inv_freq", requires_grad=False
        )

        # mindspore does not support get_default_dtype()
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = mindspore.Tensor(
            np.arange(self.max_seq_len_cached), dtype=self.inv_freq.dtype
        )
        # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        freqs = ops.operations.MatMul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = mindspore.Parameter(
            emb.cos().astype(dtype), name="cos_cached", requires_grad=False
        )
        self.sin_cached = mindspore.Parameter(
            emb.sin().astype(dtype), name="sin_cached", requires_grad=False
        )

    def construct(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].astype(dtype=x.dtype),
            self.sin_cached[:seq_len].astype(dtype=x.dtype),
        )


class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self, dim: int, max_position_embeddings=2048, base=10000, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = mindspore.Tensor(np.arange(seq_len), dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = ops.operations.MatMul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = mindspore.Parameter(
            emb.cos().astype(dtype), name="cos_cached", requires_grad=False
        )
        self.sin_cached = mindspore.Parameter(
            emb.sin().astype(dtype), name="sin_cached", requires_grad=False
        )


class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """
    FalconRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self, dim: int, max_position_embeddings=2048, base=10000, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.head_dim / (self.head_dim - 2))
            inv_freq = 1.0 / (
                base ** (ops.arange(0, self.head_dim, 2).float() / self.head_dim)
            )
            self.inv_freq = mindspore.Parameter(
                inv_freq, name="inv_freq", requires_grad=False
            )

        t = ops.arange(seq_len)
        # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        freqs = ops.operations.MatMul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = mindspore.Parameter(
            emb.cos().astype(dtype), name="cos_cached", requires_grad=False
        )
        self.sin_cached = mindspore.Parameter(
            emb.sin().astype(dtype), name="sin_cached", requires_grad=False
        )


def build_alibi_tensor(
    attention_mask: mindspore.Tensor, num_heads: int, dtype: mindspore.dtype
) -> mindspore.Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = mindspore.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=mindspore.float32
    )
    powers = ops.arange(1, 1 + closest_power_of_2, dtype=mindspore.int32)
    slopes = ops.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = mindspore.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=mindspore.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = ops.arange(
            1, 1 + 2 * num_remaining_heads, 2, dtype=mindspore.int32
        )
        slopes = ops.cat([slopes, ops.pow(extra_base, extra_powers)], axis=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(axis=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None]
    alibi.astype(mindspore.bfloat16)
    alibi = alibi * arange_tensor
    return ops.reshape(alibi, (batch_size * num_heads, 1, seq_length)).astype(dtype)


# Copied from transformers.models.bloom.modeling_bloom.dropout_add
def dropout_add(
    x: mindspore.Tensor, residual: mindspore.Tensor, prob: float, training: bool
) -> mindspore.Tensor:
    """
    Dropout add function

    Args:
        x (`mindspore.tensor`, *required*):
            input tensor
        residual (`mindspore.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class FalconAttention(nn.Cell):
    def __init__(self, config: FalconConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        if config.rotary:
            self._init_rope()

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (
                config.num_kv_heads * 2 + config.num_attention_heads
            ) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(
            self.hidden_size, qkv_out_dim, has_bias=config.bias
        )
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(
            self.hidden_size, self.hidden_size, has_bias=config.bias
        )
        self.attention_dropout = nn.Dropout(p=config.attention_dropout)
        self.num_kv_heads = (
            config.num_kv_heads
            if (self.new_decoder_architecture or not self.multi_query)
            else 1
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                base=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _split_heads(
        self, fused_qkv: mindspore.Tensor
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`mindspore.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(
                batch,
                seq_len,
                -1,
                self.num_heads // self.num_kv_heads + 2,
                self.head_dim,
            )
            query = qkv[:, :, :, :-2]
            key = qkv[:, :, :, [-2]]
            value = qkv[:, :, :, [-1]]
            key = ops.broadcast_to(key, query.shape)
            value = ops.broadcast_to(value, query.shape)

            query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
            return query, key, value
        elif not self.multi_query:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(
                batch_size, seq_length, self.num_heads, 3, self.head_dim
            )
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(
                batch_size, seq_length, self.num_heads + 2, self.head_dim
            )
            return (
                fused_qkv[..., :-2, :],
                fused_qkv[..., [-2], :],
                fused_qkv[..., [-1], :],
            )

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`mindspore.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            mindspore.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        alibi: Optional[mindspore.Tensor],
        attention_mask: mindspore.Tensor,
        position_ids: Optional[mindspore.Tensor] = None,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = (
            self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        )
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.swapaxes(1, 2).reshape(
            batch_size, self.num_heads, query_length, self.head_dim
        )
        key_layer = key_layer.swapaxes(1, 2).reshape(
            batch_size,
            num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.swapaxes(1, 2).reshape(
            batch_size, num_kv_heads, query_length, self.head_dim
        )

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, cos, sin, position_ids
            )

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, kv_length, head_dim]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer = ops.concat((past_key, key_layer), axis=-2)
            value_layer = ops.concat((past_value, value_layer), axis=-2)

        kv_length = key_layer.shape[-2]
        present = (key_layer, value_layer) if use_cache else None
        if alibi is None:
            attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
            attention_scores /= math.sqrt(self.head_dim)

            attention_scores = F.softmax(
                attention_scores + attention_mask, axis=-1, dtype=hidden_states.dtype
            )
            attn_output = ops.matmul(attention_scores, value_layer)

            attn_output = attn_output.view(
                batch_size, self.num_heads, query_length, self.head_dim
            )
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(
                batch_size, query_length, self.num_heads * self.head_dim
            )

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, attention_scores
            else:
                return output_tensor, present

        else:
            matmul_result = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(
                batch_size, self.num_heads, query_length, kv_length
            )

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype in [mindspore.float16, mindspore.bfloat16]:
                attention_scores = attention_scores.astype(mindspore.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + alibi.view(
                batch_size, self.num_heads, 1, -1
            )
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(
                attention_logits + attention_mask, axis=-1, dtype=hidden_states.dtype
            )
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(
                batch_size, self.num_heads, query_length, kv_length
            )

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = ops.matmul(attention_probs_reshaped, value_layer)
            context_layer = ops.flatten(context_layer, 0, 1)
            # change view [batch_size, q_length, num_heads * head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            if output_attentions:
                return output_tensor, present, attention_probs
            else:
                return output_tensor, present


class FalconFlashAttention2(FalconAttention):
    """
    Falcon flash attention module. This module inherits from `FalconAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        alibi: Optional[mindspore.Tensor],
        attention_mask: mindspore.Tensor,
        position_ids: Optional[mindspore.Tensor] = None,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = (
            self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        )
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.swapaxes(1, 2).reshape(
            batch_size, self.num_heads, query_length, self.head_dim
        )
        key_layer = key_layer.swapaxes(1, 2).reshape(
            batch_size, num_kv_heads, query_length, self.head_dim
        )
        value_layer = value_layer.swapaxes(1, 2).reshape(
            batch_size, num_kv_heads, query_length, self.head_dim
        )

        kv_seq_len = key_layer.shape[-2]
        if layer_past is not None:
            kv_seq_len += layer_past[0].shape[-2]
        if alibi is None:
            cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer, key_layer, cos, sin, position_ids
            )

        if layer_past is not None and use_cache:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size, self.num_heads, kv_length, head_dim]
            #  - value: [batch_size, self.num_heads, kv_length, head_dim]
            key_layer = ops.concat((past_key, key_layer), axis=-2)
            value_layer = ops.concat((past_value, value_layer), axis=-2)

        past_key_value = (key_layer, value_layer) if use_cache else None

        if alibi is not None:
            raise ValueError("`alibi` is not supported when `use_flash_attn` is True")

        attn_dropout = self.config.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_layer.dtype
        if input_dtype == mindspore.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_layer = query_layer.astype(target_dtype)
            key_layer = key_layer.astype(target_dtype)
            value_layer = value_layer.astype(target_dtype)

        attn_output = self._flash_attention_forward(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            query_length,
            dropout=attn_dropout,
        )

        attn_weights = attn_output.reshape(
            batch_size, query_length, self.num_heads * self.head_dim
        )
        attn_output = self.dense(attn_weights)

        if not output_attentions:
            attn_weights = None

        return attn_output, past_key_value, attn_weights

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`mindspore.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`mindspore.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`mindspore.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`mindspore.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if attention_mask is None:
            return flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

        batch_size = query_states.shape[0]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=self.is_causal,
        )

        return pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = mindspore.arange(
                batch_size + 1, dtype=mindspore.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class FalconMLP(nn.Cell):
    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = FalconLinear(
            hidden_size, 4 * hidden_size, bias=config.bias
        )
        self.act = nn.GELU()
        self.dense_4h_to_h = FalconLinear(
            4 * hidden_size, hidden_size, bias=config.bias
        )
        self.hidden_dropout = config.hidden_dropout

    def construct(self, x: mindspore.Tensor) -> mindspore.Tensor:
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x


class FalconDecoderLayer(nn.Cell):
    def __init__(self, config: FalconConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.self_attention = (
            FalconFlashAttention2(config)
            if getattr(config, "_flash_attn_2_enabled", False)
            else FalconAttention(config)
        )
        self.mlp = FalconMLP(config)
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if config.new_decoder_architecture:
            # The layer norm before self-attention
            self.ln_attn = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # The layer norm before the MLP
            self.ln_mlp = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        else:
            self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            if not config.parallel_attn:
                self.post_attention_layernorm = LayerNorm(
                    hidden_size, eps=config.layer_norm_epsilon
                )

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        alibi: Optional[mindspore.Tensor],
        attention_mask: mindspore.Tensor,
        position_ids: Optional[mindspore.Tensor] = None,
        layer_past: Optional[Tuple[mindspore.Tensor, mindspore.Tensor]] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        if self.config.new_decoder_architecture:
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self attention.
        attn_outputs = self.self_attention(
            attention_layernorm_out,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        attention_output = attn_outputs[0]

        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = dropout_add(
                    attention_output,
                    residual,
                    self.config.attention_dropout,
                    training=self.training,
                )
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        outputs = attn_outputs[1:]

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        output = dropout_add(
            mlp_output, residual, self.config.hidden_dropout, training=self.training
        )

        return (output,) + outputs if use_cache else (output,) + outputs[1:]



class FalconPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FalconConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FalconDecoderLayer"]
    _supports_flash_attn_2 = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Cell):
        """Initialize the weights."""
        if isinstance(module, (nn.Dense, FalconLinear)):
            # 使用正态分布初始化权重
            module.weight.set_data(
                Normal(0.0, self.config.initializer_range)(module.weight.shape)
            )
            if module.bias is not None:
                module.bias.set_data(np.zeros(module.bias.shape, dtype=np.float32))
        elif isinstance(module, nn.Embedding):
            module.weight.set_data(
                Normal(0.0, self.config.initializer_range)(module.weight.shape)
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx, :].set_data(
                    np.zeros(module.weight.shape[1:], dtype=np.float32)
                )
        elif isinstance(module, nn.LayerNorm):
            module.bias.set_data(np.zeros(module.bias.shape, dtype=np.float32))
            module.weight.set_data(np.ones(module.weight.shape, dtype=np.float32))



class FalconModel(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_alibi = config.alibi

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        # Transformer blocks
        self.h = nn.ModuleList([FalconDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: mindspore.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[-2]

        if self.use_alibi:
            mask = (
                mindspore.ones(
                    (batch_size, seq_length + past_key_values_length), device=inputs_embeds.device, dtype=mindspore.long
                )
                if attention_mask is None
                else attention_mask
            )
            alibi = build_alibi_tensor(mask, self.num_heads, dtype=hidden_states.dtype)
        else:
            alibi = None
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = mindspore.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    layer_past,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
        
class FalconForCausalLM(FalconPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: mindspore.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: mindspore.Tensor,
        past_key_values: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # Note: versions of Falcon with alibi do not use position_ids. It is used with RoPE.
        if not self.transformer.use_alibi and attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }


    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class FalconForSequenceClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = FalconModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
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
        elif input_ids is None:
            sequence_lengths = -1
            logger.warning(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        else:
            sequence_lengths = (mindspore.ne(input_ids, self.config.pad_token_id).sum(axis=-1) - 1)
        pooled_logits = logits[mindspore.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in [
                    mindspore.long,
                    mindspore.int,
                ]:
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
                loss = loss_fct(pooled_logits, labels)
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


class FalconForTokenClassification(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = FalconModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor, mindspore.Tensor], ...]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            batch_size, seq_length = labels.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class FalconForQuestionAnswering(FalconPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FalconModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
