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
import numpy as np
from typing import Optional, Tuple, Union

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import log as logger
# import torch.utils.checkpoint
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal

# from ...modeling_outputs import (
#     BaseModelOutputWithPastAndCrossAttentions,
#     CausalLMOutputWithCrossAttentions,
#     CausalLMOutputWithPast,
#     SequenceClassifierOutputWithPast,
#     TokenClassifierOutput,
# )
from ...modeling_utils import PreTrainedModel
# from ...utils import (
# is_flash_attn_2_available,
# logging,
# )
from .config_falcon import FalconConfig

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

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
        super(FalconLinear, self).__init__(
            in_channels, out_channels, has_bias=has_bias)

    def construct(self, input):
        hidden_states = ops.matmul(input, self.weight.T)
        if self.has_bias:
            hidden_states = hidden_states + self.bias
        return hidden_states


# rotary pos emb helpers (torch.jit.script does not seem to support staticmethod...)
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return ops.Concat(-1)((-x2, x1))


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(axis=-1, dtype=mindspore.int32)
    indices = ops.nonzero(padding_mask.flatten()).flatten()
    max_seqlen_in_batch = ops.ReduceMax()(seqlens_in_batch).item()
    cu_seqlens = ops.Pad(((1, 0),))(ops.CumSum()(
        seqlens_in_batch, axis=0, exclusive=True))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# TODO (joao): Is this the same implementation as in Llama? If so, let's make them the same and add the copy facilities
class FalconRotaryEmbedding(nn.Cell):
    """Implementation of RotaryEmbedding from GPT-NeoX.
    This implementation is designed to operate on queries and keys that are compatible with `[batch_size,
    n_heads_per_partition, seq_len, head_dim]` (e.g. MinGPTAttention format).
    """

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048):
        super().__init__()
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / \
            (self.base ** (ops.arange(0, head_dim, 2).float() / head_dim))
        # hrer inv_freq is a fixed parameter of the model and does not participate in the training update process
        self.inv_freq = mindspore.Parameter(
            inv_freq, name="inv_freq", requires_grad=False)
        self.head_dim = head_dim
        self.seq_len_cached = -1
        self.cos_cached: Union[mindspore.tensor, None] = None
        self.sin_cached: Union[mindspore.tensor, None] = None

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.seq_len_cached = seq_len
        t = mindspore.Tensor(np.arange(seq_len), dtype=self.inv_freq.dtype)
        # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        freqs = ops.operations.MatMul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        emb = ops.cat((freqs, freqs), axis=-1)

        if dtype in [mindspore.float16, mindspore.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

        self.cos_cached = self.cos_cached.astype(dtype)
        self.sin_cached = self.sin_cached.astype(dtype)

    def cos_sin(
        self, seq_len: int, past_key_values_length: int, position_ids: mindspore.tensor, dtype=mindspore.bfloat16
    ) -> mindspore.tensor:
        total_length = seq_len + past_key_values_length
        if total_length > self.seq_len_cached:
            self._set_cos_sin_cache(total_length, dtype)

        # Gather cos, sin at the designated position ids
        cos = self.cos_cached[position_ids]  # [bs, seq_len, dim]
        sin = self.sin_cached[position_ids]  # [bs, seq_len, dim]
        return cos, sin

    def construct(self, query, key, past_key_values_length, position_ids):
        _, seq_len, _ = query.shape
        cos, sin = self.cos_sin(
            seq_len, past_key_values_length, position_ids, query.dtype)
        # Query and key's shapes are [bs * num_heads, seq_len, dim], might need manual expansion. Ifs and elses used to
        # avoid unnecessary repeat_interleave operations.
        query_expansion_factor = int(query.shape[0] / cos.shape[0])
        if query_expansion_factor > 1:
            query_cos = ops.repeat_interleave(
                cos, query_expansion_factor, axis=0)
            query_sin = ops.repeat_interleave(
                sin, query_expansion_factor, axis=0)
        else:
            query_cos, query_sin = cos, sin

        key_expansion_factor = int(key.shape[0] / cos.shape[0])
        if key_expansion_factor > 1:
            if key_expansion_factor != query_expansion_factor:
                key_cos = ops.repeat_interleave(
                    cos, key_expansion_factor, axis=0)
                key_sin = ops.repeat_interleave(
                    sin, key_expansion_factor, axis=0)
            else:
                key_cos, key_sin = query_cos, query_sin
        else:
            key_cos, key_sin = cos, sin

        return (query * query_cos) + (rotate_half(query) * query_sin), (key * key_cos) + (rotate_half(key) * key_sin)


class FalconLinearScalingRotaryEmbedding(FalconRotaryEmbedding):
    """FalconRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(head_dim, base, max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.seq_len_cached = seq_len
        t = mindspore.Tensor(np.arange(seq_len), dtype=self.inv_freq.dtype)
        # This line is the only difference from FalconRotaryEmbedding._set_cos_sin_cache
        t = t / self.scaling_factor
        freqs = ops.operations.MatMul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        emb = ops.cat((freqs, freqs), axis=-1)

        if dtype in [mindspore.float16, mindspore.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()[None, :, :]
        self.sin_cached = emb.sin()[None, :, :]

        self.cos_cached = self.cos_cached.astype(dtype)
        self.sin_cached = self.sin_cached.astype(dtype)


class FalconDynamicNTKScalingRotaryEmbedding(FalconRotaryEmbedding):
    """
    FalconRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(self, head_dim: int, base=10000, max_position_embeddings=2048, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(head_dim, base, max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.seq_len_cached = seq_len

        # This if block is the only difference from FalconRotaryEmbedding._set_cos_sin_cache
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len /
                 self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.head_dim / (self.head_dim - 2))
            inv_freq = 1.0 / \
                (base ** (ops.arange(0, self.head_dim,
                 2).float() / self.head_dim))
            self.inv_freq = mindspore.Parameter(
                inv_freq, name="inv_freq", requires_grad=False)

        t = ops.arange(seq_len)
        # freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        freqs = ops.operations.MatMul()(t.reshape(-1, 1), self.inv_freq.reshape(1, -1))
        emb = ops.cat((freqs, freqs), axis=-1)

        if dtype in [mindspore.float16, mindspore.bfloat16]:
            emb = emb.float()

        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

        self.cos_cached = self.cos_cached.astype(dtype)
        self.sin_cached = self.sin_cached.astype(dtype)


def _make_causal_mask(
    input_ids_shape: mindspore.Tensor.shape, past_key_values_length: int
) -> mindspore.Tensor:
    """
    Make causal mask used for self-attention. This mask does not take the existing attention mask into account - it
    just blocks tokens from attending forwards in the sequence. The output shape will be `[batch_size, 1,
    target_length, target_length+past_key_values_length]`.
    """
    batch_size, target_length = input_ids_shape

    mask = mindspore.Tensor.triu(ops.ones(
        (target_length, target_length), dtype=mindspore.Tensor.bool), diagonal=1)
    # If past_key_values_length is 0 this is an empty tensor and the concatenation is a no-op.
    # This code style is an unfortunate consequence of getting your TF engineer to port models; doing it this
    # way avoids a data-dependent conditional, which will help me when I have to port this to XLA later.
    past_mask = ops.zeros(
        (target_length, past_key_values_length), dtype=mindspore.Tensor.bool)
    mask = ops.cat([past_mask, mask], axis=-1)
    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: mindspore.Tensor, past_key_values_length: int) -> mindspore.Tensor:
    """
    Expands attention_mask from `[batch_size, seq_length]` to `[batch_size, 1, seq_length, seq_length + past_length]`.
    """
    batch_size, total_length = mask.shape
    seq_length = total_length - \
        past_key_values_length if past_key_values_length is not None else total_length

    expanded_mask = ~(mask[:, None, None, :].astype(mindspore.Tensor.bool))
    return expanded_mask.expand(batch_size, 1, seq_length, total_length)


def build_alibi_tensor(attention_mask: mindspore.Tensor, num_heads: int, dtype: mindspore.dtype) -> mindspore.Tensor:
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = mindspore.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=mindspore.float32
    )
    powers = ops.arange(1, 1 + closest_power_of_2, dtype=mindspore.int32)
    slopes = ops.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = mindspore.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=mindspore.float32
        )
        num_remaining_heads = min(
            closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = ops.arange(
            1, 1 + 2 * num_remaining_heads, 2, dtype=mindspore.int32)
        slopes = ops.cat([slopes, ops.pow(extra_base, extra_powers)], axis=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1)
                     * attention_mask)[:, None, :]
    alibi = slopes[..., None].bfloat16() * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).astype(dtype)


# Copied from transformers.models.bloom.modeling_bloom.dropout_add
def dropout_add(x: mindspore.Tensor, residual: mindspore.Tensor, prob: float, training: bool) -> mindspore.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
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

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.maybe_rotary = self._init_rope() if config.rotary else lambda q, k, t, p: (q, k)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 +
                           config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size
        self.query_key_value = FalconLinear(
            self.hidden_size, qkv_out_dim, has_bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(
            self.hidden_size, self.hidden_size, has_bias=config.bias)
        self.attention_dropout = nn.Dropout(p=config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (
            self.new_decoder_architecture or not self.multi_query) else 1

    def _init_rope(self):
        if self.config.rope_scaling is None:
            rotary_emb = FalconRotaryEmbedding(
                self.head_dim,
                base=self.config.rope_theta,
                max_position_embeddings=self.config.max_position_embeddings,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                rotary_emb = FalconLinearScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                rotary_emb = FalconDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    base=self.config.rope_theta,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        return rotary_emb

    def _split_heads(self, fused_qkv: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        if self.new_decoder_architecture:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(
                batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
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
                batch_size, seq_length, self.num_heads, 3, self.head_dim)
            return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
        else:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(
                batch_size, seq_length, self.num_heads + 2, self.head_dim)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

    # Copied from transformers.models.bloom.modeling_bloom.BloomAttention._merge_heads
    def _merge_heads(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
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

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> mindspore.Tensor:
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / \
            math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = ops.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = ops.ones(
                L, S, dtype=mindspore.Tensor.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.astype(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == mindspore.Tensor.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = ops.matmul(query, key.swapaxes(-2, -1) * scale_factor)
        attn_weight += attn_bias
        attn_weight = ops.softmax(attn_weight, axis=-1)
        attn_weight = ops.dropout(attn_weight, dropout_p, training=True)
        return ops.matmul(attn_weight, value)

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
        padding_mask: Optional[mindspore.Tensor] = None,
    ):
        # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = self.query_key_value(hidden_states)
        num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.swapaxes(1, 2).reshape(
            batch_size * self.num_heads, query_length, self.head_dim)
        key_layer = key_layer.swapaxes(1, 2).reshape(
            batch_size * num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.swapaxes(1, 2).reshape(
            batch_size * num_kv_heads, query_length, self.head_dim)

        past_kv_length = 0 if layer_past is None else layer_past[0].shape[1]
        query_layer, key_layer = self.maybe_rotary(
            query_layer, key_layer, past_kv_length, position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, kv_length, head_dim]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = ops.cat((past_key, key_layer), axis=1)
            value_layer = ops.cat((past_value, value_layer), axis=1)

        _, kv_length, _ = key_layer.shape
        if use_cache:
            present = (key_layer, value_layer)
        else:
            present = None

        # mindspore don't have finfo API, so use numpy to convert
        float_min = np.finfo(mindspore.dtype_to_nptype(query_layer.dtype)).min
        attention_mask_float = (
            attention_mask * 1.0).masked_fill(attention_mask, float_min).astype(query_layer.dtype)

        query_layer_ = query_layer.reshape(
            batch_size, self.num_heads, -1, self.head_dim)
        key_layer_ = key_layer.reshape(
            batch_size, num_kv_heads, -1, self.head_dim)
        value_layer_ = value_layer.reshape(
            batch_size, num_kv_heads, -1, self.head_dim)

        if alibi is None:
            if hasattr(F, "scaled_dot_product_attention") and not output_attentions:
                # TODO: deprecate this once we add FA2 support in Falcon
                logger.warning(
                    "The current implementation of Falcon calls `torch.scaled_dot_product_attention` directly, this will be deprecated in the"
                    " future in favor of the `BetterTransformer` API. Please install the latest optimum library with `pip install -U optimum` and call "
                    "`model.to_bettertransformer()` to benefit from `torch.scaled_dot_product_attention` and future performance optimizations."
                )

                attn_output = self.scaled_dot_product_attention(
                    query_layer_, key_layer_, value_layer_, attention_mask_float, 0.0, is_causal=False
                )
                attention_scores = None
            else:
                attention_scores = ops.matmul(
                    query_layer_, key_layer_.swapaxes(-1, -2))
                attention_scores /= math.sqrt(self.head_dim)

                attention_scores = F.softmax(
                    attention_scores + attention_mask_float, axis=-1, dtype=hidden_states.dtype
                )
                attn_output = ops.matmul(attention_scores, value_layer_)

            attn_output = attn_output.view(
                batch_size, self.num_heads, query_length, self.head_dim)
            attn_output = attn_output.permute(0, 2, 1, 3)
            attn_output = attn_output.reshape(
                batch_size, query_length, self.num_heads * self.head_dim)

            output_tensor = self.dense(attn_output)

            if output_attentions:
                return output_tensor, present, attention_scores
            else:
                return output_tensor, present

        else:
            matmul_result = ops.matmul(
                query_layer_, key_layer_.swapaxes(-1, -2))

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(
                batch_size, self.num_heads, query_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == mindspore.float16 or input_dtype == mindspore.bfloat16:
                attention_scores = attention_scores.astype(mindspore.float32)
            # Matt (HF) note: We could possibly use F.scaled_dot_product_attention here too, by
            # adding (alibi * self.inv_norm_factor) to attention_mask_float. I think this would be mathematically
            # equivalent and more performant, but there might be a numerical difference. If you're reading this
            # and you'd like to experiment and maybe file a PR, feel free!
            attention_logits = attention_scores + \
                alibi.view(batch_size, self.num_heads, 1, -1)
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(
                attention_logits + attention_mask_float, axis=-1, dtype=hidden_states.dtype)
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(
                batch_size, self.num_heads, query_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            context_layer = (ops.matmul(
                attention_probs_reshaped, value_layer_)).flatten(0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            context_layer = self._merge_heads(context_layer)

            output_tensor = self.dense(context_layer)

            if output_attentions:
                return output_tensor, present, attention_probs
            else:
                return output_tensor, present


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
        if isinstance(module, nn.Dense) or isinstance(module, FalconLinear):
            # 使用正态分布初始化权重
            module.weight.set_data(
                Normal(0.0, self.config.initializer_range)(module.weight.shape))
            if module.bias is not None:
                module.bias.set_data(
                    np.zeros(module.bias.shape, dtype=np.float32))
        elif isinstance(module, nn.Embedding):
            module.weight.set_data(
                Normal(0.0, self.config.initializer_range)(module.weight.shape))
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx, :].set_data(
                    np.zeros(module.weight.shape[1:], dtype=np.float32))
        elif isinstance(module, nn.LayerNorm):
            module.bias.set_data(np.zeros(module.bias.shape, dtype=np.float32))
            module.weight.set_data(
                np.ones(module.weight.shape, dtype=np.float32))

    # Copied from transformers.models.bloom.modeling_bloom.BloomPreTrainedModel._set_gradient_checkpointing with BloomModel->FalconModel
    def _set_gradient_checkpointing(self, module: nn.Cell, value: bool = False):
        if isinstance(module, FalconModel):
            module.gradient_checkpointing = value

    @staticmethod
    def _convert_cache_to_standard_format(
        past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]], batch_size: int
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, kv_length, head_dim = past_key_value[0][0].shape
        # [batch_size * self.num_heads, kv_length, head_dim] -> [batch_size, num_heads, kv_length, head_dim]
        # Note that don't want to use self.num_attention_heads because the number of heads may vary depending
        # on whether we use multi_query attention.
        num_heads = batch_size_times_num_heads // batch_size
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, kv_length, head_dim),
                layer_past[1].view(batch_size, num_heads, kv_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_rw_cache(
        past_key_value: Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]
    ) -> Tuple[Tuple[mindspore.Tensor, mindspore.Tensor]]:
        batch_size, num_heads, kv_length, head_dim = past_key_value[0][0].shape
        batch_size_times_num_heads = batch_size * num_heads
        # [batch_size, num_heads, kv_length, head_dim] -> [batch_size * num_heads, kv_length, head_dim]
        return tuple(
            (
                layer_past[0].view(
                    batch_size_times_num_heads, kv_length, head_dim),
                layer_past[1].view(
                    batch_size_times_num_heads, kv_length, head_dim),
            )
            for layer_past in past_key_value
        )


class FalconModel(FalconPreTrainedModel):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
