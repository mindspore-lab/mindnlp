# Copyright (c) Meta Platforms, Inc. and affiliates.
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

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
# pylint: disable=C0103
""" MindNLP llama model."""

import math
from typing import Tuple, Optional
import numpy as np

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.communication import get_group_size

from mindnlp.parallel import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from .llama_config import LlamaConfig


class RMSNorm(nn.Cell):
    '''
    RMSNorm
    '''
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones(dim, dtype=mindspore.float16))

    def _norm(self, norm_x):
        return norm_x * ops.rsqrt(ops.mean(ops.pow(norm_x, 2), -1, keep_dims=True) + self.eps)

    def construct(self, construct_x):
        '''RMSNorm construct'''
        output = self._norm(construct_x.float()).astype(construct_x.dtype)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, start_pos: int, seqlen: int, theta: float = 10000.0):
    '''
    precompute_freqs_cis
    '''
    freqs = 1.0 / (theta ** (ops.cast(ops.arange(0, dim, 2)[: (dim // 2)], mindspore.float32) / dim))
    _t = ops.arange(end).astype(freqs.dtype)
    freqs = ops.cast(ops.outer(_t, freqs), mindspore.float32)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = ops.cat((freqs, freqs), axis=-1)
    cos_cached = emb.cos()[None, :, None, :][:, start_pos : start_pos + seqlen, :, ...]
    sin_cached = emb.sin()[None, :, None, :][:, start_pos : start_pos + seqlen, :, ...]
    return (cos_cached, sin_cached,)


def rotate_half(_x):  # [bsz, seqlen, n_local_heads, head_dim]
    '''
    Rotates half the hidden dims of the input.
    '''
    x_1 = _x[..., : _x.shape[-1] // 2]
    x_2 = _x[..., _x.shape[-1] // 2 :]
    return ops.cat((-x_2, x_1), axis=-1)


def apply_rotary_emb(
    x_q: mindspore.Tensor,
    x_k: mindspore.Tensor,
    cos: mindspore.Tensor,
    sin: mindspore.Tensor,
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    '''
    apply_rotary_emb
    '''
    xq_out = (x_q * cos) + (rotate_half(x_q) * sin)
    xk_out = (x_k * cos) + (rotate_half(x_k) * sin)
    return xq_out.astype(x_q.dtype), xk_out.astype(x_k.dtype)

def repeat_kv(x: mindspore.Tensor, n_rep: int) -> mindspore.Tensor:
    """repeat kv"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Cell):
    '''
    Attention
    '''
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.n_local_heads = config.n_heads // get_group_size()
        self.head_dim = config.dim // config.n_heads
        self.w_q = ColumnParallelLinear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            dtype=mindspore.float16,
        )
        self.w_k = ColumnParallelLinear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            dtype=mindspore.float16,
        )
        self.w_v = ColumnParallelLinear(
            config.dim,
            config.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            dtype=mindspore.float16,
        )
        self.w_o = RowParallelLinear(
            config.n_heads * self.head_dim,
            config.dim,
            bias=False,
            input_is_parallel=True,
            dtype=mindspore.float16,
        )

        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len

        self.cache_k = mindspore.Tensor(np.zeros(
            (self.max_batch_size, self.max_seq_len, self.n_local_heads, self.head_dim)
        ))
        self.cache_v = mindspore.Tensor(np.zeros(
            (self.max_batch_size, self.max_seq_len, self.n_local_heads, self.head_dim)
        ))
        self.inv_norm_factor = mindspore.Tensor(1.0 / math.sqrt(self.head_dim), dtype=mindspore.float16)

    def construct(self, _x: mindspore.Tensor, start_pos: int,
                  mask: Optional[mindspore.Tensor]):
        '''
        construct
        '''
        bsz, seqlen, _ = _x.shape
        # x = h = [bsz * seqlen * emb_dim]
        x_q, x_k, x_v = self.w_q(_x), self.w_k(_x), self.w_v(_x)

        x_q = x_q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        x_k = x_k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        x_v = x_v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # xq = xk = xv = [bsz, seqlen, self.n_local_heads, self.head_dim]

        cos, sin = precompute_freqs_cis(
            self.head_dim, self.max_seq_len * 2, start_pos, seqlen
        )
        x_q, x_k = apply_rotary_emb(x_q, x_k, cos, sin)

        self.cache_k = self.cache_k.astype(x_q.dtype)
        self.cache_v = self.cache_v.astype(x_q.dtype)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = x_k
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = x_v

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # xq = xk = xv = [bsz, seqlen, self.n_local_heads, self.head_dim]

        x_q = ops.transpose(x_q, (0, 2, 1, 3))
        keys = ops.transpose(keys, (0, 2, 1, 3))
        values = ops.transpose(values, (0, 2, 1, 3))
        scores = ops.matmul(x_q, ops.transpose(keys, (0, 1, 3, 2)))
        scores = ops.mul(scores, self.inv_norm_factor)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = ops.softmax(scores.astype(mindspore.float32), axis=-1, dtype=mindspore.float32)
        output = ops.matmul(scores.astype(mindspore.float16), values)  # (bs, n_local_heads, slen, head_dim)
        output = ops.transpose(output, (0, 2, 1, 3)).view(bsz, seqlen, -1) # .contiguous()???
        return self.w_o(output)


class FeedForward(nn.Cell):
    '''
    FeedForward
    '''
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, dtype=mindspore.float16
        )
        self.w_2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, dtype=mindspore.float16
        )
        self.w_3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, dtype=mindspore.float16
        )

    def silu(self, _x):
        '''
        Sigmoid Linear Unit
        '''
        return _x * ops.sigmoid(_x)

    def construct(self, _x):
        return self.w_2(self.silu(self.w_1(_x)) * self.w_3(_x))


class TransformerBlock(nn.Cell):
    '''
    TransformerBlock
    '''
    def __init__(self, layer_id: int, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.dim, hidden_dim=4 * config.dim, multiple_of=config.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def construct(self, _x: mindspore.Tensor, start_pos: int,
                  mask: Optional[mindspore.Tensor]):
        _h = _x + self.attention(self.attention_norm(_x), start_pos, mask)
        out = _h + self.feed_forward(self.ffn_norm(_h))
        return out


class Transformer(nn.Cell):
    '''
    Transformer
    '''
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = ParallelEmbedding(
            config.vocab_size, config.dim, dtype=mindspore.float16
        )

        self.layers = nn.SequentialCell()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = ColumnParallelLinear(
            config.dim, config.vocab_size, bias=False, dtype=mindspore.float16
        )

    def construct(self, tokens: mindspore.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape   # tokens.shape = [bsz * seqlen]
        _h = self.tok_embeddings(tokens) # h = [bsz * seqlen * emb_dim]

        mask = None
        if seqlen > 1:
            mask = ops.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = ops.triu(mask, diagonal=start_pos + 1).astype(_h.dtype)

        for layer in self.layers:
            _h = layer(_h, start_pos, mask)
        _h = self.norm(_h) # h = [bsz * seqlen * emb_dim]
        output = self.output(_h[:, -1, :])  # only compute last logits  output = [bsz * vocab_size]
        return ops.cast(output, mindspore.float32)
