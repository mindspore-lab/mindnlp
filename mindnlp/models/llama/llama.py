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
""" MindNLP llama model."""

import math
from typing import Tuple, Optional

import mindspore
from mindspore import nn, ops, Parameter, numpy
from .llama_config import LlamaConfig

class RMSNorm(nn.Cell):
    '''
    RMSNorm
    '''
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(ops.ones(dim))

    def _norm(self, norm_x):
        return norm_x * ops.rsqrt(ops.mean(ops.pow(norm_x, 2), -1, keep_dims=True) + self.eps)

    def construct(self, construct_x):
        '''RMSNorm construct'''
        output = self._norm(construct_x.float()).astype(construct_x.dtype)
        return output * self.weight

def polar(pabs, angle):
    '''
    polar
    '''
    return ops.Complex()(pabs * ops.cos(angle), pabs * ops.sin(angle))

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    '''
    precompute_freqs_cis
    '''
    freqs = 1.0 / (theta ** (ops.cast(numpy.arange(0, dim, 2)[: (dim // 2)], mindspore.float32) / dim))
    _t = numpy.arange(end).astype(freqs.dtype)  # type: ignore
    freqs = ops.cast(numpy.outer(_t, freqs), mindspore.float32)  # type: ignore
    # TODO(khoray): wait response of lyf
    freqs_cis = polar(numpy.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: mindspore.Tensor, _x: mindspore.Tensor):
    '''
    reshape_for_broadcast
    '''
    ndim = _x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (_x.shape[1], _x.shape[-1])
    shape = [d if i in (1, ndim - 1) else 1 for i, d in enumerate(_x.shape)]
    return freqs_cis.view(*shape)

def view_as_complex(_x: mindspore.Tensor):
    '''
    view_as_complex
    '''
    return ops.Complex()(_x[:,:,:,:,0], _x[:,:,:,:,1])

def view_as_real(_x: mindspore.Tensor):
    '''
    view_as_real
    '''
    real = ops.real(_x)
    imag = ops.imag(_x)
    return ops.stack([real, imag], -1)

def apply_rotary_emb(
    x_q: mindspore.Tensor,
    x_k: mindspore.Tensor,
    freqs_cis: mindspore.Tensor,
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    '''
    apply_rotary_emb
    '''
    xq_ = view_as_complex(x_q.float().reshape(*x_q.shape[:-1], -1, 2))
    xk_ = view_as_complex(x_k.float().reshape(*x_k.shape[:-1], -1, 2))
    # xq_ : [bsz, seqlen, n_heads, head_dim]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = ops.flatten(view_as_real(xq_ * freqs_cis), start_dim=3)
    xk_out = ops.flatten(view_as_real(xk_ * freqs_cis), start_dim=3)
    return xq_out.astype(x_q.dtype), xk_out.astype(x_k.dtype)

class Attention(nn.Cell):
    '''
    Attention
    '''
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.n_local_heads = config.n_heads # delete parallel
        self.head_dim = config.dim // config.n_heads
        self.w_q = nn.Dense(
            config.dim,
            config.n_heads * self.head_dim,
            has_bias=False
        )
        self.w_k = nn.Dense(
            config.dim,
            config.n_heads * self.head_dim,
            has_bias=False
        )
        self.w_v = nn.Dense(
            config.dim,
            config.n_heads * self.head_dim,
            has_bias=False
        )
        self.w_o = nn.Dense(
            config.n_heads * self.head_dim,
            config.dim,
            has_bias=False
        )

        self.cache_k = numpy.zeros(
            (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
        )
        self.cache_v = numpy.zeros(
            (config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim)
        )

    def construct(self, _x: mindspore.Tensor, start_pos: int,
                freqs_cis: mindspore.Tensor, mask: Optional[mindspore.Tensor]):
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

        x_q, x_k = apply_rotary_emb(x_q, x_k, freqs_cis=freqs_cis)

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
        scores = numpy.matmul(x_q, ops.transpose(keys, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = ops.softmax(scores.astype(mindspore.float32), axis=-1).astype(x_q.dtype)
        output = ops.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
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

        self.w_1 = nn.Dense(dim, hidden_dim, has_bias=False)
        self.w_2 = nn.Dense(hidden_dim, dim, has_bias=False)
        self.w_3 = nn.Dense(dim, hidden_dim, has_bias=False)

    def construct(self, _x):
        return self.w_2(ops.silu(self.w_1(_x)) * self.w_3(_x))

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
                freqs_cis: mindspore.Tensor, mask: Optional[mindspore.Tensor]):
        _h = _x + self.attention.construct(self.attention_norm(_x), start_pos, freqs_cis, mask)
        out = _h + self.feed_forward.construct(self.ffn_norm(_h))
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

        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.dim
        )

        self.layers = nn.SequentialCell()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Dense(
            config.dim, config.vocab_size, has_bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads, self.config.max_seq_len * 2
        )

    def construct(self, tokens: mindspore.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape   # tokens.shape = [bsz * seqlen]
        _h = self.tok_embeddings(tokens) # h = [bsz * seqlen * emb_dim]
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen] # freqs_cis = [seqlen, emb_dim // nheads // 2]

        mask = None
        if seqlen > 1:
            mask = numpy.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = numpy.triu(mask, k=start_pos + 1).astype(_h.dtype)

        for layer in self.layers:
            _h = layer(_h, start_pos, freqs_cis, mask)
        _h = self.norm(_h) # h = [bsz * seqlen * emb_dim]
        output = self.output(_h[:, -1, :])  # only compute last logits  output = [bsz * vocab_size]
        return ops.cast(output, mindspore.float32)
