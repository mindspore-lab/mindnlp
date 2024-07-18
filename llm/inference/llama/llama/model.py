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
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import One, Zero
from mindspore.ops._tracefunc import trace
from mindspore.ops._primitive_cache import _get_cache_prim
from mindnlp.parallel.tensor_parallel.utils import get_group_size

from mindnlp.parallel.tensor_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


class LlamaConfig():
    """
    Configuration for Llama
    """
    def __init__(
            self,
            dim=512,
            n_layers=8,
            n_heads=8,
            vocab_size=-1,  # defined later by tokenizer
            multiple_of=256,  # make SwiGLU hidden layer size multiple of large power of 2
            norm_eps=1e-5,
            max_batch_size=32,
            max_seq_len=2048,
            **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.n_heads = n_heads
        self.multiple_of = multiple_of
        self.max_batch_size = max_batch_size
        self.norm_eps = norm_eps


class RMSNorm(nn.Cell):
    '''
    RMSNorm
    '''
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(Tensor(shape=(dim,), dtype=mindspore.float16, init=One()), 'weight')

    def _norm(self, norm_x):
        return norm_x * ops.rsqrt(ops.mean(ops.pow(norm_x, 2), -1, keep_dims=True) + self.eps)

    @trace
    def construct(self, construct_x):
        '''RMSNorm construct'''
        output = self._norm(construct_x.float()).astype(construct_x.dtype)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    '''
    precompute_freqs_cis
    '''
    freqs = 1.0 / (theta ** (ops.cast(ops.arange(0, dim, 2)[: (dim // 2)], mindspore.float32) / dim))
    _t = ops.arange(end).astype(freqs.dtype)  # type: ignore
    freqs = ops.cast(ops.outer(_t, freqs), mindspore.float32)  # type: ignore
    # TODO(khoray): wait response of lyf
    freqs_cis = ops.polar(ops.ones_like(freqs), freqs)  # complex64
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
    _complex = _get_cache_prim(ops.Complex)()
    return _complex(_x[:,:,:,:,0], _x[:,:,:,:,1])


@trace
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
    xq_out = ops.flatten(ops.view_as_real(xq_ * freqs_cis), start_dim=3)
    xk_out = ops.flatten(ops.view_as_real(xk_ * freqs_cis), start_dim=3)
    return xq_out.astype(x_q.dtype), xk_out.astype(x_k.dtype)


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

        self.cache_k = Tensor(shape=(config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim),
                              dtype=mindspore.float16, init=Zero())

        self.cache_v = Tensor(shape=(config.max_batch_size, config.max_seq_len, self.n_local_heads, self.head_dim),
                              dtype=mindspore.float16, init=Zero())

    @trace
    def get_qkv(self, _x):
        bsz, seqlen, _ = _x.shape
        # x = h = [bsz * seqlen * emb_dim]
        x_q, x_k, x_v = self.w_q(_x), self.w_k(_x), self.w_v(_x)

        x_q = x_q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        x_k = x_k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        x_v = x_v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        return x_q, x_k, x_v        

    def construct(self, _x: mindspore.Tensor, start_pos: int,
                freqs_cis: mindspore.Tensor, mask: Optional[mindspore.Tensor]):
        '''
        construct
        '''
        bsz, seqlen, _ = _x.shape
        x_q, x_k, x_v = self.get_qkv(_x)

        # xq = xk = xv = [bsz, seqlen, self.n_local_heads, self.head_dim]

        x_q, x_k = apply_rotary_emb(x_q, x_k, freqs_cis=freqs_cis,)


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
        scores = ops.matmul(x_q, ops.transpose(keys, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = ops.softmax(scores.astype(mindspore.float32), axis=-1).astype(x_q.dtype)
        output = ops.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = ops.transpose(output, (0, 2, 1, 3)).view(bsz, seqlen, -1)
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

    @trace
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
        _h = _x + self.attention(self.attention_norm(_x), start_pos, freqs_cis, mask)
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

        self.layers = nn.CellList()
        for layer_id in range(config.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = ColumnParallelLinear(
            config.dim, config.vocab_size, bias=False, dtype=mindspore.float16
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
            mask = ops.full((1, 1, seqlen, seqlen), float("-inf"))
            mask = ops.triu(mask, diagonal=start_pos + 1).astype(_h.dtype)

        for layer in self.layers:
            _h = layer(_h, start_pos, freqs_cis, mask)

        _h = self.norm(_h) # h = [bsz * seqlen * emb_dim]
        output = self.output(_h[:, -1, :])  # only compute last logits  output = [bsz * vocab_size]
        return output.astype(mindspore.float32)
