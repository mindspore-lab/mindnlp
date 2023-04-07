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
