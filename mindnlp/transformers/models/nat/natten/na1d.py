#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_

from .context import is_fna_enabled
from .functional import na1d, na1d_av, na1d_qk
from .types import CausalArg1DTypeOrDed, Dimension1DTypeOrDed
from .utils import check_all_args, log

logger = log.get_logger(__name__)


class NeighborhoodAttention1D(nn.Module):
    """
    Neighborhood Attention 1D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: Dimension1DTypeOrDed,
        dilation: Dimension1DTypeOrDed = 1,
        is_causal: CausalArg1DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size_, dilation_, is_causal_ = check_all_args(
            1, kernel_size, dilation, is_causal
        )
        assert len(kernel_size_) == len(dilation_) == len(is_causal_) == 1
        if any(is_causal_) and rel_pos_bias:
            raise NotImplementedError(
                "Causal neighborhood attention is undefined with positional biases."
                "Please consider disabling positional biases, or open an issue."
            )

        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size_
        self.dilation = dilation_
        self.is_causal = is_causal_

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * self.kernel_size[0] - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"NeighborhoodAttention1D expected a rank-3 input tensor; got {x.dim()=}."
            )

        B, L, C = x.shape

        if is_fna_enabled():
            if self.attn_drop_rate > 0:
                logger.error(
                    "You're using fused neighborhood attention, and passed in a "
                    "non-zero attention dropout rate. This implementation does "
                    "support attention dropout yet, which means dropout is NOT being applied "
                    "to your attention weights."
                )

            qkv = (
                self.qkv(x)
                .reshape(B, L, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 1, 3, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = na1d(
                q,
                k,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
                scale=self.scale,
            )
            x = x.reshape(B, L, C)

        else:
            qkv = (
                self.qkv(x)
                .reshape(B, L, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na1d_qk(
                q,
                k,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
                rpb=self.rpb,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = na1d_av(
                attn,
                v,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                is_causal=self.is_causal,
            )
            x = x.permute(0, 2, 1, 3).reshape(B, L, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"has_bias={self.rpb is not None}"
        )
