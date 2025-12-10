# FILE: nanovllm/layers/rotary_embedding.py
from functools import lru_cache
import mindtorch
from mindtorch import nn
from typing import Optional, Dict


def apply_rotary_emb(
    x: mindtorch.Tensor,
    cos: mindtorch.Tensor,
    sin: mindtorch.Tensor,
) -> mindtorch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = mindtorch.chunk(x.to(mindtorch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return mindtorch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(mindtorch.arange(0, rotary_dim, 2, dtype=mindtorch.float) / rotary_dim))
        t = mindtorch.arange(max_position_embeddings, dtype=mindtorch.float)
        freqs = mindtorch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = mindtorch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @mindtorch.compile
    def forward(
        self,
        positions: mindtorch.Tensor,
        query: mindtorch.Tensor,
        key: mindtorch.Tensor,
    ) -> tuple[mindtorch.Tensor, mindtorch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Optional[Dict] = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb