import mindtorch
from mindtorch import nn

from ..utils.context import get_context


def store_kvcache(key: mindtorch.Tensor, value: mindtorch.Tensor, k_cache: mindtorch.Tensor, v_cache: mindtorch.Tensor, slot_mapping: mindtorch.Tensor):
    # pylint: disable=undefined-variable
    # These are conditionally imported from flash_attn or other backends
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = mindtorch.tensor([])

    def forward(self, q: mindtorch.Tensor, k: mindtorch.Tensor, v: mindtorch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # if context.is_prefill:
        #     if context.block_tables is not None:    # prefix cache
        #         k, v = k_cache, v_cache
        #     # pylint: disable=undefined-variable
        #     o = flash_attn_varlen_func(q, k, v,
        #                                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
        #                                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
        #                                softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        # else:    # decode
        #     # flash_attn_with_kvcache is conditionally imported from flash_attn
        #     # pylint: disable=undefined-variable
        #     o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,  # noqa: F821
        #                                 cache_seqlens=context.context_lens, block_table=context.block_tables,
        #                                 softmax_scale=self.scale, causal=True)
        # return o
