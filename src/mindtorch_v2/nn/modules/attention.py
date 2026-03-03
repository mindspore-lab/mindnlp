from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from .. import functional as F


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            w = tensor([[0.0] * embed_dim for _ in range(3 * embed_dim)])
            self.in_proj_weight = Parameter(w)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        else:
            self.register_parameter('in_proj_weight', None)
            self.q_proj_weight = Parameter(tensor([[0.0] * embed_dim for _ in range(embed_dim)]))
            self.k_proj_weight = Parameter(tensor([[0.0] * self.kdim for _ in range(embed_dim)]))
            self.v_proj_weight = Parameter(tensor([[0.0] * self.vdim for _ in range(embed_dim)]))

        if bias:
            self.in_proj_bias = Parameter(tensor([0.0] * 3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        w_out = tensor([[0.0] * embed_dim for _ in range(embed_dim)])
        self.out_proj = type('Linear', (), {
            'weight': Parameter(w_out),
            'bias': Parameter(tensor([0.0] * embed_dim)) if bias else None,
        })()
        # Register as a proper submodule by storing weight/bias
        from .linear import Linear
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        raise NotImplementedError("MultiheadAttention forward is not yet implemented")

    def extra_repr(self):
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}'
