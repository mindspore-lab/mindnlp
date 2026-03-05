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

        from .linear import Linear
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        # Handle batch_first: convert (N, L, E) -> (L, N, E)
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        # Project Q, K, V
        if self._qkv_same_embed_dim:
            w_q = self.in_proj_weight[:embed_dim]
            w_k = self.in_proj_weight[embed_dim:2*embed_dim]
            w_v = self.in_proj_weight[2*embed_dim:]
        else:
            w_q = self.q_proj_weight
            w_k = self.k_proj_weight
            w_v = self.v_proj_weight

        if self.in_proj_bias is not None:
            b_q = self.in_proj_bias[:embed_dim]
            b_k = self.in_proj_bias[embed_dim:2*embed_dim]
            b_v = self.in_proj_bias[2*embed_dim:]
        else:
            b_q = b_k = b_v = None

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        # Reshape: (L, N, E) -> (L, N, H, D) -> (N, H, L, D)
        head_dim = self.head_dim
        num_heads = self.num_heads

        # (L, N, H, D) -t(0,1)-> (N, L, H, D) -t(1,2)-> (N, H, L, D)
        q = q.reshape(tgt_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
        k = k.reshape(src_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)
        v = v.reshape(src_len, bsz, num_heads, head_dim).transpose(0, 1).transpose(1, 2)

        # Handle key_padding_mask: (N, S) bool where True = ignore
        if key_padding_mask is not None:
            from ..._creation import tensor as _tensor
            from ..._functional import where as _where
            # Expand to (N, 1, 1, S) for broadcasting with (N, H, L, S)
            kpm = key_padding_mask.reshape(bsz, 1, 1, src_len)
            neg_inf = _tensor(float('-inf'), device=query.device)
            zero = _tensor(0.0, device=query.device)
            kpm_mask = _where(kpm, neg_inf, zero)
            if attn_mask is not None:
                from ..._functional import add as _add
                attn_mask = _add(attn_mask, kpm_mask)
            else:
                attn_mask = kpm_mask

        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        # (N, H, L, D) -> (N, L, H, D) -> (N, L, E) -> (L, N, E)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = attn_output.transpose(0, 1)

        # Output projection
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # Convert back to batch_first if needed
        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, None

    def extra_repr(self):
        return f'embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}'
