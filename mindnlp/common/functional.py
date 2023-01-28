# Copyright 2022 Huawei Technologies Co., Ltd
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
# pylint: disable=C0103
# pylint: disable=E1120
# pylint: disable=E1123

"""Custom functional api for legacy mindspore"""
from math import pi
import mindspore
from mindspore import ops, Tensor
from mindspore.ops._primitive_cache import _get_cache_prim

def kl_div(inputs, target, reduction='none', log_target=False):
    """KLDiv function."""
    if log_target:
        kl_div_loss = ops.exp(target) * (target - inputs)
    else:
        output = target * (ops.log(target) - inputs)
        zeros = zeros_like(inputs)
        kl_div_loss = ops.select(target > 0, output, zeros)
    if reduction == 'sum':
        return kl_div_loss.sum()
    if reduction == 'mean':
        return kl_div_loss.mean()
    return kl_div_loss

def softmax(inputs, axis=-1):
    """inner softmax"""
    _softmax_op = _get_cache_prim(ops.Softmax)(axis)
    return _softmax_op(inputs)

def sqrt(x):
    """inner sqrt"""
    _sqrt = _get_cache_prim(ops.Sqrt)()
    return _sqrt(x)

def relu(x):
    """inner relu."""
    relu_ = _get_cache_prim(ops.ReLU)()
    return relu_(x)

def gelu(input_x, approximate='none'):
    """inner gelu"""
    if approximate not in ['none', 'tanh']:
        raise ValueError("For ops.gelu, approximate value should be either 'none' or 'tanh'.")

    output = _get_cache_prim(ops.GeLU)()(input_x)

    if approximate == 'tanh':
        output = _get_cache_prim(ops.Pow)()(input_x, Tensor([3]))
        output = output * Tensor([0.044715]) + input_x
        output = output * _get_cache_prim(ops.Sqrt)()(Tensor(2.0 / pi))
        output = _get_cache_prim(ops.Tanh)()(output) + Tensor([1.0])
        output = output * input_x * Tensor([0.5])

    return output

def is_floating_point(x):
    """inner is_floating_point"""
    return x.dtype in [mindspore.float32, mindspore.float16, mindspore.float64]

def zeros_like(x, *, dtype=None):
    """inner zeros_like"""
    _dtype = x.dtype if dtype is None else dtype
    zeros_like_op = _get_cache_prim(ops.ZerosLike)()
    output = zeros_like_op(x)
    output = output.astype(_dtype)
    return output

def linear(x, w, b):
    """inner linear"""
    out = ops.matmul(x, w.swapaxes(-1, -2))
    if b is not None:
        out = out + b
    return out

def _in_projection(
    q,
    k,
    v,
    w_q,
    w_k,
    w_v,
    b_q = None,
    b_k = None,
    b_v = None,
):
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.
    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`
        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`
    """
    Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def _in_projection_packed(q, k, v, w, b, k_is_v, q_is_k):
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.
    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.
    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension
        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.shape[-1]
    if k_is_v:
        if q_is_k:
            # self-attention
            return ops.split(linear(q, w, b), -1, 3)
        # encoder-decoder attention
        w_q, w_kv = w.split([E, E * 2])
        if b is None:
            b_q = b_kv = None
        else:
            b_q, b_kv = b.split([E, E * 2])
        return (linear(q, w_q, b_q),) + ops.split(linear(k, w_kv, b_kv), -1, 2)
    w_q, w_k, w_v = ops.split(w, output_num=3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = ops.split(b, output_num=3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)

def _scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, is_training):
    embed_size = query.shape[-1]
    scaling_factor = sqrt(sqrt(Tensor(embed_size, mindspore.float32)))
    query = query / scaling_factor

    if is_causal:
        L = query.shape[-2], S = key.shape[-2]
        attn_mask = ops.ones((L, S), mindspore.bool_).tril()

    attn = ops.matmul(query, key.swapaxes(-2, -1) / scaling_factor)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = softmax(attn, -1)
    if dropout_p > 0. and is_training:
        attn = ops.dropout(attn, dropout_p)
    output = ops.matmul(attn, value)

    return (output, attn)

def _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.ndim == 3:
        # Batched Inputs
        is_batched = True
        assert key.ndim == 3 and value.ndim == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.ndim}-D and {value.ndim}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.ndim == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.ndim}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.ndim in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.ndim}-D tensor instead")
    elif query.ndim == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.ndim == 2 and value.ndim == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.ndim}-D and {value.ndim}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.ndim == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.ndim}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.ndim in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.ndim}-D tensor instead")
            if attn_mask.ndim == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {query.ndim}-D query tensor")

    return is_batched

def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight,
    out_proj_bias,
    training: bool = True,
    key_padding_mask = None,
    attn_mask = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight = None,
    k_proj_weight = None,
    v_proj_weight = None,
    static_k = None,
    static_v = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    k_is_v: bool = False,
    q_is_k: bool = False,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
            Default: ``False``.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True
    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we expand_dims at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # expand_dims if the input is unbatched
        query = query.expand_dims(1)
        key = key.expand_dims(1)
        value = value.expand_dims(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.expand_dims(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
    #     if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
    #         raise AssertionError(
    #             "only bool and floating types of key_padding_mask are supported")
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias, k_is_v, q_is_k)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = ops.split(in_proj_bias, output_num=3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == mindspore.uint8:
            attn_mask = attn_mask.astype(mindspore.bool_)
        else:
            assert is_floating_point(attn_mask) or attn_mask.dtype == mindspore.bool_, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.ndim == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, "
                                    "but should be {correct_2d_size}.")
            attn_mask = attn_mask.expand_dims(0)
        elif attn_mask.ndim == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, "
                                    "but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.ndim} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = ops.cat([k, bias_k.repeat(1, bsz, 1)])
        v = ops.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = ops.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = ops.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).swapaxes(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.shape[0] == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.shape[2] == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).swapaxes(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.shape[0] == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.shape[2] == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = ops.cat([k, ops.zeros(zero_attn_shape, dtype=k.dtype)], axis=1)
        v = ops.cat([v, ops.zeros(zero_attn_shape, dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = ops.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = ops.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == mindspore.bool_:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == mindspore.bool_:
        new_attn_mask = zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    #
    # (deep breath) calculate attention and out projection
    #

    if attn_mask is not None:
        if attn_mask.shape[0] == 1:
            attn_mask = attn_mask.expand_dims(0)
        else:
            attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p, is_causal, training)
    attn_output = attn_output.transpose(2, 0, 1, 3).view(bsz * tgt_len, embed_dim)

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.shape[1])

    # optionally average attention weights over heads
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_output_weights = attn_output_weights.sum(axis=1) / num_heads

    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(0)
    return attn_output, attn_output_weights
