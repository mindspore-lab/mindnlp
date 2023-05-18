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
# pylint: disable=W0622
# pylint: disable=E1123
# pylint: disable=E1120

"""Custom functional api for legacy mindspore"""
import builtins
from math import pi
import mindspore
import numpy as np
from mindspore.ops.operations.array_ops import Tril
from mindspore import ops, Tensor
from mindspore.common import dtype as mstype
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops import constexpr
from mindnlp.utils import less_min_api_compatible
tensor_slice = ops.Slice()
cast_ = ops.Cast()
scalar_to_tensor_ = ops.ScalarToTensor()


def masked_select(inputs, mask):
    """
    Returns a new 1-D Tensor which indexes the `x` tensor according to the boolean `mask`.
    The shapes of the `mask` tensor and the `x` tensor don't need to match, but they must be broadcastable.

    Args:
        input (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        mask (Tensor[bool]): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        A 1-D Tensor, with the same type as `input`.

    Raises:
        TypeError: If `input` or `mask` is not a Tensor.
        TypeError: If dtype of `mask` is not bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int64)
        >>> mask = Tensor(np.array([1, 0, 1, 0]), mindspore.bool_)
        >>> output = ops.masked_select(x, mask)
        >>> print(output)
        [1 3]
    """
    masked_select_ = _get_cache_prim(ops.MaskedSelect)()
    return masked_select_(inputs, mask)


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


def split(x, size, axis=0):
    """inner split"""
    if less_min_api_compatible:
        num = x.shape[axis] // size
        return ops.split(x, axis, num)
    return ops.split(x, split_size_or_sections=size, axis=axis)

def addmm(x, mat1, mat2, *, beta=1, alpha=1):
    """inner addmm"""
    _matmul_op = _get_cache_prim(ops.MatMul)()
    return beta * x + alpha * (_matmul_op(mat1, mat2))

def tril(input_x, diagonal=0):
    """inner tril"""
    _tril_op = _get_cache_prim(Tril)(diagonal)
    return _tril_op(input_x)

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

def _cast_type(x, to_type):
    """cast input to the specified type or cast input to tensor"""
    if isinstance(x, Tensor):
        x = cast_(x, to_type)
    else:
        x = scalar_to_tensor_(x, to_type)
    return x

def _get_type(x):
    """get the dtype of input"""
    if isinstance(x, Tensor):
        return x.dtype
    return type(x)

def _get_max_type(start, end, step):
    """get max input type with `level`"""
    valid_dtypes = [mstype.int32, mstype.float32, mstype.int64, mstype.float64]
    arg_map = [start, end, step]
    arg_type_map = [str(_get_type(i)) for i in arg_map]
    for arg_value in arg_map:
        if not (isinstance(arg_value, (float, int))
                or (isinstance(arg_value, Tensor) and arg_value.dtype in valid_dtypes)):
            raise TypeError(
                f"For arange, the input type must be int or float or a TensorScalar in {valid_dtypes},"
                f" but got {_get_type(arg_value)}")

    type_map = {'Float64': '3', 'Float32': '2', "<class 'float'>": '2', 'Int64': '1', "<class 'int'>": '1',
                'Int32': '0'}
    type_map_reverse = {'3': mstype.float64, '2': mstype.float32, '1': mstype.int64, '0': mstype.int32}
    type_level = [type_map.get(i) for i in arg_type_map]
    max_level = builtins.max(type_level)
    return type_map_reverse.get(max_level)


def argmax(input, dim=None, keepdim=False):
    """
    Return the indices of the maximum values of a tensor across a dimension.

    Args:
        input (Tensor): Input tensor.
        dim (Union[int, None], optional): The dimension to reduce. If `dim` is None, the indices of the maximum
            value within the flattened input will be returned. Default: None.
        keepdim (bool, optional): Whether the output tensor retains the specified
            dimension. Ignored if `dim` is None. Default: False.

    Returns:
        Tensor, indices of the maximum values across a dimension.

    Raises:
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float32))
        >>> output = ops.argmax(x, dim=-1)
        >>> print(output)
        [1 0 0]
    """
    if not input.shape:
        return Tensor(0)
    is_dim_none = False
    if dim is None:
        input = input.reshape((-1,))
        dim = 0
        is_dim_none = True
    out = _get_cache_prim(ops.Argmax)(dim, mstype.int64)(input)
    if keepdim and not is_dim_none:
        out = out.expand_dims(dim)
    return out

def full(size, fill_value, *, dtype=None): # pylint: disable=redefined-outer-name
    """
    Create a Tensor of the specified shape and fill it with the specified value.

    Args:
        size (Union(tuple[int], list[int])): The specified shape of output tensor.
        fill_value (number.Number): Value to fill the returned tensor. Complex numbers are not supported for now.

    Keyword Args:
        dtype (mindspore.dtype): The specified type of output tensor. `bool_` and `number` are supported, for details,
            please refer to :class:`mindspore.dtype` . Default: None.

    Returns:
        Tensor.

    Raises:
        TypeError: If `size` is not a tuple or list.
        ValueError: The element in `size` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.full((2, 2), 1)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
        >>> output = ops.full((3, 3), 0)
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
    """
    if not isinstance(size, (list, tuple)):
        raise TypeError(f"For 'ops.full', 'size' must be a tuple or list of ints, but got {type(size)}.")
    if dtype is None:
        dtype = mstype.int64
    if isinstance(size, list):
        size = tuple(size)
    fill_ = _get_cache_prim(ops.Fill)()
    return fill_(dtype, size, fill_value)

def arange(start=0, end=None, step=1, *, dtype=None):
    """inner arange"""
    res = Tensor(np.arange(start, end, step))
    if dtype is not None:
        res = res.astype(dtype)
    return res

def where(condition, x, y):
    r"""
    Selects elements from `x` or `y` based on `condition` and returns a tensor.

    .. math::
        output_i = \begin{cases} x_i,\quad &if\ condition_i \\ y_i,\quad &otherwise \end{cases}

    Args:
        condition (Tensor[bool]): If True, yield `x`, otherwise yield `y`.
        x (Union[Tensor, Scalar]): When `condition` is True, values to select from.
        y (Union[Tensor, Scalar]): When `condition` is False, values to select from.

    Returns:
        Tensor, elements are selected from `x` and `y`.

    Raises:
        TypeError: If `condition` is not a Tensor.
        TypeError: If both `x` and `y` are scalars.
        ValueError: If `condition`, `x` and `y` can not broadcast to each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.arange(4).reshape((2, 2)), mstype.float32)
        >>> b = Tensor(np.ones((2, 2)), mstype.float32)
        >>> condition = a < 3
        >>> output = ops.where(condition, a, b)
        >>> print(output)
        [[0. 1.]
         [2. 1.]]
    """
    if not isinstance(condition, Tensor):
        raise TypeError(f"For 'where', 'condition' must be a Tensor, but got {type(condition)}.")
    if isinstance(x, (int, float)):
        if not isinstance(y, Tensor):
            raise TypeError(f"For 'where', at least one of 'x' and 'y' should be Tensor, \
            but got x:{type(x)}, y:{type(y)}.")
        x = cast_(x, y.dtype)
    elif isinstance(y, (int, float)):
        if not isinstance(x, Tensor):
            raise TypeError(f"For 'where', at least one of 'x' and 'y' should be Tensor, \
            but got x:{type(x)}, y:{type(y)}.")
        y = cast_(y, x.dtype)
    output_shape = _calc_broadcast_shape(x.shape, y.shape, condition.shape)
    condition = broadcast_to(condition, output_shape)
    x = broadcast_to(x, output_shape)
    y = broadcast_to(y, output_shape)
    _select = _get_cache_prim(ops.Select)()
    return _select(condition, x, y)

@constexpr
def get_max_value(x, y, z):
    """get max value"""
    if x >= y and x >= z:
        return x
    if y >= x and y >= z:
        return y
    return z

@constexpr
def _calc_broadcast_shape(cond_shape, x_shape, y_shape):
    """Calculate broadcast shape for select"""
    converted_shape = []
    cond_reverse = cond_shape[::-1]
    x_reverse = x_shape[::-1]
    y_reverse = y_shape[::-1]
    max_len = get_max_value(len(cond_reverse), len(x_reverse), len(y_reverse))
    i = 0
    while i < max_len:
        cond_element = 1 if i >= len(cond_reverse) else cond_reverse[i]
        x_element = 1 if i >= len(x_reverse) else x_reverse[i]
        y_element = 1 if i >= len(y_reverse) else y_reverse[i]
        broadcast_element = get_max_value(cond_element, x_element, y_element)
        if cond_element not in (1, broadcast_element):
            raise ValueError(f"For select, condition input can not broadcast at index {i}")
        if x_element not in (1, broadcast_element):
            raise ValueError(f"For select, x input can not broadcast at index {i}")
        if y_element not in (1, broadcast_element):
            raise ValueError(f"For select, y input can not broadcast at index {i}")
        converted_shape.append(broadcast_element)
        i = i + 1
    converted_shape.reverse()
    return tuple(converted_shape)

def broadcast_to(input, shape): # pylint: disable=redefined-outer-name
    """
    Broadcasts input tensor to a given shape. The dim of input shape must be smaller
    than or equal to that of target shape. Suppose input shape is :math:`(x_1, x_2, ..., x_m)`,
    target shape is :math:`(*, y_1, y_2, ..., y_m)`, where :math:`*` means any additional dimension.
    The broadcast rules are as follows:

    Compare the value of :math:`x_m` and :math:`y_m`, :math:`x_{m-1}` and :math:`y_{m-1}`, ...,
    :math:`x_1` and :math:`y_1` consecutively and
    decide whether these shapes are broadcastable and what the broadcast result is.

    If the value pairs at a specific dim are equal, then that value goes right into that dim of output shape.
    With an input shape :math:`(2, 3)`, target shape :math:`(2, 3)` , the inferred output shape is :math:`(2, 3)`.

    If the value pairs are unequal, there are three cases:

    Case 1: If the value of the target shape in the dimension is -1, the value of the
    output shape in the dimension is the value of the corresponding input shape in the dimension.
    With an input shape :math:`(3, 3)`, target
    shape :math:`(-1, 3)`, the output shape is :math:`(3, 3)`.

    Case 2: If the value of target shape in the dimension is not -1, but the corresponding
    value in the input shape is 1, then the corresponding value of the output shape
    is that of the target shape. With an input shape :math:`(1, 3)`, target
    shape :math:`(8, 3)`, the output shape is :math:`(8, 3)`.

    Case 3: If the corresponding values of the two shapes do not satisfy the above cases,
    it means that broadcasting from the input shape to the target shape is not supported.

    So far we got the last m dims of the outshape, now focus on the first :math:`*` dims, there are
    two cases:

    If the first :math:`*` dims of output shape does not have -1 in it, then fill the input
    shape with ones until their length are the same, and then refer to
    Case 2 mentioned above to calculate the output shape. With target shape :math:`(3, 1, 4, 1, 5, 9)`,
    input shape :math:`(1, 5, 9)`, the filled input shape will be :math:`(1, 1, 1, 1, 5, 9)` and thus the
    output shape is :math:`(3, 1, 4, 1, 5, 9)`.

    If the first :math:`*` dims of output shape have -1 in it, it implies this -1 is corresponding to
    a non-existing dim so they're not broadcastable. With target shape :math:`(3, -1, 4, 1, 5, 9)`,
    input shape :math:`(1, 5, 9)`, instead of operating the dim-filling process first, it raises errors directly.

    Args:
        input (Tensor): The input Tensor. Supported types are: float16, float32, int32, int8, uint8, bool.
        shape (tuple): The target shape to broadcast. Can be fully specified, or have -1 in one position
                       where it will be substituted by the input tensor's shape in that position, see example.

    Returns:
        Tensor, with the given `shape` and the same data type as `input`.

    Raises:
        TypeError: If `shape` is not a tuple.
        ValueError: If the target and input shapes are incompatible, or if a - 1 in the target shape is in an invalid
                    location.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> shape = (2, 3)
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> output = ops.broadcast_to(x, shape)
        >>> print(output)
        [[1. 2. 3.]
         [1. 2. 3.]]
        >>> shape = (-1, 2)
        >>> x = Tensor(np.array([[1], [2]]).astype(np.float32))
        >>> output = ops.broadcast_to(x, shape)
        >>> print(output)
        [[1. 1.]
         [2. 2.]]
    """
    _broadcast_to = _get_cache_prim(ops.BroadcastTo)(shape)
    return _broadcast_to(input)

@constexpr
def _canonicalize_axis(axis, ndim):
    """
    Check axes are within the number of dimensions of tensor x and normalize the negative axes.

    Args:
        axis (Union[int, tuple(int), list(int)]): Axes of the tensor.
        ndim (int): The number of dimensions of the tensor.

    Return:
        Axis (Union[int, tuple(int)]). If input is integer, return integer, else tuple.
    """
    if isinstance(axis, int):
        axis = [axis]
    for ax in axis:
        if not isinstance(ax, int):
            raise TypeError(f'axis should be integers, not {type(ax)}')
        if not -ndim <= ax < ndim:
            raise ValueError(f'axis {ax} is out of bounds for array of dimension {ndim}')

    def canonicalizer(ax):
        return ax + ndim if ax < 0 else ax

    axis = [canonicalizer(ax) for ax in axis]
    if all(axis.count(el) <= 1 for el in axis):
        return tuple(sorted(axis)) if len(axis) > 1 else axis[0]
    raise ValueError(f"duplicate axis in {axis}.")


def rank(input_x):
    """
    Returns the rank of a tensor.

    Returns a 0-D int32 Tensor representing the rank of input; the rank of a tensor
    is the number of indices required to uniquely select each element of the tensor.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Returns:
        Tensor. 0-D int32 Tensor representing the rank of input, i.e., :math:`R`. The data type is an int.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_tensor = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.rank(input_tensor)
        >>> print(output)
        2
        >>> print(type(output))
        <class 'int'>
    """
    rank_ = _get_cache_prim(ops.Rank)()
    return rank_(input_x)

@constexpr
def _tuple_setitem(tup, idx, value):
    """
    Returns a tuple with specified `idx` set to `value`.
    """
    tup = list(tup)
    tup[idx] = value
    return tuple(tup)


@constexpr
def _list_comprehensions(obj, item=None, return_tuple=False):
    """
    Generates a new list or tuple by list comprehension.

    Args:
        obj (Union[int, list, tuple]):
            If integer, it will be the length of the returned tuple/list.
        item: The value to be filled. Default: None.
            If None, the values in the new list/tuple are the same as obj
            or range(obj) when obj is integer.
        return_tuple(bool): If true, returns tuple, else returns list.

    Returns:
        List or tuple.
    """
    lst = obj
    if isinstance(obj, int):
        lst = np.arange(obj)
    if item is None:
        res = list(lst)
    else:
        res = [item for _ in lst]
    if return_tuple:
        return tuple(res)
    return res

def _tensor_split_sub_tensors(x, indices_or_sections, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors along the axis according to the given `indices_or_sections`
    with type of tuple or list.
    """
    length_along_dim = x.shape[axis]
    indices_or_sections = tuple(indices_or_sections)
    indices_or_sections += (length_along_dim,)

    sub_tensors = []
    strides = _list_comprehensions(x.ndim, 1, True)
    begin = _list_comprehensions(x.ndim, 0)
    end = _list_comprehensions(x.shape)
    for i, idx in enumerate(indices_or_sections):
        begin[axis] = 0 if i == 0 else indices_or_sections[i - 1]
        end[axis] = idx
        sliced_tensor = strided_slice(x, tuple(begin), tuple(end), strides)
        sub_tensors.append(sliced_tensor)
    return tuple(sub_tensors)


def _tensor_split_sub_int(x, indices_or_sections, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors along the axis according to the given `indices_or_sections`
    with type if int.
    """
    arr_shape = x.shape
    length_along_dim = arr_shape[axis]
    if indices_or_sections > length_along_dim:
        res = ops.Split(axis, length_along_dim)(x)
        indices_or_sections_n = [length_along_dim, length_along_dim + 1]
        res2 = _tensor_split_sub_tensors(x, indices_or_sections_n, axis)
        for _ in np.arange(length_along_dim, indices_or_sections):
            res += tuple(res2)[1:]
    elif length_along_dim % indices_or_sections == 0:
        res = ops.Split(axis, indices_or_sections)(x)
    else:
        num_long_tensor = length_along_dim % indices_or_sections
        num_short_tensor = indices_or_sections - num_long_tensor
        length1 = num_long_tensor * (length_along_dim // indices_or_sections + 1)
        length2 = length_along_dim - length1
        start1 = _list_comprehensions(rank(x), 0, True)
        size1 = _tuple_setitem(arr_shape, axis, length1)
        start2 = _tuple_setitem(start1, axis, length1)
        size2 = _tuple_setitem(arr_shape, axis, length2)
        res = ops.Split(axis, num_long_tensor)(tensor_slice(x, start1, size1)) + \
              ops.Split(axis, num_short_tensor)(tensor_slice(x, start2, size2))
    return res

def strided_slice(input_x,
                  begin,
                  end,
                  strides,
                  begin_mask=0,
                  end_mask=0,
                  ellipsis_mask=0,
                  new_axis_mask=0,
                  shrink_axis_mask=0):
    r"""
    Extracts a strided slice of a Tensor based on `begin/end` index and `strides`.

    This operation extracts a fragment of size (end-begin)/strides from the given 'input_tensor'.
    Starting from the beginning position, the fragment continues adding strides to the index until
    all dimensions are not less than the ending position.

    Note:
        - `begin` 、 `end` and `strides` must have the same shape.
        - `begin` 、 `end` and `strides` are all 1-D Tensor,  and their shape size
          must not greater than the dim of `input_x`.

    During the slicing process, the fragment (end-begin)/strides are extracted from each dimension.

    Example: For Tensor `input_x` with shape :math:`(5, 6, 7)`,
    set `begin`, `end` and `strides` to (1, 3, 2), (3, 5, 6),
    (1, 1, 2) respectively, then elements from index 1 to 3 are extrected for dim 0, index 3 to 5
    are extrected for dim 1 and index 2 to 6 with a `stirded` of 2 are extrected for dim 2, this
    process is equivalent to a pythonic slice `input_x[1:3, 3:5, 2:6:2]`.

    If the length of `begin` 、 `end` and `strides` is smaller than the dim of `input_x`,
    then all elements are extracted from the missing dims, it behaves like all the
    missing dims are filled with zeros, size of that missing dim and ones.

    Example: For Tensor `input_x` with shape :math:`(5, 6, 7)`,
    set `begin`, `end` and `strides` to (1, 3),
    (3, 5), (1, 1) respectively, then elements from index 1 to 3 are extrected
    for dim 0, index 3 to 5 are extrected for dim 1 and index 3 to 5 are extrected
    for dim 2, this process is equivalent to a pythonic slice `input_x[1:3, 3:5, 0:7]`.

    Here's how a mask works:
    For each specific mask, it will be converted to a binary representation internally, and then
    reverse the result to start the calculation. For Tensor `input_x` with
    shape :math:`(5, 6, 7)`. Given mask value of 3 which
    can be represented as 0b011. Reverse that we get 0b110, which implies the first and second dim of the
    original Tensor will be effected by this mask. See examples below, for simplicity all mask mentioned
    below are all in their reverted binary form:

    - `begin_mask` and `end_mask`

      If the ith bit of `begin_mask` is 1, `begin[i]` is ignored and the fullest
      possible range in that dimension is used instead. `end_mask` is analogous,
      except with the end range. For Tensor `input_x` with shape :math:`(5, 6, 7, 8)`,  if `begin_mask`
      is 0b110, `end_mask` is 0b011, the slice `input_x[0:3, 0:6, 2:7:2]` is produced.

    - `ellipsis_mask`

      If the ith bit of `ellipsis_mask` is 1, as many unspecified dimensions as needed
      will be inserted between other dimensions. Only one non-zero bit is allowed
      in `ellipsis_mask`. For a 5*6*7*8 Tensor `input_x`,  `input_x[2:,...,:6]`
      is equivalent to `input_x[2:5,:,:,0:6]` ,  `input_x[2:,...]` is equivalent
      to `input_x[2:5,:,:,:]`.

    - `new_axis_mask`

      If the ith bit of `new_axis_mask` is 1, `begin`, `end` and `strides` are
      ignored and a new length 1 dimension is added at the specified position
      in the output Tensor. For Tensor `input_x` with shape :math:`(5, 6, 7)`, if `new_axis_mask`
      is 0b110,  a new dim is added to the second dim, which will produce
      a Tensor with shape :math:`(5, 1, 6, 7)`.

    - `shrink_axis_mask`

      If the ith bit of `shrink_axis_mask` is 1, `begin`, `end` and `strides`
      are ignored and dimension i will be shrunk to 0.
      For Tensor `input_x` with shape :math:`(5, 6, 7)`,
      if `shrink_axis_mask` is 0b010, it is equivalent to slice `x[:, 5, :]`
      and results in an output shape of :math:`(5, 7)`.

    Note:
        `new_axis_mask` and  `shrink_axis_mask` are not recommended to
        use at the same time, it might incur unexpected result.

    Args:
        input_x (Tensor): The input Tensor to be extracted from.
        begin (tuple[int]): A tuple which represents the location where to start.
            Only non-negative int is allowed.
        end (tuple[int]): A tuple or which represents the maximum location where to end.
            Only non-negative int is allowed.
        strides (tuple[int]): A tuple which represents the strides is continuously added
            before reaching the maximum location. Only int is allowed, it can be negative
            which results in reversed slicing.
        begin_mask (int, optional): Starting index of the slice. Default: 0.
        end_mask (int, optional): Ending index of the slice. Default: 0.
        ellipsis_mask (int, optional): An int mask, ignore slicing operation when set to 1. Default: 0.
        new_axis_mask (int, optional): An int mask for adding new dims. Default: 0.
        shrink_axis_mask (int, optional): An int mask for shrinking dims. Default: 0.

    Returns:
        Tensor, return the extracts a strided slice of a Tensor based on `begin/end` index and `strides`.

    Raises:
        TypeError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or
            `shrink_axis_mask` is not an int.
        TypeError: If `begin`, `end` or `strides` is not tuple[int].
        ValueError: If `begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask` or
            `shrink_axis_mask` is less than 0.
        ValueError: If `begin`, `end` and `strides` have different shapes.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
        ...                   [[5, 5, 5], [6, 6, 6]]], mindspore.float32)
        >>> output = ops.strided_slice(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1))
        >>> # Take this " output = strided_slice(input_x, (1, 0, 2), (3, 1, 3), (1, 1, 1)) " as an example,
        >>> # start = [1, 0, 2] , end = [3, 1, 3], strides = [1, 1, 1], Find a segment of (start, end),
        >>> # note that end is an open interval
        >>> # To facilitate understanding, this operator can be divided into three steps:
        >>> # Step 1: Calculation of the first dimension:
        >>> # start = 1, end = 3, strides = 1, So can take 1st, 2nd rows, and then gets the final output at this time.
        >>> # output_1th =
        >>> # [
        >>> #     [
        >>> #         [3,3,3]
        >>> #         [4,4,4]
        >>> #     ]
        >>> #     [
        >>> #         [5,5,5]
        >>> #         [6,6,6]
        >>> #     ]
        >>> # ]
        >>> # Step 2: Calculation of the second dimension
        >>> # 2nd dimension, start = 0, end = 1, strides = 1. So only 0th rows
        >>> # can be taken, and the output at this time.
        >>> # output_2nd =
        >>> # [
        >>> #     [
        >>> #         [3,3,3]
        >>> #     ]
        >>> #     [
        >>> #         [5,5,5]
        >>> #     ]
        >>> # ]
        >>> # Step 3: Calculation of the third dimension
        >>> # 3nd dimension,start = 2, end = 3, strides = 1, So can take 2th cols,
        >>> # and you get the final output at this time.
        >>> # output_3ed =
        >>> # [
        >>> #     [
        >>> #         [3]
        >>> #     ]
        >>> #     [
        >>> #         [5]
        >>> #     ]
        >>> # ]
        >>> # The final output after finishing is:
        >>> print(output)
        [[[3.]]
         [[5.]]]
        >>> # another example like :
        >>> output = strided_slice(input_x, (1, 0, 0), (2, 1, 3), (1, 1, 1))
        >>> print(output)
        [[[3. 3. 3.]]]
    """
    strided_slice_ = _get_cache_prim(ops.StridedSlice)(
        begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    return strided_slice_(input_x, begin, end, strides)


def tensor_split(input, indices_or_sections, axis=0):
    r"""
    Splits a tensor into multiple sub-tensors along the given axis.

    Args:
        input (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):

            - If `indices_or_sections` is an integer n, input tensor will be split into n sections.

              - If :math:`input.shape(axis)` can be divisible by n, sub-sections will have equal size
                :math:`input.shape(axis) / n` .
              - If :math:`input.shape(axis)` is not divisible by n, the first :math:`input.shape(axis) % n` sections
                will have size :math:`x.size(axis) // n + 1` , and the rest will have
                size :math:`input.shape(axis) // n` .

            - If `indices_or_sections` is of type tuple(int) or list(int), the input tensor will be split at the
              indices in the list or tuple. For example, given parameters :math:`indices\_or\_sections=[1, 4]`
              and :math:`axis=0` , the input tensor will be split into sections :math:`input[:1]` ,
              :math:`input[1:4]` , and :math:`input[4:]` .

        axis (int): The axis along which to split. Default: 0.

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `input` is not Tensor.
        TypeError: If argument `axis` is not int.
        ValueError: If argument `axis` is out of range of :math:`[-input.ndim, input.ndim)` .
        TypeError: If each element in 'indices_or_sections' is not integer.
        TypeError: If argument `indices_or_sections` is not int, tuple(int) or list(int).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.tensor_split(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
        Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
        Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    if not isinstance(input, Tensor):
        raise TypeError(f'expect `x` is a Tensor, but got {type(input)}')

    if not isinstance(axis, int):
        raise TypeError(f"Type of Argument `axis` should be integer but got {type(axis)}")
    handle_axis = _canonicalize_axis(axis, input.ndim)
    if isinstance(indices_or_sections, int):
        if indices_or_sections > 0:
            res = _tensor_split_sub_int(input, indices_or_sections, handle_axis)
        else:
            raise ValueError(f"For tensor_split, the value of 'indices_or_sections' must be more than zero "
                             f"but got {indices_or_sections}")
    elif isinstance(indices_or_sections, (list, tuple)):
        for item in indices_or_sections:
            if not isinstance(item, int):
                raise TypeError(f"Each element in 'indices_or_sections' should be integer, but got {type(item)}.")
        res = _tensor_split_sub_tensors(input, indices_or_sections, handle_axis)
    else:
        raise TypeError(f"Type of Argument `indices_or_sections` should be integer, tuple(int) or list(int), " \
                        f"but got {type(indices_or_sections)}")

    return res

def sigmoid(x):
    """inner sigmoid"""
    sigmoid_ = _get_cache_prim(ops.Sigmoid)()
    return sigmoid_(x)

def reverse(x, axis):
    """
    Reverses specific dimensions of a tensor.

    .. warning::
        The value range of "axis" is [-dims, dims - 1]. "dims" is the dimension length of "input_x".

    Args:
        x (Tensor): The target tensor. The data type is Number except float64.
            The shape is :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        axis (Union[tuple(int), list(int)]): The indices of the dimensions to reverse.

    Outputs:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `axis` is neither list nor tuple.
        TypeError: If element of `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.int32)
        >>> output = ops.reverse(input_x, axis=[1])
        >>> print(output)
        [[4 3 2 1]
         [8 7 6 5]]
        >>> input_x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.int32)
        >>> output = ops.reverse(input_x, axis=[1, 0])
        >>> print(output)
        [[8 7 6 5]
         [4 3 2 1]]
    """
    axis = axis[0]
    dim_size = x.shape[axis]
    reversed_indexes = arange(dim_size-1, -1, -1)
    _gather = _get_cache_prim(ops.Gather)()
    output = _gather(x, reversed_indexes, axis)
    return output
