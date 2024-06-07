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
from torch import Tensor

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import NATTEN's CPP backend. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )

from .ops import av_cross_forward, qk_cross_forward
from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
    ListOrNestedTensor,
)
from .utils import (
    check_additional_keys,
    check_additional_values,
    check_all_args,
    get_num_na_weights,
    make_attn_tensor_from_input,
)


def na1d_qk_nested(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed,
    additional_keys: Optional[Tensor] = None,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
) -> Tensor:
    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )
    num_na_weights = get_num_na_weights(kernel_size_)

    if any(is_causal_) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    if not query.is_nested or not key.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not query.is_leaf or not key.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if query.requires_grad or key.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")
    if bias is not None and bias.is_nested:
        raise ValueError("Positional biases cannot be nested.")

    if bias is not None:
        bias = bias.contiguous().to(key.dtype)

    if query.size(0) != key.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    n_add_tokens_list = []
    if additional_keys is not None and additional_keys.is_nested:
        if additional_keys.size(0) != query.size(0):
            raise ValueError("Got nested inputs, but they don't match in size.")
        n_add_tokens_list = [
            check_additional_keys(q, k) for q, k in zip(query, additional_keys)
        ]
    elif additional_keys is not None:
        # n_add_tokens_list = [check_additional_keys(q, additional_keys) for q in query]
        # Banning this for now, because it will be much more complicated to check
        # tensor sizes against each other, when they can exist in both nested and non
        # nested format.
        raise ValueError(
            "Expected all inputs to be nested, but "
            "additional keys were passed and are not "
            "nested."
        )

    additional_keys_list: ListOrNestedTensor = (
        [None for _ in range(query.size(0))]
        if additional_keys is None
        else additional_keys
    )

    attn = torch.nested.nested_tensor(
        [
            make_attn_tensor_from_input(q, num_na_weights + n_add_tokens)
            for q, n_add_tokens in zip(query, n_add_tokens_list)
        ]
    )
    for q, k, a, k_add in zip(query, key, attn, additional_keys_list):
        attn_na, attn_add = a.split(
            [num_na_weights, a.shape[-1] - num_na_weights], dim=-1
        )
        libnatten.na1d_qk_forward(
            attn_na, q, k, bias, kernel_size_, dilation_, is_causal_
        )

        if len(n_add_tokens_list):
            assert k_add is not None and attn_add.numel() > 0
            qk_cross_forward(q, k_add, attn_add)

    return attn


def na1d_av_nested(
    attn: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed,
    additional_values: Optional[Tensor] = None,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
):
    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )
    num_na_weights = get_num_na_weights(kernel_size_)

    if not attn.is_nested or not value.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not attn.is_leaf or not value.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if attn.requires_grad or value.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")

    if attn.size(0) != value.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    if additional_values is not None and additional_values.is_nested:
        if additional_values.size(0) != attn.size(0):
            raise ValueError("Got nested inputs, but they don't match in size.")
        for a, v, v_add in zip(attn, value, additional_values):
            check_additional_values(a, v_add, v, num_na_weights)
    elif additional_values is not None:
        # [check_additional_values(a, additional_values, v, num_na_weights) for a, v in zip(attn, value)]
        # Banning this for now, because it will be much more complicated to check
        # tensor sizes against each other, when they can exist in both nested and non
        # nested format.
        raise ValueError(
            "Expected all inputs to be nested, but "
            "additional values were passed and are not "
            "nested."
        )

    attn = attn.to(value.dtype)
    out = torch.empty_like(value)
    additional_values_list: ListOrNestedTensor = (
        [None for _ in range(attn.size(0))]
        if additional_values is None
        else additional_values
    )
    additional_outputs_list: ListOrNestedTensor = (
        [None for _ in range(attn.size(0))]
        if additional_values is None
        else torch.empty_like(out)
    )

    for a, v, o, v_add, o_add in zip(
        attn, value, out, additional_values_list, additional_outputs_list
    ):
        attn_na, attn_add = a.split(
            [num_na_weights, a.shape[-1] - num_na_weights], dim=-1
        )
        libnatten.na1d_av_forward(o, attn_na, v, kernel_size_, dilation_, is_causal_)

        if v_add is not None and o_add is not None:
            assert attn_add.numel() > 0
            av_cross_forward(attn_add, v_add, o_add)
            o += o_add

    return out


def na2d_qk_nested(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed,
    additional_keys: Optional[Tensor] = None,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
) -> Tensor:
    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )
    num_na_weights = get_num_na_weights(kernel_size_)

    if any(is_causal_) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    if not query.is_nested or not key.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not query.is_leaf or not key.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if query.requires_grad or key.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")
    if bias is not None and bias.is_nested:
        raise ValueError("Positional biases cannot be nested.")

    if bias is not None:
        bias = bias.contiguous().to(key.dtype)

    if query.size(0) != key.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    n_add_tokens_list = []
    if additional_keys is not None and additional_keys.is_nested:
        if additional_keys.size(0) != query.size(0):
            raise ValueError("Got nested inputs, but they don't match in size.")
        n_add_tokens_list = [
            check_additional_keys(q, k) for q, k in zip(query, additional_keys)
        ]
    elif additional_keys is not None:
        # n_add_tokens_list = [check_additional_keys(q, additional_keys) for q in query]
        # Banning this for now, because it will be much more complicated to check
        # tensor sizes against each other, when they can exist in both nested and non
        # nested format.
        raise ValueError(
            "Expected all inputs to be nested, but "
            "additional keys were passed and are not "
            "nested."
        )

    additional_keys_list: ListOrNestedTensor = (
        [None for _ in range(query.size(0))]
        if additional_keys is None
        else additional_keys
    )
    attn = torch.nested.nested_tensor(
        [
            make_attn_tensor_from_input(q, num_na_weights + n_add_tokens)
            for q, n_add_tokens in zip(query, n_add_tokens_list)
        ]
    )

    for q, k, a, k_add in zip(query, key, attn, additional_keys_list):
        attn_na, attn_add = a.split(
            [num_na_weights, a.shape[-1] - num_na_weights], dim=-1
        )
        libnatten.na2d_qk_forward(
            attn_na, q, k, bias, kernel_size_, dilation_, is_causal_
        )

        if len(n_add_tokens_list):
            assert k_add is not None and attn_add.numel() > 0
            qk_cross_forward(q, k_add, attn_add)

    return attn


def na2d_av_nested(
    attn: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed,
    additional_values: Optional[Tensor] = None,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
):
    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )
    num_na_weights = get_num_na_weights(kernel_size_)

    if not attn.is_nested or not value.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not attn.is_leaf or not value.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if attn.requires_grad or value.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")

    if attn.size(0) != value.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    if additional_values is not None and additional_values.is_nested:
        if additional_values.size(0) != attn.size(0):
            raise ValueError("Got nested inputs, but they don't match in size.")
        for a, v, v_add in zip(attn, value, additional_values):
            check_additional_values(a, v_add, v, num_na_weights)
    elif additional_values is not None:
        # [check_additional_values(a, additional_values, v, num_na_weights) for a, v in zip(attn, value)]
        # Banning this for now, because it will be much more complicated to check
        # tensor sizes against each other, when they can exist in both nested and non
        # nested format.
        raise ValueError(
            "Expected all inputs to be nested, but "
            "additional values were passed and are not "
            "nested."
        )

    attn = attn.to(value.dtype)
    out = torch.empty_like(value)
    additional_values_list: ListOrNestedTensor = (
        [None for _ in range(attn.size(0))]
        if additional_values is None
        else additional_values
    )
    additional_outputs_list: ListOrNestedTensor = (
        [None for _ in range(attn.size(0))]
        if additional_values is None
        else torch.empty_like(out)
    )

    for a, v, o, v_add, o_add in zip(
        attn, value, out, additional_values_list, additional_outputs_list
    ):
        attn_na, attn_add = a.split(
            [num_na_weights, a.shape[-1] - num_na_weights], dim=-1
        )
        libnatten.na2d_av_forward(o, attn_na, v, kernel_size_, dilation_, is_causal_)

        if v_add is not None and o_add is not None:
            assert attn_add.numel() > 0
            av_cross_forward(attn_add, v_add, o_add)
            o += o_add

    return out


def na3d_qk_nested(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed,
    additional_keys: Optional[Tensor] = None,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
) -> Tensor:
    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )
    num_na_weights = get_num_na_weights(kernel_size_)

    if any(is_causal_) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    if not query.is_nested or not key.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not query.is_leaf or not key.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if query.requires_grad or key.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")
    if bias is not None and bias.is_nested:
        raise ValueError("Positional biases cannot be nested.")

    if bias is not None:
        bias = bias.contiguous().to(key.dtype)

    if query.size(0) != key.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    n_add_tokens_list = []
    if additional_keys is not None and additional_keys.is_nested:
        if additional_keys.size(0) != query.size(0):
            raise ValueError("Got nested inputs, but they don't match in size.")
        n_add_tokens_list = [
            check_additional_keys(q, k) for q, k in zip(query, additional_keys)
        ]
    elif additional_keys is not None:
        # n_add_tokens_list = [check_additional_keys(q, additional_keys) for q in query]
        # Banning this for now, because it will be much more complicated to check
        # tensor sizes against each other, when they can exist in both nested and non
        # nested format.
        raise ValueError(
            "Expected all inputs to be nested, but "
            "additional keys were passed and are not "
            "nested."
        )

    additional_keys_list: ListOrNestedTensor = (
        [None for _ in range(query.size(0))]
        if additional_keys is None
        else additional_keys
    )
    attn = torch.nested.nested_tensor(
        [
            make_attn_tensor_from_input(q, num_na_weights + n_add_tokens)
            for q, n_add_tokens in zip(query, n_add_tokens_list)
        ]
    )

    for q, k, a, k_add in zip(query, key, attn, additional_keys_list):
        attn_na, attn_add = a.split(
            [num_na_weights, a.shape[-1] - num_na_weights], dim=-1
        )
        libnatten.na3d_qk_forward(
            attn_na, q, k, bias, kernel_size_, dilation_, is_causal_
        )

        if len(n_add_tokens_list):
            assert k_add is not None and attn_add.numel() > 0
            qk_cross_forward(q, k_add, attn_add)

    return attn


def na3d_av_nested(
    attn: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed,
    additional_values: Optional[Tensor] = None,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
):
    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )
    num_na_weights = get_num_na_weights(kernel_size_)

    if not attn.is_nested or not value.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not attn.is_leaf or not value.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if attn.requires_grad or value.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")

    if attn.size(0) != value.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    if additional_values is not None and additional_values.is_nested:
        if additional_values.size(0) != attn.size(0):
            raise ValueError("Got nested inputs, but they don't match in size.")
        for a, v, v_add in zip(attn, value, additional_values):
            check_additional_values(a, v_add, v, num_na_weights)
    elif additional_values is not None:
        # [check_additional_values(a, additional_values, v, num_na_weights) for a, v in zip(attn, value)]
        # Banning this for now, because it will be much more complicated to check
        # tensor sizes against each other, when they can exist in both nested and non
        # nested format.
        raise ValueError(
            "Expected all inputs to be nested, but "
            "additional values were passed and are not "
            "nested."
        )

    attn = attn.to(value.dtype)
    out = torch.empty_like(value)
    additional_values_list: ListOrNestedTensor = (
        [None for _ in range(attn.size(0))]
        if additional_values is None
        else additional_values
    )
    additional_outputs_list: ListOrNestedTensor = (
        [None for _ in range(attn.size(0))]
        if additional_values is None
        else torch.empty_like(out)
    )

    for a, v, o, v_add, o_add in zip(
        attn, value, out, additional_values_list, additional_outputs_list
    ):
        attn_na, attn_add = a.split(
            [num_na_weights, a.shape[-1] - num_na_weights], dim=-1
        )
        libnatten.na3d_av_forward(o, attn_na, v, kernel_size_, dilation_, is_causal_)

        if v_add is not None and o_add is not None:
            assert attn_add.numel() > 0
            av_cross_forward(attn_add, v_add, o_add)
            o += o_add

    return out
