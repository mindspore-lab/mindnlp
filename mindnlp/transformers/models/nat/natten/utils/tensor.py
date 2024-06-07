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
from torch import Size, Tensor


def _get_expected_attn_shape(input_tensor: Tensor, attention_dim: int) -> Size:
    shape = [x for x in input_tensor.shape[:-1]] + [attention_dim]
    return Size(shape)


def make_attn_tensor_from_input(input_tensor: Tensor, attention_dim: int) -> Tensor:
    return torch.empty(
        _get_expected_attn_shape(input_tensor, attention_dim),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
        requires_grad=input_tensor.requires_grad,
    )


def check_additional_keys(
    input_tensor: Tensor, additional_keys: Optional[Tensor]
) -> int:
    if additional_keys is None:
        return 0

    if additional_keys.dim() != 4:
        raise ValueError(
            "Additional tokens have to be shaped as a rank-4 tensor; "
            f"got {additional_keys.dim()}."
        )
    batch_size, heads, tokens, dim = additional_keys.shape
    expected_batch_size = input_tensor.shape[0]
    expected_heads = input_tensor.shape[1]
    expected_dim = input_tensor.shape[-1]
    if (
        batch_size != expected_batch_size
        or expected_heads != heads
        or expected_dim != dim
    ):
        raise ValueError(
            "Shape mismatch between input tensor and additional tokens; "
            "they must match in batch size, heads, and dim per head. "
            f"Got {input_tensor.shape=}, {additional_keys.shape=}."
        )
    return tokens


def check_additional_values(
    attn_tensor: Tensor,
    additional_values: Optional[Tensor],
    value: Tensor,
    expected_attn_weights: int,
) -> int:
    if additional_values is None and attn_tensor.shape[-1] == expected_attn_weights:
        return 0
    if additional_values is None:
        raise ValueError(
            f"Expected {expected_attn_weights} attention weights per token, "
            f"got {attn_tensor.shape[-1]=}."
        )

    if additional_values.dim() != 4:
        raise ValueError(
            "Additional tokens have to be shaped as a rank-4 tensor; "
            f"got {additional_values.dim()}."
        )

    if additional_values.shape[-1] != value.shape[-1]:
        raise ValueError(
            "Additional value tokens must match the dimension of the "
            f"rest of the tokens, got {additional_values.shape[-1]=} != "
            f"{value.shape[-1]=}."
        )

    batch_size, heads, tokens, dim = additional_values.shape
    if tokens + expected_attn_weights != attn_tensor.shape[-1]:
        raise ValueError(
            f"Expected {expected_attn_weights + tokens} attention weights per token, "
            f"got {attn_tensor.shape[-1]=}."
        )
    expected_batch_size = attn_tensor.shape[0]
    expected_heads = attn_tensor.shape[1]
    if batch_size != expected_batch_size or expected_heads != heads:
        raise ValueError(
            "Shape mismatch between attention tensor and additional tokens; "
            "they must match in batch size and heads. "
            f"Got {attn_tensor.shape=}, {additional_values.shape=}."
        )
    return tokens
