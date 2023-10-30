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
# pylint: disable=C0103
"""
Jax like functional apis for graph mode.
"""
import numpy as np
import mindspore
from mindspore import Tensor, ops
from mindspore.ops import constexpr


@constexpr
def finfo(dtype, attr="min"):
    """finfo api to get dtype attributes."""
    info = np.finfo(mindspore.dtype_to_nptype(dtype))
    if attr == "min":
        return Tensor(info.min, dtype)
    if attr == "max":
        return Tensor(info.max, dtype)
    return Tensor(0, dtype)


def make_attention_mask(
    query_input: Tensor,
    key_input: Tensor,
    dtype=mindspore.float32,
):
    """Mask-making helper for attention weights.

    In case of 1d inputs (i.e., `[batch..., len_q]`, `[batch..., len_kv]`, the
    attention weights will be `[batch..., heads, len_q, len_kv]` and this
    function will produce `[batch..., 1, len_q, len_kv]`.

    Args:
      query_input: a batched, flat input of query_length size
      key_input: a batched, flat input of key_length size
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len_q, len_kv]` shaped mask for 1d attention.
    """
    mask = ops.greater_equal(
        ops.expand_dims(query_input, axis=-1), ops.expand_dims(key_input, axis=-2)
    )
    mask = ops.expand_dims(mask, axis=-3)
    return mask.astype(dtype)


def make_causal_mask(
    x: Tensor, dtype=mindspore.float32
) -> Tensor:
    """Make a causal mask for self-attention.

    In case of 1d inputs (i.e., `[batch..., len]`, the self-attention weights
    will be `[batch..., heads, len, len]` and this function will produce a
    causal mask of shape `[batch..., 1, len, len]`.

    Args:
      x: input array of shape `[batch..., len]`
      extra_batch_dims: number of batch dims to add singleton axes for, none by
        default
      dtype: mask return dtype

    Returns:
      A `[batch..., 1, len, len]` shaped causal mask for 1d attention.
    """
    idxs = ops.broadcast_to(ops.arange(x.shape[-1], dtype=mindspore.int32), x.shape)
    return make_attention_mask(
        idxs,
        idxs,
        dtype=dtype,
    )
