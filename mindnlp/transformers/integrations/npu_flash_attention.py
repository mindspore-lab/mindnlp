# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import math
import mindspore
from mindspore.ops import flash_attention_score
from mindspore import nn
from typing import Optional, Tuple
from mindnlp.core import ops


# FlashAttention2 is supported on Ascend NPU with down-right aligned causal mask by default.
# Set environment variable `NPU_FA2_SPARSE_MODE` to 2 when using top-left aligned causal mask.
TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", default=DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE))
if SPARSE_MODE not in [TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE, DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE]:
    raise ValueError(
        "Environment variable `NPU_FA2_SPARSE_MODE` can only be set as 2 (top-left aligned causal mask) "
        "or 3 (down-right aligned causal mask)."
    )


def is_npu_fa2_top_left_aligned_causal_mask():
    return SPARSE_MODE == TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE


class IndexFirstAxis(nn.Cell):
    def __init__(self):
        super(IndexFirstAxis, self).__init__()

    def construct(self, input: mindspore.Tensor, indices: mindspore.Tensor):
        assert input.ndim >= 2
        first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        input_flat = input.reshape(first_axis_dim, -1)
        indices_expanded = ops.expand_dims(indices, -1)
        indices_expanded = ops.broadcast_to(indices_expanded, (-1, input_flat.shape[1]))
        output_flat = ops.gather(input_flat, 0, indices_expanded)
        output = output_flat.reshape(-1, *other_shape)
        return output

    def bprop(self, input, indices, out, dout):
        assert dout.ndim >= 2
        other_shape = dout.shape[1:]
        grad_output = dout
        
        grad_flat = grad_output.reshape(grad_output.shape[0], -1)
        grad_shape = (input.shape[0], grad_flat.shape[1])
        grad_input = ops.zeros(grad_shape, grad_flat.dtype)
        
        indices_expanded = ops.expand_dims(indices, -1)
        indices_expanded = ops.broadcast_to(indices_expanded, (-1, grad_flat.shape[1]))
        grad_input.scatter_(0, indices_expanded, grad_flat)
        
        return grad_input.reshape(input.shape[0], *other_shape), None


index_first_axis = IndexFirstAxis()


class IndexPutFirstAxis(nn.Cell):
    def __init__(self):
        super(IndexPutFirstAxis, self).__init__()

    def construct(self, values: mindspore.Tensor, indices: mindspore.Tensor, first_axis_dim: int):
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = ops.zeros(
            (first_axis_dim, *values.shape[1:]),
            values.dtype
        )
        output[indices] = values
        return output

    def bprop(self, values, indices, first_axis_dim, out, dout):
        grad_values = dout[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis()


def pad_input(
    hidden_states: mindspore.Tensor,
    indices: mindspore.Tensor,
    batch: int,
    seqlen: int
):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def unpad_input(
    hidden_states: mindspore.Tensor,
    attention_mask: mindspore.Tensor,
    unused_mask: Optional[mindspore.Tensor] = None,
):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=mindspore.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=mindspore.int32)
    indices = ops.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = ops.pad(ops.cumsum(seqlens_in_batch, dim=0, dtype=mindspore.int32), (1, 0))

    hidden_states_flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    hidden_states = index_first_axis(hidden_states_flat, indices)
    return (
        hidden_states,
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def create_attn_mask(causal: bool, sparse_mode: int) -> Tuple[int, mindspore.Tensor]:
    """
    Create a causal mask for the attention scores.

    Args:
        causal (`bool`):
            If `True`, the mask will be causal.
        sparse_mode (`bool`):
            If `True`, the mask will be top-left
            aligned, otherwise it will be bottom-right aligned.
    Returns:
        `Tuple[bool, mindspore.Tensor]`:
            A tuple containing sparse_mode and the mask tensor.
    """
    if not causal:
        sparse_mode = 0
        attn_mask = None
    else:
        if sparse_mode == TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE:
            attn_mask = ops.tril(ops.ones((2048, 2048)), diagonal=-1).bool()
        else:
            attn_mask = ops.triu(ops.ones((2048, 2048)), diagonal=1).bool()
    return sparse_mode, attn_mask


def npu_flash_attn_func(
    q: mindspore.Tensor,
    k: mindspore.Tensor,
    v: mindspore.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
):
    head_num = q.shape[2]
    sparse_mode, attn_mask = create_attn_mask(causal, SPARSE_MODE)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    output = flash_attention_score(
        q,
        k,
        v,
        head_num,
        keep_prob=1.0 - dropout_p,
        scalar_value=softmax_scale,
        attn_mask=attn_mask,
        input_layout="BSND",
        sparse_mode=sparse_mode,
        prefix=None,
    )

    return output


def npu_flash_attn_varlen_func(
    q: mindspore.Tensor,
    k: mindspore.Tensor,
    v: mindspore.Tensor,
    cu_seqlens_q: Optional[mindspore.Tensor] = None,
    cu_seqlens_k: Optional[mindspore.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
):
    head_num = q.shape[1]
    sparse_mode, attn_mask = create_attn_mask(causal, SPARSE_MODE)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    output = flash_attention_score(
        q,
        k,
        v,
        head_num,
        keep_prob=1.0 - dropout_p,
        scalar_value=softmax_scale,
        attn_mask=attn_mask,
        input_layout="TND",
        actual_seq_qlen=cu_seqlens_q[1:].asnumpy().tolist(),
        actual_seq_kvlen=cu_seqlens_k[1:].asnumpy().tolist(),
        sparse_mode=sparse_mode,
        prefix=None,
    )

    return output
