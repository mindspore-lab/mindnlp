# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
# Copyright 2023 Huawei Technologies Co., Ltd
#
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
"""MindSpore Longformer model."""
# pylint: disable=relative-beyond-top-level
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
# pylint: disable=C0302
import inspect
import math
from typing import List, Set, Tuple, Callable, Optional, Union

import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from ..utils.activations import ACT2FN
from .longformer_config import LongformerConfig
from ..utils import logging
from ...abc import PreTrainedModel
logger = logging.get_logger(__name__)

def apply_chunking_to_forward(
    forward_fn: Callable[..., mindspore.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> mindspore.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts
    of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield
    the same result as directly applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn`
         would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head
        , self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method
    # -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return ops.cat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], mindspore.Tensor]:
    # qbh longTensor->tensor
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = mindspore.ops.ones((n_heads, head_size))
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).equal(1)
    index: mindspore.Tensor = mindspore.ops.arange(len(mask))[mask].long()
    return heads, index


def prune_linear_layer(layer: nn.Dense, index: mindspore.Tensor, dim: int = 0) -> nn.Dense:# qbh longtensor->tensor
    """
    Prune a linear layer to keep only entries in index.
    """
    W = layer.weight.index_select(dim, index).copy()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.copy()
        else:
            b = layer.bias[index].copy()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Dense(in_channels=new_size[1], out_channels=new_size[0], has_bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W)
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b)
        new_layer.bias.requires_grad = True
    return new_layer


def tensor_to_tuple(self):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """
    ans = ()
    for i in range(self.shape[1]):
        if self.shape[0] == 0:
            ans = ans + (mindspore.Tensor([], dtype=mindspore.int64), )
        else:
            ans = ans + (self[:, i:i+1].reshape(-1), )
    return ans


def as_strided(self, size, stride, storage_offset=None):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """
    input_ms = self
    # if len(size) != len(stride):
    #     raise RuntimeError("mismatch in length of strides and shape.")
    index = np.arange(0, size[0] * stride[0], stride[0])
    for i in range(1, len(size)):
        tmp = np.arange(0, size[i] * stride[i], stride[i])
        index = np.expand_dims(index, -1)
        index = index + tmp
    if storage_offset is not None:
        index = index + storage_offset
    input_indices = mindspore.Tensor(index)
    out = mindspore.ops.gather(input_ms.reshape(-1), input_indices, 0)
    out = mindspore.Tensor(out)
    return out


def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurrence of `sep_token_id`.
    """

    sep_token_indices = (input_ids == sep_token_id).nonzero()
    batch_size = input_ids.shape[0]

    assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
    assert sep_token_indices.shape[0] == 3 * batch_size, (
        f"There should be exactly three separator tokens: {sep_token_id} "
        f"in every sample for questions answering. You"
        " might also consider to set `global_attention_mask` manually i"
        "n the forward function to avoid this error."
    )
    return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]


def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    """
    Computes global attention mask by putting attention on all
    tokens before `sep_token_id` if `before_sep_token is
    True` else after `sep_token_id`.
    """
    question_end_index = _get_question_end_index(input_ids, sep_token_id)
    question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
    # bool attention mask with True in locations of global attention
    attention_mask = mindspore.numpy.arange(input_ids.shape[1])  # qbh delete device
    if before_sep_token is True:
        attention_mask = (attention_mask.expand_as(input_ids) < question_end_index).to(mindspore.uint8)
    else:
        # last token is separation token and should not be counted and in the middle are two separation tokens
        attention_mask = (attention_mask.expand_as(input_ids) > (question_end_index + 1)).to(mindspore.uint8) * (
            attention_mask.expand_as(input_ids) < input_ids.shape[-1]
        ).to(mindspore.uint8)

    return attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:
_
    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = mindspore.ops.not_equal(input_ids, padding_idx).astype(mindspore.int32)
    incremental_indices = mindspore.ops.cumsum(mask, axis=1, dtype=mask.dtype) * mask
    return incremental_indices.astype(mindspore.int32) + padding_idx


class LongformerEmbeddings(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(normalized_shape=(config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def construct(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        """forward"""
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = mindspore.ops.zeros(input_shape, dtype=mindspore.int64)  # delete device

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor inputs_embeds:

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = mindspore.numpy.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64  # delete device
        )
        return position_ids.unsqueeze(0).broadcast_to(input_shape)


class LongformerSelfAttention(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = None
        self.all_head_size = None
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Dense(config.hidden_size, self.embed_dim)
        self.key = nn.Dense(config.hidden_size, self.embed_dim)
        self.value = nn.Dense(config.hidden_size, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Dense(config.hidden_size, self.embed_dim)
        self.key_global = nn.Dense(config.hidden_size, self.embed_dim)
        self.value_global = nn.Dense(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2
        self.config = config

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        """
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states = hidden_states.swapaxes(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.shape
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).swapaxes(0, 1)
        query_vectors = mindspore.Tensor(query_vectors)

        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).swapaxes(0, 1)
        key_vectors = mindspore.Tensor(key_vectors)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf

        float_mask = remove_from_windowed_attention_mask.astype(query_vectors.dtype).masked_fill(
            remove_from_windowed_attention_mask,
            Tensor(np.finfo(mindspore.dtype_to_nptype(query_vectors.dtype)).min, dtype=query_vectors.dtype)
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.shape), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask
        assert list(attn_scores.shape) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key


            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = mindspore.ops.cat((global_key_attn_scores, attn_scores), axis=-1)

            # free memory
            del global_key_attn_scores

        attn_probs = mindspore.ops.softmax(
            attn_scores, axis=-1
        )  # use fp32 for numerical stability
        attn_probs = mindspore.Tensor(attn_probs, dtype=mindspore.float32)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_probs = layer_head_mask.view(1, 1, -1, 1) * attn_probs
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0

        attn_probs = mindspore.Tensor.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = mindspore.Tensor(attn_probs)
        attn_probs = attn_probs.astype(attn_scores.dtype)

        # free memory
        del attn_scores

        # apply dropout
        attn_probs = ops.dropout(attn_probs, p=self.dropout, training=self.training)  # qbh no training?
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).swapaxes(0, 1)
        value_vectors = mindspore.Tensor(value_vectors)
        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.shape == (batch_size, seq_len, self.num_heads, self.head_dim), \
            "Unexpected size"  # qbh want ask
        attn_output = attn_output.swapaxes(0, 1).reshape(seq_len, batch_size, embed_dim)
        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                layer_head_mask=layer_head_mask,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.swapaxes(0, 1),)

        if output_attentions:
            outputs += (attn_probs,)
        return outputs + (global_attn_probs,) if (is_global_attn and output_attentions) else outputs

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = mindspore.ops.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.shape[:-2], hidden_states_padded.shape[-1], hidden_states_padded.shape[-2]
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.shape
        chunked_hidden_states = mindspore.ops.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        # Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
                                :, :, :-window_overlap
                                ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        if not onnx_export:
            # non-overlapping chunks of size = 2w
            hidden_states = hidden_states.view(
                hidden_states.shape[0],
                # mindspore.ops.div(
                #     Tensor(hidden_states.shape[1], dtype=mindspore.int32),
                #     Tensor((window_overlap * 2), dtype=mindspore.int32),
                #     rounding_mode="trunc"),
                # qbh change div
                int(hidden_states.shape[1] / (window_overlap * 2)),
                window_overlap * 2,
                hidden_states.shape[2],
            )
            # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
            chunk_size = list(hidden_states.shape)
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(int(x / 4) for x in hidden_states.strides)
            chunk_stride[1] = chunk_stride[1] // 2
            value = as_strided(self=hidden_states, size=chunk_size, stride=chunk_stride)
            return value

        # When exporting to ONNX, use this separate logic
        # have to use slow implementation since as_strided, unfold and
        # 2d-tensor indexing aren't supported (yet) in ONNX export

        # TODO replace this with
        # > return hidden_states.unfold(dimension=1, size=window_overlap * 2, step=window_overlap).transpose(2, 3)
        # once `unfold` is supported
        # the case hidden_states.size(1) == window_overlap * 2 can also
        # simply return hidden_states.unsqueeze(1), but that's control flow

        chunk_size = [
            hidden_states.size(0),
            mindspore.ops.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]
        overlapping_chunks = mindspore.numpy.empty(chunk_size)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                                                 :, chunk * window_overlap: chunk * window_overlap + 2 * window_overlap,
                                                 :
                                                 ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> mindspore.Tensor:

        beginning_mask_2d = input_tensor.new_ones((affected_seq_len, affected_seq_len + 1)).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = ops.flip(beginning_mask.copy(), dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = ops.broadcast_to(beginning_mask, beginning_input.shape)

        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = mindspore.numpy.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)

        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.broadcast_to(ending_input.shape)
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):] = ops.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: mindspore.Tensor, key: mindspore.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.shape == key.shape
        chunks_count = mindspore.ops.div(
            Tensor(seq_len, dtype=mindspore.int32),
            Tensor(window_overlap, dtype=mindspore.int32),
            rounding_mode="trunc"
        ) - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.swapaxes(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.swapaxes(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        query = self._chunk(query, window_overlap, self.config.__dict__.get("onnx_export", False))
        key = self._chunk(key, window_overlap, self.config.__dict__.get("onnx_export", False))

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        # diagonal_chunked_attention_scores = ops.einsum("bcxd,bcyd->bcxy", query, key)  # multiply

        diagonal_chunked_attention_scores = mindspore.Tensor(np.einsum(
            "bcxd,bcyd->bcxy",
            mindspore.Tensor.asnumpy(query),
            mindspore.Tensor.asnumpy(key))
        )
        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap)
        # columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, int(chunks_count) + 1, window_overlap, window_overlap * 2 + 1)
        )
        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).swapaxes(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs: mindspore.Tensor, value: mindspore.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.shape

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.shape[:3] == value.shape[:3]
        assert attn_probs.shape[3] == 2 * window_overlap + 1
        chunks_count = mindspore.ops.div(
            Tensor(seq_len, dtype=mindspore.float32),
            Tensor(window_overlap, dtype=mindspore.float32),
            rounding_mode="trunc"
        ) - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.swapaxes(1, 2).reshape(
            batch_size * num_heads,
            int(mindspore.ops.div(
                Tensor(seq_len, dtype=mindspore.float32),
                Tensor(window_overlap, dtype=mindspore.float32),
                rounding_mode="trunc"
            )),
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.swapaxes(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = mindspore.ops.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, int(chunks_count) + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = tuple(int(x / 4) for x in padded_value.strides)
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        # chunked_value = padded_value.as_strided(
        # size=chunked_value_size, stride=chunked_value_stride)  # qbh no as_stride
        chunked_value = as_strided(self=padded_value, size=chunked_value_size, stride=chunked_value_stride)
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        # context = ops.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        context = Tensor(np.einsum(
            "bcwd,bcdh->bcwh",
            chunked_attn_probs.asnumpy(),
            chunked_value.asnumpy()
        ))
        return context.view(batch_size, num_heads, seq_len, head_dim).swapaxes(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(axis=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero()  # as_tuple=True
        is_index_global_attn_nonzero = tensor_to_tuple(is_index_global_attn_nonzero)
        # helper variable
        is_local_index_global_attn = mindspore.numpy.arange(
            max_num_global_attn_indices
        ) < num_global_attn_indices.unsqueeze(dim=-1)
        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero()
        # qbh delete as_tuple = false
        is_local_index_global_attn_nonzero = tensor_to_tuple(is_local_index_global_attn_nonzero)
        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero()
        # qbh delete as_tuple = false
        is_local_index_no_global_attn_nonzero = tensor_to_tuple(is_local_index_no_global_attn_nonzero)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]  # qbh don't understand hint
        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            (batch_size, int(max_num_global_attn_indices), self.num_heads, self.head_dim)
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = \
            key_vectors[is_index_global_attn_nonzero]
        key_vectors_only_global = Tensor(key_vectors_only_global)
        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        # attn_probs_from_global_key = ops.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)
        attn_probs_from_global_key = mindspore.Tensor(np.einsum(
            "blhd,bshd->blhs",
            mindspore.Tensor.asnumpy(query_vectors),
            mindspore.Tensor.asnumpy(key_vectors_only_global))
        )

        attn_probs_from_global_key = attn_probs_from_global_key.swapaxes(1, 3)

        # print(type(is_local_index_no_global_attn_nonzero))
        # print(is_local_index_no_global_attn_nonzero)
        # print(type(attn_probs_from_global_key))
        # print(attn_probs_from_global_key.shape)

        if is_local_index_no_global_attn_nonzero[0].shape[0] != 0:
            attn_probs_from_global_key[
                is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
            ] = Tensor(np.finfo(
                mindspore.dtype_to_nptype(attn_probs_from_global_key.dtype)).min,
                       dtype=attn_probs_from_global_key.dtype
                       )

        attn_probs_from_global_key = attn_probs_from_global_key.swapaxes(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, int(max_num_global_attn_indices))
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            (batch_size, int(max_num_global_attn_indices), self.num_heads, self.head_dim)
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = ops.matmul(
            attn_probs_only_global.swapaxes(1, 2).copy(), value_vectors_only_global.swapaxes(1, 2).copy()
        ).swapaxes(1, 2) # qbh clone substitute copy

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, int(max_num_global_attn_indices), attn_probs.shape[-1] - int(max_num_global_attn_indices)
        )

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(
            (int(max_num_global_attn_indices),
             batch_size, self.embed_dim)
        )
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[
            is_index_global_attn_nonzero[::-1]
        ]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.view(int(max_num_global_attn_indices),
                                                  batch_size * self.num_heads,
                                                  self.head_dim
                                                  ).swapaxes(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.view(-1, batch_size * self.num_heads, self.head_dim).swapaxes(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.view(-1, batch_size * self.num_heads, self.head_dim).swapaxes(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = ops.bmm(global_query_vectors_only_global, global_key_vectors.swapaxes(1, 2))

        assert list(global_attn_scores.shape) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], (
            "global_attn_scores have the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is"
            f" {global_attn_scores.shape}."
        )

        global_attn_scores = global_attn_scores.view(
            batch_size,
            self.num_heads, int(max_num_global_attn_indices),
            seq_len
        )

        global_attn_scores = global_attn_scores.swapaxes(1, 2)
        if is_local_index_no_global_attn_nonzero[0].shape[0] != 0:
            global_attn_scores[
                is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
            ] = mindspore.Tensor(np.finfo(
                mindspore.dtype_to_nptype(global_attn_scores.dtype)).min,
                                 dtype=global_attn_scores.dtype
                                 )
        global_attn_scores = global_attn_scores.swapaxes(1, 2)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            mindspore.Tensor(np.finfo(
                mindspore.dtype_to_nptype(global_attn_scores.dtype)).min,
                             dtype=global_attn_scores.dtype
                             ),
        )

        global_attn_scores = global_attn_scores.view(
            batch_size * self.num_heads,
            int(max_num_global_attn_indices),
            seq_len
        )

        # compute global attn probs
        global_attn_probs_float = mindspore.Tensor(ops.softmax(
            global_attn_scores, axis=-1
        ), dtype=mindspore.float32)  # use fp32 for numerical stability

        # apply layer head masking
        if layer_head_mask is not None:
            assert layer_head_mask.shape == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.shape}"
            global_attn_probs_float = layer_head_mask.view(1, -1, 1, 1) * global_attn_probs_float.view(
                batch_size, self.num_heads, max_num_global_attn_indices, seq_len
            )
            global_attn_probs_float = global_attn_probs_float.view(
                batch_size * self.num_heads, max_num_global_attn_indices, seq_len
            )
        if self.training:
            global_attn_probs = ops.dropout(
                global_attn_probs_float.type_as(global_attn_scores), p=self.dropout
            )
        else:
            global_attn_probs = global_attn_probs_float
        # global attn output
        global_attn_output = ops.bmm(global_attn_probs, global_value_vectors)

        assert list(global_attn_output.shape) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], (
            "global_attn_output tensor has the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is"
            f" {global_attn_output.shape}."
        )

        global_attn_probs = global_attn_probs.view(
            batch_size,
            self.num_heads,
            int(max_num_global_attn_indices),
            seq_len
        )
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, int(max_num_global_attn_indices), self.head_dim
        )
        return global_attn_output, global_attn_probs


class LongformerSelfOutput(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(normalized_shape=(config.hidden_size, ), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LongformerAttention(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.self = LongformerSelfAttention(config, layer_id)
        self.output = LongformerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):  # qbh pytorch.util
        """
        Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


class LongformerIntermediate(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]  # qbh remain ACT2FN
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = Tensor(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LongformerOutput(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)  # qbh remain to do 3.9 22.29
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LongformerLayer(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = LongformerAttention(config, layer_id)
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def ff_chunk(self, attn_output):
        """
        Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
        """
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output


class LongformerEncoder(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        padding_len=0,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_cache
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0

        # Record `is_global_attn == True` to enable ONNX export
        is_global_attn = bool(is_index_global_attn.flatten().any())

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None  # All local attentions.
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.shape[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}."
        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # qbh TODO
            # if self.gradient_checkpointing and self.training:
            #
            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs, is_global_attn, output_attentions)
            #
            #         return custom_forward
            #
            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(layer_module),
            #         hidden_states,
            #         attention_mask,
            #         head_mask[idx] if head_mask is not None else None,
            #         is_index_masked,
            #         is_index_global_attn,
            #     )
            # else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len +
                # 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + (layer_outputs[1].swapaxes(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len =>
                    # bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + (layer_outputs[2].swapaxes(2, 3),)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # undo padding if necessary
        # unpad `hidden_states` because the calling function is expecting a length == input_ids.size(1)
        hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
        if output_hidden_states:
            all_hidden_states = (state[:, : state.shape[1] - padding_len] for state in all_hidden_states)

        if output_attentions:
            all_attentions = (state[:, :, : state.shape[2] - padding_len, :] for state in all_attentions)

        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
        )


class LongformerPooler(nn.Cell):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongformerLMHead(nn.Cell):
    """Longformer Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)

        self.decoder = nn.Dense(config.hidden_size, config.vocab_size)
        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def construct(self, features, **kwargs):
        x = self.dense(features)
        x = ops.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        self.bias = self.decoder.bias # qbh delete if device


class LongformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def get_input_embeddings(self) -> "nn.Cell":
        pass

    def set_input_embeddings(self, value: "nn.Cell"):
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self):
        pass

    config_class = LongformerConfig
    base_model_prefix = "longformer"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"position_ids"]
    _no_split_modules = ["LongformerSelfAttention"]

    def post_init(self):
        pass

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LongformerEncoder):
            module.gradient_checkpointing = value


class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from [`RobertaModel`] and overwrote standard self-attention with longformer self-attention
    to provide the ability to process long sequences following the self-attention approach described in [Longformer:
    the Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, and Arman Cohan.
    Longformer self-attention combines a local (sliding window) and global attention to extend to long documents
    without the O(n^2) increase in memory and compute.

    The self-attention module `LongformerSelfAttention` implemented here supports the combination of local and global
    attention but it lacks support for autoregressive attention and dilated attention. Autoregressive and dilated
    attention are more relevant for autoregressive language modeling than finetuning on downstream tasks. Future
    release will add support for autoregressive attention, but the support for dilated attention requires a custom CUDA
    kernel to be memory and compute efficient.

    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = LongformerPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _pad_to_window_size(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
        inputs_embeds: Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window

        # this path should be recorded in the ONNX export, it is fine with padding_len == 0 as well
        if padding_len > 0:
            logger.info(
                "Input ids are automatically padded from %s to %s to be a multiple of "
                "`config.attention_window`: %s", seq_len, seq_len + padding_len, attention_window
            )
            if input_ids is not None:
                input_ids = ops.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = ops.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=mindspore.int64,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = ops.cat([inputs_embeds, inputs_embeds_padding], axis=-2)

            attention_mask = ops.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens
            token_type_ids = ops.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    def _merge_to_attention_mask(self, attention_mask: Tensor, global_attention_mask: Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        r"""

        Returns:

        Examples:

        ```python
        >>> import torch
        >>> from transformers import LongformerModel, AutoTokenizer

        >>> model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")

        >>> SAMPLE_TEXT = " ".join(["Hello world! "] * 1000)  # long input document
        >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        >>> attention_mask = torch.ones(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        ... )  # initialize to local attention
        >>> global_attention_mask = torch.zeros(
        ...     input_ids.shape, dtype=torch.long, device=input_ids.device
        ... )  # initialize to global attention to be deactivated for all tokens
        >>> global_attention_mask[
        ...     :,
        ...     [
        ...         1,
        ...         4,
        ...         21,
        ...     ],
        ... ] = 1  # Set global attention to random tokens for the sake of this example
        >>> # Usually, set global attention based on the task. For example,
        >>> # classification: the <s> token
        >>> # QA: question tokens
        >>> # LM: potentially on the beginning of sentences and paragraphs
        >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        >>> sequence_output = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: Tensor = self.get_extended_attention_mask(attention_mask, input_shape)[
            :, 0, 0, :
        ]

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            padding_len=padding_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict or return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return (sequence_output, pooled_output) + encoder_outputs[1:]

class LongformerForMaskedLM(LongformerPreTrainedModel):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    _keys_to_ignore_on_load_missing = ["lm_head.decoder"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
        """
        self.lm_head.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, None]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Mask filling example:

        ```python
        >>> from transformers import AutoTokenizer, LongformerForMaskedLM

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        >>> model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
        ```

        Let's try a very long input.

        ```python
        >>> TXT = (
        ...     "My friends are <mask> but they eat too many carbs."
        ...     + " That's why I decide not to eat with them." * 300
        ... )
        >>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
        ['healthy', 'skinny', 'thin', 'good', 'vegetarian']
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict or return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return None


class LongformerForSequenceClassification(LongformerPreTrainedModel):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # qbh delete docs
    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, None]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            logger.info("Initializing global attention on CLS token...")
            global_attention_mask = ops.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (mindspore.int32, mindspore.int64)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict or return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return None

class LongformerClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = mindspore.ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output


class LongformerForQuestionAnswering(LongformerPreTrainedModel):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, None]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, LongformerForQuestionAnswering
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        >>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        >>> encoding = tokenizer(question, text, return_tensors="pt")
        >>> input_ids = encoding["input_ids"]

        >>> # default is local attention everywhere
        >>> # the forward method will automatically set global attention on question tokens
        >>> attention_mask = encoding["attention_mask"]

        >>> outputs = model(input_ids, attention_mask=attention_mask)
        >>> start_logits = outputs.start_logits
        >>> end_logits = outputs.end_logits
        >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        >>> answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]
        >>> answer = tokenizer.decode(
        ...     tokenizer.convert_tokens_to_ids(answer_tokens)
        ... )  # remove space prepending space token
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            if input_ids is None:
                logger.warning(
                    "It is not possible to automatically generate the `global_attention_mask` because input_ids is"
                    " None. Please make sure that it is correctly set."
                )
            else:
                # set global attention on question tokens automatically
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict or return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return None


class LongformerForTokenClassification(LongformerPreTrainedModel):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # qbh delete docs
    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, None]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict or return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return None


class LongformerForMultipleChoice(LongformerPreTrainedModel):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super().__init__(config)

        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    # qbh delete docs
    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        global_attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, None]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # set global attention on question tokens
        if global_attention_mask is None and input_ids is not None:
            logger.info("Initializing global attention on multiple choice...")
            # put global attention on all tokens after `config.sep_token_id`
            global_attention_mask = ops.stack(
                [
                    _compute_global_attention_mask(input_ids[:, i], self.config.sep_token_id, before_sep_token=False)
                    for i in range(num_choices)
                ],
                axis=1,
            )

        flat_input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        flat_global_attention_mask = (
            global_attention_mask.view(-1, global_attention_mask.shape[-1])
            if global_attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.longformer(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            global_attention_mask=flat_global_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict or return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return None
