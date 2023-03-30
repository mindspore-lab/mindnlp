# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
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
"""PyTorch Longformer model."""
# pylint: disable=relative-beyond-top-level
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-locals
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=invalid-name
import math
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import ops


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
        return position_ids.unsqueeze(0).expand(input_shape)


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
        attn_probs = mindspore.ops.dropout(attn_probs, p=1-self.dropout, training=self.training)  # qbh no training?
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
        beginning_mask = ops.expand(beginning_mask, Tensor(beginning_input.shape))

        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = mindspore.numpy.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)

        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(Tensor(ending_input.shape))
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
