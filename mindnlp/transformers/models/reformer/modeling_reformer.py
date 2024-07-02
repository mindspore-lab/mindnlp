# coding=utf-8
# Copyright 2020 The Trax Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# ============================================================================
"""MindSpore REFORMER model."""

import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, ParameterTuple
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward
from .configuration_reformer import ReformerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/reformer-crime-and-punishment"
_CONFIG_FOR_DOC = "ReformerConfig"

REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/reformer-crime-and-punishment",
    "google/reformer-enwik8",
    # See all Reformer models at https://hf-mirror.com/models?filter=reformer
]


# Define named tuples for nn.Cells here
LSHSelfAttentionOutput = namedtuple("LSHSelfAttentionOutput", ["hidden_states", "attention_probs", "buckets"])
LocalSelfAttentionOutput = namedtuple("LocalSelfAttentionOutput", ["hidden_states", "attention_probs"])
AttentionOutput = namedtuple("AttentionOutput", ["hidden_states", "attention_probs", "buckets"])
ReformerOutput = namedtuple("ReformerOutput", ["hidden_states", "attn_output", "attention_probs", "buckets"])
ReformerBackwardOutput = namedtuple(
    "ReformerBackwardOutput", ["attn_output", "hidden_states", "grad_attn_output", "grad_hidden_states"]
)
ReformerEncoderOutput = namedtuple(
    "ReformerEncoderOutput",
    ["hidden_states", "all_hidden_states", "all_attentions", "past_buckets_states"],
)


def _stable_argsort(vector, dim):
    """
    Args:
        vector (ndarray): The input vector to be sorted.
        dim (int): The dimension along which to perform the sorting.
    
    Returns:
        None: This function does not return a value, but sorts the input vector in place based on the
            specified dimension.
    
    Raises:
        None
    """
    # this function scales the vector so that ops.argsort is stable.
    # ops.argsort is not stable on its own
    scale_offset = ops.arange(vector.shape[dim]).view(1, 1, -1)
    scale_offset = scale_offset.expand(vector.shape)
    scaled_vector = vector.shape[dim] * vector + (scale_offset % vector.shape[dim])
    return ops.argsort(scaled_vector, axis=dim)


def _get_least_common_mult_chunk_len(config):
    """

    Args:
        config (object): The configuration object containing parameters for attention layers.
            This parameter is used to determine the chunk length based on the types of attention layers.
            It should have the following attributes:

            - attn_layers (list): List of attention layer types, can contain 'lsh', 'local', or both.
            - lsh_attn_chunk_length (int): Chunk length for 'lsh' attention layer.
            - local_attn_chunk_length (int): Chunk length for 'local' attention layer.

    Returns:
        None: The function does not return a value directly, but the chunk length is determined based on the
            configuration and used within the function.

    Raises:
        NotImplementedError: If the attention layer types in `config.attn_layers` are not one of ['lsh', 'local']
            or a combination of both.
    """
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    if len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    if len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return np.lcm(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    raise NotImplementedError(
        f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
        "attn layer types from ['lsh', 'local'] only."
    )


def _get_min_chunk_len(config):
    """

    Args:
        config (object):
            A configuration object containing the following attributes:

            - attn_layers (list): List of attention layer types ('lsh', 'local').
            - lsh_attn_chunk_length (int): Chunk length for 'lsh' attention layer.
            - local_attn_chunk_length (int): Chunk length for 'local' attention layer.

    Returns:
        None: This function calculates and returns the minimum chunk length based on the configuration provided.

    Raises:
        NotImplementedError:
            Raised when the configuration contains invalid attention layer types. Only 'lsh' and 'local' are allowed.
    """
    attn_types = config.attn_layers
    attn_types_set = set(attn_types)
    if len(attn_types_set) == 1 and attn_types[0] == "lsh":
        return config.lsh_attn_chunk_length
    if len(attn_types_set) == 1 and attn_types[0] == "local":
        return config.local_attn_chunk_length
    if len(attn_types_set) == 2 and attn_types_set == {"lsh", "local"}:
        return min(config.lsh_attn_chunk_length, config.local_attn_chunk_length)
    raise NotImplementedError(
        f"Only attn layer types 'lsh' and 'local' exist, but `config.attn_layers`: {config.attn_layers}. Select "
        "attn layer types from ['lsh', 'local'] only."
    )


class AxialPositionEmbeddings(nn.Cell):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """
    def __init__(self, config):
        """
        This method initializes an instance of the AxialPositionEmbeddings class.

        Args:
            self: The instance of the AxialPositionEmbeddings class.
            config:
                An object containing configuration parameters for the axial position embeddings.

                - axial_pos_shape: A list of integers representing the shape of the axial positions.
                - axial_pos_embds_dim: A list of integers representing the dimensions of the axial position embeddings.
                - hidden_dropout_prob: A float value representing the dropout probability.
                - hidden_size: An integer representing the hidden size of the model.

        Returns:
            None.

        Raises:
            ValueError: If the sum of axial_pos_embds_dim does not match the hidden_size specified in the configuration.
        """
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob

        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = []

        if sum(self.axial_pos_embds_dim) != config.hidden_size:
            raise ValueError(
                f"Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to "
                f"config.hidden_size: {config.hidden_size}"
            )

        # create weights
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            # create expanded shapes
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)

            # create tensor and init
            self.weights.append(Parameter(ops.ones(ax_shape, dtype=mindspore.float32), name=f'weights.{axis}'))

        self.weights = ParameterTuple(self.weights)

    def construct(self, position_ids):
        """
        This method constructs position encodings based on the given position IDs and axial position weights.

        Args:
            self: The instance of the AxialPositionEmbeddings class.
            position_ids (torch.Tensor): A 2D tensor representing the position IDs of the input sequences.
              It has a shape of (batch_size, sequence_length).

        Returns:
            None: This method does not return any value.

        Raises:
            ValueError:
                Raised if the following conditions are met:

                1. During training, if the product of the axial_pos_shape factors does not match the sequence length.
                2. During training and dropout is enabled, if the operation cannot be performed.
                3. During inference, if the product of the axial_pos_shape factors is less than the sequence length.
                4. During inference, if the required position encodings columns exceed the available columns.
        """
        # broadcast weights to correct shape
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]

        broadcasted_weights = [
            weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights
        ]

        if self.training is True:
            if reduce(mul, self.axial_pos_shape) != sequence_length:
                raise ValueError(
                    f"If training, make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply to "
                    f"sequence length. Got prod({self.axial_pos_shape}) != sequence_length: {sequence_length}. "
                    f"You might want to consider padding your sequence length to {reduce(mul, self.axial_pos_shape)} "
                    "or changing config.axial_pos_shape."
                )

            if self.dropout > 0:
                weights = ops.cat(broadcasted_weights, axis=-1)
                # permute weights so that 2D correctly drops dims 1 and 2
                transposed_weights = weights.swapaxes(2, 1)
                # drop entire matrix of last two dims (prev dims 1 and 2)
                dropped_transposed_weights = ops.dropout2d(
                    transposed_weights, p=self.dropout, training=self.training
                )
                dropped_weights = dropped_transposed_weights.swapaxes(2, 1)

                position_encodings = ops.reshape(dropped_weights, (batch_size, sequence_length, -1))

            else:
                position_encodings = ops.cat(
                    [ops.reshape(weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights],
                    axis=-1,
                )

        else:
            if reduce(mul, self.axial_pos_shape) < sequence_length:
                raise ValueError(
                    f"Make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply at least to "
                    f"max(sequence_length, least_common_mult_chunk_length): max({sequence_length}, "
                    f"{self.least_common_mult_chunk_length})."
                )

            # compute how many columns are needed
            max_position_id = position_ids.max().item()
            required_pos_encodings_columns = -(-(max_position_id + 1) // self.axial_pos_shape[1])

            # cut to columns that are needed
            position_encodings = ops.cat(
                [weight[:, :required_pos_encodings_columns] for weight in broadcasted_weights], axis=-1
            )
            position_encodings = ops.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))

            # select correct position encodings
            position_encodings = ops.cat(
                [
                    ops.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0)
                    for i in range(batch_size)
                ],
                axis=0,
            )

        return position_encodings


class PositionEmbeddings(nn.Cell):
    """Constructs conventional position embeddings of shape `[max_pos_embeddings, hidden_size]`."""
    def __init__(self, config):
        """
        Initializes an instance of the PositionEmbeddings class.

        Args:
            self: The instance of the PositionEmbeddings class.
            config:
                An instance of the configuration class containing the following attributes:

                - hidden_dropout_prob (float): The dropout probability used for the hidden layers.
                - max_position_embeddings (int): The maximum number of position embeddings.
                - hidden_size (int): The size of the hidden layers.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def construct(self, position_ids):
        """
        Constructs position embeddings based on given position IDs.

        Args:
            self (PositionEmbeddings): The instance of the PositionEmbeddings class.
            position_ids (Tensor): A tensor containing the position IDs for which embeddings need to be constructed.
                It should be a 1D tensor of integers representing the positions.

        Returns:
            position_embeddings: The method returns the constructed position embeddings.

        Raises:
            ValueError: If the position_ids tensor is not provided or is not a valid 1D tensor.
            TypeError: If the dropout rate is not a float or the training flag is not a boolean.
        """
        position_embeddings = self.embedding(position_ids)
        position_embeddings = ops.dropout(position_embeddings, p=self.dropout, training=self.training)
        return position_embeddings


class ReformerEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes the ReformerEmbeddings class.

        Args:
            self (object): The instance of the ReformerEmbeddings class.
            config (object):
                An object containing configuration parameters for the embeddings.

                - max_position_embeddings (int): The maximum number of positions for position embeddings.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - axial_pos_embds (bool): A flag indicating whether to use axial position embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = float(config.hidden_dropout_prob)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = (
            AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)
        )

    def construct(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        """
        Constructs the embeddings for the Reformer model.

        Args:
            self (ReformerEmbeddings): An instance of the ReformerEmbeddings class.
            input_ids (Optional[Tensor]): The input tensor representing the tokenized input sequence.
                It has shape [batch_size, sequence_length] and each element corresponds to a token ID.
            position_ids (Optional[Tensor]): The tensor representing the position encodings for each token
                in the input sequence. It has shape [batch_size, sequence_length] and each element corresponds
                to a position ID.
            inputs_embeds (Optional[Tensor]): The tensor representing the embedded input sequence.
                It has shape [batch_size, sequence_length, embedding_size] and each element corresponds
                to an embedding vector.
            start_idx_pos_encodings (int): The starting index for the positional encodings.

        Returns:
            Tensor: The embeddings tensor representing the input sequence with positional information.
                It has shape [batch_size, sequence_length, embedding_size].

        Raises:
            ValueError: If the sequence length of the position_ids tensor is greater than the maximum allowed
                sequence length.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = ops.arange(
                start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if position_ids.shape[-1] > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {position_ids.shape[-1]} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )

        # dropout
        embeddings = ops.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # add positional embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings


class EfficientAttentionMixin:
    """
    A few utilities for nn.Cells in Reformer, to be used as a mixin.
    """
    def _look_adjacent(self, vectors, num_chunks_before, num_chunks_after):
        """
        Used to implement attention between consecutive chunks.

        Args:
            vectors: array of shape [batch_size, num_attention_heads, n_chunks, chunk_len, ...]
            num_chunks_before: chunks before current chunk to include in attention
            num_chunks_after: chunks after current chunk to include in attention

        Returns:
            tensor of shape [num_chunks, N * chunk_length, ...], where N = (1 + num_chunks_before + num_chunks_after).
        """
        if num_chunks_before == 0 and num_chunks_after == 0:
            return vectors

        slices = []
        for i in range(-num_chunks_before, num_chunks_after + 1):
            if i == 0:
                slices.append(vectors)
            else:
                slices.append(ops.cat([vectors[:, :, i:, ...], vectors[:, :, :i, ...]], axis=2))
        return ops.cat(slices, axis=3)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.shape[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.swapaxes(2, 1)

    def _merge_hidden_size_dims(self, x, num_attn_heads, attn_head_size):
        """
        merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        x = x.permute(0, 2, 1, 3)
        return ops.reshape(x, (x.shape[0], -1, num_attn_heads * attn_head_size))

    def _split_seq_length_dim_to(self, vectors, dim_factor_1, dim_factor_2, num_attn_heads, attn_head_size=None):
        """
        splits sequence length dim of vectors into `dim_factor_1` and `dim_factor_2` dims
        """
        batch_size = vectors.shape[0]
        split_dim_shape = (batch_size, num_attn_heads, dim_factor_1, dim_factor_2)

        if len(vectors.shape) == 4:
            return ops.reshape(vectors, split_dim_shape + (attn_head_size,))
        if len(vectors.shape) == 3:
            return ops.reshape(vectors, split_dim_shape)
        raise ValueError(f"Input vector rank should be one of [3, 4], but is: {len(vectors.shape)}")


class LSHSelfAttention(nn.Cell, EfficientAttentionMixin):

    """
    This class represents a self-attention mechanism using Locality Sensitive Hashing (LSH) for efficient attention
    computation.
    It inherits from nn.Cell, EfficientAttentionMixin.

    The class implements LSH self-attention mechanism for neural networks. It includes methods for initializing the
    LSH attention layer, constructing the attention mechanism, computing attention masks, hashing vectors, and other
    related operations.

    Attributes:
        config: Configuration parameters for the LSH self-attention layer.
        chunk_length: Length of each attention chunk.
        num_hashes: Number of hashes used in the LSH mechanism.
        num_buckets: Number of buckets used for hashing.
        num_chunks_before: Number of chunks to consider before the current chunk.
        num_chunks_after: Number of chunks to consider after the current chunk.
        hash_seed: Seed for random hash rotations.
        is_decoder: Flag indicating if the layer is used in a decoder.
        max_position_embeddings: Maximum position embeddings allowed.
        dropout: Dropout probability for attention weights.
        num_attention_heads: Number of attention heads.
        attention_head_size: Size of each attention head.
        all_head_size: Total size of all attention heads.
        hidden_size: Size of the hidden layer.
        query_key: Dense layer for query and key vectors.
        value: Dense layer for value vectors.
        self_mask_value_float16: Float16 value for masking in self-attention.
        self_mask_value_float32: Float32 value for masking in self-attention.
        mask_value_float16: Float16 value for general masking.
        mask_value_float32: Float32 value for general masking.

    Methods:
        construct: Constructs the LSH self-attention mechanism based on input hidden states and optional parameters.
        _query_per_attn_head: Computes query vectors per attention head.
        _value_per_attn_head: Computes value vectors per attention head.
        _hash_vectors: Hashes input vectors into buckets for attention computation.
        _get_sorted_bucket_idx_and_undo_sorted_bucket_idx: Computes sorted bucket indices for efficient attention
            calculation.
        _set_num_buckets: Sets the number of buckets based on the input sequence length.
        _attend: Computes attention scores and outputs based on query, key, and value vectors.
        _compute_attn_mask: Computes attention mask based on query and key indices.
        _get_relevant_hid_states_and_buckets: Retrieves relevant hidden states and buckets for efficient attention
            calculation.
        _expand_to_indices_in_relevant_chunk: Expands indices for relevant chunks in hidden states.
        _len_and_dim_norm: Normalizes vectors based on length and attention head size.
        _len_norm: Length normalization for input vectors.
        _gather_by_expansion: Expands indices and vectors for all hashes and gathers relevant elements.

    Note:
        This class is designed for implementing efficient self-attention mechanisms using Locality Sensitive Hashing.
    """
    def __init__(self, config):
        """
        Initializes the LSHSelfAttention class.

        Args:
            self (LSHSelfAttention): An instance of the LSHSelfAttention class.
            config (object): An object containing the configuration parameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config

        self.chunk_length = config.lsh_attn_chunk_length
        self.num_hashes = config.num_hashes
        self.num_buckets = config.num_buckets
        self.num_chunks_before = config.lsh_num_chunks_before
        self.num_chunks_after = config.lsh_num_chunks_after
        self.hash_seed = config.hash_seed
        self.is_decoder = config.is_decoder
        self.max_position_embeddings = config.max_position_embeddings

        self.dropout = config.lsh_attention_probs_dropout_prob

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query_key = nn.Dense(self.hidden_size, self.all_head_size, has_bias=False)
        self.value = nn.Dense(self.hidden_size, self.all_head_size, has_bias=False)

        # save mask value here. Need fp32 and fp16 mask values
        self.self_mask_value_float16 = mindspore.tensor(-1e3)
        self.self_mask_value_float32 = mindspore.tensor(-1e5)
        self.mask_value_float16 = mindspore.tensor(-1e4)
        self.mask_value_float32 = mindspore.tensor(-1e9)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        buckets=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        """
        Constructs the LSH self-attention mechanism.

        Args:
            self: The object instance.
            hidden_states (torch.Tensor): The input hidden states with shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional): Masking tensor for attention scores with shape
                (batch_size, sequence_length). Defaults to None.
            head_mask (torch.Tensor, optional): Masking tensor for individual attention heads with shape
                (num_attention_heads). Defaults to None.
            num_hashes (int, optional): Number of hash functions for LSH. Defaults to None.
            buckets (torch.Tensor, optional): Hash buckets for attention mechanism with shape
                (batch_size, num_attention_heads, num_hashes, sequence_length). Defaults to None.
            past_buckets_states (tuple, optional): Tuple containing past hash buckets and states. Defaults to None.
            use_cache (bool, optional): Flag to enable caching for attention mechanism. Defaults to False.
            output_attentions (bool, optional): Flag to output attention probabilities. Defaults to False.

        Returns:
            None

        Raises:
            AssertionError: If the input sequence length is not equal to 1 when `past_buckets_states` is passed.
            AssertionError: If the last dimension of query_key_vectors or value_vectors does not match the attention
                head size.
            AssertionError: If the last dimension of the buckets tensor does not match the expected value based on the
                number of hashes and sequence length.
            AssertionError: If the shape of out_vectors does not match the expected shape.
            AssertionError: If the configuration parameters are not set correctly when chunk_length is None.
        """
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        do_cached_attention = use_cache and past_buckets_states[1] is not None

        # check if cache shall be used and that hidden states are already cached
        if do_cached_attention:
            assert sequence_length == 1, (
                "At the moment, auto-regressive language generation is only possible one word at a time. Make sure"
                f" that input sequence length {sequence_length} equals 1, when `past_buckets_states` is passed."
            )
            past_buckets = past_buckets_states[0]
            past_states = past_buckets_states[1]

            # get query vector
            query_vectors = self.query_key(hidden_states)
            query_vectors = self._split_hidden_size_dim(
                query_vectors, self.num_attention_heads, self.attention_head_size
            )

            if past_buckets is not None:
                key_value_hidden_states, sorted_bucket_idx, buckets = self._get_relevant_hid_states_and_buckets(
                    query_vectors=query_vectors,
                    attention_mask=attention_mask,
                    num_hashes=num_hashes,
                    hidden_states=hidden_states,
                    past_states=past_states,
                    past_buckets=past_buckets,
                )

                query_key_vectors = self._query_per_attn_head(key_value_hidden_states)
                value_vectors = self._value_per_attn_head(key_value_hidden_states)

                # split key & value vectors by num hashes to apply
                # self attention on each separately
                query_key_vectors = self._split_seq_length_dim_to(
                    query_key_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                value_vectors = self._split_seq_length_dim_to(
                    value_vectors,
                    num_hashes,
                    -1,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                # repeat query vectors across hash dimension
                query_vectors = query_vectors.unsqueeze(2).repeat(1, 1, num_hashes, 1, 1)
            else:
                key_value_hidden_states = ops.cat([past_states, hidden_states], axis=1)

                query_key_vectors = self.query_key(key_value_hidden_states)
                value_vectors = self.value(key_value_hidden_states)

        else:
            # project hidden_states to query_key and value
            query_vectors = None
            query_key_vectors = self.query_key(hidden_states)
            value_vectors = self.value(hidden_states)

        # if query key is not already split
        if not do_cached_attention or past_buckets is None:
            query_key_vectors = self._split_hidden_size_dim(
                query_key_vectors, self.num_attention_heads, self.attention_head_size
            )
            value_vectors = self._split_hidden_size_dim(
                value_vectors, self.num_attention_heads, self.attention_head_size
            )

        # cache buckets for next incremental decoding
        if do_cached_attention and past_buckets is None and key_value_hidden_states.shape[1] >= self.chunk_length:
            buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)

        # free memory
        del hidden_states

        assert (
            query_key_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {query_key_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of value_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."

        do_standard_self_attention = (sequence_length <= self.chunk_length) or (
            use_cache and past_buckets_states[1] is not None
        )
        # LSH attention only makes sense if chunked attention should be performed
        if not do_standard_self_attention:
            # set `num_buckets` on the fly, recommended way to do it
            if self.num_buckets is None:
                self._set_num_buckets(sequence_length)

            # use cached buckets for backprop only
            if buckets is None:
                # hash query key vectors into buckets
                buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
            else:
                # make sure buckets has correct shape for LSH attention
                buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes * sequence_length)

            assert (
                int(buckets.shape[-1]) == num_hashes * sequence_length
            ), f"last dim of buckets is {buckets.shape[-1]}, but should be {num_hashes * sequence_length}"

            sorted_bucket_idx, undo_sorted_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
                sequence_length, buckets, num_hashes
            )

            # make sure bucket idx is not longer then sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx % sequence_length

            # cluster query key value vectors according to hashed buckets
            query_key_vectors = self._gather_by_expansion(query_key_vectors, sorted_bucket_idx_per_hash, num_hashes)
            value_vectors = self._gather_by_expansion(value_vectors, sorted_bucket_idx_per_hash, num_hashes)
            query_key_vectors = self._split_seq_length_dim_to(
                query_key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            if self.chunk_length is None:
                assert self.num_chunks_before == 0 and self.num_chunks_after == 0, (
                    "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and"
                    " `config.num_chunks_before` are set to 0."
                )
        elif do_cached_attention and past_buckets is not None:
            # use max sequence length
            sorted_bucket_idx_per_hash = sorted_bucket_idx
        else:
            # get sequence length indices
            sorted_bucket_idx_per_hash = ops.arange(sequence_length).repeat(
                batch_size, self.num_attention_heads, 1
            )

        # scale key vectors
        sqrt_num = np.sqrt(self.attention_head_size)
        key_vectors = self._len_and_dim_norm(query_key_vectors, sqrt_num)

        # set query_vectors to query key vectors if LSH self attention
        query_vectors = query_vectors if query_vectors is not None else query_key_vectors

        # free memory
        del query_key_vectors

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx_per_hash=sorted_bucket_idx_per_hash,
            attention_mask=attention_mask,
            head_mask=head_mask,
            do_standard_self_attention=do_standard_self_attention,
            do_cached_attention=do_cached_attention,
        )

        # free memory
        del key_vectors, value_vectors

        # re-order out_vectors and logits
        if not do_standard_self_attention:
            # sort clusters back to correct ordering
            out_vectors, logits = ReverseSort()(out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx)

        if not do_standard_self_attention or (do_cached_attention and past_buckets is not None):
            # sum up all hash rounds
            if num_hashes > 1:
                out_vectors = self._split_seq_length_dim_to(
                    out_vectors,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                )
                logits = self._split_seq_length_dim_to(
                    logits,
                    num_hashes,
                    sequence_length,
                    self.num_attention_heads,
                    self.attention_head_size,
                ).unsqueeze(-1)

                probs_vectors = ops.exp(logits - ops.logsumexp(logits, axis=2, keep_dims=True))
                out_vectors = ops.sum(out_vectors * probs_vectors, dim=2)
                # free memory
                del probs_vectors

            # free memory
            del logits

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ), (
            "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length,"
            " config.attention_head_size]`."
        )

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        if buckets is not None:
            buckets = buckets.view(batch_size, self.num_attention_heads, num_hashes, -1)

        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs, buckets=buckets)

    def _query_per_attn_head(self, hidden_states):
        """
        This method computes the query key vectors for each attention head.

        Args:
            self (LSHSelfAttention): The instance of LSHSelfAttention class.
            hidden_states (Tensor): The input hidden states with shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method does not return any value but updates the query_key_vectors attribute of the
                LSHSelfAttention instance.

        Raises:
            ValueError: If the dimensions of the input hidden_states or query_key weight are incompatible
                for the matrix multiplication.
            RuntimeError: If the einsum operation fails due to any reason.
        """
        per_head_query_key = self.query_key.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).swapaxes(-2, -1)
        # only relevant for inference and no bias => we can use einsum here
        query_key_vectors = ops.einsum("balh,ahr->balr", hidden_states, per_head_query_key)
        return query_key_vectors

    def _value_per_attn_head(self, hidden_states):
        """
        This method calculates the value vectors per attention head in the LSHSelfAttention class.

        Args:
            self (LSHSelfAttention): The instance of the LSHSelfAttention class.
            hidden_states (Tensor): The input hidden states tensor with shape [batch_size, sequence_length, hidden_size].

        Returns:
            None: This method does not return a value but updates the value vectors based on the input hidden states.

        Raises:
            ValueError: If the dimensions of the input hidden_states tensor are incorrect for the matrix multiplication.
            RuntimeError: If any runtime error occurs during the calculation process.
        """
        per_head_value = self.value.weight.reshape(
            self.num_attention_heads, self.attention_head_size, self.hidden_size
        ).swapaxes(-2, -1)
        # only relevant for inference and no bias => we can use einsum here
        value_vectors = ops.einsum("balh,ahr->balr", hidden_states, per_head_value)
        return value_vectors

    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        """
        This method '_hash_vectors' is defined in the class 'LSHSelfAttention', and it performs hashing operations
        on input vectors.

        Args:
            self: The instance of the LSHSelfAttention class.
            vectors (Tensor): Input vectors to be hashed. Shape should be (batch_size, sequence_length, hidden_size).
            num_hashes (int): Number of hashes to be generated.
            attention_mask (Tensor or None): Masking tensor to apply attention masking. Should be of shape
                (batch_size, sequence_length).
            increase_num_buckets (bool): Flag indicating whether to increase the number of buckets.

        Returns:
            None: This method does not return any value but updates the 'offset_buckets' attribute.

        Raises:
            AssertionError:
                - If the number of buckets is not an even integer.
                - If the input bucket factors are not even integers.
                - If the attention mask sum is less than the expected value based on batch size and sequence length.
            ValueError:
                - If the number of buckets is not an integer or a list of integers.
            IndexError: If the tensors' dimensions are incompatible for operations like indexing and concatenation.
            TypeError: If the data type of tensors is not supported for the operations performed in the method.
        """
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert (
                self.num_buckets % 2 == 0
            ), f"There should be an even number of buckets, but `self.num_buckets`: {self.num_buckets}"
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert (
                    bucket_factor % 2 == 0
                ), f"The number of buckets should be even, but `num_bucket`: {bucket_factor}"
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        if self.hash_seed is not None:
            # for determinism
            mindspore.set_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = ops.randn(rotations_shape, dtype=vectors.dtype)
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = ops.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = ops.cat([rotated_vectors, -rotated_vectors], axis=-1)
            buckets = ops.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = ops.cat([rotated_vectors_factor, -rotated_vectors_factor], axis=-1)
                if buckets is None:
                    buckets = ops.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * ops.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (attention_mask.sum().item() < batch_size * attention_mask.shape[-1]):
            # add an extra bucket for padding tokens only
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket
            buckets_mask = attention_mask.to(mindspore.bool_)[:, None, None, :].expand(buckets.shape)
            buckets = ops.where(
                buckets_mask, buckets, mindspore.tensor(num_buckets - 1, dtype=mindspore.int64)
            )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = ops.arange(num_hashes)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        """
        Method to get the sorted bucket indices and create an undo mapping for sorting purposes in LSHSelfAttention class.

        Args:
            self: The instance of the LSHSelfAttention class.
            sequence_length (int): The length of the input sequence.
            buckets (Tensor): A tensor containing bucket values for each element in the input sequence.
            num_hashes (int): The number of hash functions used for bucketing.

        Returns:
            two tensors: sorted_bucket_idx and undo_sorted_bucket_idx.

        Raises:
            ValueError: If the input sequence length is not a positive integer.
            TypeError: If the buckets tensor is not a valid tensor object.
            ValueError: If the number of hashes is not a positive integer.
        """
        # hash-based sort
        sorted_bucket_idx = _stable_argsort(buckets, dim=-1)

        # create simple indices to scatter to, to have undo sort
        indices = (
            ops.arange(sorted_bucket_idx.shape[-1])
            .view(1, 1, -1)
            .expand(sorted_bucket_idx.shape)
        )

        # get undo sort
        undo_sorted_bucket_idx = sorted_bucket_idx.new_zeros(sorted_bucket_idx.shape).astype(indices.dtype)
        undo_sorted_bucket_idx = undo_sorted_bucket_idx.scatter(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _set_num_buckets(self, sequence_length):
        """
        This method _set_num_buckets calculates the number of buckets to be used for locality-sensitive hashing (LSH)
        self-attention in the LSHSelfAttention class.

        Args:
            self (LSHSelfAttention): The instance of the LSHSelfAttention class.
            sequence_length (int): The total length of the input sequence.

        Returns:
            None: This method does not return any value. It sets the calculated number of buckets in the configuration
                and instance variable.

        Raises:
            None.
        """
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2**num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        logger.warning(f"config.num_buckets is not set. Setting config.num_buckets to {num_buckets}...")

        # set num buckets in config to be properly saved
        self.config.num_buckets = num_buckets
        self.num_buckets = num_buckets

    def _attend(
        self,
        query_vectors,
        key_vectors,
        value_vectors,
        sorted_bucket_idx_per_hash,
        attention_mask,
        head_mask,
        do_standard_self_attention,
        do_cached_attention,
    ):
        '''
        This method performs LSH (Locality Sensitive Hashing) self-attention calculation for the LSHSelfAttention class.

        Args:
            self (LSHSelfAttention): The LSHSelfAttention object.
            query_vectors (Tensor): The input query vectors for the attention calculation.
            key_vectors (Tensor): The input key vectors for the attention calculation.
            value_vectors (Tensor): The input value vectors for the attention calculation.
            sorted_bucket_idx_per_hash (Tensor): The sorted bucket indices per hash for the attention calculation.
            attention_mask (Tensor): The attention mask to be applied in the calculation.
            head_mask (Tensor, optional): The optional mask to be applied to the attention scores.
            do_standard_self_attention (bool): A flag indicating whether to perform standard self-attention calculation.
            do_cached_attention (bool): A flag indicating whether to use cached attention for the calculation.

        Returns:
            None.

        Raises:
            ValueError: If the input tensors do not have the expected shapes or types.
            RuntimeError: If there is an issue with the computation during the method execution.
        '''
        # look at previous and following chunks if chunked attention
        if not do_standard_self_attention:
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        # (BS, NumAttn, NumHash x NumChunk, Chunk_L x Hidden),(BS, NumAttn, NumHash x NumChunk, Chunk_L * (Num_bef + Num_aft + 1) x Hidden) -> (BS, NumAttn, NumHash x NumChunk, Chunk_L, Chunk_L * (1 + Num_bef+ Num_aft))
        query_key_dots = ops.matmul(query_vectors, key_vectors.swapaxes(-1, -2))

        # free memory
        del query_vectors, key_vectors

        # if chunked attention split bucket idxs to query and key
        if not do_standard_self_attention:
            query_bucket_idx = self._split_seq_length_dim_to(
                sorted_bucket_idx_per_hash, -1, self.chunk_length, self.num_attention_heads
            )
            key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)
        elif do_cached_attention and query_key_dots.ndim > 4:
            key_value_bucket_idx = sorted_bucket_idx_per_hash
            query_bucket_idx = (
                key_value_bucket_idx.new_ones(key_value_bucket_idx.shape[:-1] + (1,)) * key_value_bucket_idx.max()
            )
        elif do_cached_attention and query_key_dots.ndim <= 4:
            query_bucket_idx = (query_key_dots.shape[-1] - 1) * ops.ones_like(query_key_dots)[:, :, :, -1]
            key_value_bucket_idx = ops.arange(
                query_key_dots.shape[-1], dtype=mindspore.int64
            )[None, None, :].expand(query_bucket_idx.shape[:2] + (-1,))
        else:
            query_bucket_idx = key_value_bucket_idx = sorted_bucket_idx_per_hash

        # get correct mask values depending on precision
        if query_key_dots.dtype == mindspore.float16:
            self_mask_value = self.self_mask_value_float16.half()
            mask_value = self.mask_value_float16.half()
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        if not do_cached_attention:
            mask = self._compute_attn_mask(
                query_bucket_idx,
                key_value_bucket_idx,
                attention_mask,
                query_key_dots.shape,
                do_standard_self_attention,
            )

            if mask is not None:
                query_key_dots = ops.where(mask, query_key_dots, mask_value)

            # free memory
            del mask

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "

        self_mask = ops.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2))

        # apply self_mask
        query_key_dots = ops.where(self_mask, query_key_dots, self_mask_value)

        # free memory
        del self_mask

        logits = ops.logsumexp(query_key_dots, axis=-1, keep_dims=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = ops.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = ops.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = ops.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if out_vectors.ndim > 4:
            logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dot_shape, do_standard_self_attention
    ):

        """
        Compute attention mask for LSH self-attention.

        This method computes the attention mask for LSH self-attention based on the given query and key indices,
        attention mask, query-key dot shape, and the flag indicating whether to use standard self-attention.

        Args:
            self (LSHSelfAttention): An instance of the LSHSelfAttention class.
            query_indices (Tensor): The indices of the query.
            key_indices (Tensor): The indices of the key.
            attention_mask (Tensor): The attention mask to be applied (optional).
            query_key_dot_shape (tuple): The shape of the dot product between query and key.
            do_standard_self_attention (bool): A flag indicating whether to use standard self-attention.

        Returns:
            Tensor: The computed attention mask.

        Raises:
            TypeError: If the attention_mask is not of type Tensor.
        """
        # attention mask for LSH
        if attention_mask is not None:
            # if chunked attention, the attention mask has to correspond to LSH order
            attention_mask = attention_mask.to(mindspore.bool_)[:, None, :]
            if not do_standard_self_attention:
                # expand attn_mask to fit with key_value_bucket_idx shape
                attention_mask = attention_mask[:, None, :]
                attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
                # extract attention mask from LSH sorted key_indices
                attention_mask = ops.gather_elements(attention_mask, -1, key_indices)

            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dot_shape)

        # Causal mask
        if self.is_decoder is True:
            causal_mask = ops.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2))

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask & attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    def _get_relevant_hid_states_and_buckets(
        self, query_vectors, attention_mask, num_hashes, hidden_states, past_states, past_buckets
    ):

        """
        This method '_get_relevant_hid_states_and_buckets' is defined in the class 'LSHSelfAttention'.

        Args:
            self (LSHSelfAttention): The instance of the LSHSelfAttention class.
            query_vectors (Tensor): The input query vectors for which relevant hidden states and buckets need
                to be retrieved.
            attention_mask (Tensor): The attention mask to be applied during the computation.
            num_hashes (int): The number of hashes to be used in the computation.
            hidden_states (Tensor): The current hidden states that include both past and new states.
            past_states (Tensor): The past hidden states that need to be included in the computation.
            past_buckets (Tensor): The past bucket values that need to be considered.

        Returns:
            None: This method does not return any value but updates the relevant hidden states and buckets.

        Raises:
            AssertionError:

                - If the shape of 'bucket_idx' does not match the expected shape.
                - If the shape of 'relevant_hidden_states' or 'relevant_bucket_idx_chunk' does not match the
                expected shapes.
                - If the number of 'hidden_states' or 'bucket_idx' does not match the expected values.
            IndexError: If there is an index error during the computation.
        """
        # concat hidden states
        hidden_states = ops.cat([past_states, hidden_states], axis=1)

        # batch_size hidden
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        # check if cached buckets include pad bucket
        max_bucket = self.num_buckets if isinstance(self.num_buckets, int) else reduce(mul, self.num_buckets)

        # if pad bucket was cached => need to increase num buckets for caching
        increase_num_buckets = past_buckets.max() > num_hashes * max_bucket - 1

        # retrieve query buckets
        query_buckets = self._hash_vectors(
            query_vectors, num_hashes, attention_mask, increase_num_buckets=increase_num_buckets
        )

        # concat buckets
        concat_buckets = ops.cat([past_buckets, query_buckets.unsqueeze(-1)], axis=-1)

        # hash-based sort
        bucket_idx = _stable_argsort(concat_buckets, dim=-1)

        # bucket_idx has shape: BatchSize x NumAttnHeads x NumHashes x SequenceLength
        assert bucket_idx.shape == (
            batch_size,
            self.num_attention_heads,
            num_hashes,
            sequence_length,
        ), (
            f"bucket_idx should have shape {(batch_size, self.num_attention_heads, num_hashes, sequence_length)}, but"
            f" has shape {bucket_idx.shape}."
        )

        # find indices of new bucket indices
        relevant_bucket_idx = (bucket_idx == (bucket_idx.shape[-1] - 1)).nonzero()

        # expand relevant bucket indices to its chunks
        relevant_bucket_idx_chunk = self._expand_to_indices_in_relevant_chunk(relevant_bucket_idx, sequence_length)
        relevant_bucket_idx_chunk = bucket_idx[tuple(relevant_bucket_idx_chunk.swapaxes(0, 1))]

        # adapt bucket_idx for batch and hidden states for index select
        offset = ops.arange(relevant_bucket_idx_chunk.shape[-1], dtype=mindspore.int64)
        bucket_idx_batch_offset = sequence_length * (
            batch_size * ops.div(offset, relevant_bucket_idx_chunk.shape[-1], rounding_mode="floor")
        )

        # add batch offset
        relevant_bucket_idx_chunk_all_batch = relevant_bucket_idx_chunk + bucket_idx_batch_offset
        hidden_states = hidden_states.reshape((-1, self.hidden_size))

        # select all relevant hidden states
        relevant_hidden_states = hidden_states.index_select(0, relevant_bucket_idx_chunk_all_batch)

        # reshape hidden states and bucket_idx to correct output
        relevant_hidden_states = relevant_hidden_states.reshape(
            batch_size, self.num_attention_heads, -1, self.hidden_size
        )
        relevant_bucket_idx_chunk = relevant_bucket_idx_chunk.reshape(
            batch_size, self.num_attention_heads, num_hashes, -1
        )

        assert (
            relevant_hidden_states.shape[2]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes
        ), (
            "There should be"
            f" {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length * num_hashes} `hidden_states`,"
            f" there are {relevant_hidden_states.shape[2]} `hidden_states`."
        )

        assert (
            relevant_bucket_idx_chunk.shape[-1]
            == (self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length
        ), (
            "There should be"
            f" {(self.num_chunks_before + self.num_chunks_after + 1) * self.chunk_length} `hidden_states`, there are"
            f" {relevant_bucket_idx_chunk.shape[-1]} `bucket_idx`."
        )

        return relevant_hidden_states, relevant_bucket_idx_chunk, query_buckets

    def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):

        """
        This method '_expand_to_indices_in_relevant_chunk' is defined in the class 'LSHSelfAttention'.

        Args:
            self:
                An instance of the LSHSelfAttention class.

                - Purpose: Represents the current instance of the LSHSelfAttention class.
                - Restrictions: None.

            indices:
                A tensor containing indices.

                - Type: Tensor
                - Purpose: Specifies the indices to be expanded within the relevant chunk.
                - Restrictions: Should be a 2D tensor.

            sequence_length:
                An integer specifying the total length of the sequence.

                - Type: int
                - Purpose: Indicates the total length of the sequence for boundary calculations.
                - Restrictions: Must be a positive integer.

        Returns:
            None:

                - Type: None
                - Purpose: The method does not return any value, it modifies the 'indices' tensor in place.

        Raises:
            None.
        """
        # get relevant indices of where chunk starts and its size
        start_indices_chunk = ((indices[:, -1] // self.chunk_length) - self.num_chunks_before) * self.chunk_length
        total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)

        # expand start indices and add correct chunk offset via arange
        expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
        chunk_sequence_indices = expanded_start_indices + ops.arange(
            total_chunk_size, dtype=mindspore.int64
        ).unsqueeze(0).expand(indices.shape[0], total_chunk_size)

        # make sure that circular logic holds via % seq len
        chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length

        # expand indices and set indices correctly
        indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(start_dim=0, end_dim=1).copy()
        indices[:, -1] = chunk_sequence_indices

        return indices

    def _len_and_dim_norm(self, vectors, sqrt_num):
        """
        length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors / sqrt_num
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
        length normalization
        """
        variance = ops.mean(x**2, -1, keep_dims=True)
        norm_x = x * ops.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
        expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return ops.gather_elements(vectors, 2, expanded_idxs)


class ReverseSort(nn.Cell):
    """
    After chunked attention is applied which sorted clusters, original ordering has to be restored. Since customized
    backward function is used for Reformer, the gradients of the output vectors have to be explicitly sorted here.
    """
    def construct(self, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx):

        """
        Constructs and returns modified vectors and logits based on the given parameters.

        Args:
            self (ReverseSort): An instance of the ReverseSort class.
            out_vectors (Tensor): The original vectors.
            logits (Tensor): The original logits.
            sorted_bucket_idx (Tensor): The indices of sorted buckets.
            undo_sorted_bucket_idx (Tensor): The indices to undo the sorted buckets.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the modified out_vectors and logits.

        Raises:
            None.

        """
        # save sorted_bucket_idx for backprop
        self.sorted_bucket_idx = sorted_bucket_idx

        # undo sort to have correct order for next layer
        expanded_undo_sort_indices = undo_sorted_bucket_idx.unsqueeze(-1).expand(out_vectors.shape)
        out_vectors = ops.gather_elements(out_vectors, 2, expanded_undo_sort_indices)
        logits = ops.gather_elements(logits, 2, undo_sorted_bucket_idx)
        return out_vectors, logits

    def bprop(self, out_vectors, logits, sorted_bucket_idx, undo_sorted_bucket_idx, outputs, grads):

        """
        This method performs backpropagation for the ReverseSort class.

        Args:
            self: An instance of the ReverseSort class.
            out_vectors: A tensor containing the output vectors.
            logits: A tensor containing the logits.
            sorted_bucket_idx: A tensor containing the sorted bucket indices.
            undo_sorted_bucket_idx: A tensor containing the undo sorted bucket indices.
            outputs: A tensor containing the outputs.
            grads: A tuple of two tensors containing the gradients of out_vectors and logits respectively.

        Returns:
            A tuple containing the gradients of out_vectors and logits, followed by two None values.

        Raises:
            None.

        """
        grad_out_vectors, grad_logits = grads
        # get parameters saved in ctx
        sorted_bucket_idx = self.sorted_bucket_idx

        expanded_sort_indices = sorted_bucket_idx.unsqueeze(-1).expand(grad_out_vectors.shape)
        # reverse sort of forward
        grad_out_vectors = ops.gather_elements(grad_out_vectors, 2, expanded_sort_indices)
        grad_logits = ops.gather_elements(grad_logits, 2, sorted_bucket_idx)

        # return grad and `None` fillers for last 2 forward args
        return grad_out_vectors, grad_logits, None, None


class LocalSelfAttention(nn.Cell, EfficientAttentionMixin):

    """
    The `LocalSelfAttention` class is a subclass of `nn.Cell` and `EfficientAttentionMixin` that represents
    a local self-attention mechanism. This mechanism is commonly used in transformer-based models for
    processing sequential data.

    Attributes:
        `num_attention_heads` (int): The number of attention heads.
        `chunk_length` (int): The length of each attention chunk.
        `num_chunks_before` (int): The number of chunks before the current position.
        `num_chunks_after` (int): The number of chunks after the current position.
        `is_decoder` (bool): Indicates whether the attention is used in a decoder architecture.
        `pad_token_id` (int): The token ID used for padding.
        `attention_head_size` (int): The size of each attention head.
        `all_head_size` (int): The total size of all attention heads.
        `hidden_size` (int): The hidden size of the input.
        `query` (nn.Dense): The dense layer used for computing query vectors.
        `key` (nn.Dense): The dense layer used for computing key vectors.
        `value` (nn.Dense): The dense layer used for computing value vectors.
        `dropout` (float): The dropout rate for attention probabilities.
        `mask_value_float16` (mindspore.tensor): The mask value for float16 data type.
        `mask_value_float32` (mindspore.tensor): The mask value for float32 data type.

    Methods:
        `construct`: Computes the local self-attention mechanism for the given hidden states.
        `_compute_attn_mask`: Computes the attention mask based on query and key indices.
        `_retrieve_relevant_hidden_states`: Retrieves the relevant hidden states from previous states.

    """
    def __init__(self, config):

        """
        Initializes the LocalSelfAttention class.

        Args:
            self: The instance of the LocalSelfAttention class.
            config: An object containing configuration parameters for the attention mechanism.
                This parameter is expected to have the following attributes:

                - num_attention_heads (int): The number of attention heads.
                - local_attn_chunk_length (int): The length of attention chunks.
                - local_num_chunks_before (int): The number of chunks before the current position.
                - local_num_chunks_after (int): The number of chunks after the current position.
                - is_decoder (bool): Indicates if the attention mechanism is used in a decoder.
                - pad_token_id (int): The token ID used for padding.
                - attention_head_size (int): The size of each attention head.
                - hidden_size (int): The size of the hidden layer.
                - local_attention_probs_dropout_prob (float): The dropout probability for attention weights.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters provided are invalid or missing.
            TypeError: If any of the configuration attribute types are incorrect.
        """
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.chunk_length = config.local_attn_chunk_length
        self.num_chunks_before = config.local_num_chunks_before
        self.num_chunks_after = config.local_num_chunks_after
        self.is_decoder = config.is_decoder
        self.pad_token_id = config.pad_token_id

        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = config.hidden_size

        # projection matrices
        self.query = nn.Dense(self.hidden_size, self.all_head_size, has_bias=False)
        self.key = nn.Dense(self.hidden_size, self.all_head_size, has_bias=False)
        self.value = nn.Dense(self.hidden_size, self.all_head_size, has_bias=False)

        self.dropout = float(config.local_attention_probs_dropout_prob)

        # save mask value here
        self.mask_value_float16 = mindspore.tensor(-1e4)
        self.mask_value_float32 = mindspore.tensor(-1e9)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_buckets_states=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):

        """
        Constructs the local self-attention mechanism.

        Args:
            self (object): The instance of the LocalSelfAttention class.
            hidden_states (Tensor): The input hidden states with shape (batch_size, sequence_length, hidden_size).
            attention_mask (Tensor, optional): Mask to prevent attention to certain positions. Shape should be
                (batch_size, 1, sequence_length, sequence_length). Defaults to None.
            head_mask (Tensor, optional): Mask to prevent attention to certain heads.
                Shape should be (num_attention_heads,) or (num_layers, num_attention_heads). Defaults to None.
            past_buckets_states (Tuple[Tensor, Tensor], optional): Tuple containing the previous key and
                value hidden states. Defaults to None.
            use_cache (bool, optional): Flag indicating whether to use cached key and value states. Defaults to False.
            output_attentions (bool, optional): Flag indicating whether to output attention probabilities.
                Defaults to False.

        Returns:
            LocalSelfAttentionOutput: An object containing the output hidden states and attention probabilities.

        Raises:
            AssertionError: Raised if the conditions specified in the method are not met.
        """
        sequence_length = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # check if cache shall be used and that hidden states are already cached
        if use_cache and past_buckets_states[1] is not None:
            assert past_buckets_states[0] is None, (
                "LocalSelfAttention should not make use of `buckets`. There seems to be an error when caching"
                " hidden_states_and_buckets."
            )
            key_value_hidden_states = self._retrieve_relevant_hidden_states(
                past_buckets_states[1], self.chunk_length, self.num_chunks_before
            )
            key_value_hidden_states = ops.cat([key_value_hidden_states, hidden_states], axis=1)

            # only query vector for last token
            query_vectors = self.query(hidden_states)
            # compute key and value for relevant chunk
            key_vectors = self.key(key_value_hidden_states)
            value_vectors = self.value(key_value_hidden_states)

            # free memory
            del key_value_hidden_states
        else:
            # project hidden_states to query, key and value
            query_vectors = self.query(hidden_states)
            key_vectors = self.key(hidden_states)
            value_vectors = self.value(hidden_states)

        # split last dim into `config.num_attention_heads` and `config.attention_head_size`
        query_vectors = self._split_hidden_size_dim(query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_hidden_size_dim(key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert (
            query_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {query_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            key_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {key_vectors.shape[-1]} but should be {self.attention_head_size}."
        assert (
            value_vectors.shape[-1] == self.attention_head_size
        ), f"last dim of query_key_vectors is {value_vectors.shape[-1]} but should be {self.attention_head_size}."

        if self.chunk_length is None:
            assert self.num_chunks_before == 0 and self.num_chunks_after == 0, (
                "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and"
                " `config.num_chunks_before` are set to 0."
            )

        # normalize key vectors
        key_vectors = key_vectors / np.sqrt(self.attention_head_size)

        # get sequence length indices
        indices = ops.arange(sequence_length).repeat(
            batch_size, self.num_attention_heads, 1
        )

        # if one should do normal n^2 self-attention
        do_standard_self_attention = sequence_length <= self.chunk_length

        # if input should be chunked
        if not do_standard_self_attention:
            # chunk vectors
            # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size
            query_vectors = self._split_seq_length_dim_to(
                query_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            key_vectors = self._split_seq_length_dim_to(
                key_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )
            value_vectors = self._split_seq_length_dim_to(
                value_vectors,
                -1,
                self.chunk_length,
                self.num_attention_heads,
                self.attention_head_size,
            )

            # chunk indices
            query_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)
            key_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)

            # append chunks before and after
            key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
            value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
            key_indices = self._look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after)
        else:
            query_indices = key_indices = indices

        # query-key matmul: QK^T
        query_key_dots = ops.matmul(query_vectors, key_vectors.swapaxes(-1, -2))

        # free memory
        del query_vectors, key_vectors

        mask = self._compute_attn_mask(
            query_indices, key_indices, attention_mask, query_key_dots.shape, do_standard_self_attention
        )

        if mask is not None:
            # get mask tensor depending on half precision or not
            if query_key_dots.dtype == mindspore.float16:
                mask_value = self.mask_value_float16.half()
            else:
                mask_value = self.mask_value_float32

            query_key_dots = ops.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # softmax
        logits = ops.logsumexp(query_key_dots, axis=-1, keep_dims=True)
        attention_probs = ops.exp(query_key_dots - logits)

        # free memory
        del logits

        # dropout
        attention_probs = ops.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = ops.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        if not do_standard_self_attention:
            out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        )

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(
        self, query_indices, key_indices, attention_mask, query_key_dots_shape, do_standard_self_attention
    ):

        """
        Computes the attention mask for the LocalSelfAttention module.

        Args:
            self (LocalSelfAttention): The instance of the LocalSelfAttention class.
            query_indices (Tensor): The indices of the query sequence. Shape: (batch_size, query_seq_length).
            key_indices (Tensor): The indices of the key sequence. Shape: (batch_size, key_seq_length).
            attention_mask (Tensor): The attention mask tensor. Shape: (batch_size, query_seq_length, key_seq_length).
            query_key_dots_shape (Tensor): The shape of the query-key dot products.
                Shape: (batch_size, num_attention_heads, query_seq_length, key_seq_length).
            do_standard_self_attention (bool): A flag indicating whether to use standard self-attention.

        Returns:
            Tensor or None: The computed attention mask tensor.
                Shape: (batch_size, num_attention_heads, query_seq_length, key_seq_length).

        Raises:
            TypeError: If the input arguments are not of the expected types.
            ValueError: If the shapes of the input arguments are not valid.
        """
        # chunk attention mask and look before and after
        if attention_mask is not None:
            attention_mask = attention_mask.to(mindspore.bool_)[:, None, :]

            if not do_standard_self_attention:
                attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
                attention_mask = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
            # create attn_mask
            attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)

        # Causal mask
        if self.is_decoder is True:
            causal_mask = ops.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2))

            # add attention mask if not None
            if attention_mask is not None:
                attention_mask = causal_mask & attention_mask
            else:
                attention_mask = causal_mask

        return attention_mask

    @staticmethod
    def _retrieve_relevant_hidden_states(previous_hidden_states, chunk_length, num_chunks_before):

        """
        Retrieves relevant hidden states from previous hidden states based on the provided parameters.

        Args:
            previous_hidden_states (ndarray): An array of shape (batch_size, sequence_length, hidden_size)
                representing the previous hidden states.
            chunk_length (int): The length of each chunk of hidden states.
            num_chunks_before (int): The number of chunks before the relevant hidden states to retrieve.

        Returns:
            None

        Raises:
            None

        This static method retrieves relevant hidden states from previous hidden states based on the given parameters.
        It calculates the start position of the relevant hidden states based on the chunk length and the number of
        chunks before. The retrieved hidden states are then returned as an array with shape
        (batch_size, sequence_length, hidden_size), starting from the calculated start position.
        """
        start_position = ((previous_hidden_states.shape[1] // chunk_length) - num_chunks_before) * chunk_length
        return previous_hidden_states[:, start_position:]


class ReformerSelfOutput(nn.Cell):

    """
    This class represents the self-attention output module of the Reformer model.

    The ReformerSelfOutput class inherits from the nn.Cell class and is responsible for processing the hidden states
    of the Reformer model's self-attention layer. It applies a linear transformation followed by dropout to the
    input hidden states.

    Attributes:
        dropout (float): The dropout probability used during the forward pass.
        dense (nn.Dense): The linear transformation layer that maps the input hidden states to the output.

    Methods:
        construct(hidden_states): Applies the linear transformation and dropout to the input hidden states.

    Example:
        ```python
        >>> # Create an instance of the ReformerSelfOutput class
        >>> config = Configuration()
        >>> reformer_self_output = ReformerSelfOutput(config)
        ...
        >>> # Apply the self-attention output module to the hidden states
        >>> hidden_states = torch.randn(batch_size, sequence_length, hidden_size)
        >>> output = reformer_self_output.construct(hidden_states)
        ```
    Note:
        This class assumes that the input hidden_states have already been processed by the self-attention module
        of the Reformer model.
    """
    def __init__(self, config):

        """
        Initializes a new instance of the ReformerSelfOutput class.

        Args:
            self (object): The current instance of the class.
            config (object): An object containing configuration parameters for the ReformerSelfOutput.
                This object should have the following attributes:

                - num_attention_heads (int): Number of attention heads.
                - attention_head_size (int): Size of each attention head.
                - hidden_dropout_prob (float): Dropout probability for hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        all_head_size = config.num_attention_heads * config.attention_head_size
        self.dropout = float(config.hidden_dropout_prob)

        self.dense = nn.Dense(all_head_size, config.hidden_size, has_bias=False)

    def construct(self, hidden_states):

        """Constructs the output of the Reformer self-attention layer.

        Args:
            self (ReformerSelfOutput): The instance of the ReformerSelfOutput class.
            hidden_states (torch.Tensor): The input hidden states of the self-attention layer.

        Returns:
            None

        Raises:
            None

        This method takes in the input hidden states and processes them to construct the output of the Reformer
        self-attention layer. The hidden states are first passed through a dense layer to transform the dimensions.
        Then, a dropout operation is applied to the transformed hidden states with a dropout probability specified by
        the instance variable 'dropout'. The dropout operation is only applied during training.
        Finally, the processed hidden states are returned as the output of the layer.

        Note:
            The output does not have any explicit restrictions as it is of type None and does not affect subsequent
            operations in the model.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ReformerAttention(nn.Cell):

    """
    This class represents a ReformerAttention module, which is used in the Reformer model for attention mechanisms.
    It inherits from the nn.Cell class.

    Attributes:
        layer_id (int): The ID of the attention layer.
        attn_layers (list): The types of attention layers used in the Reformer model.
        layer_norm (nn.LayerNorm): Layer normalization module applied to the input hidden states.
        self_attention (LSHSelfAttention or LocalSelfAttention): The self-attention module used based on
            the attention layer type.
        output (ReformerSelfOutput): The module responsible for the final output of the self-attention mechanism.

    Methods:
        construct: Applies the ReformerAttention module to the input hidden_states and returns the attention output.

    Note:
        - Only 'lsh' and 'local' attention layer types are supported. The attention layer types can be selected from
        ['lsh', 'local'] only.

    Raises:
        NotImplementedError: If the input attention layer types are not 'lsh' or 'local'.

    """
    def __init__(self, config, layer_id=0):

        """
        Initialize the ReformerAttention class.

        Args:
            self (object): The instance of the ReformerAttention class.
            config (object): An object containing configuration settings for the attention layer.
            layer_id (int, optional): The ID of the layer within the attention module. Defaults to 0.

        Returns:
            None.

        Raises:
            NotImplementedError: If the specified attention layer types are not 'lsh' or 'local'.
        """
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = config.attn_layers

        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "lsh":
            self.self_attention = LSHSelfAttention(config)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == "local":
            self.self_attention = LocalSelfAttention(config)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == {"lsh", "local"}:
            # get correct attn layers
            if self.attn_layers[self.layer_id] == "lsh":
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            raise NotImplementedError(
                f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. "
                "Select attn layer types from ['lsh', 'local'] only."
            )
        self.output = ReformerSelfOutput(config)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
        buckets=None,
    ):

        """
        This method constructs the attention output in the ReformerAttention class.

        Args:
            self: Reference to the class instance.
            hidden_states (torch.Tensor): Input tensor containing the hidden states.
            attention_mask (torch.Tensor, optional): Mask tensor for controlling attention computation.
            head_mask (torch.Tensor, optional): Mask tensor for controlling attention within each head.
            num_hashes (int, optional): Number of hashes to use in hashing attention.
            past_buckets_states (torch.Tensor, optional): Tensor containing past bucket states.
            use_cache (bool, optional): Flag indicating whether to use cache for attention computation.
            orig_sequence_length (int, optional): Original length of the sequence.
            output_attentions (bool, optional): Flag indicating whether to output attention weights.
            buckets (torch.Tensor, optional): Tensor containing bucket information.

        Returns:
            None.

        Raises:
            ValueError: If the orig_sequence_length is less than 1.
            AttributeError: If the self_attention_outputs object does not have the 'buckets' attribute.
            IndexError: If the layer_id in past_buckets_states is out of bounds.
            RuntimeError: If there is an issue with concatenating tensors using ops.cat.
            TypeError: If the input types are not as expected.
            Exception: For any other unforeseen errors during the method execution.
        """
        hidden_states = self.layer_norm(hidden_states)

        # make sure cached hidden states is set to None for backward pass
        if past_buckets_states is not None:
            past_buckets_states_layer = past_buckets_states[self.layer_id]
        else:
            past_buckets_states_layer = None

        # use cached buckets for backprob if buckets not None for LSHSelfAttention
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states_layer,
            use_cache=use_cache,
            output_attentions=output_attentions,
            buckets=buckets,
        )

        # add buckets if necessary
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        # cache hidden states for future use
        if use_cache:
            if past_buckets_states[self.layer_id][0] is None:
                # padded input should not be cached
                past_buckets = (
                    buckets[:, :, :, :orig_sequence_length]
                    if (buckets is not None and orig_sequence_length > 1)
                    else buckets
                )
            else:
                past_buckets = ops.cat([past_buckets_states[self.layer_id][0], buckets], axis=-1)

            if past_buckets_states[self.layer_id][1] is None:
                # padded input should not be cached
                past_states = hidden_states[:, :orig_sequence_length]
            else:
                past_states = ops.cat([past_buckets_states[self.layer_id][1], hidden_states], axis=1)

            past_buckets_states[self.layer_id] = (past_buckets, past_states)
        # compute attention feed forward output
        attention_output = self.output(self_attention_outputs.hidden_states)

        return AttentionOutput(
            hidden_states=attention_output,
            attention_probs=self_attention_outputs.attention_probs,
            buckets=buckets,
        )


class ReformerFeedForwardDense(nn.Cell):

    """
    ReformerFeedForwardDense represents a feedforward dense layer used in a Reformer model for neural network operations.

    Attributes:
        dropout (float): The dropout rate for the hidden states.
        act_fn (function): The activation function used for the hidden states.
        dense (nn.Dense): The dense layer for transforming hidden states.

    Methods:
        __init__: Initializes the ReformerFeedForwardDense instance with the provided configuration.
        construct: Constructs the feedforward dense layer by applying dense transformation, dropout,
            and activation function to the hidden states.

    This class inherits from nn.Cell and includes methods to initialize and construct the feedforward dense layer in
    a Reformer model.
    """
    def __init__(self, config):

        """
        Initializes a ReformerFeedForwardDense object with the specified configuration.

        Args:
            self (ReformerFeedForwardDense): The instance of the ReformerFeedForwardDense class.
            config (object): The configuration object containing various settings for the dense layer.
                Expected to have the following attributes:

                - hidden_dropout_prob (float): The dropout probability for the hidden layer.
                - hidden_act (str or function): The activation function for the hidden layer.
                - hidden_size (int): The size of the hidden layer.
                - feed_forward_size (int): The size of the feed-forward layer.

        Returns:
            None.

        Raises:
            TypeError: If the config.hidden_dropout_prob is not a float.
            KeyError: If the config.hidden_act is not a valid activation function name.
            AttributeError: If the config object is missing any of the required attributes.
        """
        super().__init__()
        self.dropout = float(config.hidden_dropout_prob)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        self.dense = nn.Dense(config.hidden_size, config.feed_forward_size)

    def construct(self, hidden_states):

        """
        Constructs the feedforward dense layer for the Reformer model.

        Args:
            self (ReformerFeedForwardDense): An instance of the ReformerFeedForwardDense class.
            hidden_states (tensor): The input hidden states to be processed by the feedforward dense layer.

        Returns:
            tensor: The processed hidden states after passing through the feedforward dense layer.

        Raises:
            ValueError: If the hidden_states tensor is not provided.
            TypeError: If the input hidden_states tensor is not of type tensor.
            RuntimeError: If an error occurs during the dropout operation.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states


class ReformerFeedForwardOutput(nn.Cell):

    """
    Represents the output of the feed forward layer in a Reformer neural network.

    This class inherits from nn.Cell and contains methods for initializing and constructing the feed forward layer output.

    Attributes:
        dropout (float): The dropout rate for the hidden units.
        dense (nn.Dense): The fully connected layer for transforming input hidden states.

    Methods:
        __init__: Initializes the ReformerFeedForwardOutput with the given configuration.
        construct: Constructs the output of the feed forward layer using the provided hidden states.
    """
    def __init__(self, config):

        """
        Initializes an instance of the ReformerFeedForwardOutput class.

        Args:
            self: The object instance.
            config:
                An object containing the configuration parameters.

                - Type: Any valid object.
                - Purpose: Specifies the configuration settings for the ReformerFeedForwardOutput instance.
                - Restrictions: None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dropout = float(config.hidden_dropout_prob)

        self.dense = nn.Dense(config.feed_forward_size, config.hidden_size)

    def construct(self, hidden_states):

        """
        Constructs the output of the feed-forward layer in the Reformer model.

        Args:
            self (ReformerFeedForwardOutput): The instance of the ReformerFeedForwardOutput class.
            hidden_states (tensor): The input hidden states to be processed by the feed-forward layer.

        Returns:
            tensor: The processed hidden states after passing through the feed-forward layer.

        Raises:
            ValueError: If the hidden_states tensor is not valid or has incorrect dimensions.
            RuntimeError: If an error occurs during the computation of the output tensor.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        return hidden_states


class ChunkReformerFeedForward(nn.Cell):

    """
    This class represents a feed-forward module for chunked reformer attention output in a neural network.

    The ChunkReformerFeedForward class inherits from the nn.Cell class and is designed to process attention output
    in a chunked manner. It applies layer normalization, dense transformation, and output transformation to the input
    hidden states.

    Attributes:
        chunk_size_feed_forward (int): The size of each chunk for processing the attention output.
        seq_len_dim (int): The dimension representing the sequence length in the attention output.
        layer_norm (nn.LayerNorm): The layer normalization module applied to the hidden states.
        dense (ReformerFeedForwardDense): The dense transformation module applied to the layer-normalized hidden states.
        output (ReformerFeedForwardOutput): The final output transformation module applied to the
            transformed hidden states.

    Methods:
        __init__(self, config):
            Initializes a new instance of the ChunkReformerFeedForward class.

            Args:

            - config (object): The configuration object containing hyperparameters and settings for the module.

        construct(self, attention_output):
            Constructs the output of the ChunkReformerFeedForward module.

            Args:

            - attention_output (Tensor): The input attention output tensor.

            Returns:

            - Tensor: The constructed output tensor after applying layer normalization, dense transformation,
            and output transformation.

        construct_chunk(self, hidden_states):
            Constructs a chunk of the ChunkReformerFeedForward module.

            Args:

            - hidden_states (Tensor): The input hidden states tensor.

            Returns:

            - Tensor: The constructed chunk of the feed-forward module after applying layer normalization,
            dense transformation, and output transformation.
    """
    def __init__(self, config):

        """
        Initializes a ChunkReformerFeedForward instance.

        Args:
            self (ChunkReformerFeedForward): The ChunkReformerFeedForward instance itself.
            config:
                A configuration object containing the necessary parameters for initialization.

                - chunk_size_feed_forward (int): The chunk size for feed-forward operations.
                - hidden_size (int): The size of the hidden layers.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not of the expected type.
            ValueError: If any required parameter is missing in the config.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dense = ReformerFeedForwardDense(config)
        self.output = ReformerFeedForwardOutput(config)

    def construct(self, attention_output):

        """
        Constructs the feed-forward chunk reformer for the given attention output.

        Args:
            self (ChunkReformerFeedForward): Instance of the ChunkReformerFeedForward class.
            attention_output (Tensor): The attention output tensor to be processed by the chunk reformer.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the chunk size for feed-forward is invalid.
            RuntimeError: If there is an issue with applying chunking during the forward pass.
        """
        return apply_chunking_to_forward(
            self.construct_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

    def construct_chunk(self, hidden_states):

        """
        Constructs a chunk of reformer feed forward layer.

        Args:
            self (ChunkReformerFeedForward): The instance of the ChunkReformerFeedForward class.
            hidden_states (tensor): The input tensor containing the hidden states.

        Returns:
            None.

        Raises:
            AttributeError: If the 'layer_norm', 'dense', or 'output' attributes are not found in the instance.
            ValueError: If the 'hidden_states' parameter is not a valid tensor.
        """
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


class ReformerLayer(nn.Cell):

    """
    Represents a Reformer layer that consists of an attention mechanism and a feed forward network.
    This class inherits from nn.Cell.

    Attributes:
        attention_seed (int): Seed for the attention layer to ensure deterministic dropout behavior.
        feed_forward_seed (int): Seed for the feed forward layer to ensure deterministic dropout behavior.

    Methods:
        __init__: Initializes the ReformerLayer with attention and feed forward components.
        _init_attention_seed: Sets a new seed for the attention layer to ensure deterministic dropout behavior.
        _init_feed_forward_seed: Sets a new seed for the feed forward layer to ensure deterministic dropout behavior.
        construct: Constructs the Reformer layer by applying attention and feed forward operations and returning the output.

    Raises:
        None.
    """
    def __init__(self, config, layer_id=0):

        """
        Initializes a new instance of the ReformerLayer class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing various settings and parameters.
            layer_id (int, optional): The identifier for the layer. Defaults to 0.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention = ReformerAttention(config, layer_id)
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = ChunkReformerFeedForward(config)

    def _init_attention_seed(self):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """
        # randomize seeds
        np.random.seed(None)
        self.attention_seed = int(np.random.randint(0, 100000) % sys.maxsize)

        mindspore.set_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
        This function sets a new seed for the feed forward layer to make dropout deterministic for both forward calls:
        1 normal forward call and 1 forward call in backward to recalculate activations.
        """
        # randomize seeds
        np.random.seed(None)
        self.feed_forward_seed = int(np.random.randint(0, 100000) % sys.maxsize)

        mindspore.set_seed(self.feed_forward_seed)

    def construct(
        self,
        prev_attn_output,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_attentions=False,
    ):

        """
        Constructs a Reformer layer by performing attention and feed forward operations on the input hidden states.

        Args:
            self: The object instance.
            prev_attn_output (torch.Tensor): The previous attention output tensor.
                Shape: (batch_size, sequence_length, hidden_size)
            hidden_states (torch.Tensor): The input hidden states tensor.
                Shape: (batch_size, sequence_length, hidden_size)
            attention_mask (torch.Tensor, optional): The attention mask tensor.
                Shape: (batch_size, sequence_length). Defaults to None.
            head_mask (torch.Tensor, optional): The head mask tensor.
                Shape: (num_heads, sequence_length, sequence_length). Defaults to None.
            num_hashes (int, optional): The number of hashes for hashing attention. Defaults to None.
            past_buckets_states (torch.Tensor, optional): The tensor of past bucket states.
                Shape: (batch_size, sequence_length, num_hashes, buckets). Defaults to None.
            use_cache (bool, optional): Whether to use cache for attention. Defaults to False.
            orig_sequence_length (int, optional): The original sequence length before padding. Defaults to None.
            output_attentions (bool, optional): Whether to output attention probabilities. Defaults to False.

        Returns:
            ReformerOutput:
                An object containing the attention output, hidden states, attention probabilities, and buckets.

        Raises:
            None.
        """
        # every forward pass we sample a different seed
        # for dropout and save for forward fn in backward pass
        # to have correct dropout

        if self.training:
            self._init_attention_seed()

        attn_outputs = self.attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs.hidden_states

        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # Y_1 = X_1 + f(X_2)
        attn_output = prev_attn_output + attn_output

        # free memory
        del prev_attn_output

        # every forward pass we sample a different seed
        # for dropout and save seed for forward fn in backward
        # to have correct dropout
        if self.training:
            self._init_feed_forward_seed()
        # Y_2 = X_2 + g(Y_1)
        hidden_states = hidden_states + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets,
        )

    # def backward_pass(
    #     self,
    #     next_attn_output,
    #     hidden_states,
    #     grad_attn_output,
    #     grad_hidden_states,
    #     attention_mask=None,
    #     head_mask=None,
    #     buckets=None,
    # ):
    #     # Implements the backward pass for reversible ResNets.
    #     # A good blog post on how this works can be found here:
    #     # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
    #     # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py

    #     assert self.training, (
    #         "If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the"
    #         " model into training mode."
    #     )

    #     with torch.enable_grad():
    #         next_attn_output.requires_grad = True

    #         # set seed to have correct dropout
    #         torch.manual_seed(self.feed_forward_seed)
    #         # g(Y_1)
    #         res_hidden_states = self.feed_forward(next_attn_output)
    #         res_hidden_states.backward(grad_hidden_states, retain_graph=True)

    #     with torch.no_grad():
    #         # X_2 = Y_2 - g(Y_1)
    #         hidden_states = hidden_states - res_hidden_states
    #         del res_hidden_states

    #         grad_attn_output = grad_attn_output + next_attn_output.grad
    #         next_attn_output.grad = None

    #     with torch.enable_grad():
    #         hidden_states.requires_grad = True

    #         # set seed to have correct dropout
    #         torch.manual_seed(self.attention_seed)
    #         # f(X_2)
    #         # use cached buckets for backprob if buckets not None for LSHSelfAttention
    #         output = self.attention(
    #             hidden_states=hidden_states,
    #             head_mask=head_mask,
    #             attention_mask=attention_mask,
    #             buckets=buckets,
    #         ).hidden_states
    #         output.backward(grad_attn_output, retain_graph=True)

    #     # X_1 = Y_1 - f(X_2)
    #     attn_output = next_attn_output - output
    #     del output, next_attn_output

    #     grad_hidden_states = grad_hidden_states + hidden_states.grad
    #     hidden_states.grad = None
    #     hidden_states = hidden_states.detach()

    #     return ReformerBackwardOutput(
    #         attn_output=attn_output,
    #         hidden_states=hidden_states,
    #         grad_attn_output=grad_attn_output,
    #         grad_hidden_states=grad_hidden_states,
    #     )


class _ReversibleFunction(nn.Cell):
    """
    To prevent PyTorch from performing the usual backpropagation, a customized backward function is implemented here.
    This way it is made sure that no memory expensive activations are saved during the forward pass. This function is
    heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py
    """
    def construct(
        self,
        hidden_states,
        layers,
        attention_mask,
        head_mask,
        num_hashes,
        all_hidden_states,
        all_attentions,
        past_buckets_states,
        use_cache,
        orig_sequence_length,
        output_hidden_states,
        output_attentions,
    ):

        """
        This method 'construct' is defined within the class '_ReversibleFunction' and is responsible for constructing
        the output based on the provided input parameters.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The input hidden states.
            layers (List[Layer]): List of layers to be applied during the construction process.
            attention_mask (Tensor): The attention mask to be applied during computations.
            head_mask (List[Tensor]): List of head masks for each layer.
            num_hashes (int): The number of hashes used in computations.
            all_hidden_states (List[Tensor]): List to store all hidden states if 'output_hidden_states' is True.
            all_attentions (List[Tensor]): List to store all attention probabilities if 'output_attentions' is True.
            past_buckets_states (Tensor): Past bucket states used in computations.
            use_cache (bool): Indicates whether to use caching during computations.
            orig_sequence_length (int): The original length of the input sequence.
            output_hidden_states (bool): Flag to determine if hidden states should be stored.
            output_attentions (bool): Flag to determine if attention probabilities should be stored.

        Returns:
            None: This method does not return any value explicitly; it updates the state of the class instance.

        Raises:
            None: This method does not raise any exceptions.
        """
        all_buckets = ()

        # split duplicated tensor
        hidden_states, attn_output = ops.chunk(hidden_states, 2, axis=-1)

        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                prev_attn_output=attn_output,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                num_hashes=num_hashes,
                past_buckets_states=past_buckets_states,
                use_cache=use_cache,
                orig_sequence_length=orig_sequence_length,
                output_attentions=output_attentions,
            )

            attn_output = layer_outputs.attn_output
            hidden_states = layer_outputs.hidden_states
            all_buckets = all_buckets + (layer_outputs.buckets,)

            if output_attentions:
                all_attentions.append(layer_outputs.attention_probs)

        # Add last layer
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)

        # # attach params to ctx for backward
        # ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        # ctx.layers = layers
        # ctx.all_buckets = all_buckets
        # ctx.head_mask = head_mask
        # ctx.attention_mask = attention_mask

        # Concatenate 2 RevNet outputs
        return ops.cat([attn_output, hidden_states], axis=-1)

    # @staticmethod
    # def backward(ctx, grad_hidden_states):
    #     grad_attn_output, grad_hidden_states = ops.chunk(grad_hidden_states, 2, dim=-1)

    #     # retrieve params from ctx for backward
    #     attn_output, hidden_states = ctx.saved_tensors

    #     # create tuple
    #     output = ReformerBackwardOutput(
    #         attn_output=attn_output,
    #         hidden_states=hidden_states,
    #         grad_attn_output=grad_attn_output,
    #         grad_hidden_states=grad_hidden_states,
    #     )

    #     # free memory
    #     del grad_attn_output, grad_hidden_states, attn_output, hidden_states

    #     layers = ctx.layers
    #     all_buckets = ctx.all_buckets
    #     head_mask = ctx.head_mask
    #     attention_mask = ctx.attention_mask

    #     for idx, layer in enumerate(layers[::-1]):
    #         # pop last buckets from stack
    #         buckets = all_buckets[-1]
    #         all_buckets = all_buckets[:-1]

    #         # backprop
    #         output = layer.backward_pass(
    #             next_attn_output=output.attn_output,
    #             hidden_states=output.hidden_states,
    #             grad_attn_output=output.grad_attn_output,
    #             grad_hidden_states=output.grad_hidden_states,
    #             head_mask=head_mask[len(layers) - idx - 1],
    #             attention_mask=attention_mask,
    #             buckets=buckets,
    #         )

    #     assert all_buckets == (), "buckets have to be empty after backpropagation"
    #     grad_hidden_states = ops.cat([output.grad_attn_output, output.grad_hidden_states], axis=-1)

    #     # num of return vars has to match num of forward() args
    #     # return gradient for hidden_states arg and None for other args
    #     return grad_hidden_states, None, None, None, None, None, None, None, None, None, None, None


class ReformerEncoder(nn.Cell):

    """
    The 'ReformerEncoder' class is a Python class that represents the encoder component of the Reformer model.
    It inherits from the 'nn.Cell' class.

    Attributes:
        dropout (float): The dropout probability for the hidden states.
        layers (nn.CellList): A list of 'ReformerLayer' instances representing the layers of the encoder.
        layer_norm (nn.LayerNorm): A layer normalization module.

    Methods:
        __init__: Initializes a new instance of the 'ReformerEncoder' class.
        construct: Constructs the encoder by applying the Reformer layers to the input hidden states.

    """
    def __init__(self, config):

        """
        Initializes a ReformerEncoder instance.

        Args:
            self (ReformerEncoder): The ReformerEncoder instance to be initialized.
            config (Config):
                A configuration object containing settings for the ReformerEncoder.

                - config.hidden_dropout_prob (float): The dropout probability for hidden layers.
                - config.num_hidden_layers (int): The number of hidden layers in the ReformerEncoder.
                - config.hidden_size (int): The size of hidden layers.
                - config.layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            TypeError: If config is not of type Config.
            ValueError: If config is missing any required attributes.
            ValueError: If config.hidden_dropout_prob is not a float.
            ValueError: If config.num_hidden_layers is not an integer.
            ValueError: If config.hidden_size is not an integer.
            ValueError: If config.layer_norm_eps is not a float.
        """
        super().__init__()
        self.dropout = float(config.hidden_dropout_prob)

        self.layers = nn.CellList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        num_hashes=None,
        past_buckets_states=None,
        use_cache=False,
        orig_sequence_length=None,
        output_hidden_states=False,
        output_attentions=False,
    ):

        """
        Constructs the ReformerEncoder output given the input parameters.

        Args:
            self: The class instance.
            hidden_states (Tensor): The input hidden states. Shape [batch_size, sequence_length, hidden_size].
            attention_mask (Tensor, optional): The attention mask. Shape [batch_size, sequence_length, sequence_length].
                Masks the attention scores for padding tokens. Defaults to None.
            head_mask (Tensor, optional): The head mask. Shape [num_attention_heads, sequence_length, sequence_length].
                Masks the attention scores for specific attention heads. Defaults to None.
            num_hashes (int, optional): The number of hashes to use for LSH attention. Defaults to None.
            past_buckets_states (list, optional): The list of past bucket states. Each element is a tuple of two tensors.
                Shape [(buckets, num_hashes, sequence_length // bucket_size, embedding_dim),
                (buckets, num_hashes, sequence_length // bucket_size)].
                Defaults to None.
            use_cache (bool, optional): Whether to use cache for fast decoding. Defaults to False.
            orig_sequence_length (int, optional): The original sequence length before padding. Defaults to None.
            output_hidden_states (bool, optional): Whether to output all hidden states. Defaults to False.
            output_attentions (bool, optional): Whether to output all attention matrices. Defaults to False.

        Returns:
            ReformerEncoderOutput:
                An instance of the ReformerEncoderOutput class containing the following attributes:

                - hidden_states (Tensor): The output hidden states. Shape [batch_size, sequence_length, hidden_size].
                - all_hidden_states (list): List of hidden states at each layer. Each element has shape
                [batch_size, sequence_length, hidden_size].
                - all_attentions (list): List of attention matrices at each layer. Each element has shape
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                - past_buckets_states (list): List of past bucket states for fast decoding.
                Each element is a tuple of two tensors.
                Shape [(buckets, num_hashes, sequence_length // bucket_size, embedding_dim),
                (buckets, num_hashes, sequence_length // bucket_size)].

        Raises:
            None.
        """
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # init cached hidden states if necessary
        if past_buckets_states is None:
            past_buckets_states = [((None), (None)) for i in range(len(self.layers))]

        # concat same tensor for reversible ResNet
        hidden_states = ops.cat([hidden_states, hidden_states], axis=-1)
        hidden_states = _ReversibleFunction()(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            past_buckets_states,
            use_cache,
            orig_sequence_length,
            output_hidden_states,
            output_attentions,
        )
        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        return ReformerEncoderOutput(
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_attentions=all_attentions,
            past_buckets_states=past_buckets_states,
        )


class ReformerOnlyLMHead(nn.Cell):

    """
    The class 'ReformerOnlyLMHead' represents a language model head for the Reformer model.
    It inherits from the 'nn.Cell' class and contains methods for initialization, construction, chunking,
    and weight tying.

    Attributes:
        seq_len_dim (int): The dimension representing the sequence length.
        chunk_size_lm_head (int): The chunk size for the language model head.
        decoder (nn.Dense): The dense layer for decoding hidden states.
        bias (Parameter): The bias parameter for the decoder.

    Methods:
        __init__: Initializes the ReformerOnlyLMHead instance with the provided configuration.
        construct: Constructs the language model head using chunking for the given hidden states.
        construct_chunk: Constructs a chunk of the language model head for the given hidden states.
        _tie_weights: Ties the weights of the bias to the decoder bias.
    """
    def __init__(self, config):

        """
        Initializes a new instance of the ReformerOnlyLMHead class.

        Args:
            self: The object itself.
            config: An instance of the Configuration class containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.seq_len_dim = 1
        self.chunk_size_lm_head = config.chunk_size_lm_head
        self.decoder = nn.Dense(2 * config.hidden_size, config.vocab_size, has_bias=False)
        self.bias = Parameter(initializer('zeros', (config.vocab_size,)), 'bias')
        self.decoder.bias = self.bias

    def construct(self, hidden_states):

        """
        Constructs the LM head output for the Reformer model.

        Args:
            self (ReformerOnlyLMHead): The instance of the ReformerOnlyLMHead class.
            hidden_states (Tensor): The input hidden states from the Reformer model.
                Expected shape: (batch_size, sequence_length, hidden_size).
                Purpose: Represents the hidden states from the Reformer model.
                Restrictions: Must be a valid tensor with the correct shape.

        Returns:
            None.

        Raises:
            TypeError: If the input hidden_states is not a valid tensor.
            ValueError: If the input hidden_states has an incorrect shape.
        """
        return apply_chunking_to_forward(self.construct_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)

    def construct_chunk(self, hidden_states):

        """
        Args:
            self (ReformerOnlyLMHead): The instance of the ReformerOnlyLMHead class.
            hidden_states (Tensor): The input hidden states to be processed.

        Returns:
            Tensor: The processed hidden states after passing through the decoder.

        Raises:
            None.
        """
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self):

        """
        Method _tie_weights in class ReformerOnlyLMHead.

        Args:
            self: (ReformerOnlyLMHead) The instance of the ReformerOnlyLMHead class.
                It is used to access the attributes and methods of the class.

        Returns:
            None.

        Raises:
            None
        """
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class ReformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ReformerConfig
    base_model_prefix = "reformer"

    @property
    def dummy_inputs(self):

        """
        Returns dummy input tensors for the Reformer PreTrained Model.

        Args:
            self: An instance of the ReformerPreTrainedModel class.

        Returns:
            dict: A dictionary containing dummy input tensors for the model.
                The dictionary has two keys:

                - 'input_ids': A tensor representing dummy input IDs.
                - 'attention_mask': A tensor representing dummy attention mask.

        Raises:
            None.
        """
        input_ids = mindspore.tensor(DUMMY_INPUTS)
        input_mask = mindspore.tensor(DUMMY_MASK)
        dummy_inputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, AxialPositionEmbeddings):
            for weight in cell.weights:
                weight.set_data(initializer(Normal(self.config.axial_norm_std), weight.shape, weight.dtype))
        elif isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(mindspore.Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


@dataclass
class ReformerModelOutput(ModelOutput):
    """
    Output type of [`ReformerModel`].

    Args:
        last_hidden_state (`mindspore.Tensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        past_buckets_states (`List[Tuple(mindspore.Tensor, mindspore.Tensor)]`, *optional*, returned
            when `use_cache=True` is passed or when `config.use_cache=True`):
            List of `Tuple(mindspore.Tensor, mindspore.Tensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
            up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: mindspore.Tensor
    past_buckets_states: Optional[List[Tuple[mindspore.Tensor, mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class ReformerModelWithLMHeadOutput(ModelOutput):
    """
    Output type of [`ReformerModelWithLMHead`].

    Args:
        loss (`mindspore.Tensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`mindspore.Tensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        past_buckets_states (`List[Tuple(mindspore.Tensor, mindspore.Tensor)]`, *optional*, returned when `use_cache=True`
            is passed or when `config.use_cache=True`):
            List of `Tuple(mindspore.Tensor, mindspore.Tensor` of length `config.n_layers`, with the first element
            being the previous *buckets* of shape `(batch_size, num_heads, num_hashes, sequence_length)`) and the
            second being the previous *hidden_states* of shape `(batch_size, sequence_length, hidden_size)`).

            Contains precomputed buckets and hidden-states that can be used (see `past_buckets_states` input) to speed
            up sequential decoding.
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            TTuple of `mindspore.Tensor` (one for the output of the embeddings and one for the output of each layer)
            of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    past_buckets_states: Optional[List[Tuple[mindspore.Tensor, mindspore.Tensor]]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class ReformerModel(ReformerPreTrainedModel):

    """ReformerModel

    This class represents a Reformer model. It is a subclass of ReformerPreTrainedModel,
    which provides the base implementation for a pre-trained Reformer model.

    Attributes:
        config (ReformerConfig): The configuration instance of the Reformer model.
        embeddings (ReformerEmbeddings): The embeddings layer of the model.
        encoder (ReformerEncoder): The encoder layer of the model.

    Methods:
        __init__: Initializes a new instance of the ReformerModel.
        get_input_embeddings: Retrieves the input embeddings layer of the model.
        set_input_embeddings: Sets the input embeddings layer of the model.
        _prune_heads: Prunes the attention heads of the model.
        construct: Constructs the Reformer model with the given input and configurations.
        _pad_to_mult_of_chunk_length: Pads the input tensors to be a multiple of the chunk length according to
            the Reformer model configuration.
    """
    def __init__(self, config):

        """
        Initializes an instance of the ReformerModel class.

        Args:
            self: The current instance of the ReformerModel class.
            config: An object containing the configuration settings for the ReformerModel.
                It must have the following attributes:

                - num_hidden_layers (int): The number of hidden layers in the model. Must be greater than 0.

        Returns:
            None.

        Raises:
            AssertionError: If the `config.num_hidden_layers` attribute is not greater than 0.

        """
        super().__init__(config)
        self.config = config
        assert (
            self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"

        self.embeddings = ReformerEmbeddings(config)
        self.encoder = ReformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):

        """

        Description:
            This method retrieves the input embeddings from the ReformerModel.

        Args:
            self (object): The instance of the ReformerModel class.

        Returns:
            None: This method returns the input embeddings of type 'None'.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):

        """
        Sets the input embeddings for the ReformerModel.

        Args:
            self (ReformerModel): The instance of the ReformerModel.
            value: The input embeddings to be set. Should be of type torch.Tensor.

        Returns:
            None.

        Raises:
            TypeError: If the input embeddings are not of type torch.Tensor.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        num_hashes: Optional[int] = None,
        past_buckets_states: Optional[List[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ReformerModelOutput]:

        """
        Constructs the Reformer model.

        Args:
            self (ReformerModel): The instance of the ReformerModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the token indices. Default: None.
            attention_mask (Optional[mindspore.Tensor]): The tensor indicating which tokens should be attended to.
                Default: None.
            position_ids (Optional[mindspore.Tensor]): The tensor containing the position indices. Default: None.
            head_mask (Optional[mindspore.Tensor]): The tensor indicating which heads should be masked. Default: None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing the input embeddings. Default: None.
            num_hashes (Optional[int]): The number of hashes to use for the LSH attention. Default: None.
            past_buckets_states (Optional[List[Tuple[mindspore.Tensor]]]):
                The list of tensors containing the past bucket states. Default: None.
            use_cache (Optional[bool]): Whether to use the cache for the attention computation. Default: None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default: None.
            output_attentions (Optional[bool]): Whether to output attention weights. Default: None.
            return_dict (Optional[bool]): Whether to return a dictionary as the output. Default: None.

        Returns:
            Union[Tuple, ReformerModelOutput]: A tuple or an instance of the ReformerModelOutput class
                containing the model output.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            AssertionError: If the input_ids shape is not [batch_size, sequence_length].
            AssertionError: If past_buckets_states is used during training.
            ValueError: If the input sequence length is not a multiple of the least common multiple chunk_length.
            ValueError: If the input sequence length is not a multiple of the least common multiple chunk_length
                during training.

        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape  # noqa: F841
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]  # noqa: F841
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert (
            len(input_shape) == 2
        ), f"`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {input_shape}"

        if past_buckets_states is not None:
            assert not self.training, "`past_buckets_states` can only be used for inference, not for training`."

        # prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, is_attention_chunked=True)

        # original sequence length for padding
        orig_sequence_length = input_shape[-1]

        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        min_chunk_length = _get_min_chunk_len(self.config)

        must_pad_to_match_chunk_length = (
            input_shape[-1] % least_common_mult_chunk_length != 0
            and input_shape[-1] > min_chunk_length
            and past_buckets_states is None
        )

        if must_pad_to_match_chunk_length:
            padding_length = least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length

            if self.training is True:
                raise ValueError(
                    f"If training, sequence length {input_shape[-1]} has to be a multiple of least common multiple "
                    f"chunk_length {least_common_mult_chunk_length}. Please consider padding the input to a length "
                    f"of {input_shape[-1] + padding_length}."
                )

            # pad input
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
            )

        # start index for position encoding depends on incremental decoding
        if past_buckets_states is not None:
            start_idx_pos_encodings = past_buckets_states[0][1].shape[1]
        else:
            start_idx_pos_encodings = 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            start_idx_pos_encodings=start_idx_pos_encodings,
        )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        sequence_output = encoder_outputs.hidden_states

        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        past_buckets_states = encoder_outputs.past_buckets_states if use_cache else None
        hidden_states = encoder_outputs.all_hidden_states if output_hidden_states else None
        attentions = encoder_outputs.all_attentions if output_attentions else None

        if not return_dict:
            return tuple(v for v in [sequence_output, past_buckets_states, hidden_states, attentions] if v is not None)
        return ReformerModelOutput(
            last_hidden_state=sequence_output,
            past_buckets_states=past_buckets_states,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def _pad_to_mult_of_chunk_length(
        self,
        input_ids,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        input_shape=None,
        padding_length=None,
        padded_seq_length=None,
    ):

        """
        This method '_pad_to_mult_of_chunk_length' in the class 'ReformerModel' pads input data to be a multiple of
        a specified chunk length.

        Args:
            self: The instance of the class.
            input_ids (Tensor): The input tensor containing token ids.
            inputs_embeds (Tensor, optional): The input embeddings tensor.
            attention_mask (Tensor, optional): The tensor specifying the attention mask.
            position_ids (Tensor, optional): The tensor specifying the position ids.
            input_shape (Tuple): The shape of the input tensor.
            padding_length (int): The length to pad the input data to be a multiple of the chunk length.
            padded_seq_length (int): The length of the padded sequence.

        Returns:
            Tuple: A tuple containing the padded input_ids, inputs_embeds, attention_mask, position_ids, and
                updated input_shape.

        Raises:
            ValueError: If the input data is not in the expected format.
            RuntimeError: If an unexpected error occurs during padding.
        """
        logger.warning_once(
            f"Input ids are automatically padded from {input_shape[-1]} to {input_shape[-1] + padding_length} to be a "
            f"multiple of `config.chunk_length`: {padded_seq_length}"
        )

        padded_input_ids = ops.full(
            (input_shape[0], padding_length),
            self.config.pad_token_id,
            dtype=mindspore.int64,
        )

        # Extend `attention_mask`
        if attention_mask is not None:
            pad_attention_mask = ops.zeros(input_shape[0], padding_length, dtype=attention_mask.dtype)

            attention_mask = ops.cat([attention_mask, pad_attention_mask], axis=-1)
        else:
            attention_mask = ops.cat(
                [
                    ops.ones(input_shape, dtype=mindspore.bool_),
                    ops.zeros((input_shape[0], padding_length), dtype=mindspore.bool_),
                ],
                axis=-1,
            )

        # Extend `input_ids` with padding to match least common multiple chunk_length
        if input_ids is not None:
            input_ids = ops.cat([input_ids, padded_input_ids], axis=-1)
            input_shape = input_ids.shape

            # Pad position ids if given
            if position_ids is not None:
                padded_position_ids = ops.arange(input_shape[-1], padded_seq_length, dtype=mindspore.int64)
                padded_position_ids = position_ids.unsqueeze(0).expand(input_shape[0], padding_length)
                position_ids = ops.cat([position_ids, padded_position_ids], axis=-1)

        # Extend `inputs_embeds` with padding to match least common multiple chunk_length
        if inputs_embeds is not None:
            padded_inputs_embeds = self.embeddings(padded_input_ids, position_ids)
            inputs_embeds = ops.cat([inputs_embeds, padded_inputs_embeds], axis=-2)
            input_shape = inputs_embeds.shape
        return input_ids, inputs_embeds, attention_mask, position_ids, input_shape


class ReformerModelWithLMHead(ReformerPreTrainedModel):

    """
    A Python class representing a Reformer model with a language modeling head (LMHead).
    This class inherits from the ReformerPreTrainedModel class.

    The ReformerModelWithLMHead class is designed to be used as a decoder for the Reformer model.
    It incorporates a ReformerModel, which performs the main computation, and a ReformerOnlyLMHead, which generates
    the language modeling predictions.

    Attributes:
        reformer (ReformerModel): The Reformer model used for the main computation.
        lm_head (ReformerOnlyLMHead): The language modeling head used for generating predictions.

    Methods:
        __init__: Initializes the ReformerModelWithLMHead instance with the given configuration.
        get_output_embeddings: Retrieves the decoder of the lm_head.
        set_output_embeddings: Sets the decoder of the lm_head to the provided new_embeddings.
        construct: Constructs the Reformer model with the given inputs and returns the output.
        prepare_inputs_for_generation: Prepares the inputs for generation by selecting the last token and
            returning a dictionary of inputs.
        _reorder_cache: Reorders the past key values for beam search.

    Note:
        The ReformerModelWithLMHead class assumes that the config parameter has an 'is_decoder' attribute set to True.
        It also checks specific conditions related to the 'attn_layers' attribute in the config to ensure the correct
        configuration for causal mask usage.

    """
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):

        """
        Initialize the ReformerModelWithLMHead class with the provided configuration.

        Args:
            self (object): The instance of the ReformerModelWithLMHead class.
            config (object): An object containing configuration settings for the model.
                This parameter is required for initializing the model and must be an instance of the configuration class.
                It should include specific settings such as whether the model is a decoder or not.

        Returns:
            None.

        Raises:
            AssertionError: Raised if the 'is_decoder' flag in the config is not set to True, indicating that the model
                being used is not a decoder.
            AssertionError: Raised if the 'local' key is found in the 'attn_layers' attribute of the config and
                'local_num_chunks_after' is not set to 0 when causal mask is enabled.
            AssertionError: Raised if the 'lsh' key is found in the 'attn_layers' attribute of the config and
                'lsh_num_chunks_after' is not set to 1 when causal mask is enabled.
        """
        super().__init__(config)
        assert config.is_decoder, "If you want to use `ReformerModelWithLMHead` make sure that `is_decoder=True`."
        assert "local" not in self.config.attn_layers or config.local_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.local_num_chunks_after` is set to 0 and not"
            f" {config.local_num_chunks_after}."
        )
        assert "lsh" not in self.config.attn_layers or config.lsh_num_chunks_after == 0, (
            "If causal mask is enabled, make sure that `config.lsh_num_chunks_after` is set to 1 and not"
            f" {config.lsh_num_chunks_after}."
        )

        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):

        """
        Returns the output embeddings of the ReformerModelWithLMHead.

        This method, 'get_output_embeddings', returns the decoder of the language model head of the
        ReformerModelWithLMHead. The decoder is responsible for mapping the hidden states of the model to the
        output vocabulary.

        Args:
            self (ReformerModelWithLMHead): The instance of the ReformerModelWithLMHead class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):

        """
        Sets the output embeddings of the ReformerModelWithLMHead.

        Args:
            self (ReformerModelWithLMHead): The instance of the ReformerModelWithLMHead class.
            new_embeddings: The new embeddings to be set for the output layer.
                It can be any compatible object that can replace the current embeddings.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method replaces the current embeddings in the lm_head.decoder attribute with the provided
            new embeddings. The lm_head.decoder is responsible for generating the output of the ReformerModelWithLMHead.
            By setting new embeddings, the model can be fine-tuned or customized for different tasks or requirements.
        """
        self.lm_head.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        num_hashes: Optional[int] = None,
        past_buckets_states: Optional[List[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        Args:
            labels(`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
                config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
                labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            past_buckets_states=past_buckets_states,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = ops.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ReformerModelWithLMHeadOutput(
            loss=loss,
            logits=logits,
            past_buckets_states=reformer_outputs.past_buckets_states,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, use_cache=None, num_hashes=None, **kwargs
    ):

        """
        Prepare inputs for generation.

        Args:
            self (ReformerModelWithLMHead): The instance of the ReformerModelWithLMHead class.
            input_ids (torch.Tensor): The tensor of input token IDs of shape (batch_size, sequence_length).
            past_key_values (Tuple[torch.Tensor], optional): The tuple of past key values for efficient generation.
                Defaults to None.
            use_cache (bool, optional): Whether to use the cache for fast decoding. Defaults to None.
            num_hashes (int, optional): The number of hashes for LSH attention. Defaults to None.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (torch.Tensor): The tensor of input token IDs.
                - 'past_buckets_states' (Tuple[torch.Tensor]): The tuple of past key values.
                - 'use_cache' (bool): Whether to use the cache for fast decoding.
                - 'num_hashes' (int): The number of hashes for LSH attention.

        Raises:
            None

        """
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        inputs_dict = {
            "input_ids": input_ids,
            "past_buckets_states": past_key_values,
            "use_cache": use_cache,
            "num_hashes": num_hashes,
        }

        return inputs_dict

    def _reorder_cache(self, past_key_values, beam_idx):

        """
        Reorders the cache for the ReformerModelWithLMHead class.

        Args:
            self: An instance of the ReformerModelWithLMHead class.
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples, where each tuple contains
                a past key tensor and a past value tensor for each layer. The past key tensor represents the
                previous key values, and the past value tensor represents the previous hidden states for each layer.
                The length of the list corresponds to the number of layers in the model.
            beam_idx (torch.Tensor): A tensor representing the indices to reorder the cache.
                These indices determine the new order of the cache.

        Returns:
            reord_past_buckets_states (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples, where each tuple
                contains the reordered past key tensor and the reordered past value tensor for each layer.
                The reordered past key tensor represents the previous key values after reordering, and the reordered
                past value tensor represents the previous hidden states after reordering.
                The length of the list corresponds to the number of layers in the model.

        Raises:
            None.
        """
        reord_past_buckets_states = []
        for layer_past in past_key_values:
            # buckets
            if layer_past[0] is not None:
                reord_buckets = layer_past[0].index_select(0, beam_idx)
            else:
                reord_buckets = None

            # hidden states
            reord_hidden_states = layer_past[1].index_select(0, beam_idx)
            reord_past_buckets_states.append((reord_buckets, reord_hidden_states))
        return reord_past_buckets_states


class ReformerForMaskedLM(ReformerPreTrainedModel):
    r"""
    A Reformer model with a language modeling head for masked language modeling tasks.

    This class inherits from `ReformerPreTrainedModel` and utilizes the Reformer architecture and a language modeling head for
    masking language modeling tasks. The class is capable of generating output embeddings and setting new embeddings with
    the provided methods, `get_output_embeddings()` and `set_output_embeddings()`, respectively, and the `construct()`
    method constructs the model and computes the masked language modeling loss if the `labels` argument is provided.

    The `ReformerForMaskedLM` class takes a `config` argument, which is an instance of `ReformerConfig`. The class
    implements the `__init__()` method that initializes the parent class with the provided `config`. The method also
    checks that `config.is_decoder=False` for bi-directional self-attention.

    The class has the following methods:

    - `get_output_embeddings()`: Returns the decoder for the language modeling head.
    - `set_output_embeddings(new_embeddings)`: Sets the decoder for the language modeling head to `new_embeddings`.
    - `construct(input_ids=None, position_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None,
    num_hashes=None, labels=None, output_hidden_states=None, output_attentions=None, return_dict=None)`:
    Constructs the model and computes the masked language modeling loss if the `labels` argument is provided.
    This method takes several optional input arguments and returns a tuple with the following elements:

        - If `labels` is not `None`, returns a tuple `(masked_lm_loss, logits, hidden_states, attentions)`.
        - If `labels` is `None`, returns a tuple `(logits, hidden_states, attentions)`.

            - `masked_lm_loss` is the masked language modeling loss, computed as cross-entropy loss between the logits
              and the labels.
            - `logits` is a tensor of shape `(batch_size, sequence_length, vocab_size)` containing the unnormalized
              log probabilities for each vocabulary token.
            - `hidden_states` is a tuple of length `config.num_hidden_layers + 1` with each tensor of shape
              `(batch_size, sequence_length, hidden_size)` representing the hidden states of the model at each layer.
            - `attentions` is a tuple of length `config.num_hidden_layers` with each tensor of shape
              `(batch_size, num_heads, sequence_length, sequence_length)` representing the attention weights for each
              layer.

    Note:
        This class utilizes a false checkpoint since there is no available pre-trained model for the masked language
        modeling task with the Reformer architecture.

    Example:
        ```python
        >>> from transformers import ReformerForMaskedLM, ReformerConfig
        ...
        >>> # Initializing a Reformer configuration
        >>> config = ReformerConfig()
        ...
        >>> # Initializing a ReformerForMaskedLM model with the configuration
        >>> model = ReformerForMaskedLM(config)
        ...
        >>> # Getting the decoder for the language modeling head
        >>> decoder = model.get_output_embeddings()
        ...
        >>> # Setting new embeddings for the language modeling head
        >>> model.set_output_embeddings(new_embeddings)
        ...
        >>> # Constructing the model and computing the masked language modeling loss
        >>> masked_lm_loss, logits, hidden_states, attentions = model.construct(input_ids, position_ids, attention_mask,
        ...     head_mask, inputs_embeds, num_hashes, labels, output_hidden_states, output_attentions, return_dict)
        ```

    """
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):

        """
        Initializes an instance of the ReformerForMaskedLM class.

        Args:
            self: The instance of the class.
            config: An instance of the configuration class that specifies the model configuration.

        Returns:
            None

        Raises:
            AssertionError: If `config.is_decoder` is set to True. To use the ReformerForMaskedLM, `config.is_decoder`
                should be set to False for bi-directional self-attention.

        """
        super().__init__(config)
        assert not config.is_decoder, (
            "If you want to use `ReformerForMaskedLM` make sure `config.is_decoder=False` for bi-directional"
            " self-attention."
        )
        self.reformer = ReformerModel(config)
        self.lm_head = ReformerOnlyLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):

        """
        This method retrieves the output embeddings from the ReformerForMaskedLM model.

        Args:
            self: ReformerForMaskedLM - The instance of the ReformerForMaskedLM class.

        Returns:
            None: This method returns None as it only retrieves the output embeddings without any additional processing.

        Raises:
            None.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):

        """
        Sets the output embeddings for the Reformer model.

        Args:
            self (ReformerForMaskedLM): The instance of the ReformerForMaskedLM class.
            new_embeddings: The new embeddings to be set as the output embeddings. It can be of any type.

        Returns:
            None.

        Raises:
            None.

        This method sets the output embeddings of the ReformerForMaskedLM model to the provided new_embeddings.
        The new_embeddings can be any type and will be assigned to the decoder of the lm_head.
        """
        self.lm_head.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        num_hashes: Optional[int] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels

        Returns:
            Union[Tuple, MaskedLMOutput]

        <Tip warning={true}>

        This example uses a false checkpoint since we don't have any available pretrained model for the masked language
        modeling task with the Reformer architecture.

        </Tip>

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoTokenizer, ReformerForMaskedLM
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-reformer")
            >>> model = ReformerForMaskedLM.from_pretrained("hf-internal-testing/tiny-random-reformer")
            ...
            >>> # add mask_token
            >>> tokenizer.add_special_tokens({"mask_token": "[MASK]"})  # doctest: +IGNORE_RESULT
            >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
            ...
            >>> # resize model's embedding matrix
            >>> model.resize_token_embeddings(new_num_tokens=model.config.vocab_size + 1)  # doctest: +IGNORE_RESULT
            ...
            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits
            ...
            >>> # retrieve index of [MASK]
            >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            ...
            >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            >>> predicted_token = tokenizer.decode(predicted_token_id)
            ```

            ```python
            >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
            >>> # mask labels of non-[MASK] tokens
            >>> labels = torch.where(
            ...     inputs.input_ids == tokenizer.mask_token_id, labels[:, : inputs["input_ids"].shape[-1]], -100
            ... )
            ...
            >>> outputs = model(**inputs, labels=labels)
            >>> loss = round(outputs.loss.item(), 2)
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            use_cache=False,  # no causal mask
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]
        logits = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + reformer_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )


class ReformerForSequenceClassification(ReformerPreTrainedModel):

    """
    ReformerForSequenceClassification
    ----------------------------------

    The `ReformerForSequenceClassification` class is a Python class that represents a sequence classification model
    based on the Reformer architecture. It inherits from the `ReformerPreTrainedModel` class.

    Summary
    -------
    The `ReformerForSequenceClassification` class provides a sequence classification model that can be used for
    tasks such as sentiment analysis, text classification, and natural language inference.

    Attributes:
        `num_labels` : int
            The number of labels for the sequence classification task.
        `config` : ReformerConfig
            The configuration object that holds all the hyperparameters of the model.
        `reformer` : ReformerModel
            The Reformer model used for encoding the input sequence.
        `classifier` : ReformerClassificationHead
            The classification head that takes the encoded sequence and produces the final logits.

    Methods:
        `construct`:
            Constructs the sequence classification model and performs forward propagation.

            Parameters:

            - `input_ids` : Optional[mindspore.Tensor] The input sequence tensor with shape
            `(batch_size, sequence_length)`.
            - `position_ids` : Optional[mindspore.Tensor] The position ids tensor with shape
            `(batch_size, sequence_length)` specifying the position of each token in the input sequence.
            - `attention_mask` : Optional[mindspore.Tensor] The attention mask tensor with shape
            `(batch_size, sequence_length)` indicating which tokens should be attended to and which ones should not.
            - `head_mask` : Optional[mindspore.Tensor] The head mask tensor with shape `(num_heads,)` or
            `(num_layers, num_heads)` indicating which heads or layers to mask during attention computation.
            - `inputs_embeds` : Optional[mindspore.Tensor] The embedded input sequence tensor with shape
            `(batch_size, sequence_length, embedding_size)`.
            - `num_hashes` : Optional[int] The number of hashes to use for the locality-sensitive hashing
            attention mechanism.
            - `labels` : Optional[mindspore.Tensor] The tensor of labels for computing the sequence
            classification/regression loss.
            - `output_hidden_states` : Optional[bool] Whether to output hidden states of the Reformer model.
            - `output_attentions` : Optional[bool] Whether to output attention weights of the Reformer model.
            - `return_dict` : Optional[bool] Whether to return the output as a dictionary or a tuple.

            Returns:

            - If `return_dict` is True, returns a `SequenceClassifierOutput` object containing the loss, logits,
            hidden states, and attention weights.
            - If `return_dict` is False, returns a tuple containing the loss, logits, and any additional hidden
            states or attention weights.

    Example:
        ```python
        >>> import mindspore as ms
        >>> from transformers import ReformerForSequenceClassification
        ...
        >>> model = ReformerForSequenceClassification(config)
        ...
        >>> inputs = {
        ...     'input_ids': ms.Tensor([[1, 2, 3, 4, 5]]),
        ...     'attention_mask': ms.Tensor([[1, 1, 1, 1, 1]])
        ... }
        ...
        >>> outputs = model.construct(**inputs)
        >>> logits = outputs.logits
        >>> predicted_class_id = logits.argmax(axis=1).item()
        >>> label = model.config.id2label[predicted_class_id]
        >>> ```
        >>> To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to the `ReformerForSequenceClassification` constructor:
        >>> ```python
        >>> num_labels = len(model.config.id2label)
        >>> model = ReformerForSequenceClassification(config, num_labels=num_labels)
        ...
        >>> inputs = {
        ...     'input_ids': ms.Tensor([[1, 2, 3, 4, 5]]),
        ...     'attention_mask': ms.Tensor([[1, 1, 1, 1, 1]]),
        ...     'labels': ms.Tensor([1])
        ... }
        ...
        >>> outputs = model.construct(**inputs)
        >>> loss = outputs.loss
        ```
    """
    def __init__(self, config):

        """
        Initializes the ReformerForSequenceClassification class.

        Args:
            self: The instance of the class.
            config: An instance of the configuration containing settings for the ReformerForSequenceClassification.
                It should contain the following attributes:

                - num_labels (int): The number of labels for sequence classification.
                - is_decoder (bool): If True, the ReformerForSequenceClassification is used as a decoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.reformer = ReformerModel(config)
        self.classifier = ReformerClassificationHead(config)
        if config.is_decoder is True:
            logger.warning("You might want to disable causal masking for sequence classification")

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        num_hashes: Optional[int] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple, SequenceClassifierOutput]

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoTokenizer, ReformerForSequenceClassification
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")
            >>> model = ReformerForSequenceClassification.from_pretrained("google/reformer-crime-and-punishment")
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            ...
            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits
            ...
            >>> predicted_class_id = logits.argmax().item()
            >>> label = model.config.id2label[predicted_class_id]
            ```

            ```python
            >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
            >>> num_labels = len(model.config.id2label)
            >>> model = ReformerForSequenceClassification.from_pretrained(
            ...     "google/reformer-crime-and-punishment", num_labels=num_labels
            ... )
            ...
            >>> labels = mindspore.tensor(1)
            >>> loss = model(**inputs, labels=labels).loss
            ```
            """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ReformerClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):

        """
        Initializes the ReformerClassificationHead class.

        Args:
            self (ReformerClassificationHead): The instance of the ReformerClassificationHead class.
            config (object): The configuration object containing the settings for the Reformer model.
                It should have attributes such as hidden_size (int), classifier_dropout (float, optional),
                hidden_dropout_prob (float), and num_labels (int). The config object is required and should not be None.

        Returns:
            None.

        Raises:
            ValueError: If the config parameter is None or if any of the required attributes in the
                config object are missing.
            TypeError: If the config parameter is not of the expected type or if any attribute in the
                config object has an unexpected type.
            RuntimeError: If there is an issue with the initialization of the Dense and Dropout layers.
        """
        super().__init__()
        self.dense = nn.Dense(2 * config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, hidden_states, **kwargs):

        """
        Constructs the classification head for the Reformer model.

        Args:
            self (ReformerClassificationHead): An instance of the ReformerClassificationHead class.
            hidden_states (torch.Tensor): The hidden states of the input sequence.
                It should have shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method does not return any value.

        Raises:
            None: No exceptions are raised by this method.
        """
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class ReformerForQuestionAnswering(ReformerPreTrainedModel):

    """
    This class represents a Reformer model for question answering tasks. It is a subclass of the ReformerPreTrainedModel.

    The ReformerForQuestionAnswering class implements the necessary methods to perform question answering on a
    given input sequence. It includes an initialization method, a construction method, and helper methods.

    The initialization method (__init__) takes a configuration object as input and initializes the Reformer model,
    as well as the necessary layers for question answering. The number of labels is also set in this method.

    The construction method (construct) takes various input tensors and parameters and performs question answering.
    It utilizes the Reformer model to obtain the sequence output, which is then passed through a dense layer for
    classification. The start and end logits are obtained from the output and are returned along with other relevant
    outputs.

    The construct method also allows optional inputs for start_positions and end_positions, which are used to compute
    the token classification loss. The loss is calculated using the cross-entropy loss function. If start_positions and
    end_positions are provided, the total loss is computed as the average of the start and end losses.

    The construct method returns the start and end logits, along with other relevant outputs, depending on the value of
    the return_dict parameter. If return_dict is False, a tuple of outputs is returned. If return_dict is True,
    a QuestionAnsweringModelOutput object is returned, containing the start_logits, end_logits, hidden_states,
    and attentions.

    Note:
        This class inherits from the ReformerPreTrainedModel class, which provides additional functionality for
        pre-training and fine-tuning the Reformer model.
    """
    def __init__(self, config):

        """
        Initializes a new instance of the ReformerForQuestionAnswering class.

        Args:
            self (ReformerForQuestionAnswering): The instance of the class.
            config:
                The configuration for the ReformerForQuestionAnswering model.

                - Type: Any
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None:
                This method does not return any value.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.reformer = ReformerModel(config)
        # 2 * config.hidden_size because we use reversible residual layers
        self.qa_outputs = nn.Dense(2 * config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        num_hashes: Optional[int] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        reformer_outputs = self.reformer(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            num_hashes=num_hashes,
            use_cache=False,  # no causal mask
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        sequence_output = reformer_outputs[0]

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

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + reformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=reformer_outputs.hidden_states,
            attentions=reformer_outputs.attentions,
        )

__all__ = [
    "REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
    "ReformerAttention",
    "ReformerForMaskedLM",
    "ReformerForQuestionAnswering",
    "ReformerForSequenceClassification",
    "ReformerLayer",
    "ReformerModel",
    "ReformerModelWithLMHead",
    "ReformerPreTrainedModel",
]
