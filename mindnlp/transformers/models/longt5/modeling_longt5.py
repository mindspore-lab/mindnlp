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
"""
MindSpore LongT5 model
"""

import copy
import math
from typing import List, Tuple

import numpy as np
import mindspore
from mindspore.common.initializer import initializer, Constant, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from mindnlp.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_longt5 import LongT5Config


logger = logging.get_logger(__name__)
####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
# TODO: Update before the merge
LongT5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/long-t5-local-base",
    "google/long-t5-local-large",
    "google/long-t5-tglobal-base",
    "google/long-t5-tglobal-large",
]


def _pad_to_multiple(x: mindspore.Tensor, block_len: int, dim: int, pad_value: int = 0) -> (mindspore.Tensor):
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return ops.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)
    return x

def _split_into_blocks(x: mindspore.Tensor, block_len: int, dim: int) -> mindspore.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return ops.zeros(output_shape, dtype=x.dtype)
    return x.reshape(output_shape)

def _concatenate_3_blocks(x: mindspore.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> (mindspore.Tensor):
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[mindspore.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return ops.cat(blocks_list, dim=sequence_dim)

def _make_3block_relative_position_ids(block_len: int) -> mindspore.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = ops.arange(3 * block_len, dtype=mindspore.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids

def _mask_local_attention_mask(local_attention_mask: mindspore.Tensor, block_len: int) -> mindspore.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = _make_3block_relative_position_ids(block_len)
    locality_mask = ops.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return ops.logical_and(local_attention_mask, locality_mask)

def _get_local_attention_mask(attention_mask: mindspore.Tensor, block_len: int) -> mindspore.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = ops.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)
    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1)


def _make_global_fixed_block_ids(
    attention_mask: mindspore.Tensor, global_block_size: int
) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.

    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.

    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.

    Padding tokens from the original sequence are represented by -1.
    """
    batch_size, seq_len = attention_mask.shape[:2]

    def handle_orphan_tokens(block_ids: mindspore.Tensor) -> mindspore.Tensor:
        block_ends = (ops.arange(seq_len) % global_block_size) == global_block_size - 1
        block_ends = block_ends.to(block_ids.device)
        true_block_ends = ops.logical_and(block_ends, block_ids >= 0)
        full_blocks = true_block_ends.sum(-1).unsqueeze(-1).type(block_ids.dtype) - 1
        block_ids = ops.where(block_ids < full_blocks, block_ids, full_blocks)
        return block_ids

    fixed_block_mask = ops.ones_like(attention_mask) / global_block_size
    fixed_block_mask = ops.cumsum(fixed_block_mask, dim=1) - fixed_block_mask
    mask = ops.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = ops.floor(mask + fixed_block_mask - 1.0).type(attention_mask.dtype)
    _global_block_ids_lower_bound = mindspore.tensor(-1, dtype=global_block_ids.dtype)
    global_block_ids = ops.where(
        global_block_ids > _global_block_ids_lower_bound, global_block_ids, _global_block_ids_lower_bound
    )
    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    global_block_ids = handle_orphan_tokens(global_block_ids)
    num_globals = seq_len // global_block_size
    # [batch_size, seq_len // global_block_size]
    if num_globals > 0:
        _sequence_block_ids_max = ops.max(global_block_ids, dim=-1).values.repeat(num_globals, 1).transpose(0, 1)
    else:
        _sequence_block_ids_max = ops.zeros(
            batch_size, dtype=global_block_ids.dtype
        )
    global_segment_ids = ops.cumsum(ops.ones(batch_size, num_globals), dim=-1) - 1
    global_segment_ids = global_segment_ids.to(attention_mask.device)
    global_segment_ids = ops.where(global_segment_ids <= _sequence_block_ids_max, 1, 0)
    return global_block_ids.type(mindspore.Tensor.int), global_segment_ids.type(mindspore.Tensor.int)


def _make_side_relative_position_ids(attention_mask: mindspore.Tensor, global_block_size: int) -> mindspore.Tensor:
    """Create the relative position tensor for local -> global attention."""
    block_ids, global_segment_ids = _make_global_fixed_block_ids(attention_mask, global_block_size)
    global_seq_len = global_segment_ids.shape[-1]
    global_positions = ops.arange(global_seq_len)
    side_relative_position = global_positions - block_ids[..., None]
    return side_relative_position.type(mindspore.int64)

def _create_global_aggregates(
    hidden_states: mindspore.Tensor, block_ids: mindspore.Tensor, global_seq_len: int
) -> mindspore.Tensor:
    """Compute individual block aggregates by summing over individual blocks."""
    # (batch..., seq_len, global_seq_len))
    block_ids = block_ids.where(
        block_ids >= 0, mindspore.tensor(global_seq_len, dtype=block_ids.dtype)
    )
    one_hot_block_ids = nn.functional.one_hot(block_ids.type(mindspore.int64), global_seq_len + 1)[:, :, :-1]
    return ops.einsum("...nd,...ng->...gd", hidden_states, one_hot_block_ids.type(hidden_states.dtype))


class LongT5LayerNorm(nn.Module):
    """LongT5LayerNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the LongT5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size, mindspore.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Constructs the LongT5LayerNorm for normalization of hidden states.
        
        Args:
            self (LongT5LayerNorm): An instance of the LongT5LayerNorm class.
            hidden_states (numpy.ndarray): A numpy array containing hidden states to be normalized.
                The array should have a dtype of mindspore.float32.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        variance = hidden_states.astype(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states / ops.sqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16]:
            hidden_states = hidden_states.astype(self.weight.dtype)

        return self.weight * hidden_states

ALL_LAYERNORM_LAYERS.append(LongT5LayerNorm)

class LongT5DenseActDense(nn.Module):
    """LongT5DenseActDense"""
    def __init__(self, config: LongT5Config):
        """
        This method initializes an instance of the LongT5DenseActDense class.
        
        Args:
            self: Represents the instance of the class.
            config (LongT5Config): An object of type LongT5Config containing configuration parameters for the
            dense layers. It specifies the dimensions of the input and output tensors, as well as the dropout
            rate and activation function to be used.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type LongT5Config.
            ValueError: If the config parameter contains invalid configuration values.
            RuntimeError: If there is an issue with initializing the dense layers, dropout, or activation function.
        """
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        """
        This method forwards and processes hidden states in the LongT5DenseActDense class.

        Args:
            self: An instance of the LongT5DenseActDense class, representing the current object.
            hidden_states: A tensor containing the hidden states to be processed.

        Returns:
            hidden_states: A tensor representing the processed hidden states.

        Raises:
            TypeError: If the weight datatype of self.wo is not matching with hidden_states.dtype or mindspore.int8.
        """
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.wo.weight.dtype not in (hidden_states.dtype, mindspore.int8):
            hidden_states = hidden_states.astype(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class LongT5DenseGatedActDense(nn.Module):
    """LongT5DenseGatedActDense"""
    def __init__(self, config: LongT5Config):
        """
        Initializes an instance of the LongT5DenseGatedActDense class.

        Args:
            self: The instance of the class.
            config (LongT5Config):
                An object containing configuration parameters for the dense layers.

                - config.d_model (int): The dimensionality of the model.
                - config.d_ff (int): The dimensionality of the feed-forward layer.
                - config.dropout_rate (float): The dropout rate for regularization.
                - config.dense_act_fn (str): The name of the activation function to be used.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        """
        Constructs the hidden states of the LongT5DenseGatedActDense model.

        Args:
            self (LongT5DenseGatedActDense): An instance of the LongT5DenseGatedActDense class.
            hidden_states (Tensor): The input hidden states.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class LongT5LayerFF(nn.Module):
    """LongT5LayerFF"""
    def __init__(self, config: LongT5Config):
        """
        Initializes the LongT5LayerFF class.

        Args:
            self (object): The instance of the LongT5LayerFF class.
            config (LongT5Config): An instance of LongT5Config containing configuration settings for the LongT5LayerFF.
                This parameter is used to configure the behavior of the LongT5LayerFF.
                It is expected to be an instance of the LongT5Config class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = LongT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = LongT5DenseActDense(config)

        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(self, hidden_states):
        """
        Method to forward the forward pass through the LongT5LayerFF feed-forward layer.

        Args:
            self (LongT5LayerFF): The instance of the LongT5LayerFF class.
            hidden_states (tensor): The input hidden states to be processed by the feed-forward layer.

        Returns:
            None: This method modifies the hidden_states in-place.

        Raises:
            TypeError: If the input hidden_states are not of type tensor.
            ValueError: If the input hidden_states are empty or have incompatible dimensions.
        """
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class LongT5Attention(nn.Module):
    """LongT5Attention"""
    def __init__(self, config: LongT5Config, has_relative_attention_bias=False):
        """
        Initializes an instance of the LongT5Attention class.

        Args:
            self: The instance of the LongT5Attention class.
            config (LongT5Config): An instance of LongT5Config containing configuration parameters
                for the attention mechanism.
            has_relative_attention_bias (bool): A boolean flag indicating whether relative attention bias is used.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'LongT5Attention' and is responsible for pruning the
        attention heads in the LongT5 model based on the provided 'heads'.

        Args:
            self (LongT5Attention): The instance of the LongT5Attention class.
            heads (List[int]): A list of integers representing the heads to be pruned from the attention mechanism.

        Returns:
            None: This method does not return any value explicitly but modifies the internal state of the
                LongT5Attention instance by pruning the specified attention heads.

        Raises:
            TypeError: If the 'heads' parameter is not a list of integers.
            ValueError: If the 'heads' list is empty, as there are no heads to prune.
            ValueError: If the number of heads to prune exceeds the total number of available heads in the
                LongT5Attention instance.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        This method calculates the relative position bucket for the LongT5Attention class.

        Args:
            relative_position (Tensor): The relative position value to calculate the bucket for.
            bidirectional (bool, optional): Whether the bucket calculation should be bidirectional. Default is True.
            num_buckets (int, optional): The total number of buckets to use for the calculation. Default is 32.
            max_distance (int, optional): The maximum distance value to consider for the calculation. Default is 128.

        Returns:
            Tensor: The calculated relative position bucket value.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the input parameters do not meet the specified restrictions.
            RuntimeError: If an unexpected error occurs during the calculation.
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = 0 - \
                ops.minimum(relative_position, ops.zeros(relative_position.shape)).astype(mindspore.int64)
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.astype(mindspore.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += ops.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = ops.arange(query_length, dtype=mindspore.int64)[:, None]
        memory_position = ops.arange(key_length, dtype=mindspore.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).expand_dims(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).swapaxes(1, 2)

        def unshape(states):
            """reshape"""
            return states.swapaxes(1, 2).view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = ops.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = ops.matmul(
            query_states, key_states.swapaxes(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.n_heads, real_seq_length, key_length), scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = ops.ones(position_bias.shape[1], mindspore.float32)
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = ops.softmax(scores.astype(mindspore.float32), dim=-1).astype(
            scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        if self.training:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout
            )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(ops.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class LongT5LocalAttention(nn.Module):
    """LongT5LocalAttention"""
    def __init__(self, config: LongT5Config, has_relative_attention_bias=False):
        """
        Initializes an instance of the LongT5LocalAttention class.

        Args:
            self: The instance of the class.
            config (LongT5Config): An object containing configuration parameters for the attention mechanism.
            has_relative_attention_bias (bool): A flag indicating whether relative attention bias is enabled.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.local_radius = config.local_radius     #
        self.block_len = self.local_radius + 1      #
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        This method computes the relative position bucket for a given relative position in the LongT5LocalAttention class.

        Args:
            relative_position (Tensor): A tensor representing the relative position.
            bidirectional (bool, optional): A boolean indicating whether the attention is bidirectional. Defaults to True.
            num_buckets (int, optional): An integer specifying the number of buckets. Defaults to 32.
            max_distance (int, optional): An integer representing the maximum distance. Defaults to 128.

        Returns:
            Tensor: A tensor representing the relative position bucket.

        Raises:
            TypeError: If the relative_position is not a tensor.
            ValueError: If the num_buckets or max_distance are non-positive integers.

        Note:
            - The relative_position should have a shape compatible with other tensors in the computation.
            - The num_buckets should be a positive integer.
            - The max_distance should be a positive integer greater than num_buckets.
            - The bidirectional flag determines whether the attention is computed bidirectionally or unidirectionally.

        Example:
            ```python
            >>> relative_position = tensor([1, -2, 3, -4])
            >>> bucket = LongT5LocalAttention._relative_position_bucket(relative_position)
            ```
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = 0 - \
                ops.minimum(relative_position, ops.zeros(relative_position.shape)).astype(mindspore.int64)
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.astype(mindspore.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.fill(relative_position_if_large.dtype, \
                                                 relative_position_if_large.shape, num_buckets - 1)
        )
        # relative_buckets += ops.where(is_small, relative_position\
        # , relative_position_if_large) # mindspore 2.0
        relative_buckets += ops.select(is_small.astype(mindspore.bool_), \
                                relative_position, relative_position_if_large) # mindspore 1.10
        return relative_buckets

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        memory_position = ops.arange(3 * block_length, dtype=mindspore.int64)
        context_position = memory_position[block_length:-block_length]
        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.transpose([2, 0, 1]).expand_dims(0).expand_dims(0)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        '''
        Constructs the local attention mechanism for the LongT5 model.

        Args:
            self (LongT5LocalAttention): An instance of the LongT5LocalAttention class.
            hidden_states (Tensor): The input hidden states tensor of shape (batch_size, seq_length, hidden_dim).
            mask (Tensor, optional): The attention mask tensor of shape (batch_size, seq_length). Defaults to None.
            position_bias (Tensor, optional): The position bias tensor of shape (1, 1, n_heads, block_len, 3 * block_len).
                Defaults to None.
            layer_head_mask (Tensor, optional):
                The layer head mask tensor of shape (batch_size, n_heads, seq_length, seq_length). Defaults to None.
            output_attentions (bool, optional): Flag to output attention weights. Defaults to False.

        Returns:
            Tuple:
                A tuple containing the following elements:

                - attn_output (Tensor): The output tensor of shape (batch_size, seq_length, hidden_dim).
                - present_key_value_state (None): Placeholder for future use.
                - position_bias (Tensor): The position bias tensor of shape (1, 1, n_heads, block_len, 3 * block_len).
                - attn_weights (Tensor, optional): The attention weights tensor of shape
                (batch_size, n_heads, seq_length, seq_length), returned only if output_attentions is set to True.

        Raises:
            None.
        '''
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        def unshape(states):
            """reshape"""
            return states.view(batch_size, -1, self.inner_dim)

        # get query/key/value states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # compute scores
        scores = ops.einsum(
            "...qhd,...khd->...hqk", query_states, key_states
        )  # (batch_size, num_block, n_heads, block_len, 3 * block_len)

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, 1, self.n_heads, self.block_len, 3 * self.block_len), scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(self.block_len)

            if mask is not None:
                # Replace masked positions with -1e10 (according to the original implementation)
                mask = ops.where(mask > 0, 0.0, -1e10)
                # We need to adjust position bias shape to be sum with mask
                position_bias = position_bias + mask.transpose(1, 2)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = ops.softmax(scores.astype(mindspore.float32), dim=-1).astype(
            scores.dtype
        )
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        if self.training:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout
            )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_weights = attn_weights.type(value_states.dtype)    # 存疑
        attn_output = unshape(ops.einsum("...hqk,...khd->...qhd", attn_weights, value_states))   # (batch_size, seq_length, dim)
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

class LongT5TransientGlobalAttention(nn.Module):
    """LongT5TransientGlobalAttention"""
    def __init__(self, config: LongT5Config, has_relative_attention_bias=False):
        """
        Initializes an instance of the LongT5TransientGlobalAttention class.

        Args:
            self: The instance of the class.
            config (LongT5Config): An object of the LongT5Config class containing configuration parameters.
            has_relative_attention_bias (bool, optional): Specifies whether relative attention bias is present.
                Default is False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.local_radius = config.local_radius     # new
        self.block_len = self.local_radius + 1      # new
        self.global_block_size = config.global_block_size   # new
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

        # Relativen attention bias & Layer norm for global attention
        if self.has_relative_attention_bias:
            self.global_relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.global_input_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Method to calculate the relative position bucket for LongT5TransientGlobalAttention.

        Args:
            relative_position (Tensor): The relative position value to calculate the bucket for.
            bidirectional (bool, optional): Flag indicating if the attention is bidirectional. Default is True.
            num_buckets (int, optional): Number of buckets to use for bucketing the relative positions. Default is 32.
            max_distance (int, optional): Maximum distance for bucketing. Default is 128.

        Returns:
            None: This method does not return any value explicitly, but updates the relative_buckets variable.

        Raises:
            ValueError: If the relative_position is not a valid tensor.
            TypeError: If the bidirectional flag is not a boolean.
            ValueError: If the num_buckets is not a positive integer.
            ValueError: If the max_distance is not a positive integer.
            ValueError: If the relative_position is out of range when calculating the bucket.
            ValueError: If an error occurs during the bucketing calculation process.
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).astype(mindspore.int64) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_position = 0 - \
                ops.minimum(relative_position, ops.zeros(relative_position.shape)).astype(mindspore.int64)
        # now relative_position is in the range [0, inf)
        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            ops.log(relative_position.astype(mindspore.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mindspore.int64)
        relative_position_if_large = ops.minimum(
            relative_position_if_large, ops.fill(relative_position_if_large.dtype, \
                                                 relative_position_if_large.shape, num_buckets - 1)
        )
        # relative_buckets += ops.where(is_small, relative_position\
        # , relative_position_if_large) # mindspore 2.0
        relative_buckets += ops.select(is_small.astype(mindspore.bool_), \
                                relative_position, relative_position_if_large) # mindspore 1.10
        return relative_buckets

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        memory_position = ops.arange(3 * block_length, dtype=mindspore.int64)
        context_position = memory_position[block_length:-block_length]
        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.transpose([2, 0, 1]).expand_dims(0).expand_dims(0)
        return values

    def compute_side_bias(self, mask: mindspore.Tensor, global_segment_ids: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method computes the side bias for attention calculation in the LongT5TransientGlobalAttention class.

        Args:
            self (LongT5TransientGlobalAttention): The instance of the LongT5TransientGlobalAttention class.
            mask (mindspore.Tensor): A tensor representing the mask used in attention calculation.
            global_segment_ids (mindspore.Tensor): A tensor containing global segment ids for attention calculation.

        Returns:
            mindspore.Tensor: A tensor representing the computed attention side bias.

        Raises:
            ValueError: If the input tensors are not of the expected shape or type.
            RuntimeError: If there is an issue during the computation process.
        """
        # (batch_size, 1, seq_len, global_seq_len)
        side_attention_mask = ops.equal(mask[..., None], global_segment_ids[:, None, :])[:, None, ...]
        attention_side_bias = ops.where(side_attention_mask > 0, 0.0, -1e10)
        # (batch_size, seq_len, global_seq_len)
        side_relative_position = _make_side_relative_position_ids(mask, self.global_block_size)
        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (batch_size, seq_len, global_seq_len, num_heads)
        side_bias = self.global_relative_attention_bias(side_relative_position_bucket)

        # (batch_size, num_heads, seq_len, global_seq_len)
        side_bias = side_bias.permute([0, 3, 1, 2])
        # (batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias
        return attention_side_bias

    def forward(
        self,
        hidden_states,
        mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        This method forwards the transient global attention mechanism for the LongT5 model.

        Args:
            self: The instance of the LongT5TransientGlobalAttention class.
            hidden_states (Tensor): The input hidden states with shape (batch_size, seq_length, hidden_size).
            mask (Tensor, optional): An optional mask tensor with shape (batch_size, seq_length) to
                mask the attention scores.
            position_bias (Tensor, optional): An optional position bias tensor with shape
                (1, 1, n_heads, block_len, 3 * block_len).
            layer_head_mask (Tensor, optional): An optional mask tensor with shape (n_heads, block_len, block_len)
                to mask specific heads and blocks.
            output_attentions (bool): A boolean flag indicating whether to include attention weights in the output.

        Returns:
            None: This method does not return any value, it updates internal states and variables.

        Raises:
            ValueError: If the shape of input tensors does not match the expected shapes.
            RuntimeError: If there is a runtime error during the computation.
            TypeError: If the input arguments are not of the expected types.
            AssertionError: If the input assertions fail during the computation.
        """
        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim)

        def unshape(states):
            """reshape"""
            return states.view(batch_size, -1, self.inner_dim)

        # Prepare components for transient-global attention
        # Obtain block_ids and global_segment_ids
        # global_seq_len := seq_len // self.global_block_size
        # shapes: (batch_size, seq_len) & (batch_size, global_seq_len)
        block_ids, global_segment_ids = _make_global_fixed_block_ids(
            mask if mask is not None else ops.ones(hidden_states.shape[:-1]),
            self.global_block_size,
        )

        # Create global inputs
        _global_seq_len = global_segment_ids.shape[-1]
        global_inputs = _create_global_aggregates(hidden_states, block_ids, _global_seq_len)
        global_inputs = self.global_input_layer_norm(global_inputs)

        # get query/key/value states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        # Get global/side key/value states  shape: (batch_size, global_seq_len, n_heads, dim_per_head)
        side_key_states = shape(self.k(global_inputs))
        side_value_states = shape(self.v(global_inputs))

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Tile side inputs across local key/value blocks
        # New shape: (batch_size, num_blocks, global_seq_len, n_heads, dim_per_head)
        reps = [1] * (side_key_states.ndim + 1)
        reps[1] = key_states.shape[1]
        side_key_states = side_key_states.unsqueeze(1).repeat(reps)
        side_value_states = side_value_states.unsqueeze(1).repeat(reps)

        # Concatenate "local" and "side"/"global" key/value states to allow each token to attend global aggregated ones
        # New shape: (batch_size, num_blocks, 3 * block_len + global_seq_len, n_heads, dim_per_head)
        key_states = ops.cat([key_states, side_key_states], dim=2)
        value_states = ops.cat([value_states, side_value_states], dim=2)

        # Compute scores -> (batch_size, num_block, n_heads, block_len, 3 * block_len + global_seq_len)
        scores = ops.einsum(
            "...qhd,...khd->...hqk", query_states, key_states
        )

        if mask is not None:
            # We need to adjust position bias shape to be sum with mask
            local_attention_mask = _get_local_attention_mask(mask, self.block_len)
            # Replace masked positions with -10_000 (according to the original implementation)
            local_attention_mask = ops.where(local_attention_mask > 0, 0.0, -1e10)
        else:
            local_attention_mask = None

        if position_bias is None:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, 1, self.n_heads, self.block_len, 3 * self.block_len), scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(self.block_len)

            if local_attention_mask is not None:
                # (batch_size, 1, n_heads, block_len, 3 * block_len)
                position_bias = position_bias + local_attention_mask.transpose(1, 2)
            position_bias = position_bias.type(scores.dtype)

            # Calculate global/side bias - shape: # (batch_size, num_heads, seq_len, global_seq_len)
            if mask is None:
                mask = ops.ones(batch_size, seq_length)
            # (batch_size, num_heads, seq_len, global_seq_len)
            side_position_bias = self.compute_side_bias(mask, global_segment_ids)
            # (batch_size, num_blocks, num_heads, block_len, global_seq_len)
            side_position_bias = _split_into_blocks(side_position_bias, self.block_len, dim=-2).transpose(1, 2)
            side_position_bias = side_position_bias.type(scores.dtype).to(scores.device)
            # (batch_size, num_blocks, num_heads, block_len, 3 * block_len + global_seq_len)
            position_bias = ops.cat([position_bias, side_position_bias], dim=-1)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = ops.softmax(scores.astype(mindspore.float32), dim=-1).astype(
            scores.dtype
        )
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        if self.training:
            attn_weights = F.dropout(
                attn_weights, p=self.dropout
            )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        attn_weights = attn_weights.type(value_states.dtype)    # 存疑
        attn_output = unshape(ops.einsum("...hqk,...khd->...qhd", attn_weights, value_states))   # (batch_size, seq_length, dim)
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)

        present_key_value_state = None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class LongT5LayerSelfAttention(nn.Module):
    """LongT5LayerSelfAttention"""
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initializes a LongT5LayerSelfAttention object.

        Args:
            self: The object itself.
            config (object): An instance of configuration for the LongT5LayerSelfAttention.
            has_relative_attention_bias (bool, optional): Indicates whether relative attention bias is applied.
                Default is False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.SelfAttention = LongT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Method 'forward' in the class 'LongT5LayerSelfAttention'.

        This method forwards the output hidden states by applying self-attention mechanism.

        Args:
            self: Instance of the class.
            hidden_states (Tensor): Input hidden states.
            attention_mask (Tensor, optional): Mask for attention scores, default is None.
            position_bias (Tensor, optional): Bias for relative position encoding, default is None.
            layer_head_mask (Tensor, optional): Mask for specific layers and heads, default is None.
            past_key_value (Tuple, optional): Tuple containing past key and value tensors, default is None.
            use_cache (bool, optional): Flag to use cache for faster decoding, default is False.
            output_attentions (bool, optional): Flag to output attention scores, default is False.

        Returns:
            Tuple: A tuple containing updated hidden states and attention outputs.

        Raises:
            ValueError: If any of the input tensors have incompatible shapes.
            TypeError: If any input parameter is not of the expected type.
            RuntimeError: If cache is not initialized properly or if there is an issue with the attention mechanism.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class LongT5LayerLocalSelfAttention(nn.Module):
    """LongT5LayerSelfAttention"""
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the attention mechanism.
            has_relative_attention_bias (bool, optional): A flag indicating whether the attention mechanism 
                has relative attention bias. Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.LocalSelfAttention = LongT5LocalAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        This method forwards the LongT5LayerLocalSelfAttention and performs the local self-attention operation.

        Args:
            self: The instance of the LongT5LayerLocalSelfAttention class.
            hidden_states (tensor): The input hidden states. It is of type tensor and represents the input sequence
                of hidden states.
            attention_mask (tensor, optional): An optional mask tensor. It is of type tensor and is used to mask the
                attention scores. Default is None.
            position_bias (tensor, optional): An optional tensor for positional bias.
                It is of type tensor and provides positional information to the attention mechanism. Default is None.
            layer_head_mask (tensor, optional): An optional mask tensor.
                It is of type tensor and is applied to the attention scores for specific layers and heads.
                Default is None.
            output_attentions (bool, optional): A flag to indicate whether to output attentions.
                It is of type bool and determines whether to include attention outputs in the return value.
                Default is False.

        Returns:
            tuple:
                A tuple containing the following elements:

                - hidden_states (tensor): The updated hidden states after the local self-attention operation.
                - additional_outputs (tuple): Additional outputs including attention scores if 'output_attentions' is True.

        Raises:
            None
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.LocalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class LongT5LayerTransientGlobalSelfAttention(nn.Module):
    """LongT5LayerSelfAttention"""
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initializes the LongT5LayerTransientGlobalSelfAttention instance.

        Args:
            self: The instance itself.
            config: An object containing configuration settings for the LongT5LayerTransientGlobalSelfAttention.
            has_relative_attention_bias (bool, optional): Specifies whether the attention has relative bias.
                Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.TransientGlobalSelfAttention = LongT5TransientGlobalAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    ):
        """
        Method 'forward' in the class 'LongT5LayerTransientGlobalSelfAttention'.
        This method forwards the output of the layer by applying transient global self-attention mechanism.

        Args:
            self: Reference to the instance of the class.
            hidden_states (tensor): The input hidden states to be processed.
            attention_mask (tensor, optional): Masking tensor indicating which positions should be attended to.
            position_bias (tensor, optional): Tensor providing positional biases for the attention mechanism.
            layer_head_mask (tensor, optional): Masking tensor for individual attention heads within the layer.
            output_attentions (bool, optional): Flag to indicate whether to output attention scores.

        Returns:
            tuple:
                A tuple containing the following elements:

                - hidden_states (tensor): The updated hidden states after applying attention mechanism.
                - additional_outputs (tuple): Any additional outputs returned by the attention mechanism.

        Raises:
            None
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.TransientGlobalSelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class LongT5LayerCrossAttention(nn.Module):
    """LongT5LayerCrossAttention"""
    def __init__(self, config):
        """
        Initialize the LongT5LayerCrossAttention class.

        Args:
            self: An instance of the LongT5LayerCrossAttention class.
            config:
                A dictionary containing configuration settings for the LongT5LayerCrossAttention.

                - Type: dict
                - Purpose: Contains the configuration settings for the LongT5LayerCrossAttention.
                - Restrictions: Must be a valid dictionary with required configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.EncDecAttention = LongT5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(p=config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        """
        Constructs the cross-attention layer for the LongT5 model.

        Args:
            self (LongT5LayerCrossAttention): An instance of the LongT5LayerCrossAttention class.
            hidden_states (torch.Tensor): The input hidden states of the layer.
                Shape: (batch_size, sequence_length, hidden_size).
            key_value_states (torch.Tensor): The key-value states for attention.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional): The attention mask tensor.
                Shape: (batch_size, sequence_length).
            position_bias (torch.Tensor, optional): The position bias tensor.
                Shape: (batch_size, num_heads, sequence_length, sequence_length).
            layer_head_mask (torch.Tensor, optional): The layer head mask tensor.
                Shape: (batch_size, num_heads, sequence_length, sequence_length).
            past_key_value (tuple, optional): The past key-value states for attention.
                Tuple containing two tensors: (past_key_states, past_value_states).
            use_cache (bool, optional): Whether to use cache for the attention outputs.
            query_length (int, optional): The length of the query.
            output_attentions (bool, optional): Whether to output the attention outputs.

        Returns:
            tuple:
                A tuple containing the following elements:

                - layer_output (torch.Tensor): The output hidden states of the layer.
                Shape: (batch_size, sequence_length, hidden_size).
                - attention_probs (torch.Tensor, optional): The attention probabilities.
                Shape: (batch_size, num_heads, sequence_length, sequence_length).
                This is only returned when output_attentions=True.
                - cross_attentions (torch.Tensor, optional): The cross-attention probabilities.
                Shape: (batch_size, num_heads, sequence_length, sequence_length).
                This is only returned when output_attentions=True.

        Raises:
            None.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class LongT5Block(nn.Module):
    """LongT5Block"""
    def __init__(self, config, has_relative_attention_bias=False):
        """
        Initialize the LongT5Block.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the settings for the LongT5Block.
            has_relative_attention_bias (bool): A boolean indicating whether the attention mechanism
                has relative attention bias.

        Returns:
            None.

        Raises:
            ValueError: If the configuration for the encoder attention mechanism is invalid, a ValueError is raised.
        """
        super().__init__()
        self.is_decoder = config.is_decoder

        if config.is_decoder:
            attention_layer = LongT5LayerSelfAttention
        elif config.encoder_attention_type == "local":
            attention_layer = LongT5LayerLocalSelfAttention
        elif config.encoder_attention_type == "transient-global":
            attention_layer = LongT5LayerTransientGlobalSelfAttention
        else:
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {config.encoder_attention_type}."
            )

        self.layer = nn.ModuleList()
        self.layer.append(attention_layer(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(LongT5LayerCrossAttention(config))

        self.layer.append(LongT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        # return_dict=True,
    ):
        """
        Constructs a LongT5Block layer.

        Args:
            self: The object instance.
            hidden_states (Tensor): The input hidden states for the layer.
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding tokens.
            position_bias (Tensor, optional): Bias for relative position encoding.
            encoder_hidden_states (Tensor, optional): Hidden states from the encoder for cross-attention.
            encoder_attention_mask (Tensor, optional): Mask for encoder attention.
            encoder_decoder_position_bias (Tensor, optional): Bias for cross-attention position encoding.
            layer_head_mask (Tensor, optional): Mask for specific attention heads in the layer.
            cross_attn_layer_head_mask (Tensor, optional): Mask for specific attention heads in cross-attention.
            past_key_value (Tuple, optional): Tuple containing past key and value states for caching.
            use_cache (bool, optional): Flag to indicate whether to use caching.
            output_attentions (bool, optional): Flag to indicate whether to output attentions.

        Returns:
            tuple:
                Tuple of output tensors including the updated hidden states and additional information
                based on the input parameters.

        Raises:
            ValueError: If the number of past key values does not match the expected number.
            Warning: If past_key_values is passed to the encoder when not intended.
            TypeError: If the input tensors have incompatible data types.
            RuntimeError: If there are issues during the computation process.
        """
        if past_key_value is not None:
            if not self.is_decoder:
                logging.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 inference - check https://github.com/huggingface/transformers/pull/19229/
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 inference - check https://github.com/huggingface/transformers/pull/19229/
            if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
                clamp_value = mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
                hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 inference - check https://github.com/huggingface/transformers/pull/19229/
        if hidden_states.dtype == mindspore.float16 and ops.isinf(hidden_states).any():
            clamp_value = mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(hidden_states.dtype)).max) - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs
        # hidden-states, present_key_value_states, (self-attention position bias),
        # (self-attention weights), (cross-attention position bias),(cross-attention weights)


class LongT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongT5Config
    base_model_prefix = "transformer"

    supports_gradient_checkpointing = True
    _no_split_modules = ["LongT5Block"]

    @property
    def dummy_inputs(self):
        """
        This method generates dummy inputs for the LongT5PreTrainedModel class.

        Args:
            self: An instance of the LongT5PreTrainedModel class.

        Returns:
            None

        Raises:
            None
        """
        input_ids = mindspore.tensor(DUMMY_INPUTS)
        input_mask = mindspore.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, cell):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(cell, LongT5LayerNorm):
            cell.weight.assign_value(initializer(Constant(factor * 1.0), cell.weight.shape, cell.weight.dtype))
        elif isinstance(cell, (LongT5Model, LongT5ForConditionalGeneration, LongT5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            cell.shared.weight.assign_value(initializer(Normal(factor * 1.0),
                                                    cell.shared.weight.shape, cell.shared.weight.dtype))
        elif isinstance(cell, LongT5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            cell.wi.weight.assign_value(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                cell.wi.weight.shape, cell.wi.weight.dtype))
            if hasattr(cell.wi, "bias") and cell.wi.bias is not None:
                cell.wi.bias.assign_value(initializer('zeros', cell.wi.bias.shape, cell.wi.bias.dtype))
            cell.wo.weight.assign_value(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)),
                                                cell.wo.weight.shape, cell.wo.weight.dtype))
            if hasattr(cell.wo, "bias") and cell.wo.bias is not None:
                cell.wo.bias.assign_value(initializer('zeros', cell.wo.bias.shape, cell.wo.bias.dtype))
        elif isinstance(cell, LongT5DenseGatedActDense):
            cell.wi_0.weight.assign_value(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                  cell.wi_0.weight.shape, cell.wi_0.weight.dtype))
            if hasattr(cell.wi_0, "bias") and cell.wi_0.bias is not None:
                cell.wi_0.bias.assign_value(initializer('zeros', cell.wi_0.bias.shape, cell.wi_0.bias.dtype))
            cell.wi_1.weight.assign_value(initializer(Normal(factor * ((self.config.d_model) ** -0.5)),
                                                  cell.wi_1.weight.shape, cell.wi_1.weight.dtype))
            if hasattr(cell.wi_1, "bias") and cell.wi_1.bias is not None:
                cell.wi_1.bias.assign_value(initializer('zeros', cell.wi_1.bias.shape, cell.wi_1.bias.dtype))
            cell.wo.weight.assign_value(initializer(Normal(factor * ((self.config.d_ff) ** -0.5)),
                                                cell.wo.weight.shape, cell.wo.weight.dtype))
            if hasattr(cell.wo, "bias") and cell.wo.bias is not None:
                cell.wo.bias.assign_value(initializer('zeros', cell.wo.bias.shape, cell.wo.bias.dtype))

        elif isinstance(cell, (LongT5Attention, LongT5LocalAttention, LongT5TransientGlobalAttention)):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads

            cell.q.weight.assign_value(initializer(Normal(factor * ((d_model * key_value_proj_dim) ** -0.5)),
                                               cell.q.weight.shape, cell.q.weight.dtype))
            cell.k.weight.assign_value(initializer(Normal(factor * (d_model ** -0.5)),
                                               cell.k.weight.shape, cell.k.weight.dtype))
            cell.v.weight.assign_value(initializer(Normal(factor * (d_model ** -0.5)),
                                               cell.v.weight.shape, cell.v.weight.dtype))
            cell.o.weight.assign_value(initializer(Normal(factor * ((n_heads * key_value_proj_dim) ** -0.5)),
                                               cell.o.weight.shape, cell.o.weight.dtype))
            if cell.has_relative_attention_bias:
                cell.relative_attention_bias.weight.assign_value(initializer(Normal(factor * (d_model**-0.5)),
                                                    cell.relative_attention_bias.weight.shape, cell.relative_attention_bias.weight.dtype))
                if isinstance(cell, LongT5TransientGlobalAttention):
                    cell.global_relative_attention_bias.weight.assign_value(initializer(Normal(factor * (d_model ** -0.5)),
                                                                             cell.global_relative_attention_bias.weight.shape,
                                                                             cell.global_relative_attention_bias.weight.dtype))

    def _shift_right(self, input_ids):
        """
        Shifts the input_ids to the right by one position and fills the shifted position with the decoder_start_token_id.

        Args:
            self (LongT5PreTrainedModel): The instance of the LongT5PreTrainedModel class.
            input_ids (torch.Tensor): The input tensor containing token ids to be shifted to the right.

        Returns:
            torch.Tensor: The shifted input_ids tensor with the first position filled with the decoder_start_token_id.

        Raises:
            ValueError: If self.model.config.decoder_start_token_id is not defined
                or if self.model.config.pad_token_id is not defined.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In LongT5 it is usually set to the pad_token_id. "
                "See LongT5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].copy()
        shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class LongT5Stack(LongT5PreTrainedModel):
    """LongT5Stack"""
    def __init__(self, config, embed_tokens=None):
        """
        Initializes an instance of the LongT5Stack class.

        Args:
            self (LongT5Stack): An instance of the LongT5Stack class.
            config: A configuration object containing various parameters for the LongT5Stack.
            embed_tokens: An optional nn.Embedding object representing the embedding tokens. Defaults to None.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes the LongT5Stack instance by setting various attributes and creating
            the necessary layers. It takes in the following parameters:

            - self: The instance of the LongT5Stack class itself.
            - config: A configuration object which contains the parameters for the LongT5Stack.
            - embed_tokens: An optional nn.Embedding object that represents the embedding tokens.
            If provided, the weight of the embed_tokens will be set to the weight of the provided object.

        The method performs the following steps:

        1. Calls the __init__ method of the super class to initialize the parent class.
        2. Sets the embed_tokens attribute to an nn.Embedding object with the specified vocabulary size and d_model.
        3. If embed_tokens is not None, it sets the weight of self.embed_tokens to the weight of the provided embed_tokens.
        4. Sets the is_decoder attribute to the value of config.is_decoder.
        5. Sets the local_radius attribute to the value of config.local_radius.
        6. Sets the block_len attribute to the local_radius + 1.
        7. Creates a block attribute as an nn.ModuleList containing LongT5Block objects. The number of blocks is
        determined by config.num_layers. Each block is initialized with a relative_attention_bias if it is the
        first block in the list.
        8. Sets the final_layer_norm attribute to a LongT5LayerNorm object with the specified d_model and layer_norm_epsilon.
        9. Sets the dropout attribute to an nn.Dropout object with the specified dropout_rate.
        10. Sets the gradient_checkpointing attribute to False.
        11. Calls the post_init method.

        Note:
            The LongT5Stack class is part of the LongT5 model and is responsible for stacking multiple LongT5Blocks
            to form the complete LongT5 model.
        """
        super().__init__(config)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight
        self.is_decoder = config.is_decoder

        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1

        self.block = nn.ModuleList(
            [LongT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
            This method retrieves the input embeddings from the LongT5Stack class.

        Args:
            self: The instance of the LongT5Stack class. It is used to access the embed_tokens attribute.

        Returns:
            The embed_tokens attribute: which represents the input embeddings.

        Raises:
            None
        """
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the LongT5Stack class.

        Args:
            self (LongT5Stack): The instance of the LongT5Stack class.
            new_embeddings (Any): The new embeddings to be set for the input tokens. It can be any object type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        '''
        This method forwards the LongT5Stack model. It takes 13 parameters:

        Args:
            self (object): The instance of the class.
            input_ids (Tensor, optional): The input tensor of token indices. Default is None.
            attention_mask (Tensor, optional): The attention mask tensor. Default is None.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder. Default is None.
            encoder_attention_mask (Tensor, optional): The attention mask for the encoder. Default is None.
            inputs_embeds (Tensor, optional): The embedded input tensor. Default is None.
            head_mask (Tensor, optional): The head mask tensor. Default is None.
            cross_attn_head_mask (Tensor, optional): The cross-attention head mask tensor. Default is None.
            past_key_values (list, optional): The list of past key values. Default is None.
            use_cache (bool, optional): Flag indicating whether to use cache. Default is None.
            output_attentions (bool, optional): Flag indicating whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Flag indicating whether to output hidden states. Default is None.
            return_dict (bool, optional): Flag indicating whether to return a dictionary. Default is None.

        Returns:
            None.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously,
                or if neither input_ids nor inputs_embeds are specified.
            AssertionError: If the model is used as a decoder and use_cache is set to True,
                or if the model is used as a decoder and encoder_attention_mask is not specified
                while encoder_hidden_states is provided.
        '''
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids.astype(mindspore.int64))

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, mask_seq_length), mindspore.float32)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = ops.ones(
                (batch_size, encoder_seq_length), mindspore.int64
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape, inputs_embeds.device
            )
        elif self.config.encoder_attention_type == "local":
            extended_attention_mask = _get_local_attention_mask(attention_mask, self.block_len)
        else:  # we need to use both local attention mask and standard extended mask for transient-global attention
            extended_attention_mask = attention_mask

        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), \
            # (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class LongT5Model(LongT5PreTrainedModel):
    """LongT5Model"""
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: LongT5Config):
        """
        Initializes a LongT5Model instance.

        Args:
            self: The instance of the LongT5Model class.
            config (LongT5Config): An instance of LongT5Config containing the configuration parameters for the model.
                It specifies the model's architecture, including vocab size and model dimension.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LongT5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = LongT5Stack(decoder_config)

    def get_input_embeddings(self):
        """
        Method to retrieve input embeddings in the LongT5Model class.

        Args:
            self: The instance of the LongT5Model class.

        Returns:
            The shared input embeddings used in the LongT5Model.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the LongT5Model.

        Args:
            self (LongT5Model): The instance of the LongT5Model class.
            new_embeddings: The new embeddings to be set for the input.
                It should be a tensor representing the embeddings.
                The shape of the tensor should match the expected input shape of the model.

        Returns:
            None.

        Raises:
            None.

        """
        self.shared = new_embeddings
        # self.encoder.set_input_embeddings(new_embeddings)
        # self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        Tie the weights of the encoder and decoder word embeddings if specified in the configuration.

        Args:
            self (LongT5Model): The instance of the LongT5Model class.

        Returns:
            None.

        Raises:
            None
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        """get encoder"""
        return self.encoder

    def get_decoder(self):
        """get decoder"""
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        This method forwards a LongT5 model with the specified parameters.

        Args:
            self (object): The instance of the class.
            input_ids (list): The input token IDs for the encoder.
            attention_mask (list): The attention mask for the encoder input.
            decoder_input_ids (list): The input token IDs for the decoder.
            decoder_attention_mask (list): The attention mask for the decoder input.
            head_mask (list): The mask applied to the encoder's attention heads.
            decoder_head_mask (list): The mask applied to the decoder's attention heads.
            cross_attn_head_mask (list): The mask applied to the cross-attention heads.
            encoder_outputs (object): The output of the encoder.
            past_key_values (object): The past key values for the decoder.
            inputs_embeds (object): The embeddings for the encoder inputs.
            decoder_inputs_embeds (object): The embeddings for the decoder inputs.
            use_cache (bool): Flag indicating whether to use cache.
            output_attentions (bool): Flag indicating whether to output attentions.
            output_hidden_states (bool): Flag indicating whether to output hidden states.
            return_dict (bool): Flag indicating whether to return a dictionary.

        Returns:
            None

        Raises:
            None
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class LongT5ForConditionalGeneration(LongT5PreTrainedModel):
    """LongT5ForConditionalGeneration"""
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: LongT5Config):
        """
        Args:
            self: The instance of the LongT5ForConditionalGeneration class.
            config (LongT5Config): An instance of LongT5Config class containing the configuration parameters
                for the LongT5 model. It specifies the model dimensions, vocabulary size, and other relevant settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LongT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = LongT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the LongT5ForConditionalGeneration model.

        Args:
            self:
                An instance of the LongT5ForConditionalGeneration class.

                - Type: LongT5ForConditionalGeneration
                - Purpose: Represents the current instance of the LongT5ForConditionalGeneration class.
                - Restrictions: None

        Returns:
            None: The method returns None as it retrieves the input embeddings from the model.

        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """set input embeddings"""
        self.shared = new_embeddings
        # self.encoder.set_input_embeddings(new_embeddings)
        # self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        This method ties the weights of the encoder and decoder embeddings if the configuration specifies
        to tie the word embeddings.

        Args:
            self (LongT5ForConditionalGeneration): The instance of the LongT5ForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        """set output embeddings"""
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        """get output embeddings"""
        return self.lm_head

    def get_encoder(self):
        """get encoder"""
        return self.encoder

    def get_decoder(self):
        """get decoder"""
        return self.decoder

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        This method forwards a LongT5 model for conditional generation.

        Args:
            self: The instance of the class.
            input_ids (torch.Tensor, optional): The input token IDs for the encoder. Default is None.
            attention_mask (torch.Tensor, optional): The attention mask for the encoder input. Default is None.
            decoder_input_ids (torch.Tensor, optional): The input token IDs for the decoder. Default is None.
            decoder_attention_mask (torch.Tensor, optional): The attention mask for the decoder input. Default is None.
            head_mask (torch.Tensor, optional): The head mask for the encoder. Default is None.
            decoder_head_mask (torch.Tensor, optional): The head mask for the decoder. Default is None.
            cross_attn_head_mask (torch.Tensor, optional): The cross-attention head mask. Default is None.
            encoder_outputs (torch.Tensor, optional): The encoder outputs. Default is None.
            past_key_values (torch.Tensor, optional): The past key values for the decoder. Default is None.
            inputs_embeds (torch.Tensor, optional): The input embeddings for the encoder. Default is None.
            decoder_inputs_embeds (torch.Tensor, optional): The input embeddings for the decoder. Default is None.
            labels (torch.Tensor, optional): The target labels for prediction. Default is None.
            use_cache (bool, optional): Whether to use cache for decoding. Default is None.
            output_attentions (bool, optional): Whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default is None.
            return_dict (bool, optional): Whether to return a dictionary as output. Default is None.

        Returns:
            None

        Raises:
            NotImplementedError: If the method encounters an operation that is not implemented.
            ValueError: If incorrect arguments are provided or if the input dimensions are not valid.
            RuntimeError: If there is an issue during model execution.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1), ignore_index=-100)
            # TODO(thom): Add z_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """prepare inputs for generation"""
        # cut decoder_input_ids if past is used
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """prepare decoder input ids from labels"""
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        '''
        This method '_reorder_cache' is defined within the class 'LongT5ForConditionalGeneration' and
        is responsible for reordering the cache for the T5 model during decoding.

        Args:
            self: The instance of the class.
            past_key_values (tuple): A tuple containing the past key and value states for each layer in the decoder.
                The past key and value states are used to speed up decoding.
                If None, a warning is logged suggesting to set 'use_cache=True' to enhance decoding speed.
            beam_idx (tensor): The indices of the selected beams to be used for reordering the past key and value states.

        Returns:
            tuple: The reordered past key and value states for the decoder.
                If the 'past_key_values' parameter is None, it returns None.

        Raises:
            AssertionError: If the shape or length of the reordered layer past states does not match the
                original layer past states.
        '''
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logging.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class LongT5EncoderModel(LongT5PreTrainedModel):
    """LongT5EncoderModel"""
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: LongT5Config):
        """
        Initializes a new instance of the LongT5EncoderModel class.

        Args:
            self: The object instance.
            config (LongT5Config):
                The configuration object for the model.

                - The 'config' parameter is of type LongT5Config, which holds various configuration settings for the model.
                - It is used to initialize the base class with the provided configuration.
                - This parameter is required and must be provided.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = LongT5Stack(encoder_config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the LongT5EncoderModel.
        
        Args:
            self: An instance of the LongT5EncoderModel class.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the LongT5EncoderModel.
        
        Args:
            self (LongT5EncoderModel): The instance of the LongT5EncoderModel class.
            new_embeddings (object): New input embeddings to be set for the model.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        """
        Ties the word embeddings weights with the shared layer weights if specified in the configuration.
        
        Args:
            self (LongT5EncoderModel): The instance of the LongT5EncoderModel class.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        """get encoder"""
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        head_mask = None,
        inputs_embeds = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        """
        This method forwards the LongT5EncoderModel by passing the input parameters to the encoder.
        
        Args:
            self: The instance of the LongT5EncoderModel class.
            input_ids (Optional[Tensor]): The input token IDs for the encoder. Default is None.
            attention_mask (Optional[Tensor]): The attention mask tensor for the encoder. Default is None.
            head_mask (Optional[Tensor]): The head mask tensor for the encoder. Default is None.
            inputs_embeds (Optional[Tensor]): The input embeddings for the encoder. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default is None.
        
        Returns:
            None.
        
        Raises:
            None
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

__all__ = [
    "LongT5_PRETRAINED_MODEL_ARCHIVE_LIST",
    "LongT5EncoderModel",
    "LongT5ForConditionalGeneration",
    "LongT5Model",
    "LongT5PreTrainedModel",
]
