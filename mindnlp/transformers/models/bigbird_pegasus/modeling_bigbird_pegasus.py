# coding=utf-8
# Copyright 2021 Google Research The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore BigBirdPegasus model."""


import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.modules.functional import finfo
from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from .configuration_bigbird_pegasus import BigBirdPegasusConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/bigbird-pegasus-large-arxiv"
_CONFIG_FOR_DOC = "BigBirdPegasusConfig"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 1024]


def shift_tokens_right(input_ids: mindspore.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].copy()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = shifted_input_ids.masked_fill(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class BigBirdPegasusLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """Initialize the BigBirdPegasusLearnedPositionalEmbedding class.
        
        This method initializes an instance of the BigBirdPegasusLearnedPositionalEmbedding class with the given parameters.
        
        Args:
            self (BigBirdPegasusLearnedPositionalEmbedding): The instance of the class.
            num_embeddings (int): The number of embeddings to be generated.
            embedding_dim (int): The dimension of each embedding.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__(num_embeddings, embedding_dim)

    def construct(self, input_ids_shape, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = ops.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=mindspore.int64
        )
        return super().construct(positions)


# Copied from transformers.models.big_bird.modeling_big_bird.BigBirdSelfAttention with BigBird->BigBirdPegasus
class BigBirdPegasusSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism implementation based on the BigBird Pegasus architecture. 
    It is designed for neural network models and inherits from nn.Cell. 
    
    Attributes:
        num_attention_heads (int): Number of attention heads in the self-attention mechanism.
        attention_head_size (int): Size of each attention head in the self-attention mechanism.
        all_head_size (int): Total size of all attention heads combined.
        query (nn.Dense): Linear transformation layer for query computation.
        key (nn.Dense): Linear transformation layer for key computation.
        value (nn.Dense): Linear transformation layer for value computation.
        dropout (nn.Dropout): Dropout layer for attention probabilities.
        is_decoder (bool): Indicates whether the self-attention mechanism is used in a decoder context.
    
    Methods:
        swapaxes_for_scores(self, x): Reshapes the input tensor for attention score computation.
        construct(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, 
            encoder_attention_mask=None, past_key_value=None, output_attentions=False): 
            Performs the self-attention mechanism computation based on the provided inputs.
    
    Raises:
        ValueError: If the hidden size is not a multiple of the number of attention heads.
    
    Returns:
        tuple: A tuple containing the context layer and optionally attention probabilities and past key-value pairs.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BigBirdPegasusSelfAttention class.
        
        Args:
            self: An instance of the BigBirdPegasusSelfAttention class.
            config: An object containing configuration parameters for the self-attention layer. 
                It should have the following attributes:
                
                - hidden_size (int): The size of the hidden layer.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int, optional): The size of the embedding layer. 
                Only required if hidden_size is not a multiple of num_attention_heads.
                - use_bias (bool): Flag indicating whether to include bias in the linear transformations.
                - attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
                - is_decoder (bool): Flag indicating whether the self-attention layer is used as a decoder.

        Returns:
            None.

        Raises:
            ValueError: 
                If the hidden_size is not a multiple of the num_attention_heads and embedding_size is not provided.

        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.use_bias)
        self.key = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.use_bias)
        self.value = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.use_bias)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def swapaxes_for_scores(self, x):
        """
        Performs axis swapping and reshaping operations on the input tensor to prepare it for BigBirdPegasusSelfAttention.

        Args:
            self: An instance of the BigBirdPegasusSelfAttention class.
            x (Tensor): The input tensor to be processed. It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        This method constructs the self-attention mechanism for the BigBirdPegasus model.

        Args:
            self: The instance of the class.
            hidden_states (Tensor): The input hidden states for the attention mechanism.
            attention_mask (Tensor, optional): Mask to avoid attending to specific positions. Default is None.
            head_mask (Tensor, optional): Mask for specific attention heads. Default is None.
            encoder_hidden_states (Tensor, optional): Hidden states of the encoder if cross-attention is used. Default is None.
            encoder_attention_mask (Tensor, optional): Mask for encoder attention. Default is None.
            past_key_value (Tuple[Tensor, Tensor], optional): Tuple of past key and value if available. Default is None.
            output_attentions (bool): Flag to output attention probabilities. Default is False.

        Returns:
            Tuple[Tensor, Tensor] or Tuple[Tensor, Tuple[Tensor, Tensor]]: 
                The context layer and optionally the attention probabilities and past key/value.

        Raises:
            ValueError: If the dimensions of the input tensors are incompatible.
            TypeError: If the input parameters are not of the expected types.
            RuntimeError: If there are runtime issues during the computation.
        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.swapaxes_for_scores(self.key(encoder_hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.swapaxes_for_scores(self.key(hidden_states))
            value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        query_layer = self.swapaxes_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BigBirdPegasusModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.big_bird.modeling_big_bird.BigBirdBlockSparseAttention with BigBird->BigBirdPegasus
class BigBirdPegasusBlockSparseAttention(nn.Cell):

    """
    The `BigBirdPegasusBlockSparseAttention` class represents a block sparse attention mechanism used in the 
    BigBird model for text generation tasks. 
    This class inherits from the `nn.Cell` class and provides methods for performing block sparse attention operations.

    The class initializes and constructs the block sparse attention mechanism using the provided configuration parameters. 
    It also includes methods for fast matrix multiplication, creating random attention masks, and generating adjacency 
    lists for random attention.

    Methods:
        `__init__(self, config, seed=None)`: Initializes the BigBirdPegasusBlockSparseAttention with the given 
            configuration and optional seed for randomization.
        `swapaxes_for_scores(self, x)`: Swaps axes for the attention scores.
        `construct(self, hidden_states, band_mask=None, from_mask=None, to_mask=None, from_blocked_mask=None, 
            to_blocked_mask=None, output_attentions=None)`: 
            Constructs the block sparse attention mechanism using the given hidden states and masks.
        `ms_bmm_nd(self, inp_1, inp_2, ndim=None)`: Fast n-dimensional matrix multiplication.
        `ms_bmm_nd_swapaxes(self, inp_1, inp_2, ndim=None)`: Fast n-dimensional matrix multiplication with swapaxes.
        `bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer, band_mask, from_mask, to_mask, 
            from_blocked_mask, to_blocked_mask, n_heads, n_rand_blocks, attention_head_size, from_block_size, 
            to_block_size, batch_size, from_seq_len, to_seq_len, seed, plan_from_length, plan_num_rand_blocks, 
            output_attentions)`:
            Performs the block sparse attention mechanism with the given input tensors and parameters.
        `ms_gather_b2(params, indices)`: Gathers elements from the input tensors based on the provided indices.
        `_create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, num_attention_heads, 
            num_rand_blocks, batch_size, from_seq_length, from_block_size)`: 
            Creates a 3D attention mask from 2D tensor masks.
        `_get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks)`: 
            Generates a plan for random attention placement.
        `_bigbird_block_rand_mask(self, from_seq_length, to_seq_length, from_block_size, to_block_size, 
            num_rand_blocks, last_idx=-1)`: Creates an adjacency list of random attention.
        `_bigbird_block_rand_mask_with_head(self, from_seq_length, to_seq_length, from_block_size, to_block_size, 
            num_heads, plan_from_length, plan_num_rand_blocks, window_block_left=1, window_block_right=1,
            global_block_top=1, global_block_bottom=1, global_block_left=1, global_block_right=1)`: 
                Creates an adjacency list of random attention with multiple heads.
        `_get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks, 
            window_block_left=1, window_block_right=1, global_block_left=1, global_block_right=1)`: 
            Gets random row attention for a single block row.

    This class provides a comprehensive implementation of the block sparse attention mechanism for efficient text generation tasks.
    """
    def __init__(self, config, seed=None):
        """
        Initializes a BigBirdPegasusBlockSparseAttention object.

        Args:
            self (object): The instance of the BigBirdPegasusBlockSparseAttention class.
            config (object): 
                An object containing configuration parameters for the attention block.
                
                - max_position_embeddings (int): Maximum sequence length.
                - hidden_size (int): Size of the hidden layer.
                - num_attention_heads (int): Number of attention heads.
                - num_random_blocks (int): Number of random blocks.
                - block_size (int): Size of each block.
                - use_bias (bool): Indicates whether bias is used in Dense layers.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.
        """
        super().__init__()

        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.use_bias)
        self.key = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.use_bias)
        self.value = nn.Dense(config.hidden_size, self.all_head_size, has_bias=config.use_bias)

    def swapaxes_for_scores(self, x):
        """
        Performs the axis swapping operation on the input tensor for bigbird's block-sparse attention.

        Args:
            self (BigBirdPegasusBlockSparseAttention): An instance of the BigBirdPegasusBlockSparseAttention class.
            x (torch.Tensor): The input tensor to be processed. 
                Its shape should be compatible with (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: The input tensor with axes swapped. 
                The shape of the output tensor is (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None.

        This method takes in two parameters, 'self' and 'x'. 
        
        'self' is an instance of the BigBirdPegasusBlockSparseAttention class, 
        used to access other class attributes and methods. 
        
        'x' is a torch.Tensor object representing the input tensor to be processed. 
        The input tensor should have a shape compatible with (batch_size, sequence_length, hidden_size).

        The method performs an axis swapping operation on the input tensor 'x'. 
        It first calculates the new shape of 'x' by combining the existing dimensions with the number of 
        attention heads and the size of each attention head. 
        Then, using the 'view' method of the tensor, the shape of 'x' is rearranged to match the new shape. 
        Finally, the 'permute' method is applied to the tensor to swap the axes according to the specified 
        permutation order (0, 2, 1, 3).

        The method returns the processed tensor with the axes swapped. T
        he shape of the output tensor is (batch_size, num_attention_heads, sequence_length, attention_head_size).

        No exceptions are raised by this method.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
    ):
        """
        This method 'construct' is a part of the class 'BigBirdPegasusBlockSparseAttention'. It takes the following parameters:

        Args:
            self: The instance of the class.
            hidden_states (tensor): The input tensor of shape (batch_size, seqlen, _), 
                where batch_size is the number of sequences in a batch, seqlen is the sequence length, 
                and _ represents the hidden size.
            band_mask (tensor, optional): The band mask tensor used for sparse attention computation. Default is None.
            from_mask (tensor, optional): The from mask tensor. Default is None.
            to_mask (tensor, optional): The to mask tensor. Default is None.
            from_blocked_mask (tensor, optional): The from blocked mask tensor. Default is None.
            to_blocked_mask (tensor, optional): The to blocked mask tensor. Default is None.
            output_attentions (bool, optional): Whether to output attention probabilities. Default is None.

        Returns:
            outputs: A tuple containing the context layer tensor of shape (batch_size, from_seq_length, -1) 
                and the attention probabilities tensor if output_attentions is True, else only the context layer tensor.

        Raises:
            ValueError: If the query sided sequence length is not a multiple of the block size 
                or if the key/value sided sequence length is not a multiple of the block size.
        """
        # Currently this `class` can't be used in decoder.

        batch_size, seqlen, _ = hidden_states.shape
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        query_layer = self.swapaxes_for_scores(self.query(hidden_states))
        key_layer = self.swapaxes_for_scores(self.key(hidden_states))
        value_layer = self.swapaxes_for_scores(self.value(hidden_states))

        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        context_layer = context_layer.view(batch_size, from_seq_length, -1)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def ms_bmm_nd(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication"""
        # faster replacement of ops.einsum ("bhqk,bhkd->bhqd")
        return ops.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def ms_bmm_nd_swapaxes(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication with swapaxes"""
        # faster replacement of ops.einsum (bhqd,bhkd->bhqk)
        return ops.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).swapaxes(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))

    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
    ):
        """
        This method calculates the context layer and attention probabilities using the BigBird block-sparse attention mechanism.

        Args:
            self: The instance of the class.
            query_layer: The input query tensor of shape (batch_size, from_seq_len, hidden_size).
            key_layer: The input key tensor of shape (batch_size, to_seq_len, hidden_size).
            value_layer: The input value tensor of shape (batch_size, to_seq_len, hidden_size).
            band_mask: Band mask tensor to apply attention masking of shape (batch_size, n_heads, from_seq_len, to_seq_len).
            from_mask: Mask tensor to apply masking to the 'from' sequence of shape (batch_size, 1, 1, from_seq_len).
            to_mask: Mask tensor to apply masking to the 'to' sequence of shape (batch_size, 1, 1, to_seq_len).
            from_blocked_mask: Mask tensor for blocked 'from' sequence of shape (batch_size, n_heads, from_seq_len // from_block_size, from_block_size).
            to_blocked_mask: Mask tensor for blocked 'to' sequence of shape (batch_size, n_heads, to_seq_len // to_block_size, to_block_size).
            n_heads: Number of attention heads.
            n_rand_blocks: Number of random blocks in the BigBird attention mechanism.
            attention_head_size: Size of each attention head.
            from_block_size: Size of each block in the 'from' sequence.
            to_block_size: Size of each block in the 'to' sequence.
            batch_size: Size of the input batch.
            from_seq_len: Length of the 'from' sequence.
            to_seq_len: Length of the 'to' sequence.
            seed: Seed value for random operations.
            plan_from_length: Length of the plan for random attention if applicable.
            plan_num_rand_blocks: Number of random blocks in the plan if applicable.
            output_attentions: Flag to indicate whether to output attention probabilities.

        Returns:
            context_layer: The context layer tensor of shape (batch_size, from_seq_len, hidden_size).
            attention_probs: 
                The attention probabilities tensor of shape (batch_size, n_heads, from_seq_len, to_seq_len) 
                if 'output_attentions' is True, else None.

        Raises:
            ValueError: Raised if the number of blocks in 'from' and 'to' sequences are not equal.
        """
        # BigBirdPegasus block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        # attention is calculated separately for q[0], q[1], q[2:-2], q[-2], q[-1] in order to use special trick of shifting tokens (for calculating sliding attention)
        # hence following code can be divided into 5 parts.

        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0

        # generate random attention and corresponding masks
        np.random.seed(seed)
        if from_seq_len in [1024, 3072, 4096]:  # old plans used in paper
            rand_attn = [
                self._bigbird_block_rand_mask(
                    self.max_seqlen, self.max_seqlen, from_block_size, to_block_size, n_rand_blocks, last_idx=1024
                )[: (from_seq_len // from_block_size - 2)]
                for _ in range(n_heads)
            ]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = self._get_rand_attn_plan(
                    from_seq_len, from_block_size, n_rand_blocks
                )

            rand_attn = self._bigbird_block_rand_mask_with_head(
                from_seq_length=from_seq_len,
                to_seq_length=to_seq_len,
                from_block_size=from_block_size,
                to_block_size=to_block_size,
                num_heads=n_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks,
            )

        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = mindspore.tensor(rand_attn, dtype=mindspore.int64)
        rand_attn = rand_attn.unsqueeze(0)
        rand_attn = ops.cat([rand_attn for _ in range(batch_size)], axis=0)

        rand_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask, to_blocked_mask, rand_attn, n_heads, n_rand_blocks, bsz, from_seq_len, from_block_size
        )

        blocked_query_matrix = query_layer.view(bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1)
        blocked_key_matrix = key_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)
        blocked_value_matrix = value_layer.view(bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1)

        # preparing block for randn attn
        gathered_key = self.ms_gather_b2(blocked_key_matrix, rand_attn)
        gathered_key = gathered_key.view(
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]
        gathered_value = self.ms_gather_b2(blocked_value_matrix, rand_attn)
        gathered_value = gathered_value.view(
            bsz, n_heads, to_seq_len // to_block_size - 2, n_rand_blocks * to_block_size, -1
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, n_rand_blocks, to_block_size, -1]

        # 1st PART
        # 1st block (global block) attention scores
        # q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        first_product = self.ms_bmm_nd_swapaxes(blocked_query_matrix[:, :, 0], key_layer, ndim=4)

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = ops.softmax(
            first_product, axis=-1
        )  # [bsz, n_heads, from_block_size, to_seq_len]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        first_context_layer = self.ms_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer = first_context_layer.unsqueeze(2)

        # 2nd PART
        # 2nd block attention scores
        # q[1] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> 2nd, 3rd blocks
        # global key blocks -> 1st block

        second_key_mat = ops.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 0],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]
        second_value_mat = ops.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 0],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_product = self.ms_bmm_nd_swapaxes(blocked_query_matrix[:, :, 1], second_key_mat, ndim=4)
        second_seq_pad = ops.cat(
            [
                to_mask[:, :, :, : 3 * to_block_size],
                to_mask[:, :, :, -to_block_size:],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            axis=3,
        )
        second_rand_pad = ops.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, 0],
            ],
            axis=3,
        )
        second_product = second_product * rsqrt_d
        second_product += (1.0 - ops.minimum(second_seq_pad, second_rand_pad)) * attn_mask_penalty
        second_attn_weights = ops.softmax(
            second_product, axis=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_context_layer = self.ms_bmm_nd(second_attn_weights, second_value_mat, ndim=4)

        second_context_layer = second_context_layer.unsqueeze(2)

        # 3rd PART
        # Middle blocks attention scores
        # q[-2:2] x (sliding_keys, random_keys, global_keys)
        # sliding attn is calculated using special trick of shifting tokens as discussed in paper
        # random keys are generated by taking random indices as per `rand_attn`
        # global keys -> 1st & last block

        exp_blocked_key_matrix = ops.cat(
            [blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], blocked_key_matrix[:, :, 3:-1]], axis=3
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        exp_blocked_value_matrix = ops.cat(
            [blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], blocked_value_matrix[:, :, 3:-1]],
            axis=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [b, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        inner_band_product = self.ms_bmm_nd_swapaxes(middle_query_matrix, exp_blocked_key_matrix, ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size]
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        rand_band_product = self.ms_bmm_nd_swapaxes(middle_query_matrix, gathered_key[:, :, 1:-1], ndim=5)
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size]
        rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        first_band_product = ops.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        last_band_product = ops.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)) * attn_mask_penalty
        last_band_product += (1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)) * attn_mask_penalty
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * attn_mask_penalty

        # completing attention scores matrix for all q[-2:2]
        band_product = ops.cat(
            [first_band_product, inner_band_product, rand_band_product, last_band_product], axis=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # safely doing softmax since attention matrix is completed
        attn_weights = ops.softmax(
            band_product, axis=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+n_rand_blocks)*to_block_size]

        # contribution of sliding keys
        # [bsz, n_heads, m//from_block_size-4, from_block_size, 3*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        context_layer = self.ms_bmm_nd(
            attn_weights[:, :, :, :, to_block_size : 4 * to_block_size], exp_blocked_value_matrix, ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of random keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, n_rand_blocks*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, n_rand_blocks*to_block_size, -1]
        context_layer += self.ms_bmm_nd(
            attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size], gathered_value[:, :, 1:-1], ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of global keys
        context_layer += ops.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :to_block_size], blocked_value_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        context_layer += ops.einsum(
            "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -to_block_size:], blocked_value_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_mat = ops.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -1],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+n_random_blocks)*to_block_size, -1]
        second_last_value_mat = ops.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -1],
            ],
            axis=2,
        )  # [bsz, n_heads, (4+r)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]
        second_last_product = self.ms_bmm_nd_swapaxes(blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4)
        second_last_seq_pad = ops.cat(
            [
                to_mask[:, :, :, :to_block_size],
                to_mask[:, :, :, -3 * to_block_size :],
                to_mask.new_ones([bsz, 1, 1, n_rand_blocks * to_block_size]),
            ],
            axis=3,
        )
        second_last_rand_pad = ops.cat(
            [
                rand_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                rand_mask[:, :, -1],
            ],
            axis=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (1.0 - ops.minimum(second_last_seq_pad, second_last_rand_pad)) * attn_mask_penalty
        second_last_attn_weights = ops.softmax(
            second_last_product, axis=-1
        )  # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+n_rand_blocks)*to_block_size] x [bsz, n_heads, (4+n_rand_blocks)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_last_context_layer = self.ms_bmm_nd(second_last_attn_weights, second_last_value_mat, ndim=4)
        second_last_context_layer = second_last_context_layer.unsqueeze(2)

        # 5th PART
        # last block (global) attention scores
        # q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        last_product = self.ms_bmm_nd_swapaxes(blocked_query_matrix[:, :, -1], key_layer, ndim=4)
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = ops.softmax(last_product, axis=-1)  # [bsz, n_heads, from_block_size, n]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        last_context_layer = self.ms_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer = last_context_layer.unsqueeze(2)

        # combining representations of all tokens
        context_layer = ops.cat(
            [first_context_layer, second_context_layer, context_layer, second_last_context_layer, last_context_layer],
            axis=2,
        )
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = ops.swapaxes(context_layer, 1, 2)

        # this is just for visualizing; forward pass doesn't depend on following code
        if output_attentions:
            # TODO(PVP): need to verify if below code is correct
            attention_probs = ops.zeros(
                bsz, n_heads, from_seq_len, to_seq_len, dtype=mindspore.float32)

            # 1st query block
            # corresponding to `first_context_layer`
            attention_probs[:, :, :from_block_size, :] = first_attn_weights  # all keys global

            # 2nd query block
            # corresponding to `second_context_layer`
            attention_probs[:, :, from_block_size : 2 * from_block_size, : 3 * to_block_size] = second_attn_weights[
                :, :, :, : 3 * to_block_size
            ]  # 1st three key blocks (global + sliding)
            attention_probs[:, :, from_block_size : 2 * from_block_size, -to_block_size:] = second_attn_weights[
                :, :, :, 3 * to_block_size : 4 * to_block_size
            ]  # last key block (global)
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, 1][:, i2[0]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

            # Middle query blocks
            # corresponding to `context_layer`
            # sliding keys
            for q_idx in range(from_seq_len // from_block_size - 4):
                attn_probs_view = attention_probs.view(
                    bsz,
                    n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                )[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[:, :, q_idx, :, to_block_size : 4 * to_block_size]
                attn_probs_view[:, :, q_idx, :, q_idx : q_idx + 3, :] = right_slice.view(
                    bsz, n_heads, from_block_size, 3, to_block_size
                )  # inner_band_product
            # global keys (corresponding to 1st key block)
            attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, :to_block_size] = attn_weights[
                :, :, :, :, :to_block_size
            ].view(bsz, n_heads, -1, to_block_size)  # first_band_product
            # global keys (corresponding to last key block)
            attention_probs[:, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:] = attn_weights[
                :, :, :, :, -to_block_size:
            ].view(bsz, n_heads, -1, to_block_size)  # last_band_product
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    for q_idx in range(1, len(i2) - 1):
                        attn_probs_view = attention_probs.view(
                            bsz,
                            n_heads,
                            from_seq_len // from_block_size,
                            from_block_size,
                            to_seq_len // to_block_size,
                            to_block_size,
                        )
                        right_slice = w2[q_idx - 1, :, 4 * to_block_size : -to_block_size]
                        attn_probs_view[p1, p2, q_idx + 1][:, i2[q_idx]] = right_slice.view(
                            from_block_size, n_rand_blocks, to_block_size
                        )

            # Second-last query block
            # corresponding to `second_last_context_layer`
            attention_probs[:, :, -2 * from_block_size : -from_block_size, :to_block_size] = second_last_attn_weights[
                :, :, :, :to_block_size
            ]  # 1st key block (global)
            attention_probs[
                :, :, -2 * from_block_size : -from_block_size, -3 * to_block_size :
            ] = second_last_attn_weights[
                :, :, :, to_block_size : 4 * to_block_size
            ]  # last three blocks (global + sliding)
            # random keys
            for p1, i1, w1 in zip(range(bsz), rand_attn, second_last_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, -2][:, i2[-1]] = right_slice.view(
                        from_block_size, n_rand_blocks, to_block_size
                    )

            # last query block
            # corresponding to `last_context_layer`
            attention_probs[:, :, -from_block_size:, :] = last_attn_weights  # all keys global

        else:
            attention_probs = None

        return context_layer, attention_probs

    @staticmethod
    def ms_gather_b2(params, indices):
        """
        Performs block sparse attention gathering using the given parameters and indices.

        Args:
            params (Tensor): The input tensor containing the parameters for gathering.
            indices (Tensor): The input tensor containing the indices for gathering.

        Returns:
            None: 
                This method does not return any value directly, 
                but the result of the gathering operation is reflected in the output tensor.

        Raises:
            ValueError: 
                If the first two dimensions of params and indices are not identical, 
                a ValueError is raised with a descriptive message.
        """
        # this operation is equivalent to tf.gather when batch_dims=2

        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical,                 but"
                f" they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}"
            )
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]

        shift = ops.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather)
        indices_shift = ops.div(shift, num_indices_to_gather, rounding_mode="floor") * num_indices_to_pick_from

        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])

        out_flattened = flattened_params.index_select(0, flattened_indices)

        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask,
        to_blocked_mask,
        rand_attn,
        num_attention_heads,
        num_rand_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].
            rand_attn: [batch_size, num_attention_heads,
            from_seq_length//from_block_size-2, num_rand_blocks]
            num_attention_heads: int. Number of attention heads.
            num_rand_blocks: int. Number of random chunks per row.
            batch_size: int. Batch size for computation.
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
            from_block_size, num_rand_blocks*to_block_size].
        """
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = ops.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        rand_mask = ops.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence.
            from_block_size: int. size of block in from sequence.
            num_rand_blocks: int. Number of random chunks per row.

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
            each block
        """
        plan_from_length = []
        plan_num_rand_blocks = []
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    def _bigbird_block_rand_mask(
        self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        # During inference (eval) no randomness
        if not self.training:
            return rand_attn
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_heads: int. total number of heads.
            plan_from_length: list. plan from length where num_random_blocks are chosen from.
            plan_num_rand_blocks: list. number of rand blocks within the plan.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_top: int. number of blocks at the top.
            global_block_bottom: int. number of blocks at the bottom.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            adjacency list of size num_head where each element is of size from_seq_length//from_block_size-2 by
            num_rand_blocks
        """
        # using this method when from_seq_length not in [1024, 3072, 4096]

        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        if from_seq_length not in plan_from_length:
            raise ValueError("Error from sequence length not in plan!")

        # Total number of blocks in the mmask
        num_blocks = from_seq_length // from_block_size
        # Number of blocks per plan
        plan_block_length = np.array(plan_from_length) // from_block_size
        # till when to follow plan
        max_plan_idx = plan_from_length.index(from_seq_length)

        # Random Attention adjacency list
        rand_attn = [
            np.zeros((num_blocks, np.sum(plan_num_rand_blocks[: max_plan_idx + 1])), dtype=np.int32)
            for i in range(num_heads)
        ]
        # During inference (eval) no randomness
        if not self.training:
            for nh in range(num_heads):
                rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]
            return rand_attn

        # We will go iteratively over the plan blocks and pick random number of
        # Attention blocks from the legally allowed blocks
        for plan_idx in range(max_plan_idx + 1):
            rnd_r_cnt = 0
            if plan_idx > 0:
                # set the row for all from_blocks starting from 0 to
                # plan_block_length[plan_idx-1]
                # column indx start fromm plan_block_length[plan_idx-1] and ends at
                # plan_block_length[plan_idx]
                if plan_num_rand_blocks[plan_idx] > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                    for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx - 1]):
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

                for pl_id in range(plan_idx):
                    if plan_num_rand_blocks[pl_id] == 0:
                        continue
                    for blk_rw_idx in range(plan_block_length[plan_idx - 1], plan_block_length[plan_idx]):
                        rnd_r_cnt = 0
                        to_start_block_id = 0
                        if pl_id > 0:
                            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                            to_start_block_id = plan_block_length[pl_id - 1]
                        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                        for h in range(num_heads):
                            rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )

            if plan_num_rand_blocks[plan_idx] == 0:
                continue
            curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
            from_start_block_id = global_block_top
            to_start_block_id = 0
            if plan_idx > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                from_start_block_id = plan_block_length[plan_idx - 1]
                to_start_block_id = plan_block_length[plan_idx - 1]

            for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
                for h in range(num_heads):
                    rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = self._get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right,
                    )

        for nh in range(num_heads):
            rand_attn[nh] = rand_attn[nh][global_block_top : num_blocks - global_block_bottom, :]

        return rand_attn

    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        For a single row block get random row attention.

        Args:
            block_id: int. block id of row.
            to_start_block_id: int. random attention column start id.
            to_end_block_id: int. random attention column end id.
            num_rand_blocks: int. number of random blocks to be selected.
            window_block_left: int. number of blocks of window to left of a block.
            window_block_right: int. number of blocks of window to right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            row containing the random attention vector of size num_rand_blocks.
        """
        # list of to_blocks from which to choose random attention
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # permute the blocks
        perm_block = np.random.permutation(to_block_list)

        # illegal blocks for the current block id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)


class BigBirdPegasusEncoderAttention(nn.Cell):

    """
    This class represents the attention mechanism for the BigBirdPegasus encoder. 
    It handles different types of attention mechanisms based on the 'attention_type' specified in the configuration. 
    The class provides methods to set the attention type and construct the attention outputs based on the specified type.

    The class inherits from nn.Cell and contains methods to initialize the attention type, set the attention type, 
    and construct the attention outputs. 
    The supported attention types are 'original_full' and 'block_sparse'.

    Attributes:
        config (object): The configuration object specifying the attention type and other parameters.
        seed (int): The seed value used for random initialization, default is None.
        attention_type (str): The type of attention mechanism used, either 'original_full' or 'block_sparse'.
        self (object): The attention mechanism object based on the specified type.
        output (object): The output layer applied to the attention outputs.

    Methods:
        set_attention_type: Sets the attention type to the specified value.
        construct: Constructs the attention outputs based on the attention type.

    Raises:
        ValueError: If an invalid attention type is provided or if the attention type cannot be set to the specified value.
    """
    def __init__(self, config, seed=None):
        """
        Initializes the BigBirdPegasusEncoderAttention class.

        Args:
            self: The instance of the class.
            config: The configuration object containing settings for the encoder attention.
                    It should have attributes like attention_type, hidden_size, and use_bias.
            seed: An optional integer value representing the random seed for reproducibility.

        Returns:
            None.

        Raises:
            ValueError: If the attention_type specified in the config is not 'original_full' or 'block_sparse'.
        """
        super().__init__()
        self.config = config
        self.seed = seed

        self.attention_type = config.attention_type

        if self.attention_type == "original_full":
            self.self = BigBirdPegasusSelfAttention(config)
        elif self.attention_type == "block_sparse":
            self.self = BigBirdPegasusBlockSparseAttention(config, seed)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        self.output = nn.Dense(config.hidden_size, config.hidden_size, has_bias=config.use_bias)

    def set_attention_type(self, value: str):
        """
        This method sets the attention type for the BigBirdPegasusEncoderAttention class.

        Args:
            self: The instance of the BigBirdPegasusEncoderAttention class.
            value (str): The attention type to be set. It must be either 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the provided attention type 'value' is not 'original_full' or 'block_sparse'.
        """
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return

        self.attention_type = value
        if value == "original_full":
            # copy all weights to new full attention class
            attn_weights = BigBirdPegasusSelfAttention(self.config)
        else:
            # copy all weights to new sparse attention class
            attn_weights = BigBirdPegasusBlockSparseAttention(self.config, self.seed)

        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value

        if not self.training:
            self.self.set_train(False)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
    ):
        """
        Args:
            self: The instance of the class.
            hidden_states: Tensor containing the input hidden states with shape (batch_size, sequence_length, hidden_size).
            attention_mask: Optional tensor with shape (batch_size, sequence_length) 
                containing the attention mask values for the input sequence. Defaults to None.
            head_mask: Optional tensor with shape (num_heads,) containing the mask for attention heads. Defaults to None.
            past_key_value: Optional tuple containing the past key and value tensors for autoregressive decoding. 
                Defaults to None.
            output_attentions: Boolean indicating whether to return the attention weights. Defaults to False.
            band_mask: Optional tensor with shape (batch_size, num_heads, from_seq_length, to_seq_length) 
                containing the band mask values for the attention mechanism. Defaults to None.
            from_mask: Optional tensor with shape (batch_size, from_seq_length) 
                containing the from mask values for the attention mechanism. Defaults to None.
            to_mask: Optional tensor with shape (batch_size, to_seq_length) 
                containing the to mask values for the attention mechanism. Defaults to None.
            from_blocked_mask: Optional tensor with shape (batch_size, num_attention_heads, from_blocks, from_bandwidth) 
                containing the from blocked mask values for the attention mechanism. Defaults to None.
            to_blocked_mask: Optional tensor with shape (batch_size, num_attention_heads, to_blocks, to_bandwidth) 
                containing the to blocked mask values for the attention mechanism. Defaults to None.

        Returns:
            Tuple containing the attention output tensor and additional outputs from the attention mechanism.

        Raises:
            None.
        """
        # Expand dims to enable multiplication in the self-attention module
        head_mask = head_mask.reshape(1, -1, 1, 1) if head_mask is not None else None

        if self.config.attention_type == "original_full":
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        else:
            self_outputs = self.self(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )

        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bart.modeling_bart.BartAttention with BartConfig->BigBirdPegasusConfig, Bart->BigBirdPegasusDecoder
class BigBirdPegasusDecoderAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BigBirdPegasusConfig] = None,
    ):
        """
        Initializes the BigBirdPegasusDecoderAttention class.

        Args:
            self: The instance of the class.
            embed_dim (int): The dimension of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability. Defaults to 0.0.
            is_decoder (bool, optional): Indicates if the attention mechanism is used as a decoder. Defaults to False.
            bias (bool, optional): Indicates whether to include bias in the projection layers. Defaults to True.
            is_causal (bool, optional): Indicates if the attention is causal. Defaults to False.
            config (Optional[BigBirdPegasusConfig], optional): The configuration for the BigBirdPegasus model. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.v_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.q_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)
        self.out_proj = nn.Dense(embed_dim, embed_dim, has_bias=bias)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        This method '_shape' in the class 'BigBirdPegasusDecoderAttention' reshapes the input tensor to match 
        the required dimensions for decoder attention in the BigBird Pegasus model.

        Args:
            self (BigBirdPegasusDecoderAttention): The instance of the BigBirdPegasusDecoderAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped. 
                It should have the shape (batch size, sequence length, hidden size).
            seq_len (int): The length of the input sequence.
            bsz (int): The batch size of the input tensor.

        Returns:
            None: 
                This method does not return any value but reshapes the input tensor in place to match 
                the required dimensions for decoder attention.

        Raises:
            None.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        key_value_states: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(mindspore.Tensor, mindspore.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(mindspore.Tensor, mindspore.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.bmm(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if layer_head_mask is not None:
            if layer_head_mask.shape != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.shape}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.bmm(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BigBirdPegasusEncoderLayer(nn.Cell):

    """
    This class represents a BigBirdPegasusEncoderLayer in a neural network model. 
    It is used for encoding input sequences using a combination of attention mechanisms and feedforward neural networks. 
    The layer includes mechanisms for self-attention, layer normalization, activation functions, and dropout regularization. 
    \It also supports different types of attention, such as 'original_full' and 'block_sparse'.

    The BigBirdPegasusEncoderLayer inherits from the nn.Cell class and consists of methods for initialization, 
    constructing the layer, and setting the attention type. 
    The layer's parameters and operations are defined based on the provided configuration, including attention types, 
    embedding dimensions, and feedforward neural network dimensions.

    The `construct` method processes the input hidden states through self-attention mechanisms, normalization, 
    activation functions, and feedforward neural networks. 
    It allows for optional outputs of attention tensors from all attention layers. 
    The method handles different types of masks for attention and dropout regularization based on the provided configurations.

    The `set_attention_type` method allows for setting the type of attention mechanism used in the layer, 
    ensuring that it is either 'original_full' or 'block_sparse'. It updates the attention type parameter
    and adjusts the self-attention mechanism accordingly.

    Overall, the BigBirdPegasusEncoderLayer class encapsulates the functionality of an encoder layer 
    within the BigBirdPegasus model, providing essential operations for processing input sequences and capturing
    complex patterns in the data.
    """
    def __init__(self, config: BigBirdPegasusConfig, seed=None):
        """
        __init__

        Initializes a new instance of the BigBirdPegasusEncoderLayer class.

        Args:
            self: The instance of the class.
            config (BigBirdPegasusConfig): An instance of BigBirdPegasusConfig containing the configuration parameters for the encoder layer.
            seed (int, optional): An integer representing the seed for random number generation. Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.attention_type = config.attention_type
        self.embed_dim = config.d_model
        self.self_attn = BigBirdPegasusEncoderAttention(config, seed=seed)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Dense(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Dense(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        layer_head_mask: mindspore.Tensor,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=from_blocked_mask,
            to_blocked_mask=to_blocked_mask,
        )
        hidden_states = self_attention_outputs[0]

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))

        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == mindspore.float16 and (
            ops.isinf(hidden_states).any() or ops.isnan(hidden_states).any()
        ):
            clamp_value = finfo(hidden_states.dtype, 'max') - 1000
            hidden_states = ops.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attention_outputs[1],)

        return outputs

    def set_attention_type(self, value: str):
        """
        Sets the attention type for the BigBirdPegasusEncoderLayer.

        Args:
            self: The instance of the BigBirdPegasusEncoderLayer class.
            value (str): The attention type to be set. It can only have the values 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the provided attention type is not 'original_full' or 'block_sparse'.

        This method sets the attention type of the BigBirdPegasusEncoderLayer to either 'original_full' or 'block_sparse'.
        If the provided attention type is the same as the current attention type, the method will return without making any changes.
        If the provided attention type is not valid, a ValueError will be raised.

        Note:
            This method also updates the attention type of the self-attention sub-layer within the BigBirdPegasusEncoderLayer.
        """
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return
        self.attention_type = value
        self.self_attn.set_attention_type(value)


class BigBirdPegasusDecoderLayer(nn.Cell):

    """
    This class represents a decoder layer for the BigBirdPegasus model. 
    It includes self-attention and cross-attention mechanisms, feedforward neural network, layer normalization, and dropout.

    The class inherits from nn.Cell and has an initialization method that takes a BigBirdPegasusConfig object 
    as input and constructs the decoder layer. 
    The construct method takes input hidden states and various optional arguments to perform decoding and returns 
    the output tensor. 
    It also includes detailed parameter descriptions and option flags for controlling the output.

    The decoder layer consists of self-attention, encoder attention, layer normalization, feedforward neural network, 
    and dropout operations. 
    The construct method also supports caching of past key and value projection states and returning attention 
    weights if specified.

    This class provides a comprehensive implementation of a decoder layer for the BigBirdPegasus model with detailed parameter and method descriptions.
    """
    def __init__(self, config: BigBirdPegasusConfig):
        """Initializes an instance of the BigBirdPegasusDecoderLayer class.

        Args:
            self (BigBirdPegasusDecoderLayer): The current instance of the BigBirdPegasusDecoderLayer class.
            config (BigBirdPegasusConfig): The configuration object that holds various settings for the decoder layer.

        Returns:
            None.

        Raises:
            None.

        This method initializes the BigBirdPegasusDecoderLayer by setting the values of various attributes. 
        It takes in the self parameter, which is a reference to the current instance of the class, and the
        config parameter, which is an instance of the BigBirdPegasusConfig class.

        Attributes:
            embed_dim (int): The embedding dimension size, which is set to the value of config.d_model.
            self_attn (BigBirdPegasusDecoderAttention): The self-attention mechanism for the decoder layer, 
                which is created using the embed_dim, config.decoder_attention_heads, config.attention_dropout,
                config.is_decoder, and config.use_bias parameters.
            dropout (float): The dropout rate, which is set to the value of config.dropout.
            activation_fn (function): The activation function used in the decoder layer, 
                which is determined based on the value of config.activation_function.
            activation_dropout (float): The dropout rate for the activation function, 
                which is set to the value of config.activation_dropout.
            self_attn_layer_norm (nn.LayerNorm): The layer normalization module applied to the output of 
                the self-attention mechanism, which is created using the embed_dim parameter.
            encoder_attn (BigBirdPegasusDecoderAttention): The encoder attention mechanism for the decoder layer, 
                which is created using the embed_dim, config.decoder_attention_heads, config.attention_dropout,
                config.is_decoder, and config.use_bias parameters.
            encoder_attn_layer_norm (nn.LayerNorm): The layer normalization module applied to the output of 
                the encoder attention mechanism, which is created using the embed_dim parameter.
            fc1 (nn.Dense): The first fully connected layer in the decoder feed-forward network, 
                which has an input size of embed_dim and an output size of config.decoder_ffn_dim.
            fc2 (nn.Dense): The second fully connected layer in the decoder feed-forward network, 
                which has an input size of config.decoder_ffn_dim and an output size of embed_dim.
            final_layer_norm (nn.LayerNorm): The layer normalization module applied to the output of the decoder layer, 
                which is created using the embed_dim parameter.

        Note:
            The BigBirdPegasusDecoderAttention and nn.Dense classes are assumed to be imported from the appropriate libraries.
        """
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BigBirdPegasusDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.use_bias,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BigBirdPegasusDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.use_bias,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Dense(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Dense(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # Copied from transformers.models.mbart.modeling_mbart.MBartDecoderLayer.forward
    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        layer_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_layer_head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> mindspore.Tensor:
        """
        Args:
            hidden_states (`mindspore.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`mindspore.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`mindspore.Tensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`mindspore.Tensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`mindspore.Tensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`mindspore.Tensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(mindspore.Tensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = ops.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# Copied from transformers.models.bart.modeling_bart.BartClassificationHead with Bart->BigBirdPegasus
class BigBirdPegasusClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        """
        Args:
            self (object): The instance of the class.
            input_dim (int): The dimension of the input features.
            inner_dim (int): The dimension of the inner layer.
            num_classes (int): The number of classes for classification.
            pooler_dropout (float): The dropout probability for the pooler layer.

        Returns:
            None.

        Raises:
            TypeError: If the input_dim, inner_dim, num_classes, or pooler_dropout is not of the expected type.
            ValueError: If the input_dim, inner_dim, num_classes, or pooler_dropout does not meet the specified restrictions.
        """
        super().__init__()
        self.dense = nn.Dense(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Dense(inner_dim, num_classes)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the classification head for the BigBirdPegasus model.

        Args:
            self: The instance of the BigBirdPegasusClassificationHead class.
            hidden_states (mindspore.Tensor): The hidden states input tensor.
                It represents the input data for the classification head.

        Returns:
            mindspore.Tensor: A tensor representing the output of the classification head.

        Raises:
            None
        """
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = ops.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BigBirdPegasusPreTrainedModel(PreTrainedModel):

    """
    The 'BigBirdPegasusPreTrainedModel' class represents a pre-trained model for natural language processing tasks. 
    It inherits from the 'PreTrainedModel' class and includes methods for initializing weights
    and generating dummy inputs for the model.

    The '_init_weights' method initializes the weights of the model's cells based on the specified standard deviation. 
    It handles different cell types such as 'nn.Dense' and 'nn.Embedding', setting their weights and biases accordingly. 
    For 'nn.Embedding' cells, it also handles padding indices to ensure proper weight initialization.

    The 'dummy_inputs' property returns a dictionary of dummy inputs for the model, including an attention mask and input IDs. 
    It uses the specified pad token ID to generate the inputs and handles padding for the input sequences.

    This class provides essential functionality for initializing model weights and generating dummy inputs, 
    making it suitable for use in natural language processing tasks.
    """
    config_class = BigBirdPegasusConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BigBirdPegasusEncoderLayer", "BigBirdPegasusDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, cell):
        """Initialize the weights"""
        std = self.config.init_std
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(std),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))

    @property
    def dummy_inputs(self):
        """
        Retrieves dummy inputs for the 'BigBirdPegasusPreTrainedModel' class.

        Args:
            self: The current instance of the class (BigBirdPegasusPreTrainedModel).

        Returns:
            dict: 
                A dictionary containing dummy inputs for the model, with the following keys:
                
                - 'attention_mask': A tensor representing the attention mask. 
                It is obtained by applying the 'ne' (not equal) operation on the 'input_ids' tensor, 
                with the padding token as the argument.
                - 'input_ids': A tensor representing the input IDs for the model. 
                It contains two rows, where each row represents a different sequence. 
                The first row consists of the values [0, 6, 10, 4, 2],
                and the second row consists of the values [0, 8, 12, 2, pad_token], 
                where 'pad_token' is the padding token ID obtained from the model's configuration.

        Raises:
            None.
        """
        pad_token = self.config.pad_token_id
        input_ids = mindspore.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class BigBirdPegasusEncoder(BigBirdPegasusPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BigBirdPegasusEncoderLayer`].

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes a new instance of the BigBirdPegasusEncoder class.

        Args:
            self: The object itself.
            config (BigBirdPegasusConfig): The configuration object containing the settings for the encoder.
            embed_tokens (Optional[nn.Embedding]): The optional embedding tokens for the encoder. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.attention_type = config.attention_type
        self.block_size = config.block_size

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.CellList([BigBirdPegasusEncoderLayer(config, seed=i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        attention_mask = attention_mask.long()

        # in order to use block_sparse attention, sequence_length has to be at least
        # bigger than all global attentions: 2 * block_size
        # + sliding tokens: 3 * block_size
        # + random tokens: 2 * num_random_blocks * block_size
        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == "block_sparse" and input_shape[1] <= max_tokens_to_attend:
            # change attention_type from block_sparse to original_full
            sequence_length = input_shape[1]
            logger.warning(
                "Attention type 'block_sparse' is not possible if sequence_length: "
                f"{sequence_length} <= num global tokens: 2 * config.block_size "
                "+ min. num sliding tokens: 3 * config.block_size "
                "+ config.num_random_blocks * config.block_size "
                "+ additional buffer: config.num_random_blocks * config.block_size "
                f"= {max_tokens_to_attend} with config.block_size "
                f"= {self.config.block_size}, config.num_random_blocks "
                f"= {self.config.num_random_blocks}. "
                "Changing attention type to 'original_full'..."
            )
            self.set_attention_type("original_full")

        if self.attention_type == "block_sparse":
            padding_len, hidden_states, attention_mask = self._pad_to_block_size(hidden_states, attention_mask)
        else:
            padding_len = 0

        # expand attention_mask
        if self.attention_type == "original_full":
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
            blocked_encoder_mask = band_mask = from_mask = to_mask = None
        elif self.attention_type == "block_sparse":
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
                attention_mask, self.block_size
            )
            attention_mask = None
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.attention_type}"
            )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.shape[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.shape[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        band_mask,
                        from_mask,
                        to_mask,
                        blocked_encoder_mask,
                        blocked_encoder_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        band_mask=band_mask,
                        from_mask=from_mask,
                        to_mask=to_mask,
                        from_blocked_mask=blocked_encoder_mask,
                        to_blocked_mask=blocked_encoder_mask,
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm_embedding(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.shape[1]
            hidden_states = hidden_states[:, :-padding_len]

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        self.encoder_o = hidden_states

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

    def set_attention_type(self, value: str):
        '''
        Sets the attention type for the BigBirdPegasusEncoder.

        Args:
            self (BigBirdPegasusEncoder): The instance of the BigBirdPegasusEncoder.
            value (str): The attention type to be set. It can only be set to either 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the provided attention type is not 'original_full' or 'block_sparse'.
        '''
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layers:
            layer.set_attention_type(value)

    @staticmethod  # Copied from transformers.models.big_bird.modeling_big_bird.BigBirdModel.create_masks_for_block_sparse_attn
    def create_masks_for_block_sparse_attn(attention_mask: mindspore.Tensor, block_size: int):
        """
        Method: create_masks_for_block_sparse_attn

        Description:
            This method creates various types of masks for block-sparse attention in the BigBirdPegasusEncoder class. 
            It takes an attention_mask and the block_size as input and returns multiple masks required for
            block-sparse attention.

        Args:
            attention_mask (mindspore.Tensor): A 2D tensor representing the attention mask for the input sequence.
            block_size (int): The size of each block for block-sparse attention.

        Returns:
            None

        Raises:
            ValueError: If the sequence length is not a multiple of the block size.

        Note:
            This method is a static method.

        """
        batch_size, seq_length = attention_mask.shape
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            exp_blocked_to_pad = ops.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], axis=2
            )
            band_mask = ops.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask = band_mask.unsqueeze(1)
            return band_mask

        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def _pad_to_block_size(self, hidden_states: mindspore.Tensor, attention_mask: mindspore.Tensor):
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        # padding
        block_size = self.config.block_size
        batch_size, seq_len = hidden_states.shape[:2]

        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )
            pad_id = self.config.pad_token_id
            input_ids_padding = ops.ones((batch_size, padding_len), dtype=mindspore.int64) * pad_id
            inputs_embeds_padding = self.embed_tokens(input_ids_padding)
            hidden_states = ops.cat([hidden_states, inputs_embeds_padding], axis=-2)

            attention_mask = ops.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens

        return padding_len, hidden_states, attention_mask


class BigBirdPegasusDecoder(BigBirdPegasusPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BigBirdPegasusDecoderLayer`]

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        """
        Initializes an instance of the BigBirdPegasusDecoder class.

        Args:
            self: The instance of the class.
            config (BigBirdPegasusConfig): The configuration object containing various settings for the decoder.
            embed_tokens (Optional[nn.Embedding]): Optional pre-trained embedding matrix for the tokens. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.CellList([BigBirdPegasusDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the BigBirdPegasusDecoder.

        Args:
            self: An instance of the BigBirdPegasusDecoder class.

        Returns:
            embed_tokens: This method returns the input embeddings for the decoder.

        Raises:
            None.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """Set the input embeddings for the BigBirdPegasusDecoder.

        Args:
            self (BigBirdPegasusDecoder): The instance of the BigBirdPegasusDecoder class.
            value: The input embeddings to be set for the decoder. It should be a compatible data type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = value

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.
                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed
                or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions

        hidden_states = ops.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.shape[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.shape[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = ops.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layernorm_embedding(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BigBirdPegasusModel(BigBirdPegasusPreTrainedModel):

    """
    This class represents a BigBirdPegasus model for sequence-to-sequence tasks. It is a variant of the BigBird model
    that is specifically designed for text generation tasks using the Pegasus architecture.

    The BigBirdPegasusModel class inherits from the BigBirdPegasusPreTrainedModel class, which is a base class
    for all pre-trained BigBirdPegasus models. It provides common methods and attributes for loading
    and saving models.

    Methods:
        __init__(self, config: BigBirdPegasusConfig): Initializes the BigBirdPegasusModel instance with a given configuration.
        get_input_embeddings(self): Returns the shared input embeddings used by the model.
        set_input_embeddings(self, value): Sets the shared input embeddings of the model.
        _tie_weights(self): Ties the weights of the encoder and decoder embedding layers if specified in the configuration.
        get_encoder(self): Returns the encoder module of the model.
        get_decoder(self): Returns the decoder module of the model.
        construct(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, head_mask,
            decoder_head_mask, cross_attn_head_mask, encoder_outputs, past_key_values, inputs_embeds,
            decoder_inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
            Constructs the model by performing encoding and decoding operations on the input sequence.

    Please refer to the documentation of the individual methods for more details on their parameters and return values.

    Note:
        This docstring is generated based on the provided code snippet and may not include all the class attributes,
        methods, and their details. Please refer to the source code or official documentation for
        complete information.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BigBirdPegasusConfig):
        """
        Initializes a new instance of the BigBirdPegasusModel class.

        Args:
            self: The current instance of the class.
            config (BigBirdPegasusConfig):
                The configuration object containing various settings for the model.

                - `pad_token_id` (int): The index of the padding token in the vocabulary.
                - `vocab_size` (int): The size of the vocabulary.
                - `d_model` (int): The dimensionality of the model's hidden states.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BigBirdPegasusEncoder(config, self.shared)
        self.decoder = BigBirdPegasusDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings for the BigBirdPegasusModel.

        Args:
            self: An instance of the BigBirdPegasusModel class.

        Returns:
            None.

        Raises:
            None
        """
        return self.shared

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the BigBirdPegasusModel.

        Args:
            self (BigBirdPegasusModel): The instance of the BigBirdPegasusModel class.
            value (object): The input embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def _tie_weights(self):
        """
        Ties the weights of the encoder and decoder token embeddings if the tie_word_embeddings flag is set to True.

        Args:
            self (BigBirdPegasusModel): An instance of the BigBirdPegasusModel class.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method is used to ensure that the encoder and decoder token embeddings share the same weights
            when the tie_word_embeddings flag is set to True.
            It helps in reducing the number of parameters in the model and improves training efficiency.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        """
        This method returns the encoder associated with the BigBirdPegasusModel.

        Args:
            self: The instance of the BigBirdPegasusModel class.

        Returns:
            None: This method returns the encoder associated with the BigBirdPegasusModel.

        Raises:
            This method does not raise any exceptions.
        """
        return self.encoder

    def get_decoder(self):
        """
        Returns the decoder of the BigBirdPegasusModel.

        Args:
            self: An instance of the BigBirdPegasusModel class.

        Returns:
            decoder: This method returns the decoder of the BigBirdPegasusModel.
                The decoder is responsible for decoding the input and generating the output.

        Raises:
            None.
        """
        return self.decoder

    # Copied from transformers.models.bart.modeling_bart.BartModel.forward with Bart->BigBirdPegasus
    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        """
        Constructs the BigBirdPegasusModel.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor, optional): The input token IDs of shape (batch_size, sequence_length).
                Defaults to None.
            attention_mask (mindspore.Tensor, optional): The attention mask of shape (batch_size, sequence_length).
                Defaults to None.
            decoder_input_ids (mindspore.Tensor, optional): The decoder input token IDs of shape
                (batch_size, sequence_length). Defaults to None.
            decoder_attention_mask (mindspore.Tensor, optional): The decoder attention mask of shape
                (batch_size, sequence_length). Defaults to None.
            head_mask (mindspore.Tensor, optional): The head mask tensor of shape (num_layers, num_heads) or
                (num_layers, num_heads, sequence_length, sequence_length). Defaults to None.
            decoder_head_mask (mindspore.Tensor, optional): The decoder head mask tensor of shape
                (num_layers, num_heads) or (num_layers, num_heads, sequence_length, sequence_length). Defaults to None.
            cross_attn_head_mask (mindspore.Tensor, optional): The cross-attention head mask tensor of shape
                (num_layers, num_heads) or (num_layers, num_heads, sequence_length, sequence_length). Defaults to None.
            encoder_outputs (List[mindspore.Tensor], optional): The encoder outputs of shape
                [(batch_size, sequence_length, hidden_size), ...]. Defaults to None.
            past_key_values (List[mindspore.Tensor], optional): The past key values of shape
                [(batch_size, num_heads, past_sequence_length, hidden_size), ...]. Defaults to None.
            inputs_embeds (mindspore.Tensor, optional): The embedded inputs tensor of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            decoder_inputs_embeds (mindspore.Tensor, optional): The embedded decoder inputs tensor of shape
                (batch_size, sequence_length, hidden_size). Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Whether to return as a dictionary. Defaults to None.

        Returns:
            Union[Tuple, Seq2SeqModelOutput]: A tuple or a Seq2SeqModelOutput containing the model outputs.

        Raises:
            ValueError: If no `decoder_input_ids` or `decoder_inputs_embeds` are passed and `input_ids` is None.
        """
        # different to other models, BigBirdPegasus automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
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


# Copied from transformers.models.bart.modeling_bart.BartForConditionalGeneration with Bart->BigBirdPegasus, BART->BIGBIRD_PEGASUS
class BigBirdPegasusForConditionalGeneration(BigBirdPegasusPreTrainedModel):

    """
    This class represents a conditional generation model based on BigBirdPegasus.
    It is a subclass of BigBirdPegasusPreTrainedModel.

    The BigBirdPegasusForConditionalGeneration class extends the functionality of its parent class by adding methods
    for conditional generation tasks, such as generating text given a prompt or a set of input tokens.

    Methods:
        __init__: Initializes the model with the given configuration.
        get_encoder: Returns the encoder of the model.
        get_decoder: Returns the decoder of the model.
        resize_token_embeddings: Resizes the token embeddings of the model.
        _resize_final_logits_bias: Resizes the bias tensor used for final logits.
        get_output_embeddings: Returns the output embedding layer of the model.
        set_output_embeddings: Sets the output embedding layer of the model.
        construct: Constructs the model for conditional generation tasks.
        prepare_inputs_for_generation: Prepares the input tensors for generation.
        prepare_decoder_input_ids_from_labels: Prepares the decoder input IDs from the given labels.
        _reorder_cache: Reorders the past key values for beam search.

    The BigBirdPegasusForConditionalGeneration class is designed to be used for various conditional generation tasks,
    such as text generation, text completion, and text summarization.
    """
    base_model_prefix = "model"
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = ["final_logits_bias"]

    def __init__(self, config: BigBirdPegasusConfig):
        """
        Initializes an instance of the BigBirdPegasusForConditionalGeneration class.

        Args:
            self: The instance of the class.
            config (BigBirdPegasusConfig): An instance of BigBirdPegasusConfig containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.model = BigBirdPegasusModel(config)
        self.final_logits_bias = ops.zeros((1, self.model.shared.vocab_size))
        self.lm_head = nn.Dense(config.d_model, self.model.shared.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Retrieve the encoder component from the model.

        Args:
            self: An instance of the BigBirdPegasusForConditionalGeneration class.

        Returns:
            None: The method returns None as the encoder component.

        Raises:
            None.
        """
        return self.model.get_encoder()

    def get_decoder(self):
        """
        This method returns the decoder from the BigBirdPegasusForConditionalGeneration model.

        Args:
            self (BigBirdPegasusForConditionalGeneration): The instance of the BigBirdPegasusForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        """
        Resize the token embeddings for the model.

        Args:
            self: The instance of the BigBirdPegasusForConditionalGeneration class.
            new_num_tokens (int): The new number of tokens to resize the embeddings to.
            pad_to_multiple_of (Optional[int]): A value to pad the new number of tokens to a multiple of, if specified.

        Returns:
            nn.Embedding: The resized token embeddings of type nn.Embedding.

        Raises:
            ValueError: If new_num_tokens is not a positive integer.
            TypeError: If new_num_tokens is not an integer.
            TypeError: If pad_to_multiple_of is not an integer.
        """
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        """
        Resizes the final logits bias tensor in the BigBirdPegasusForConditionalGeneration class.

        Args:
            self (BigBirdPegasusForConditionalGeneration): An instance of the BigBirdPegasusForConditionalGeneration class.
            new_num_tokens (int): The desired number of tokens for the resized final logits bias tensor.

        Returns:
            None:
                This method modifies the `final_logits_bias` attribute of the BigBirdPegasusForConditionalGeneration
                instance.

        Raises:
            None.

        The method `_resize_final_logits_bias` resizes the `final_logits_bias` tensor based on the provided
        `new_num_tokens` parameter. If the `new_num_tokens` is less than or equal to the current number of
        tokens in `final_logits_bias`, the tensor is sliced to retain only the first `new_num_tokens` columns.
        Otherwise, extra bias columns are added to the tensor using `zeros` and `cat` operations.

        Note:
            This method directly modifies the `final_logits_bias` attribute of the
            BigBirdPegasusForConditionalGeneration instance.
        """
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = ops.zeros((1, new_num_tokens - old_num_tokens))
            new_bias = ops.cat([self.final_logits_bias, extra_bias], axis=1)
        self.final_logits_bias = new_bias

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the BigBirdPegasus model.

        Args:
            self (BigBirdPegasusForConditionalGeneration):
                The instance of the BigBirdPegasusForConditionalGeneration class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the BigBirdPegasusForConditionalGeneration model.

        Args:
            self (BigBirdPegasusForConditionalGeneration): The instance of the BigBirdPegasusForConditionalGeneration class.
            new_embeddings (torch.nn.Embedding): The new output embeddings to be set for the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            `Union[Tuple, Seq2SeqLMOutput]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        This method prepares inputs for generation in the BigBirdPegasusForConditionalGeneration class.

        Args:
            self: The instance of the class.
            decoder_input_ids (Tensor): The input tensor for the decoder.
            past_key_values (tuple, optional): A tuple of past key values for the model's autoregressive decoding. Defaults to None.
            attention_mask (Tensor, optional): The attention mask for the input. Defaults to None.
            decoder_attention_mask (Tensor, optional): The attention mask for the decoder input. Defaults to None.
            head_mask (Tensor, optional): The mask for the attention heads. Defaults to None.
            decoder_head_mask (Tensor, optional): The mask for the decoder's attention heads. Defaults to None.
            cross_attn_head_mask (Tensor, optional): The mask for cross-attention heads. Defaults to None.
            use_cache (bool, optional): Whether to use caching for the model. Defaults to None.
            encoder_outputs (Tensor, optional): The outputs from the encoder. Defaults to None.

        Returns:
            dict: A dictionary containing
                'input_ids', 'encoder_outputs', 'past_key_values', 'decoder_input_ids', 'attention_mask',
                'decoder_attention_mask', 'head_mask', 'decoder_head_mask', 'cross_attn_head_mask', and 'use_cache'.

        Raises:
            None
        """
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: mindspore.Tensor):
        """
        Prepare decoder input IDs from labels.

        This method takes two parameters: self, labels.

        Args:
            self (BigBirdPegasusForConditionalGeneration): An instance of the BigBirdPegasusForConditionalGeneration class.
            labels (mindspore.Tensor): The labels tensor representing the ground truth sequence.

        Returns:
            None.

        Raises:
            None.
        """
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the past key values based on the provided beam index.

        Args:
            past_key_values (tuple): A tuple containing past key values for each layer.
                Each element in the tuple is a tuple representing the past key values for a layer.
            beam_idx (Tensor): A tensor containing the indices of the beams to reorder the past key values.

        Returns:
            tuple: A tuple of reordered past key values,
                where each element in the tuple represents the reordered past key values for a layer.

        Raises:
            None
        """
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


class BigBirdPegasusForSequenceClassification(BigBirdPegasusPreTrainedModel):

    """
    This class represents a BigBirdPegasus model for sequence classification. It inherits from
    BigBirdPegasusPreTrainedModel and includes methods for model initialization and construction of the sequence
    classifier.
    The construct method takes various input parameters for decoding and attention masks, and returns the sequence
    classifier output including logits and optional loss.
    The class also handles different problem types such as regression, single label classification, and multi-label
    classification.
    Additionally, it ensures consistency in the number of <eos> tokens for all examples.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BigBirdPegasusConfig, **kwargs):
        """
        Initializes a new instance of the BigBirdPegasusForSequenceClassification class.

        Args:
            self: The object itself.
            config (BigBirdPegasusConfig): The configuration for the BigBirdPegasus model.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config, **kwargs)
        self.model = BigBirdPegasusModel(config)
        self.classification_head = BigBirdPegasusClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bart.modeling_bart.BartForSequenceClassification.forward
    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id)

        if len(ops.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask].view(hidden_states.shape[0], -1, hidden_states.shape[-1])[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.config.num_labels == 1:
                    loss = ops.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = ops.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = ops.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = ops.binary_cross_entropy_with_logits(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BigBirdPegasusForQuestionAnswering(BigBirdPegasusPreTrainedModel):

    """
    This class represents a BigBirdPegasus model for question answering tasks. It is designed to perform question
    answering using the BigBirdPegasus architecture.

    The class includes methods for initialization and constructing the model for question answering tasks.
    It inherits from the BigBirdPegasusPreTrainedModel class and utilizes a sequence-to-sequence model for
    processing input and generating output.

    The __init__ method initializes the model with configuration settings, including setting the number of labels for classification.
    The construct method constructs the model for question answering by processing input tensors and generating
    start and end position logits for the answer span.

    The class provides functionality for computing the token classification loss based on the start and end positions
    of the labelled span.
    It handles the calculation of loss and returns the output in the desired format based on the return_dict parameter.

    For detailed information on the methods and parameters of this class, please refer to the class code and method
    docstrings.
    """
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the BigBirdPegasusForQuestionAnswering class.

        Args:
            self: The object instance itself.
            config:
                An object containing configuration settings for the model.

                - Type: Any
                - Purpose: Contains configuration settings for the model initialization.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BigBirdPegasusModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bart.modeling_bart.BartForQuestionAnswering.forward
    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        decoder_input_ids: Optional[mindspore.Tensor] = None,
        decoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        decoder_head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        encoder_outputs: Optional[List[mindspore.Tensor]] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        decoder_inputs_embeds: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
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

            start_loss = ops.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = ops.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


# Copied from transformers.models.pegasus.modeling_pegasus.PegasusDecoderWrapper with Pegasus->BigBirdPegasus
class BigBirdPegasusDecoderWrapper(BigBirdPegasusPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """
    def __init__(self, config):
        """
        Initialize the BigBirdPegasusDecoderWrapper class.

        Args:
            self (object): The instance of the class.
            config (dict): A dictionary containing configuration parameters for the decoder.
                Must include the necessary settings for initializing the BigBirdPegasusDecoder.

        Returns:
            None.

        Raises:
            None: However, if the configuration parameters are invalid or if there are issues during the
                initialization of the BigBirdPegasusDecoder, exceptions related to those
                actions may be raised during the execution of this method.
        """
        super().__init__(config)
        self.decoder = BigBirdPegasusDecoder(config)

    def construct(self, *args, **kwargs):
        """
        Method to construct a decoder using the BigBirdPegasusDecoderWrapper.

        Args:
            self (object): The instance of the BigBirdPegasusDecoderWrapper class.
                This parameter is required to access the methods and attributes of the class.

        Returns:
            None: This method does not return any value explicitly. It delegates the construction
                to the decoder specified in self.decoder.

        Raises:
            None:
                However, exceptions may be raised by the decoder method called within this construct method.
        """
        return self.decoder(*args, **kwargs)


class BigBirdPegasusForCausalLM(BigBirdPegasusPreTrainedModel):

    """
    The `BigBirdPegasusForCausalLM` class represents a BigBird Pegasus model for causal language modeling tasks.
    It inherits from the `BigBirdPegasusPreTrainedModel` class.

    The class initializes the model with the provided configuration and defines methods for getting and
    setting input and output embeddings, setting the decoder, and constructing the model for generation.
    Additionally, it provides methods for preparing inputs for generation and reordering cache for beam search.

    The `construct` method processes the input data for the model and returns the model outputs.
    The `prepare_inputs_for_generation` method prepares input data for generation, and the `_reorder_cache` method
    reorders the cache for beam search.

    The class also includes detailed documentation for the input and output parameters of the `construct` method,
    providing information on the usage and functionality of each parameter.

    Example usage of the `BigBirdPegasusForCausalLM` class is provided in the docstring, demonstrating how to
    initialize the model and generate predictions.

    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        """
        Initializes the BigBirdPegasusForCausalLM class.

        Args:
            self: The instance of the class.
            config: A configuration object containing the model's configuration parameters.
                It is expected to be a dictionary or an object that can be deep-copied.
                It should include the necessary parameters for initializing the model.
                The 'is_decoder' and 'is_encoder_decoder' attributes will be modified within this method.

        Returns:
            None.

        Raises:
            AttributeError: If the 'config' parameter is missing required attributes.
            TypeError: If the 'config' parameter is not of the expected type.
            ValueError: If the 'config' parameter contains invalid values.
        """
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        self.model = BigBirdPegasusDecoderWrapper(config)

        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
            Returns the input embeddings used by the BigBirdPegasusForCausalLM model's decoder.

        Args:
            self (object): The instance of the BigBirdPegasusForCausalLM class.

        Returns:
            None:
                This method returns None as it directly retrieves and returns the input embeddings from the decoder of the model.

        Raises:
            None
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BigBirdPegasusForCausalLM model.

        Args:
            self (BigBirdPegasusForCausalLM): The instance of the BigBirdPegasusForCausalLM class.
            value: The input embeddings to be set for the model. This should be of type torch.Tensor.

        Returns:
            None.

        Raises:
            None.

        This method sets the input embeddings for the BigBirdPegasusForCausalLM model.
        It assigns the given 'value' to the 'embed_tokens' attribute of the decoder in the model.
        The 'embed_tokens' attribute represents the embedding layer used for token inputs in the decoder.
        By setting the input embeddings, the model will use the provided embeddings during inference and decoding.

        Note:
            It is important to ensure that the 'value' parameter is a tensor of shape (vocab_size, embedding_dim)
            where 'vocab_size' is the size of the vocabulary and 'embedding_dim' is the dimensionality of the
            embedding space.
        """
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the BigBirdPegasusForCausalLM model.

        Args:
            self (BigBirdPegasusForCausalLM): The instance of the BigBirdPegasusForCausalLM class.
                This parameter refers to the current instance of the model.

        Returns:
            lm_head: The method returns the 'lm_head' attribute of the model, which represents the output embeddings.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings for the BigBirdPegasusForCausalLM model.

        Args:
            self (BigBirdPegasusForCausalLM): The instance of the BigBirdPegasusForCausalLM class.
            new_embeddings (Any): The new embeddings to be set for the output layer.

        Returns:
            None:
                This method updates the lm_head attribute of the BigBirdPegasusForCausalLM instance with the new embeddings.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        Sets the decoder for the BigBirdPegasusForCausalLM model.

        Args:
            self (BigBirdPegasusForCausalLM): The instance of the BigBirdPegasusForCausalLM class.
            decoder: The decoder object to be set for the model. It should be of the appropriate type.

        Returns:
            None.

        Raises:
            None.
        """
        self.model.decoder = decoder

    def get_decoder(self):
        """
        Retrieve the decoder component from the BigBirdPegasusForCausalLM model.

        Args:
            self (object): Instance of the BigBirdPegasusForCausalLM class.
                This parameter is required to access the model attributes.

        Returns:
            NoneType: This method returns the decoder component of the model.
                The decoder is responsible for generating the output sequences.

        Raises:
            No specific exceptions are raised by this method.
        """
        return self.model.decoder

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        cross_attn_head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (`mindspore.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(mindspore.Tensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, BigBirdPegasusForCausalLM
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
            >>> model = BigBirdPegasusForCausalLM.from_pretrained(
            ...     "google/bigbird-pegasus-large-arxiv", add_cross_attention=False
            ... )
            >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> logits = outputs.logits
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self (BigBirdPegasusForCausalLM): The instance of the BigBirdPegasusForCausalLM class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
            past_key_values (Optional[Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]):
                Optional tuple of past key and value tensors.
            attention_mask (Optional[torch.Tensor]): The attention mask tensor of shape (batch_size, sequence_length).
                If not provided, it will be initialized with ones.
            use_cache (Optional[bool]): Whether to use cache for faster decoding.

        Returns:
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor], bool]]:
                A dictionary containing the following items:

                - 'input_ids' (torch.Tensor): The input tensor.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                - 'past_key_values' (Optional[Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]):
                Optional tuple of past key and value tensors.
                - 'use_cache' (Optional[bool]): Whether to use cache for faster decoding.
        
        Raises:
            None.
        """
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            input_ids = input_ids[:, -1:]
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Method to reorder the cache values according to the given beam index.
        
        Args:
            past_key_values (tuple): A tuple containing the past key values for each layer.
            beam_idx (Tensor): A tensor representing the indices of beams.
        
        Returns:
            None:
                This method does not return any value, but it updates the past key values based on the specified beam index.
        
        Raises:
            None
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

__all__ = [
    "BigBirdPegasusForCausalLM",
    "BigBirdPegasusForConditionalGeneration",
    "BigBirdPegasusForQuestionAnswering",
    "BigBirdPegasusForSequenceClassification",
    "BigBirdPegasusModel",
    "BigBirdPegasusPreTrainedModel",
]
