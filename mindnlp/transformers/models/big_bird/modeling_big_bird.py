# coding=utf-8
# Copyright 2021 Google Research and The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore BigBird model."""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import (
    ModelOutput,
    logging,
)
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward
from .configuration_big_bird import BigBirdConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/bigbird-roberta-base"
_CONFIG_FOR_DOC = "BigBirdConfig"

BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/bigbird-roberta-base",
    "google/bigbird-roberta-large",
    "google/bigbird-base-trivia-itc",
    # See all BigBird models at https://hf-mirror.com/models?filter=big_bird
]


class BigBirdEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        """
        Initializes an instance of the BigBirdEmbeddings class.
        
        Args:
            self: The object instance.
            config: An object that contains the configuration parameters for the embeddings.
            vocab_size (int): The size of the vocabulary.
            hidden_size (int): The size of the hidden layers.
            pad_token_id (int): The ID of the padding token.
            max_position_embeddings (int): The maximum number of positions for the position embeddings.
            type_vocab_size (int): The size of the token type vocabulary.
            layer_norm_eps (float): The epsilon value for layer normalization.
            hidden_dropout_prob (float): The dropout probability for the hidden layers.
            position_embedding_type (str, optional): The type of position embedding. Defaults to 'absolute'.
            rescale_embeddings (bool): Whether to rescale the embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)
        # End copy

        self.rescale_embeddings = config.rescale_embeddings
        self.hidden_size = config.hidden_size

    def construct(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        '''
        Constructs the embeddings for the BigBird model.

        Args:
            self (BigBirdEmbeddings): An instance of the BigBirdEmbeddings class.
            input_ids (Optional[Tensor]): A tensor of shape (batch_size, sequence_length)
                containing the input token IDs. Default is None.
            token_type_ids (Optional[Tensor]): A tensor of shape (batch_size, sequence_length)
                containing the token type IDs. Default is None.
            position_ids (Optional[Tensor]): A tensor of shape (batch_size, sequence_length)
                containing the position IDs. Default is None.
            inputs_embeds (Optional[Tensor]): A tensor of shape (batch_size, sequence_length, hidden_size)
                containing the input embeddings. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            Tensor: A tensor of shape (batch_size, sequence_length, hidden_size) containing the constructed embeddings.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None.
            ValueError: If input_ids is not None and inputs_embeds is not None.
            ValueError: If input_ids is not None and token_type_ids is None while self.token_type_ids is None.
            ValueError: If inputs_embeds is not None and token_type_ids is not None.
            ValueError: If inputs_embeds is not None and position_ids is not None.
        '''
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.rescale_embeddings:
            inputs_embeds = inputs_embeds * (self.hidden_size**0.5)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class BigBirdSelfAttention(nn.Cell):

    """
    Represents a self-attention mechanism implementation based on the BigBird model.
    This class provides methods for performing self-attention computations in neural networks.

    The 'BigBirdSelfAttention' class inherits from nn.Cell and implements functionalities for self-attention mechanisms.
    It includes methods for initializing the attention mechanism, swapping axes for attention scores calculations,
    and constructing the attention mechanism.

    Attributes:
        num_attention_heads: Number of attention heads in the self-attention mechanism.
        attention_head_size: Size of each attention head in the self-attention mechanism.
        all_head_size: Total size of all attention heads combined.
        query: Dense layer for query transformations.
        key: Dense layer for key transformations.
        value: Dense layer for value transformations.
        dropout: Dropout layer for attention probabilities.
        is_decoder: Flag indicating if the attention mechanism is used in a decoder context.

    Methods:
        swapaxes_for_scores(x): Swaps axes in the input tensor for calculating attention scores.
        construct(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions):
            Constructs the self-attention mechanism with given inputs and parameters.

    Raises:
        ValueError: When the hidden size is not a multiple of the number of attention heads.

    Note:
        This class is specifically designed for implementing self-attention mechanisms in neural networks based on the BigBird model.
    """
    def __init__(self, config):
        """
        Initializes the BigBirdSelfAttention class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the self-attention mechanism.
            hidden_size (int): The size of the hidden layer.
            num_attention_heads (int): The number of attention heads to use.
            embedding_size (int): The size of the embedding. Optional.
            use_bias (bool): Flag indicating whether to use bias in the Dense layers.
            attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
            is_decoder (bool): Flag indicating if the self-attention mechanism is used in a decoder.

        Returns:
            None.

        Raises:
            ValueError:
                If the hidden size is not a multiple of the number of attention heads
                or if the 'embedding_size' attribute is missing in the config object.
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
        Method to rearrange the axes of the input tensor for BigBird self-attention scoring.

        Args:
            self (BigBirdSelfAttention): An instance of the BigBirdSelfAttention class.
            x (torch.Tensor): The input tensor to be reshaped and permuted.
                It should have a shape of (batch_size, seq_length, hidden_size).

        Returns:
            torch.Tensor: The reshaped and permuted tensor with the axes rearranged for scoring purposes.
            The returned tensor will have a shape of (batch_size, num_attention_heads, seq_length, attention_head_size).

        Raises:
            None.
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
        This method constructs the self-attention mechanism in the BigBird model.

        Args:
            self (object): The instance of the BigBirdSelfAttention class.
            hidden_states (tensor): The input tensor representing the hidden states of the model.
            attention_mask (tensor, optional): A tensor specifying which elements should be attended to and which should not. Defaults to None.
            head_mask (tensor, optional): A tensor used to mask certain heads of the attention mechanism. Defaults to None.
            encoder_hidden_states (tensor, optional): The hidden states of the encoder if cross-attention is required. Defaults to None.
            encoder_attention_mask (tensor, optional): The attention mask to be applied in case of cross-attention. Defaults to None.
            past_key_value (tuple, optional): A tuple containing the previous key and value tensors for optimization purposes. Defaults to None.
            output_attentions (bool): A flag indicating whether to output attention probabilities. Defaults to False.

        Returns:
            tuple:
                A tuple containing the context layer tensor and optionally the attention probabilities tensor
                or past key-value tuple.

        Raises:
            ValueError: If the provided dimensions of tensors are incompatible.
            NotImplementedError: If the method is not implemented for a specific use case.
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
            # Apply the attention mask is (precomputed for all layers in BigBirdModel forward() function)
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


class BigBirdBlockSparseAttention(nn.Cell):

    '''
    """
    A class representing the BigBirdBlockSparseAttention, which is used for implementing block-sparse attention
    mechanism in neural networks.

    This class inherits from nn.Cell and provides methods for constructing the block-sparse attention mechanism
    using the BigBird algorithm.
    It includes methods for initializing the attention mechanism, creating random attention masks, and performing
    block-sparse attention operations.

    Attributes:
        config: A configuration object used to set the parameters for the attention mechanism.
        seed: An optional parameter for setting the random seed.

    Methods:
        __init__: Initializes the BigBirdBlockSparseAttention with the given configuration and seed.
        swapaxes_for_scores: Swaps the axes of the input tensor for calculating attention scores.
        construct: Constructs the block-sparse attention mechanism using the given input tensors and masks.
        ms_bmm_nd: Performs fast nd matrix multiplication for the input tensors.
        ms_bmm_nd_swapaxes: Performs fast nd matrix multiplication with swapped axes for the input tensors.
        bigbird_block_sparse_attention: Implements the block-sparse attention mechanism using the BigBird algorithm.
        ms_gather_b2: Gathers elements from the input tensor based on the provided indices.
        _create_rand_mask_from_inputs: Creates a 3D attention mask from a 2D tensor mask.
        _get_rand_attn_plan: Provides the plan for placing random attention within the sequence.
        _bigbird_block_rand_mask: Creates an adjacency list of random attention for the given sequence lengths and block sizes.
        _bigbird_block_rand_mask_with_head: Creates an adjacency list of random attention with multiple heads.
        _get_single_block_row_attention: Retrieves random row attention for a single row block.

    """
    '''
    def __init__(self, config, seed=None):
        """
        Initializes an instance of the BigBirdBlockSparseAttention class.

        Args:
            self: The instance of the class.
            config:
                A configuration object that contains various settings for the attention mechanism.
                It should have the following attributes:

                - max_position_embeddings (int): The maximum sequence length.
                - hidden_size (int): The size of the hidden state.
                - num_attention_heads (int): The number of attention heads.
                - num_random_blocks (int): The number of random blocks.
                - block_size (int): The block size for block sparse attention.
                - use_bias (bool): Whether to include bias in the dense layers.

            seed (Optional[int]): The random seed for reproducibility. Default is None.

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
        This method 'swapaxes_for_scores' is defined in the class 'BigBirdBlockSparseAttention'.
        It reshapes and transposes the input tensor 'x' to match the specified dimensions for attention calculations.

        Args:
            self (BigBirdBlockSparseAttention): The instance of the BigBirdBlockSparseAttention class.
            x (Tensor): The input tensor to be reshaped and transposed. It is expected to be a multi-dimensional tensor.

        Returns:
            Tensor: The reshaped and transposed tensor based on the input 'x',
                with dimensions modified to match the requirements for attention calculations.

        Raises:
            ValueError: If the dimensions of the input tensor 'x' do not align with the expected shape
                for reshaping and transposition.
            RuntimeError: If any runtime error occurs during the reshaping and transposition process.
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
        The 'construct' method in the 'BigBirdBlockSparseAttention' class constructs the context layer
        using BigBird block sparse attention mechanism.

        Args:
            self: The instance of the class.
            hidden_states (torch.Tensor): The input hidden states of shape (batch_size, seqlen, _).
            band_mask (torch.Tensor, optional): The band mask to be applied during attention computation.
            from_mask (torch.Tensor, optional): Mask for 'from' sequence to be applied during attention computation.
            to_mask (torch.Tensor, optional): Mask for 'to' sequence to be applied during attention computation.
            from_blocked_mask (torch.Tensor, optional): Mask for blocked 'from' sequence.
            to_blocked_mask (torch.Tensor, optional): Mask for blocked 'to' sequence.
            output_attentions (bool, optional): Flag indicating whether to output attention probabilities.

        Returns:
            outputs (tuple):
                A tuple containing the context layer and attention probabilities if 'output_attentions' is True,
                else only the context layer.

        Raises:
            ValueError:
                Raised if the query sided sequence length is not a multiple of the block size or if the key/value
                sided sequence length is not a multiple of the block size.
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
        This method, 'bigbird_block_sparse_attention', is defined in the class 'BigBirdBlockSparseAttention'
        and is used to perform block-sparse attention computation in the BigBird model.

        Args:
            self: Reference to the current instance of the class.
            query_layer: Tensor representing the query input for the attention mechanism.
            key_layer: Tensor representing the key input for the attention mechanism.
            value_layer: Tensor representing the value input for the attention mechanism.
            band_mask: Tensor representing the band mask for attention computation.
            from_mask: Tensor representing the mask for the 'from' sequence length.
            to_mask: Tensor representing the mask for the 'to' sequence length.
            from_blocked_mask: Tensor representing the blocked mask for the 'from' sequence.
            to_blocked_mask: Tensor representing the blocked mask for the 'to' sequence.
            n_heads: Integer, number of attention heads.
            n_rand_blocks: Integer, number of random blocks for BigBird attention.
            attention_head_size: Integer, size of each attention head.
            from_block_size: Integer, size of each block in the 'from' sequence.
            to_block_size: Integer, size of each block in the 'to' sequence.
            batch_size: Integer, size of the batch.
            from_seq_len: Integer, length of the 'from' sequence.
            to_seq_len: Integer, length of the 'to' sequence.
            seed: Integer, seed for random number generation.
            plan_from_length: Integer, length of the plan for random attention.
            plan_num_rand_blocks: Integer, number of random blocks in the plan.
            output_attentions: Boolean flag indicating whether to output attention weights.

        Returns:
            context_layer: Tensor containing the output of the block-sparse attention computation.
            attention_probs: Tensor containing the attention probabilities if 'output_attentions' is True, otherwise None.

        Raises:
            ValueError: Raised if the number of blocks in 'from' and 'to' sequences are not the same.
        """
        # BigBird block-sparse attention as suggested in paper

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
                bsz, n_heads, from_seq_len, to_seq_len, dtype=mindspore.float32
            )

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
                    # attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(
                    #     from_block_size, n_rand_blocks, to_block_size
                    # )
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
                        # attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = right_slice.view(
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
                    # attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(
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
        Performs a block sparse attention operation on the given parameters and indices in the
        BigBirdBlockSparseAttention class.

        Args:
            params (Tensor): A tensor representing the input parameters of shape
                (batch_size, sequence_length, num_params, hidden_dim).
            batch_size (int): The number of sequences in a batch.
            sequence_length (int): The length of each sequence.
            num_params (int): The number of parameters.
            hidden_dim (int): The dimensionality of the hidden representation.
            indices (Tensor): A tensor representing the indices for gathering the parameters of shape
                (batch_size, sequence_length, num_indices_x, num_indices_y).
            batch_size (int): The number of sequences in a batch.
            sequence_length (int): The length of each sequence.
            num_indices_x (int): The number of indices in the x-direction.
            num_indices_y (int): The number of indices in the y-direction.

        Returns:
            Tensor:
                A tensor representing the output of the block sparse attention operation of shape
                (batch_size, sequence_length, num_indices_x, num_indices_y, hidden_dim).

        Raises:
            ValueError: If the first two dimensions of params and indices are not identical.

        Note:
            - The first two dimensions of params and indices must be identical.
            - The output tensor has the same batch size, sequence length, and hidden dimension as the input params.
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


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BigBird
class BigBirdSelfOutput(nn.Cell):

    """
    This class represents the self-output layer for the BigBird model.
    It applies dense transformation, layer normalization, and dropout to the input hidden states, and then adds
    the input tensor to the normalized hidden states.

    This class inherits from nn.Cell and includes an __init__ method for initialization and a construct method for
    performing the self-output transformation.

    The __init__ method initializes the dense transformation, layer normalization, and dropout layers with the given
    configuration parameters.

    The construct method takes the input hidden states and input tensor as input, applies the dense transformation,
    dropout, layer normalization, and adds the input tensor to the normalized hidden states, and returns the resulting
    hidden states.

    """
    def __init__(self, config):
        """
        Initializes the BigBirdSelfOutput class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters.
            hidden_size (int): The size of the hidden layer.
            layer_norm_eps (float): The epsilon value for LayerNorm.
            hidden_dropout_prob (float): The dropout probability for hidden layers.

        Returns:
            None.

        Raises:
            ValueError: If the provided config does not contain the required parameters.
            TypeError: If the data types of the config parameters are incorrect.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the BigBird self-attention layer.

        Args:
            self (BigBirdSelfOutput): The instance of the BigBirdSelfOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor produced by the self-attention mechanism.
                Shape: (batch_size, sequence_length, hidden_size)
            input_tensor (mindspore.Tensor): The input tensor to the self-attention layer.
                Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            mindspore.Tensor: The output tensor of the self-attention layer after processing the hidden_states and input_tensor.
                Shape: (batch_size, sequence_length, hidden_size)

        Raises:
            TypeError: If the input arguments are not of the expected types.
            ValueError: If the shapes of hidden_states and input_tensor are not compatible for addition.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdAttention(nn.Cell):

    """
    This class represents the BigBirdAttention module, which is used for attention computation in the BigBird model.
    It is a subclass of the nn.Cell class.

    Attributes:
        attention_type (str): The type of attention used in the module. It can be either 'original_full' or 'block_sparse'.
        config (object): The configuration object for the BigBird model.
        seed (int or None): The seed value used for random number generation. Default is None.
        self (object): The self-attention module used in the BigBirdAttention module.
        output (object): The module that processes the attention output.

    Methods:
        __init__: Initializes a new instance of the BigBirdAttention module.
        set_attention_type: Sets the attention_type attribute to the specified value.
        construct: Computes attention and returns the outputs.

    """
    def __init__(self, config, seed=None):
        """
        Initialize the BigBirdAttention class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration settings for the attention mechanism.
            seed (optional): An integer representing the random seed for reproducible results.

        Returns:
            None.

        Raises:
            ValueError: If the attention_type specified in the config is neither 'original_full' nor 'block_sparse'.
        """
        super().__init__()
        self.attention_type = config.attention_type
        self.config = config
        self.seed = seed

        if self.config.attention_type == "original_full":
            self.self = BigBirdSelfAttention(config)
        elif self.config.attention_type == "block_sparse":
            self.self = BigBirdBlockSparseAttention(config, seed)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        self.output = BigBirdSelfOutput(config)

    def set_attention_type(self, value: str):
        """
        This method sets the attention type for the BigBirdAttention class.

        Args:
            self: BigBirdAttention object, the instance of the class.
            value: str, the attention type to be set. It must be either 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the provided value is not 'original_full' or 'block_sparse'.
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
            attn_weights = BigBirdSelfAttention(self.config)
        else:
            # copy all weights to new sparse attention class
            attn_weights = BigBirdBlockSparseAttention(self.config, self.seed)

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
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        # block_sparse config
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
    ):
        """
        This method constructs the BigBird attention mechanism.

        Args:
            self: The instance of the BigBirdAttention class.
            hidden_states (Tensor): The input hidden states.
            attention_mask (Tensor, optional): Mask for attention computation.
            head_mask (Tensor, optional): Mask for individual attention heads.
            encoder_hidden_states (Tensor, optional): Hidden states of the encoder.
            encoder_attention_mask (Tensor, optional): Mask for encoder attention.
            past_key_value (Tuple, optional): Tuple containing past key and value tensors.
            output_attentions (bool): Whether to output attentions or not.
            band_mask (Tensor, optional): Mask for band attention computation.
            from_mask (Tensor, optional): Mask for 'from' sequence attention.
            to_mask (Tensor, optional): Mask for 'to' sequence attention.
            from_blocked_mask (Tensor, optional): Mask for 'from' sequence block attention.
            to_blocked_mask (Tensor, optional): Mask for 'to' sequence block attention.

        Returns:
            Tuple: A tuple containing the attention output and additional outputs from the attention mechanism.

        Raises:
            ValueError: If BigBird is used as a decoder when config.attention_type is not 'original_full'.
        """
        # fp16 compatibility
        if band_mask is not None:
            band_mask = band_mask.to(hidden_states.dtype)
        if from_mask is not None:
            from_mask = from_mask.to(hidden_states.dtype)
        if to_mask is not None:
            to_mask = to_mask.to(hidden_states.dtype)
        if self.attention_type == "original_full":
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        else:
            if encoder_hidden_states is not None:
                raise ValueError("BigBird cannot be used as a decoder when config.attention_type != 'original_full'")
            self_outputs = self.self(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BigBird
class BigBirdIntermediate(nn.Cell):

    """
    This class represents the intermediate layer of the BigBird model, which is used for processing hidden states.
    """
    def __init__(self, config):
        '''
        Initializes a new instance of the BigBirdIntermediate class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration parameters for the BigBirdIntermediate model.
                It is expected to have the following attributes:

                - hidden_size: An integer specifying the size of the hidden layer.
                - intermediate_size: An integer specifying the size of the intermediate layer.
                - hidden_act: A string or a function representing the activation function for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            KeyError: If the config.hidden_act value is not found in the ACT2FN dictionary.
        '''
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        The 'construct' method in the 'BigBirdIntermediate' class applies transformations to the input hidden states.

        Args:
            self: Represents the instance of the class.
            hidden_states (mindspore.Tensor): Input tensor containing the hidden states.
                It should be a valid instance of mindspore.Tensor.

        Returns:
            mindspore.Tensor: The transformed tensor representing the intermediate hidden states.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BigBird
class BigBirdOutput(nn.Cell):

    """
    This class represents the output layer of the BigBird model. It inherits from the nn.Cell class.

    The BigBirdOutput class applies various transformations to the input hidden states and produces the final output tensor.

    Attributes:
        dense (mindspore.nn.Dense): A fully connected layer that maps the input hidden states to the intermediate size.
        LayerNorm (mindspore.nn.LayerNorm): A layer normalization module that normalizes the hidden states.
        dropout (mindspore.nn.Dropout): A dropout module that applies dropout to the hidden states.

    Methods:
        __init__: Initializes the BigBirdOutput instance.
        construct: Applies the necessary transformations to the input hidden states and returns the output tensor.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the BigBirdOutput class.

        Args:
            self (BigBirdOutput): The current instance of the BigBirdOutput class.
            config: An object containing configuration parameters for the BigBirdOutput class.
                It should have the following attributes:

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the BigBird model.

        Args:
            self (BigBirdOutput): A reference to the current instance of the BigBirdOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor produced by the model.
                Shape should be (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor to the layer.
                Shape should be (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The output tensor of the BigBird model.
                Shape is the same as the input tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BigBirdLayer(nn.Cell):

    """
    This class represents a layer of the BigBird model, which is used for attention-based computations. It inherits from the nn.Cell class.

    Attributes:
        config: An object that stores the configuration settings for the BigBirdLayer.
        attention_type: A string representing the type of attention used by the layer.
        chunk_size_feed_forward: An integer specifying the chunk size for feed-forward computations.
        seq_len_dim: An integer indicating the dimension along which the sequence length is defined.
        attention: An instance of the BigBirdAttention class, responsible for performing attention computations.
        is_decoder: A boolean value indicating whether the BigBirdLayer is used as a decoder model.
        add_cross_attention: A boolean value specifying whether cross attention is added to the model.
        crossattention: An instance of the BigBirdAttention class used for cross-attention computations.
        intermediate: An instance of the BigBirdIntermediate class, responsible for intermediate computations.
        output: An instance of the BigBirdOutput class, used for the final output computations.

    Methods:
        set_attention_type: Sets the attention type to either 'original_full' or 'block_sparse'.
        construct: Constructs the layer by performing attention-based computations and returning the outputs.
        feed_forward_chunk: Applies the feed-forward computation on the attention output.

    """
    def __init__(self, config, seed=None):
        """
        Initializes a BigBirdLayer instance with the provided configuration and optional seed.

        Args:
            self: The BigBirdLayer instance itself.
            config: An object containing configuration settings for the BigBirdLayer.
                This parameter is required and should not be None.
            seed: An integer used for random seed initialization. Default is None.

        Returns:
            None.

        Raises:
            TypeError: If add_cross_attention is True but the model is not set as a decoder model.
        """
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BigBirdAttention(config, seed=seed)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BigBirdAttention(config)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)

    def set_attention_type(self, value: str):
        """
        Sets the attention type for the BigBirdLayer.

        Args:
            self (BigBirdLayer): The instance of the BigBirdLayer class.
            value (str): The attention type to be set. It can only be 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the provided attention type is not 'original_full' or 'block_sparse'.

        This method sets the attention type for the BigBirdLayer. The attention type determines the type of attention
        mechanism used in the layer.

        If the provided attention type is not 'original_full' or 'block_sparse', a ValueError is raised.
        Otherwise, if the provided attention type is the same as the current attention type, the method returns
        without making any changes. Otherwise, the attention type is updated, and the set_attention_type method is
        called on the attention object of the layer.

        If the layer has cross-attention enabled, the set_attention_type method is also called on the crossattention object.
        """
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return
        self.attention_type = value
        self.attention.set_attention_type(value)

        if self.add_cross_attention:
            self.crossattention.set_attention_type(value)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        blocked_encoder_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        '''
        Constructs the BigBirdLayer.

        Args:
            self: The instance of the BigBirdLayer class.
            hidden_states (Tensor): The input hidden states.
            attention_mask (Tensor, optional): The attention mask for the self-attention mechanism. Default is None.
            head_mask (Tensor, optional): The mask for the attention heads. Default is None.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder. Default is None.
            encoder_attention_mask (Tensor, optional): The attention mask for the encoder. Default is None.
            band_mask (Tensor, optional): The band mask for attention. Default is None.
            from_mask (Tensor, optional): The 'from' mask for attention. Default is None.
            to_mask (Tensor, optional): The 'to' mask for attention. Default is None.
            blocked_encoder_mask (Tensor, optional): The mask for blocked encoder. Default is None.
            past_key_value (Tensor, optional): The past key-value pair. Default is None.
            output_attentions (bool): Whether to output attentions. Default is False.

        Returns:
            None.

        Raises:
            ValueError: If `encoder_hidden_states` are passed and cross-attention layers are not instantiated
                by setting `config.add_cross_attention=True`.
        '''
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=blocked_encoder_mask,
            to_blocked_mask=blocked_encoder_mask,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with                    "
                    " cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Method: feed_forward_chunk

        Description:
        Performs a feed-forward chunk operation within the BigBirdLayer class.

        Args:
            self (BigBirdLayer): The instance of the BigBirdLayer class.
            attention_output (Tensor): The input tensor representing the attention output.

        Returns:
            Tensor: The output tensor after the feed-forward chunk operation.

        Raises:
            ValueError: If the input tensor dimensions are not compatible.
            RuntimeError: If there is an issue during the intermediate or output computations.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BigBirdEncoder(nn.Cell):

    """
    BigBirdEncoder represents an encoder module for the BigBird model.

    This class inherits from nn.Cell and includes methods to initialize the encoder, set attention type,
    and construct the encoder with various input parameters and return options.

    The 'construct' method processes the input hidden states and optional masks, producing hidden states,
    attentions, and cross-attentions based on the configuration and layer operations.

    The class also supports gradient checkpointing and cache management to optimize memory usage during training.

    Attributes:
        config: The configuration for the encoder.
        attention_type: The type of attention used in the encoder.
        layer: The list of BigBirdLayer instances used in the encoder.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled.

    Methods:
        __init__: Initializes the BigBirdEncoder with the provided configuration.
        set_attention_type: Sets the attention type for the encoder and its layers.
        construct: Constructs the encoder with the provided input and return options.

    Raises:
        ValueError: If the provided attention type is not 'original_full' or 'block_sparse'.

    Note:
        This class assumes the use of specific layer types and configurations as part of the BigBird model architecture.
    """
    def __init__(self, config):
        """
        Initializes the BigBirdEncoder class.

        Args:
            self: The instance of the BigBirdEncoder class.
            config: A configuration object containing various settings for the encoder.
                It should be an instance of a configuration class and must have the following attributes:

                - attention_type: A string specifying the type of attention to be used.
                - num_hidden_layers: An integer representing the number of hidden layers in the encoder configuration.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.attention_type = config.attention_type

        self.layer = nn.CellList(
            [BigBirdLayer(config, seed=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def set_attention_type(self, value: str):
        """
        Sets the attention type for the BigBirdEncoder.

        Args:
            self (BigBirdEncoder): An instance of the BigBirdEncoder class.
            value (str): The attention type to be set. It can only be set to either 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the provided value is not 'original_full' or 'block_sparse'.

        This method sets the attention type of the BigBirdEncoder instance to the specified value.
        If the provided value is not 'original_full' or 'block_sparse', a ValueError is raised.
        If the provided value is the same as the current attention type, the method does nothing.

        After setting the attention type, this method also updates the attention type for each layer in the
        BigBirdEncoder instance by calling the set_attention_type method on each layer.
        """
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return
        self.attention_type = value
        for layer in self.layer:
            layer.set_attention_type(value)

    def construct(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        blocked_encoder_mask=None,
        return_dict=True,
    ) -> Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]:
        """
        Constructs the BigBird encoder layer.

        Args:
            self: The BigBirdEncoder instance.
            hidden_states (torch.Tensor): The input hidden states for the encoder layer.
            attention_mask (Optional[torch.Tensor]): Mask to avoid attending to certain positions in the input sequence.
            head_mask (Optional[List[torch.Tensor]]): Mask for multi-head attention mechanism.
            encoder_hidden_states (Optional[torch.Tensor]): Hidden states from the encoder.
            encoder_attention_mask (Optional[torch.Tensor]): Mask for encoder's attention mechanism.
            past_key_values (Optional[List[torch.Tensor]]): Cached key and value tensors for previous steps.
            use_cache (Optional[bool]): Whether to use caching for intermediate results.
            output_attentions (bool): Whether to output attention tensors.
            output_hidden_states (bool): Whether to output hidden states at each layer.
            band_mask (Optional[torch.Tensor]): Mask for BigBird attention mechanism.
            from_mask (Optional[torch.Tensor]): Mask for source positions in BigBird attention.
            to_mask (Optional[torch.Tensor]): Mask for target positions in BigBird attention.
            blocked_encoder_mask (Optional[torch.Tensor]): Mask for blocked encoder positions.
            return_dict (bool): Whether to return the output as a dictionary.

        Returns:
            Union[BaseModelOutputWithPastAndCrossAttentions, Tuple]: The output of the BigBird encoder layer,
            which can be a BaseModelOutputWithPastAndCrossAttentions object or a tuple of relevant tensors based on
            the 'return_dict' flag.

        Raises:
            Warning: If 'use_cache' is set to True while using gradient checkpointing.
        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                band_mask,
                from_mask,
                to_mask,
                blocked_encoder_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->BigBird
class BigBirdPredictionHeadTransform(nn.Cell):

    """
    This class represents the transformation module used in the BigBirdPredictionHead model.
    It is responsible for applying a series of transformations to the input hidden states.

    The 'BigBirdPredictionHeadTransform' class inherits from the 'nn.Cell' class.

    Attributes:
        dense (nn.Dense): A fully connected layer that maps the input hidden states to the same hidden size.
        transform_act_fn (function): The activation function applied to the transformed hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module that normalizes the hidden states.

    Methods:
        __init__: Initializes a new instance of the 'BigBirdPredictionHeadTransform' class.
        construct: Applies the transformation operations to the input hidden states.

    """
    def __init__(self, config):
        """
        Initializes the BigBirdPredictionHeadTransform class.

        Args:
            self (object): The instance of the class.
            config (object):
                A configuration object containing the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - hidden_act (str or function): Activation function for the hidden layer. If it's a string, it refers to
                a predefined activation function in the ACT2FN dictionary; otherwise, it should be a custom activation function.
                - layer_norm_eps (float): The epsilon value for LayerNorm.

        Returns:
            None: The method initializes the instance variables of the BigBirdPredictionHeadTransform class.

        Raises:
            TypeError: If the config.hidden_act is not a string or a function.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the BigBirdPredictionHeadTransform.

        This method applies various transformations to the input hidden_states tensor and returns the transformed tensor.

        Args:
            self (BigBirdPredictionHeadTransform): The instance of the BigBirdPredictionHeadTransform.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the hidden states from the previous layer.

        Returns:
            mindspore.Tensor: The transformed tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the hidden states after applying the transformations.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->BigBird
class BigBirdLMPredictionHead(nn.Cell):

    """
    This class represents a prediction head for a BigBird Language Model (LM). It includes a transformer for processing
    hidden states and a decoder for generating predictions based on the transformed states.
    The decoder does not have bias and utilizes a custom bias parameter.
    The class inherits from nn.Cell and provides a 'construct' method to perform the transformation and decoding
    of hidden states.

    Attributes:
        transform: An instance of BigBirdPredictionHeadTransform used to transform hidden states.
        decoder: An instance of nn.Dense for decoding transformed states into predictions.
        bias: A custom bias parameter for the decoder.

    Methods:
        construct(hidden_states): Applies the transformation and decoding process to the input hidden states, returning the final predictions.
    """
    def __init__(self, config):
        """Initializes an instance of the BigBirdLMPredictionHead class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the prediction head.

                - Type: Any
                - Purpose: Configures the prediction head with specific settings.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.transform = BigBirdPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        Constructs the hidden states for the BigBirdLMPredictionHead.

        Args:
            self (BigBirdLMPredictionHead): The instance of the BigBirdLMPredictionHead class.
            hidden_states (Tensor): The input hidden states to be processed.
                Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: The constructed hidden states after applying transformations.
                Shape: (batch_size, sequence_length, hidden_size).

        Raises:
            None.

        This method takes the input hidden states and performs transformations on them to construct the
        final hidden states for the BigBirdLMPredictionHead.
        It first applies the 'transform' method to the hidden states, which applies any necessary pre-processing.
        Then, it passes the transformed hidden states through the 'decoder' to generate the final constructed hidden states.
        The constructed hidden states are then returned.

        Note:
            This method does not modify the input hidden states in-place.
            Instead, it returns the constructed hidden states as a new tensor.

        Example:
            ```python
            >>> head = BigBirdLMPredictionHead()
            >>> hidden_states = torch.randn(32, 128, 768)
            >>> constructed_states = head.construct(hidden_states)
            ```
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->BigBird
class BigBirdOnlyMLMHead(nn.Cell):

    """
    This class represents the BigBirdOnlyMLMHead, which is responsible for generating prediction scores
    for masked language modeling (MLM) tasks using the BigBird transformer model.

    The BigBirdOnlyMLMHead class inherits from the nn.Cell class.

    Example:
        ```python
        >>> config = BigBirdConfig(...)
        >>> model = BigBirdOnlyMLMHead(config)
        >>> sequence_output = mindspore.Tensor(...)
        >>> prediction_scores = model.construct(sequence_output)
        ```

    Attributes:
        predictions (BigBirdLMPredictionHead): An instance of the BigBirdLMPredictionHead class that
        handles the prediction generation.

    Methods:
        __init__:
            Initializes a BigBirdOnlyMLMHead object.

            Args:

            - config (BigBirdConfig): The configuration object for the BigBird transformer model.

        construct:
            Generates prediction scores for masked language modeling tasks.

            Args:

            - sequence_output (mindspore.Tensor):
            The output tensor from the preceding layer of the BigBird transformer model.

            Returns:

            - mindspore.Tensor: The prediction scores for the MLM tasks.
    """
    def __init__(self, config):
        """
        Initializes a BigBirdOnlyMLMHead object with the provided configuration.

        Args:
            self: The instance of the BigBirdOnlyMLMHead class.
            config: A dictionary containing the configuration settings for the BigBirdOnlyMLMHead.
                The configuration should include parameters such as hidden_size, num_labels, etc.
                This parameter is required for initializing the BigBirdLMPredictionHead object.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not a dictionary or if it is missing required keys.
            ValueError: If the config values are invalid or out of range.
        """
        super().__init__()
        self.predictions = BigBirdLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """Constructs the masked language model head for the BigBirdOnlyMLMHead class.

        Args:
            self (BigBirdOnlyMLMHead): The instance of the BigBirdOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The output tensor from the previous layer of the model.
                It represents the sequence of hidden states for each input token.
                The shape of the tensor should be (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The prediction scores tensor for the masked language model.
                It represents the scores assigned to each vocabulary token for the given input sequence.
                The shape of the tensor is (batch_size, sequence_length, vocab_size).

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->BigBird
class BigBirdOnlyNSPHead(nn.Cell):

    """
    This class represents a BigBird Only Next Sentence Prediction (NSP) Head, which is used in
    natural language processing tasks. It inherits from the `nn.Cell` class provided by the MindSpore framework.

    The BigBirdOnlyNSPHead class contains methods for initializing and constructing the NSP head.

    Methods:
        __init__(self, config):
            Initializes an instance of the BigBirdOnlyNSPHead class.

            Args:

            - config (object): An object containing the configuration parameters for the NSP head.

            Returns:

            - None

        construct(self, pooled_output):
            Constructs and returns the next sentence prediction score based on the given pooled_output.

            Args:

            - pooled_output (Tensor): A tensor representing the pooled output from the previous layer.

            Returns:

            - seq_relationship_score (Tensor): A tensor representing the next sentence prediction score.
    """
    def __init__(self, config):
        """
        Initializes a BigBirdOnlyNSPHead object.

        Args:
            self (object): The instance of the class.
            config (object): Configuration object containing parameters for the head.
                 Requires 'hidden_size' attribute for the size of the hidden layer.

        Returns:
            None: The method initializes the 'seq_relationship' attribute of the instance.

        Raises:
            AttributeError: If the 'config' object does not have the required 'hidden_size' attribute.
        """
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        """
        This method constructs a BigBirdOnlyNSPHead by calculating the sequence relationship score based on the pooled output.

        Args:
            self (object): The BigBirdOnlyNSPHead instance.
            pooled_output (tensor): The pooled output tensor representing the sequence information.

        Returns:
            None.

        Raises:
            None.
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->BigBird
class BigBirdPreTrainingHeads(nn.Cell):

    """
    The BigBirdPreTrainingHeads class represents the pre-training heads for the BigBird model.
    It includes methods for predicting masked tokens and determining sequence relationships.
    This class inherits from nn.Cell and contains the necessary components for pre-training tasks in the BigBird model.
    """
    def __init__(self, config):
        """
        Initializes the BigBirdPreTrainingHeads class.

        Args:
            self: The instance of the class.
            config:
                The configuration object containing various settings and hyperparameters for the model.

                - Type: object
                - Purpose: To configure the pre-training heads based on the provided settings.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None: The method initializes the predictions and sequence relationship attributes of the class.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = BigBirdLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        """
        Construct the prediction and relation scores for the BigBirdPreTrainingHeads model.

        Args:
            self (BigBirdPreTrainingHeads): The instance of the BigBirdPreTrainingHeads class.
            sequence_output (tensor): The sequence output tensor from the model.
            pooled_output (tensor): The pooled output tensor from the model.

        Returns:
            tuple:
                A tuple containing two elements:

                - prediction_scores (tensor): The prediction scores for the sequence output.
                - seq_relationship_score (tensor): The sequence relationship score for the pooled output.

        Raises:
            None
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BigBirdPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BigBirdConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
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

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


BIG_BIRD_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Cell](https://pytorch.org/docs/stable/nn.html#torch.nn.Cell) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BigBirdConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BIG_BIRD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`mindspore.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`mindspore.Tensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`mindspore.Tensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`mindspore.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`mindspore.Tensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@dataclass
class BigBirdForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BigBirdForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `mindspore.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`mindspore.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    prediction_logits: mindspore.Tensor = None
    seq_relationship_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


@dataclass
class BigBirdForQuestionAnsweringModelOutput(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        pooler_output (`mindspore.Tensor` of shape `(batch_size, 1)`):
            pooler output from BigBigModel
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    start_logits: mindspore.Tensor = None
    end_logits: mindspore.Tensor = None
    pooler_output: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class BigBirdModel(BigBirdPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the BigBirdModel class.

        Args:
            self (BigBirdModel): The instance of the class.
            config (object): The configuration object containing model parameters.
            add_pooling_layer (bool): Flag to indicate whether to add a pooling layer.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.attention_type = self.config.attention_type
        self.config = config

        self.block_size = self.config.block_size

        self.embeddings = BigBirdEmbeddings(config)
        self.encoder = BigBirdEncoder(config)

        if add_pooling_layer:
            self.pooler = nn.Dense(config.hidden_size, config.hidden_size)
            self.activation = nn.Tanh()
        else:
            self.pooler = None
            self.activation = None

        if self.attention_type != "original_full" and config.add_cross_attention:
            logger.warning(
                "When using `BigBirdForCausalLM` as decoder, then `attention_type` must be `original_full`. Setting"
                " `attention_type=original_full`"
            )
            self.set_attention_type("original_full")

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the BigBirdModel.

        Args:
            self: An instance of the BigBirdModel class.

        Returns:
            word_embeddings: The method returns the word embeddings stored in the BigBirdModel instance.

        Raises:
            word_embeddings
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Set the input embeddings of the BigBirdModel.

        Args:
            self (BigBirdModel): An instance of the BigBirdModel class.
            value: The new input embeddings to be set. This should be an instance of a compatible embedding object.

        Returns:
            None.

        Raises:
            None.

        This method is used to update the input embeddings of the BigBirdModel with a new set of embeddings.
        It takes in the instance of the BigBirdModel class and the new embeddings to be set as parameters.
        The 'value' parameter should be an instance of a compatible embedding object.

        Note that changing the input embeddings can have a significant impact on the model's performance, s
        o it should be done carefully and with consideration of the specific task and data being used.

        This method does not return any value, as it directly modifies the input embeddings of the BigBirdModel instance.

        Example:
            ```python
            >>> model = BigBirdModel()
            >>> embeddings = WordEmbeddings()
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.embeddings.word_embeddings = value

    def set_attention_type(self, value: str):
        """
        Method to set the attention type for the BigBirdModel.

        Args:
            self: Instance of the BigBirdModel class.
            value (str): The specified attention type to set. It can only be either 'original_full' or 'block_sparse'.

        Returns:
            None.

        Raises:
            ValueError: If the value provided is not 'original_full' or 'block_sparse'.
        """
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return
        self.attention_type = value
        self.encoder.set_attention_type(value)

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPoolingAndCrossAttentions, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors
                of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones(((batch_size, seq_length + past_key_values_length)))
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # in order to use block_sparse attention, sequence_length has to be at least
        # bigger than all global attentions: 2 * block_size
        # + sliding tokens: 3 * block_size
        # + random tokens: 2 * num_random_blocks * block_size
        max_tokens_to_attend = (5 + 2 * self.config.num_random_blocks) * self.config.block_size
        if self.attention_type == "block_sparse" and seq_length <= max_tokens_to_attend:
            # change attention_type from block_sparse to original_full
            sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
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
            (
                padding_len,
                input_ids,
                attention_mask,
                token_type_ids,
                position_ids,
                inputs_embeds,
            ) = self._pad_to_block_size(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pad_token_id=self.config.pad_token_id,
            )
        else:
            padding_len = 0

        if self.attention_type == "block_sparse":
            blocked_encoder_mask, band_mask, from_mask, to_mask = self.create_masks_for_block_sparse_attn(
                attention_mask, self.block_size
            )
            extended_attention_mask = None

        elif self.attention_type == "original_full":
            blocked_encoder_mask = None
            band_mask = None
            from_mask = None
            to_mask = None
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.attention_type}"
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            blocked_encoder_mask=blocked_encoder_mask,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        pooler_output = self.activation(self.pooler(sequence_output[:, 0, :])) if (self.pooler is not None) else None

        # undo padding
        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.shape[1]
            sequence_output = sequence_output[:, :-padding_len]

        if not return_dict:
            return (sequence_output, pooler_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

    @staticmethod
    def create_masks_for_block_sparse_attn(attention_mask: mindspore.Tensor, block_size: int):
        """
        Creates masks for block sparse attention in the BigBirdModel class.

        Args:
            attention_mask (mindspore.Tensor): A 2D tensor representing the attention mask.
                Shape: [batch_size, seq_length].
            block_size (int): The size of each attention block.

        Returns:
            tuple:
                A tuple containing the following four tensors:

                - blocked_encoder_mask (mindspore.Tensor): A 3D tensor representing the attention mask in blocked format.
                Shape: [batch_size, seq_length // block_size, block_size].
                - band_mask (mindspore.Tensor): A 5D tensor representing the band mask for block sparse attention.
                Shape: [batch_size, 1, seq_length // block_size - 4, block_size, 3 * block_size].
                - from_mask (mindspore.Tensor): A 4D tensor representing the attention mask for the "from" sequence.
                Shape: [batch_size, 1, seq_length, 1].
                - to_mask (mindspore.Tensor): A 4D tensor representing the attention mask for the "to" sequence.
                Shape: [batch_size, 1, 1, seq_length].

        Raises:
            ValueError: If the sequence length is not a multiple of the block size.

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
                from_blocked_mask: 2D Tensor of shape [batch_size, from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size, to_seq_length//to_block_size, to_block_size].

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

    def _pad_to_block_size(
        self,
        input_ids: mindspore.Tensor,
        attention_mask: mindspore.Tensor,
        token_type_ids: mindspore.Tensor,
        position_ids: mindspore.Tensor,
        inputs_embeds: mindspore.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        # padding
        block_size = self.config.block_size

        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )
            if input_ids is not None:
                input_ids = ops.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_bigbird.BigBirdEmbeddings
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
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens
            token_type_ids = ops.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds


class BigBirdForPreTraining(BigBirdPreTrainedModel):

    """
    This class represents a BigBird model for pre-training tasks, inheriting from BigBirdPreTrainedModel.
    It includes methods for initialization, getting and setting output embeddings, and constructing the
    model for pre-training tasks. The constructor initializes the model with the provided configuration,
    sets up the BigBird model and pre-training heads, and executes post-initialization steps.
    Methods are provided for retrieving and updating the output embeddings.
    The 'construct' method takes various input parameters for constructing the model, computes
    the masked language modeling loss and next sequence
    prediction loss if labels are provided, and returns the pre-training outputs.
    An example usage is provided in the docstring.
    """
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        """
        Initializes an instance of the BigBirdForPreTraining class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' that contains the configuration parameters for the model.
                It should be an instance of the BigBirdConfig class and must contain the following attributes:

                - add_pooling_layer (bool): Whether to add a pooling layer to the model. Default is True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.bert = BigBirdModel(config, add_pooling_layer=True)
        self.cls = BigBirdPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings for the BigBirdForPreTraining model.

        Args:
            self: An instance of the BigBirdForPreTraining class.

        Returns:
            None: The method returns the output embeddings for the model.

        Raises:
            This method does not raise any exceptions.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the BigBirdForPreTraining model.

        Args:
            self (BigBirdForPreTraining): The instance of the BigBirdForPreTraining class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings.
                This can be a tensor or any object that can be assigned to the output embeddings attribute.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method allows the user to set the output embeddings of the BigBirdForPreTraining model.
            The output embeddings are assigned to the `predictions.decoder` attribute of the model's `cls` object.
            By setting new embeddings, the user can customize or update the output embeddings used in the model's predictions.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        next_sentence_label: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BigBirdForPreTrainingOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

            next_sentence_label (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. If specified, nsp loss will be
                added to masked_lm loss. Input should be a sequence pair (see `input_ids` docstring) Indices should be in
                `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:
            Union[BigBirdForPreTrainingOutput, Tuple[mindspore.Tensor]]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, BigBirdForPreTraining
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
            >>> model = BigBirdForPreTraining.from_pretrained("google/bigbird-roberta-base")
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None:
            total_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if next_sentence_label is not None and total_loss is not None:
            next_sentence_loss = ops.cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = total_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BigBirdForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BigBirdForMaskedLM(BigBirdPreTrainedModel):

    """
    BigBirdForMaskedLM includes methods to create a BigBird model for masked language modeling tasks.

    This class inherits from BigBirdPreTrainedModel, and provides functionality to initialize the model,
    get and set the output embeddings, construct the model for masked language modeling, and prepare inputs
    for generation.

    Example:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, BigBirdForMaskedLM
        >>> from datasets import load_dataset
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        >>> model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
        >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT
        ...
        >>> # select random long article
        >>> LONG_ARTICLE_TARGET = squad_ds[81514]["context"]
        >>> # select random sentence
        >>> LONG_ARTICLE_TARGET[332:398]
        'the highest values are very close to the theoretical maximum value'
        >>> # add mask_token
        >>> LONG_ARTICLE_TO_MASK = LONG_ARTICLE_TARGET.replace("maximum", "[MASK]")
        >>> inputs = tokenizer(LONG_ARTICLE_TO_MASK, return_tensors="pt")
        >>> # long article input
        >>> list(inputs["input_ids"].shape)
        [1, 919]
        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits
        >>> # retrieve index of [MASK]
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        'maximum'
        ```
    """
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        """
        Initializes an instance of the BigBirdForMaskedLM class.

        Args:
            self: The instance of the class.
            config: An object of the Config class containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BigBirdForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the BigBirdForMaskedLM model.

        Args:
            self (BigBirdForMaskedLM): The instance of the BigBirdForMaskedLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the BigBirdForMaskedLM model.

        Args:
            self (object): The instance of the BigBirdForMaskedLM class.
            new_embeddings (object): The new embeddings to be set as the output embeddings for the model.
                It can be of any valid type supported for model embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Union[MaskedLMOutput, Tuple[mindspore.Tensor]]

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoTokenizer, BigBirdForMaskedLM
            >>> from datasets import load_dataset
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
            >>> model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
            >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT
            ...
            >>> # select random long article
            >>> LONG_ARTICLE_TARGET = squad_ds[81514]["context"]
            >>> # select random sentence
            >>> LONG_ARTICLE_TARGET[332:398]
            'the highest values are very close to the theoretical maximum value'
            >>> # add mask_token
            >>> LONG_ARTICLE_TO_MASK = LONG_ARTICLE_TARGET.replace("maximum", "[MASK]")
            >>> inputs = tokenizer(LONG_ARTICLE_TO_MASK, return_tensors="pt")
            >>> # long article input
            >>> list(inputs["input_ids"].shape)
            [1, 919]
            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits
            >>> # retrieve index of [MASK]
            >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            >>> tokenizer.decode(predicted_token_id)
            'maximum'
            ```

            ```python
            >>> labels = tokenizer(LONG_ARTICLE_TARGET, return_tensors="pt")["input_ids"]
            >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
            >>> outputs = model(**inputs, labels=labels)
            >>> round(outputs.loss.item(), 2)
            1.99
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
        """
        Prepares inputs for generation in the BigBirdForMaskedLM model.

        Args:
            self (BigBirdForMaskedLM): The instance of the BigBirdForMaskedLM class.
            input_ids (Tensor): The input tensor of shape (batch_size, sequence_length).
                The tensor represents the input token IDs.
            attention_mask (Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                It masks the padding tokens. Defaults to None.

        Returns:
            dict: A dictionary containing the prepared inputs for generation.
                The dictionary has the following keys:

                - 'input_ids' (Tensor): The input tensor of shape (batch_size, sequence_length + 1).
                It includes an additional dummy token at the end.
                - 'attention_mask' (Tensor): The attention mask tensor of shape (batch_size, sequence_length + 1).
                It includes an additional attention mask for the dummy token.

        Raises:
            ValueError: If the PAD token is not defined for generation.
        """
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
        attention_mask = ops.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], axis=-1)
        dummy_token = ops.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=mindspore.int64
        )
        input_ids = ops.cat([input_ids, dummy_token], axis=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class BigBirdForCausalLM(BigBirdPreTrainedModel):

    """
    This class represents a BigBird model for Causal Language Modeling (LM).
    It is designed for generating text sequences autoregressively,
    predicting the next token in a sequence given the previous tokens.

    The class includes methods for initializing the model, getting and setting output embeddings,
    constructing the model with various input parameters, preparing inputs for text generation, and reordering
    cache during decoding.

    Attributes:
        bert: BigBirdModel instance representing the core BigBird model.
        cls: BigBirdOnlyMLMHead instance for predicting masked tokens in the input sequence.

    Methods:
        __init__(self, config): Initializes the BigBirdForCausalLM model with the provided configuration.
        get_output_embeddings(self): Retrieves the output embeddings from the model.
        set_output_embeddings(self, new_embeddings): Sets new output embeddings for the model.
        construct(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
            encoder_hidden_states, encoder_attention_mask, past_key_values, labels, use_cache, output_attentions,
            output_hidden_states, return_dict): Constructs the model for LM generation, taking various input parameters.
        prepare_inputs_for_generation(self, input_ids, past_key_values, attention_mask): Prepares inputs for text generation, handling past key values and attention mask.
        _reorder_cache(self, past_key_values, beam_idx): Reorders the cache during decoding based on beam index for faster processing.

    Note:
        This class is inherited from BigBirdPreTrainedModel for leveraging pre-trained weights and configurations.
    """
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        """
        Initializes an instance of the BigBirdForCausalLM class.

        Args:
            self: The instance of the class.
            config: An instance of the BigBirdConfig class containing the configuration settings for the model.
                It must have the following attributes:

                - is_decoder (bool): Indicates whether the model is used as a decoder. If False, a warning message is logged.

        Returns:
            None.

        Raises:
            None.

        This method initializes the BigBirdForCausalLM instance by calling the superclass's __init__ method with the provided config.
        If the is_decoder attribute in the config is False, a warning message is logged to remind the user to set it to True
        if they want to use BigBirdForCausalLM as a standalone model.
        The method then initializes the bert attribute with an instance of the BigBirdModel class, using the provided config.
        Finally, the cls attribute is initialized with an instance of the BigBirdOnlyMLMHead class, using the provided config.
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BigBirdForCausalLM` as a standalone, add `is_decoder=True.`")

        self.bert = BigBirdModel(config)
        self.cls = BigBirdOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the BigBirdForCausalLM model.

        Args:
            self (BigBirdForCausalLM): The instance of the BigBirdForCausalLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the BigBirdForCausalLM model.

        Args:
            self (BigBirdForCausalLM): The instance of the BigBirdForCausalLM class.
            new_embeddings: The new embeddings to be set for the output layer. It should be a tensor of shape
                (vocab_size, hidden_size), where vocab_size is the size of the output vocabulary
                and hidden_size is the size of the hidden layers in the model.

        Returns:
            None: This method modifies the output embeddings of the BigBirdForCausalLM model in place.

        Raises:
            None.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        labels: Optional[mindspore.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[CausalLMOutputWithCrossAttentions, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = ops.cross_entropy(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None):
        """
        This method prepares inputs for generation in the BigBirdForCausalLM class.

        Args:
            self: The instance of the class.
            input_ids (torch.Tensor): The input tensor containing the token ids.
                Shape should be (batch_size, sequence_length).
            past_key_values (tuple, optional): The tuple of past key values for attention mechanism.
                Default is None.
            attention_mask (torch.Tensor, optional): The attention mask tensor.
                If not provided, it is initialized with ones of the same shape as input_ids.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (torch.Tensor): The input tensor for generation with potentially removed prefix.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                - 'past_key_values' (tuple): The past key values for attention mechanism.

        Raises:
            ValueError: If the input_ids shape does not match the expected shape.
            IndexError: If the past_key_values tuple does not have the expected structure.
        """
        input_shape = input_ids.shape

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

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

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache for the BigBirdForCausalLM model based on the provided beam index.

        Args:
            self (BigBirdForCausalLM): The instance of the BigBirdForCausalLM class.
            past_key_values (tuple): A tuple containing past key values for the model.
            beam_idx (Tensor): A tensor representing the beam index to reorder the cache.

        Returns:
            None: This method does not return any value but modifies the cache in-place.

        Raises:
            IndexError: If the beam index is out of bounds or invalid.
            ValueError: If the past_key_values are not in the expected format.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


class BigBirdClassificationHead(nn.Cell):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        """
        Initializes a BigBirdClassificationHead instance.

        Args:
            self (BigBirdClassificationHead): The instance of the BigBirdClassificationHead class.
            config (object): An object containing configuration parameters for the classification head.
                The config object should have the following attributes:

                - hidden_size (int): The size of the hidden layers in the classification head.
                - classifier_dropout (float or None): The dropout probability for the classifier layer.
                If None, it defaults to the hidden dropout probability.
                - hidden_dropout_prob (float): The dropout probability for the hidden layers.
                - num_labels (int): The number of output labels.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If the config object is missing any of the required attributes.
            AttributeError: If the config object does not have the necessary attributes to initialize the classification head.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Dense(config.hidden_size, config.num_labels)

        self.config = config

    def construct(self, features, **kwargs):
        """
        Constructs the BigBird classification head.

        Args:
            self (BigBirdClassificationHead): The instance of the BigBirdClassificationHead class.
            features (tensor): The input features tensor of shape (batch_size, sequence_length, feature_size).

        Returns:
            None: This method does not return any value directly, but modifies the input features tensor in place.

        Raises:
            None.
        """
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BigBirdForSequenceClassification(BigBirdPreTrainedModel):

    """
    BigBirdForSequenceClassification is a class that represents a BigBird model for sequence classification tasks.
    It extends the functionality of BigBirdPreTrainedModel to include sequence classification capabilities.

    This class includes an initialization method '__init__' that initializes the model with the provided configuration.
    It also includes a 'construct' method that constructs the model for inference or training, taking input tensors and optional arguments.
    The method computes the loss based on the provided labels and returns the sequence classifier output.

    The 'construct' method accepts various input tensors such as input_ids, attention_mask, token_type_ids, etc.,
    and computes the sequence classification/regression loss based on the provided labels.
    The method supports different types of loss calculations depending on the configuration and the number of labels.

    The class provides an example usage demonstrating how to load the model, tokenize input text, and perform sequence classification using the model.
    It also showcases how to compute the loss for a given input and labels.

    For detailed usage and examples, refer to the code snippets provided in the docstring.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BigBirdForSequenceClassification class.

        Args:
            self (BigBirdForSequenceClassification): The instance of the class itself.
            config: The configuration object containing the model configuration.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BigBirdModel(config)
        self.classifier = BigBirdClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[SequenceClassifierOutput, Tuple[mindspore.Tensor]]

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoTokenizer, BigBirdForSequenceClassification
            >>> from datasets import load_dataset
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
            >>> model = BigBirdForSequenceClassification.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
            >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT
            ...
            >>> LONG_ARTICLE = squad_ds[81514]["context"]
            >>> inputs = tokenizer(LONG_ARTICLE, return_tensors="pt")
            >>> # long input article
            >>> list(inputs["input_ids"].shape)
            [1, 919]
            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits
            >>> predicted_class_id = logits.argmax().item()
            >>> model.config.id2label[predicted_class_id]
            'LABEL_0'
            ```

            ```python
            >>> num_labels = len(model.config.id2label)
            >>> model = BigBirdForSequenceClassification.from_pretrained(
            ...     "l-yohai/bigbird-roberta-base-mnli", num_labels=num_labels
            ... )
            >>> labels = mindspore.tensor(1)
            >>> loss = model(**inputs, labels=labels).loss
            >>> round(loss.item(), 2)
            1.13
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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
                elif self.num_labels > 1 and labels.dtype in (mindspore.int32, mindspore.int64):
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


class BigBirdForMultipleChoice(BigBirdPreTrainedModel):

    """
    BigBirdForMultipleChoice is a class for multiple choice question answering using the BigBird model.
    It inherits from BigBirdPreTrainedModel and provides methods to construct the model for multiple choice tasks.

    Attributes:
        bert (BigBirdModel): The BigBird model used for processing input sequences.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Dense): Dense layer for classification.

    Methods:
        __init__(config): Initializes the BigBirdForMultipleChoice class with the given configuration.
        construct(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels,
            output_attentions, output_hidden_states, return_dict): Constructs the model for multiple choice tasks.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BigBirdForMultipleChoice class.

        Args:
            self: The instance of the class.
            config: An object containing configuration settings for the BigBirdModel.

        Returns:
            None.

        Raises:
            NotImplementedError: If the method 'post_init' is not implemented in the derived class.
            TypeError: If the 'config' parameter is not of the expected type.
        """
        super().__init__(config)

        self.bert = BigBirdModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MultipleChoiceModelOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
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
            loss = ops.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BigBirdForTokenClassification(BigBirdPreTrainedModel):

    """
    BigBirdForTokenClassification is a token classification model based on the BigBird architecture.
    It inherits from BigBirdPreTrainedModel and is designed for token-level classification tasks, such as Named
    Entity Recognition or Part-of-Speech tagging.

    The class's constructor initializes the model with the provided configuration and sets up the necessary components,
    including the BigBirdModel, dropout layers, and classifier. It also calls the post_init method for additional setup.

    The construct method takes input tensors and optional arguments for various model outputs, such as attentions and hidden states.
    It returns the token classification output, including logits for each token, and computes the token classification loss if labels are provided.

    The labels parameter in the construct method is an optional tensor containing the target labels for token classification.
    The indices in the labels tensor should be in the range [0, num_labels - 1].

    If return_dict is set to False, the method returns a tuple containing the token logits and additional model outputs.
    If return_dict is True, the method returns a TokenClassifierOutput object containing the loss, logits, hidden states,
    and attentions.

    Note:
        This docstring is generated based on the provided code snippet and may need to be updated with additional
        details about the class and its methods.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BigBirdForTokenClassification class.

        Args:
            self: The object itself.
            config: An instance of the BigBirdConfig class that contains the configuration parameters for the model.
                It should have the following attributes:

                - num_labels (int): The number of labels for token classification.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BigBirdModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: mindspore.Tensor = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[mindspore.Tensor]]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
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
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BigBirdForQuestionAnsweringHead(nn.Cell):
    """Head for question answering tasks."""
    def __init__(self, config):
        """
        Initializes an instance of the BigBirdForQuestionAnsweringHead class.

        Args:
            self (BigBirdForQuestionAnsweringHead): An instance of the BigBirdForQuestionAnsweringHead class.
            config: The configuration object for the BigBird model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.intermediate = BigBirdIntermediate(config)
        self.output = BigBirdOutput(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, encoder_output):
        """
        Constructs the question-answering head for the BigBird model.

        Args:
            self (BigBirdForQuestionAnsweringHead): An instance of the BigBirdForQuestionAnsweringHead class.
            encoder_output (Tensor): The output tensor from the encoder.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.dropout(encoder_output)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states, encoder_output)
        hidden_states = self.qa_outputs(hidden_states)
        return hidden_states


class BigBirdForQuestionAnswering(BigBirdPreTrainedModel):

    """
    The `BigBirdForQuestionAnswering` class represents a model for question answering using the BigBird architecture.
    It is a subclass of `BigBirdPreTrainedModel` and provides methods for training, evaluating,
    and predicting question answering tasks.

    Attributes:
        `config`: An instance of `BigBirdConfig` that holds the model configuration.
        `num_labels`: The number of labels for the question answering task.
        `sep_token_id`: The token ID for the separator token in the input.
        `bert`: The BigBirdModel instance that serves as the base model.
        `qa_classifier`: The BigBirdForQuestionAnsweringHead instance that performs question answering classification.

    Methods:
        `__init__(self, config, add_pooling_layer=False)`: Initializes the `BigBirdForQuestionAnswering` instance.
        `construct(self, input_ids, attention_mask, question_lengths, token_type_ids, position_ids, head_mask,
            inputs_embeds, start_positions, end_positions, output_attentions, output_hidden_states, return_dict)`:
            Constructs the model for question answering.
        `prepare_question_mask(q_lengths, maxlen)`: Prepares a question mask for question answering.

    Example:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, BigBirdForQuestionAnswering
        >>> from datasets import load_dataset
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
        >>> model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
        >>> squad_ds = load_dataset("squad_v2", split="train")
        ...
        >>> # select random article and question
        >>> LONG_ARTICLE = squad_ds[81514]["context"]
        >>> QUESTION = squad_ds[81514]["question"]
        >>> inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors="pt")
        ...
        >>> with torch.no_grad():
        >>>     outputs = model(**inputs)
        ...
        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()
        >>> predict_answer_token_ids = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        >>> predict_answer_token = tokenizer.decode(predict_answer_token_ids)
        ...
        >>> target_start_index, target_end_index = mindspore.tensor([130]), mindspore.tensor([132])
        >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
        >>> loss = outputs.loss
        ```
    """
    def __init__(self, config, add_pooling_layer=False):
        '''
        __init__

        Initializes an instance of the BigBirdForQuestionAnswering class.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the model configuration.
            add_pooling_layer (bool, optional): A boolean indicating whether to add a pooling layer. Defaults to False.

        Returns:
            None.

        Raises:
            None
        '''
        super().__init__(config)

        config.num_labels = 2
        self.num_labels = config.num_labels
        self.sep_token_id = config.sep_token_id

        self.bert = BigBirdModel(config, add_pooling_layer=add_pooling_layer)
        self.qa_classifier = BigBirdForQuestionAnsweringHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        question_lengths: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BigBirdForQuestionAnsweringModelOutput, Tuple[mindspore.Tensor]]:
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

        Returns:
            Union[BigBirdForQuestionAnsweringModelOutput, Tuple[mindspore.Tensor]]

        Example:
            ```python
            >>> import torch
            >>> from transformers import AutoTokenizer, BigBirdForQuestionAnswering
            >>> from datasets import load_dataset
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-base")
            >>> model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base")
            >>> squad_ds = load_dataset("squad_v2", split="train")  # doctest: +IGNORE_RESULT
            ...
            >>> # select random article and question
            >>> LONG_ARTICLE = squad_ds[81514]["context"]
            >>> QUESTION = squad_ds[81514]["question"]
            >>> QUESTION
            'During daytime how high can the temperatures reach?'
            >>> inputs = tokenizer(QUESTION, LONG_ARTICLE, return_tensors="pt")
            >>> # long article and question input
            >>> list(inputs["input_ids"].shape)
            [1, 929]
            >>> with torch.no_grad():
            ...     outputs = model(**inputs)
            ...
            >>> answer_start_index = outputs.start_logits.argmax()
            >>> answer_end_index = outputs.end_logits.argmax()
            >>> predict_answer_token_ids = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
            >>> predict_answer_token = tokenizer.decode(predict_answer_token_ids)
            ```

            ```python
            >>> target_start_index, target_end_index = mindspore.tensor([130]), mindspore.tensor([132])
            >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
            >>> loss = outputs.loss
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seqlen = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        if question_lengths is None and input_ids is not None:
            # assuming input_ids format: <cls> <question> <sep> context <sep>
            question_lengths = ops.argmax(input_ids.eq(self.sep_token_id).int(), dim=-1) + 1
            question_lengths = question_lengths.unsqueeze(1)

        logits_mask = None
        if question_lengths is not None:
            # setting lengths logits to `-inf`
            logits_mask = self.prepare_question_mask(question_lengths, seqlen)
            if token_type_ids is None:
                token_type_ids = ops.ones(logits_mask.shape, dtype=mindspore.int32) - logits_mask
            logits_mask[:, 0] = False
            logits_mask = logits_mask.unsqueeze(2)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.qa_classifier(sequence_output)

        if logits_mask is not None:
            # removing question tokens from the competition
            logits = logits - logits_mask * 1e6

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
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BigBirdForQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            pooler_output=outputs.pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def prepare_question_mask(q_lengths: mindspore.Tensor, maxlen: int):
        """
        Prepare a binary mask for the question tokens in the BigBirdForQuestionAnswering class.
        
        Args:
            q_lengths (mindspore.Tensor): A tensor containing the lengths of the question tokens.
                  Each element represents the length of a question in the batch.
                  Shape: (batch_size,)
            maxlen (int): The maximum length of the question tokens.
                  The mask will be padded with zeros up to this length.
            
        Returns:
            None
            
        Raises:
            TypeError: If q_lengths is not of type mindspore.Tensor.
            ValueError: If q_lengths or maxlen is not a positive integer.
        
        Note:
            This method generates a binary mask for the question tokens in a batch, indicating which tokens are valid based on their lengths.
            The mask will be of shape (batch_size, maxlen), where each element is either 1 (valid token) or 0 (padding token).
            The mask is used in subsequent operations to ignore the padding tokens during computation.
        """
        # q_lengths -> (bz, 1)
        mask = ops.arange(0, maxlen)
        mask = mask.unsqueeze(0)  # -> (1, maxlen)
        mask = ops.where(mask < q_lengths, mindspore.tensor(1), mindspore.tensor(0))
        return mask

__all__ = [
    "BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST",
    "BigBirdForCausalLM",
    "BigBirdForMaskedLM",
    "BigBirdForMultipleChoice",
    "BigBirdForPreTraining",
    "BigBirdForQuestionAnswering",
    "BigBirdForSequenceClassification",
    "BigBirdForTokenClassification",
    "BigBirdLayer",
    "BigBirdModel",
    "BigBirdPreTrainedModel",
]
