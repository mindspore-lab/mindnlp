# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
# Copyright 2022 The HuggingFace Inc. team.
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
"""MindSpore ERNIE model."""


import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.injection import LESS_MS_2_2
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_ernie import ErnieConfig


logger = logging.get_logger(__name__)


ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nghuyong/ernie-1.0-base-zh",
    "nghuyong/ernie-2.0-base-en",
    "nghuyong/ernie-2.0-large-en",
    "nghuyong/ernie-3.0-base-zh",
    "nghuyong/ernie-3.0-medium-zh",
    "nghuyong/ernie-3.0-mini-zh",
    "nghuyong/ernie-3.0-micro-zh",
    "nghuyong/ernie-3.0-nano-zh",
    "nghuyong/ernie-gram-zh",
    "nghuyong/ernie-health-zh",
    # See all ERNIE models at https://hf-mirror.com/models?filter=ernie
]


class MSErnieEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes an instance of the `MSErnieEmbeddings` class.
        
        Args:
            self: The object itself.
            config: An instance of the `Config` class containing the configuration parameters for the embeddings.
            
        Returns:
            None
            
        Raises:
            None
            
        Description:
            This method initializes the `MSErnieEmbeddings` object by setting up the necessary embedding layers and other
            attributes. It takes in the `config` object which contains the configuration parameters for the embeddings.
            
            - `word_embeddings`: A `nn.Embedding` layer that maps word indices to word embeddings. It has dimensions
            (config.vocab_size, config.hidden_size) and uses the `config.pad_token_id` as the padding index.
            - `position_embeddings`: A `nn.Embedding` layer that maps position indices to position embeddings.
            It has dimensions (config.max_position_embeddings, config.hidden_size).
            - `token_type_embeddings`: A `nn.Embedding` layer that maps token type indices to token type embeddings.
            It has dimensions (config.type_vocab_size, config.hidden_size).
            - `use_task_id`: A boolean indicating whether to use task type embeddings. If `True`, an additional
            `task_type_embeddings` layer is created with dimensions (config.task_type_vocab_size, config.hidden_size).
            - `LayerNorm`: A `nn.LayerNorm` layer that applies layer normalization to the embeddings.
            It has dimensions [config.hidden_size] and uses `config.layer_norm_eps` as epsilon.
            - `dropout`:
                A `nn.Dropout` layer that applies dropout to the embeddings with probability `config.hidden_dropout_prob`.
            - `position_embedding_type`:
                A string indicating the type of position embeddings to use. It defaults to 'absolute'.
            - `position_ids`: A tensor containing the position indices.
                It is created using `ops.arange` and has dimensions (1, config.max_position_embeddings).
            - `token_type_ids`: A tensor containing the token type indices. It is created using `ops.zeros` with the same
            dimensions as `position_ids`.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.use_task_id = config.use_task_id
        if config.use_task_id:
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        Constructs the MSErnie embeddings for the given input.

        Args:
            self (MSErnieEmbeddings): The instance of the MSErnieEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the token ids. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The tensor containing the token type ids. Default is None.
            task_type_ids (Optional[mindspore.Tensor]): The tensor containing the task type ids. Default is None.
            position_ids (Optional[mindspore.Tensor]): The tensor containing the position ids. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing the input embeddings. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            mindspore.Tensor: The tensor representing the constructed embeddings.

        Raises:
            None
        """
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
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # add `task_type_id` for ERNIE model
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings += task_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->Ernie
class MSErnieSelfAttention(nn.Cell):

    """
    This class represents the self-attention mechanism for the MSErnie model.
    It calculates attention scores between input sequences and produces context layers based on the attention weights.
    The class inherits from nn.Cell and is designed to be used within the MSErnie model for natural language processing
    tasks.

    Attributes:
        num_attention_heads (int): The number of attention heads in the self-attention mechanism.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): The total size of all attention heads combined.
        query (nn.Dense): A dense layer for query transformations.
        key (nn.Dense): A dense layer for key transformations.
        value (nn.Dense): A dense layer for value transformations.
        dropout (nn.Dropout): Dropout layer for attention probabilities.
        position_embedding_type (str): The type of position embedding used in the self-attention mechanism.
        max_position_embeddings (int): The maximum number of position embeddings.
        distance_embedding (nn.Embedding): Embedding layer for distance-based positional encodings.
        is_decoder (bool): Indicates if the self-attention mechanism is used in a decoder context.

    Methods:
        transpose_for_scores:
            Transposes the input tensor to prepare it for attention score calculations.

        construct:
            Constructs the self-attention mechanism using the provided input tensors and masks.
            It calculates attention scores, applies position embeddings, performs softmax, and produces context layers.

            Args:

            - hidden_states (mindspore.Tensor): The input tensor to the self-attention mechanism.
            - attention_mask (Optional[mindspore.Tensor]): Optional tensor for masking attention scores.
            - head_mask (Optional[mindspore.Tensor]): Optional tensor for masking attention heads.
            - encoder_hidden_states (Optional[mindspore.Tensor]):
                Hidden states from an encoder, if cross-attention is used.
            - encoder_attention_mask (Optional[mindspore.Tensor]): Attention mask for encoder hidden states.
            - past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                Cached key and value tensors from previous iterations.
            - output_attentions (Optional[bool]): Flag to indicate whether to output attention weights.

            Returns:

            - Tuple[mindspore.Tensor]:
                Tuple containing the context layer and optionally the attention probabilities and cached key-value pairs.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the MSErnieSelfAttention class.

        Args:
            self: The object instance.
            config:
                The configuration object containing various parameters.

                - Type: object
                - Purpose: Specifies the configuration settings for the self-attention module.
                - Restrictions: None

            position_embedding_type:
                The type of position embedding to use.

                - Type: str or None
                - Purpose: Specifies the type of position embedding to use in the self-attention module.
                - Restrictions: If None, the position_embedding_type will default to 'absolute'.
        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and
                the config object does not have an 'embedding_size' attribute.
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

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method transposes the input tensor for attention scores calculation.

        Args:
            self (MSErnieSelfAttention): The instance of the MSErnieSelfAttention class.
            x (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor:
                A transposed tensor of shape (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        This method constructs the self-attention mechanism for MSErnie model.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states. Shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                An optional attention mask tensor. Shape (batch_size, num_heads, sequence_length, sequence_length).
            head_mask (Optional[mindspore.Tensor]):
                An optional head mask tensor for controlling the attention heads. Shape (num_heads,).
            encoder_hidden_states (Optional[mindspore.Tensor]):
                Optional encoder hidden states for cross-attention. Shape (batch_size, encoder_seq_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]):
                Optional attention mask for encoder_hidden_states.
                Shape (batch_size, num_heads, sequence_length, encoder_seq_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                Optional tuple of past key and value tensors. Shape ((past_key_tensor, past_value_tensor)).
            output_attentions (Optional[bool]): Flag to output attentions. Default is False.

        Returns:
            Tuple[mindspore.Tensor]: Tuple containing the context layer tensor and optionally
                the attention probabilities tensor.
            The context layer tensor represents the output of the self-attention mechanism.
                Shape (batch_size, sequence_length, hidden_size).
            The attention probabilities tensor represents the attention distribution.
                Shape (batch_size, num_heads, sequence_length, encoder_seq_length).

        Raises:
            ValueError: If the dimensions of input tensors are not compatible for matrix multiplication.
            IndexError: If accessing past key and value tensors leads to index out of range.
            RuntimeError: If there is an issue with the computation or masking operations.
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
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
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

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = mindspore.Tensor(key_length - 1, dtype=mindspore.int64).view(
                    -1, 1
                )
            else:
                position_ids_l = ops.arange(query_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(key_length, dtype=mindspore.int64).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = ops.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / ops.sqrt(ops.scalar_to_tensor(self.attention_head_size, attention_scores.dtype))
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ErnieModel forward() function)
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
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Ernie
class MSErnieSelfOutput(nn.Cell):

    """
    MSErnieSelfOutput represents the self-output layer of the Ernie model in MindSpore.

    This class inherits from nn.Cell and contains methods for initializing and constructing the self-output layer,
    which includes dense, LayerNorm, and dropout operations.

    Attributes:
        dense (nn.Dense): The dense layer for linear transformation of hidden states.
        LayerNorm (nn.LayerNorm): The layer normalization for normalizing hidden states.
        dropout (nn.Dropout): The dropout layer for adding regularization to hidden states.

    Methods:
        __init__: Initializes the MSErnieSelfOutput instance with the provided configuration.
        construct: Constructs the self-output layer by performing dense, dropout, and LayerNorm operations on
            the hidden states.

    Returns:
        mindspore.Tensor: The output tensor after passing through the self-output layer transformations.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieSelfOutput class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the MSErnieSelfOutput class.

                - Type: object
                - Purpose: Specifies the configuration settings for the MSErnieSelfOutput instance.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the output of the MSErnieSelfOutput class by performing a series of operations on the
        input hidden_states and input_tensor.

        Args:
            self (MSErnieSelfOutput): The instance of the MSErnieSelfOutput class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            input_tensor (mindspore.Tensor): The input tensor used for the addition operation.

        Returns:
            mindspore.Tensor: The output tensor after the series of operations.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Ernie
class MSErnieAttention(nn.Cell):

    """
    This class represents the attention mechanism used in the MSErnie model.
    It is responsible for calculating the attention scores between the input sequence and itself or encoder hidden states.
    The attention scores are then used to weigh the importance of different parts of the input sequence during the
    model's computation.

    This class inherits from the nn.Cell class.

    Methods:
        __init__: Initializes the MSErnieAttention instance.
        prune_heads: Prunes the specified attention heads from the model.
        construct: Constructs the attention mechanism by calculating attention scores and applying them to the
            input sequence.

    Attributes:
        self: An instance of MSErnieSelfAttention, representing the self-attention mechanism.
        self_attn: An instance of MSErnieSelfAttention,
            representing the self-attention mechanism (used in older versions of MindSpore).
        output: An instance of MSErnieSelfOutput, representing the output layer of the attention mechanism.
        pruned_heads: A set that stores the indices of the pruned attention heads.

    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the MSErnieAttention class.

        Args:
            self: The object itself.
            config: An object of type 'config' containing configuration settings.
            position_embedding_type: (Optional) A string specifying the type of position embedding. Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = MSErnieSelfAttention(config, position_embedding_type=position_embedding_type)
        if LESS_MS_2_2:
            self.self_attn = self.self
        self.output = MSErnieSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes the specified attention heads from the MSErnieAttention layer.

        Args:
            self (MSErnieAttention): The instance of the MSErnieAttention class.
            heads (List[int]): A list of attention head indices to be pruned.

        Returns:
            None

        Raises:
            None
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
        self.output.dense = prune_linear_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        This method constructs the MSErnieAttention module.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor):
                The input tensor of shape (batch_size, sequence_length, hidden_size) containing the hidden states.
            attention_mask (Optional[mindspore.Tensor]):
                An optional input tensor of shape (batch_size, sequence_length) representing the attention mask for
                the input sequence. Defaults to None.
            head_mask (Optional[mindspore.Tensor]):
                An optional tensor of shape (num_heads,) representing the mask for the attention heads.
                Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                An optional tensor of shape (batch_size, sequence_length, hidden_size) containing the hidden states of
                the encoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                An optional tensor of shape (batch_size, sequence_length) representing the attention mask for the
                encoder hidden states. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple containing the past key and value tensors for fast decoding. Defaults to None.
            output_attentions (Optional[bool]): A boolean flag indicating whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor of shape
                (batch_size, sequence_length, hidden_size).

        Raises:
            No specific exceptions are raised by this method.
        """
        if LESS_MS_2_2:
            self_outputs = self.self_attn(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        else:
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Ernie
class MSErnieIntermediate(nn.Cell):

    """
    This class represents the intermediate layer of the MSErnie model, which is used for feature extraction and transformation.

    The MSErnieIntermediate class inherits from the nn.Cell class, which is a base class for all neural network layers
    in the MindSpore framework.

    Attributes:
        dense (nn.Dense): A fully connected layer that transforms the input tensor to the hidden size defined
            in the configuration.
        intermediate_act_fn (function): The activation function applied to the hidden states after the dense layer.

    Methods:
        __init__(self, config): Initializes the MSErnieIntermediate instance with the given configuration.
        construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
            Performs the forward pass of the intermediate layer.

    """
    def __init__(self, config):
        """Initializes an instance of the MSErnieIntermediate class.

        Args:
            self: An instance of the MSErnieIntermediate class.
            config:
                A configuration object that contains the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str or function): The activation function to use in the hidden layer.
                If a string is provided, the corresponding function will be retrieved from the ACT2FN dictionary.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method to construct intermediate hidden states in the MSErnieIntermediate class.

        Args:
            self (MSErnieIntermediate): The instance of the MSErnieIntermediate class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states to be processed.

        Returns:
            mindspore.Tensor: The processed hidden states after passing through the dense layer and activation function.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Ernie
class MSErnieOutput(nn.Cell):

    """
    MSErnieOutput is a class that represents the output layer for the MSErnie model in MindSpore.
    This class inherits from nn.Cell and contains methods to process hidden states and input tensors.

    Attributes:
        dense (nn.Dense): A fully connected layer to transform the hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module to normalize the hidden states.
        dropout (nn.Dropout): A dropout layer to apply dropout to the hidden states.

    Methods:
        __init__: Initializes the MSErnieOutput class with the provided configuration.
        construct: Processes the hidden states and input tensor to generate the output tensor.

    Note:
        This class is specifically designed for the MSErnie model in MindSpore and should be used as the final output layer.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieOutput class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' containing configuration settings for the MSErnieOutput.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output tensor of the MSErnie model.

        Args:
            self (MSErnieOutput): The instance of the MSErnieOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor generated by the model.
                Shape: (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor to the layer.
                Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The output tensor of the MSErnie model after processing the hidden states and input tensor.
                Shape: (batch_size, sequence_length, hidden_size).

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the shapes of the input tensors are incompatible for addition.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Ernie
class MSErnieLayer(nn.Cell):

    """
    This class represents a layer of the MSErnie model, designed for natural language processing tasks.
    The MSErnieLayer class is responsible for handling self-attention and cross-attention mechanisms within the model.
    It inherits from nn.Cell and contains methods for initialization, constructing the layer,
    and performing feed-forward operations on the attention output.

    Attributes:
        chunk_size_feed_forward: The size of the chunk for feed-forward operations.
        seq_len_dim: The dimension of the sequence length.
        attention: The self-attention mechanism used in the layer.
        is_decoder: A flag indicating whether the layer is used as a decoder in the model.
        add_cross_attention: A flag indicating whether cross-attention is added to the layer.
        crossattention: The cross-attention mechanism used in the layer.
        intermediate: The intermediate layer in the feed-forward network.
        output: The output layer in the feed-forward network.

    Methods:
        __init__: Initializes the MSErnieLayer with the provided configuration.
        construct: Constructs the layer by processing the input hidden states and optional arguments.
        feed_forward_chunk: Performs feed-forward operations on the attention output to generate the final layer output.

    Note:
        If cross-attention is added, the layer should be used as a decoder model.
        Instantiation with cross-attention layers requires setting `config.add_cross_attention=True`.
        The construct method processes hidden states and optional arguments to generate the final outputs.
        The feed_forward_chunk method handles the feed-forward operations on the attention output to produce the layer output.
    """
    def __init__(self, config):
        """
        Initializes a MSErnieLayer object with the given configuration.

        Args:
            self: The MSErnieLayer instance.
            config:
                An object containing the configuration parameters for the MSErnieLayer.

                - chunk_size_feed_forward (int): The chunk size for feed forward operations.
                - is_decoder (bool): Indicates if the layer is used as a decoder model.
                - add_cross_attention (bool): Indicates if cross attention is added.
                - position_embedding_type (str): The type of position embedding for cross attention.
                Only applicable if add_cross_attention is True.

        Returns:
            None.

        Raises:
            ValueError: If add_cross_attention is True and the layer is not used as a decoder model.

        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MSErnieAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = MSErnieAttention(config, position_embedding_type="absolute")
        self.intermediate = MSErnieIntermediate(config)
        self.output = MSErnieOutput(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        Constructs an MSErnieLayer.

        This method applies the MSErnie layer to the input hidden states and returns the output of the layer.
        The MSErnie layer consists of self-attention, cross-attention (if decoder), and feed-forward sublayers.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states. Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask indicating which positions should be attended to and which should be ignored.
                Shape: (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]):
                The mask for the individual attention heads.
                Shape: (num_heads,) or (num_layers, num_heads) or (batch_size, num_heads, sequence_length, sequence_length).
            encoder_hidden_states (Optional[mindspore.Tensor]):
                The hidden states of the encoder if cross-attention is enabled.
                Shape: (batch_size, encoder_sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]):
                The attention mask for the encoder if cross-attention is enabled.
                Shape: (batch_size, encoder_sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The cached key-value pairs of the self-attention and cross-attention layers from previous steps.
                Shape: (2, num_layers, num_heads, sequence_length, key_value_size).
            output_attentions (Optional[bool]): Whether to output the attention weights. Default: False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the output of the MSErnie layer.
                The first element is the output of the feed-forward sublayer.
                If the layer is a decoder, the tuple also includes the cached key-value pairs for self-attention
                and cross-attention.

        Raises:
            ValueError: If `encoder_hidden_states` are provided but the model is not instantiated with cross-attention
                layers by setting `config.add_cross_attention=True`.
        """
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
            present_key_value = None

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
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

        # do not support `apply_chunking_to_forward` on graph mode
        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )
        layer_output = self.feed_forward_chunk(attention_output)
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Performs a feed forward chunk operation on the given attention output.

        Args:
            self (MSErnieLayer): An instance of the MSErnieLayer class.
            attention_output: The attention output tensor to be processed.
                It should be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method does not directly return any value. Instead, it updates the layer output.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Ernie
class MSErnieEncoder(nn.Cell):

    """
    MSErnieEncoder represents a customized encoder for the MSErnie model that inherits from nn.Cell.

    Attributes:
        config: A dictionary containing configuration parameters for the encoder.
        layer: A CellList containing MSErnieLayer instances for each hidden layer in the encoder.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled in the encoder.

    Methods:
        __init__: Initializes the MSErnieEncoder with the given configuration.
        construct: Constructs the forward
        pass of the encoder with optional outputs based on the input parameters.

    Returns:
        Union[Tuple[mindspore.Tensor], dict]:
            A tuple containing relevant output tensors or a dictionary with optional outputs based on the method parameters.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieEncoder class.

        Args:
            self (object): The instance of the MSErnieEncoder class.
            config (object): Configuration object containing parameters for the MSErnieEncoder.
                This object should include the following attributes:

                - num_hidden_layers (int): Number of hidden layers for the encoder.
                - Other configuration parameters specific to the MSErnieEncoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([MSErnieLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        """
        This method constructs the MSErnie encoder with the provided input parameters and returns the
        output hidden states, decoder cache, all hidden states, self attentions, and cross attentions.

        Args:
            self: The instance of the MSErnieEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states to the encoder.
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid attention on padding tokens.
            head_mask (Optional[mindspore.Tensor]): Mask for masked multi-head attention.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Tuple of past key values for fast decoding.
            use_cache (Optional[bool]): Flag to use the cache for decoding.
            output_attentions (Optional[bool]): Flag to output attentions.
            output_hidden_states (Optional[bool]): Flag to output hidden states.

        Returns:
            Union[Tuple[mindspore.Tensor], dict]: Depending on the output flags, returns a tuple containing hidden states,
                next decoder cache, all hidden states, self attentions, and cross attentions. If any of these values are
                None, they are excluded from the tuple.

        Raises:
            None

        """
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

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


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Ernie
class MSErniePooler(nn.Cell):

    """
    This class represents a pooler for the MSErnie model. It inherits from nn.Cell.

    Attributes:
        dense (nn.Dense): A fully connected layer used for pooling operations.
        activation (nn.Tanh): An activation function applied to the pooled output.

    Methods:
        __init__: Initializes the MSErniePooler class.
        construct: Constructs the pooled output tensor.

    """
    def __init__(self, config):
        """
        __init__

        Initializes an instance of the MSErniePooler class.

        Args:
            self (MSErniePooler): The instance of the MSErniePooler class.
            config: The configuration object containing the parameters for the MSErniePooler instance.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required attributes.
            RuntimeError: If there is an issue with initializing the dense layer or activation function.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method is part of the class MSErniePooler and is used to construct a pooled output from the given hidden states tensor.

        Args:
            self: The instance of the MSErniePooler class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                It is expected to be of shape (batch_size, sequence_length, hidden_size)
                where batch_size represents the number of input sequences in the batch, sequence_length represents
                the length of the sequences, and hidden_size represents the size of the hidden states.
                The hidden states are the output of the Ernie model and are used to construct the pooled output.

        Returns:
            mindspore.Tensor: The constructed pooled output tensor.
                It represents the aggregated representation of the input sequences and is of shape
                (batch_size, hidden_size) where batch_size represents the number of input sequences in the batch and
                hidden_size represents the size of the hidden states.

        Raises:
            None
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Ernie
class MSErniePredictionHeadTransform(nn.Cell):

    '''
    The MSErniePredictionHeadTransform class represents a transformation module for an ERNIE prediction head.
    This class inherits from nn.Cell and is used to process hidden states for ERNIE predictions.

    Attributes:
        dense: A fully connected neural network layer with input and output size of config.hidden_size.
        transform_act_fn: Activation function used for transforming hidden states.
        LayerNorm: Layer normalization module with hidden size specified by config.hidden_size
            and epsilon specified by config.layer_norm_eps.

    Methods:
        __init__: Initializes the MSErniePredictionHeadTransform instance with the provided configuration.
        construct: Applies transformations to the input hidden states and returns the processed hidden states.

    Usage:
        Instantiate an MSErniePredictionHeadTransform object with the desired configuration and utilize the
        construct method to process hidden states for ERNIE predictions.
    '''
    def __init__(self, config):
        """
        Initializes the MSErniePredictionHeadTransform class.

        Args:
            self (MSErniePredictionHeadTransform): The instance of the MSErniePredictionHeadTransform class.
            config:
                A configuration object containing settings for the transformation.

                - Type: object
                - Purpose: Specifies the configuration parameters for the transformation.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            TypeError: If the configuration object is not of the expected type.
            KeyError: If the specified hidden activation function is not found in the ACT2FN dictionary.
            ValueError: If there are issues with the provided configuration parameters.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method 'construct' in the class 'MSErniePredictionHeadTransform' processes the hidden states using
        a series of transformations and returns the processed hidden states as a 'mindspore.Tensor'  object.

        Args:
            self (MSErniePredictionHeadTransform): The instance of the class MSErniePredictionHeadTransform.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.
                It should be a tensor object containing the hidden states information.

        Returns:
            mindspore.Tensor:
                Returns the processed hidden states after applying dense layer, activation function,
                and layer normalization.

        Raises:
            No specific exceptions are documented to be raised by this method.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Ernie
class MSErnieLMPredictionHead(nn.Cell):

    """
    This class represents a prediction head for the MSErnie language model, which is used for language modeling tasks.
    It is a subclass of `nn.Cell`.

    Attributes:
        transform (MSErniePredictionHeadTransform):
            An instance of the MSErniePredictionHeadTransform class that applies transformations to the input hidden states.
        decoder (nn.Dense):
            A fully connected layer that takes the transformed hidden states as input and produces predictions.
        bias (Parameter): The bias term used in the fully connected layer.

    Methods:
        __init__: Initializes an instance of the MSErnieLMPredictionHead class.
        construct: Applies transformations and produces predictions based on the input hidden states.

    """
    def __init__(self, config):
        """
        Initialize the MSErnieLMPredictionHead class.

        Args:
            self: The current instance of the class.
            config: An object containing configuration settings for the prediction head.
                It is expected to have attributes including 'hidden_size' and 'vocab_size'.
                'hidden_size' specifies the size of the hidden layer, and 'vocab_size' specifies
                the size of the vocabulary.

        Returns:
            None:
                This method initializes various components of the prediction head such as the transform, decoder, and bias.

        Raises:
            None.
        """
        super().__init__()
        self.transform = MSErniePredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size), 'bias')

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        Constructs the MSErnieLMPredictionHead.

        Args:
            self (MSErnieLMPredictionHead): An instance of the MSErnieLMPredictionHead class.
            hidden_states (Tensor): The input hidden states. Expected shape is (batch_size, sequence_length, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Ernie
class MSErnieOnlyMLMHead(nn.Cell):

    """
    This class represents a prediction head for Masked Language Modeling (MLM) tasks using the MSErnie model.

    This class inherits from nn.Cell and is responsible for constructing prediction scores based on the sequence output
    from the MSErnie model.

    Attributes:
        predictions (MSErnieLMPredictionHead): Instance of MSErnieLMPredictionHead used for generating prediction scores.

    Methods:
        construct(sequence_output: mindspore.Tensor) -> mindspore.Tensor:
            Constructs prediction scores based on the input sequence_output tensor.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieOnlyMLMHead class.

        Args:
            self: The instance of the MSErnieOnlyMLMHead class.

            config:
                A configuration object containing settings for the MSErnieOnlyMLMHead instance.

                - Type: Any
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: The config object must be compatible with the MSErnieOnlyMLMHead class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.predictions = MSErnieLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs the masked language model (MLM) head for the MSErnie model.

        Args:
            self (MSErnieOnlyMLMHead): The instance of the MSErnieOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The output tensor from the preceding layer, typically the encoder.
                It represents the sequence output that will be used for predicting masked tokens.

        Returns:
            mindspore.Tensor: The prediction scores tensor generated by the MLM head.
                This tensor contains the predicted scores for each token in the input sequence.

        Raises:
            None
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->Ernie
class MSErnieOnlyNSPHead(nn.Cell):

    """
    The `MSErnieOnlyNSPHead` class is a subclass of `nn.Cell` that represents a neural network head for the MSErnie model,
    specifically designed for the Next Sentence Prediction (NSP) task.

    This class initializes an instance of `MSErnieOnlyNSPHead` with a configuration object, which is used to define
    the hidden size of the model.
    The `config` parameter should be an instance of `MSErnieConfig` or a class derived from it.

    The `construct` method takes a `pooled_output` tensor as input and computes the next sentence prediction score
    using a dense layer.
    The `pooled_output` tensor should be of shape (batch_size, hidden_size), where `hidden_size` is the size of
    the hidden layers in the model.

    The `seq_relationship` attribute is an instance of `nn.Dense` that performs the computation of the next sentence
    prediction score.
    It takes the `pooled_output` tensor as input and returns a tensor of shape (batch_size, 2),
    where the second dimension represents the probability scores for two possible sentence relationships.

    The `construct` method returns the computed `seq_relationship_score` tensor.

    Example:
        ```python
        >>> config = MSErnieConfig(hidden_size=768)
        >>> head = MSErnieOnlyNSPHead(config)
        >>> output = head.construct(pooled_output)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieOnlyNSPHead class.

        Args:
            self (MSErnieOnlyNSPHead): An instance of the MSErnieOnlyNSPHead class.
            config:
                A configuration object containing the model's settings.

                - Type: Any valid object
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        """
        This method constructs the sequence relationship score based on the pooled output for the MSErnieOnlyNSPHead class.

        Args:
            self (object): The instance of the MSErnieOnlyNSPHead class.
            pooled_output (object): The pooled output obtained from the model.

        Returns:
            None: This method does not return any value, but calculates the sequence relationship score
                based on the pooled output.

        Raises:
            NNone.
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->Ernie
class MSErniePreTrainingHeads(nn.Cell):

    """
    This class represents the pre-training heads of the MSErnie model, which includes prediction scores and
    sequence relationship scores.

    The class inherits from the nn.Cell class.

    Attributes:
        predictions (MSErnieLMPredictionHead):
            An instance of the MSErnieLMPredictionHead class, responsible for generating prediction scores
            based on sequence outputs.
        seq_relationship (nn.Dense):
            A fully connected layer that produces sequence relationship scores based on pooled outputs.

    Methods:
        construct(sequence_output, pooled_output):
            Constructs the pre-training heads by generating prediction scores and sequence relationship scores
            based on the given sequence and pooled outputs.

            Args:

            - sequence_output (Tensor): The sequence output from the MSErnie model.
            - pooled_output (Tensor): The pooled output from the MSErnie model.

            Returns:

            - prediction_scores (Tensor): The prediction scores generated by the predictions module.
            - seq_relationship_score (Tensor): The sequence relationship scores generated by the seq_relationship module.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErniePreTrainingHeads class.

        Args:
            self (MSErniePreTrainingHeads): The instance of the class itself.
            config: An object containing the configuration parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.predictions = MSErnieLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        """
        This method constructs prediction scores and sequence relationship scores for pre-training tasks in the MSErnie model.

        Args:
            self (object): The instance of the MSErniePreTrainingHeads class.
            sequence_output (object):
                The output sequence generated by the model.

                - Type: Any
                - Purpose: Represents the input sequence for pre-training tasks.
                - Restrictions: Should be a valid output sequence object.
            pooled_output (object):
                The pooled output generated by the model.

                - Type: Any
                - Purpose: Represents the pooled output for pre-training tasks.
                - Restrictions: Should be a valid pooled output object.

        Returns:
            tuple:
                A tuple containing the prediction scores and sequence relationship score.

                - Type: tuple
                - Purpose: Contains the prediction scores for the input sequence and the sequence relationship score.
                - Restrictions: None

        Raises:
            None
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MSErniePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ErnieConfig
    base_model_prefix = "ernie"
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


class MSErnieModel(MSErniePreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Ernie
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes an instance of the MSErnieModel class.

        Args:
            self: The object instance.
            config (object): The configuration object that contains the model parameters.
            add_pooling_layer (bool): Indicates whether to add a pooling layer. Default is True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embeddings = MSErnieEmbeddings(config)
        self.encoder = MSErnieEncoder(config)

        self.pooler = MSErniePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.get_input_embeddings
    def get_input_embeddings(self):
        """
        Get the input embeddings for the MSErnieModel.

        Args:
            self (MSErnieModel): An instance of the MSErnieModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    # Copied from transformers.models.bert.modeling_bert.BertModel.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MSErnieModel.

        Args:
            self (MSErnieModel): The instance of the MSErnieModel class.
            value: The new input embeddings to be set. This should be of type torch.Tensor.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = value

    # Copied from transformers.models.bert.modeling_bert.BertModel._prune_heads
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
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
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

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
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
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

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
            task_type_ids=task_type_ids,
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
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class MSErnieForPreTraining(MSErniePreTrainedModel):

    """
    MSErnieForPreTraining is a class that extends MSErniePreTrainedModel and is designed for pre-training the
    Ernie model for masked language modeling and next sentence prediction tasks.

    The class includes methods for initializing the model with configuration, getting and setting output embeddings,
    and constructing the model for training. The 'construct' method takes various input tensors
    such as input_ids, attention_mask, token_type_ids, etc., and computes the total loss for masked language modeling
    and next sentence prediction. The method returns the total loss, prediction scores, sequence relationship scores,
    and additional outputs if specified.

    Example usage of the MSErnieForPreTraining class involves initializing a tokenizer and the model, processing inputs
    using the tokenizer, and obtaining prediction and sequence relationship logits from the model's outputs.
    """
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of MSErnieForPreTraining.

        Args:
            self (object): The instance of the class.
            config (object): Configuration object containing settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.ernie = MSErnieModel(config)
        self.cls = MSErniePreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings from the MSErnieForPreTraining model.

        Args:
            self: An instance of the MSErnieForPreTraining class.

        Returns:
            The output embeddings from the model.

        Raises:
            None.

        Example:
            ```python
            >>> model = MSErnieForPreTraining()
            >>> embeddings = model.get_output_embeddings()
            ```
        """
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings in the MSErnieForPreTraining model.

        Args:
            self (MSErnieForPreTraining): The instance of the MSErnieForPreTraining class.
                This parameter represents the current instance of the class.
            new_embeddings (object): The new output embeddings to be set in the model.
                This parameter should be of the desired type for output embeddings.

        Returns:
            None.

        Raises:
            None.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        next_sentence_label: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:
            Union[Tuple[mindspore.Tensor], dict]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, ErnieForPreTraining
            ...
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
            >>> model = ErnieForPreTraining.from_pretrained("nghuyong/ernie-1.0-base-zh")
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
            ```
        """
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = ops.cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        output = (prediction_scores, seq_relationship_score) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


class MSErnieForCausalLM(MSErniePreTrainedModel):

    """
    MSErnieForCausalLM
    ------------------

    This class is an implementation of the MSErnie model for causal language modeling (LM).
    It inherits from the MSErniePreTrainedModel class.

    Attributes:
        ernie (MSErnieModel): The main MSErnie model.
        cls (MSErnieOnlyMLMHead): The MLM head for generating predictions.

    Methods:
        __init__: Initializes the MSErnieForCausalLM class.
        get_output_embeddings: Retrieves the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model.
        construct: Constructs the MSErnie model for causal language modeling.
        prepare_inputs_for_generation: Prepares the inputs for text generation.
        _reorder_cache: Reorders the cache for beam search decoding.
    """
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.__init__ with BertLMHeadModel->ErnieForCausalLM,Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of MSErnieForCausalLM class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings for the model.

                - Type: Config
                - Purpose: Specifies the model configuration parameters.
                - Restrictions: Must be provided to initialize the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `ErnieForCausalLM` as a standalone, add `is_decoder=True.`")

        self.ernie = MSErnieModel(config, add_pooling_layer=False)
        self.cls = MSErnieOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MSErnieForCausalLM model.

        Args:
            self: The instance of the MSErnieForCausalLM class.

        Returns:
            decoder: The method returns the output embeddings of the model which are of type None.
                These embeddings represent the learned representation of the input data.

        Raises:
            None.

        """
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets new output embeddings for the model.

        Args:
            self (MSErnieForCausalLM): The instance of the MSErnieForCausalLM class.
            new_embeddings (Any): The new embeddings to be set for the output layer.

        Returns:
            None.

        Raises:
            None.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
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
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having
                4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        if labels is not None:
            use_cache = False

        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = ops.cross_entropy(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((lm_loss,) + output) if lm_loss is not None else output

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        """
        Prepare inputs for generation.

        This method prepares the input tensors for generating text using the MSErnie model for causal language modeling.

        Args:
            self (MSErnieForCausalLM): The instance of the MSErnieForCausalLM class.
            input_ids (torch.Tensor): The input tensor containing the tokenized input text.
                Shape: [batch_size, sequence_length].
            past_key_values (Tuple[torch.Tensor]): The past key-value pairs used for fast decoding.
                Each tuple element contains past key-value tensors.
                Shape: [(batch_size, num_heads, sequence_length, hidden_size // num_heads)] * num_layers.
                Default: None.
            attention_mask (torch.Tensor): The attention mask tensor to avoid attending to padding tokens.
                Shape: [batch_size, sequence_length].
                Default: None.
            use_cache (bool): Whether to use the past key-value cache for fast decoding.
                Default: True.
            **model_kwargs: Additional model-specific keyword arguments.

        Returns:
            dict:
                A dictionary containing the prepared input tensors.

                - 'input_ids' (torch.Tensor): The modified input tensor.
                Shape: [batch_size, modified_sequence_length].
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                Shape: [batch_size, modified_sequence_length].
                - 'past_key_values' (Tuple[torch.Tensor]): The past key-value pairs.
                Each tuple element contains past key-value tensors.
                Shape: [(batch_size, num_heads, modified_sequence_length, hidden_size // num_heads)] * num_layers.
                - 'use_cache' (bool): The flag indicating whether to use the past key-value cache.

        Raises:
            None
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache items based on the provided beam index.

        Args:
            self (MSErnieForCausalLM): The instance of the MSErnieForCausalLM class.
            past_key_values (tuple): A tuple containing the past key-value states for each layer.
            beam_idx (torch.Tensor): An index tensor representing the beam index.

        Returns:
            tuple: A tuple containing the reordered past key-value states.

        Raises:
            None.

        Description:
            This method takes in the past key-value states and reorders them based on the provided beam index.
            It returns a tuple containing the reordered past key-value states.

            The 'self' parameter refers to the instance of the MSErnieForCausalLM class in which this method is called.

            The 'past_key_values' parameter is a tuple containing the past key-value states for each layer.
            These states are used to preserve information over time steps during generation.

            The 'beam_idx' parameter is a tensor representing the beam index.
            It is used to determine the order in which the past key-value states should be reordered.

            The method does not raise any exceptions.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class MSErnieForMaskedLM(MSErniePreTrainedModel):

    """
    The `MSErnieForMaskedLM` class is a Python class that represents a model for masked language modeling (MLM) using
    the MSErnie architecture. It is designed to generate predictions for masked tokens in a given input sequence.

    This class inherits from the `MSErniePreTrainedModel` class, which provides the basic functionality and configuration
    for the MSErnie model.

    The `MSErnieForMaskedLM` class contains the following methods:

    - `__init__(self, config)`: Initializes the `MSErnieForMaskedLM` instance with a given configuration.
    It creates the MSErnie model and MLM head, and performs additional initialization steps.
    - `get_output_embeddings`: Returns the decoder layer of the MLM head.
    - `set_output_embeddings`: Sets the decoder layer of the MLM head to the given embeddings.
    - `construct`: Constructs the MSErnie model and performs the forward pass.
    It takes various input tensors and returns the masked language modeling loss and other outputs.
    - `prepare_inputs_for_generation`:
    Prepares the inputs for generation by adding a dummy token for each input sequence and adjusting the
    attention mask accordingly.

    Please note that the detailed docstring provided here omits method signatures and any other code.
    Refer to the actual implementation for complete details on the method signatures and any additional code.

    """
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the `MSErnieForMaskedLM` class.

        Args:
            self: The instance of the class.
            config: An object of type `Config` containing the configuration parameters.

        Returns:
            None

        Raises:
            None

        This method initializes the `MSErnieForMaskedLM` instance by setting up the configuration and the model components.
        It takes in the `config` parameter, which is an object of type `Config` and contains
        the necessary configuration parameters for the model.
        The `self` parameter refers to the instance of the class itself.
        If the `config.is_decoder` attribute is True, a warning message is logged to ensure that the `config.is_decoder`
        attribute is set to False for bi-directional self-attention.
        The method then initializes the `ernie` attribute by creating an instance of the `MSErnieModel` class,
        passing the `config` object and setting the `add_pooling_layer` attribute to False.
        The `cls` attribute is initialized with an instance of the `MSErnieOnlyMLMHead` class, using the `config` object.
        Finally, the `post_init` method is called to perform any additional initialization tasks.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `ErnieForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.ernie = MSErnieModel(config, add_pooling_layer=False)
        self.cls = MSErnieOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.get_output_embeddings
    def get_output_embeddings(self):
        """
        Returns the output embeddings for the MSErnieForMaskedLM model.

        Args:
            self: An instance of the MSErnieForMaskedLM class.

        Returns:
            None: The method returns a value of type 'None'.

        Raises:
            None.

        This method retrieves the output embeddings of the MSErnieForMaskedLM model.
        The output embeddings represent the predicted decoder values for the given input.

        Note:
            The output embeddings are obtained using the `predictions.decoder` attribute of the `self.cls` object.

        Example:
            ```python
            >>> model = MSErnieForMaskedLM()
            >>> embeddings = model.get_output_embeddings()
            ```
        """
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """"
        Sets the output embeddings for the MSErnieForMaskedLM model.

        Args:
            self (object): The instance of the MSErnieForMaskedLM class.
            new_embeddings (object): The new embeddings to be set as the output embeddings for the model.
                Should be of the same type as the current output embeddings.

        Returns:
            None: This method does not return any value.

        Raises:
            None.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        """
        Prepare inputs for generation.

        This method takes three parameters: self, input_ids, and attention_mask.
        It prepares the input data for generation by modifying the input_ids and attention_mask tensors.

        Args:
            self (MSErnieForMaskedLM): The instance of the MSErnieForMaskedLM class.
            input_ids (Tensor): The input tensor of shape [batch_size, sequence_length].
                It contains the input token IDs.
            attention_mask (Tensor, optional): The attention mask tensor of shape [batch_size, sequence_length].
                It is used to mask out the padding tokens. Defaults to None.

        Returns:
            dict:
                A dictionary containing the modified input_ids and attention_mask tensors.

                - 'input_ids' (Tensor): The modified input tensor of shape [batch_size, sequence_length+1].
                It contains the input token IDs with an additional dummy token appended.
                - 'attention_mask' (Tensor): The modified attention mask tensor of shape [batch_size, sequence_length+1].
                It is used to mask out the padding tokens with an additional padding token appended.

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


class MSErnieForNextSentencePrediction(MSErniePreTrainedModel):

    """
    This class represents a model for next sentence prediction using MSErnie, a pre-trained model for natural language
    understanding. It inherits from the MSErniePreTrainedModel class.

    The class has an initializer method that takes a configuration object as input.
    It initializes an instance of the MSErnieModel class and the MSErnieOnlyNSPHead class, and then calls the post_init method.

    The construct method is used to perform the next sentence prediction task. It takes several input tensors,
    such as input_ids, attention_mask, token_type_ids, and labels.
    It returns a tuple containing the next sentence prediction loss, the sequence relationship scores, and additional outputs.

    The labels parameter is optional and is used for computing the next sequence prediction loss.
    The labels should be a tensor of shape (batch_size,) containing indices in the range [0, 1]. A label of 0 indicates
    that sequence B is a continuation of sequence A, while a label of 1 indicates that sequence B is a random sequence.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, MSErnieForNextSentencePrediction
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
        >>> model = MSErnieForNextSentencePrediction.from_pretrained("nghuyong/ernie-1.0-base-zh")
        ...
        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
        ...
        >>> outputs = model(**encoding, labels=mindspore.Tensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```

    Note:
        The 'next_sentence_label' argument in the construct method is deprecated and will be removed in a future version.
        Use the 'labels' argument instead.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForNextSentencePrediction.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of MSErnieForNextSentencePrediction.

        Args:
            self: The instance of the class.
            config: The configuration object containing settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not of the expected type.
            ValueError: If any required attribute in the 'config' object is missing.
        """
        super().__init__(config)

        self.ernie = MSErnieModel(config)
        self.cls = MSErnieOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring). Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

        Returns:
            Union[Tuple[mindspore.Tensor], dict]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, ErnieForNextSentencePrediction
            ...
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0-base-zh")
            >>> model = ErnieForNextSentencePrediction.from_pretrained("nghuyong/ernie-1.0-base-zh")
            ...
            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")
            ...
            >>> outputs = model(**encoding, labels=mindspore.Tensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
            ```
        """
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            next_sentence_loss = ops.cross_entropy(seq_relationship_scores.view(-1, 2), labels.view(-1))

        output = (seq_relationship_scores,) + outputs[2:]
        return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output


class MSErnieForSequenceClassification(MSErniePreTrainedModel):

    """
    This class represents an implementation of MSErnie for sequence classification. It is a subclass of `MSErniePreTrainedModel`.

    Attributes:
        `num_labels` (int): The number of labels for sequence classification.
        `config` (object): The configuration object for MSErnie.
        `ernie` (object): The MSErnieModel instance for feature extraction.
        `dropout` (object): The dropout layer for regularization.
        `classifier` (object): The fully connected layer for classification.
        `problem_type` (str): The type of problem being solved for classification.
            Options are 'regression', 'single_label_classification', and 'multi_label_classification'.

    Methods:
        `construct`: Constructs the MSErnie model for sequence classification.

    """
    # Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """Initializes an instance of the `MSErnieForSequenceClassification` class.

        Args:
            self: The instance of the class.
            config (MSErnieConfig): The configuration object for the model.
                It contains various hyperparameters and settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ernie = MSErnieModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MSErnieForMultipleChoice(MSErniePreTrainedModel):

    """
    MSErnieForMultipleChoice represents a multiple choice question answering model based on the ERNIE
    (Enhanced Representation through kNowledge Integration) architecture.
    This class extends MSErniePreTrainedModel and provides methods for constructing the model,
    including processing input data, computing logits, and calculating loss for training.
    The model utilizes an ERNIE model for encoding input sequences and a classifier for predicting the correct choice
    among multiple options.
    The construct method takes various input tensors such as input_ids, attention_mask, token_type_ids, and labels,
    and returns the loss and reshaped logits for the multiple choice classification task.
    Additionally, the class includes functionality for handling dropout during training and post-initialization tasks.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieForMultipleChoice class.

        Args:
            self (MSErnieForMultipleChoice): The current instance of the MSErnieForMultipleChoice class.
            config:
                An object containing configuration settings for the model.

                - Type: Dict
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: Must be a valid dictionary object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.ernie = MSErnieModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
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

        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels)

        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MSErnieForTokenClassification(MSErniePreTrainedModel):

    """
    MSErnieForTokenClassification is a class that represents a token classification model based on MSErnie
    (MindSpore implementation of ERNIE) for sequence labeling tasks.

    This class inherits from MSErniePreTrainedModel and provides functionality for token classification by utilizing
    an ERNIE-based model architecture.
    It includes methods for initializing the model with configuration parameters, constructing the model
    for inference or training, and computing token classification loss.

    Attributes:
        num_labels (int): The number of labels for token classification tasks.
        ernie (MSErnieModel): The ERNIE model used for token classification.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Dense): Fully connected layer for classification.

    Methods:
        __init__: Initializes the MSErnieForTokenClassification model with the given configuration.
        construct: Constructs the model for inference or training and computes token classification loss if labels
            are provided.

    Args:
        config (object): The configuration object containing model hyperparameters.
        input_ids (mindspore.Tensor, optional): Tensor of input token IDs for the model.
        attention_mask (mindspore.Tensor, optional): Tensor representing the attention mask for input tokens.
        token_type_ids (mindspore.Tensor, optional): Tensor for token type IDs.
        task_type_ids (mindspore.Tensor, optional): Tensor for task type IDs.
        position_ids (mindspore.Tensor, optional): Tensor for position IDs.
        head_mask (mindspore.Tensor, optional): Tensor for head mask.
        inputs_embeds (mindspore.Tensor, optional): Tensor for input embeddings.
        labels (mindspore.Tensor, optional): Tensor of labels for token classification.
        output_attentions (bool, optional): Flag to output attentions.
        output_hidden_states (bool, optional): Flag to output hidden states.

    Returns:
        Union[Tuple[mindspore.Tensor], dict]:
            Tuple containing model outputs and optionally additional information such as attentions and hidden states.

    Raises:
        ValueError: If the number of labels provided is not compatible with the model architecture.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes a new instance of the `MSErnieForTokenClassification` class.

        Args:
            self: The object itself.
            config: An instance of the `MSErnieConfig` class containing the model configuration.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie = MSErnieModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MSErnieForQuestionAnswering(MSErniePreTrainedModel):

    """
    MSErnieForQuestionAnswering represents a model for question answering tasks using the MSErnie architecture.
    This class inherits from MSErniePreTrainedModel and includes methods for initializing the model and constructing the forward pass
    for predicting start and end positions of answers within a text sequence.

    Attributes:
        num_labels (int): The number of labels for the classifier output.
        ernie (MSErnieModel): The MSErnie model used as the base for question answering.
        qa_outputs (nn.Dense): The fully connected layer for predicting start and end positions within the sequence.

    Methods:
        __init__: Initializes the model with the given configuration.
        construct:
            Constructs the forward pass of the model for question answering, predicting start and end positions within
            the input sequence.
            Returns the total loss and output logits for start and end positions, along with any additional model outputs.

    Args:
        config: The configuration object containing model hyperparameters.
        input_ids (mindspore.Tensor): The input token IDs of the sequence.
        attention_mask (mindspore.Tensor): The attention mask to prevent attention to padding tokens.
        token_type_ids (mindspore.Tensor): The token type IDs to distinguish between question and context tokens.
        task_type_ids (mindspore.Tensor): The task type IDs for multi-task learning.
        position_ids (mindspore.Tensor): The position IDs for positional embeddings.
        head_mask (mindspore.Tensor): The mask for attention heads.
        inputs_embeds (mindspore.Tensor): The input embeddings instead of input IDs.
        start_positions (mindspore.Tensor): The start positions of the answer span in the sequence.
        end_positions (mindspore.Tensor): The end positions of the answer span in the sequence.
        output_attentions (bool): Flag to output attentions weights.
        output_hidden_states (bool): Flag to output hidden states of the model.

    Returns:
        Tuple containing the total loss, start logits, end logits, and any additional model outputs.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes a new instance of the MSErnieForQuestionAnswering class.

        Args:
            self: The object instance.
            config:
                An instance of the MSErnieConfig class containing the configuration parameters for the model.

                - Type: MSErnieConfig
                - Purpose: Specifies the model configuration.
                - Restrictions: Must be a valid MSErnieConfig object.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie = MSErnieModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        task_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], dict]:
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
        outputs = self.ernie(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


class MSUIE(MSErniePreTrainedModel):
    """
    Ernie Model with two linear layer on top of the hidden-states output to compute `start_prob` and `end_prob`,
    designed for Universal Information Extraction.

    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct UIE
    """
    def __init__(self, config: ErnieConfig):
        """
        Initializes an instance of the MSUIE class.

        Args:
            self: The instance of the class.
            config (ErnieConfig): The configuration object for the Ernie model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.ernie = MSErnieModel(config)
        self.linear_start = nn.Dense(config.hidden_size, 1)
        self.linear_end = nn.Dense(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`ErnieModel`.
            token_type_ids (Tensor, optional):
                See :class:`ErnieModel`.
            position_ids (Tensor, optional):
                See :class:`ErnieModel`.
            attention_mask (Tensor, optional):
                See :class:`ErnieModel`.

        Example:
            ```python
            >>> import paddle
            >>> from paddlenlp.transformers import UIE, ErnieTokenizer
            >>> tokenizer = ErnieTokenizer.from_pretrained('uie-base')
            >>> model = UIE.from_pretrained('uie-base')
            >>> inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
            >>> inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
            >>> start_prob, end_prob = model(**inputs)
            ```
        """
        outputs = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        start_logits = self.linear_start(sequence_output)
        start_logits = ops.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = ops.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_loss = ops.binary_cross_entropy_with_logits(start_prob, start_positions)
            end_loss = ops.binary_cross_entropy_with_logits(end_prob, end_positions)
            total_loss = (start_loss + end_loss) / 2.0

        output = (start_prob, end_prob) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


__all__ = [
    "ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MSErnieForCausalLM",
    "MSErnieForMaskedLM",
    "MSErnieForMultipleChoice",
    "MSErnieForNextSentencePrediction",
    "MSErnieForPreTraining",
    "MSErnieForQuestionAnswering",
    "MSErnieForSequenceClassification",
    "MSErnieForTokenClassification",
    "MSErnieModel",
    "MSErniePreTrainedModel",
    "MSUIE"
]
