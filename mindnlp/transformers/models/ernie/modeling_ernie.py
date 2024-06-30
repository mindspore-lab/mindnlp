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


import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
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


class ErnieEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes an instance of the ErnieEmbeddings class.
        
        Args:
            self: The instance of the ErnieEmbeddings class.
            config: An object containing configuration parameters for the ErnieEmbeddings class.
                The config object should have the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - pad_token_id (int): The ID of the padding token.
                - max_position_embeddings (int): The maximum number of position embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - use_task_id (bool): Whether to use task IDs.
                - task_type_vocab_size (int): The size of the task type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - position_embedding_type (str): The type of position embedding to use. Default is 'absolute'.

        Returns:
            None

        Raises:
            None
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
        Constructs the embeddings for the ERNIE model.

        Args:
            self (ErnieEmbeddings): The instance of the ErnieEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input tensor of shape [batch_size, sequence_length].
            token_type_ids (Optional[mindspore.Tensor]): The token type tensor of shape [batch_size, sequence_length].
            task_type_ids (Optional[mindspore.Tensor]): The task type tensor of shape [batch_size, sequence_length].
            position_ids (Optional[mindspore.Tensor]): The position ids tensor of shape [batch_size, sequence_length].
            inputs_embeds (Optional[mindspore.Tensor]): The input embeddings tensor of shape [batch_size, sequence_length, embedding_size].
            past_key_values_length (int): The length of past key values.

        Returns:
            mindspore.Tensor: The embeddings tensor of shape [batch_size, sequence_length, embedding_size].

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
class ErnieSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism for the ERNIE (Enhanced Representation through kNowledge Integration) model.
    It is used to compute attention scores and produce context layers during the processing of input data.
    The class inherits from nn.Cell and includes methods for initializing the self-attention mechanism,
    transposing tensors for scoring calculations, and constructing the attention mechanism outputs.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initialize the ErnieSelfAttention class.

        Args:
            self: The instance of the class.
            config:
                An instance of the configuration class containing the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int, optional): The size of the embedding layer.
                If not provided, it is expected to be an attribute of the config.
                - attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
                - position_embedding_type (str, optional): The type of position embedding.
                If not provided, it defaults to 'absolute'.
                - max_position_embeddings (int): The maximum number of position embeddings.
                - is_decoder (bool): Indicates if the model is a decoder.

            position_embedding_type (str, optional): The type of position embedding. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and the config does not
                have an 'embedding_size' attribute.
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
        Transpose the input tensor for calculating attention scores.

        Args:
            self (ErnieSelfAttention): The instance of the ErnieSelfAttention class.
            x (mindspore.Tensor): The input tensor with shape (batch_size, seq_length, hidden_size).

        Returns:
            mindspore.Tensor:
                The transposed tensor with shape (batch_size, num_attention_heads, seq_length, attention_head_size).

        Raises:
            TypeError: If the input tensor is not of type mindspore.Tensor.
            ValueError: If the input tensor shape is not compatible with the expected shape for transposition.
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
        Constructs the self-attention mechanism for the ERNIE model.

        Args:
            self (ErnieSelfAttention): The instance of the ErnieSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of the model.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
                It is a binary tensor of shape (batch_size, sequence_length) where 1 indicates a valid token
                and 0 indicates a padded token. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
                It is a binary tensor of shape (num_attention_heads,) indicating which heads to mask. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                The hidden states of the encoder. Shape: (batch_size, encoder_sequence_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor for the encoder. Shape: (batch_size, encoder_sequence_length). Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The cached key-value pairs from previous attention computations. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attention probabilities. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the context layer tensor and optionally the attention probabilities tensor.
                The context layer tensor has shape (batch_size, sequence_length, hidden_size)
                and represents the output of the self-attention mechanism.

        Raises:
            None: This method does not raise any exceptions.

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

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
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
class ErnieSelfOutput(nn.Cell):

    """
    The ErnieSelfOutput class represents a module for self-attention mechanism in ERNIE (Enhanced Representation
    through kNowledge Integration) model.
    This class inherits from nn.Cell and contains methods to apply dense, layer normalization, and dropout operations
    to the input tensor.

    Attributes:
        dense (nn.Dense): A dense layer to transform the input tensor's hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module to normalize the hidden states.
        dropout (nn.Dropout): A dropout module to apply dropout to the hidden states.

    Methods:
        construct:
            Applies dense, dropout, and layer normalization operations to the input tensor's hidden states
            and returns the output tensor.
    """
    def __init__(self, config):
        """
        Initializes an instance of the ErnieSelfOutput class.

        Args:
            self (ErnieSelfOutput): The instance of the ErnieSelfOutput class.
            config (object): An object containing configuration parameters for the ErnieSelfOutput instance.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are invalid or inconsistent.
            TypeError: If the configuration object is not of the expected type.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the ERNIE self-attention layer.

        Args:
            self (ErnieSelfOutput): An instance of the ErnieSelfOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor.
                It should have a shape of (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor.
                It should have the same shape as the hidden_states tensor.

        Returns:
            mindspore.Tensor: The output tensor of the ERNIE self-attention layer.
                It has the same shape as the input_tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Ernie
class ErnieAttention(nn.Cell):

    '''
    This class represents the ErnieAttention module, which is a part of the ERNIE
    (Enhanced Representation through kNowledge Integration) model.
    The ErnieAttention module is used for self-attention mechanism and output processing.
    It includes methods for head pruning and attention construction.
    This class inherits from nn.Cell and is designed to be used within the ERNIE model architecture for
    natural language processing tasks.
    '''
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the ErnieAttention class.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing the model's settings and hyperparameters.
            position_embedding_type (str, optional): The type of position embedding to be used. Defaults to None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = ErnieSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = ErnieSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'ErnieAttention' and is responsible for
        pruning the attention heads based on the provided 'heads' parameter.

        Args:
            self (ErnieAttention): The instance of the ErnieAttention class.
                This parameter represents the instance of the ErnieAttention class which contains the attention heads to be pruned.

            heads (list): A list of integers representing the indices of attention heads to be pruned.
                This parameter specifies the indices of the attention heads that need to be pruned from the model.

        Returns:
            None: This method does not return any value.
                It operates by modifying the attributes of the ErnieAttention instance in-place.

        Raises:
            None.
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
        This method constructs the attention mechanism for the Ernie model.

        Args:
            self (ErnieAttention): The instance of the ErnieAttention class.
            hidden_states (mindspore.Tensor): The input hidden states for the attention mechanism.
            attention_mask (Optional[mindspore.Tensor]): An optional mask tensor for the attention scores.
                Defaults to None.
            head_mask (Optional[mindspore.Tensor]): An optional mask tensor for controlling the attention heads.
                Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                An optional tensor containing the hidden states of the encoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                An optional mask tensor for the encoder attention scores. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple containing the past key and value tensors. Defaults to None.
            output_attentions (Optional[bool]): A flag indicating whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the attention output tensor and any additional outputs from the attention mechanism.

        Raises:
            None
        """
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
class ErnieIntermediate(nn.Cell):

    '''
    Represents an intermediate layer for the ERNIE (Enhanced Representation through kNowledge Integration) model.
    This class provides methods to perform intermediate operations on input hidden states.

    This class inherits from nn.Cell and contains methods for initialization and constructing the intermediate layer.

    Attributes:
        dense (nn.Dense): A dense layer with the specified hidden size and intermediate size.
        intermediate_act_fn (function): The activation function applied to the intermediate hidden states.

    Methods:
        __init__: Initializes the ERNIE intermediate layer with the provided configuration.
        construct: Constructs the intermediate layer by applying dense and activation functions to the input hidden states.
    '''
    def __init__(self, config):
        """
        Initialize the ErnieIntermediate class with the provided configuration.

        Args:
            self (object): The instance of the ErnieIntermediate class.
            config (object):
                An object containing the configuration parameters.

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str or function): The activation function for the hidden layer.
                If provided as a string, it should be a key in ACT2FN dictionary.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are invalid or missing.
            TypeError: If the provided hidden activation function is not a string or function.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the intermediate layer of the ERNIE model.

        Args:
            self (ErnieIntermediate): An instance of the ErnieIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the output from the previous layer of the ERNIE model.

        Returns:
            mindspore.Tensor: The tensor representing the intermediate hidden states of shape (batch_size, sequence_length, hidden_size).
                It is the result of applying the intermediate layer operations on the input hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Ernie
class ErnieOutput(nn.Cell):

    """
    The ErnieOutput class represents a neural network cell for processing output in the ERNIE model.
    This class inherits from the nn.Cell class and includes methods for initializing and constructing the output layer for the model.

    The __init__ method initializes the ErnieOutput instance with the specified configuration.
    It initializes the dense layer, LayerNorm, and dropout for processing the output.

    The construct method processes the hidden states and input tensor to generate the final output tensor using the
    initialized dense layer, dropout, and LayerNorm.

    """
    def __init__(self, config):
        """
        Initializes an instance of ErnieOutput.

        Args:
            self (ErnieOutput): The instance of the ErnieOutput class.
            config:
                An object containing configuration parameters.

                - Type: Any
                - Purpose: Configuration object specifying model settings.
                - Restrictions: Must be compatible with the specified configuration format.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of the expected type.
            ValueError: If the config parameter does not contain required attributes.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output tensor for the Ernie model.

        Args:
            self (ErnieOutput): The instance of the ErnieOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor generated by the model.
                This tensor is processed through dense layers and normalization.
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden states.
                It serves as additional information for the final output.

        Returns:
            mindspore.Tensor: The constructed output tensor that combines the processed hidden states
                with the input tensor to produce the final output of the Ernie model.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Ernie
class ErnieLayer(nn.Cell):

    """ErnieLayer is a class representing a layer in the Ernie model.
    This class inherits from nn.Cell and contains methods for initializing the layer and constructing
    the layer's feed forward chunk.

    Attributes:
        chunk_size_feed_forward (int): The chunk size for the feed forward operation.
        seq_len_dim (int): The dimension of the sequence length.
        attention (ErnieAttention): The attention mechanism used in the layer.
        is_decoder (bool): Indicates whether the layer is a decoder model.
        add_cross_attention (bool): Indicates whether cross attention is added to the layer.
        crossattention (ErnieAttention): The cross attention mechanism used in the layer.
        intermediate (ErnieIntermediate): The intermediate layer in the Ernie model.
        output (ErnieOutput): The output layer in the Ernie model.

    Methods:
        __init__: Initializes the ErnieLayer with the provided configuration.
        construct: Constructs the layer using the given input tensors and parameters.
        feed_forward_chunk(attention_output): Executes the feed forward operation on the attention output.

    Raises:
        ValueError: If the layer is not instantiated with cross-attention layers when `encoder_hidden_states` are passed.

    Returns:
        Tuple: Outputs of the layer's construct method, including the layer output and present key value if the layer is a decoder model.
    """
    def __init__(self, config):
        """
        Initializes an instance of the ErnieLayer class.

        Args:
            self: The instance of the ErnieLayer class.
            config:
                A configuration object containing various settings for the ErnieLayer.

                - Type: object
                - Purpose: Configures the behavior of the ErnieLayer.
                - Restrictions: Must contain the following attributes:

                    - chunk_size_feed_forward: Chunk size for feed-forward operations.
                    - is_decoder: Boolean indicating whether the layer is used as a decoder model.
                    - add_cross_attention: Boolean indicating whether cross attention is added.
                    - position_embedding_type: Optional parameter specifying the position embedding type for cross
                    attention.

        Returns:
            None.

        Raises:
            ValueError: Raised if cross attention is added but the ErnieLayer is not used as a decoder model.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ErnieAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ErnieAttention(config, position_embedding_type="absolute")
        self.intermediate = ErnieIntermediate(config)
        self.output = ErnieOutput(config)

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
        Constructs an ERNIE (Enhanced Representation through kNowledge Integration) layer.

        Args:
            self: The object itself.
            hidden_states (mindspore.Tensor): The input hidden states for the layer.
            attention_mask (Optional[mindspore.Tensor]): Mask for the attention mechanism. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): Mask for the attention heads. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states from the encoder layer. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for the encoder attention mechanism.
                Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Cached key and value tensors for fast inference.
                Defaults to None.
            output_attentions (Optional[bool]): Whether to return attentions weights. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the layer output tensor.

        Raises:
            ValueError: If `encoder_hidden_states` are passed, and cross-attention layers are not instantiated
                by setting `config.add_cross_attention=True`.
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
        This method calculates the feed-forward output for a chunk in the ErnieLayer.

        Args:
            self (object): The instance of the ErnieLayer class.
            attention_output (object): The attention output from the previous layer,
                expected to be a tensor representing the attention scores.

        Returns:
            None: This method does not return any value explicitly but updates the layer_output attribute of
                the ErnieLayer instance.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Ernie
class ErnieEncoder(nn.Cell):

    """
    The ErnieEncoder class represents a multi-layer Ernie (Enhanced Representation through kNowledge Integration)
    encoder module for processing sequential inputs. It inherits from the nn.Cell class.

    Attributes:
        config: The configuration settings for the ErnieEncoder.
        layer: A list of ErnieLayer instances representing the individual layers of the encoder.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled.

    Methods:
        __init__: Initializes the ErnieEncoder with the provided configuration.
        construct:
            Constructs the ErnieEncoder module with the given inputs and returns the output either as a tuple of tensors
            or as a BaseModelOutputWithPastAndCrossAttentions object.

    Notes:
        - The construct method supports various optional input parameters and returns different types of outputs based
        on the provided arguments.
        - The class supports gradient checkpointing when enabled during training.
    """
    def __init__(self, config):
        """
        Initialize the ErnieEncoder class.

        Args:
            self (ErnieEncoder): The instance of the ErnieEncoder class.
            config (dict): A dictionary containing configuration parameters for the ErnieEncoder.
                It should include the following keys:

                - num_hidden_layers (int): The number of hidden layers in the ErnieEncoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([ErnieLayer(config) for _ in range(config.num_hidden_layers)])
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
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        '''
        Constructs the ErnieEncoder.

        Args:
            self (ErnieEncoder): The instance of the ErnieEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states of the encoder.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the encoder.
                Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values. Defaults to None.
            use_cache (Optional[bool]): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Defaults to False.
            return_dict (Optional[bool]): Whether to return a dictionary. Defaults to True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]: The output of the ErnieEncoder.

        Raises:
            None.
        '''
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

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
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


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->Ernie
class ErniePooler(nn.Cell):

    """
    ErniePooler class represents a pooler layer for an ERNIE model.

    This class inherits from nn.Cell and implements a pooler layer that takes hidden states as input,
    processes the first token tensor through a dense layer and activation function, and returns the pooled output.

    Attributes:
        dense (nn.Dense): A dense layer with the specified hidden size.
        activation (nn.Tanh): A hyperbolic tangent activation function.

    Methods:
        __init__: Initializes the ErniePooler object with the provided configuration.
        construct: Constructs the pooled output from the hidden states input.
    """
    def __init__(self, config):
        """
        Initializes an instance of the ErniePooler class.

        Args:
            self: The instance of the class.
            config:
                An object of type 'config' which contains the configuration parameters.

                - Type: Any valid object.
                - Purpose: Specifies the configuration parameters for the ErniePooler instance.
                - Restrictions: None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a pooled output tensor from the given hidden states.

        Args:
            self (ErniePooler): An instance of the ErniePooler class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                Shape should be (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: A tensor representing the pooled output.
                Shape is (batch_size, hidden_size).

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Ernie
class ErniePredictionHeadTransform(nn.Cell):

    """
    This class represents the transformation head for the ERNIE prediction model.
    It performs various operations such as dense transformation, activation function application,
    and layer normalization on the input hidden states.

    Inherits from:
        nn.Cell

    Attributes:
        dense (nn.Dense): A dense layer used for transforming the input hidden states.
        transform_act_fn (function): The activation function applied to the transformed hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module applied to the hidden states.

    Methods:
        __init__: Initializes the class instance with the provided configuration.
        construct: Applies the transformation operations on the input hidden states and returns the transformed states.

    """
    def __init__(self, config):
        """Initializes an instance of the ErniePredictionHeadTransform class.

        Args:
            self (ErniePredictionHeadTransform): An instance of the ErniePredictionHeadTransform class.
            config: The configuration object containing the settings for the ErniePredictionHeadTransform.

        Returns:
            None

        Raises:
            None
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
        Constructs the ErniePredictionHeadTransform.

        Args:
            self (ErniePredictionHeadTransform): An instance of the ErniePredictionHeadTransform class.
            hidden_states (mindspore.Tensor): The input hidden states to be transformed.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The transformed hidden states. It has the same shape as the input hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Ernie
class ErnieLMPredictionHead(nn.Cell):

    """
    Represents a prediction head for ERNIE Language Model that performs decoding and transformation operations
    on hidden states.

    This class inherits from nn.Cell and provides methods for initializing the prediction head and constructing
    predictions based on the input hidden states.

    Attributes:
        transform: ErniePredictionHeadTransform object for transforming hidden states.
        decoder: nn.Dense object for decoding hidden states into output predictions.
        bias: Parameter object for bias initialization.

    Methods:
        __init__: Initializes the prediction head with the given configuration.
        construct: Constructs predictions based on the input hidden states by applying
            transformation and decoding operations.

    Example:
        ```python
        >>> config = get_config()
        >>> prediction_head = ErnieLMPredictionHead(config)
        >>> hidden_states = get_hidden_states()
        >>> predictions = prediction_head.construct(hidden_states)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the ErnieLMPredictionHead class.

        Args:
            self: The instance of the ErnieLMPredictionHead class.
            config: An object that holds configuration settings for the ErnieLMPredictionHead.
                It is expected to contain properties like hidden_size, vocab_size, and any other relevant
                configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.transform = ErniePredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size), 'bias')

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        This method 'construct' is part of the class 'ErnieLMPredictionHead' and is responsible for constructing
        the hidden states using transformation and decoding.

        Args:
            self: Represents the instance of the class. It is implicitly passed and does not need to be provided as
                an argument.

            hidden_states (Tensor): The input hidden states to be processed. It is expected to be a tensor containing
                the initial hidden states.

        Returns:
            Tensor: The processed hidden states after transformation and decoding.

        Raises:
            TypeError: If the input 'hidden_states' is not of type Tensor.
            ValueError: If the input 'hidden_states' is empty or invalid for transformation and decoding.
            RuntimeError: If there is an issue during the transformation or decoding process.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Ernie
class ErnieOnlyMLMHead(nn.Cell):

    """
    This class represents the implementation of the ErnieOnlyMLMHead, which is used for masked language model (MLM)
    prediction in Ernie language model. It inherits from the nn.Cell class and contains methods for initializing the
    class and constructing MLM predictions based on the input sequence output.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the ErnieOnlyMLMHead class.

        Args:
            self: The instance of the ErnieOnlyMLMHead class.
            config: An object of the ErnieConfig class containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = ErnieLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the masked language model (MLM) head for the ERNIE model.

        Args:
            self (ErnieOnlyMLMHead): An instance of the ErnieOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The output tensor of the ERNIE model's sequence encoder.
                This tensor represents the contextualized representations of input sequences.
                Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The prediction scores generated by the MLM head.
                The prediction scores represent the likelihood of each token being masked and need to be compared with
                the corresponding ground truth labels during the training process.

                Shape: (batch_size, sequence_length, vocab_size).

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->Ernie
class ErnieOnlyNSPHead(nn.Cell):

    """
    Represents a head for Next Sentence Prediction (NSP) task in the ERNIE model.

    This class inherits from the nn.Cell module and provides functionality to predict whether two input sequences
    are consecutive in the ERNIE model. It contains methods to initialize the head and construct the NSP score based
    on the pooled output of the model.

    Methods:
        __init__: Initializes the NSP head with a Dense layer for sequence relationship prediction.
        construct: Constructs the NSP score by passing the pooled output through the Dense layer.

    Attributes:
        seq_relationship: A Dense layer with hidden_size neurons for predicting the relationship between sequences.
    """
    def __init__(self, config):
        """
        Initializes an instance of the ErnieOnlyNSPHead class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration settings.

                - Type: Any
                - Purpose: Specifies the configuration settings for the head.
                - Restrictions: Must be compatible with the nn.Dense module.

        Returns:
            None. This method does not return any value.

        Raises:
            TypeError: If the 'config' parameter is not provided.
            ValueError: If the 'config.hidden_size' value is invalid or incompatible with nn.Dense.
        """
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        """
        Constructs the sequence relationship score based on the pooled output.

        Args:
            self: Instance of the ErnieOnlyNSPHead class.
            pooled_output:
                Tensor containing the pooled output from the model.

                - Type: Tensor
                - Purpose: Represents the output features obtained after pooling.
                - Restrictions: Must be a valid tensor object.

        Returns:
            seq_relationship_score:
                The calculated sequence relationship score based on the pooled output.

                - Type: None
                - Purpose: Represents the score indicating the relationship between two sequences.

        Raises:
            None
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->Ernie
class ErniePreTrainingHeads(nn.Cell):

    """
    The ErniePreTrainingHeads class represents the pre-training heads for ERNIE model, used for predicting masked tokens
    and sequence relationships. It inherits from nn.Cell and provides methods for initializing the prediction heads
    and making predictions.

    Methods:
        __init__: Initializes the ErniePreTrainingHeads instance with the given configuration.
        construct: Constructs the pre-training heads using the sequence output and pooled output, and returns the
            prediction scores and sequence relationship score.

    Attributes:
        predictions: Instance of ErnieLMPredictionHead for predicting masked tokens.
        seq_relationship: Dense layer for predicting sequence relationships.
    """
    def __init__(self, config):
        """
        Initialize the ErniePreTrainingHeads class.

        Args:
            self: The instance of the ErniePreTrainingHeads class.
            config: A configuration object containing the settings for the ErniePreTrainingHeads.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required settings.
        """
        super().__init__()
        self.predictions = ErnieLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        """
        Constructs the prediction scores and sequence relationship scores for the ErniePreTrainingHeads model.

        Args:
            self (ErniePreTrainingHeads): An instance of the ErniePreTrainingHeads class.
            sequence_output (Tensor): The output tensor from the sequence model.
                This tensor contains the contextualized representations for each token in the input sequence.
                Shape: (batch_size, sequence_length, hidden_size)
            pooled_output (Tensor): The output tensor from the pooling model.
                This tensor contains the pooled representation of the input sequence.
                Shape: (batch_size, hidden_size)

        Returns:
            Tuple[Tensor, Tensor]:
                A tuple of prediction scores and sequence relationship scores.

                - prediction_scores (Tensor): The prediction scores for each token in the input sequence.
                Each score represents the probability of the token being masked in pre-training.
                Shape: (batch_size, sequence_length, vocab_size)
                - seq_relationship_score (Tensor): The sequence relationship score.
                This score represents the probability of the input sequence being a continuation of another sequence.
                Shape: (batch_size, num_labels)

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class ErniePreTrainedModel(PreTrainedModel):
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


@dataclass
# Copied from transformers.models.bert.modeling_bert.BertForPreTrainingOutput with Bert->Ernie
class ErnieForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ErnieForPreTraining`].

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


class ErnieModel(ErniePreTrainedModel):
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
        Initializes an instance of the ErnieModel class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing settings for the Ernie model.
            add_pooling_layer (bool): A flag indicating whether to add a pooling layer to the model. Default is True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ErnieEmbeddings(config)
        self.encoder = ErnieEncoder(config)

        self.pooler = ErniePooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.get_input_embeddings
    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the ErnieModel.

        Args:
            self: The instance of the ErnieModel class.

        Returns:
            The input embeddings for the ErnieModel.

        Raises:
            This method does not raise any exceptions.
        """
        return self.embeddings.word_embeddings

    # Copied from transformers.models.bert.modeling_bert.BertModel.set_input_embeddings
    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the ErnieModel.

        Args:
            self (ErnieModel): The instance of the ErnieModel class.
            value: The input embeddings to be set for the ErnieModel.
                It should be of type that is compatible with the embeddings.word_embeddings attribute.

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
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
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

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
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ErnieForPreTraining(ErniePreTrainedModel):

    """
    This class represents an Ernie model for pre-training tasks. It inherits from the ErniePreTrainedModel.

    The class includes methods for initializing the model, getting and setting output embeddings, and constructing the
    model for pre-training tasks. The `construct` method takes various input tensors and optional arguments, and returns
    the output of the model for pre-training. It also includes detailed information about the expected input parameters,
    optional arguments, and return values.

    The class also provides an example of how to use the model for pre-training tasks using the AutoTokenizer and
    example inputs. The example demonstrates how to tokenize input text, generate model outputs, and access specific
    logits from the model.

    For more details on the usage and functionality of the ErnieForPreTraining class, refer to the provided code and
    docstring examples.
    """
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """Initializes an instance of the ErnieForPreTraining class.

        Args:
            self (ErnieForPreTraining): An instance of the ErnieForPreTraining class.
            config (object): The configuration object for the ErnieForPreTraining class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.ernie = ErnieModel(config)
        self.cls = ErniePreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.get_output_embeddings
    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from the ErnieForPreTraining model.

        Args:
            self (ErnieForPreTraining): The instance of the ErnieForPreTraining class.

        Returns:
            None: This method does not return anything but directly accesses and returns the output embeddings
                from the model.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the ErnieForPreTraining model.

        Args:
            self (ErnieForPreTraining): An instance of the ErnieForPreTraining class.
            new_embeddings: The new embeddings to be set for the model predictions decoder. This can be of any type.

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], ErnieForPreTrainingOutput]:
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
            Union[Tuple[mindspore.Tensor], ErnieForPreTrainingOutput]

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = ops.cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ErnieForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieForCausalLM(ErniePreTrainedModel):

    """
    This class represents a causal language modeling model based on the ERNIE
    (Enhanced Representation through kNowledge Integration) architecture.
    It is designed for generating text predictions based on input sequences, with a focus on predicting the next word
    in a sequence.
    The model includes functionality for constructing the model, setting and getting output embeddings, preparing inputs
    for text generation, and reordering cache during generation.

    The class includes methods for initializing the model, constructing the model for inference or training, setting
    and getting output embeddings, preparing inputs for text generation, and reordering cache during generation.

    The 'construct' method constructs the model for inference or training, taking various input tensors such as
    input ids, attention masks, token type ids, and more. It returns the model outputs including the language modeling
    loss and predictions.

    The 'prepare_inputs_for_generation' method prepares input tensors for text generation, including handling past key
    values and attention masks. It returns a dictionary containing the input ids, attention  mask, past key values,
    and use_cache flag.

    The '_reorder_cache' method reorders the past key values during generation based on the beam index used for parallel
    decoding.

    For more detailed information on each method's parameters and return values, refer to the method docstrings within
    the class code.
    """
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.__init__ with BertLMHeadModel->ErnieForCausalLM,Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the `ErnieForCausalLM` class.

        Args:
            self: The instance of the class.
            config (object):
                The configuration object containing various settings for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `ErnieForCausalLM` as a standalone, add `is_decoder=True.`")

        self.ernie = ErnieModel(config, add_pooling_layer=False)
        self.cls = ErnieOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings
    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the ErnieForCausalLM model.

        Args:
            self (ErnieForCausalLM): The instance of the ErnieForCausalLM class.

        Returns:
            decoder: This method returns the output embeddings from the model's predictions decoder layer.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the ErnieForCausalLM model.

        Args:
            self (ErnieForCausalLM): The instance of the ErnieForCausalLM class.
            new_embeddings: The new embeddings to be set as output embeddings.
                It should be of the same shape as the existing embeddings.

        Returns:
            None.

        Raises:
            None.

        Note:
            This method updates the output embeddings of the ErnieForCausalLM model to the provided new_embeddings.
            The new_embeddings should be of the same shape as the existing embeddings.

        Example:
            ```python
            >>> model = ErnieForCausalLM()
            >>> new_embeddings = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            >>> model.set_output_embeddings(new_embeddings)
            ```
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]:
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
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4
                tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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

    # Copied from transformers.models.bert.modeling_bert.BertLMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self: The instance of the class.
            input_ids (torch.Tensor): The input tensor containing the input ids.
            past_key_values (tuple, optional): The tuple containing past key values. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
            use_cache (bool, optional): Flag indicating whether to use cache. Defaults to True.

        Returns:
            dict: A dictionary containing the prepared input_ids, attention_mask, past_key_values, and use_cache.

        Raises:
            ValueError: If input_ids shape is incompatible with past_key_values.
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
        This method '_reorder_cache' reorders the past states based on the provided beam indices.

        Args:
            self (ErnieForCausalLM): The instance of the ErnieForCausalLM class.
            past_key_values (tuple): A tuple containing past states for each layer.
            beam_idx (Tensor): A tensor representing the beam indices used for reordering.

        Returns:
            None: This method does not return any value but updates the 'reordered_past' variable within the method.

        Raises:
            IndexError: If the provided beam indices are out of bounds.
            TypeError: If the input types are not as expected.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class ErnieForMaskedLM(ErniePreTrainedModel):

    """
    This class represents a model for Masked Language Modeling using the ERNIE
    (Enhanced Representation through kNowledge Integration) architecture.
    It is designed for generating predictions for masked tokens within a sequence of text.

    The class inherits from ErniePreTrainedModel and implements methods for initializing the model, getting and setting
    output embeddings, constructing the model for training or inference, and preparing inputs for text generation.

    Methods:
        __init__: Initializes the ErnieForMaskedLM model with the given configuration.
        get_output_embeddings: Retrieves the output embeddings from the model.
        set_output_embeddings: Sets new output embeddings for the model.
        construct: Constructs the model for training or inference, computing the masked language modeling loss
            and prediction scores.
        prepare_inputs_for_generation: Prepares inputs for text generation, including handling padding and dummy tokens.

    Note:
        This class assumes the existence of the ErnieModel and ErnieOnlyMLMHead classes for the ERNIE architecture.
    """
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the 'ErnieForMaskedLM' class.

        Args:
            self: The current object instance.
            config: An instance of the 'Config' class containing the configuration settings for the model.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes the 'ErnieForMaskedLM' class by setting the configuration and initializing the
            'ErnieModel' and 'ErnieOnlyMLMHead' objects.

            The 'config' parameter is an instance of the 'Config' class, which contains various configuration settings for the model.
            This method also logs a warning if the 'is_decoder' flag in the 'config' parameter is set to True,
            indicating that the model is being used as a decoder.

            The 'ErnieModel' object is initialized with the given 'config' and the 'add_pooling_layer' flag set to False.

            The 'ErnieOnlyMLMHead' object is also initialized with the given 'config'.

            Finally, the 'post_init' method is called to perform any additional initialization steps.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `ErnieForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.ernie = ErnieModel(config, add_pooling_layer=False)
        self.cls = ErnieOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.get_output_embeddings
    def get_output_embeddings(self):
        """
        Retrieve the output embeddings from the ErnieForMaskedLM model.

        Args:
            self (ErnieForMaskedLM): An instance of the ErnieForMaskedLM class.
                Represents the model object that contains the output embeddings.

        Returns:
            None: This method returns the output embeddings stored in the 'decoder' of the 'predictions' object
                within the ErnieForMaskedLM model.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the ErnieForMaskedLM model.

        Args:
            self (ErnieForMaskedLM): The instance of the ErnieForMaskedLM class.
            new_embeddings (object): The new embeddings to be set for the model's output.
                It can be any object that is compatible with the existing model's output embeddings.
                The new embeddings will replace the current embeddings.

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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        """
        Prepare inputs for generation.

        This method prepares input data for generation in the ErnieForMaskedLM model.

        Args:
            self: The instance of the ErnieForMaskedLM class.
            input_ids (Tensor): The input token IDs. Shape (batch_size, sequence_length).
            attention_mask (Tensor, optional): The attention mask tensor. Shape (batch_size, sequence_length).
            **model_kwargs: Additional model-specific keyword arguments.

        Returns:
            dict: A dictionary containing the prepared input_ids and attention_mask.

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


class ErnieForNextSentencePrediction(ErniePreTrainedModel):

    """
    ErnieForNextSentencePrediction is a class that represents a model for next sentence prediction using the ERNIE
    (Enhanced Representation through kNowledge IntEgration) architecture.
    This class inherits from the ErniePreTrainedModel class.

    The ERNIE model is designed for various natural language processing tasks, including next sentence prediction.
    It takes input sequences and predicts whether the second sequence follows the first sequence in a given pair.

    The class's code initializes an instance of the ErnieForNextSentencePrediction class with the provided configuration.
    It creates an ERNIE model and a next sentence prediction head.
    The post_init() method is called to perform additional setup after the initialization.

    The construct() method constructs the model using the provided input tensors and other optional arguments.
    It returns the predicted next sentence relationship scores. The method also supports computing the next sequence
    prediction loss if labels are provided.

    The labels parameter is used to compute the next sequence prediction loss.
    It should be a tensor of shape (batch_size,) where each value indicates the relationship between the input sequences:

    - 0 indicates sequence B is a continuation of sequence A.
    - 1 indicates sequence B is a random sequence.
    The method returns a tuple of the next sentence prediction loss, the next sentence relationship scores,
    and other optional outputs such as hidden states and attentions.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, ErnieForNextSentencePrediction
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
    # Copied from transformers.models.bert.modeling_bert.BertForNextSentencePrediction.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of ErnieForNextSentencePrediction.

        Args:
            self (ErnieForNextSentencePrediction): The instance of the ErnieForNextSentencePrediction class.
            config (dict): The configuration dictionary containing parameters for initializing the model.
                It should include necessary settings for model configuration.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.ernie = ErnieModel(config)
        self.cls = ErnieOnlyNSPHead(config)

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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[mindspore.Tensor], NextSentencePredictorOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring). Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

        Returns:
            Union[Tuple[mindspore.Tensor], NextSentencePredictorOutput]

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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            next_sentence_loss = ops.cross_entropy(seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieForSequenceClassification(ErniePreTrainedModel):

    """
    This class represents an ERNIE model for sequence classification tasks.
    It is a subclass of the `ErniePreTrainedModel` class.

    The `ErnieForSequenceClassification` class has an initialization method and a `construct` method.
    The initialization method initializes the ERNIE model and sets up the classifier layers.
    The `construct` method performs the forward pass of the model and returns the output.

    Attributes:
        num_labels (int): The number of labels for the sequence classification task.
        config (ErnieConfig): The configuration object for the ERNIE model.
        ernie (ErnieModel): The ERNIE model instance.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Dense): Dense layer for classification.

    Methods:
        __init__: Initializes the `ErnieForSequenceClassification` instance.
        construct: Performs the forward
            pass of the ERNIE model and returns the output.

    Example:
        ```python
        >>> # Initialize the model
        >>> model = ErnieForSequenceClassification(config)
        ...
        >>> # Perform forward pass
        >>> output = model.construct(input_ids, attention_mask, token_type_ids, task_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ```
    """
    # Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the 'ErnieForSequenceClassification' class.

        Args:
            self: The instance of the class.
            config: An instance of 'Config' class containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ernie = ErnieModel(config)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            return_dict=return_dict,
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
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ErnieForMultipleChoice(ErniePreTrainedModel):

    """
    This class represents an Ernie model for multiple choice tasks. It inherits from the ErniePreTrainedModel class.

    The ErnieForMultipleChoice class initializes an Ernie model with the given configuration.
    It constructs the model by passing input tensors through the Ernie model layers and applies dropout and
    classification layers to generate the logits for multiple choice classification.

    Example:
        ```python
        >>> model = ErnieForMultipleChoice(config)
        >>> outputs = model.construct(input_ids, attention_mask, token_type_ids, task_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ```
    Args:
        config (ErnieConfig): The configuration for the Ernie model.

    Methods:
        __init__:
            Initializes the ErnieForMultipleChoice class with the given configuration.

        construct:
            Constructs the Ernie model for multiple choice tasks and returns the model outputs.

    Returns:
        Union[Tuple[mindspore.Tensor], MultipleChoiceModelOutput]:
            The model outputs, which can include the loss, logits, hidden states, and attentions.

    Note:
        The labels argument should be provided for computing the multiple choice classification loss.
        Indices in labels should be in the range [0, num_choices-1], where num_choices is the size of the second
        dimension of the input tensors (input_ids).
    """
    # Copied from transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the ErnieForMultipleChoice class.

        Args:
            self (ErnieForMultipleChoice): The instance of the ErnieForMultipleChoice class.
            config: The configuration object containing various hyperparameters and settings for the model initialization.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the configuration object is missing required attributes.
            RuntimeError: If there are issues during model initialization or post-initialization steps.
        """
        super().__init__(config)

        self.ernie = ErnieModel(config)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], MultipleChoiceModelOutput]:
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


class ErnieForTokenClassification(ErniePreTrainedModel):

    """
    This class represents a token classification model based on the Ernie architecture.
    It is used for token-level classification tasks such as Named Entity Recognition (NER) and part-of-speech tagging.
    The model inherits from the ErniePreTrainedModel class and utilizes the ErnieModel for token embeddings and
    hidden representations.
    It includes methods for model initialization and forward propagation to compute token classification logits and loss.

    The class's constructor initializes the model with the provided configuration, sets the number of classification
    labels, and configures the ErnieModel with the specified parameters.
    Additionally, it sets up the dropout and classifier layers.

    The construct method takes input tensors and optional arguments for token classification, and returns the
    token classification output. It also computes the token classification loss if labels are provided.
    The method supports various optional parameters for controlling the model's behavior during inference.

    Note:
        The docstring is based on the provided information and does not include specific code signatures.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the ErnieForTokenClassification class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing the settings for the model.
                This object must have the following attributes:

                - num_labels (int): The number of labels for token classification.
                - classifier_dropout (float or None): The dropout rate for the classifier layer.
                If None, it defaults to the hidden dropout probability from the configuration.
                - hidden_dropout_prob (float): The dropout probability for the hidden layers.

        Returns:
            None.

        Raises:
            ValueError: If the config object is missing the num_labels attribute.
            TypeError: If the config object does not have the expected data types for the attributes.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie = ErnieModel(config, add_pooling_layer=False)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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


class ErnieForQuestionAnswering(ErniePreTrainedModel):

    """
    ErnieForQuestionAnswering is a class that represents a model for question answering tasks using the
    ERNIE (Enhanced Representation through kNowledge Integration) architecture.
    This class inherits from ErniePreTrainedModel and provides methods for constructing the model and
    performing question answering inference.

    The class constructor initializes the model with the provided configuration.
    The model architecture includes an ERNIE model with the option to add a pooling layer.
    Additionally, it includes a dense layer for question answering outputs.

    The construct method takes various input tensors and performs the question answering computation.
    It supports optional inputs for start and end positions, attention masks, token type IDs, task type IDs,
    position IDs, head masks, and input embeddings.
    The method returns the question-answering model output, which includes the start and end logits for the predicted
    answer spans.

    The method also allows for customizing the return of outputs by specifying the return_dict parameter.
    If the return_dict parameter is not provided, the method uses the default value from the model's configuration.

    Overall, the ErnieForQuestionAnswering class encapsulates the functionality for performing question answering tasks
    using the ERNIE model and provides a high-level interface for constructing the model and
    performing inference.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ with Bert->Ernie,bert->ernie
    def __init__(self, config):
        """
        Initializes an instance of the ErnieForQuestionAnswering class.

        Args:
            self (ErnieForQuestionAnswering): The object instance of the ErnieForQuestionAnswering class.
            config (object): An object containing configuration settings for the Ernie model.
                This parameter is required for initializing the ErnieForQuestionAnswering instance.
                It should include the following attributes:

                - num_labels (int): The number of labels for the classification task.
                - hidden_size (int): The size of the hidden layers in the model.
                - add_pooling_layer (bool): Flag indicating whether to add a pooling layer in the Ernie model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected object type.
            ValueError: If the config object is missing any required attributes.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie = ErnieModel(config, add_pooling_layer=False)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]:
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
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class UIEModelOutput(ModelOutput):
    """
    Output class for outputs of UIE.

    Args:
        loss (`mindspore.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_prob (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (after Sigmoid).
        end_prob (`mindspore.Tensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (after Sigmoid).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `mindspore.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    start_prob: mindspore.Tensor = None
    end_prob: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class UIE(ErniePreTrainedModel):
    """
    Ernie Model with two linear layer on top of the hidden-states
    output to compute `start_prob` and `end_prob`,
    designed for Universal Information Extraction.
    Args:
        config (:class:`ErnieConfig`):
            An instance of ErnieConfig used to construct UIE
    """
    def __init__(self, config: ErnieConfig):
        """
        Initializes an instance of the UIE class.

        Args:
            self (UIE): The instance of the UIE class.
            config (ErnieConfig): An instance of ErnieConfig containing the configuration parameters for the UIE model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.ernie = ErnieModel(config)
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
        return_dict: Optional[bool] = None
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
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

        if not return_dict:
            output = (start_prob, end_prob) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return UIEModelOutput(
            loss=total_loss,
            start_prob=start_prob,
            end_prob=end_prob,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST",
    "ErnieForCausalLM",
    "ErnieForMaskedLM",
    "ErnieForMultipleChoice",
    "ErnieForNextSentencePrediction",
    "ErnieForPreTraining",
    "ErnieForQuestionAnswering",
    "ErnieForSequenceClassification",
    "ErnieForTokenClassification",
    "ErnieModel",
    "ErniePreTrainedModel",
    "UIE"
]
