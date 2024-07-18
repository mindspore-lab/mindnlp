# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
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
"""MindSpore MegatronBERT model."""


import math
import warnings
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
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration_megatron_bert import MegatronBertConfig
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MegatronBertConfig"
_CHECKPOINT_FOR_DOC = "nvidia/megatron-bert-cased-345m"

MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/megatron-bert-cased-345m",
    # See all MegatronBERT models at https://hf-mirror.com/models?filter=megatron_bert
]


class MegatronBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initialize the MegatronBertEmbeddings class.
        
        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the embeddings.

                - Type: Object
                - Purpose: Contains various configuration parameters such as vocab_size, hidden_size,
                max_position_embeddings, type_vocab_size, pad_token_id, hidden_dropout_prob, and position_embedding_type.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # In Megatron, layer-norm is applied after the 1st dropout.
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        Construct embeddings for the MegatronBertEmbeddings class.

        Args:
            self: The instance of the MegatronBertEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input token IDs. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The token type IDs. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position IDs. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded input tokens. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            mindspore.Tensor: The constructed embeddings.

        Raises:
            None.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Megatron BERT moves that layer norm after the drop-out (and to each layer).
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->MegatronBert
class MegatronBertSelfAttention(nn.Cell):

    """
    This class represents the self-attention mechanism used in the Megatron-BERT model.
    It is used to calculate the attention scores and apply attention weights to the input hidden states.

    Args:
        config (object): The configuration object containing various model parameters.
        position_embedding_type (str, optional): The type of position embedding to be used. Defaults to None.

    Raises:
        ValueError: If the hidden size is not a multiple of the number of attention heads.

    Attributes:
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): The total size of all attention heads.
        query (nn.Dense): The dense layer for query projection.
        key (nn.Dense): The dense layer for key projection.
        value (nn.Dense): The dense layer for value projection.
        dropout (nn.Dropout): The dropout layer for attention probabilities.
        position_embedding_type (str): The type of position embedding used.
        max_position_embeddings (int): The maximum number of position embeddings.
        distance_embedding (nn.Embedding): The embedding layer for relative position distances.
        is_decoder (bool): Indicates if the self-attention is used in the decoder.

    Methods:
        transpose_for_scores:
            Transposes the input tensor to match the attention scores shape.

        construct:
            Computes the self-attention scores and applies attention weights to the input hidden states.

    Returns:
        Tuple[mindspore.Tensor]: A tuple containing the context layer, and optionally the attention probabilities
            and past key-value states.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes a new instance of the MegatronBertSelfAttention class.

        Args:
            self: The object itself.
            config: An instance of the configuration class containing various settings for the self-attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to use. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and no
                embedding size is provided.

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
        Transpose the input tensor for scores calculation in the MegatronBertSelfAttention class.

        Args:
            self (MegatronBertSelfAttention): The instance of the MegatronBertSelfAttention class.
            x (mindspore.Tensor): The input tensor to be transposed.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The transposed tensor of shape
                (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            ValueError: If the input tensor x does not have the expected shape for transposition.
            TypeError: If the input tensor x is not of type mindspore.Tensor.
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
        Method to perform self-attention mechanism in Megatron-style BERT models.

        Args:
            self: Instance of the MegatronBertSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states to be attended over.
            attention_mask (Optional[mindspore.Tensor]): Mask to prevent attention to certain positions.
            head_mask (Optional[mindspore.Tensor]): Mask to zero out some heads of the attention calculation.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states of the encoder if cross-attention is needed.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder hidden states if cross-attention is needed.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key and value tensors for caching.
            output_attentions (Optional[bool]): Flag to indicate if attention probabilities should be returned.

        Returns:
            Tuple[mindspore.Tensor]: Tuple containing the context layer and optionally attention probabilities or
                past key and value.

        Raises:
            ValueError: If the dimensions of the input tensors are not compatible for matrix multiplication.
            TypeError: If there are issues with the types of the inputs.
            RuntimeError: If there are runtime issues while executing the attention mechanism.
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
            # Apply the attention mask is (precomputed for all layers in MegatronBertModel forward() function)
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


# Based transformers.models.bert.modeling_bert.BertSelfOutput. Moved LayerNorm to MegatronBertAttention below.
class MegatronBertSelfOutput(nn.Cell):

    """
    The MegatronBertSelfOutput class represents a neural network cell for processing self-attention output in a
    Megatron-style BERT model. This class is designed to be used within a neural network architecture.

    This class inherits from the nn.Cell class, and it contains methods for initializing the cell and constructing the
    self-attention output.

    The __init__ method initializes the MegatronBertSelfOutput cell with the given configuration, including setting up
    dense layers and dropout for processing the hidden states.

    The construct method takes the hidden_states and residual tensors as input and processes the hidden states using
    the defined dense and dropout layers. It then returns the sum of the original residual and the processed hidden
    states.

    Note:
        This class assumes the availability of the mindspore library for specific tensor operations.
    """
    def __init__(self, config):
        """
        Initializes the MegatronBertSelfOutput class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings.

                - hidden_size (int): The size of the hidden layer.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, residual: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the self-attention output for the MegatronBert model.

        Args:
            self (MegatronBertSelfOutput): An instance of the MegatronBertSelfOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the input to the self-attention layer.
            residual (mindspore.Tensor): The residual tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor is added to the output of the self-attention layer.

        Returns:
            mindspore.Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the self-attention output obtained by applying a dense layer and dropout to
                the hidden states tensor, and then adding it to the residual tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


# Based transformers.models.bert.modeling_bert.BertAttention. Added LayerNorm.
class MegatronBertAttention(nn.Cell):

    """
    This class represents the attention mechanism used in Megatron-BERT models. It is a part of the Megatron-BERT
    architecture and is responsible for performing self-attention operations.

    The MegatronBertAttention class inherits from the nn.Cell class.

    Attributes:
        ln (nn.LayerNorm): Layer normalization module used in the attention mechanism.
        self (MegatronBertSelfAttention): Self-attention module responsible for computing attention scores.
        output (MegatronBertSelfOutput): Output module that combines attention output with the input hidden states.
        pruned_heads (set): A set of pruned attention heads.

    Methods:
        __init__: Initializes the MegatronBertAttention instance.
        prune_heads: Prunes the specified attention heads.
        construct: Performs the attention mechanism computation.

    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertAttention class.

        Args:
            self (MegatronBertAttention): The current instance of the class.
            config (object):
                The configuration object containing the hyperparameters for the attention mechanism.

                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.ln = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.self = MegatronBertSelfAttention(config)
        self.output = MegatronBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the 'MegatronBertAttention' class. It prunes specific attention
        heads from the self-attention mechanism based on the provided 'heads' parameter.

        Args:
            self: Represents the instance of the MegatronBertAttention class.
                It is used to access the attributes and methods of the class.
            heads: A list that contains the indices of the attention heads to be pruned.
                These indices correspond to the specific attention heads that should be removed from the self-attention
                mechanism.

        Returns:
            None: However, it modifies the internal state of the MegatronBertAttention instance by pruning the specified
                attention heads from the self-attention mechanism.

        Raises:
            None:
                However, potential exceptions that might occur during the execution could include:

                - TypeError: If the input parameters are not of the expected types.
                - IndexError: If there are issues with accessing elements within the 'heads' list or other data structures.
                - ValueError: If there are inconsistencies or unexpected values encountered during the pruning process.
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
        Args:
            self: The instance of the MegatronBertAttention class.
            hidden_states (mindspore.Tensor): The input hidden states for the attention mechanism.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor specifying which elements should be attended to.
            head_mask (Optional[mindspore.Tensor]): Optional tensor for masking individual attention heads.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional tensor representing the hidden states of the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional tensor specifying which elements of the encoder
                hidden states should be attended to.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple of past key and value tensors for
                fast decoding.
            output_attentions (Optional[bool]): Optional flag indicating whether to return attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output and additional outputs from
                the attention mechanism.

        Raises:
            ValueError: If the input tensors have incompatible shapes or types.
            TypeError: If the input parameters are not of the expected types.
            RuntimeError: If there is an issue during the attention computation process.
        """
        ln_outputs = self.ln(hidden_states)
        self_outputs = self.self(
            ln_outputs,
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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->MegatronBert
class MegatronBertIntermediate(nn.Cell):

    """
    Represents an intermediate layer of a Megatron-style BERT model for processing hidden states.

    This class inherits from nn.Cell and contains methods for initializing the intermediate layer and processing
    hidden states through dense and activation functions.

    Attributes:
        dense (nn.Dense): The dense layer used for processing hidden states.
        intermediate_act_fn (function): The activation function applied to the hidden states.

    Methods:
        __init__: Initializes the MegatronBertIntermediate instance with the provided configuration.
        construct: Processes the input hidden states through the dense layer and activation function, returning
            the transformed hidden states.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertIntermediate class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing the settings for the MegatronBertIntermediate.
                It should have the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - intermediate_size (int): The size of the intermediate layer.
                - hidden_act (str or function): The activation function for the hidden layer.

                    - If it is a string, it should be one of the predefined activation functions available in the
                    ACT2FN dictionary.
                    - If it is a function, it should be a custom activation function.

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
        Constructs the intermediate layer of the Megatron BERT model.

        Args:
            self (MegatronBertIntermediate): An instance of the MegatronBertIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.

        Returns:
            mindspore.Tensor: The output hidden states tensor after applying the intermediate layer.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Based on transformers.models.bert.modeling_bert.BertOutput. Moved LayerNorm to MegatronBertLayer below.
class MegatronBertOutput(nn.Cell):

    """A module that serves as the output layer of the Megatron-BERT model.

    This module applies a dense layer followed by a dropout layer to the input tensor and adds it to the original input
    tensor. It is designed to be used as the output layer of the Megatron-BERT model.

    Args:
        config (obj): The configuration object that contains the required hyperparameters.

    Example:
        ```python
        >>> config = BertConfig(hidden_size=768, intermediate_size=3072, hidden_dropout_prob=0.1)
        >>> output_layer = MegatronBertOutput(config)
        >>> hidden_states = mindspore.Tensor([[0.5, 0.3, 0.2], [0.1, 0.7, 0.4]], mindspore.float32)
        >>> input_tensor = mindspore.Tensor([[0.2, 0.6, 0.9], [0.3, 0.4, 0.8]], mindspore.float32)
        >>> output = output_layer.construct(hidden_states, input_tensor)
        ```
    Attributes:
        dense (obj): The dense layer that applies a linear transformation to the input tensor.
        dropout (obj): The dropout layer that randomly sets elements of the input tensor to zero.

    Methods:
        construct(hidden_states, input_tensor):
            Applies the dense layer and dropout layer to the input tensor, and returns the sum of the input tensor
            and the transformed tensor.

    Note:
        This class inherits from `nn.Cell` and is typically used as a component within the Megatron-BERT
        model architecture.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the MegatronBertOutput class.

        Args:
            self: The object itself.
            config:
                An object of type 'config' which represents the configuration settings.

                - Type: object
                - Purpose: This parameter is used to configure the MegatronBertOutput instance.
                - Restrictions: Must be a valid 'config' object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the MegatronBertOutput by adding the hidden states to the input tensor.

        Args:
            self (MegatronBertOutput): An instance of the MegatronBertOutput class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states.
                The shape of the tensor should be compatible with the dense layer.
            input_tensor (mindspore.Tensor): A tensor containing the input values.
                The shape of the tensor should be compatible with the hidden states tensor.

        Returns:
            mindspore.Tensor: A tensor representing the result of adding the hidden states to the input tensor.

        Raises:
            None.

        Note:
            - The hidden states tensor is processed using the dense layer.
            - Dropout is applied to the hidden states tensor.
            - The input tensor and hidden states tensor should have compatible shapes.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return input_tensor + hidden_states


# Based on transformers.models.bert.modeling_bert.BertLayer. Added LayerNorm.
class MegatronBertLayer(nn.Cell):

    """
    This class represents a layer of the Megatron-Bert model. It is used to perform attention and feed-forward
    operations on input hidden states.

    Attributes:
        chunk_size_feed_forward (int): The chunk size used for chunking the feed-forward operation.
        seq_len_dim (int): The dimension of the sequence length.
        attention (MegatronBertAttention): The attention module used for self-attention.
        is_decoder (bool): Indicates whether the layer is used as a decoder model.
        add_cross_attention (bool): Indicates whether cross-attention is added.
        crossattention (MegatronBertAttention): The attention module used for cross-attention
            if add_cross_attention is True.
        ln (nn.LayerNorm): The layer normalization module.
        intermediate (MegatronBertIntermediate): The intermediate module used for the feed-forward operation.
        output (MegatronBertOutput): The output module used for the feed-forward operation.

    Methods:
        construct(hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
            encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            Constructs the layer by performing attention and feed-forward operations on the input hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states.
            - attention_mask (mindspore.Tensor, optional): The attention mask tensor. Defaults to None.
            - head_mask (mindspore.Tensor, optional): The head mask tensor. Defaults to None.
            - encoder_hidden_states (mindspore.Tensor, optional): The hidden states of the encoder if the layer
            is used as a decoder model. Defaults to None.
            - encoder_attention_mask (mindspore.Tensor, optional): The attention mask of the encoder if the layer
            is used as a decoder model. Defaults to None.
            - past_key_value (Tuple[Tuple[mindspore.Tensor]], optional): The past key-value pairs for caching
            attention outputs. Defaults to None.
            - output_attentions (bool, optional): Whether to output attention scores. Defaults to False.

            Returns:

            - Tuple[mindspore.Tensor]: The outputs of the layer.

        feed_forward_chunk(attention_output):
            Applies the feed-forward operation to the attention output.

            Args:

            - attention_output (mindspore.Tensor): The attention output.

            Returns:

            - mindspore.Tensor: The output of the feed-forward operation.
    """
    def __init__(self, config):
        """Initializes an instance of the MegatronBertLayer class.

        Args:
            self: An instance of the MegatronBertLayer class.
            config:
                A configuration object containing the following attributes:

                - chunk_size_feed_forward: An integer indicating the chunk size for feedforward layers.
                - is_decoder: A boolean indicating whether the layer is a decoder.
                - add_cross_attention: A boolean indicating whether to add cross attention to the layer.
                - hidden_size: An integer indicating the size of the hidden layer.
                - layer_norm_eps: A float indicating the epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            TypeError: If add_cross_attention is True and is_decoder is False.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MegatronBertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = MegatronBertAttention(config)
        self.ln = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.intermediate = MegatronBertIntermediate(config)
        self.output = MegatronBertOutput(config)

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
        Constructs a MegatronBertLayer.

        This method performs the forward pass of a MegatronBertLayer. It takes in various input tensors and returns
        the outputs after applying self-attention and cross-attention mechanisms, as well as feed-forward layers.

        Args:
            self (MegatronBertLayer): An instance of the MegatronBertLayer class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): An optional attention mask tensor of shape
                (batch_size, seq_length) where 1s indicate tokens to attend to and 0s indicate tokens to mask.
            head_mask (Optional[mindspore.Tensor]): An optional head mask tensor of shape (num_heads,) or
                (num_layers, num_heads) where 1s indicate heads to keep and 0s indicate heads to mask.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, seq_length, hidden_size) representing the hidden states of the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional attention mask tensor of shape
                (batch_size, seq_length) for the encoder.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key-value tensors
                for self-attention and cross-attention.
            output_attentions (Optional[bool]): An optional flag indicating whether to output attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the outputs of the MegatronBertLayer.
                The first element is the layer output tensor of shape (batch_size, seq_length, hidden_size).
                If the layer is a decoder, the tuple also contains the present key-value tensor of shape
                (2, batch_size, num_heads, seq_length, hidden_size).

        Raises:
            AttributeError: If `encoder_hidden_states` are passed and cross-attention layers are not instantiated
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
                raise AttributeError(
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
        Feed forward chunk of the MegatronBertLayer class.

        This method applies feed forward operations to the attention_output tensor.

        Args:
            self (MegatronBertLayer): An instance of the MegatronBertLayer class.
            attention_output (Tensor): The input tensor to be processed. It represents the attention output.

        Returns:
            None.

        Raises:
            None.

        """
        ln_output = self.ln(attention_output)
        intermediate_output = self.intermediate(ln_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MegatronBertEncoder(nn.Cell):

    """
    The MegatronBertEncoder class represents a transformer encoder for Megatron-BERT. It inherits from nn.Cell and
    is responsible for encoding input sequences using multiple layers of transformer blocks. The class provides methods
    for constructing the encoder and performing forward pass computations, including handling gradient checkpointing
    and caching for efficient training and inference.

    Attributes:
        config: The configuration parameters for the encoder.
        layer: A list of MegatronBertLayer instances representing the stacked transformer layers in the encoder.
        ln: A LayerNorm instance for layer normalization.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled.

    Methods:
        __init__: Initializes the MegatronBertEncoder with the provided configuration.
        construct: Constructs the encoder and performs forward pass computations, optionally returning hidden states,
            attentions, and cross-attentions based on the specified parameters.

    The construct method handles the processing of input hidden states, attention masks, head masks, encoder hidden
    states, encoder attention masks, past key values, and caching options. It iterates through the stacked transformer
    layers, applying gradient checkpointing if enabled, and computes the final hidden states with layer normalization.
    Additionally, it returns the output as a BaseModelOutputWithPastAndCrossAttentions object if return_dict is True.

    Note:
        The MegatronBertEncoder class is designed for use in the Megatron-BERT architecture and is designed to work in
        conjunction with other components such as MegatronBertLayer and LayerNorm for efficient transformer-based
        encoding.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the MegatronBertEncoder class.

        Args:
            self: The instance of the MegatronBertEncoder class.
            config (object): An object containing the configuration parameters for the MegatronBertEncoder.
                It should include the following attributes:

                - num_hidden_layers (int): The number of hidden layers in the encoder.
                - hidden_size (int): The size of the hidden layers.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([MegatronBertLayer(config) for _ in range(config.num_hidden_layers)])

        # The final layer norm. We removed the 1st LN, moved LN to each hidden layer and this one
        # is simply the final LN (Transformer's BERT has it attached to each hidden layer).
        self.ln = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
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
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        '''
        Constructs the MegatronBertEncoder.

        Args:
            self (MegatronBertEncoder): The instance of MegatronBertEncoder.
            hidden_states (mindspore.Tensor): The hidden states of the input sequence.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
                Shape: (batch_size, sequence_length) or (batch_size, sequence_length, sequence_length). Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
                Shape: (num_heads,) or (num_layers, num_heads) or (batch_size, num_layers, num_heads) or
                (batch_size, num_heads, sequence_length, sequence_length). Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder sequence.
                Shape: (batch_size, encoder_sequence_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the encoder.
                Shape: (batch_size, encoder_sequence_length) or (batch_size, encoder_sequence_length,
                encoder_sequence_length). Defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key value tensors. Defaults to None.
            use_cache (Optional[bool]): Whether to use cache. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Defaults to False.
            return_dict (Optional[bool]): Whether to return a dictionary as the output. Defaults to True.

        Returns:
            Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]: The output of the MegatronBertEncoder.
                It can be either a tuple of tensors or an instance of BaseModelOutputWithPastAndCrossAttentions.

        Raises:
            None

        '''
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

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

            # Because we moved the layer-norm at the end of the hidden layer, we have non-normali-
            # zed data here. If that's really needed, we must apply LN to match Transformer's BERT.

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Finalize the hidden states.
        hidden_states = self.ln(hidden_states)

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


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->MegatronBert
class MegatronBertPooler(nn.Cell):

    """This class represents a Pooler for the MegatronBert model.

    The MegatronBertPooler class is responsible for pooling the hidden states of the MegatronBert model
    and producing a pooled output tensor. It inherits from the nn.Cell class.

    Attributes:
        dense (nn.Dense): A fully connected layer that maps the input tensor to the desired output size.
        activation (nn.Tanh): An activation function that applies the hyperbolic tangent element-wise to the input tensor.

    Methods:
        __init__: Initializes the MegatronBertPooler instance.
        construct: Constructs the pooled output tensor.

    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertPooler class.

        Args:
            self: The instance of the class.
            config:
                An object of type 'Config' that contains the configuration settings for the pooler.

                - Type: Config
                - Purpose: Stores the configuration settings for the pooler.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        '''
        This method constructs pooled output from the hidden states of the MegatronBertPooler model.

        Args:
            self (MegatronBertPooler): The instance of the MegatronBertPooler class.
            hidden_states (mindspore.Tensor): The input tensor containing hidden states.
                It should be of shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: A tensor representing the pooled output. It has the shape (batch_size, hidden_size).

        Raises:
            None
        '''
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->MegatronBert
class MegatronBertPredictionHeadTransform(nn.Cell):

    """Represents a transformation head for the Megatron-BERT prediction head.

    This class inherits from nn.Cell and provides methods for transforming hidden states as part of the Megatron-BERT
    prediction head. It includes a dense layer, activation function transformation, and layer normalization.

    Attributes:
        dense (nn.Dense): The dense layer for transforming the hidden states.
        transform_act_fn (function): The activation function for transforming the hidden states.
        LayerNorm (nn.LayerNorm): The layer normalization for normalizing the hidden states.

    Methods:
        construct:
            Transforms the input hidden states using the dense layer, activation function, and layer normalization,
            and returns the transformed hidden states.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the MegatronBertPredictionHeadTransform class.

        Args:
            self: The object itself.
            config: An object of type 'Config' containing the configuration settings for the
                MegatronBertPredictionHeadTransform.

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
        Constructs the MegatronBertPredictionHeadTransform.

        This method applies a series of transformations to the input tensor `hidden_states` to prepare it for
        the Megatron-BERT prediction head.

        Args:
            self (MegatronBertPredictionHeadTransform): The instance of the MegatronBertPredictionHeadTransform class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, hidden_size).
                It represents the hidden states.

        Returns:
            mindspore.Tensor: The transformed hidden states tensor of shape (batch_size, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->MegatronBert
class MegatronBertLMPredictionHead(nn.Cell):

    """MegatronBertLMPredictionHead

    This class represents the prediction head for the Megatron-BERT language model. It is responsible for
    transforming the hidden states and generating predictions for the next token in a sequence.

    This class inherits from the nn.Cell class.

    Attributes:
        transform (MegatronBertPredictionHeadTransform): An instance of the MegatronBertPredictionHeadTransform class,
            used to transform the hidden states.
        decoder (nn.Dense): A fully connected layer that maps the transformed hidden states to the vocabulary size.
        bias (Parameter): A learnable bias parameter used in the decoder layer.

    Methods:
        construct(hidden_states): Transforms the input hidden states and generates predictions for the next token
            in the sequence.

    """
    def __init__(self, config):
        """
        Initialize the MegatronBertLMPredictionHead object with the provided configuration.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the prediction head.
                It is expected to have attributes like 'hidden_size' and 'vocab_size' required for initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.transform = MegatronBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size), 'bias')

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        Constructs the MegatronBertLMPredictionHead.

        Args:
            self (MegatronBertLMPredictionHead): The instance of the MegatronBertLMPredictionHead class.
            hidden_states (Tensor): The input hidden states to be processed. It should be a tensor of shape
                (batch_size, sequence_length, hidden_size).

        Returns:
            hidden_states (Tensor): The processed hidden states. It is a tensor of shape
                (batch_size, sequence_length, hidden_size) after applying the transformation and decoding.

        Raises:
            None.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->MegatronBert
class MegatronBertOnlyMLMHead(nn.Cell):

    """
    Represents a Megatron-style MLM head for BERT models, which includes only the MLM prediction head without the rest
    of the model.

    This class inherits from nn.Cell and is designed to be used in conjunction with a BERT model for masked language
    modeling tasks. It contains methods for initializing the prediction head and generating prediction scores based on
    the input sequence output.

    The class includes an __init__ method to initialize the prediction head with the provided configuration, and a
    construct method to generate prediction scores using the sequence output tensor. The prediction scores are obtained
    by passing the sequence output through the prediction head.

    Note:
        This class assumes that the MegatronBertLMPredictionHead class is available for use in creating the MLM
        prediction head.

    """
    def __init__(self, config):
        """
        Initialize the MegatronBertOnlyMLMHead class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration settings for the MegatronBertOnlyMLMHead class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method constructs predictions for masked language modeling using the MegatronBertOnlyMLMHead class.

        Args:
            self (MegatronBertOnlyMLMHead): The instance of the MegatronBertOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The output tensor from the previous layer representing the
                input sequence for prediction. This tensor should be compatible with the model architecture and contain
                the necessary information for prediction.

        Returns:
            mindspore.Tensor: A tensor containing the prediction scores generated by the model for masked language modeling.
                The prediction scores represent the likelihood of each token being the correct masked token.

        Raises:
            ValueError: If the input sequence_output is not a valid mindspore.Tensor object.
            RuntimeError: If there are issues during the prediction process within the self.predictions() method.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertOnlyNSPHead with Bert->MegatronBert
class MegatronBertOnlyNSPHead(nn.Cell):

    """
    This class represents the NSP (Next Sentence Prediction) head for the Megatron-BERT model.

    The MegatronBertOnlyNSPHead class inherits from the nn.Cell class and is responsible for predicting whether
    two sentences follow each other in a text. It is used in the Megatron-BERT model to perform the next sentence
    prediction task.

    Attributes:
        seq_relationship (nn.Dense): A densely connected layer that maps the input features to a score indicating the
            likelihood of the next sentence prediction. The layer has a hidden size of `config.hidden_size` and output size
            of 2, representing the two possible classes (follows or does not follow).

    Methods:
        __init__:
            Initializes a new instance of the MegatronBertOnlyNSPHead class.

            Args:

            - config (object): The configuration object for the Megatron-BERT model.

        construct:
            Constructs the NSP head by forwarding the input pooled_output through the seq_relationship layer.

            Args:

            - pooled_output (Tensor): The pooled output tensor from the Megatron-BERT model.

            Returns:

            - seq_relationship_score (Tensor): The predicted score for the next sentence prediction task.

    Note:
        This class assumes that the Megatron-BERT model has already been instantiated and its output features
        have been pooled.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertOnlyNSPHead class.

        Args:
            self (MegatronBertOnlyNSPHead): The object instance.
            config (object): The configuration object containing the model's settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        """
        Method 'construct' in the class 'MegatronBertOnlyNSPHead'.

        Args:
            self (object):
                The instance of the class.

                - Purpose: Represents the current instance of the class.
                - Restrictions: This parameter is automatically passed when the method is called.

            pooled_output (any):
                The pooled output from the model.

                - Purpose: The output obtained from pooling the sequence representations.
                - Restrictions: Expects a valid pooled output object.

        Returns:
            None:
                - Purpose: The method does not explicitly return any value but assigns the 'seq_relationship_score'
                to the pooled output.

        Raises:
            None.
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# Copied from transformers.models.bert.modeling_bert.BertPreTrainingHeads with Bert->MegatronBert
class MegatronBertPreTrainingHeads(nn.Cell):

    """
    This class represents the pre-training heads of the Megatron-BERT model. It is responsible for predicting masked
    tokens and determining the relationship between input sequences.

    The MegatronBertPreTrainingHeads class is a subclass of nn.Cell.

    Attributes:
        predictions (MegatronBertLMPredictionHead): An instance of the MegatronBertLMPredictionHead class that
            handles predicting masked tokens.
        seq_relationship (nn.Dense): A dense layer that determines the relationship between input sequences.

    Methods:
        __init__: Initializes the MegatronBertPreTrainingHeads instance.
        construct: Constructs the pre-training heads by generating prediction scores for masked tokens and calculating
            the sequence relationship score.

    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertPreTrainingHeads class.

        Args:
            self (MegatronBertPreTrainingHeads): The instance of the class.
            config: The configuration object containing the necessary parameters for initializing the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.predictions = MegatronBertLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        """
        Construct method in the MegatronBertPreTrainingHeads class.

        Args:
            self (object): The instance of the class.
            sequence_output (object): The output sequence tensor from the pre-trained BERT model.
                It is of type tensor and contains the contextual embeddings for each token in the input sequence.
            pooled_output (object): The pooled output tensor from the pre-trained BERT model.
                It is of type tensor and contains the aggregated representation of the input sequence.

        Returns:
            tuple:
                A tuple containing prediction_scores and seq_relationship_score.

                - prediction_scores (object): The prediction scores for the next sequence token.
                It is of type tensor and is obtained from the predictions method.
                - seq_relationship_score (object): The score for the next sequence relationship.
                It is of type tensor and is obtained from the seq_relationship method.

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MegatronBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MegatronBertConfig
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


@dataclass
# Copied from transformers.models.bert.modeling_bert.BertForPreTrainingOutput with Bert->MegatronBert
class MegatronBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`MegatronBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `mindspore.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`mindspore.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed or
            when `config.output_attentions=True`):
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


class MegatronBertModel(MegatronBertPreTrainedModel):
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
        __init__ method in the MegatronBertModel class.

        Args:
            self: The instance of the class.
            config: A dictionary containing configuration parameters for the MegatronBertModel.
                It is used to initialize the model's embeddings, encoder, and pooler.
            add_pooling_layer: A boolean flag indicating whether to add a pooling layer to the model.
                Default is True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = MegatronBertEmbeddings(config)
        self.encoder = MegatronBertEncoder(config)

        self.pooler = MegatronBertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
        This method returns the word embeddings used for input in a MegatronBertModel instance.

        Args:
            self (MegatronBertModel): The instance of the MegatronBertModel class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MegatronBertModel instance.

        Args:
            self (MegatronBertModel): The instance of the MegatronBertModel class.
            value: The new input embeddings to be set for the model. Should be of type torch.Tensor.

        Returns:
            None.

        Raises:
            None.
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
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
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
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having
                4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
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


class MegatronBertForPreTraining(MegatronBertPreTrainedModel):

    """
    The `MegatronBertForPreTraining` class represents a pre-trained Megatron-BERT model for pre-training tasks.
    It inherits from the `MegatronBertPreTrainedModel` class and provides methods for constructing
    the model, retrieving and setting output embeddings, and performing pre-training tasks such as masked
    language modeling and next sentence prediction.

    The `construct` method takes input tensors for various model inputs and optional labels, and returns pre-training
    outputs including loss, prediction logits, sequence relationship logits, hidden states, and attentions.
    This method supports both batch and sequence-level losses for masked language modeling and next sentence prediction.

    The `get_output_embeddings` method returns the decoder for predictions, while the `set_output_embeddings` method
    allows for updating the decoder with new embeddings.

    This class is designed to work with the Megatron-BERT model and is intended to be used for pre-training tasks in
    natural language processing applications.
    """
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config, add_binary_head=True):
        """
        Initializes a new instance of the MegatronBertForPreTraining class.

        Args:
            self (MegatronBertForPreTraining): The instance of the class.
            config (object): The configuration object containing the model's settings.
            add_binary_head (bool): Indicates whether to add a binary head to the model. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.cls = MegatronBertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MegatronBertForPreTraining model.

        Args:
            self (MegatronBertForPreTraining): The instance of the MegatronBertForPreTraining class.

        Returns:
            None.

        Raises:
            None.

        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the MegatronBertForPreTraining model.

        Args:
            self (MegatronBertForPreTraining): An instance of the MegatronBertForPreTraining class.
            new_embeddings: The new embeddings to be set for the model's output.
                This should be a tensor of the same shape as the previous embeddings.

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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        next_sentence_label: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MegatronBertForPreTrainingOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:
            Union[Tuple, MegatronBertForPreTrainingOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MegatronBertForPreTraining
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
            >>> model = MegatronBertForPreTraining.from_pretrained("nvidia/megatron-bert-cased-345m")
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
        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = ops.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = ops.cross_entropy(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MegatronBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MegatronBertForCausalLM(MegatronBertPreTrainedModel):

    '''
    A class that represents the MegatronBERT model for Causal Language Modeling. This class inherits from
    MegatronBertPreTrainedModel and provides methods for model initialization, output embeddings, input
    preparation for generation, cache reordering, and model construction. It also includes detailed explanations for
    the model's input and output parameters, as well as usage examples. The methods within the class
    enable fine-tuning and using the model for causal language modeling tasks.
    '''
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        """
        Initializes an instance of MegatronBertForCausalLM class.

        Args:
            self: The instance of MegatronBertForCausalLM class.
            config:
                A configuration object containing settings for the model initialization.

                - Type: object
                - Purpose: To configure the model with specific settings.
                - Restrictions: Must be a valid configuration object compatible with the model.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `MegatronBertForCausalLM` as a standalone, add `is_decoder=True.`")

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.cls = MegatronBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from MegatronBertForCausalLM model.

        Args:
            self (MegatronBertForCausalLM): The instance of the MegatronBertForCausalLM class.
                It represents the model for which the output embeddings are being retrieved.

        Returns:
            None.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the MegatronBertForCausalLM model.

        Args:
            self (MegatronBertForCausalLM): The instance of the MegatronBertForCausalLM class.
            new_embeddings (object): The new output embeddings to be set for the model.
                It could be a tensor, array, or any object representing the new embeddings.

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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
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

        Returns:
            Union[Tuple, CausalLMOutputWithCrossAttentions]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MegatronBertForCausalLM, MegatronBertConfig
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
            >>> model = MegatronBertForCausalLM.from_pretrained("nvidia/megatron-bert-cased-345m", is_decoder=True)
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            ...
            >>> prediction_logits = outputs.logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

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

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        """
        Prepare inputs for generation.

        Args:
            self (object): The instance of the class.
            input_ids (tensor): The input tensor containing the token ids.
                Its shape should be (batch_size, sequence_length).
            past_key_values (tuple, optional): The past key values if available for autoregressive generation.
                Default is None.
            attention_mask (tensor, optional): The attention mask tensor.
                If not provided, it is initialized with ones of the same shape as input_ids.

        Returns:
            dict: A dictionary containing the prepared input ids, attention mask, and past key values.

        Raises:
            ValueError: If the input_ids shape is invalid for past_key_values removal.
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
        Method to reorder the cache for a MegatronBertForCausalLM model.

        Args:
            self (object): The instance of the MegatronBertForCausalLM class.
            past_key_values (tuple): A tuple containing the past key values from the model.
            beam_idx (tensor): A tensor representing the indices for reordering the cache.

        Returns:
            None.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class MegatronBertForMaskedLM(MegatronBertPreTrainedModel):

    """
    This class represents a MegatronBert model for Masked Language Modeling (MLM). It inherits from the
    MegatronBertPreTrainedModel and includes methods for initializing the model, getting and setting output
    embeddings, constructing the model, and preparing inputs for generation. The class provides functionality
    for performing masked language modeling tasks using the MegatronBert model.

    Attributes:
        config (MegatronBertConfig): The configuration for the MegatronBert model.

    Methods:
        __init__: Initializes the MegatronBertForMaskedLM model with the given configuration.
        get_output_embeddings: Retrieves the output embeddings of the model.
        set_output_embeddings: Sets the output embeddings of the model to the specified new embeddings.
        construct: Constructs the model with the given input and optional arguments, and returns the MaskedLMOutput.
        prepare_inputs_for_generation: Prepares the input for generation by updating the input_ids and attention_mask
            for the model.

    Note:
        For consistency, always use triple double quotes around docstrings.
    """
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        """
        Initializes an instance of MegatronBertForMaskedLM.

        Args:
            self: The instance of the class.
            config: A configuration object containing settings for the MegatronBertForMaskedLM model.
                It must have attributes like 'is_decoder', which is a boolean indicating if the model is a decoder.
                The configuration object is used to configure the model's behavior.

        Returns:
            None.

        Raises:
            Warning: If the 'is_decoder' attribute in the config is set to True, a warning message is logged.
            AttributeError: If the config object does not have the required attributes, an AttributeError may be raised.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `MegatronBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.cls = MegatronBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the MegatronBertForMaskedLM model.

        Args:
            self (MegatronBertForMaskedLM): An instance of the MegatronBertForMaskedLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the MegatronBertForMaskedLM model.

        Args:
            self (MegatronBertForMaskedLM): An instance of the MegatronBertForMaskedLM class.
            new_embeddings: The new embeddings to be set for the model's output.

        Returns:
            None: This method modifies the model in-place.

        Raises:
            None.

        This method is used to set the output embeddings for the MegatronBertForMaskedLM model. The new embeddings are
        assigned to the model's predictions.decoder attribute, which represents the decoder layer responsible for
        generating output embeddings during inference. The method does not return any value and modifies the model
        directly.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
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
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
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

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        """
        Prepare inputs for generation.

        This method prepares input tensors for generation in the MegatronBertForMaskedLM model.

        Args:
            self: (object) The instance of the MegatronBertForMaskedLM class.
            input_ids: (Tensor) The input token IDs. Shape [batch_size, sequence_length].
            attention_mask: (Tensor, optional) The attention mask tensor. Shape [batch_size, sequence_length].

        Returns:
            dict:
                A dictionary containing the prepared input tensors for generation:

                - 'input_ids': (Tensor) The prepared input token IDs with dummy token appended.
                Shape [batch_size, sequence_length + 1].
                - 'attention_mask': (Tensor) The prepared attention mask tensor with an additional column of zeros appended.
                Shape [batch_size, sequence_length + 1].

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
            (effective_batch_size, 1), self.config.pad_token_id, dtype=mindspore.int64)
        input_ids = ops.cat([input_ids, dummy_token], axis=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class MegatronBertForNextSentencePrediction(MegatronBertPreTrainedModel):

    """
    Represents a MegatronBert model for next sentence prediction.

    This class inherits from the MegatronBertPreTrainedModel and provides next sentence prediction functionality
    using the Megatron BERT model.

    The class constructor initializes the MegatronBertForNextSentencePrediction model with the given configuration.

    The `construct` method takes input tensors and computes the next sentence prediction loss.
    It returns the next sentence predictor output.

    Args:
        input_ids (Optional[mindspore.Tensor], optional): The input tensor containing the indices of input sequence
            tokens in the vocabulary. Defaults to None.
        attention_mask (Optional[mindspore.Tensor], optional): The input tensor containing indices specifying which
            tokens should be attended to. Defaults to None.
        token_type_ids (Optional[mindspore.Tensor], optional): The input tensor containing the segment token indices
            to differentiate the sequences. Defaults to None.
        position_ids (Optional[mindspore.Tensor], optional): The input tensor containing the position indices of
            each input token. Defaults to None.
        head_mask (Optional[mindspore.Tensor], optional): The input tensor containing the mask for the attention heads.
            Defaults to None.
        inputs_embeds (Optional[mindspore.Tensor], optional): The input tensor containing the embedded inputs.
            Defaults to None.
        labels (Optional[mindspore.Tensor], optional): The tensor containing the labels for computing the next sequence
            prediction loss. Defaults to None.
        output_attentions (Optional[bool], optional): Whether to return attentions. Defaults to None.
        output_hidden_states (Optional[bool], optional): Whether to return hidden states. Defaults to None.
        return_dict (Optional[bool], optional): Whether to return a dictionary. Defaults to None.

    Returns:
        Union[Tuple, NextSentencePredictorOutput]: A tuple containing the next sentence prediction loss and the
            next sentence predictor output.

    Raises:
        FutureWarning: If the `next_sentence_label` argument is used, as it is deprecated.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, MegatronBertForNextSentencePrediction
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
        >>> model = MegatronBertForNextSentencePrediction.from_pretrained("nvidia/megatron-bert-cased-345m")
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
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertForNextSentencePrediction class.

        Args:
            self (MegatronBertForNextSentencePrediction): The instance of the class.
            config: The configuration object containing the settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.cls = MegatronBertOnlyNSPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring). Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

        Returns:
            Union[Tuple, NextSentencePredictorOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MegatronBertForNextSentencePrediction
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
            >>> model = MegatronBertForNextSentencePrediction.from_pretrained("nvidia/megatron-bert-cased-345m")
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


class MegatronBertForSequenceClassification(MegatronBertPreTrainedModel):

    """
    This class represents a MegatronBERT model for sequence classification tasks. It inherits from the
    MegatronBertPreTrainedModel class and includes methods for initializing the model and generating
    classification outputs.

    The `construct` method takes various input tensors and computes the sequence classification/regression loss based
    on the configured problem type. It returns the classification logits and optionally the loss, hidden states, and
    attentions.

    The `__init__` method initializes the model with the provided configuration and sets up the BERT model, dropout layer,
    and classifier for sequence classification.

    The class also provides detailed documentation for the `construct` method, including information about the input and
    output tensors, as well as the optional labels for computing the classification/regression loss.

    For complete method signatures and code, please refer to the source code.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertForSequenceClassification class.

        Args:
            self : The object instance.
            config : An object of type 'Config' containing the configuration settings for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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


class MegatronBertForMultipleChoice(MegatronBertPreTrainedModel):

    """
    A Python class representing the MegatronBertForMultipleChoice model, which is designed for multiple choice
    classification tasks. It is a subclass of the MegatronBertPreTrainedModel.

    The MegatronBertForMultipleChoice model consists of a MegatronBertModel, a dropout layer, and a classifier.
    The MegatronBertModel encodes the input sequence using the BERT architecture, while the dropout layer helps prevent
    overfitting. The classifier then produces logits for each choice in the multiple choice question.

    Methods:
        __init__: Initializes the MegatronBertForMultipleChoice model with the given configuration.
        construct: Constructs the model and performs forward pass given the input tensors. It returns the logits for
            each choice and optionally computes the loss.

    Attributes:
        bert: The MegatronBertModel used for encoding the input sequence.
        dropout: The dropout layer for regularization.
        classifier: The linear layer for producing logits.

    Note:
        - The input tensors should be either `mindspore.Tensor` objects or `None` if not applicable.
        - The `labels` tensor should have shape `(batch_size,)` and contain indices in `[0, ..., num_choices-1]`.
        - The `return_dict` argument is optional and defaults to the `use_return_dict` value from the model configuration.

    Example:
        ```python
        >>> config = MegatronBertConfig(...)
        >>> model = MegatronBertForMultipleChoice(config)
        >>> input_ids = ...
        >>> attention_mask = ...
        >>> token_type_ids = ...
        >>> position_ids = ...
        >>> head_mask = ...
        >>> inputs_embeds = ...
        >>> labels = ...
        >>> output_attentions = ...
        >>> output_hidden_states = ...
        >>> return_dict = ...
        >>> logits, loss = model.construct(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
        ... inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertForMultipleChoice class.

        Args:
            self (object): The instance of the class itself.
            config (object): The configuration object containing parameters for model initialization.
                It should have attributes like hidden_dropout_prob, hidden_size, etc.
                This parameter is required for configuring the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
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


class MegatronBertForTokenClassification(MegatronBertPreTrainedModel):

    """
    This class represents a token classification model based on the Megatron BERT architecture.
    It inherits from the MegatronBertPreTrainedModel class and includes functionality for token classification tasks.

    The __init__ method initializes the MegatronBertForTokenClassification instance with the provided configuration.
    It sets the number of labels, initializes the BERT model without a pooling layer, sets the dropout probability,
    and initializes the classifier.

    The construct method takes input tensors for token classification, such as input_ids, attention_mask, token_type_ids,
    position_ids, head_mask, and inputs_embeds. It also supports optional arguments for labels, output_attentions,
    output_hidden_states, and return_dict. The method returns TokenClassifierOutput containing the loss, logits,
    hidden states, and attentions. If labels are provided, it computes the token classification loss using cross-entropy.

    The class provides detailed docstrings for each method, including parameter descriptions and return types for
    improved documentation and understanding.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MegatronBertForTokenClassification class.

        Args:
            self: The instance of the class.
            config: An object containing configuration parameters for the model.
                It should include the following attributes:

                - num_labels (int): The number of labels for token classification.
                - hidden_dropout_prob (float): The dropout probability for the hidden layers.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the correct type.
            ValueError: If the num_labels attribute in the config is not provided or is not a positive integer.
            ValueError: If the hidden_dropout_prob attribute in the config is not provided or is not a valid
                probability value (0 <= hidden_dropout_prob <= 1).
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
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


class MegatronBertForQuestionAnswering(MegatronBertPreTrainedModel):

    """A class representing a Megatron-BERT model for question answering.

    This class inherits from the MegatronBertPreTrainedModel class and is specifically designed for question answering tasks.
    It includes methods for constructing the model and generating predictions.

    Attributes:
        num_labels (int): The number of labels for token classification.
        bert (MegatronBertModel): The Megatron-BERT model.
        qa_outputs (nn.Dense): The dense layer for question answering outputs.

    Methods:
        __init__: Initializes the MegatronBertForQuestionAnswering instance.
        construct: Constructs the Megatron-BERT model and generates predictions for question answering tasks.
    
    """
    def __init__(self, config):
        """
        Initialize the MegatronBertForQuestionAnswering class.
        
        Args:
            self (object): The instance of the class.
            config (object):
                The configuration object containing the settings for the model.

                - num_labels (int): The number of labels for question answering.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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

__all__ = [
    "MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MegatronBertForCausalLM",
    "MegatronBertForMaskedLM",
    "MegatronBertForMultipleChoice",
    "MegatronBertForNextSentencePrediction",
    "MegatronBertForPreTraining",
    "MegatronBertForQuestionAnswering",
    "MegatronBertForSequenceClassification",
    "MegatronBertForTokenClassification",
    "MegatronBertModel",
    "MegatronBertPreTrainedModel",
]
