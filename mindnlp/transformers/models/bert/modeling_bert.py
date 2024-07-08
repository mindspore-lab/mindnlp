# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# ============================================================================

"""MindNLP bert model"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np
import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.utils import ModelOutput
from .configuration_bert import BertConfig
from ...modeling_utils import PreTrainedModel
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
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer

try:
    from mindspore.hypercomplex.dual.dual_operators import Dense
    from mindspore.hypercomplex.utils import to_2channel, get_x_and_y
    from mindspore.hypercomplex.dual.dual_functions import matmul
except:
    from mindnlp._legacy.hypercomplex.dual import Dense
    from mindnlp._legacy.hypercomplex.utils import to_2channel, get_x_and_y
    from mindnlp._legacy.hypercomplex.dual.dual_functions import matmul

logger = logging.get_logger(__name__)

BERT_SUPPORT_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking"
]

@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

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

class BertEmbeddings(nn.Cell):
    """
    Embeddings for BERT, include word, position and token_type
    """
    def __init__(self, config):
        """
        This method initializes an instance of the BertEmbeddings class.
        
        Args:
            self: The instance of the BertEmbeddings class.
            config: An object containing configuration parameters for the embeddings.
                It should have the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layer.
                - pad_token_id (int): The index of the padding token.
                - max_position_embeddings (int): The maximum number of positional embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.
                - position_embedding_type (str, optional): The type of positional embedding, defaults to 'absolute'.

        Returns:
            None.

        Raises:
            AttributeError: If the config object does not have the required attributes.
            ValueError: If the config attributes have invalid values or types.
            TypeError: If the config parameters are of incorrect types.
            RuntimeError: If there is an error during the initialization process.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).reshape((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ):
        """
        This method constructs the embeddings for input tokens in the BERT model.

        Args:
            self (BertEmbeddings): The instance of the BertEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input token IDs. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The token type IDs for the input tokens. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position IDs for the input tokens. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The pre-computed input embeddings. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            None.

        Raises:
            TypeError: If the input_ids, token_type_ids, position_ids, or inputs_embeds are not of type mindspore.Tensor.
            ValueError: If the input_shape is not valid or if there is an issue with the dimensions of the input tensors.
            RuntimeError: If there is a runtime issue during the construction of embeddings.
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
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
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
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Cell):
    """
    Self attention layer for BERT.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes the BertSelfAttention instance.

        Args:
            self: The instance of the class.
            config: A configuration object containing the model's settings and hyperparameters.
            position_embedding_type (str, optional): The type of position embedding to be used. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size specified in the config is not a multiple of the number of attention heads.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}"
            )
        self.output_attentions = config.output_attentions

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

    def transpose_for_scores(self, input_x):
        r"""
        transpose for scores
        """
        new_x_shape = input_x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        input_x = input_x.view(*new_x_shape)
        return input_x.transpose(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Constructs the self-attention mechanism for the Bert model.

        Args:
            self (BertSelfAttention): The instance of the BertSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of the model.
                It has shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor to mask certain positions in the input.
                It has shape (batch_size, sequence_length) or (batch_size, 1, 1, sequence_length).
            head_mask (Optional[mindspore.Tensor]):
                The head mask tensor to mask certain heads of the attention mechanism.
                It has shape (num_attention_heads,) or (batch_size, num_attention_heads) or
                (batch_size, num_attention_heads, sequence_length, sequence_length).
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder.
                It has shape (batch_size, encoder_sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the encoder.
                It has shape (batch_size, 1, 1, encoder_sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The cached key and value tensors from previous steps.
                It is a tuple containing two tensors of shape (batch_size, num_attention_heads, sequence_length, head_size).
            output_attentions (Optional[bool]): Whether to output attention probabilities.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the computed context layer and the attention probabilities.
                The context layer has shape (batch_size, sequence_length, hidden_size) and the attention
                probabilities have shape (batch_size, num_attention_heads, sequence_length, sequence_length).

        Raises:
            None

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
                position_ids_l = Tensor(key_length - 1, dtype=mindspore.int64).view(-1, 1)
            else:
                position_ids_l = ops.arange(query_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(key_length, dtype=mindspore.int64).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding.broadcast_to((query_length, -1, -1)))
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = ops.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = ops.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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

        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Cell):
    r"""
    Bert Self Output
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertSelfOutput class.

        Args:
            self: The instance of the class.
            config:
                An object that holds configuration parameters for the self-attention mechanism.

                - Type: object
                - Purpose: Specifies the configuration settings for the self-attention mechanism.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required configuration settings.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm  = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        This method 'construct' is a part of the 'BertSelfOutput' class and is responsible
        for processing hidden states in a BERT model.

        Args:
            self:
                The instance of the class.

                - Type: BertSelfOutput.
                - Purpose: Represents the current instance of the class.
                - Restrictions: None.

            hidden_states:
                The hidden states obtained from the previous layer.

                - Type: Tensor.
                - Purpose: Represents the input hidden states to be processed.
                - Restrictions: Should be a valid Tensor object.

            input_tensor:
                The input tensor that needs to be added to the processed hidden states.

                - Type: Tensor.
                - Purpose: Represents the additional input tensor to be combined with the processed hidden states.
                - Restrictions: Should be a valid Tensor object.

        Returns:
            None: This method does not return any value but directly modifies the hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Cell):
    r"""
    Bert Attention
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes a BertAttention object.

        Args:
            self: The instance of the class itself.
            config (object): The configuration object containing settings for the BertAttention.
            position_embedding_type (str, optional): The type of position embedding to be used. Defaults to None.

        Returns:
            None: This method initializes the BertAttention object and does not return any value.

        Raises:
            None
        """
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """prune heads"""
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
    ):
        """
        This method constructs the BertAttention layer.

        Args:
            self (BertAttention): The instance of the BertAttention class.
            hidden_states (mindspore.Tensor): The input tensor containing the hidden states of the model.
                The shape should be [batch_size, sequence_length, hidden_size].
            attention_mask (Optional[mindspore.Tensor]): An optional tensor containing the attention mask for the input.
                If provided, the shape should be [batch_size, 1, sequence_length, sequence_length] and
                the values should be 0 or 1. Default is None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor containing the head mask for the input.
                If provided, the shape should be [num_heads] and the values should be 0 or 1. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                An optional tensor containing the hidden states of the encoder.
                If provided, the shape should be [batch_size, sequence_length, hidden_size].
                Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                An optional tensor containing the attention mask for the encoder input.
                If provided, the shape should be [batch_size, 1, sequence_length,
                sequence_length] and the values should be 0 or 1. Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple containing the past key and value tensors.
                If provided, the shape should be [(batch_size, num_heads, sequence_length,
                head_size), (batch_size, num_heads, sequence_length, head_size)]. Default is None.
            output_attentions (Optional[bool]): An optional boolean value indicating whether to output attentions.
                Default is False.

        Returns:
            outputs (Tuple[mindspore.Tensor]):
                A tuple of output tensors containing the attention_output and any additional outputs from the layer.

        Raises:
            ValueError: If the shapes or types of input tensors are invalid.
            RuntimeError: If there is a runtime error during the execution of the method.
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


class BertIntermediate(nn.Cell):
    r"""
    Bert Intermediate
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertIntermediate class.

        Args:
            self: The instance of the class.
            config: An object of type 'Config' containing the configuration settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states):
        """
        Constructs the intermediate layer of the BERT model.

        Args:
            self (BertIntermediate): An instance of the BertIntermediate class.
            hidden_states: The input hidden states to the intermediate layer.
                It should be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Cell):
    r"""
    Bert Output
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertOutput class.

        Args:
            self: The object itself.
            config: An instance of a configuration class containing various hyperparameters and settings.
                It is expected to have the following attributes:

                - intermediate_size: An integer representing the size of the intermediate layer.
                - hidden_size: An integer representing the size of the hidden layer.
                - layer_norm_eps: A floating-point number representing the epsilon value for layer normalization.
                - hidden_dropout_prob: A floating-point number representing the dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        This method constructs the output of a BERT model by applying transformations to the hidden states.

        Args:
            self: The instance of the BertOutput class.
            hidden_states (tensor): The hidden states from the BERT model, typically of shape (batch_size, sequence_length, hidden_size).
                It is the input to the method and represents the encoded information from the input tokens.
            input_tensor (tensor): The input tensor to be added to the hidden_states after transformation.
                It is typically of the same shape as hidden_states and serves as additional input for the transformation.

        Returns:
            tensor: The transformed hidden_states after applying a series of operations including dense layer, dropout, and layer normalization.
                The returned tensor represents the constructed output of the BERT model.

        Raises:
            ValueError: If the shapes of hidden_states and input_tensor are not compatible for the addition operation.
            RuntimeError: If any runtime error occurs during the transformation process.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Cell):
    r"""
    Bert Layer
    """
    def __init__(self, config):
        """
        Initialize a BertLayer object.

        Args:
            self (BertLayer): The instance of the BertLayer class.
            config (object):
                A configuration object containing various settings for the BertLayer.

                - chunk_size_feed_forward (int): The chunk size used for feed-forward operations.
                - is_decoder (bool): Indicates whether the model is designed as a decoder.
                - add_cross_attention (bool): Specifies if cross attention is to be added.
                - position_embedding_type (str): The type of position embedding to be used if cross attention is added.

        Returns:
            None.

        Raises:
            ValueError:
                If add_cross_attention is True and the model is not configured as a decoder, an exception is raised.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        This method constructs a BertLayer by processing the input hidden_states through self-attention
        and potentially cross-attention mechanisms.

        Args:
            self: The instance of the BertLayer class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]):
                An optional tensor for masking the attention scores. Defaults to None.
            head_mask (Optional[mindspore.Tensor]):
                An optional tensor to mask the heads of the attention mechanism. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                An optional tensor representing hidden states from the encoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                An optional tensor for masking the encoder attention scores. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple of past key and value tensors for efficient incremental decoding.
                Defaults to None.
            output_attentions (Optional[bool]): A flag indicating whether to output attention weights. Defaults to False.

        Returns:
            None:
                This method does not return any value explicitly,
                but it updates the internal state of the BertLayer instance.

        Raises:
            ValueError:
                If `encoder_hidden_states` are provided but cross-attention layers are not instantiated
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
        """feed forward chunk"""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Cell):
    r"""
    Bert Encoder
    """
    def __init__(self, config):
        """
        BertEncoder.__init__

        Initializes a new BertEncoder object.

        Args:
            self (object): The instance of the BertEncoder class.
            config (object): The configuration object containing settings for the BertEncoder.
                This parameter is required to initialize the BertEncoder object.

                - It should be an instance of the configuration class containing the necessary settings.

                    - Example: config = BertConfig(num_hidden_layers=12, ...)

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BertLayer(config) for _ in range(config.num_hidden_layers)])

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
    ):
        """
        This method 'construct' is a part of the class 'BertEncoder' and is responsible for processing
        hidden states through the encoder layers.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed through the encoder layers.
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid attention on padding tokens, defaults to None.
            head_mask (Optional[mindspore.Tensor]): Mask for attention heads in the encoder layers, defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states of the encoder, defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask to avoid attention on padding tokens in the encoder, defaults to None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key values for caching, defaults to None.
            use_cache (Optional[bool]): Indicates whether to use cache for the next decoder step, defaults to None.
            output_attentions (Optional[bool]): Flag to output attention weights, defaults to False.
            output_hidden_states (Optional[bool]): Flag to output hidden states, defaults to False.
            return_dict (Optional[bool]): Flag to return the output as a dictionary, defaults to True.

        Returns:
            None: This method does not return any value directly.
                It processes the input hidden states through the encoder layers and updates the states internally.

        Raises:
            None: This method does not raise any exceptions explicitly.
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


class BertPooler(nn.Cell):
    r"""
    Bert Pooler
    """
    def __init__(self, config):
        """
        Initializes the BertPooler class.

        Args:
            self: The instance of the BertPooler class.
            config:
                An instance of the configuration class containing the hidden size parameter.

                - Type: object
                - Purpose: To configure the BertPooler with specified hidden size.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        """
        Constructs the pooled output tensor from the given hidden states tensor.

        Args:
            self (BertPooler): The instance of the BertPooler class.
            hidden_states (torch.Tensor): The tensor of shape (batch_size, sequence_length, hidden_size)
                containing the hidden states of the input sequence.

        Returns:
            torch.Tensor: The pooled output tensor of shape (batch_size, hidden_size)
                representing the contextualized representation of the entire input sequence.

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding.
        # to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Cell):
    r"""
    Bert Prediction Head Transform
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertPredictionHeadTransform class.

        Args:
            self (BertPredictionHeadTransform): The instance of the class.
            config: A configuration object containing the necessary parameters for the transformation.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        """
        Constructs the transformed hidden states for the BertPredictionHeadTransform class.

        Args:
            self (BertPredictionHeadTransform): An instance of the BertPredictionHeadTransform class.
            hidden_states: The input hidden states to be transformed.
                Expected to be of shape [batch_size, sequence_length, hidden_size].

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Cell):
    r"""
    Bert LM Prediction Head
    """
    def __init__(self, config):
        """
        This method initializes the BertLMPredictionHead class.

        Args:
            self: The object instance of the BertLMPredictionHead class.
            config: A configuration object containing settings for the prediction head.
                It is of type dict or a custom configuration class.
                The config parameter is used to configure the prediction head's behavior and settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(initializer('zeros', config.vocab_size), 'bias')

        self.decoder.bias = self.bias
        # for mindspore.nn.Dense
        self.decoder.has_bias = True
        self.decoder.bias_add = ops.add

    def construct(self, hidden_states):
        """
        This method 'construct' is defined in the class 'BertLMPredictionHead' and is responsible for processing the hidden states.

        Args:
            self: The instance of the class.
            hidden_states (tensor): The input hidden states to be processed.
                It should be of type tensor and contain the information about the hidden states.

        Returns:
            hidden_states (tensor): The processed hidden states.
                It is of type tensor and contains the transformed and decoded information from the input hidden states.

        Raises:
            This method does not raise any exceptions.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def _tie_weights(self):
        """
        Ties the weights of the bias in the BertLMPredictionHead decoder to the main decoder weights.

        Args:
            self (BertLMPredictionHead): The instance of the BertLMPredictionHead class.
                This parameter is a reference to the current object.

        Returns:
            None: This method does not return any value. It updates the bias weights in-place.

        Raises:
            None
        """
        self.bias = self.decoder.bias


class BertOnlyMLMHead(nn.Cell):
    """BertOnlyMLMHead"""
    def __init__(self, config):
        """
        __init__

        This method initializes an instance of the BertOnlyMLMHead class.

        Args:
            self (BertOnlyMLMHead): The instance of the BertOnlyMLMHead class.
            config: The configuration parameters for the BertLMPredictionHead.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the masked language modeling (MLM) head for the BERT model.

        Args:
            self (BertOnlyMLMHead): The instance of the BertOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The output tensor from the BERT model's sequence output layer.
                It should have the shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The prediction scores for the masked language modeling task.
                It has the shape (batch_size, sequence_length, vocab_size).

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the input tensor does not have the expected shape.

        Note:
            The MLM head is responsible for generating prediction scores for the masked tokens in the input sequence.
            The prediction scores are computed by passing the sequence output through the predictions layer.
            The predictions layer maps the hidden states of each token to the vocabulary size, representing the probabilities
            of each token being the correct masked token.

        Example:
            ```python
            >>> head = BertOnlyMLMHead()
            >>> output = head.construct(sequence_output)
            ```
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Cell):
    """BertOnlyNSPHead"""
    def __init__(self, config):
        """
        Initializes a BertOnlyNSPHead object with the specified configuration.

        Args:
            self (BertOnlyNSPHead): The instance of the BertOnlyNSPHead class.
            config: The configuration object containing parameters for the NSP head.
                Expected to be an instance of a class that includes a 'hidden_size' attribute.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            AttributeError: If the config object does not have the 'hidden_size' attribute.
        """
        super().__init__()
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, pooled_output):
        """
        This method constructs a sequence relationship score based on the pooled output.

        Args:
            self (object): The instance of the class.
            pooled_output (object): The pooled output from the BERT model.

        Returns:
            None:
                This method returns None,
                as the constructed sequence relationship score is directly assigned to the seq_relationship_score variable.

        Raises:
            None.
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Cell):
    r"""
    Bert PreTraining Heads
    """
    def __init__(self, config):
        """
        Initializes the BertPreTrainingHeads class.

        Args:
            self (BertPreTrainingHeads): The instance of the BertPreTrainingHeads class.
            config: A configuration object containing settings for the BertPreTrainingHeads instance.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the necessary settings.
        """
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Dense(config.hidden_size, 2)

    def construct(self, sequence_output, pooled_output):
        """
        Construct the prediction scores and sequence relationship scores for pre-training heads in BERT.

        Args:
            self (BertPreTrainingHeads): An instance of the BertPreTrainingHeads class.
            sequence_output (Tensor): The output sequence tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the contextualized representation of each token in the input sequence.
            pooled_output (Tensor): The pooled output tensor of shape (batch_size, hidden_size).
                It represents the contextualized representation of the entire input sequence.

        Returns:
            tuple:
                A tuple containing two elements:

                - prediction_scores (Tensor): The prediction scores tensor of shape (batch_size, sequence_length, vocab_size).
                It represents the scores for predicting the masked tokens in the input sequence.
                - seq_relationship_score (Tensor): The sequence relationship score tensor of shape (batch_size, 2).
                It represents the scores for predicting the next sentence relationship.

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertPreTrainedModel(PreTrainedModel):
    """BertPretrainedModel"""
    config_class = BertConfig
    base_model_prefix = 'bert'

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


class BertModel(BertPreTrainedModel):
    r"""
    Bert Model
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a BertModel instance.

        Args:
            self: The instance of the BertModel class.
            config (object): The configuration object for the BertModel.
                It contains the required parameters for initializing the model.
            add_pooling_layer (bool): A flag indicating whether to add a pooling layer to the model.
                Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        """
        Gets the input embeddings for the BertModel.

        Args:
            self (BertModel): The instance of the BertModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        This method sets the input embeddings for the BertModel.

        Args:
            self (BertModel): The instance of the BertModel class.
            new_embeddings (object): The new input embeddings to be set for the BertModel.
                It can be of any valid object type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = new_embeddings

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
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        This method constructs a BERT model with the specified input parameters.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing token indices.
            attention_mask (Optional[mindspore.Tensor]): Mask tensor indicating which tokens should be attended to.
            token_type_ids (Optional[mindspore.Tensor]): Tensor indicating token types for different sequences in the input.
            position_ids (Optional[mindspore.Tensor]): Tensor containing position indices.
            head_mask (Optional[mindspore.Tensor]): Mask tensor specifying which heads to prune in the attention layers.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded input tensor.
            encoder_hidden_states (Optional[mindspore.Tensor]): Tensor containing hidden states from the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask tensor for encoder attention.
            past_key_values (Optional[List[mindspore.Tensor]]): List of tensors containing past key values.
            use_cache (Optional[bool]): Flag indicating whether to use cache in the decoder.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
            return_dict (Optional[bool]): Flag indicating whether to return a dictionary.

        Returns:
            None.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are specified.
            ValueError: If padding is present without an attention mask.
            ValueError: If the function encounters any other invalid input configuration.
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


class BertForPretraining(BertPreTrainedModel):
    r"""
    Bert For Pretraining
    """
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of BertForPretraining.

        Args:
            self: The instance of the class.
            config: A dictionary containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not a dictionary.
            ValueError: If the configuration settings are invalid or missing required fields.
        """
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method retrieves the output embeddings for the BertForPretraining model.

        Args:
            self: The instance of the class BertForPretraining.
                It is the implicit parameter representing the instance of the class itself.

        Returns:
            None: This method returns the output embeddings through the self.cls.predictions.decoder attribute.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the BertForPretraining model.

        Args:
            self (BertForPretraining): An instance of the BertForPretraining class.
            new_embeddings: The new embeddings that will be set as the output embeddings.
                Should be compatible with the model's architecture.

        Returns:
            None: This method modifies the model in-place.

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
    ):
        """
        Construct method in the BertForPretraining class.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor containing the indices of input sequence tokens in the vocabulary. Default is None.
            attention_mask (Optional[mindspore.Tensor]):
                The input tensor containing the mask to avoid performing attention on padding token indices.
                Default is None.
            token_type_ids (Optional[mindspore.Tensor]):
                The input tensor containing the token type ids to differentiate two sequences in the input.
                Default is None.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor containing the position indices of each input token in the sequence. Default is None.
            head_mask (Optional[mindspore.Tensor]):
                The input tensor containing the mask to nullify selected heads of the self-attention modules.
                Default is None.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor containing the embedded input sequence tokens. Default is None.
            labels (Optional[mindspore.Tensor]):
                The input tensor containing the labels for the masked language model. Default is None.
            next_sentence_label (Optional[mindspore.Tensor]):
                The input tensor containing the labels for the next sentence prediction. Default is None.
            output_attentions (Optional[bool]): Whether to return the attentions array. Default is None.
            output_hidden_states (Optional[bool]): Whether to return the hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return outputs as a dict. Default is None.

        Returns:
            None.

        Raises:
            None

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

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertLMHeadModel(BertPreTrainedModel):
    """BertLMHeadModel"""
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of BertLMHeadModel.

        Args:
            self: The instance of the class.
            config: A dictionary containing the configuration parameters for the model.
                It should include the following keys:

                - is_decoder (bool): Indicates whether the model is used as a decoder.

                    - If False, a warning will be logged.
                    - If True, the model will be initialized with is_decoder set to True.
                    - Default is False.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the BertLMHeadModel.

        Args:
            self (BertLMHeadModel): An instance of the BertLMHeadModel class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the output embeddings of the BertLMHeadModel.
        The output embeddings are obtained by predicting the decoder of the model's predictions.

        Note:
            The output embeddings represent the encoded representation of the input tokens in the model.
            They are useful for downstream tasks such as clustering or classification.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the language model head in a BERT model.

        Args:
            self (BertLMHeadModel): The instance of the BertLMHeadModel class.
            new_embeddings (Tensor): The new embeddings to be set for the output layer.
                Should be a tensor compatible with the existing model architecture.

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
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        '''
        This method constructs the BertLMHeadModel.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                Input tensor containing the indices of input tokens in the vocabulary.
            attention_mask (Optional[mindspore.Tensor]):
                Masking tensor used to avoid performing attention on padding token indices.
            token_type_ids (Optional[mindspore.Tensor]):
                Tensor containing the segment indices to differentiate between two sequences in the input.
            position_ids (Optional[mindspore.Tensor]):
                Tensor containing the position indices of each input token in the sequence.
            head_mask (Optional[mindspore.Tensor]):
                Masking tensor for attention heads.
            inputs_embeds (Optional[mindspore.Tensor]):
                Tensor containing the embedded input tokens.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                Tensor containing the hidden states of the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                Masking tensor for encoder attention.
            labels (Optional[mindspore.Tensor]):
                Tensor containing the labels for the prediction scores.
            past_key_values (Optional[List[mindspore.Tensor]]):
                List of tensors containing cached key and value states from previous attention mechanisms.
            use_cache (Optional[bool]):
                Flag indicating whether to use the cached key and value states for attention mechanisms.
            output_attentions (Optional[bool]):
                Flag indicating whether to output the attention weights of all layers.
            output_hidden_states (Optional[bool]):
                Flag indicating whether to output the hidden states of all layers.
            return_dict (Optional[bool]):
                Flag indicating whether to return the output as a dictionary.

        Returns:
            None.

        Raises:
            None
        '''
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        """
        Prepare inputs for generation.

        Args:
            self (BertLMHeadModel): The instance of the BertLMHeadModel class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length) containing the input ids.
            past_key_values (tuple, optional): The tuple of past key values used for generation. Defaults to None.
            attention_mask (torch.Tensor, optional):
                The attention mask tensor of shape (batch_size, sequence_length) containing the attention masks.
                Defaults to None.
            use_cache (bool, optional): Whether to use cache for generation. Defaults to True.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following keys:

                - 'input_ids' (torch.Tensor):
                The input tensor of shape (batch_size, sequence_length) containing the updated input ids.
                - 'attention_mask' (torch.Tensor):
                The attention mask tensor of shape (batch_size, sequence_length) containing the updated attention masks.
                - 'past_key_values' (tuple):
                The tuple of past key values used for generation.
                - 'use_cache' (bool):
                Whether to use cache for generation.

        Raises:
            None.
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

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache based on the provided beam index for a BERT language model head model.

        Args:
            self (BertLMHeadModel): The instance of the BertLMHeadModel class.
            past_key_values (tuple): A tuple containing past key values for each layer of the model.
            beam_idx (torch.Tensor): A tensor representing the beam index to reorder the cache.

        Returns:
            None: This method modifies the cache in-place.

        Raises:
            None
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class BertForMaskedLM(BertPreTrainedModel):
    """BertForMaskedLM"""
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the BertForMaskedLM class.

        Args:
            self: The instance of the class.
            config (BertConfig): The configuration object for the BertForMaskedLM model.

        Returns:
            None

        Raises:
            None

        Description:
        This method is the constructor for the BertForMaskedLM class. It initializes the instance by setting up
        the model architecture and loading the configuration.

        The 'config' parameter is an instance of the BertConfig class, which contains various settings
        and hyperparameters for the model.
        It is used to configure the model architecture and behavior.

        Note that if the 'is_decoder' attribute of the 'config' parameter is set to True, a warning message is logged,
        reminding the user to set 'is_decoder' to False when using the 'BertForMaskedLM' model
        with bi-directional self-attention.

        The method initializes two attributes of the instance:

        - 'bert': An instance of the 'BertModel' class, which represents the BERT model without the MLM head.
        The 'config' parameter is passed to the 'BertModel' constructor to configure the model architecture.
        - 'cls': An instance of the 'BertOnlyMLMHead' class, which represents the MLM head of the BERT model.
        The 'config' parameter is passed to the 'BertOnlyMLMHead' constructor to configure the MLM head.

        After the initialization, the 'post_init' method is called to execute any additional setup steps specific to the BertForMaskedLM class.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings for the BertForMaskedLM model.

        Args:
            self (BertForMaskedLM): The instance of the BertForMaskedLM class.

        Returns:
            None.

        Raises:
            None
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for the BertForMaskedLM model.

        Args:
            self (BertForMaskedLM): The instance of the BertForMaskedLM class.
            new_embeddings (Any): The new embeddings to set for the output layer.

        Returns:
            None.

        Raises:
            None
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
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
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

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
        """
        Method: prepare_inputs_for_generation

        Description:
            This method prepares inputs for generation by adding a dummy token at the end of the input_ids
            and updating the attention_mask accordingly.

        Args:
            self: The instance of the BertForMaskedLM class.
            input_ids (Tensor): The input token IDs for generation.
            attention_mask (Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            dict: A dictionary containing the updated 'input_ids' and 'attention_mask'.

        Raises:
            ValueError: If the PAD token is not defined in the configuration.
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


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BertForNextSentencePrediction"""
    def __init__(self, config):
        """
        Initializes an instance of BertForNextSentencePrediction class.

        Args:
            self (BertForNextSentencePrediction): The instance of the BertForNextSentencePrediction class.
            config: The configuration object containing settings for the BERT model.

        Returns:
            None: This method initializes the BertForNextSentencePrediction instance with the specified config settings.

        Raises:
            None.
        """
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

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
    ):
        """Constructs the BertForNextSentencePrediction model.

        Args:
            self (BertForNextSentencePrediction): An instance of the BertForNextSentencePrediction class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the indices of input sequence tokens.
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor indicating which tokens should be attended to (1) and which should not (0).
            token_type_ids (Optional[mindspore.Tensor]):
                The token type tensor indicating the type of each token in the input sequence.
            position_ids (Optional[mindspore.Tensor]): The tensor containing the position indices of each input token.
            head_mask (Optional[mindspore.Tensor]):
                The tensor indicating which heads should be masked in the attention layers.
            inputs_embeds (Optional[mindspore.Tensor]):
                The tensor containing the embedded representation of the input tokens.
            labels (Optional[mindspore.Tensor]): The tensor containing the labels for the next sentence prediction task.
            output_attentions (Optional[bool]): Whether to include the attention probabilities in the output.
            output_hidden_states (Optional[bool]): Whether to include the hidden states in the output.
            return_dict (Optional[bool]): Whether to return a dictionary instead of a tuple as the output.

        Returns:
            None.

        Raises:
            None.
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


class BertForSequenceClassification(BertPreTrainedModel):
    """Bert Model for classification tasks"""
    def __init__(self, config):
        """
        Initializes the BertForSequenceClassification class.

        Args:
            self (BertForSequenceClassification): The current instance of the BertForSequenceClassification class.
            config (BertConfig): The configuration object for the BertModel.
                It specifies the model architecture and parameters.

        Returns:
            None.

        Raises:
            ValueError: If the provided configuration object is invalid or missing required parameters.
            TypeError: If the configuration object is not of type BertConfig.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        '''
        This method constructs the Bert model for sequence classification.

        Args:
            self (BertForSequenceClassification): The instance of the BertForSequenceClassification class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor containing the indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[mindspore.Tensor]):
                The input tensor containing the attention mask to avoid performing attention on padding token indices.
            token_type_ids (Optional[mindspore.Tensor]):
                The input tensor containing the token type ids to differentiate between two sequences in the input.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor containing the position indices to position embeddings.
            head_mask (Optional[mindspore.Tensor]):
                The input tensor containing the mask for the heads which controls which head is executed.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor containing the embeddings of the input sequence tokens.
            labels (Optional[mindspore.Tensor]): The input tensor containing the labels for computing the loss.
            output_attentions (Optional[bool]): Whether to return attentions.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a sequence classifier output as a dictionary.

        Returns:
            None

        Raises:
            TypeError: If the input tensors are not of type mindspore.Tensor.
            ValueError: If there is a mismatch in the dimensions or types of the input tensors.
        '''
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


class BertForMultipleChoice(BertPreTrainedModel):
    """BertForMultipleChoice"""
    def __init__(self, config):
        """
        Initializes a BertForMultipleChoice instance.

        Args:
            self (BertForMultipleChoice): The current instance of the BertForMultipleChoice class.
            config: An instance of the configuration class that holds various hyperparameters and settings for the model.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not of the expected type.
            ValueError: If the provided config does not contain necessary attributes.
            RuntimeError: If there are issues during the initialization process.
        """
        super().__init__(config)

        self.bert = BertModel(config)
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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
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


class BertForTokenClassification(BertPreTrainedModel):
    """BertForTokenClassification"""
    def __init__(self, config):
        """
        Initialize the BertForTokenClassification model.

        Args:
            self (BertForTokenClassification): The instance of the BertForTokenClassification class.
            config:
                A configuration object containing settings for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
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


class BertForQuestionAnswering(BertPreTrainedModel):
    """BertForQuestionAnswering"""
    def __init__(self, config):
        """
        Initializes a new instance of the BertForQuestionAnswering class.

        Args:
            self: The object itself.
            config (BertConfig):
                The configuration for the Bert model. It contains various hyperparameters and settings.

                - Type: BertConfig
                - Purpose: Specifies the configuration for the Bert model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
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
    ):
        """
        Constructs the BertForQuestionAnswering model.

        Args:
            self (BertForQuestionAnswering): The instance of the BertForQuestionAnswering class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the indices of the input sequence tokens.
            attention_mask (Optional[mindspore.Tensor]):
                The input tensor containing the attention mask to avoid performing attention on padding tokens.
            token_type_ids (Optional[mindspore.Tensor]):
                The input tensor containing the segment token indices to indicate which tokens belong to the question
                and which belong to the context.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor containing the position indices to indicate the position of each token in the
                input sequence.
            head_mask (Optional[mindspore.Tensor]):
                The input tensor containing the mask to nullify selected heads of the self-attention modules.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor containing the embedded representation of the inputs.
            start_positions (Optional[mindspore.Tensor]):
                The input tensor containing the indices of the start positions for the answer span.
            end_positions (Optional[mindspore.Tensor]):
                The input tensor containing the indices of the end positions for the answer span.
            output_attentions (Optional[bool]): Whether to return the attentions weights of each layer in the outputs.
            output_hidden_states (Optional[bool]): Whether to return the hidden states of all layers in the outputs.
            return_dict (Optional[bool]): Whether to return a dictionary as the output instead of a tuple.

        Returns:
            None.

        Raises:
            None.
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
            end_loss = ops.cross_entropy(end_logits, end_positions)
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


class BertForPreTraining(BertPreTrainedModel):
    """BertForPreTraining"""
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        """
        Initializes a new instance of the BertForPreTraining class.

        Args:
            self: The object itself.
            config: A configuration object that specifies the model hyperparameters and other settings.
                It should be an instance of the BertConfig class.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from a BERT model for pre-training.

        Args:
            self: Instance of the BertForPreTraining class.
                This parameter refers to the current instance of the BertForPreTraining class.

        Returns:
            None:
                This method returns None, as it retrieves the output embeddings for further processing.

        Raises:
            None.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the model with the provided new embeddings.

        Args:
            self (BertForPreTraining): An instance of the BertForPreTraining class.
            new_embeddings (Any): The new embeddings to be set for the model's output.

        Returns:
            None: This method modifies the model's output embeddings in-place.

        Raises:
            None.

        Note:
            The 'new_embeddings' parameter should be of the same shape and type as the original output embeddings.
            Modifying the output embeddings may affect the model's performance and downstream tasks.
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
    ) -> Union[Tuple[mindspore.Tensor], BertForPreTrainingOutput]:
        """
        Constructs the pre-training model for BERT.

        Args:
            self: The instance of the BertForPreTraining class.
            input_ids (Optional[mindspore.Tensor]): The input token IDs. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask for the input. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The token type IDs. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position IDs of tokens. Default is None.
            head_mask (Optional[mindspore.Tensor]): The mask for heads in the self-attention mechanism. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded input tokens. Default is None.
            labels (Optional[mindspore.Tensor]): The labels for masked language modeling task. Default is None.
            next_sentence_label (Optional[mindspore.Tensor]): The label for next sentence prediction task. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return the output as a dictionary. Default is None.

        Returns:
            Union[Tuple[mindspore.Tensor], BertForPreTrainingOutput]:
                A tuple containing the prediction scores for masked language modeling
                and next sentence prediction tasks, and additional outputs if specified.

        Raises:
            None
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

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertDualSelfAttention(nn.Cell):

    """
    The BertDualSelfAttention class represents the dual self-attention mechanism used in the BERT model.
    This class implements the mechanism for both real and imaginary parts of the self-attention mechanism.
    It inherits from the nn.Cell class and provides methods for attention score computation and context layer generation.

    Attributes:
        config: A configuration object containing the model's hyperparameters.
        output_attentions: A boolean indicating whether to output attention scores.
        num_attention_heads: An integer representing the number of attention heads.
        attention_head_size: An integer representing the size of each attention head.
        all_head_size: An integer representing the total size of all attention heads combined.
        query: A Dense layer for computing queries for the attention mechanism.
        key: A Dense layer for computing keys for the attention mechanism.
        value: A Dense layer for computing values for the attention mechanism.
        dropout: A dropout layer for performing dropout on the attention scores.
        position_embedding_type: A string representing the type of position embedding used.

    Methods:
        transpose_for_scores(input_x): Transposes the input tensor for computing attention scores.
        construct(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions):
            Constructs the dual self-attention mechanism using the provided input tensors.

    Note:
        The construct method raises a NotImplementedError for cross-attention and past_key_value arguments,
        as these functionalities are not implemented yet.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the BertDualSelfAttention class.

        Args:
            self: The instance of the class.
            config (object): An object of the configuration class containing the model's configuration parameters.
            position_embedding_type (str, optional): The type of position embedding. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.

        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.config = config
        self.output_attentions = config.output_attentions
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Dense(config.hidden_size//2, self.all_head_size//2)
        self.key = Dense(config.hidden_size//2, self.all_head_size//2)
        self.value = Dense(config.hidden_size//2, self.all_head_size//2)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

    def transpose_for_scores(self, input_x):
        r"""
        transpose for scores
        """
        new_x_shape = input_x.shape[:-1] + (self.num_attention_heads, self.attention_head_size //2)
        input_x = input_x.view(*new_x_shape)
        return input_x.transpose(0, 1, 3, 2, 4)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        This method 'construct' in the class 'BertDualSelfAttention' implements the dual self-attention mechanism
        for the BERT model.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor):
                The input hidden states tensor with shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                An optional tensor with shape (batch_size, sequence_length) containing values of 0 or 1 to mask
                the attention scores for padded tokens.
            head_mask (Optional[mindspore.Tensor]): An optional tensor to mask the attention scores of specific heads.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                An optional tensor containing the hidden states of the encoder if performing cross-attention.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                An optional tensor to mask the attention scores for cross-attention.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                An optional tuple containing the past key and value tensors for incremental decoding.
            output_attentions (Optional[bool]):
                An optional boolean flag indicating whether to output the attention scores.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor]]:
                A tuple containing the context layer tensor with shape (batch_size, sequence_length, hidden_size)
                and optionally the attention scores tensor with shape
                (batch_size, num_attention_heads, sequence_length, sequence_length).

        Raises:
            NotImplementedError: If the functionality for cross-attention or incremental decoding is not implemented.
        """
        hidden_states_r = hidden_states[:,:,:self.config.hidden_size//2]
        hidden_states_d = hidden_states[:,:,self.config.hidden_size//2:]

        new_hidden_states = to_2channel(hidden_states_r, hidden_states_d)
        mixed_query_layer = self.query(new_hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention or past_key_value is not None:
            raise NotImplementedError("This functionality is not implemented.")

        mixed_key_layer = self.key(new_hidden_states)

        key_layer = self.transpose_for_scores(mixed_key_layer)

        mixed_value_layer = self.value(new_hidden_states)

        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_scores_r, attention_scores_i = get_x_and_y(attention_scores)
        attention_scores_r = ops.softmax(attention_scores_r, axis=-1)
        attention_scores_i = ops.softmax(attention_scores_i, axis=-1)

        p_attn = to_2channel(attention_scores_r, attention_scores_i)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        context_layer = matmul(p_attn, value_layer)

        context_layer_r, context_layer_i = get_x_and_y(context_layer)
        context_layer = ops.cat([context_layer_r, context_layer_i], -1)

        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, p_attn) if output_attentions else (context_layer,)

        return outputs

class BertDualSelfOutput(nn.Cell):

    """
    The 'BertDualSelfOutput' class represents a module that performs dual self-attention mechanism for BERT.
    It inherits from nn.Cell and contains methods for initializing the module and constructing
    the dual self-attention mechanism.

    Attributes:
        hidden_size (int): The size of the hidden states.
        dense (Dense): The dense layer for the dual self-attention mechanism.
        LayerNorm (LayerNorm): The layer normalization for the dual self-attention mechanism.
        dropout (Dropout): The dropout layer for the dual self-attention mechanism.

    Methods:
        __init__(config): Initializes the 'BertDualSelfOutput' module with the provided configuration.
        construct(hidden_states, input_tensor):
            Constructs the dual self-attention mechanism using the provided hidden states and input tensor.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertDualSelfOutput class.

        Args:
            self (BertDualSelfOutput): An instance of the BertDualSelfOutput class.
            config: A configuration object containing the parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense = Dense(config.hidden_size//2, config.hidden_size//2)
        self.LayerNorm  = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        Method 'construct' in the class 'BertDualSelfOutput'.

        This method constructs the hidden states by processing the input hidden states and input tensor.

        Args:
            self: Instance of the class BertDualSelfOutput. It represents the current instance of the class.
            hidden_states: Tensor of shape (batch_size, sequence_length, hidden_size).
                The input hidden states to be processed.
            input_tensor: Tensor of shape (batch_size, sequence_length, hidden_size).
                The input tensor to be added to the processed hidden states.

        Returns:
            None: This method does not return any value.

        Raises:
            None.
        """
        hidden_states_r = hidden_states[:,:,:self.hidden_size//2]
        hidden_states_d = hidden_states[:,:,self.hidden_size//2:]

        hidden_states = to_2channel(hidden_states_r, hidden_states_d)
        hidden_states = self.dense(hidden_states)
        hidden_states_r, hidden_states_d = get_x_and_y(hidden_states)
        hidden_states = ops.cat([hidden_states_r, hidden_states_d], -1)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertDualAttention(nn.Cell):

    """
    This class represents a BertDualAttention module that inherits from nn.Cell.
    It contains methods for initializing the module, pruning attention heads,
    and constructing the attention mechanism for BERT models.

    Attributes:
        config: Configuration for the BertDualAttention module.
        position_embedding_type: Type of position embedding to be used (optional).

    Methods:
        __init__(self, config, position_embedding_type=None):
            Initializes the BertDualAttention module with the given configuration and position embedding type.

        prune_heads(self, heads):
            Prunes the specified attention heads from the self-attention mechanism.

        construct(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
        encoder_attention_mask=None, past_key_value=None, output_attentions=False):
            Constructs the attention mechanism for BERT models using the provided inputs and past key values.

    Raises:
        ValueError: If the number of heads to be pruned is invalid.

    Returns:
        outputs: Tuple containing the attention output and optional additional outputs.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes the BertDualAttention class with the provided configuration and position embedding type.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing settings for the dual attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to be used. Default is None.

        Returns:
            None: This method does not return any value.

        Raises:
            None
        """
        super().__init__()
        self.self = BertDualSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertDualSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """prune heads"""
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
    ):
        """
        Constructs the attention mechanism for the BertDualAttention class.

        Args:
            self (BertDualAttention): The instance of the BertDualAttention class.
            hidden_states (mindspore.Tensor):
                The input hidden states tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor of shape (batch_size, seq_length) or (batch_size, seq_length, seq_length).
                Defaults to None.
            head_mask (Optional[mindspore.Tensor]):
                The head mask tensor of shape (num_heads, seq_length, seq_length). Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                The encoder hidden states tensor of shape (batch_size, seq_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                The encoder attention mask tensor of shape (batch_size, seq_length) or (batch_size, seq_length, seq_length).
                Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The previous key-value pairs tensor. Defaults to None.
            output_attentions (Optional[bool]): Whether to output the attention weights. Defaults to False.

        Returns:
            outputs (tuple):
                A tuple containing the attention output tensor of shape (batch_size, seq_length, hidden_size)
                and any additional outputs.

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

class BertDualIntermediate(nn.Cell):

    """
    This class represents a dual intermediate layer in a BERT model.

    The BertDualIntermediate class is a subclass of nn.Cell and is used to construct the dual intermediate layer in a BERT model.
    It takes in a configuration object as input, which specifies the hidden size and intermediate size.
    The class initializes the hidden size and intermediate size attributes based on the provided configuration.

    Attributes:
        hidden_size (int): The size of the hidden state in the dual intermediate layer.
        intermediate_size (int): The size of the intermediate state in the dual intermediate layer.
        dense (Dense): A dense layer that transforms the input hidden states.
        intermediate_act_fn (function): The activation function to be applied to the intermediate states.

    Methods:
        construct(hidden_states):
            Constructs the dual intermediate layer using the given hidden states as input.
            The method first splits the input hidden states into two channels: hidden_states_r and hidden_states_d.
            Then, it combines the two channels into a single input using the to_2channel function.
            The combined input is passed through the dense layer.
            The resulting intermediate states are then split back into two channels: hidden_states_r and hidden_states_d.
            Finally, the two channels are concatenated and passed through the intermediate activation function.
            The method returns the resulting hidden states.

    Note:
        - This class assumes that the given configuration object contains the necessary parameters for initialization.
        - The intermediate activation function can be either a string representing a predefined activation function or a custom activation function.

    Example:
        ```python
        >>> # Create a configuration object
        >>> config = {
        >>>     'hidden_size': 768,
        >>>     'intermediate_size': 3072,
        >>>     'hidden_act': 'gelu'
        >>> }
        ...
        >>> # Create an instance of the BertDualIntermediate class
        >>> dual_intermediate = BertDualIntermediate(config)
        ...
        >>> # Use the dual_intermediate instance to construct the dual intermediate layer
        >>> hidden_states = ... # input hidden states
        >>> output = dual_intermediate.construct(hidden_states)
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertDualIntermediate class.

        Args:
            self: The current object instance.
            config:
                An object of the configuration class that holds the configuration settings.

                - Type: object
                - Purpose: To provide the necessary configuration for initializing the BertDualIntermediate instance.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dense = Dense(config.hidden_size//2, config.intermediate_size//2)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states):
        """
        The construct method in the BertDualIntermediate class processes the hidden_states tensor to produce an intermediate representation.

        Args:
            self (BertDualIntermediate): The instance of the BertDualIntermediate class.
            hidden_states (tensor): A tensor of shape (batch_size, sequence_length, hidden_size)
                representing the hidden states of the input sequence. The hidden_size is expected to be an even number.

        Returns:
            None: This method does not return any value. The input hidden_states tensor is modified in place.

        Raises:
            ValueError: If the hidden_states tensor does not have the expected shape or if the hidden_size is not an even number.
            RuntimeError: If any runtime error occurs during the processing of hidden_states.
        """
        hidden_states_r = hidden_states[:,:,:self.hidden_size//2]
        hidden_states_d = hidden_states[:,:,self.hidden_size//2:]
        hidden_states = to_2channel(hidden_states_r, hidden_states_d)
        hidden_states = self.dense(hidden_states)
        hidden_states_r, hidden_states_d = get_x_and_y(hidden_states)
        hidden_states = ops.cat([hidden_states_r, hidden_states_d], -1)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertDualOutput(nn.Cell):

    """
    The 'BertDualOutput' class represents a custom neural network layer for processing dual outputs in a BERT model.
    This class inherits functionality from nn.Cell and implements methods for initialization and processing of hidden states.

    Attributes:
        intermediate_size (int): The size of the intermediate layer in the network.
        dense (Dense): A dense layer for processing the intermediate hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module for normalizing hidden states.
        dropout (nn.Dropout): A dropout layer for regularization during training.

    Methods:
        __init__(self, config): Initializes the BertDualOutput instance with the provided configuration.
        construct(self, hidden_states, input_tensor): Processes the hidden states and input tensor to produce the final output.

    The '__init__' method initializes the instance by setting the intermediate_size, dense layer,
    LayerNorm module, and dropout layer based on the provided configuration.
    The 'construct' method processes the hidden states by splitting them, applying transformations,
    and combining the outputs to produce the final hidden states.

    This class is designed to be used as a component in BERT models for handling dual outputs efficiently.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertDualOutput class.

        Args:
            self: The instance of the BertDualOutput class.
            config:
                A configuration object containing the following attributes:

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If the config parameter does not contain the required attributes
                or if their values are not within the expected range.
            AttributeError: If the config parameter does not have the necessary attributes.
        """
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.dense = Dense(config.intermediate_size//2, config.hidden_size//2)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states, input_tensor):
        """
        This method 'construct' is a member of the class 'BertDualOutput' and is used to process hidden states
        and input tensors in a specific manner.

        Args:
            self: The instance of the class.
            hidden_states (tensor): The hidden states to be processed.
                It is expected to be a tensor with shape (batch_size, sequence_length, hidden_size).
            input_tensor (tensor): The input tensor to be added to the processed hidden states.
                It is expected to be a tensor with the same shape as hidden_states.

        Returns:
            None: This method does not return any value explicitly,
                but it modifies the hidden_states and input_tensor in place.

        Raises:
            None: This method does not raise any exceptions explicitly.
        """
        hidden_states_r = hidden_states[:,:,:self.intermediate_size//2]
        hidden_states_d = hidden_states[:,:,self.intermediate_size//2:]
        hidden_states = to_2channel(hidden_states_r, hidden_states_d)
        hidden_states = self.dense(hidden_states)
        hidden_states_r, hidden_states_d = get_x_and_y(hidden_states)
        hidden_states = ops.cat([hidden_states_r, hidden_states_d], -1)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertDualLayer(nn.Cell):

    """
    BertDualLayer

    This class represents a layer in a dual-attention BERT model.
    It is a subclass of nn.Cell and is responsible for performing attention and feed-forward operations.

    Attributes:
        chunk_size_feed_forward (int): The size of chunks for feed-forward operation.
        seq_len_dim (int): The dimension of the sequence length.
        attention (BertDualAttention): The attention module used for self-attention.
        is_decoder (bool): Indicates whether the layer is used as a decoder model.
        add_cross_attention (bool): Indicates whether cross-attention is added.
        crossattention (BertAttention): The attention module used for cross-attention (if add_cross_attention is True).
        intermediate (BertDualIntermediate): The intermediate module used in the feed-forward operation.
        output (BertDualOutput): The output module used in the feed-forward operation.

    Methods:
        construct(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions):
            Constructs the layer by performing attention and feed-forward operations.

        feed_forward_chunk(attention_output):
            Performs the feed-forward operation on a chunk of attention output.

    Note:
        The class assumes that the imported modules (BertDualAttention, BertAttention, BertDualIntermediate, BertDualOutput)
        are available and properly implemented.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BertDualLayer class.

        Args:
            self: The object instance.
            config (object):
                The configuration object that contains the settings for the BertDualLayer.

                - chunk_size_feed_forward (int): The chunk size for feed-forward attention.
                - is_decoder (bool): Indicates whether the model is a decoder.
                - add_cross_attention (bool): Indicates whether cross attention is added.
                    Raises a ValueError if cross attention is added and the model is not a decoder.
                - position_embedding_type (str): The type of position embedding for cross attention.

        Returns:
            None

        Raises:
            ValueError: If cross attention is added and the model is not a decoder.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertDualAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertDualIntermediate(config)
        self.output = BertDualOutput(config)

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        This method constructs a BertDualLayer by performing self-attention and potentially cross-attention operations.

        Args:
            self: The instance of the BertDualLayer class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor specifying which elements should be attended to.
            head_mask (Optional[mindspore.Tensor]): An optional tensor providing a mask for the attention heads.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional hidden states from an encoder layer for cross-attention.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional attention mask for the encoder hidden states.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple containing the past key and value tensors.
            output_attentions (Optional[bool]): Flag indicating whether to output attention weights.

        Returns:
            None.

        Raises:
            ValueError: Raised if `encoder_hidden_states` are provided but cross-attention layers are not instantiated in the BertDualLayer instance.
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

        # layer_output = apply_chunking_to_forward(
        #     self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        # )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """feed forward chunk"""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertDualEncoder(nn.Cell):

    """
    The BertDualEncoder class represents a dual encoder model based on the BERT architecture.
    This class inherits from the nn.Cell class in MindSpore.

    Attributes:
        config: The configuration parameters for the model.
        layer: A list of BertDualLayer instances representing the stacked layers in the encoder.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled in the model.

    Methods:
        __init__(self, config):
            Initializes the BertDualEncoder instance with the provided configuration.

        construct(self, hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict):
            Constructs the dual encoder model with the given input tensors and parameters.
            Returns the final hidden states, past key values, hidden states at all layers,
            self-attentions at all layers, and cross-attentions at all layers.
    """
    def __init__(self, config):
        """Initialize the BertDualEncoder class.

        Args:
            self: The instance of the BertDualEncoder class.
            config:
                A dictionary containing the configuration parameters for the BertDualEncoder.

                - Type: dict
                - Purpose: Specifies the configuration settings for the BertDualEncoder.
                - Restrictions: Must be a valid dictionary object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BertDualLayer(config) for _ in range(config.num_hidden_layers)])
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
    ):
        """
        This method constructs the BertDualEncoder model.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            attention_mask (Optional[mindspore.Tensor]): Mask indicating which elements in the input should be attended to.
            head_mask (Optional[mindspore.Tensor]): Mask for attention heads.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states from the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key values for caching.
            use_cache (Optional[bool]): Flag indicating whether to use caching.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
            return_dict (Optional[bool]): Flag indicating whether to return a dictionary.

        Returns:
            None.

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


class BertDualModel(BertPreTrainedModel):
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
        Initializes an instance of the BertDualModel class.

        Args:
            self: The instance of the class.
            config (object): The configuration object that contains the settings for the model.
            add_pooling_layer (bool): A flag indicating whether to add a pooling layer. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertDualEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the BertDualModel instance.

        Args:
            self: The instance of the BertDualModel class.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings, represented by the 'word_embeddings' attribute of the BertDualModel instance.
        The embeddings are used to encode the input data into numerical representations suitable for processing by the model.

        Note that this method does not modify any attributes or perform any calculations. It simply returns the existing input embeddings.

        Example:
            ```python
            >>> model = BertDualModel()
            >>> embeddings = model.get_input_embeddings()
            ```
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BertDualModel.

        Args:
            self (BertDualModel): The instance of the BertDualModel class.
            value: The input embeddings to be set for the model. Should be of type WordEmbeddings.

        Returns:
            None: This method updates the input embeddings for the BertDualModel in-place.

        Raises:
            TypeError: If the provided 'value' is not of type WordEmbeddings.
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
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input.
                This mask is used in the cross-attention if the model is configured as a decoder.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with
                each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
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


class BertDualForSequenceClassification(BertPreTrainedModel):

    """
    The BertDualForSequenceClassification class represents a dual BERT model for sequence classification tasks.
    This class inherits from BertPreTrainedModel and provides methods for initializing the model and processing
    input data for sequence classification.

    The __init__ method initializes the BertDualForSequenceClassification instance by setting the number of labels,
    BERT model configuration, dropout, and classifier layers.

    The construct method processes input data for sequence classification using the BERT model.
    It accepts input tensors such as input_ids, attention_mask, token_type_ids, position_ids, head_mask,
    inputs_embeds, labels, and additional parameters for controlling the output format.
    The method returns the classification logits and can also calculate the loss based on the problem type
    and labels provided.

    Note:
        This docstring is based on the provided code and may need to be updated with additional information
        about the class attributes, methods, and usage.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BertDualForSequenceClassification class.

        Args:
            self: The instance of the class.
            config:
                An instance of the configuration class containing the model configuration parameters.

                - Type: config class
                - Purpose: Specifies the configuration parameters for the model.
                - Restrictions: Must be a valid instance of the configuration class.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config.num_labels is not provided or is invalid.
            AttributeError: If the required attributes are not found in the config object.
            RuntimeError: If an error occurs during model initialization or post-initialization.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertDualModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
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
    ):
        """
        This method constructs a dual BERT model for sequence classification.
        
        Args:
            self (BertDualForSequenceClassification): The instance of the BertDualForSequenceClassification class.
            input_ids (Optional[mindspore.Tensor]): The input token IDs representing the sequences. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask to avoid attending to padding tokens. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The token type IDs to distinguish different sequences in the input. Default is None.
            position_ids (Optional[mindspore.Tensor]): The position IDs to specify the position of each token in the input. Default is None.
            head_mask (Optional[mindspore.Tensor]): The head mask to nullify selected heads of the self-attention mechanism. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The embedded representation of the input sequences. Default is None.
            labels (Optional[mindspore.Tensor]): The labels for the input sequences. Default is None.
            output_attentions (Optional[bool]): Whether to return the attentions of all layers. Default is None.
            output_hidden_states (Optional[bool]): Whether to return the hidden states of all layers. Default is None.
            return_dict (Optional[bool]): Whether to return outputs as a dictionary. Default is None.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the provided problem type is not supported or recognized.
            RuntimeError: If the number of labels is not compatible with the problem type.
            NotImplementedError: If the problem type is not implemented.
        
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

__all__ = [
    'BertEmbeddings', 'BertAttention', 'BertEncoder', 'BertIntermediate', 'BertLayer',
    'BertModel', 'BertForPretraining', 'BertLMPredictionHead', 'BertForSequenceClassification',
    'BertForMaskedLM', 'BertForMultipleChoice', 'BertForNextSentencePrediction', 'BertForPreTraining',
    'BertForQuestionAnswering', 'BertForTokenClassification', 'BertLMHeadModel'
]
