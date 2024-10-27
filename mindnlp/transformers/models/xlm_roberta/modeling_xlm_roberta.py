# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
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

"""MindSpore XLM-RoBERTa model."""
import math
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from .configuration_xlm_roberta import XLMRobertaConfig
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)


logger = logging.get_logger(__name__)

# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->XLMRoberta
class XLMRobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        """
        __init__
        
        Initializes a new instance of the XLMRobertaEmbeddings class.
        
        Args:
            self: The instance of the XLMRobertaEmbeddings class.
            config: An object containing configuration parameters for the XLMRoberta model.
                It includes the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The dimension of the hidden layers.
                - max_position_embeddings (int): The maximum number of positional embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability.
                - position_embedding_type (str, optional): The type of position embedding. Defaults to 'absolute'.
                - pad_token_id (int): The id of the padding token.

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
        self.LayerNorm = nn.LayerNorm([config.hidden_size,], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        """
        This method forwards the embeddings for the XLM-Roberta model.

        Args:
            self: (object) The instance of the class.
            input_ids: (Tensor, optional) The input tensor containing the token ids. Default is None.
            token_type_ids: (Tensor, optional) The input tensor containing the token type ids. Default is None.
            position_ids: (Tensor, optional) The input tensor containing the position ids. Default is None.
            inputs_embeds: (Tensor, optional) The input embeddings tensor. Default is None.
            past_key_values_length: (int) The length of the past key values. Default is 0.

        Returns:
            embeddings: (Tensor) The forwarded embeddings for the XLM-Roberta model.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None, or if an unsupported position_embedding_type
                is provided.
            IndexError: If input_ids or inputs_embeds do not have the expected shape.
            AttributeError: If the 'token_type_ids' attribute is missing in the class.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in forwardor where it is all zeros, which usually occurs
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

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: mindspore.Tensor

        Returns: mindspore.Tensor
        """
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = ops.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64)
        return position_ids.unsqueeze(0).expand(input_shape)


class XLMRobertaSelfAttention(nn.Module):
    """XLMRobertaSelfAttention"""
    def __init__(self, config, position_embedding_type=None):
        """
        This method initializes an instance of the XLMRobertaSelfAttention class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration settings for the XLMRobertaSelfAttention.
                It should have attributes like hidden_size, num_attention_heads, embedding_size,
                attention_probs_dropout_prob, position_embedding_type, max_position_embeddings, and is_decoder.
            position_embedding_type: (optional) A string specifying the type of position embedding. Defaults to None.
                It should be one of 'absolute', 'relative_key', or 'relative_key_query'.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size in the config is not a multiple of the number of attention heads,
                and the config does not have the attribute 'embedding_size'.
            AttributeError: If the config does not have the attribute 'embedding_size' when the hidden size is
                not a multiple of the number of attention heads.
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

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """transpose_for_scores"""
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
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
        Constructs the self-attention mechanism for the XLMRoberta model.

        Args:
            self: An instance of the XLMRobertaSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states. Shape (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor. Shape
                (batch_size, seq_length, seq_length). Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
                Shape (num_attention_heads, seq_length, seq_length). Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states from the encoder.
                Shape (batch_size, seq_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder hidden states.
                Shape (batch_size, seq_length, seq_length). Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The past key-value pairs for each layer in the encoder. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attention probabilities. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the context layer tensor.
                Shape (batch_size, seq_length, hidden_size).
            If output_attentions is True, the tuple also contains attention probabilities tensor.
                Shape (batch_size, num_attention_heads, seq_length, seq_length).
            If the model is a decoder, the tuple also contains the past key-value pairs.

        Raises:
            None.
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
            key_layer = ops.cat([past_key_value[0], key_layer], dim=2)
            value_layer = ops.cat([past_key_value[1], value_layer], dim=2)
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
                position_ids_l = mindspore.Tensor(key_length - 1, dtype=mindspore.int64).view(-1, 1)
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
            # Apply the attention mask is (precomputed for all layers in XLMRobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

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


class XLMRobertaSelfOutput(nn.Module):
    """XLMRobertaSelfOutput"""
    def __init__(self, config):
        """
        Initializes the XLMRobertaSelfOutput class.

        Args:
            self: The object instance.
            config (object):
                An object containing configuration parameters.

                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): Epsilon value for LayerNorm.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not specified in the config.
            ValueError: If the layer_norm_eps is not specified in the config.
            ValueError: If the hidden_dropout_prob is not specified in the config.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the output of the XLMRoberta model's self-attention layer.

        Args:
            self (XLMRobertaSelfOutput): The instance of the XLMRobertaSelfOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor representing the output of the
                self-attention layer. This tensor is processed through dense layers and normalization.
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden_states tensor.
                This tensor is used for residual connection in the self-attention layer.

        Returns:
            mindspore.Tensor: The output tensor after processing the hidden_states tensor through dense layers, dropout,
                normalization, and adding the input_tensor for the self-attention layer.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaAttention(nn.Module):
    """XLMRobertaAttention"""
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of XLMRobertaAttention.

        Args:
            self (object): The instance of the class itself.
            config (object): An object containing configuration settings.
            position_embedding_type (str, optional): The type of position embedding to use. Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = XLMRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = XLMRobertaSelfOutput(config)
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
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
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
        This method 'forward' in the class 'XLMRobertaAttention' forwards the output of the attention mechanism
        based on the input parameters.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states. Shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid performing attention on padding tokens.
                Shape (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): Mask to zero out selected heads of the attention mechanism.
                Shape (num_heads,).
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states of the encoder if applicable.
                Shape (batch_size, sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention if applicable.
                Shape (batch_size, sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key and value tensors for fast decoding.
            output_attentions (Optional[bool]): Flag to indicate whether to output attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor.
                Shape (batch_size, sequence_length, hidden_size).

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


class XLMRobertaIntermediate(nn.Module):
    """XLMRobertaIntermediate"""
    def __init__(self, config):
        """
        Initializes an instance of the XLMRobertaIntermediate class.

        Args:
            self: The instance of the XLMRobertaIntermediate class.
            config:
                Configuration object containing parameters for the intermediate layer.

                - Type: object
                - Purpose: Specifies the configuration settings for the intermediate layer.

        Returns:
            None

        Raises:
            TypeError: If the 'config' parameter is not provided.
            ValueError: If the 'hidden_act' attribute of the 'config' parameter is not a string or a
                valid activation function.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the hidden states using the specified intermediate layers in the XLMRoberta model.

        Args:
            self (XLMRobertaIntermediate): The instance of the XLMRobertaIntermediate class.
            hidden_states (mindspore.Tensor): The tensor representing the hidden states to be processed.
                It should be of type mindspore.Tensor and must adhere to the input requirements of the intermediate
                layers.

        Returns:
            mindspore.Tensor: Returns the processed hidden states as a tensor of type mindspore.Tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class XLMRobertaOutput(nn.Module):
    """XLMRobertaOutput"""
    def __init__(self, config):
        """
        Initializes an instance of XLMRobertaOutput class.

        Args:
            self (object): The instance of the XLMRobertaOutput class.
            config (object):
                An object containing configuration parameters.

                - Type: Any
                - Purpose: Specifies the configuration settings for the XLMRobertaOutput instance.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        '''
        This method forwards the output of the XLMRoberta model by performing a series of operations on the
        hidden states and input tensor.

        Args:
            self (XLMRobertaOutput): The instance of the XLMRobertaOutput class.
            hidden_states (mindspore.Tensor): The tensor containing the hidden states of the XLMRoberta model.
                It is expected to be a tensor of shape [batch_size, sequence_length, hidden_size].
            input_tensor (mindspore.Tensor): The input tensor to be added to the hidden states after normalization.
                It is expected to be a tensor of the same shape as hidden_states.

        Returns:
            mindspore.Tensor: The tensor representing the forwarded output of the XLMRoberta model.
                It has the same shape as the input_tensor and hidden_states.

        Raises:
            ValueError: If the shapes of hidden_states and input_tensor are not compatible for addition.
            RuntimeError: If an error occurs during the execution of the method.
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaLayer(nn.Module):
    """XLMRobertaLayer"""
    def __init__(self, config):
        """
        This method initializes an instance of the XLMRobertaLayer class.

        Args:
            self: The instance of the XLMRobertaLayer class.
            config: An object containing configuration parameters for the XLMRobertaLayer instance.
                It should include the following attributes:

                - chunk_size_feed_forward: An integer specifying the chunk size for feed-forward processing.
                - is_decoder: A boolean indicating whether the model is used as a decoder.
                - add_cross_attention: A boolean indicating whether cross-attention is added to the model.

        Returns:
            None.

        Raises:
            ValueError: If add_cross_attention is True and the model is not configured as a decoder, a ValueError
                is raised indicating that XLMRobertaLayer should be used as a decoder model when cross attention is added.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = XLMRobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = XLMRobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = XLMRobertaIntermediate(config)
        self.output = XLMRobertaOutput(config)

    def forward(
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
        Method to forward the XLMRobertaLayer.

        Args:
            self: The instance of the XLMRobertaLayer class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor containing attention mask values for the
                self-attention mechanism.
            head_mask (Optional[mindspore.Tensor]): Optional tensor containing head mask values for the
                self-attention mechanism.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional tensor containing hidden states
                from the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional tensor containing attention mask values
                for the encoder.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple of past key and value tensors
                for speeding up inference.
            output_attentions (Optional[bool]): Flag indicating whether to output attention weights.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the computed layer output tensor and any additional outputs
                depending on the decoder mode.

        Raises:
            ValueError: Raised if `encoder_hidden_states` are provided but cross-attention layers were not instantiated.
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
        """feed_forward_chunk"""
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class XLMRobertaEncoder(nn.Module):
    """XLMRobertaEncoder"""
    def __init__(self, config):
        """
        Initializes a new XLMRobertaEncoder object.

        Args:
            self (XLMRobertaEncoder): The XLMRobertaEncoder instance.
            config (object): The configuration object containing parameters for the encoder.
                It is expected to have attributes such as 'num_hidden_layers' to specify the number of hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
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
        """
        This method forwards the XLMRobertaEncoder.

        Args:
            self: The instance of the XLMRobertaEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor to mask the attention scores.
                Default is None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor to mask the attention scores of each head.
                Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor containing the hidden states of
                the encoder. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor to mask the encoder attention scores.
                Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key values.
                Default is None.
            use_cache (Optional[bool]): An optional boolean to use caching. Default is None.
            output_attentions (Optional[bool]): An optional boolean to output attention. Default is False.
            output_hidden_states (Optional[bool]): An optional boolean to output hidden states. Default is False.
            return_dict (Optional[bool]): An optional boolean to return a dictionary. Default is True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                Returns either a tuple of tensors or an instance of BaseModelOutputWithPastAndCrossAttentions based on
                the return_dict value.

        Raises:
            Warning: If use_cache is set to True while using gradient checkpointing, a warning is raised notifying that
                it is incompatible, and use_cache is set to False.
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


class XLMRobertaPooler(nn.Module):
    """XLMRobertaPooler"""
    def __init__(self, config):
        """
        Initializes an instance of the XLMRobertaPooler class.

        Args:
            self (object): The instance of the XLMRobertaPooler class.
            config (object):
                An object containing configuration parameters.

                - hidden_size (int): The size of the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
            Constructs the pooled output tensor from the given hidden states.

            Args:
                self: An instance of the XLMRobertaPooler class.
                hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size)
                    containing the hidden states of the XLM-Roberta model.

            Returns:
                mindspore.Tensor: The pooled output tensor of shape (batch_size, hidden_size) representing the
                    aggregated representation of the input sequence.

            Raises:
                None.

            """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class XLMRobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = False
    _no_split_modules = ["XLMRobertaEmbeddings", "XLMRobertaSelfAttention"]

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.assign_value(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.assign_value(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.assign_value(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Sets the gradient checkpointing attribute of the given module.

        Args:
            self: The instance of the XLMRobertaPreTrainedModel class.
            module: The module for which to set the gradient checkpointing attribute.
                Must be an instance of XLMRobertaEncoder.
            value: The value to set for the gradient checkpointing attribute. (Default: False)

        Returns:
            None.

        Raises:
            TypeError: If the module is not an instance of XLMRobertaEncoder.
        """
        if isinstance(module, XLMRobertaEncoder):
            module.gradient_checkpointing = value


class XLMRobertaModel(XLMRobertaPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """
    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->XLMRoberta
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes an instance of the XLMRobertaModel class.

        Args:
            self (XLMRobertaModel): The instance of the class itself.
            config (XLMRobertaConfig): The configuration object that holds the model configuration settings.
            add_pooling_layer (bool): A flag indicating whether to add a pooling layer to the model. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method 'get_input_embeddings' is defined in the class 'XLMRobertaModel' and retrieves the input
        embeddings from the model.

        Args:
            self (XLMRobertaModel): The instance of the XLMRobertaModel class.
                This parameter is required to access the embeddings within the model.

        Returns:
            None: This method returns None as it simply retrieves the input embeddings without any further processing.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the XLMRobertaModel.

        Args:
            self (XLMRobertaModel): The current instance of the XLMRobertaModel class.
            value: The new input embeddings to be set. This should be of type torch.Tensor.

        Returns:
            None.

        Raises:
            None.

        Note:
            The 'value' parameter should have the same dimensions as the current word_embeddings of the model.
            The input embeddings are used to represent the input tokens as vectors in the XLMRobertaModel.
            By setting new input embeddings, the model can be fine-tuned or updated with custom embeddings.

        Example:
            ```python
            >>> model = XLMRobertaModel()
            >>> embeddings = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
            >>> model.set_input_embeddings(embeddings)
            ```
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
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
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
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


class XLMRobertaForCausalLM(XLMRobertaPreTrainedModel):
    """XLMRobertaForCausalLM"""
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        """
        Initializes an instance of the XLMRobertaForCausalLM class.

        Args:
            self: The instance of the class.
            config: An object representing the configuration for the XLMRobertaForCausalLM model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `XLMRobertaLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Method to retrieve the output embeddings from XLMRobertaForCausalLM model.

        Args:
            self (XLMRobertaForCausalLM): The instance of the XLMRobertaForCausalLM class.
                It is used to access the decoder of the model to get the output embeddings.

        Returns:
            decoder: This method does not return any value but directly provides access to the output embeddings
                through the decoder.

        Raises:
            None.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the XLMRobertaForCausalLM model.

        Args:
            self (XLMRobertaForCausalLM): The instance of the XLMRobertaForCausalLM class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as the output embeddings for the model.

        Returns:
            None.

        Raises:
            None.

        Note:
            The output embeddings are used in the decoder layer of the XLMRobertaForCausalLM model.
            By setting new embeddings, users can customize the output layer of the model according to their specific
            requirements.

        Example:
            ```python
            >>> model = XLMRobertaForCausalLM.from_pretrained('xlm-roberta-base')
            >>> new_embeddings = torch.nn.Embedding(10, 768)
            >>> model.set_output_embeddings(new_embeddings)
            ```
        """
        self.lm_head.decoder = new_embeddings

    def forward(
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
        past_key_values: Tuple[Tuple[mindspore.Tensor]] = None,
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
                Mask to avoid performing attention on the padding token indices of the encoder input.
                This mask is used in the cross-attention if the model is configured as a decoder.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4
                tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).

        Returns:
            Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, XLMRobertaForCausalLM, AutoConfig
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            >>> config = AutoConfig.from_pretrained("roberta-base")
            >>> config.is_decoder = True
            >>> model = XLMRobertaForCausalLM.from_pretrained("roberta-base", config=config)
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="ms")
            >>> outputs = model(**inputs)
            ...
            >>> prediction_logits = outputs.logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.roberta(
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
        prediction_scores = self.lm_head(sequence_output)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = F.cross_entropy(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

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
            self (XLMRobertaForCausalLM): The instance of the XLMRobertaForCausalLM class.
            input_ids (torch.Tensor): The input tensor containing token ids. Shape should be
                (batch_size, sequence_length).
            past_key_values (Optional[torch.Tensor]): A tensor containing past key values. Default is None.
            attention_mask (Optional[torch.Tensor]): A tensor containing attention mask. If None is provided,
                it will be initialized with ones. Default is None.

        Returns:
            None.

        Raises:
            None.
        """
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache of past key values based on the provided beam index.

        Args:
            self (XLMRobertaForCausalLM): The instance of XLMRobertaForCausalLM.
            past_key_values (tuple): A tuple of past key values for each layer.
            beam_idx (torch.Tensor): A tensor containing the beam indices.

        Returns:
            None.

        Raises:
            IndexError: If the beam index is out of range for the past_key_values.
            TypeError: If the input types are incorrect or incompatible.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


class XLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    """XLMRobertaForMaskedLM"""
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        """
        Initializes an instance of XLMRobertaForMaskedLM.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing the settings for the model.
                It should have attributes like 'is_decoder' to control the behavior of the model.
                If 'is_decoder' is set to True, a warning message will be logged.
                Ensure that 'is_decoder' is set to False for bi-directional self-attention.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `XLMRobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """Get the output embeddings for the XLM-Roberta model.

        Args:
            self (XLMRobertaForMaskedLM): The instance of the XLMRobertaForMaskedLM class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the XLMRobertaForMaskedLM model.

        Args:
            self (XLMRobertaForMaskedLM): The instance of the XLMRobertaForMaskedLM class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as the output embeddings for the model.
                It should be an instance of torch.nn.Module representing the new embeddings.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not an instance of torch.nn.Module.
            AttributeError: If the lm_head.decoder attribute does not exist or is not accessible within
                the XLMRobertaForMaskedLM instance.
        """
        self.lm_head.decoder = new_embeddings

    def forward(
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
    ) -> Union[Tuple[mindspore.Tensor], MaskedLMOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead
class XLMRobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""
    def __init__(self, config):
        """
        This method initializes an instance of the XLMRobertaLMHead class.

        Args:
            self: An instance of the XLMRobertaLMHead class.
            config: A configuration object containing parameters for initializing the XLMRobertaLMHead instance.
                It is of type 'config' and is used to set the hidden size, vocabulary size, and layer
                normalization epsilon for the XLMRobertaLMHead instance.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = Parameter(ops.zeros(config.vocab_size), 'bias')
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        """
        Construct the LM head for the XLM-Roberta model.

        Args:
            self (XLMRobertaLMHead): The instance of the XLMRobertaLMHead class.
            features (Tensor): The input features to be used for LM head forwardion.

        Returns:
            None.

        Raises:
            None
        """
        x = self.dense(features)
        x = ops.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        """
        This method ties the weights of the XLMRobertaLMHead decoder to the bias.

        Args:
            self (XLMRobertaLMHead): The instance of the XLMRobertaLMHead class.
                This parameter is used to access the decoder and its bias.

        Returns:
            None.

        Raises:
            None
        """
        # To tie those two weights if they get disconnected
        self.bias = self.decoder.bias


class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    """XLMRobertaForSequenceClassification"""
    def __init__(self, config):
        """
        Initializes an instance of XLMRobertaForSequenceClassification.

        Args:
            self (object): The instance of the class.
            config (object):
                The configuration object containing the model hyperparameters and settings.

                - Type: XLMRobertaConfig
                - Purpose: Specifies the configuration for the XLM-Roberta model.
                - Restrictions: Must be a valid instance of XLMRobertaConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
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
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
            # move labels to correct device to enable model parallelism
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in (mindspore.int64, mindspore.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss = F.binary_cross_entropy_with_logits(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMRobertaForMultipleChoice(XLMRobertaPreTrainedModel):
    """XLMRobertaForMultipleChoice"""
    def __init__(self, config):
        """
        __init__

        Initialize the XLMRobertaForMultipleChoice model.

        Args:
            self: The instance of the class.
            config: An instance of the configuration class containing the model configuration.
                It is used to initialize the XLMRobertaModel, dropout, and classifier.
                It should be of type XLMRobertaConfig.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
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

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
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
            # move labels to correct device to enable model parallelism
            loss = F.cross_entropy(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMRobertaForTokenClassification(XLMRobertaPreTrainedModel):
    """XLMRobertaForTokenClassification"""
    def __init__(self, config):
        """
        Initializes the XLMRobertaForTokenClassification model.

        Args:
            self: The instance of the XLMRobertaForTokenClassification class.
            config:
                An object containing configuration settings for the model.
                It must provide the following attributes:

                - num_labels (int): The number of labels for token classification.
                - classifier_dropout (float, optional): The dropout probability for the classifier layer.
                If not specified, it defaults to the hidden dropout probability specified in the configuration.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - hidden_size (int): The size of the hidden layers in the model.

        Returns:
            None.

        Raises:
            TypeError: If config is not provided or is not an instance of the expected configuration object.
            ValueError: If the required attributes (num_labels, hidden_dropout_prob, hidden_size) are missing
                from the config.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
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
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaClassificationHead with Roberta->XLMRoberta
class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        """
        Initializes an instance of the XLMRobertaClassificationHead class.

        Args:
            self: The current instance of the class.
            config: An instance of the configuration class containing the model's configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Constructs the XLMRobertaClassificationHead.

        This method forwards the classification head for the XLM-RoBERTa model. It takes in a set of features and
        applies several operations to generate the final output.

        Args:
            self (XLMRobertaClassificationHead): An instance of the XLMRobertaClassificationHead class.
            features (Tensor): The input features for the classification head. It should have the shape
                (batch_size, sequence_length, num_features).

        Returns:
            Tensor: The output tensor of the classification head. It has the shape
                (batch_size, sequence_length, output_size).

        Raises:
            None.
        """
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ops.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XLMRobertaForQuestionAnswering(XLMRobertaPreTrainedModel):
    """XLMRobertaForQuestionAnswering"""
    def __init__(self, config):
        """
        Initializes the XLMRobertaForQuestionAnswering class.

        Args:
            self (XLMRobertaForQuestionAnswering): The instance of the XLMRobertaForQuestionAnswering class.
            config (XLMRobertaConfig): The configuration object for the XLM-RoBERTa model.
                It contains various parameters for model initialization, such as num_labels, hidden_size, and more.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not of type XLMRobertaConfig.
            ValueError: If the number of labels in the config is not a positive integer.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
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

        outputs = self.roberta(
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
        start_logits, end_logits = logits.split(1, dim=-1)
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
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = F.cross_entropy(start_logits, start_positions, ignore_index=ignored_index)
            end_loss = F.cross_entropy(end_logits, end_positions, ignore_index=ignored_index)
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


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: mindspore.Tensor x:

    Returns:
        mindspore.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (ops.cumsum(mask, dim=1).astype(mask.dtype) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

__all__ = [
        "XLMRobertaForCausalLM",
        "XLMRobertaForMaskedLM",
        "XLMRobertaForMultipleChoice",
        "XLMRobertaForQuestionAnswering",
        "XLMRobertaForSequenceClassification",
        "XLMRobertaForTokenClassification",
        "XLMRobertaModel",
        "XLMRobertaPreTrainedModel",
    ]
