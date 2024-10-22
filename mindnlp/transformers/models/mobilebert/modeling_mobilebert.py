# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
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
"""
MobileBert model
"""

import math
import warnings
from typing import Optional, Tuple
from dataclasses import dataclass

import mindspore
from mindnlp.core.nn import Parameter, Tensor

from mindnlp.core import nn, ops
from mindnlp.core.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from mindnlp.utils import ModelOutput
from .configuration_mobilebert import MobileBertConfig
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_conv1d_layer
from ...activations import ACT2FN

MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["google/mobilebert-uncased"]


class NoNorm(nn.Module):
    """NoNorm"""
    def __init__(self, feat_size):
        """
        Initializes an instance of the NoNorm class.
        
        Args:
            self (NoNorm): The instance of the NoNorm class.
            feat_size (int): The size of the feature.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        self.bias = Parameter(ops.zeros(feat_size))
        self.weight = Parameter(ops.ones(feat_size))

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Constructs a normalized tensor by applying weight and bias to the input tensor.
        
        Args:
            self (NoNorm): An instance of the NoNorm class.
            input_tensor (Tensor): The input tensor to be normalized.
                It should have the same shape as the weight and bias tensors.
        
        Returns:
            Tensor: A normalized tensor obtained by multiplying the input tensor with the weight and adding the bias.
        
        Raises:
            None.
        """
        return input_tensor * self.weight + self.bias


NORM2FN = {"layer_norm": nn.LayerNorm, "no_norm": NoNorm}


class MobileBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Initializes an instance of the MobileBertEmbeddings class.
        
        Args:
            self (MobileBertEmbeddings): The current instance of the MobileBertEmbeddings class.
            config: An object containing configuration parameters for the embeddings layer.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.config=config
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = nn.Linear(embedded_input_size, config.hidden_size)

        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.position_ids= ops.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Constructs the embeddings for MobileBERT model.
        
        Args:
            self (MobileBertEmbeddings): The instance of the MobileBertEmbeddings class.
            input_ids (Optional[Tensor]): The input tensor containing the token IDs. Default is None.
            token_type_ids (Optional[Tensor]): The input tensor containing the token type IDs. Default is None.
            position_ids (Optional[Tensor]): The input tensor containing the position IDs. Default is None.
            inputs_embeds (Optional[Tensor]): The input tensor containing pre-computed embeddings. Default is None.
        
        Returns:
            Tensor: The forwarded embeddings tensor for the MobileBERT model.
        
        Raises:
            ValueError: If the input_ids shape is invalid or if there are shape mismatches during concatenation.
            IndexError: If there are indexing errors during embeddings calculation.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int32)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.trigram_input:
            # From the paper MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited
            # Devices (https://arxiv.org/abs/2004.02984)
            #
            # The embedding table in BERT models accounts for a substantial proportion of model size. To compress
            # the embedding layer, we reduce the embedding dimension to 128 in MobileBERT.
            # Then, we apply a 1D convolution with kernel size 3 on the raw token embedding to produce a 512
            # dimensional output.
            inputs_embeds = ops.concat(
                [
                    ops.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0]),
                    inputs_embeds,
                    ops.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0]),
                ],
                dim=2
            )
        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MobileBertSelfAttention(nn.Module):
    """MobileBertSelfAttention"""
    def __init__(self, config):
        """
        Initializes the MobileBertSelfAttention object.
        
        Args:
            self (MobileBertSelfAttention): The instance of the MobileBertSelfAttention class.
            config (object): Configuration object containing various parameters.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """transpose_for_scores"""
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def forward(
        self,
        query_tensor: Tensor,
        key_tensor: Tensor,
        value_tensor: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        """
        Constructs the self-attention mechanism in the MobileBert model.
        
        Args:
            self (MobileBertSelfAttention): An instance of the MobileBertSelfAttention class.
            query_tensor (Tensor): The input tensor representing the query.
            key_tensor (Tensor): The input tensor representing the key.
            value_tensor (Tensor): The input tensor representing the value.
            attention_mask (Optional[Tensor]): An optional tensor representing the attention mask. Defaults to None.
            head_mask (Optional[Tensor]): An optional tensor representing the head mask. Defaults to None.
            output_attentions (Optional[bool]): An optional boolean indicating whether to output attentions.
                Defaults to None.
        
        Returns:
            Tuple[Tensor]: A tuple containing the context layer tensor.
                If output_attentions is set to True, the tuple also includes the attention probabilities tensor.
        
        Raises:
            None.
        """
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.transpose((0, 1, -1, -2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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
        return outputs


class MobileBertSelfOutput(nn.Module):
    """MobileBertSelfOutput"""
    def __init__(self, config):
        """
        Initializes a new instance of the MobileBertSelfOutput class.
        
        Args:
            self: The instance of the class.
            config: An object containing configuration settings for the MobileBertSelfOutput.
                It must have the following attributes:

                - use_bottleneck (bool): A flag indicating whether to use bottleneck layer.
                - true_hidden_size (int): The true hidden size for the dense layer.
                - normalization_type (str): The type of normalization to be used.
                - hidden_dropout_prob (float): The dropout probability for the layer.

        Returns:
            None. This method initializes the MobileBertSelfOutput instance with the specified configuration settings.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter is missing any required attributes.
        """
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.true_hidden_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, residual_tensor: Tensor) -> Tensor:
        """
        Constructs the output of the MobileBERT self-attention layer.

        Args:
            self (MobileBertSelfOutput): An instance of the MobileBertSelfOutput class.
            hidden_states (Tensor): The input hidden states tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the output of the self-attention layer.
            residual_tensor (Tensor): The residual tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the original input hidden states, which are added to the layer
                outputs after normalization.

        Returns:
            Tensor: The output tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the forwarded output of the MobileBERT self-attention layer.

        Raises:
            None.
        """
        layer_outputs = self.dense(hidden_states)
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class MobileBertAttention(nn.Module):
    """MobileBertAttention"""
    def __init__(self, config):
        """
        Initializes an instance of MobileBertAttention.

        Args:
            self: The instance of the class.
            config (dict): A dictionary containing configuration parameters for the attention mechanism.
                This dictionary should include the necessary settings for configuring the MobileBertSelfAttention
                and MobileBertSelfOutput components.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = MobileBertSelfAttention(config)
        self.output = MobileBertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """prune_heads"""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_conv1d_layer(self.self.query, index)
        self.self.key = prune_conv1d_layer(self.self.key, index)
        self.self.value = prune_conv1d_layer(self.self.value, index)
        self.output.dense = prune_conv1d_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query_tensor: Tensor,
        key_tensor: Tensor,
        value_tensor: Tensor,
        layer_input: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        """
        Constructs the attention mechanism for the MobileBert model.

        Args:
            self (MobileBertAttention): The instance of the MobileBertAttention class.
            query_tensor (Tensor): The input tensor representing the queries for attention calculation.
            key_tensor (Tensor): The input tensor representing the keys for attention calculation.
            value_tensor (Tensor): The input tensor representing the values for attention calculation.
            layer_input (Tensor): The input tensor representing the previous layer's output.
            attention_mask (Optional[Tensor], optional): A tensor representing the attention mask to be
                applied during attention calculation. Defaults to None.
            head_mask (Optional[Tensor], optional): A tensor representing a mask to be applied to the heads
                of the attention mechanism. Defaults to None.
            output_attentions (Optional[bool], optional): A flag indicating whether to output the attentions.
                Defaults to None.

        Returns:
            Tuple[Tensor]: A tuple containing the attention output tensor and any additional outputs from
                the attention mechanism.

        Raises:
            None

        This method forwards the attention mechanism for the MobileBert model. It takes the query_tensor, key_tensor,
        and value_tensor as inputs and performs self-attention calculation. The attention_output is then computed using
        the attention mechanism and the layer_input. The method returns a tuple containing the attention_output and
        any additional outputs obtained from the attention mechanism. The attention_mask and head_mask can be optionally
        provided to control the attention calculation, and output_attentions can be set to True to obtain the attention
        values.
        """
        self_outputs = self.self(
            query_tensor,
            key_tensor,
            value_tensor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        attention_output = self.output(self_outputs[0], layer_input)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MobileBertIntermediate(nn.Module):
    """MobileBertIntermediate"""
    def __init__(self, config):
        """
        Initialize the MobileBertIntermediate class.

        Args:
            self: The instance of the MobileBertIntermediate class.
            config:
                An object containing configuration parameters.

                - Type: Any
                - Purpose: Configuration settings for the intermediate layer.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            KeyError: If the 'hidden_act' parameter in the config object is not recognized.
            TypeError: If the 'hidden_act' parameter is not a string or callable function.
        """
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Method 'forward' in the class 'MobileBertIntermediate'.

        This method forwards the intermediate hidden states using the provided input tensor.

        Args:
            hidden_states (Tensor): The input tensor containing hidden states.
                This tensor will be processed to generate intermediate hidden states.

        Returns:
            Tensor: A tensor containing the intermediate hidden states after processing.
                The output tensor represents the intermediate hidden states obtained from the input tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputBottleneck(nn.Module):
    """OutputBottleneck"""
    def __init__(self, config):
        """
        Initializes an instance of the OutputBottleneck class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings.

                - Type: Custom class
                - Purpose: Contains configuration parameters for the OutputBottleneck.
                - Restrictions: Must be properly initialized with required attributes.

        Returns:
            None.

        Raises:
            AttributeError: If the config object is missing required attributes.
            TypeError: If the config object is not of the expected type.
            ValueError: If there are issues with the configuration parameters provided.
        """
        super().__init__()
        self.dense = nn.Linear(config.true_hidden_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, residual_tensor: Tensor) -> Tensor:
        """
        Constructs the output of the OutputBottleneck layer.

        Args:
            self: The instance of the OutputBottleneck class.
            hidden_states (Tensor): The input hidden states tensor.
                It is a tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the hidden states of the previous layer.
            residual_tensor (Tensor): The residual tensor.
                It is a tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the residual connection from the previous layer.
                The hidden states and the residual tensor are element-wise added together.

        Returns:
            Tensor: The output tensor.
                It is a tensor of shape (batch_size, sequence_length, hidden_size).
                This tensor represents the output of the OutputBottleneck layer.

        Raises:
            None.
        """
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class MobileBertOutput(nn.Module):
    """MobileBertOutput"""
    def __init__(self, config):
        """
        Initializes the MobileBertOutput class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the MobileBertOutput.

                Parameters:

                - use_bottleneck (bool): A boolean indicating whether to use bottleneck.
                - intermediate_size (int): The intermediate size for the dense layer.
                - true_hidden_size (int): The true hidden size for the dense layer.
                - normalization_type (str): The type of normalization to be used.
                - hidden_dropout_prob (float): The dropout probability for the dense layer.

        Returns:
            None.

        Raises:
            ValueError: If the configuration parameters are invalid or missing.
            TypeError: If the data type of the configuration parameters is incorrect.
        """
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(config)

    def forward(
        self, intermediate_states: Tensor, residual_tensor_1: Tensor, residual_tensor_2: Tensor
    ) -> Tensor:
        '''
        Constructs the output layer of the MobileBert model.

        Args:
            self (object): The instance of the MobileBertOutput class.
            intermediate_states (Tensor): The tensor representing the intermediate states of the model.
            residual_tensor_1 (Tensor): The tensor representing the first residual connection.
            residual_tensor_2 (Tensor): The tensor representing the second residual connection.

        Returns:
            Tensor: The output tensor representing the forwarded layer.

        Raises:
            None
        '''
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output


class BottleneckLayer(nn.Module):
    """BottleneckLayer"""
    def __init__(self, config):
        """
        Initializes a BottleneckLayer instance.

        Args:
            self: The instance itself.
            config (object): An object containing configuration parameters for the bottleneck layer.
                It should have the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - intra_bottleneck_size (int): The size of the bottleneck layer.
                - normalization_type (str): The type of normalization to be used.

        Returns:
            None: This method initializes the BottleneckLayer instance with the provided configuration.

        Raises:
            TypeError: If the config parameter is not an instance of the expected object type.
            ValueError: If the config object is missing any of the required attributes.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.intra_bottleneck_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Constructs a bottleneck layer.

        Args:
            self: An instance of the BottleneckLayer class.
            hidden_states (Tensor): The input hidden states. A tensor of shape (batch_size, input_size).

        Returns:
            Tensor: The output layer after applying the bottleneck transformation.
                A tensor of shape (batch_size, output_size).

        Raises:
            TypeError: If the input hidden_states is not a tensor.
            ValueError: If the input hidden_states is not of the expected shape.

        This method applies the bottleneck transformation to the input hidden states.
        It first applies a linear transformation using a dense layer, followed by layer normalization.
        The resulting tensor is then returned as the output of the bottleneck layer.
        """
        layer_input = self.dense(hidden_states)
        layer_input = self.LayerNorm(layer_input)
        return layer_input


class Bottleneck(nn.Module):
    """Bottleneck"""
    def __init__(self, config):
        """
        Initializes an instance of the 'Bottleneck' class.

        Args:
            self: The instance of the class.
            config (object):
                The configuration object containing the settings for the 'Bottleneck' instance.

                - key_query_shared_bottleneck (bool): A flag indicating whether to use a shared bottleneck for key and query.
                Set to True to enable the shared bottleneck, and False otherwise.
                - use_bottleneck_attention (bool): A flag indicating whether to use bottleneck attention.
                Set to True to enable bottleneck attention, and False otherwise.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.input = BottleneckLayer(config)
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor]:
        """
        This method forwards the bottlenecked hidden states based on the input hidden states.

        Args:
            self (Bottleneck): The instance of the Bottleneck class.
            hidden_states (Tensor): The input hidden states tensor.

        Returns:
            Tuple[Tensor]: A tuple containing the forwarded bottlenecked hidden states tensor(s).

        Raises:
            None.
        """
        # This method can return three different tuples of values. These different values make use of bottlenecks,
        # which are linear layers used to project the hidden states to a lower-dimensional vector, reducing memory
        # usage. These linear layer have weights that are learned during training.
        #
        # If `config.use_bottleneck_attention`, it will return the result of the bottleneck layer four times for the
        # key, query, value, and "layer input" to be used by the attention layer.
        # This bottleneck is used to project the hidden. This last layer input will be used as a residual tensor
        # in the attention self output, after the attention scores have been computed.
        #
        # If not `config.use_bottleneck_attention` and `config.key_query_shared_bottleneck`, this will return
        # four values, three of which have been passed through a bottleneck: the query and key, passed through the same
        # bottleneck, and the residual layer to be applied in the attention self output, through another bottleneck.
        #
        # Finally, in the last case, the values for the query, key and values are the hidden states without bottleneck,
        # and the residual layer will be this value passed through a bottleneck.

        bottlenecked_hidden_states = self.input(hidden_states)
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        if self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states
        return hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states


class FFNOutput(nn.Module):
    """FFNOutput"""
    def __init__(self, config):
        """
        Initializes an instance of the FFNOutput class.

        Args:
            self: The object instance.
            config: An object of type 'Config' containing configuration settings for the FFNOutput.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)

    def forward(self, hidden_states: Tensor, residual_tensor: Tensor) -> Tensor:
        """
        Method 'forward' in class 'FFNOutput'.

        Args:
            self: The instance of the class.
            hidden_states (Tensor):
                The input tensor containing hidden states.

                - Purpose: Represents the hidden states to be processed by the method.
                - Restrictions: Should be a valid tensor object.
            residual_tensor (Tensor):
                The input tensor containing residual values.

                - Purpose: Represents the residual tensor to be added to the processed output.
                - Restrictions: Should be a valid tensor object.

        Returns:
            Tensor:
                The output tensor after processing the hidden states and adding the residual tensor.

                - Purpose: Represents the final processed output tensor.

        Raises:
            None.
        """
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class FFNLayer(nn.Module):
    """FFNLayer"""
    def __init__(self, config):
        """
        Initialize the FFNLayer class with the provided configuration.

        Args:
            self (object): The instance of the FFNLayer class.
            config (object): An object containing configuration settings for the FFNLayer.
                This configuration is used to initialize the MobileBertIntermediate and FFNOutput components
                within the FFNLayer. It should include all necessary parameters required for the proper functioning
                of the FFNLayer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        This method forwards the output of the feedforward neural network layer.

        Args:
            self (FFNLayer): The instance of the FFNLayer class.
            hidden_states (Tensor): The input tensor representing the hidden states of the layer.

        Returns:
            Tensor: A tensor containing the output of the feedforward neural network layer.

        Raises:
            None
        """
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


class MobileBertLayer(nn.Module):
    """MobileBertLayer"""
    def __init__(self, config):
        """
        Initializes a MobileBertLayer instance.

        Args:
            self (MobileBertLayer): The MobileBertLayer instance to be initialized.
            config:
                An object containing configuration parameters for the MobileBertLayer.

                - use_bottleneck (bool): Indicates whether to use bottleneck layer.
                - num_feedforward_networks (int): Number of feedforward networks to be used.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks

        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks > 1:
            self.ffn = nn.ModuleList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        """
        The 'forward' method forwards the MobileBERT layer using the provided input parameters.

        Args:
            self: The object instance of the MobileBertLayer class.
            hidden_states (Tensor): The input tensor representing the hidden states.
            attention_mask (Optional[Tensor]): An optional tensor representing the attention mask. Defaults to None.
            head_mask (Optional[Tensor]): An optional tensor representing the head mask. Defaults to None.
            output_attentions (Optional[bool]): An optional boolean indicating whether to output attentions.
                Defaults to None.

        Returns:
            Tuple[Tensor]: A tuple containing the forwarded MobileBERT layer output tensor.

        Raises:
            None
        """
        if self.use_bottleneck:
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.num_feedforward_networks != 1:
            for _, ffn_module in enumerate(self.ffn):
                attention_output = ffn_module(attention_output)
                s += (attention_output,)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        outputs = (
            (layer_output,)
            + outputs
            + (
                Tensor(1000),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        return outputs


class MobileBertEncoder(nn.Module):
    """MobileBertEncoder"""
    def __init__(self, config):
        """
        Initialize the MobileBertEncoder.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the MobileBertEncoder.
                It is used to initialize the MobileBertLayer instances in the encoder.
                This object should have the attribute 'num_hidden_layers' which specifies the number
                of hidden layers to be created.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.layer = nn.ModuleList([MobileBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Tuple:
        """
        Constructs the MobileBertEncoder.

        Args:
            self: The MobileBertEncoder instance.
            hidden_states (Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[Tensor]): The attention mask of shape (batch_size, sequence_length)
                indicating which tokens should be attended to. Defaults to None.
            head_mask (Optional[Tensor]): The head mask of shape (num_hidden_layers, num_attention_heads)
                indicating which heads should be masked out. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions tensor. Defaults to False.
            output_hidden_states (Optional[bool]): Whether to output hidden states tensor. Defaults to False.
            return_dict (Optional[bool]): Whether to return the output as a dictionary. Defaults to True.

        Returns:
            Tuple:
                A tuple containing:

                - last_hidden_state (Tensor): The last layer hidden states of shape
                (batch_size, sequence_length, hidden_size).
                - hidden_states (Tuple): A tuple of hidden states from all layers, each of shape
                (batch_size, sequence_length, hidden_size). Only included if output_hidden_states is True.
                - attentions (Tuple): A tuple of attention tensors from all layers, each of shape
                (batch_size, num_attention_heads, sequence_length, sequence_length).
                Only included if output_attentions is True.

        Raises:
            None.
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class MobileBertPooler(nn.Module):
    """MobileBertPooler"""
    def __init__(self, config):
        """
        Initializes the MobileBertPooler class.

        Args:
            self (MobileBertPooler): The instance of the MobileBertPooler class.
            config (object):
                An object containing configuration parameters.

                - classifier_activation (bool): A boolean indicating whether to activate the classifier.
                - hidden_size (int): The size of the hidden layer.

        Returns:
            None.

        Raises:
            ValueError: If the config parameter is missing or does not contain the required attributes.
            TypeError: If the config parameter is not of the expected type.
        """
        super().__init__()
        self.do_activate = config.classifier_activation
        if self.do_activate:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Constructs a pooled output tensor from the given hidden states using the MobileBERT pooling algorithm.

        Args:
            self: An instance of the MobileBertPooler class.
            hidden_states (Tensor): A tensor containing the hidden states.
                Its shape should be (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: A tensor representing the pooled output. Its shape is (batch_size, hidden_size).

        Raises:
            None.

        Notes:
            - The 'hidden_states' tensor should have the first token as the token representing the entire sequence.
            - If the 'do_activate' flag is set to False, the function returns the first token tensor directly.
            - If the 'do_activate' flag is set to True, the function applies a dense layer followed by a hyperbolic
            tangent (tanh) activation function to the first token tensor.
            - The returned pooled output tensor can be used as input to downstream tasks such as classification or
            regression.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        if not self.do_activate:
            return first_token_tensor
        pooled_output = self.dense(first_token_tensor)
        pooled_output = ops.tanh(pooled_output)
        return pooled_output


class MobileBertPredictionHeadTransform(nn.Module):
    """MobileBertPredictionHeadTransform"""
    def __init__(self, config):
        """
        Initializes an instance of the MobileBertPredictionHeadTransform class.

        Args:
            self (MobileBertPredictionHeadTransform): The instance of the class itself.
            config: The configuration object for the MobileBertPredictionHeadTransform instance.
                It contains the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - hidden_act (str or callable): The activation function for the hidden layers.
                Can be either a string specifying a pre-defined activation function or a callable function.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = NORM2FN["layer_norm"]([config.hidden_size])

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Constructs a MobileBert prediction head transformation.

        Args:
            self (MobileBertPredictionHeadTransform): An instance of the MobileBertPredictionHeadTransform class.
            hidden_states (Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the hidden states from the previous layer.

        Returns:
            Tensor: The transformed tensor of shape (batch_size, sequence_length, hidden_size).
                It represents the hidden states after applying the transformation.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MobileBertLMPredictionHead(nn.Module):
    """MobileBertLMPredictionHead"""
    def __init__(self, config):
        """
        Initializes the MobileBertLMPredictionHead.

        Args:
            self (MobileBertLMPredictionHead): The instance of the MobileBertLMPredictionHead class.
            config (MobileBertConfig): The configuration object containing the model hyperparameters.
                This parameter specifies the configuration for the prediction head.

        Returns:
            None.

        Raises:
            ValueError: If the configuration object 'config' is not of type MobileBertConfig.
            TypeError: If the configuration object 'config' is missing required attributes.
        """
        super().__init__()
        self.transform = MobileBertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.dense = nn.Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = Parameter(ops.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Constructs the mobileBERT language model prediction head.

        Args:
            self (MobileBertLMPredictionHead): An instance of the MobileBertLMPredictionHead class.
            hidden_states (Tensor): The input hidden states to be processed.
                Expected shape is (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: The output tensor after processing the hidden states.
                The shape of the output tensor is (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(ops.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        hidden_states += self.decoder.bias
        return hidden_states


class MobileBertOnlyMLMHead(nn.Module):
    """MobileBertOnlyMLMHead"""
    def __init__(self, config):
        """
        Initialize the MobileBertOnlyMLMHead.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object for MobileBertLMPredictionHead.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(config)

    def forward(self, sequence_output: Tensor) -> Tensor:
        """
        Constructs the masked language model head for MobileBERT.

        Args:
            self (MobileBertOnlyMLMHead): The instance of the MobileBertOnlyMLMHead class.
            sequence_output (Tensor): The output tensor from the MobileBERT model, representing
                the sequence of hidden-states for each input token. The shape of the tensor should be (batch_size,
                sequence_length, hidden_size).

        Returns:
            Tensor: The prediction scores for the masked tokens. The shape of the tensor is
                (batch_size, sequence_length, vocab_size).

        Raises:
            ValueError: If the sequence_output tensor does not have the expected shape.
            TypeError: If the sequence_output parameter is not of type Tensor.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class MobileBertPreTrainingHeads(nn.Module):
    """MobileBertPreTrainingHeads"""
    def __init__(self, config):
        """
        This method initializes an instance of the MobileBertPreTrainingHeads class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = MobileBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output: Tensor, pooled_output: Tensor) -> Tuple[Tensor]:
        """
        This method forwards the pre-training heads for MobileBERT model.

        Args:
            self (MobileBertPreTrainingHeads): Instance of the MobileBertPreTrainingHeads class.
            sequence_output (Tensor): The output tensor from the model's sequence output.
            pooled_output (Tensor): The output tensor from the model's pooled output.

        Returns:
            Tuple[Tensor]: A tuple containing the prediction scores and sequence relationship score.

        Raises:
            None
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MobileBertConfig
    pretrained_model_archive_map = MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "mobilebert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if cell.bias is not None:
                cell.bias.data.zero_()
        elif isinstance(cell, nn.Embedding):
            cell.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if cell.padding_idx is not None:
                cell.weight.data[cell.padding_idx].zero_()
        elif isinstance(cell, (nn.LayerNorm, NoNorm)):
            cell.bias.data.zero_()
            cell.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the MobileBertPreTrainedModel.

        Args:
            self: An instance of the MobileBertPreTrainedModel class.

        Returns:
            None: The method modifies the instance by updating its input embeddings.

        Raises:
            None.

        This method retrieves the input embeddings of the MobileBertPreTrainedModel instance.
        The input embeddings are used as the initial embeddings for the model's input tokens.
        The method does not return any value but updates the instance by setting the input embeddings.

        Note:
            The MobileBertPreTrainedModel class should be initialized before calling this method.
        """

    def set_input_embeddings(self):
        """
        Method to set input embeddings for the MobileBertPreTrainedModel.

        Args:
            self: MobileBertPreTrainedModel object. Represents the instance of the MobileBertPreTrainedModel class.

        Returns:
            None.

        Raises:
            This method does not raise any exceptions.
        """

    def resize_position_embeddings(self):
        """
        This method resizes the position embeddings in the MobileBertPreTrainedModel.

        Args:
            self (MobileBertPreTrainedModel): The instance of the MobileBertPreTrainedModel class.
                It represents the current instance of the class to operate on.

        Returns:
            None: This method does not return any value.
                It modifies the position embeddings within the MobileBertPreTrainedModel instance.

        Raises:
            None.
        """

    def get_position_embeddings(self):
        """
        Returns the position embeddings for the MobileBertPreTrainedModel.

        Args:
            self: An instance of the MobileBertPreTrainedModel class.

        Returns:
            None.

        Raises:
            None.

        This method is responsible for retrieving the position embeddings for the MobileBertPreTrainedModel.
        Position embeddings are used in natural language processing models to capture the positional information of
        words or tokens in a sequence.

        The 'self' parameter represents an instance of the MobileBertPreTrainedModel class, which is required to
        access the position embeddings specific to that instance.

        Note that this method does not return any value. Instead, it updates the position embeddings within the
        MobileBertPreTrainedModel instance directly.

        Example:
            ```python
            >>> model = MobileBertPreTrainedModel()
            >>> model.get_position_embeddings()
            ```
        """

class MobileBertModel(MobileBertPreTrainedModel):
    """
    https://arxiv.org/pdf/2004.02984.pdf
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the MobileBertModel class.

        Args:
            self (MobileBertModel): The instance of the MobileBertModel class.
            config (object): The configuration object containing model hyperparameters.
            add_pooling_layer (bool): Flag indicating whether to add a pooling layer. Defaults to True.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes a MobileBertModel object with the provided configuration and optionally
            adds a pooling layer. The MobileBertModel is a deep learning model for natural language processing tasks.

            The 'self' parameter refers to the instance of the MobileBertModel class that is being initialized.

            The 'config' parameter is an object that contains hyperparameters and settings for the model.
            It is used to configure the MobileBertEmbeddings, MobileBertEncoder, and MobileBertPooler components of the
            MobileBertModel.

            The 'add_pooling_layer' parameter is a boolean flag that determines whether a pooling layer should be added
            to the MobileBertModel. If set to True, a MobileBertPooler component will be added to the model.
            If set to False, no pooling layer will be added.

            The method does not return any value.

        Note:
            The MobileBertModel consists of three main components: MobileBertEmbeddings, MobileBertEncoder,
            and MobileBertPooler. These components are initialized within this method and can be accessed using
            the respective instance variables: 'self.embeddings', 'self.encoder', and 'self.pooler'.
            The number of hidden layers in the model can be accessed using the 'self.num_hidden_layers' instance variable.
        """
        super().__init__(config)
        self.embeddings = MobileBertEmbeddings(config)
        self.encoder = MobileBertEncoder(config)

        self.pooler = MobileBertPooler(config) if add_pooling_layer else None
        self.num_hidden_layers = config.num_hidden_layers

    def get_input_embeddings(self):
        """get_input_embeddings"""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """set_input_embeddings"""
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
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        """
        This method 'forward' in the class 'MobileBertModel' takes 10 parameters:

        Args:
            self: The instance of the class.
            input_ids (Optional[Tensor]): The input tensor containing the token ids. Default is None.
            attention_mask (Optional[Tensor]): The attention mask tensor. Default is None.
            token_type_ids (Optional[Tensor]): The token type ids tensor. Default is None.
            position_ids (Optional[Tensor]): The position ids tensor. Default is None.
            head_mask (Optional[Tensor]): The head mask tensor. Default is None.
            inputs_embeds (Optional[Tensor]): The input embeddings tensor. Default is None.
            output_hidden_states (Optional[bool]): Flag to output hidden states. Default is None.
            output_attentions (Optional[bool]): Flag to output attentions. Default is None.
            return_dict (Optional[bool]): Flag to return a dictionary. Default is None.

        Returns:
            Tuple:
                A Tuple containing the model outputs, including the sequence output, pooled output, hidden states,
                and attentions.

        Raises:
            ValueError: Raised if both 'input_ids' and 'inputs_embeds' are provided simultaneously.
            ValueError: Raised if neither 'input_ids' nor 'inputs_embeds' are specified.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int32)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = attention_mask.expand_dims(1).expand_dims(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.ndim == 1:
                head_mask = head_mask.expand_dims(0).expand_dims(0).expand_dims(-1).expand_dims(-1)
                head_mask = ops.broadcast_to(head_mask, (self.num_hidden_layers, -1, -1, -1, -1))
            elif head_mask.ndim == 2:
                head_mask = head_mask.expand_dims(1).expand_dims(-1).expand_dims(-1)
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

@dataclass
class MobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`MobileBertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `mindspore.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`mindspore.Tensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(mindspore.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `mindspore.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(mindspore.Tensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
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


class MobileBertForPreTraining(MobileBertPreTrainedModel):
    """ MobileBertForPreTraining"""
    _keys_to_ignore_on_load_missing = [
        "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias",
        "embeddings.position_ids",
    ]

    def __init__(self, config):
        """
        Initializes a new instance of MobileBertForPreTraining.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object for MobileBertForPreTraining.
                It contains the settings and hyperparameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.mobilebert = MobileBertModel(config)
        self.cls = MobileBertPreTrainingHeads(config)

    def get_output_embeddings(self):
        """get_output_embeddings"""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """set_output_embeddings"""
        self.cls.predictions.decoder = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """resize_token_embeddings"""
        # resize dense output embedings at first
        self.cls.predictions.dense = self._get_resized_lm_head(
            self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
        )

        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        next_sentence_label: Optional[Tensor] = None,
        output_attentions: Optional[Tensor] = None,
        output_hidden_states: Optional[Tensor] = None,
        return_dict: Optional[Tensor] = None,
    ) -> Tuple:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

        Returns:
            Tuple

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MobileBertForPreTraining
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
            >>> model = MobileBertForPreTraining.from_pretrained("google/mobilebert-uncased")
            ...
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
            >>> # Batch size 1
            >>> outputs = model(input_ids)
            ...
            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MobileBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MobileBertForMaskedLM(MobileBertPreTrainedModel):
    """MobileBertForMaskedLM"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        "cls.predictions.decoder.weight",
        "cls.predictions.decoder.bias",
        "embeddings.position_ids",
    ]

    def __init__(self, config):
        """
        Initializes a MobileBertForMaskedLM instance.

        Args:
            self (MobileBertForMaskedLM): The instance of the MobileBertForMaskedLM class.
            config (MobileBertConfig): The configuration for the MobileBert model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        self.cls = MobileBertOnlyMLMHead(config)

    def get_output_embeddings(self):
        """get_output_embeddings"""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """set_output_embeddings"""
        self.cls.predictions.decoder = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """ resize_token_embeddings"""
        # resize dense output embedings at first
        self.cls.predictions.dense = self._get_resized_lm_head(
            self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
        )
        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MobileBertOnlyNSPHead(nn.Module):
    """MobileBertOnlyNSPHead"""
    def __init__(self, config):
        """
        Initializes the MobileBertOnlyNSPHead class.

        Args:
            self: The instance of the MobileBertOnlyNSPHead class.
            config: A configuration object containing the hidden size parameter for the neural network.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: Tensor) -> Tensor:
        """
        This method forwards the next sentence prediction (NSP) head for MobileBERT models.

        Args:
            self (MobileBertOnlyNSPHead): The instance of the MobileBertOnlyNSPHead class.
            pooled_output (Tensor): The pooled output tensor obtained from the MobileBERT model.
                It represents the contextualized aggregated representation of the input sequence and is used to
                predict the relationship between two sequences.

        Returns:
            Tensor: A tensor representing the score for the sequence relationship prediction task.
                The higher the score, the more likely the two input sequences are to be consecutive in the original text.

        Raises:
            None
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class MobileBertForNextSentencePrediction(MobileBertPreTrainedModel):
    """MobileBertForNextSentencePrediction"""
    def __init__(self, config):
        """
        Initializes an instance of the MobileBertForNextSentencePrediction class.

        Args:
            self: The object instance.
            config: A dictionary containing the configuration parameters for the MobileBert model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.mobilebert = MobileBertModel(config)
        self.cls = MobileBertOnlyNSPHead(config)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring) Indices should be in `[0, 1]`.

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.

        Returns:
            Tuple

        Example:
            ```python
            >>> from transformers import AutoTokenizer, MobileBertForNextSentencePrediction
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
            >>> model = MobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")
            ...
            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors="ms")
            ...
            >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
            >>> loss = outputs.loss
            >>> logits = outputs.logits
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

        outputs = self.mobilebert(
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
        seq_relationship_score = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_score,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    """MobileBertForSequenceClassification"""
    def __init__(self, config):
        """
        Initializes an instance of the MobileBertForSequenceClassification class.

        Args:
            self: The object instance.
            config: An instance of the MobileBertConfig class containing the configuration parameters for MobileBERT.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.mobilebert = MobileBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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
                elif self.num_labels > 1 and (labels.dtype in (mindspore.int32, mindspore.int_)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MobileBertForQuestionAnswering(MobileBertPreTrainedModel):
    """MobileBertForQuestionAnswering"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes a new instance of the MobileBertForQuestionAnswering class.

        Args:
            self (MobileBertForQuestionAnswering): An instance of the MobileBertForQuestionAnswering class.
            config: The configuration object for the MobileBert model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        r"""

        Args:
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
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

class MobileBertForMultipleChoice(MobileBertPreTrainedModel):
    """MobileBertForMultipleChoice"""
    def __init__(self, config):
        """
        Initialize the MobileBertForMultipleChoice class.

        Args:
            self (MobileBertForMultipleChoice): The instance of the MobileBertForMultipleChoice class.
            config (MobileBertConfig): The configuration for the MobileBert model.

        Returns:
            None.

        Raises:
            AttributeError: If the config parameter is missing required attributes.
            ValueError: If the config parameter contains invalid values.
        """
        super().__init__(config)
        self.mobilebert = MobileBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[Tensor]:
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
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.mobilebert(
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
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MobileBertForTokenClassification(MobileBertPreTrainedModel):
    """MobileBertForTokenClassification"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes a new instance of MobileBertForTokenClassification.

        Args:
            self: The object itself.
            config (MobileBertConfig): The configuration for the MobileBertForTokenClassification model.
                It contains various parameters for model initialization and behavior.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type MobileBertConfig.
            ValueError: If the num_labels attribute in the config is not defined or is invalid.
            AttributeError: If the required attributes are not found in the config.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
        "MobileBertForMaskedLM",
        "MobileBertForMultipleChoice",
        "MobileBertForNextSentencePrediction",
        "MobileBertForPreTraining",
        "MobileBertForQuestionAnswering",
        "MobileBertForSequenceClassification",
        "MobileBertForTokenClassification",
        "MobileBertLayer",
        "MobileBertModel",
        "MobileBertPreTrainedModel",
    ]
