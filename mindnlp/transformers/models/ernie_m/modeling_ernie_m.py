# coding=utf-8
# Copyright 2023 Huawei Technologies Co., Ltd
# Copyright 2023 Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang The HuggingFace Inc. team. All rights reserved.
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
""" MindSpore ErnieM model."""


import math
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_ernie_m import ErnieMConfig


logger = logging.get_logger(__name__)


ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "susnato/ernie-m-base_pytorch",
    "susnato/ernie-m-large_pytorch",
    # See all ErnieM models at https://hf-mirror.com/models?filter=ernie_m
]


# Adapted from paddlenlp.transformers.ernie_m.modeling.ErnieEmbeddings
class ErnieMEmbeddings(nn.Cell):
    """Construct the embeddings from word and position embeddings."""
    def __init__(self, config):
        """
        Args:
            self (object): The instance of the ErnieMEmbeddings class.
            config (object): An object containing configuration parameters for the ErnieMEmbeddings instance,
                including the hidden size, vocabulary size, maximum position embeddings, padding token ID, layer
                normalization epsilon, and hidden dropout probability.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain required attributes or if the padding token ID is not valid.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        This method 'construct' in the class 'ErnieMEmbeddings' constructs the embeddings for the input tokens.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input token IDs. Default is None. If None, 'inputs_embeds' is used to generate the embeddings.
            position_ids (Optional[mindspore.Tensor]): The position IDs for the input tokens.
                Default is None. If None, position IDs are calculated based on the input shape.
            inputs_embeds (Optional[mindspore.Tensor]): The input embeddings.
                Default is None. If None, input embeddings are generated using 'word_embeddings' based on 'input_ids'.
            past_key_values_length (int): The length of past key values.
                Default is 0. It is used to adjust the 'position_ids' if past key values are present.

        Returns:
            mindspore.Tensor: The constructed embeddings for the input tokens.

        Raises:
            ValueError: If the input shape is invalid or if 'position_ids' cannot be calculated.
            TypeError: If the input types are not as expected.
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if position_ids is None:
            input_shape = inputs_embeds.shape[:-1]
            ones = ops.ones(input_shape, dtype=mindspore.int64)
            seq_length = ops.cumsum(ones, axis=1)
            position_ids = seq_length - ones

            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length
        # to mimic paddlenlp implementation
        position_ids += 2
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->ErnieM,self.value->self.v_proj,self.key->self.k_proj,self.query->self.q_proj
class ErnieMSelfAttention(nn.Cell):
    """
    A module that implements the self-attention mechanism used in ERNIE model.

    This module contains the `ErnieMSelfAttention` class, which represents the self-attention mechanism used in the
    ERNIE model. It is a subclass of `nn.Cell` and is responsible for calculating the attention scores and producing
    the context layer.

    Attributes:
        num_attention_heads (int): The number of attention heads in the self-attention mechanism.
        attention_head_size (int): The size of each attention head.
        all_head_size (int): The total size of all attention heads combined.
        q_proj (nn.Dense): The projection layer for the query tensor.
        k_proj (nn.Dense): The projection layer for the key tensor.
        v_proj (nn.Dense): The projection layer for the value tensor.
        dropout (nn.Dropout): The dropout layer applied to the attention probabilities.
        position_embedding_type (str): The type of position embedding used in the attention mechanism.
        distance_embedding (nn.Embedding): The embedding layer for computing relative positions in the attention scores.
        is_decoder (bool): Whether the self-attention mechanism is used in a decoder module.

    Methods:
        transpose_for_scores:
            Reshapes the input tensor for calculating attention scores.

        construct:
            Constructs the self-attention mechanism by calculating attention scores and producing the context layer.

    Example:
        ```python
        >>> config = ErnieConfig(hidden_size=768, num_attention_heads=12, attention_probs_dropout_prob=0.1)
        >>> self_attention = ErnieMSelfAttention(config)
        ```
        """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes the ErnieMSelfAttention class.

        Args:
            self: The object itself.
            config (object): An object containing configuration parameters for the self-attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to use. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads.
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

        self.q_proj = nn.Dense(config.hidden_size, self.all_head_size)
        self.k_proj = nn.Dense(config.hidden_size, self.all_head_size)
        self.v_proj = nn.Dense(config.hidden_size, self.all_head_size)

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
        Transposes the input tensor for calculating attention scores in the ErnieMSelfAttention class.

        Args:
            self (ErnieMSelfAttention): The instance of the ErnieMSelfAttention class.
            x (mindspore.Tensor): The input tensor to be transposed.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor:
                The transposed tensor with shape (batch_size, num_attention_heads, sequence_length, attention_head_size).

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
        This method constructs the self-attention mechanism for the ErnieMSelfAttention class.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]):
                Optional tensor for masking attention scores. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): Optional tensor for masking attention heads. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                Optional tensor representing hidden states from an encoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                Optional tensor for masking encoder attention scores. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                Optional tuple of past key and value tensors. Defaults to None.
            output_attentions (Optional[bool]):
                Flag indicating whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the context layer tensor and optionally the attention probabilities tensor.

        Raises:
            ValueError: If the input tensor shapes are incompatible for matrix multiplication.
            ValueError: If the position_embedding_type specified is not supported.
            RuntimeError: If there is an issue with applying softmax or dropout operations.
            RuntimeError: If there is an issue with reshaping the context layer tensor.
        """
        mixed_query_layer = self.q_proj(hidden_states)

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
            key_layer = self.transpose_for_scores(self.k_proj(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.v_proj(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
            value_layer = self.transpose_for_scores(self.v_proj(hidden_states))
            key_layer = ops.cat([past_key_value[0], key_layer], axis=2)
            value_layer = ops.cat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.k_proj(hidden_states))
            value_layer = self.transpose_for_scores(self.v_proj(hidden_states))

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
                position_ids_l = mindspore.tensor(key_length - 1, dtype=mindspore.int64).view(
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
            # Apply the attention mask is (precomputed for all layers in ErnieMModel forward() function)
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


class ErnieMAttention(nn.Cell):

    """
    ErnieMAttention is a class that represents an attention mechanism used in the ERNIE-M model.
    It contains methods for initializing the attention mechanism, pruning attention heads, and constructing attention outputs.
    This class inherits from nn.Cell and utilizes an ErnieMSelfAttention module for self-attention calculations.
    The attention mechanism includes projection layers for query, key, and value, as well as an output projection layer.
    The `prune_heads` method allows for pruning specific attention heads based on provided indices.
    The `construct` method processes input hidden states through the self-attention mechanism and output projection
    layer to generate attention outputs.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initialize the ErnieMAttention class.

        Args:
            self: The instance of the class.
            config: An object containing configuration parameters.
            position_embedding_type: Type of position embedding to be used, default is None.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self_attn = ErnieMSelfAttention(config, position_embedding_type=position_embedding_type)
        self.out_proj = nn.Dense(config.hidden_size, config.hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' belongs to the class 'ErnieMAttention' and is responsible for pruning specific
        attention heads in the model based on the provided list of heads.

        Args:
            self: Instance of the 'ErnieMAttention' class. It is used to access attributes and methods within the class.
            heads: A list containing the indices of the attention heads that need to be pruned. Each element in the list
                should be an integer representing the index of the head to be pruned.

        Returns:
            None: This method does not return any value but modifies the attention heads in the model in-place.

        Raises:
            None:
                However, it is assumed that the functions called within this method, 
                such as 'find_pruneable_heads_and_indices' and 'prune_linear_layer', may raise exceptions related to 
                input validation or processing errors.
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self_attn.num_attention_heads, self.self_attn.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self_attn.q_proj = prune_linear_layer(self.self_attn.q_proj, index)
        self.self_attn.k_proj = prune_linear_layer(self.self_attn.k_proj, index)
        self.self_attn.v_proj = prune_linear_layer(self.self_attn.v_proj, index)
        self.out_proj = prune_linear_layer(self.out_proj, index, axis=1)

        # Update hyper params and store pruned heads
        self.self_attn.num_attention_heads = self.self_attn.num_attention_heads - len(heads)
        self.self_attn.all_head_size = self.self_attn.attention_head_size * self.self_attn.num_attention_heads
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
        This method constructs the ErnieMAttention module.

        Args:
            self: The instance of the ErnieMAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            attention_mask (Optional[mindspore.Tensor]): Optional tensor containing attention mask values.
            head_mask (Optional[mindspore.Tensor]): Optional tensor containing head mask values.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional tensor containing encoder hidden states.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional tensor containing encoder attention mask values.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional tuple containing past key and value tensors.
            output_attentions (Optional[bool]): Optional boolean indicating whether to output attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor.

        Raises:
            None
        """
        self_outputs = self.self_attn(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.out_proj(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ErnieMEncoderLayer(nn.Cell):

    """
    The ErnieMEncoderLayer class represents a single layer of the ErnieM (Enhanced Representation through kNowledge 
    Integration) encoder, which is designed for natural language processing tasks. This class inherits from the nn.Cell 
    class and implements the functionality for processing input hidden states using multi-head self-attention mechanism 
    and feedforward neural network layers with layer normalization and dropout.

    Attributes:
        self_attn: Instance of ErnieMAttention for multi-head self-attention mechanism.
        linear1: Instance of nn.Dense for the first feedforward neural network layer.
        dropout: Instance of nn.Dropout for applying dropout within the feedforward network.
        linear2: Instance of nn.Dense for the second feedforward neural network layer.
        norm1: Instance of nn.LayerNorm for the first layer normalization.
        norm2: Instance of nn.LayerNorm for the second layer normalization.
        dropout1: Instance of nn.Dropout for applying dropout after the first feedforward network layer.
        dropout2: Instance of nn.Dropout for applying dropout after the second feedforward network layer.
        activation: Activation function for the feedforward network.

    Methods:
        construct(self, hidden_states, attention_mask=None, head_mask=None, past_key_value=None, output_attentions=True):
            Applies the multi-head self-attention mechanism and feedforward network layers to the input hidden states, 
            optionally producing attention weights.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states.
            - attention_mask (Optional[mindspore.Tensor]): Optional tensor for masking the attention scores.
            - head_mask (Optional[mindspore.Tensor]): Optional tensor for masking specific attention heads.
            - past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]):
            Optional tuple containing past key and value tensors for fast decoding.
            - output_attentions (Optional[bool]): Optional boolean indicating whether to return attention weights.

            Returns:

            - mindspore.Tensor or Tuple[mindspore.Tensor]: The processed hidden states and optionally the attention weights.
    """
    def __init__(self, config):
        """
        Initialize an instance of the ErnieMEncoderLayer class.

        Args:
            self (ErnieMEncoderLayer): The instance of the ErnieMEncoderLayer class.
            config (object): 
                An object containing configuration parameters for the encoder layer.
                
                - hidden_dropout_prob (float): The probability of dropout for hidden layers. Default is 0.1.
                - act_dropout (float): The probability of dropout for activation functions. 
                Default is the value of hidden_dropout_prob.
                - hidden_size (int): The size of the hidden layers.
                - intermediate_size (int): The size of the intermediate layers.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_act (str or function): The activation function to be used. 
                If a string, it will be converted to a function using ACT2FN dictionary.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        # to mimic paddlenlp implementation
        dropout = 0.1 if config.hidden_dropout_prob is None else config.hidden_dropout_prob
        act_dropout = config.hidden_dropout_prob if config.act_dropout is None else config.act_dropout

        self.self_attn = ErnieMAttention(config)
        self.linear1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(p=act_dropout)
        self.linear2 = nn.Dense(config.intermediate_size, config.hidden_size)
        self.norm1 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act

    def construct(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_value: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = True,
    ):
        """
        Constructs an ErnieMEncoderLayer.

        This method applies the ErnieMEncoderLayer transformation to the input hidden states.

        Args:
            self: An instance of the ErnieMEncoderLayer class.
            hidden_states (mindspore.Tensor): The input hidden states. This should be a tensor.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key value tensor. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attention weights. Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        residual = hidden_states
        if output_attentions:
            hidden_states, attention_opt_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        hidden_states = residual + self.dropout1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        residual = hidden_states

        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        hidden_states = residual + self.dropout2(hidden_states)
        hidden_states = self.norm2(hidden_states)

        if output_attentions:
            return hidden_states, attention_opt_weights
        return hidden_states


class ErnieMEncoder(nn.Cell):

    """
    ErnieMEncoder represents a multi-layer Transformer-based encoder model for processing sequences of input data.

    The ErnieMEncoder class inherits from nn.Cell and implements a multi-layer Transformer-based encoder,
    with the ability to return hidden states and attention weights if specified.
    The class provides methods for initializing the model and processing input data through its layers.

    Attributes:
        config: A configuration object containing the model's hyperparameters.
        layers: A list of ErnieMEncoderLayer instances representing the individual layers of the encoder model.

    Methods:
        construct: Processes input embeddings through the encoder layers, optionally returning hidden states and
        attention weights based on the specified parameters.

    Please note that the actual code implementation is not included in this docstring.
    """
    def __init__(self, config):
        """
        Initializes an instance of the ErnieMEncoder class.

        Args:
            self (ErnieMEncoder): The instance of the ErnieMEncoder class.
            config (object): The configuration object containing settings for the ErnieMEncoder.
                This parameter is required for configuring the ErnieMEncoder instance.
                It should be an object that provides necessary configuration details.
                It is expected to have attributes such as num_hidden_layers to specify the number of hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layers = nn.CellList([ErnieMEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        input_embeds: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        """
        Constructs the ErnieMEncoder.

        Args:
            self: The instance of the class.
            input_embeds (mindspore.Tensor): The input embeddings. Shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask. Shape (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask. Shape (num_layers, num_heads).
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values.
                Shape (num_layers, 2, batch_size, num_heads, sequence_length // num_heads, hidden_size // num_heads).
            output_attentions (Optional[bool]): Whether to output attention weights. Default is False.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is False.
            return_dict (Optional[bool]): Whether to return a BaseModelOutputWithPastAndCrossAttentions. Default is True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                The encoded last hidden state, optional hidden states, and optional attention weights.

        Raises:
            None.
        """
        hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None

        output = input_embeds
        if output_hidden_states:
            hidden_states = hidden_states + (output,)
        for i, layer in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            output, opt_attn_weights = layer(
                hidden_states=output,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                past_key_value=past_key_value,
            )

            if output_hidden_states:
                hidden_states = hidden_states + (output,)
            if output_attentions:
                attentions = attentions + (opt_attn_weights,)

        last_hidden_state = output
        if not return_dict:
            return tuple(v for v in [last_hidden_state, hidden_states, attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state, hidden_states=hidden_states, attentions=attentions
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->ErnieM
class ErnieMPooler(nn.Cell):
    """
    This class represents the MPooler module of the ERNIE model, which is responsible for pooling the hidden states to
    obtain a single representation of the input sequence.

    Inherits from:
        nn.Cell

    Attributes:
        dense (nn.Dense): A fully connected layer that projects the input hidden states to a new hidden size.
        activation (nn.Tanh): The activation function applied to the projected hidden states.

    Methods:
        __init__(config): Initializes the ERNIE MPooler module.
        construct(hidden_states): Constructs the MPooler module by pooling the hidden states.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the ErnieMPooler class.

        Args:
            self: The object instance.
            config: An instance of the configuration class used to configure the ErnieMPooler.
                It provides various settings and parameters for the ErnieMPooler's behavior. This parameter is required.

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
        Constructs the pooled output tensor for the ERNIE model.

        Args:
            self (ErnieMPooler): An instance of the ErnieMPooler class.
            hidden_states (mindspore.Tensor): A tensor containing the hidden states from the ERNIE model.
                It should have the shape (batch_size, sequence_length, hidden_size) where:

                - batch_size: The number of sequences in the batch.
                - sequence_length: The length of each input sequence.
                - hidden_size: The size of the hidden state vectors.

        Returns:
            mindspore.Tensor: A tensor representing the pooled output of the ERNIE model.
                The pooled output is obtained by applying dense and activation layers to the first token tensor
                extracted from the hidden states tensor.

        Raises:
            None
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ErnieMConfig
    base_model_prefix = "ernie_m"

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


class ErnieMModel(ErnieMPreTrainedModel):

    """
    This class represents an ERNIE-M (Enhanced Representation through kNowledge Integration) model for multi-purpose
    pre-training and fine-tuning on downstream tasks. It incorporates ERNIE-M embeddings, encoder, and optional pooling
    layer. The class provides methods for initializing, getting and setting input embeddings, pruning model heads,
    and constructing the model with various input and output options.
    The class inherits from ErnieMPreTrainedModel and extends its functionality to support specific ERNIE-M model
    architecture and operations.
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes the ErnieMModel.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing model settings.
            add_pooling_layer (bool): A flag indicating whether to add a pooling layer to the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.initializer_range = config.initializer_range
        self.embeddings = ErnieMEmbeddings(config)
        self.encoder = ErnieMEncoder(config)
        self.pooler = ErnieMPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings from the ErnieMModel.

        Args:
            self: ErnieMModel object. The instance of the ErnieMModel class.

        Returns:
            word_embeddings: The method returns the input embeddings from the ErnieMModel.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Set the input embeddings for the ErnieMModel.

        Args:
            self (ErnieMModel): The instance of the ErnieMModel class.
            value: The input embeddings value to be set. It should be a tensor representing the input embeddings.

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
            self.encoder.layers[layer].self_attn.prune_heads(heads)

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Constructs the ERNIE-M model.

        Args:
            self: The object instance.
            input_ids (Optional[mindspore.Tensor]): The input tensor of token indices. Default is None.
            position_ids (Optional[mindspore.Tensor]): The tensor indicating the position of tokens. Default is None.
            attention_mask (Optional[mindspore.Tensor]):
                The tensor indicating which elements in the input do not need to be attended to. Default is None.
            head_mask (Optional[mindspore.Tensor]):
                The tensor indicating the heads in the multi-head attention layer to be masked. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The input embeddings. Default is None.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The previous key values. Default is None.
            use_cache (Optional[bool]): Whether to use the cache. Default is None.
            output_hidden_states (Optional[bool]): Whether to output the hidden states. Default is None.
            output_attentions (Optional[bool]): Whether to output the attentions. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default is None.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
                Depending on the value of `return_dict`, returns a tuple of tensors including the last hidden state and
                the pooler output, or a BaseModelOutputWithPoolingAndCrossAttentions object.

        Raises:
            ValueError: If both `input_ids` and `inputs_embeds` are specified.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # Adapted from paddlenlp.transformers.ernie_m.ErnieMModel
        if attention_mask is None:
            attention_mask = (input_ids == 0).to(self.dtype)
            attention_mask *= mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(attention_mask.dtype)).min, attention_mask.dtype)
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = ops.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = ops.concat([past_mask, attention_mask], axis=-1)
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = attention_mask.to(self.dtype)
            attention_mask = 1.0 - attention_mask
            attention_mask *= mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(attention_mask.dtype)).min, attention_mask.dtype)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            sequence_output = encoder_outputs[0]
            pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
            return (sequence_output, pooler_output) + encoder_outputs[1:]

        sequence_output = encoder_outputs["last_hidden_state"]
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
        hidden_states = None if not output_hidden_states else encoder_outputs["hidden_states"]
        attentions = None if not output_attentions else encoder_outputs["attentions"]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class ErnieMForSequenceClassification(ErnieMPreTrainedModel):

    """
    ErnieMForSequenceClassification is a class that represents a fine-tuned ErnieM model for sequence classification tasks.
    It inherits from ErnieMPreTrainedModel and implements methods for initializing the model and constructing predictions.

    Attributes:
        num_labels: Number of labels for sequence classification.
        config: Configuration object for the model.
        ernie_m: ErnieMModel instance for processing input sequences.
        dropout: Dropout layer for regularization.
        classifier: Dense layer for classification predictions.

    Methods:
        __init__: Initializes the ErnieMForSequenceClassification instance with the provided configuration.
        construct:
            Constructs the model for making predictions on input sequences and computes the loss based on predicted labels.

            Args:

            - input_ids (Optional[mindspore.Tensor]): Tensor of input token IDs.
            - attention_mask (Optional[mindspore.Tensor]): Tensor of attention masks.
            - position_ids (Optional[mindspore.Tensor]): Tensor of position IDs.
            - head_mask (Optional[mindspore.Tensor]): Tensor of head masks.
            - inputs_embeds (Optional[mindspore.Tensor]): Tensor of input embeddings.
            - past_key_values (Optional[List[mindspore.Tensor]]): List of past key values for caching.
            - use_cache (Optional[bool]): Flag for using caching.
            - output_hidden_states (Optional[bool]): Flag for outputting hidden states.
            - output_attentions (Optional[bool]): Flag for outputting attentions.
            - return_dict (Optional[bool]): Flag for returning output in a dictionary format.
            - labels (Optional[mindspore.Tensor]): Tensor of target labels for computing loss.

            Returns:

            - Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]: Tuple of model outputs and loss.

            Raises:

            - ValueError: If the provided labels are not in the expected format or number.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes an instance of the ErnieMForSequenceClassification class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing settings for the model initialization.
                It must have the following attributes:

                - num_labels (int): The number of labels for classification.
                - classifier_dropout (float, optional): The dropout probability for the classifier layer.
                If not provided, it defaults to the hidden dropout probability.
                - hidden_dropout_prob (float): The default hidden dropout probability.

        Returns:
            None.

        Raises:
            ValueError: If the config object is missing the num_labels attribute.
            TypeError: If the config object does not have the expected attributes or if their types are incorrect.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ernie_m = ErnieMModel(config)
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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple[mindspore.Tensor], SequenceClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
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


class ErnieMForMultipleChoice(ErnieMPreTrainedModel):

    """
    ErnieMForMultipleChoice is a class that represents a multiple choice question answering model based on the
    ERNIE-M architecture.
    It inherits from ErnieMPreTrainedModel and implements methods for constructing the model and computing the multiple
    choice classification loss.

    Attributes:
        ernie_m (ErnieMModel): The ERNIE-M model used for processing inputs.
        dropout (nn.Dropout): Dropout layer used in the classifier.
        classifier (nn.Dense): Dense layer for classification.

    Methods:
        __init__: Initializes the ErnieMForMultipleChoice model with the given configuration.
        construct: Constructs the model for multiple choice question answering and computes the classification loss.

    The construct method takes various input tensors and parameters, processes them through the ERNIE-M model,
    applies dropout, and computes the classification logits.
    If labels are provided, it calculates the cross-entropy loss. The method returns the loss and model outputs based on
    the return_dict parameter.

    This class is designed to be used for multiple choice question answering tasks with ERNIE-M models.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes an instance of the ErnieMForMultipleChoice class.

        Args:
            self: The object instance.
            config: An instance of the ErnieMConfig class containing the model configuration.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.ernie_m = ErnieMModel(config)
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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
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
        position_ids = position_ids.view(-1, position_ids.shape[-1]) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
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


class ErnieMForTokenClassification(ErnieMPreTrainedModel):

    """
    This class represents a fine-tuned ErnieM model for token classification tasks. It inherits from the ErnieMPreTrainedModel class.

    The ErnieMForTokenClassification class implements the necessary methods and attributes for token classification tasks.
    It takes a configuration object as input during initialization and sets up the model architecture accordingly.
    The model consists of an ErnieMModel instance, a dropout layer, and a classifier layer.

    Methods:
        __init__: Initializes the ErnieMForTokenClassification instance with the given configuration.
            It sets the number of labels, creates an ErnieMModel object, initializes the dropout layer, and
            creates the classifier layer.

        construct: Constructs the forward pass of the model. It takes various input tensors and returns the token
            classification output. Optionally, it can also compute the token classification loss if labels are provided.

    Attributes:
        num_labels: The number of possible labels for the token classification task.

    Example:
        ```python
        >>> config = ErnieMConfig()
        >>> model = ErnieMForTokenClassification(config)
        >>> input_ids = ...
        >>> attention_mask = ...
        >>> output = model.construct(input_ids=input_ids, attention_mask=attention_mask)
        ```

    Note:
        It is important to provide the input tensors in the correct shape and format to ensure proper model functioning.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes an instance of the ErnieMForTokenClassification class.

        Args:
            self: The instance of the ErnieMForTokenClassification class.
            config: An instance of the configuration class containing the model configuration settings.

        Returns:
            None

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie_m = ErnieMModel(config, add_pooling_layer=False)
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
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        labels: Optional[mindspore.Tensor] = None,
    ) -> Union[Tuple[mindspore.Tensor], TokenClassifierOutput]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
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


class ErnieMForQuestionAnswering(ErnieMPreTrainedModel):

    """
    ErnieMForQuestionAnswering is a class that represents a fine-tuned ErnieM model for question answering tasks.
    It is a subclass of ErnieMPreTrainedModel.

    This class extends the functionality of the base ErnieM model by adding a question answering head on top of it.
    It takes as input the configuration of the model and initializes the necessary components.
    The class provides a method called 'construct' which performs the forward pass of the model for question answering.

    The 'construct' method takes several input tensors such as 'input_ids', 'attention_mask', 'position_ids',
    'head_mask', and 'inputs_embeds'. It also supports optional inputs like 'start_positions', 'end_positions',
    'output_attentions', 'output_hidden_states', and 'return_dict'.
    The method returns the question answering model output, which includes the start and end logits, hidden states,
    attentions, and an optional total loss.

    The 'construct' method internally calls the 'ernie_m' method of the base ErnieM model to obtain the sequence output.
    It then passes the sequence output through a dense layer 'qa_outputs' to get the logits for the start and end
    positions. The logits are then processed to obtain the final start and end logits. If 'start_positions' and
    'end_positions' are provided, the method calculates the cross-entropy loss for the predicted logits and the provided
    positions. The total loss is computed as the average of the start and end losses.

    The 'construct' method returns the model output in a structured manner based on the 'return_dict' parameter.

    - If 'return_dict' is False, the method returns a tuple containing the total loss, start logits, end logits, and any
    additional hidden states or attentions.
    - If 'return_dict' is True, the method returns an instance of the 'QuestionAnsweringModelOutput' class, which
    encapsulates the output elements as attributes.

    Note:
        - If 'start_positions' and 'end_positions' are not provided, the total loss will be None.
        - The start and end positions are clamped to the length of the sequence and positions outside the sequence are
        ignored for computing the loss.
    
    """
    # Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """Initializes a new instance of the ErnieMForQuestionAnswering class.
        
        Args:
            self: The object itself.
            config: An instance of the ErnieMConfig class containing the model configuration.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie_m = ErnieMModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
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

        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
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


# Copied from paddlenlp.transformers.ernie_m.modeling.UIEM
class ErnieMForInformationExtraction(ErnieMPreTrainedModel):

    """
    ErnieMForInformationExtraction is a class that represents an ErnieM model for information extraction tasks. 
    It inherits from ErnieMPreTrainedModel and includes methods for initializing the model and constructing the forward pass.
    
    Attributes:
        ernie_m (ErnieMModel): The ErnieM model used for information extraction.
        linear_start (nn.Dense): Linear layer for predicting the start position in the input sequence.
        linear_end (nn.Dense): Linear layer for predicting the end position in the input sequence.
        sigmoid (nn.Sigmoid): Sigmoid activation function for probability calculation.
    
    Methods:
        __init__: Initializes the ErnieMForInformationExtraction class with the provided configuration.
        construct: Constructs the forward pass of the model for information extraction tasks.
    
    Args:
        input_ids (mindspore.Tensor): Input tensor containing token ids.
        attention_mask (mindspore.Tensor): Tensor specifying which tokens should be attended to.
        position_ids (mindspore.Tensor): Tensor specifying the position ids of tokens.
        head_mask (mindspore.Tensor): Tensor for masking specific heads in the self-attention layers.
        inputs_embeds (mindspore.Tensor): Tensor for providing custom embeddings instead of token ids.
        start_positions (mindspore.Tensor): Labels for start positions in the input sequence.
        end_positions (mindspore.Tensor): Labels for end positions in the input sequence.
        output_attentions (bool): Flag to output attention weights.
        output_hidden_states (bool): Flag to output hidden states.
        return_dict (bool): Flag to return outputs as a dictionary.
    
    Returns:
        Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]: Tuple of output tensors or a QuestionAnsweringModelOutput object.
    
    Raises:
        ValueError: If start_positions or end_positions are not of the expected shape.
    
    """
    def __init__(self, config):
        """
        Initializes a new instance of the ErnieMForInformationExtraction class.
        
        Args:
            self: The instance of the class.
            config: An instance of the ErnieMConfig class containing the configuration parameters for the model.
        
        Returns:
            None
        
        Raises:
            None
        """
        super().__init__(config)
        self.ernie_m = ErnieMModel(config)
        self.linear_start = nn.Dense(config.hidden_size, 1)
        self.linear_end = nn.Dense(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.post_init()

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for position (index) for computing the start_positions loss. Position outside of the sequence are
                not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) for computing the end_positions loss. Position outside of the sequence are not
                taken into account for computing the loss.
        """
        result = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            sequence_output = result.last_hidden_state
        elif not return_dict:
            sequence_output = result[0]

        start_logits = self.linear_start(sequence_output)
        start_logits = start_logits.squeeze(-1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = end_logits.squeeze(-1)
        end_prob = self.sigmoid(end_logits)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1 and start_positions.shape[-1] == 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1 and end_positions.shape[-1] == 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.binary_cross_entropy(start_prob, start_positions)
            end_loss = ops.binary_cross_entropy(end_prob, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            return tuple(
                i
                for i in [total_loss, start_prob, end_prob, result.hidden_states, result.attentions]
                if i is not None
            )

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_prob,
            end_logits=end_prob,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
        )

class UIEM(ErnieMForInformationExtraction):
    """UIEM model"""
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        start_positions: Optional[mindspore.Tensor] = None,
        end_positions: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]:
        r"""
        Args:
            start_positions (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for position (index) for computing the start_positions loss. Position outside of the sequence are
                not taken into account for computing the loss.
            end_positions (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) for computing the end_positions loss. Position outside of the sequence are not
                taken into account for computing the loss.
        """
        result = self.ernie_m(
            input_ids,
            # attention_mask=attention_mask,
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if return_dict:
            sequence_output = result.last_hidden_state
        elif not return_dict:
            sequence_output = result[0]

        start_logits = self.linear_start(sequence_output)
        start_logits = start_logits.squeeze(-1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = end_logits.squeeze(-1)
        end_prob = self.sigmoid(end_logits)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1 and start_positions.shape[-1] == 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1 and end_positions.shape[-1] == 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = ops.binary_cross_entropy(start_prob, start_positions)
            end_loss = ops.binary_cross_entropy(end_prob, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            return tuple(
                i
                for i in [total_loss, start_prob, end_prob, result.hidden_states, result.attentions]
                if i is not None
            )

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_prob,
            end_logits=end_prob,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
        )

__all__ = [
    "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
    "ErnieMForMultipleChoice",
    "ErnieMForQuestionAnswering",
    "ErnieMForSequenceClassification",
    "ErnieMForTokenClassification",
    "ErnieMModel",
    "ErnieMPreTrainedModel",
    "ErnieMForInformationExtraction",
    "UIEM"
]
