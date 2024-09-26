# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
"""MindSpore ALBERT model."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from mindnlp.utils import (
    ModelOutput,
    logging,
)
from .configuration_albert import AlbertConfig
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "albert-base-v2"
_CONFIG_FOR_DOC = "AlbertConfig"


ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    # See all ALBERT models at https://hf-mirror.com/models?filter=albert
]


class AlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the `AlbertEmbeddings` class.
        
        Args:
            self: The object itself.
            config (AlbertConfig):
                The configuration object containing various parameters for the embeddings.

                - `vocab_size` (int): The size of the vocabulary.
                - `embedding_size` (int): The size of the embeddings.
                - `pad_token_id` (int): The ID of the padding token.
                - `max_position_embeddings` (int): The maximum number of positions for the embeddings.
                - `type_vocab_size` (int): The size of the token type vocabulary.
                - `layer_norm_eps` (float): The epsilon value for LayerNorm.
                - `hidden_dropout_prob` (float): The dropout probability for embeddings.
                - `position_embedding_type` (str, optional): The type of position embeddings. Defaults to 'absolute'.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm([config.embedding_size], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.token_type_ids = ops.zeros(*self.position_ids.shape, dtype=mindspore.int64)

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        This method 'forward' is a part of the 'AlbertEmbeddings' class and is used to forward the embeddings for input tokens in the Albert model.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input token IDs, representing the index of each token in the vocabulary. Default is None.
            token_type_ids (Optional[mindspore.Tensor]):
                The token type IDs, representing the segment ID for each token (e.g., sentence A or B). Default is None.
            position_ids (Optional[mindspore.Tensor]):
                The position IDs, representing the position of each token in the sequence. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input embeddings directly provided instead of input_ids. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            mindspore.Tensor: The forwarded embeddings for the input tokens.

        Raises:
            ValueError: If the input shape and inputs_embeds shape are incompatible.
            ValueError: If the position embedding type is not supported.
            ValueError: If the token type embeddings shape and input_shape are incompatible.
            ValueError: If the position embeddings shape and input_shape are incompatible.
            ValueError: If the dimensions of input_shape are not as expected during computations.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in forwardor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(*input_shape, dtype=mindspore.int64)

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


class AlbertAttention(nn.Module):

    """
    A class representing the attention mechanism for the ALBERT (A Lite BERT) model.

    This class implements the attention mechanism used in the ALBERT model for processing input sequences.
    It includes methods for processing queries, keys, and values, calculating attention scores, applying attention masks,
    handling position embeddings, and generating the final contextualized output.

    This class inherits from the nn.Module class and contains the following methods:

    - __init__(self, config: AlbertConfig): Initializes the AlbertAttention instance with the provided configuration.
    - transpose_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor: Transposes the input tensor for calculating attention scores.
    - prune_heads(self, heads: List[int]) -> None: Prunes specific attention heads from the model.
    - forward(self, hidden_states: mindspore.Tensor, attention_mask: Optional[mindspore.Tensor] = None,
    head_mask: Optional[mindspore.Tensor] = None, output_attentions: bool = False) ->
    Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]: Constructs the output based on the input
    hidden states, applying attention and head masks if provided.

    The AlbertAttention class is a crucial component in the ALBERT model architecture, responsible for capturing
    interactions between tokens in the input sequence to generate contextualized representations.

    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertAttention class.

        Args:
            self: The instance of the class.
            config (AlbertConfig):
                An object of type AlbertConfig containing configuration parameters for the Albert model.

                - config.hidden_size (int): The size of the hidden layers in the model.
                - config.num_attention_heads (int): The number of attention heads in the model.
                - config.embedding_size (int, optional): The size of the embeddings in the model.
                - config.attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
                - config.hidden_dropout_prob (float): The dropout probability for hidden layers.
                - config.layer_norm_eps (float): The epsilon value for LayerNorm.
                - config.position_embedding_type (str): The type of position embedding ('absolute', 'relative_key', 'relative_key_query').
                - config.max_position_embeddings (int): The maximum position embeddings allowed.

        Returns:
            None.

        Raises:
            ValueError: Raised if the hidden_size is not a multiple of the num_attention_heads and no embedding_size is provided.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.pruned_heads = set()

        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    # Copied from transformers.models.bert.modeling_bert.BertSelfAttention.transpose_for_scores
    def transpose_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Transpose the input tensor for calculating attention scores in the AlbertAttention class.

        Args:
            self (AlbertAttention): The instance of the AlbertAttention class.
            x (mindspore.Tensor): The input tensor to be transposed. It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor:
                The transposed tensor with shape (batch_size, num_attention_heads, sequence_length, attention_head_size).
                The attention_head_size is calculated as hidden_size / num_attention_heads.

        Raises:
            None.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def prune_heads(self, heads: List[int]) -> None:
        """
        This method prunes specific attention heads from the AlbertAttention class.

        Args:
            self: The instance of the AlbertAttention class.
            heads (List[int]): A list of integers representing the attention heads to be pruned. If the list is empty, no action is taken.

        Returns:
            None: This method does not return any value, it modifies the internal state of the AlbertAttention instance.

        Raises:
            None
        """
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:
        '''
        Constructs the attention mechanism for the Albert model.

        Args:
            self (AlbertAttention): An instance of the AlbertAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape (batch_size, seq_length, seq_length).
                Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor of shape (num_attention_heads, seq_length, seq_length).
                Defaults to None.
            output_attentions (bool): Whether to output the attention probabilities. Defaults to False.

        Returns:
            Union[Tuple[mindspore.Tensor], Tuple[mindspore.Tensor, mindspore.Tensor]]:

                - If output_attentions is False, returns a tuple containing:

                    - layernormed_context_layer (mindspore.Tensor): The output tensor after applying layer normalization
                    of shape (batch_size, seq_length, hidden_size).

                - If output_attentions is True, returns a tuple containing:

                    - layernormed_context_layer (mindspore.Tensor): The output tensor after applying layer normalization
                    of shape (batch_size, seq_length, hidden_size).

                    - attention_probs (mindspore.Tensor): The attention probabilities tensor of shape
                    (batch_size, num_attention_heads, seq_length, seq_length).

        Raises:
            None.
        '''
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            seq_length = hidden_states.shape[1]
            position_ids_l = ops.arange(seq_length, dtype=mindspore.int64).view(-1, 1)
            position_ids_r = ops.arange(seq_length, dtype=mindspore.int64).view(1, -1)
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

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        context_layer = context_layer.swapaxes(2, 1).flatten(start_dim=2)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(hidden_states + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):

    '''
    This class represents an AlbertLayer module, which is a single layer of the Albert model.
    It inherits from nn.Module and contains methods for initialization and forward pass computation.

    The __init__ method initializes the AlbertLayer with the provided configuration.
    It sets various attributes based on the configuration, including chunk size for feed forward, sequence length dimension,
    layer normalization, attention module, feed forward network, activation function, and dropout.

    The forward method computes the forward pass for the AlbertLayer.
    It takes hidden_states, attention_mask, head_mask, output_attentions, and output_hidden_states as input and returns the hidden states
    along with optional attention outputs.

    The ff_chunk method is a helper function used within the forward method to perform the feed forward computation.

    Note:
        This class assumes that the nn module is imported as nn and that the AlbertAttention and ACT2FN classes are defined elsewhere.
    '''
    def __init__(self, config: AlbertConfig):
        """Initializes an instance of the AlbertLayer class.

        Args:
            self: The instance of the class.
            config (AlbertConfig): The configuration object for the Albert model.
                This object contains various settings and hyperparameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        '''
        Constructs an AlbertLayer.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states.
                Tensor of shape (batch_size, seq_len, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): Mask for attention computation.
                Tensor of shape (batch_size, seq_len).
            head_mask (Optional[mindspore.Tensor]): Mask for attention computation.
                Tensor of shape (num_heads,) or (num_layers, num_heads).
            output_attentions (bool): Whether to output attentions.
            output_hidden_states (bool): Whether to output hidden states.

        Returns:
            Tuple[mindspore.Tensor, mindspore.Tensor]: A tuple containing the updated hidden states
            and additional outputs based on the arguments.

        Raises:
            ValueError: If the shapes of input tensors are invalid.
            TypeError: If the input types are incorrect.
            RuntimeError: If an error occurs during the computation.
        '''
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them

    def ff_chunk(self, attention_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        Performs a feedforward chunk operation on the input attention output tensor.

        Args:
            self: Instance of the AlbertLayer class.
            attention_output (mindspore.Tensor): The input tensor representing the attention output.
                This tensor is expected to have the shape (batch_size, seq_length, hidden_size).
                It serves as the input to the feedforward network.

        Returns:
            mindspore.Tensor: The output tensor after applying the feedforward chunk operation.
                The shape of the returned tensor is expected to be (batch_size, seq_length, hidden_size).

        Raises:
            None
        """
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Module):

    """
    This class represents a group of Albert layers within the Albert model. It inherits from the nn.Module class.

    Attributes:
        albert_layers (nn.ModuleList): A list of AlbertLayer instances that make up the group.

    Methods:
        __init__:
            Initializes an instance of the AlbertLayerGroup class.

        forward:
            Constructs the AlbertLayerGroup by applying each AlbertLayer in the group to the input hidden_states.
            This method returns the resulting hidden states and optionally the layer attentions and hidden states.

    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertLayerGroup class.

        Args:
            self: The instance of the class.
            config (AlbertConfig):
                An instance of the AlbertConfig class that holds the configuration parameters for the Albert model.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes an instance of the AlbertLayerGroup class.
            It takes in a configuration object of type AlbertConfig, which holds the configuration parameters for the Albert model.
            The method initializes the superclass and creates a list of AlbertLayer objects, each with the given configuration parameters.
            The number of AlbertLayer objects in the list is determined by the 'inner_group_num' parameter of the configuration object.
        """
        super().__init__()

        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Tuple[Union[mindspore.Tensor, Tuple[mindspore.Tensor]], ...]:
        """
        Constructs an Albert Layer Group.

        Args:
            self: An instance of the AlbertLayerGroup class.
            hidden_states (mindspore.Tensor): The input hidden states of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask tensor of shape (num_hidden_layers, num_attention_heads).
            output_attentions (bool): Whether to return the attention weights. Default is False.
            output_hidden_states (bool): Whether to return the hidden states of all layers. Default is False.

        Returns:
            Tuple[Union[mindspore.Tensor, Tuple[mindspore.Tensor]], ...]:
                A tuple containing the output hidden states of shape (batch_size, sequence_length, hidden_size).

                - If output_hidden_states is True, the tuple also contains the hidden states of all layers.
                - If output_attentions is True, the tuple also contains the attention weights of all layers.

        Raises:
            None.
        """
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(nn.Module):

    """
    This class represents the AlbertTransformer, which is a part of the Albert model in the MindSpore library. It is responsible for forwarding the Albert transformer layers.

    The AlbertTransformer class inherits from the nn.Module class.

    Attributes:
        config (AlbertConfig): The configuration object for the Albert model.
        embedding_hidden_mapping_in (nn.Linear): The dense layer to map the input hidden states to the embedding size.
        albert_layer_groups (nn.ModuleList): A list of AlbertLayerGroup instances representing the transformer layers.

    Methods:
        forward(hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
            Constructs the Albert transformer layers.

            - Args:

                - hidden_states (mindspore.Tensor): The input hidden states.
                - attention_mask (Optional[mindspore.Tensor]): The attention mask tensor (default None).
                - head_mask (Optional[mindspore.Tensor]): The head mask tensor (default None).
                - output_attentions (bool): Whether to output attentions (default False).
                - output_hidden_states (bool): Whether to output hidden states (default False).
                - return_dict (bool): Whether to return the output as a BaseModelOutput instance (default True).

            - Returns:

                - Union[BaseModelOutput, Tuple]: The output as a BaseModelOutput instance or a tuple of tensors.

    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertTransformer class.

        Args:
            self: The instance of the AlbertTransformer class.
            config (AlbertConfig): An instance of AlbertConfig specifying the configuration settings for the transformer.
                The config parameter defines the model's architecture, including the embedding size, hidden size, and the number of hidden groups.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self,
        hidden_states: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        """
        Constructs the AlbertTransformer.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states to be transformed.
            attention_mask (Optional[mindspore.Tensor]): A tensor representing the attention mask, defaults to None.
            head_mask (Optional[mindspore.Tensor]): A tensor representing the head mask, defaults to None.
            output_attentions (bool): A boolean indicating whether to output attentions, defaults to False.
            output_hidden_states (bool): A boolean indicating whether to output hidden states, defaults to False.
            return_dict (bool): A boolean indicating whether to return a dictionary, defaults to True.

        Returns:
            Union[BaseModelOutput, Tuple]:
                The output value, which could be either BaseModelOutput or a tuple.

        Raises:
            None.
        """
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        head_mask = [None] * self.config.num_hidden_layers if head_mask is None else head_mask

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class AlbertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AlbertConfig
    base_model_prefix = "albert"

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


@dataclass
class AlbertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`AlbertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `mindspore.Tensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`mindspore.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        sop_logits (`mindspore.Tensor` of shape `(batch_size, 2)`):
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
    sop_logits: mindspore.Tensor = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None


class AlbertModel(AlbertPreTrainedModel):

    """
    This class represents the AlbertModel, which inherits from AlbertPreTrainedModel.
    It includes methods for initializing the model, getting and setting input embeddings, pruning heads of the model, and
    forwarding the model. The 'forward' method takes various input parameters and returns the model output.
    The class also includes detailed comments and error handling for certain scenarios.
    The 'prune_heads' method is used to prune heads of the model, and the 'forward' method forwards the model based on input parameters.
    The model outputs are returned based on the specified conditions.

    For more information and usage details, refer to the base class 'PreTrainedModel'.
    """
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, config: AlbertConfig, add_pooling_layer: bool = True):
        """
        Initializes an instance of the AlbertModel class.

        Args:
            self: The instance of the class.
            config (AlbertConfig): An instance of AlbertConfig containing the model configuration.
            add_pooling_layer (bool, optional): A flag indicating whether to add a pooling layer. Defaults to True.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type AlbertConfig.
            ValueError: If the config parameter is invalid or if the add_pooling_layer parameter is not a boolean.
        """
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        if add_pooling_layer:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.pooler_activation = nn.Tanh()
        else:
            self.pooler = None
            self.pooler_activation = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Retrieve the input embeddings for the AlbertModel.

        Args:
            self (object): The instance of the AlbertModel class.
                This parameter is required to access the instance attributes and methods.

        Returns:
            nn.Embedding: An instance of the nn.Embedding class representing the input embeddings.
                The input embeddings are used to convert input tokens into their corresponding word embeddings.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """
        Set input embeddings for the AlbertModel.

        Args:
            self (object): The instance of the AlbertModel class.
            value (nn.Embedding): The input embeddings to be set for the model. It should be an instance of nn.Embedding class.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT has
        a different architecture in that its layers are shared across groups, which then has inner groups. If an ALBERT
        model has 12 hidden layers and 2 hidden groups, with two inner groups, there is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error. See base class PreTrainedModel for more
        information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPooling, Tuple]:
        """
        Constructs the AlbertModel.

        Args:
            self (AlbertModel): The instance of the AlbertModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the IDs of the tokens. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The tensor specifying which tokens should be attended to. Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The tensor containing the type IDs of the tokens. Default is None.
            position_ids (Optional[mindspore.Tensor]): The tensor containing the position IDs of the tokens. Default is None.
            head_mask (Optional[mindspore.Tensor]): The tensor specifying which heads to mask in the attention layers. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing the embedded representations of the input tokens. Default is None.
            output_attentions (Optional[bool]): Whether to output the attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output the hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a BaseModelOutputWithPooling object. Default is None.

        Returns:
            Union[BaseModelOutputWithPooling, Tuple]: Either a BaseModelOutputWithPooling object or a tuple containing
                the sequence output, pooled output, hidden states, and attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are specified.
            ValueError: If neither input_ids nor inputs_embeds are specified.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        if attention_mask is None:
            attention_mask = ops.ones(*input_shape)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.broadcast_to((batch_size, seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(*input_shape, dtype=mindspore.int64)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0])) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AlbertForPreTraining(AlbertPreTrainedModel):

    """
    The `AlbertForPreTraining` class represents an Albert model for pre-training, inheriting from `AlbertPreTrainedModel`.
    It includes methods for initializing the model with the specified configuration, retrieving output embeddings,
    setting new output embeddings, retrieving input embeddings, and forwarding the model for pre-training tasks.
    The `forward` method accepts various input parameters and returns pre-training outputs. I
    t also includes examples of usage.

    The `AlbertForPreTraining` class provides functionality for masked language modeling and next sequence prediction (classification) loss.
    It utilizes the Albert model, prediction heads, and sentence order prediction head to compute the total loss for pre-training tasks.

    For additional details and examples on how to use the `AlbertForPreTraining` class,
    please refer to the provided code example and the official documentation for the `transformers` library.
    """
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertForPreTraining class.

        Args:
            self: The instance of the class.
            config (AlbertConfig): An object of the AlbertConfig class containing the configuration parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        """
        Retrieves the output embeddings from the AlbertForPreTraining model.

        Args:
            self (AlbertForPreTraining): The current instance of the AlbertForPreTraining class.

        Returns:
            nn.Linear: The output embeddings of the model.

        Raises:
            None.

        This method returns the output embeddings of the AlbertForPreTraining model. The output embeddings
        represent the encoded representation of the input sequence. The embeddings are obtained from the
        predictions decoder of the model.

        Example:
            ```python
            >>> model = AlbertForPreTraining()
            >>> embeddings = model.get_output_embeddings()
            ```
        """
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """
        Set the output embeddings for the AlbertForPreTraining model.

        Args:
            self (AlbertForPreTraining): The current instance of the AlbertForPreTraining model.
            new_embeddings (nn.Linear): The new embeddings to be set as the output embeddings for the model.
                It should be an instance of nn.Linear representing the new output embeddings.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of type nn.Linear.
        """
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Retrieve the input embeddings for the ALBERT model.

        Args:
            self: An instance of the AlbertForPreTraining class.

        Returns:
            nn.Embedding: An instance of the nn.Embedding class representing the input embeddings for the ALBERT model.

        Raises:
            None
        """
        return self.albert.embeddings.word_embeddings

    def forward(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        sentence_order_label: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            sentence_order_label (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
                (see `input_ids` docstring) Indices should be in `[0, 1]`. `0` indicates original order (sequence A, then
                sequence B), `1` indicates switched order (sequence B, then sequence A).

        Returns:
            Union[AlbertForPreTrainingOutput, Tuple]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, AlbertForPreTraining
            ...
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
            >>> model = AlbertForPreTraining.from_pretrained("albert-base-v2")
            ...
            >>> input_ids = mindspore.Tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)
            >>> # Batch size 1
            >>> outputs = model(input_ids)
            ...
            >>> prediction_logits = outputs.prediction_logits
            >>> sop_logits = outputs.sop_logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
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

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentence_order_label is not None:
            masked_lm_loss = F.cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_order_loss = F.cross_entropy(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertMLMHead(nn.Module):

    """
    AlbertMLMHead class represents the MLM (Masked Language Model) head for an ALBERT (A Lite BERT) model in a neural network.
    It includes methods for initializing the MLM head, forwarding the prediction scores, and tying the weights.

    This class inherits from the nn.Module class and implements the following methods:

    1. __init__(self, config: AlbertConfig):

        - Initializes the AlbertMLMHead with the provided AlbertConfig settings.
        - Initializes the LayerNorm, bias, dense, decoder, activation, and ties the weights.

    2. forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:

        - Constructs the prediction scores based on the input hidden_states tensor.
        - Applies the dense layer, activation function, LayerNorm, and decoder to generate the prediction scores.

    3. _tie_weights(self) -> None:

        - Ties the weights by setting the bias attribute equal to the decoder's bias.

    This class is designed to be used as part of an ALBERT model architecture for masked language modeling tasks.
    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertMLMHead class.

        Args:
            self: The instance of the class itself.
            config (AlbertConfig):
                An object of the AlbertConfig class containing the configuration settings for the model.

                - config.embedding_size (int): The size of the embedding.
                - config.layer_norm_eps (float): The epsilon value for layer normalization.
                - config.vocab_size (int): The size of the vocabulary.
                - config.hidden_size (int): The size of the hidden layer.
                - config.hidden_act (str): The activation function for the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()

        self.LayerNorm = nn.LayerNorm([config.embedding_size], eps=config.layer_norm_eps)
        self.bias = Parameter(ops.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]
        self.decoder.bias = self.bias

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the Albert Masked Language Model (MLM) head.

        Args:
            self (AlbertMLMHead): An instance of the AlbertMLMHead class.
            hidden_states (mindspore.Tensor): The input hidden states tensor to be processed. It represents the output
                of the previous layer and serves as input to the MLM head. It must be a tensor of shape compatible with
                the internal operations of the method.

        Returns:
            mindspore.Tensor: The prediction scores tensor generated by the MLM head. It represents the model's predictions
                for masked tokens based on the input hidden states.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores

    def _tie_weights(self) -> None:
        """
        This method ties the weights of the decoder bias to the main decoder weights.

        Args:
            self (AlbertMLMHead): The instance of the AlbertMLMHead class.

        Returns:
            None.

        Raises:
            None
        """
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class AlbertSOPHead(nn.Module):

    """
    This class represents the AlbertSOPHead, which is responsible for forwarding the sentence-order prediction (SOP) head in an ALBERT (A Lite BERT) model.

    The AlbertSOPHead class inherits from nn.Module and provides methods for initializing the SOP head and forwarding the logits for SOP classification.

    Attributes:
        config (AlbertConfig): The configuration object for the ALBERT model.

    Methods:
        __init__:
            Initializes the AlbertSOPHead instance.

        forward:
            Constructs the logits for SOP classification based on the pooled_output tensor.

    Example:
        ```python
        >>> from mindspore import nn
        >>> import mindspore.numpy as np
        >>> import mindspore.ops as ops
        ...
        >>> config = AlbertConfig()  # create the ALBERT configuration object
        >>> albert_sop_head = AlbertSOPHead(config)  # create an instance of AlbertSOPHead
        ...
        >>> pooled_output = np.random.randn(2, config.hidden_size)  # create a random pooled_output tensor
        >>> logits = albert_sop_head.forward(pooled_output)  # forward the logits for SOP classification
        ```
    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertSOPHead class.

        Args:
            self: The current instance of the class.
            config (AlbertConfig): The configuration object for the Albert model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()

        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the AlbertSOPHead by applying dropout and classifier operations on the provided pooled_output.

        Args:
            self (object): The instance of the AlbertSOPHead class.
            pooled_output (mindspore.Tensor): The pooled output tensor obtained from the previous layer. It serves as the input to the method.

        Returns:
            mindspore.Tensor:
                The output tensor (logits) obtained after applying the dropout and classifier operations on the pooled_output.
                This tensor represents the final result of the AlbertSOPHead forwardion process.

        Raises:
            None
        """
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class AlbertForMaskedLM(AlbertPreTrainedModel):

    """
    AlbertForMaskedLM is a class that represents an Albert model for Masked Language Modeling tasks.
    It inherits from AlbertPreTrainedModel and provides methods for setting and getting output embeddings, input
    embeddings, and for forwarding the model for masked language modeling.
    The class includes an initialization method that sets up the model with AlbertModel and AlbertMLMHead components, as well as methods for
    manipulating embeddings and forwarding the model for training or inference.
    The 'forward' method takes various input tensors and parameters for the model and returns the masked language modeling output
    including the loss and prediction scores. The class is designed to be used in natural language processing tasks where masked language modeling is required.
    """
    _tied_weights_keys = ["predictions.decoder.bias", "predictions.decoder.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the AlbertForMaskedLM class.

        Args:
            self: The current instance of the class.
            config (AlbertConfig): An instance of AlbertConfig containing the model configuration settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.albert = AlbertModel(config, add_pooling_layer=False)
        self.predictions = AlbertMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        """
        Retrieve the output embeddings from the AlbertForMaskedLM model.

        Args:
            self (AlbertForMaskedLM): The instance of the AlbertForMaskedLM class.
                This parameter is automatically passed when calling the method.
                It is used to access the model's predictions.decoder attribute.

        Returns:
            nn.Linear: The output embeddings of the model.
                These embeddings are used for generating predictions for masked tokens.

        Raises:
            None: This method does not raise any exceptions.
        """
        return self.predictions.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """
        Sets the output embeddings for the AlbertForMaskedLM model.

        Args:
            self (AlbertForMaskedLM): The instance of the AlbertForMaskedLM class.
            new_embeddings (nn.Linear): The new embeddings to be set for the output layer of the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.predictions.decoder = new_embeddings

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Retrieve the input embeddings for the AlbertForMaskedLM model.

        Args:
            self (AlbertForMaskedLM): An instance of the AlbertForMaskedLM class.

        Returns:
            nn.Embedding:
                The input embeddings used by the AlbertForMaskedLM model. These embeddings are of type nn.Embedding and
                represent the mapping of input tokens to their respective embeddings.

        Raises:
            None.

        Note:
            The input embeddings are obtained from the 'word_embeddings' attribute of the ALBERT model's 'embeddings' module.
        """
        return self.albert.embeddings.word_embeddings

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
    ) -> Union[MaskedLMOutput, Tuple]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            Union[MaskedLMOutput, Tuple]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, AlbertForMaskedLM
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
            >>> model = AlbertForMaskedLM.from_pretrained("albert-base-v2")
            ...
            >>> # add mask_token
            >>> inputs = tokenizer("The capital of [MASK] is Paris.", return_tensors="ms")
            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits
            ...
            >>> # retrieve index of [MASK]
            >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            >>> predicted_token_id = logits[0, mask_token_index].argmax(dim=-1)
            >>> tokenizer.decode(predicted_token_id)
            'france'
            ```

            ```python
            >>> labels = tokenizer("The capital of France is Paris.", return_tensors="ms")["input_ids"]
            >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
            >>> outputs = model(**inputs, labels=labels)
            >>> round(outputs.loss.item(), 2)
            0.81
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
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


class AlbertForSequenceClassification(AlbertPreTrainedModel):

    """
    This class represents an Albert model for sequence classification.
    It inherits from AlbertPreTrainedModel and includes methods for initializing the model and forwarding the sequence classification
    output.
    The model utilizes the Albert architecture for natural language processing tasks, such as text classification and regression.

    The __init__ method initializes the AlbertForSequenceClassification model with the provided AlbertConfig.
    It sets the number of labels, config, Albert model, dropout layer, and classifier for sequence classification.

    The forward method takes input tensors and optional arguments for sequence classification and returns the sequence classifier output.
    It also handles the computation of loss based on the problem type and labels provided.

    Note:
        This docstring is a high-level summary and is not meant to be executed as code.
    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes a new instance of the `AlbertForSequenceClassification` class.

        Args:
            self: The instance of the class.
            config (AlbertConfig): The configuration object for the model.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes a new instance of the `AlbertForSequenceClassification` class. It takes in two parameters: `self` and `config`. The `self` parameter represents the instance of the class itself.
            The `config` parameter is an object of the `AlbertConfig` class, which holds the configuration settings for the model.

            This method performs the following operations:

            1. Calls the `__init__` method of the base class to initialize the inherited attributes.
            2. Sets the `num_labels` attribute of the instance to the `num_labels` value from the `config` parameter.
            3. Sets the `config` attribute of the instance to the `config` parameter.
            4. Creates a new instance of the `AlbertModel` class, named `albert`, using the `config` parameter.
            5. Creates a new instance of the `nn.Dropout` class, named `dropout`, with the dropout probability specified in `config.classifier_dropout_prob`.
            6. Creates a new instance of the `nn.Linear` class, named `classifier`, with the input size of `config.hidden_size` and the output size of `config.num_labels`.
            7. Calls the `post_init` method to perform any additional initialization steps.

        Note:
            The `AlbertForSequenceClassification` class is typically used for sequence classification tasks using the ALBERT model.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

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
    ) -> Union[SequenceClassifierOutput, Tuple]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
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


class AlbertForTokenClassification(AlbertPreTrainedModel):

    """
    This class represents an Albert model for token classification, specifically designed for tasks
    like named entity recognition or part-of-speech tagging. It extends the AlbertPreTrainedModel class.

    Attributes:
        num_labels (int): The number of labels for the token classification task.
        albert (AlbertModel): The underlying AlbertModel instance for feature extraction.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Dense layer for classification.
        config (AlbertConfig): The configuration object for the model.

    Methods:
        __init__: Initializes the AlbertForTokenClassification instance.
        forward: Constructs the AlbertForTokenClassification model.

    Example:
        ```python
        >>> # Initialize the configuration object
        >>> config = AlbertConfig(num_labels=10, hidden_size=256, classifier_dropout_prob=0.1)
        ...
        >>> # Create an instance of AlbertForTokenClassification
        >>> model = AlbertForTokenClassification(config)
        ...
        >>> # Perform forward pass
        >>> outputs = model.forward(input_ids, attention_mask, labels=labels)
        ...
        >>> # Extract the logits
        >>> logits = outputs.logits
        ...
        >>> # Calculate the loss
        >>> loss = outputs.loss
        ```

    Note:
        The labels should be tensor of shape `(batch_size, sequence_length)` with indices in the range `[0, ..., num_labels - 1]`.
    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of the AlbertForTokenClassification class.

        Args:
            self: The instance of the class.
            config (AlbertConfig): The configuration for the Albert model.
                It contains various hyperparameters to customize the model.
                The config parameter should be an instance of AlbertConfig class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config, add_pooling_layer=False)
        classifier_dropout_prob = (
            config.classifier_dropout_prob
            if config.classifier_dropout_prob is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

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
    ) -> Union[TokenClassifierOutput, Tuple]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
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


class AlbertForQuestionAnswering(AlbertPreTrainedModel):

    """
    AlbertForQuestionAnswering represents a fine-tuned Albert model for question answering tasks.
    This class inherits from AlbertPreTrainedModel and includes functionality to handle question answering tasks
    by computing start and end logits for the labelled spans in the input sequence.

    Attributes:
        num_labels (int): Number of labels for the classification task.
        albert (AlbertModel): The Albert model used for question answering.
        qa_outputs (nn.Linear): A dense layer for computing logits for start and end positions.

    Methods:
        __init__: Initializes the AlbertForQuestionAnswering class with the provided configuration.
        forward:
            Constructs the Albert model for question answering and computes the loss for token classification based on start and end positions.
            Returns the total loss along with start and end logits if return_dict is False, otherwise returns a QuestionAnsweringModelOutput object.
    """
    def __init__(self, config: AlbertConfig):
        """
        Initializes an instance of AlbertForQuestionAnswering.

        Args:
            self: The instance of the class.
            config (AlbertConfig): An instance of AlbertConfig containing the configuration parameters for the model.
                It is used to set up the model architecture and initialize its components.
                The parameter 'config' should be of type AlbertConfig.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not of type AlbertConfig.
            ValueError: If the 'num_labels' attribute is not found in the 'config' parameter.
            AttributeError: If an attribute error occurs while initializing the model components.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config, add_pooling_layer=False)
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
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
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

        outputs = self.albert(
            input_ids=input_ids,
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

        logits: mindspore.Tensor = self.qa_outputs(sequence_output)
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


class AlbertForMultipleChoice(AlbertPreTrainedModel):

    """
    This class represents the Albert model for multiple choice classification tasks. It is a subclass of the AlbertPreTrainedModel.

    The AlbertForMultipleChoice class contains methods for model initialization and forwardion.
    It inherits the configuration from AlbertConfig and utilizes the AlbertModel for the underlying Albert architecture.

    Methods:
        __init__: Initializes the AlbertForMultipleChoice model with the given configuration.
        forward: Constructs the AlbertForMultipleChoice model with the given input tensors and returns the output.

    Attributes:
        albert: The underlying AlbertModel instance.
        dropout: Dropout layer for regularization.
        classifier: Dense layer for classification.
        config: The AlbertConfig instance used for model initialization.

    Note:
        The forward method follows the multiple choice classification setup and returns either the classification loss
        and logits or a tuple containing the loss, logits, hidden states, and attentions, depending on the return_dict parameter.

    Please refer to the AlbertConfig documentation for more details on the configuration options used by this class.
    """
    def __init__(self, config: AlbertConfig):
        """
        Initialize the AlbertForMultipleChoice model.

        Args:
            self: The object instance of the AlbertForMultipleChoice class.
            config (AlbertConfig): An instance of AlbertConfig class containing the configuration settings for the Albert model.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type AlbertConfig.
            ValueError: If the classifier_dropout_prob attribute in the config parameter is not within the valid range [0, 1].
        """
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(p=config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

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
    ) -> Union[AlbertForPreTrainingOutput, Tuple]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where *num_choices* is the size of the second dimension of the input tensors. (see
                *input_ids* above)
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
        outputs = self.albert(
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
        logits: mindspore.Tensor = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
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

__all__ = [
    "ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
    "AlbertForMaskedLM",
    "AlbertForMultipleChoice",
    "AlbertForPreTraining",
    "AlbertForQuestionAnswering",
    "AlbertForSequenceClassification",
    "AlbertForTokenClassification",
    "AlbertModel",
    "AlbertPreTrainedModel",
]
