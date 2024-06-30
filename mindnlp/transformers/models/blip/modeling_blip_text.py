# coding=utf-8
# Copyright 2022 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the BSD-3-clause license (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MindSpore BLIP Text model."""

import math
from typing import List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ....utils import logging
from .configuration_blip import BlipTextConfig


logger = logging.get_logger(__name__)


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L52
class BlipTextEmbeddings(nn.Cell):
    """Construct the embeddings from word and position embeddings."""
    def __init__(self, config):
        """
        Initializes a BlipTextEmbeddings instance.
        
        Args:
            self (BlipTextEmbeddings): The current instance of the BlipTextEmbeddings class.
            config (object):
                An object containing configuration settings. It must have the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - pad_token_id (int): The ID of the padding token.
                - max_position_embeddings (int): The maximum position for position embeddings.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.
                - position_embedding_type (str, optional): The type of position embedding. Defaults to 'absolute'.

        Returns:
            None.

        Raises:
            AttributeError: If the config object is missing any of the required attributes.
            ValueError: If any of the configuration settings are invalid or out of range.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = ops.arange(config.max_position_embeddings).expand((1, -1))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> mindspore.Tensor:
        """
        Constructs the BlipTextEmbeddings.

        Args:
            self (BlipTextEmbeddings): The instance of the BlipTextEmbeddings class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the indices of tokens in the vocabulary.
            position_ids (Optional[mindspore.Tensor]): The input tensor containing the position indices of tokens.
            inputs_embeds (Optional[mindspore.Tensor]): The input tensor containing the embeddings of tokens.
            past_key_values_length (int): The length of past key values.

        Returns:
            mindspore.Tensor: The tensor containing the constructed embeddings.

        Raises:
            ValueError: If input_ids is not None and its shape is invalid.
            ValueError: If the position embedding type is invalid.
            ValueError: If the shape of inputs_embeds is invalid.
            ValueError: If the length of past_key_values_length is negative.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L97
class BlipTextSelfAttention(nn.Cell):

    """
    A class representing self-attention mechanism for BlipText models.

    This class inherits from nn.Cell and implements the self-attention mechanism for BlipText models.
    It includes methods for saving and retrieving attention gradients and maps,
    as well as methods for constructing self-attention scores and context layers.

    Attributes:
        config (object): The configuration object for the self-attention mechanism.
        is_cross_attention (bool): A flag indicating whether cross-attention is being used.

    Methods:
        save_attn_gradients(attn_gradients): Saves the attention gradients.
        get_attn_gradients(): Retrieves the saved attention gradients.
        save_attention_map(attention_map): Saves the attention map.
        get_attention_map(): Retrieves the saved attention map.
        swapaxes_for_scores(x): Swaps axes for the input scores.
        construct(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
            past_key_value, output_attentions):
            Constructs the self-attention scores and context layers.

    Raises:
        ValueError: If the hidden size is not a multiple of the number of attention heads.

    """
    def __init__(self, config, is_cross_attention):
        """
        Initialize the BlipTextSelfAttention class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings.

                - Type: object
                - Purpose: Configuration settings for the self-attention mechanism.
                - Restrictions: None
            is_cross_attention (bool):
                A flag indicating if cross-attention is enabled.

                - Type: bool
                - Purpose: Specifies whether the attention mechanism is cross-attention or not.
                - Restrictions: Must be a boolean value.

        Returns:
            None.

        Raises:
            ValueError: Raised if the hidden size is not a multiple of the number of attention heads.
        """
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Dense(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Dense(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Dense(config.hidden_size, self.all_head_size)
            self.value = nn.Dense(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def save_attn_gradients(self, attn_gradients):
        """
        Save the attention gradients in the BlipTextSelfAttention class.

        Args:
            self (BlipTextSelfAttention): The instance of the BlipTextSelfAttention class.
                The class object where the attention gradients will be saved.
            attn_gradients (any): The attention gradients to be saved.
                This parameter represents the gradients associated with the attention mechanism.

        Returns:
            None.

        Raises:
            None.
        """
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        """
        Returns the attention gradients of the BlipTextSelfAttention layer.

        Args:
            self (BlipTextSelfAttention): The instance of BlipTextSelfAttention class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        """
        Save the attention map in the BlipTextSelfAttention class.

        Args:
            self (BlipTextSelfAttention): The instance of the BlipTextSelfAttention class.
            attention_map (object): The attention map to be saved in the class.

        Returns:
            None.

        Raises:
            None.
        """
        self.attention_map = attention_map

    def get_attention_map(self):
        """
        This method returns the attention map for BlipTextSelfAttention.

        Args:
            self: The instance of the BlipTextSelfAttention class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.attention_map

    def swapaxes_for_scores(self, x):
        """
        Performs a swap axes operation on the given input tensor to prepare it for self-attention scoring in the
        BlipTextSelfAttention class.

        Args:
            self (BlipTextSelfAttention): An instance of the BlipTextSelfAttention class.
            x (torch.Tensor): The input tensor to be reshaped and permuted.
                It should have shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: The reshaped and permuted tensor.
                It has shape (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None.

        Note:
            - The 'self' parameter is automatically passed when calling the method on an instance of the class.
            - The 'x' parameter represents the input tensor that will be modified.
            - The reshaping and permuting operations are done to prepare the tensor for subsequent self-attention scoring operations.
            - The shape of the input tensor is modified to (batch_size, num_attention_heads, sequence_length, attention_head_size).
            - The function returns None if successful.
        """
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
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
        Constructs the BlipTextSelfAttention.

        This method applies the BlipTextSelfAttention mechanism to the given hidden states and returns the
        context layer and optionally the attention probabilities.

        Args:
            self (BlipTextSelfAttention): The instance of the BlipTextSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of shape (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): Optional tensor of shape (batch_size, seq_length)
                containing attention mask values.
            head_mask (Optional[mindspore.Tensor]): Optional tensor of shape (num_heads,) or (num_layers, num_heads)
                containing head mask values.
            encoder_hidden_states (Optional[mindspore.Tensor]): Optional tensor of shape (batch_size, seq_length, hidden_size)
                representing the hidden states from an encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Optional tensor of shape (batch_size, seq_length)
                containing attention mask values for the encoder hidden states.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Optional past key-value pairs to be used
                for incremental decoding.
            output_attentions (Optional[bool]): If True, returns attention probabilities along with the context layer.

        Returns:
            Tuple[mindspore.Tensor]: The context layer tensor of shape (batch_size, seq_length, hidden_size)
                and optionally attention probabilities.

        Raises:
            None.
        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
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

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

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

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BlipTextModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = ops.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert -> BlipText
class BlipTextSelfOutput(nn.Cell):

    """
    This class represents the self-output layer of the BlipText model in the MindSpore library.

    The BlipTextSelfOutput class is a subclass of the nn.Cell class, and it is responsible for applying various
    transformations to the hidden states of the BlipText model's self-attention layer. It performs
    dense linear transformation, dropout, layer normalization, and residual connection to produce the final hidden states.

    Attributes:
        dense (nn.Dense): The dense linear transformation layer that projects the hidden states to a higher-dimensional space.
        LayerNorm (nn.LayerNorm): The layer normalization layer that applies normalization to the hidden states.
        dropout (nn.Dropout): The dropout layer that applies dropout regularization to the hidden states.

    Methods:
        construct:
            Applies the transformations to the hidden states and returns the final hidden states.

            Args:

            - hidden_states (mindspore.Tensor): The input hidden states.
            - input_tensor (mindspore.Tensor): The tensor representing the output of the previous layer.

            Returns:

            - mindspore.Tensor: The final hidden states after applying the transformations.

    Note:
        - The hidden states are projected to a higher-dimensional space using the dense layer.
        - Dropout regularization is applied to the hidden states using the dropout layer.
        - Layer normalization is applied to the sum of the transformed hidden states and the input tensor using the LayerNorm layer.
        - The final hidden states are obtained by adding the layer-normalized hidden states to the input tensor.

    Example:
        ```python
        >>> # Create an instance of the BlipTextSelfOutput class
        >>> self_output = BlipTextSelfOutput(config)
        ...
        >>> # Apply the transformations to the hidden states
        >>> final_hidden_states = self_output.construct(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """Initialize the BlipTextSelfOutput class.

        Args:
            self (BlipTextSelfOutput): An instance of the BlipTextSelfOutput class.
            config: The configuration object that contains the parameters for initializing the BlipTextSelfOutput.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the self-attention output of the BlipText model.

        Args:
            self (BlipTextSelfOutput): The instance of the BlipTextSelfOutput class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
                It should have a shape of (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor representing the self-attention output from the previous layer.
                It should have the same shape as the hidden_states.

        Returns:
            mindspore.Tensor: The tensor representing the self-attention output of the BlipText model.
                It has the same shape as the hidden_states.

        Raises:
            None: This method does not raise any exceptions.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#242
class BlipTextAttention(nn.Cell):

    """
    A class that represents a self-attention mechanism for BlipText.

    This class inherits from nn.Cell and implements methods for initializing, pruning heads, and constructing self-attention.

    Attributes:
        self (BlipTextSelfAttention): An instance of BlipTextSelfAttention for self-attention mechanism.
        output (BlipTextSelfOutput): An instance of BlipTextSelfOutput for self-attention output.
        pruned_heads (set): A set containing the indices of pruned attention heads.

    Methods:
        __init__: Initializes the BlipTextAttention instance with the given configuration and cross-attention flag.
        prune_heads: Prunes the specified attention heads from the self-attention mechanism.
        construct: Constructs the self-attention mechanism with the given inputs and optional parameters,
            and returns the attention output.

    """
    def __init__(self, config, is_cross_attention=False):
        """
        Initializes a new instance of the BlipTextAttention class.

        Args:
            self: The instance of the class.
            config: A configuration object containing settings and parameters for the attention mechanism.
                It is of type 'config'.
            is_cross_attention: A boolean flag indicating whether cross-attention is enabled. It is of type 'bool'.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = BlipTextSelfAttention(config, is_cross_attention)
        self.output = BlipTextSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes specified attention heads from the BlipTextAttention layer.

        Args:
            self (BlipTextAttention): The current instance of the BlipTextAttention class.
            heads (list): A list of integers representing the attention heads to be pruned.

        Returns:
            None

        Raises:
            None

        Description:
            This method prunes the specified attention heads from the BlipTextAttention layer.
            The attention heads are identified by their indices provided in the 'heads' parameter.
            If the 'heads' parameter is an empty list, the method returns without performing any pruning.

            The 'heads' parameter is a list of integers which represent the attention heads to be pruned.
            Each integer in the list should be a valid attention head index within the BlipTextAttention layer.
            If an invalid attention head index is provided, it will be ignored.

            The method updates the 'query', 'key', 'value', and 'output.dense' attributes of the BlipTextAttention
            instance by pruning the corresponding linear layers. The pruning is performed using the 'index' obtained
            from the 'find_pruneable_heads_and_indices' function.

            Additionally, the method updates the 'num_attention_heads' attribute of the BlipTextAttention instance
            by subtracting the number of pruned heads from the current number of attention heads. The 'all_head_size'
            attribute is also updated accordingly.

            The 'pruned_heads' attribute of the BlipTextAttention instance is updated by adding the pruned attention
            heads to the existing set of pruned heads.

            Note that the 'prune_linear_layer' function is used internally to perform the actual pruning of the
            linear layers.

        Example:
            ```python
            >>> blip_attention = BlipTextAttention()
            >>> heads_to_prune = [2, 4, 6]
            >>> blip_attention.prune_heads(heads_to_prune)
            ...
            >>> # The attention heads with indices 2, 4, and 6 will be pruned from the BlipTextAttention layer.
            >>> # The corresponding linear layers will be pruned, and the 'num_attention_heads' and 'all_head_size' attributes will be updated accordingly.
            ```
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
        This method constructs the attention mechanism for the BlipTextAttention class.

        Args:
            self: The instance of the BlipTextAttention class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask.
            head_mask (Optional[mindspore.Tensor]): An optional tensor representing the head mask.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor representing the encoder hidden states.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the encoder attention mask.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key-value states.
            output_attentions (Optional[bool]): A flag indicating whether to output attentions.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor.

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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert -> BlipText
class BlipTextIntermediate(nn.Cell):

    """
    This class represents a BlipTextIntermediate module that is used for intermediate processing of text data.
    It is a subclass of nn.Cell and provides functionality for constructing hidden states.

    Attributes:
        dense (nn.Dense): A dense layer that applies linear transformation to the input hidden states.
        intermediate_act_fn (function): The activation function applied to the intermediate hidden states.

    Methods:
        __init__: Initializes the BlipTextIntermediate module.
        construct: Constructs the intermediate hidden states.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BlipTextIntermediate class.

        Args:
            self: The instance of the class being initialized.
            config:
                An object containing the configuration parameters for the BlipTextIntermediate instance.

                - Type: Any
                - Purpose: Specifies the configuration settings for the BlipTextIntermediate instance.
                - Restrictions: None

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

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the BlipTextIntermediate.

        This method takes in 'hidden_states' and constructs the BlipTextIntermediate using the following steps:

        1. Applies a dense layer to the 'hidden_states'.
        2. Applies the intermediate activation function to the result of the dense layer.

        Args:
            self: An instance of the BlipTextIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states. A tensor of shape [batch_size, sequence_length, hidden_size].

        Returns:
            mindspore.Tensor: The constructed BlipTextIntermediate. A tensor of shape [batch_size, sequence_length, hidden_size].

        Raises:
            TypeError: If 'hidden_states' is not a tensor.
            ValueError: If the shape of 'hidden_states' is not [batch_size, sequence_length, hidden_size].

        Note:
            - The 'hidden_states' tensor represents the input hidden states, which are typically the output of a previous layer.
            - The 'hidden_size' refers to the dimensionality of the hidden states.
            - The intermediate activation function is applied element-wise to the hidden states.

        Example:
            ```python
            >>> hidden_states = mindspore.Tensor([[1, 2, 3], [4, 5, 6]], dtype=mindspore.float32)
            >>> blip_text = BlipTextIntermediate()
            >>> output = blip_text.construct(hidden_states)
            ```
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert -> BlipText
class BlipTextOutput(nn.Cell):

    """
    This class represents a BlipTextOutput module that is used in neural network models for natural language processing tasks. It is a subclass of nn.Cell and is responsible for constructing the output of the
    Blip model.

    Attributes:
        dense (nn.Dense): A fully connected layer that maps the input tensor to an intermediate size.
        LayerNorm (nn.LayerNorm): A layer normalization module that normalizes the hidden states.
        dropout (nn.Dropout): A dropout module that applies dropout regularization to the hidden states.

    Methods:
        __init__: Initializes the BlipTextOutput module with the given configuration.
        construct: Constructs the output tensor by applying dense layer, dropout, layer normalization, and adding the input tensor to the hidden states.

    """
    def __init__(self, config):
        """
        Initializes a BlipTextOutput instance.

        Args:
            self (BlipTextOutput): The current instance of the BlipTextOutput class.
            config: An object containing configuration parameters for the BlipTextOutput instance.
                It should have attributes 'intermediate_size', 'hidden_size', 'layer_norm_eps', and 'hidden_dropout_prob'.

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            NotImplementedError: If any required attributes are missing in the 'config' object.
            ValueError: If any of the attribute values in the 'config' object are invalid or out of range.
            TypeError: If the 'config' parameter is not of the expected type.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a BlipTextOutput by applying a series of operations on the given hidden states and input tensor.

        Args:
            self (BlipTextOutput): The instance of the BlipTextOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor to be processed.
                It should have the shape (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden states tensor.
                It should have the same shape as the hidden_states tensor.

        Returns:
            mindspore.Tensor: The processed hidden states tensor after applying the operations.
                It will have the same shape as the hidden_states and input_tensor.

        Raises:
            None: This method does not raise any exceptions.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BlipTextLayer(nn.Cell):

    """
    The BlipTextLayer class represents a layer of a neural network model for text processing.
    It is designed to handle self-attention and cross-attention mechanisms, and it can be used as part of a transformer
    architecture. This class inherits from nn.Cell and contains methods for initialization and constructing the layer.

    Attributes:
        config: A configuration object for the layer.
        chunk_size_feed_forward: The chunk size for feed-forward operations.
        seq_len_dim: The dimension for sequence length.
        attention: An instance of the BlipTextAttention class for handling self-attention.
        layer_num: The number of the layer in the model.
        crossattention: An instance of the BlipTextAttention class for handling cross-attention (if the layer is part of a decoder).
        intermediate: An instance of the BlipTextIntermediate class for intermediate processing.
        output: An instance of the BlipTextOutput class for producing the final output.

    Methods:
        __init__: Initializes the BlipTextLayer with the given configuration and layer number.
        construct: Constructs the layer using the provided inputs and optional arguments.
        feed_forward_chunk: Performs a feed-forward operation on the given attention output.

    The BlipTextLayer class is a fundamental component for building transformer-based models for text processing tasks,
    providing the necessary functionality for attention mechanisms and intermediate processing.
    """
    def __init__(self, config, layer_num):
        """
        Initializes a BlipTextLayer object.

        Args:
            self (BlipTextLayer): The instance of the BlipTextLayer class.
            config (object): The configuration object containing the settings for the BlipTextLayer.
            layer_num (int): The layer number for the BlipTextLayer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BlipTextAttention(config)
        self.layer_num = layer_num
        if self.config.is_decoder:
            self.crossattention = BlipTextAttention(config, is_cross_attention=self.config.is_decoder)
        self.intermediate = BlipTextIntermediate(config)
        self.output = BlipTextOutput(config)

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
        Constructs the BlipTextLayer.

        Args:
            self (BlipTextLayer): The instance of the BlipTextLayer class.
            hidden_states (mindspore.Tensor): The input hidden states of the layer.
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor, indicating which positions should be attended to and which should not.
                Defaults to None.
            head_mask (Optional[mindspore.Tensor]):
                The head mask tensor, indicating which heads should be masked out and which should not. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder layer. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the encoder layer. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key value tensor. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: The output tensor of the BlipTextLayer.

        Raises:
            None: This method does not raise any exceptions.
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

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        This method 'feed_forward_chunk' is a part of the 'BlipTextLayer' class and is used to perform
        the feed-forward chunk operation.

        Args:
            self (BlipTextLayer): The instance of the BlipTextLayer class.
            attention_output (tensor): The input tensor representing the attention output.

        Returns:
            layer_output (tensor): The output tensor obtained after the feed-forward chunk operation.

        Raises:
            None
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L386
class BlipTextEncoder(nn.Cell):

    """
    This class represents a BlipTextEncoder module that is used for encoding text data. It inherits from the nn.Cell class.

    Attributes:
        config: The configuration object for the BlipTextEncoder module.
        layer: A list of BlipTextLayer instances, representing the hidden layers of the encoder.
        gradient_checkpointing: A boolean indicating whether gradient checkpointing is enabled.

    Methods:
        __init__: Initializes a new instance of the BlipTextEncoder class.
        construct: Constructs

    the BlipTextEncoder module and performs the encoding of the input text.

    Detailed Description:
        The BlipTextEncoder module is responsible for encoding text data using a series of hidden layers.
        It takes in an input tensor representing the hidden states of the text, along with optional attention
        masks, head masks, encoder hidden states, encoder attention masks, past key values, and other parameters.
        The module applies the specified number of hidden layers to the input tensor, performing attention and
        other operations as necessary. The output of the module includes the encoded hidden states,
        as well as optional outputs such as attentions and cross-attentions.

        The BlipTextEncoder class contains an initialization method (__init__) that takes a configuration object as input.
        This configuration object is used to set up the module's parameters, such as the number of
        hidden layers. The hidden layers are stored in the 'layer' attribute as a list of BlipTextLayer instances.
        The 'gradient_checkpointing' attribute is a boolean flag that indicates whether gradient checkpointing
        is enabled for this module.

        The main method of the BlipTextEncoder class is the 'construct' method. This method takes the input
        hidden states tensor, along with optional arguments such as attention masks, head masks, and past key
        values. It performs the encoding of the input text by applying the hidden layers sequentially to the input tensor.
        The method returns the encoded hidden states, as well as optional outputs such as attentions
        and cross-attentions. The method also allows for customization of the output format through the use of
        boolean flags such as 'output_attentions' and 'output_hidden_states'.

    Note:
        - The BlipTextEncoder module supports gradient checkpointing, which can be enabled
        by setting the 'gradient_checkpointing' attribute to True.
        - The BlipTextEncoder module is designed to be used within a larger model architecture
        for natural language processing tasks.
    """
    def __init__(self, config):
        """
        Initializes a BlipTextEncoder object.

        Args:
            self (BlipTextEncoder): The instance of the BlipTextEncoder class.
            config (dict): A dictionary containing configuration parameters for the encoder.
                This dict must include the following keys:

                - num_hidden_layers (int): The number of hidden layers to be created. Must be a positive integer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type dict.
            ValueError: If the config dictionary is missing the 'num_hidden_layers' key or if it has an invalid value.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BlipTextLayer(config, i) for i in range(config.num_hidden_layers)])
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
        """
        Constructs the BlipTextEncoder.

        Args:
            self (BlipTextEncoder): The instance of the BlipTextEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states of the encoder. Shape: (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor indicating which tokens should be attended to.
                Defaults to None. Shape: (batch_size, seq_length).
            head_mask (Optional[mindspore.Tensor]): The head mask tensor indicating which heads should be masked.
                Defaults to None. Shape: (num_attention_heads).
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder.
                Defaults to None. Shape: (batch_size, seq_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for the encoder.
                Defaults to None. Shape: (batch_size, seq_length).
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The tuple of past key-value tensors for each layer of the decoder.
                Defaults to None. Shape: (num_hidden_layers, 2, batch_size, num_attention_heads, seq_length, hidden_size // num_attention_heads).
            use_cache (Optional[bool]): Whether to use cache for faster decoding. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Defaults to False.
            return_dict (Optional[bool]): Whether to return a dictionary as the output. Defaults to True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
            The output of the BlipTextEncoder.
            If return_dict is False, returns a tuple of tensors containing the hidden states, next decoder cache, all hidden states,
            self attentions, and cross attentions, if applicable.
            If return_dict is True, returns a BaseModelOutputWithPastAndCrossAttentions object containing the same tensors.

        Raises:
            None.
        """
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None

        next_decoder_cache = () if use_cache else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
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


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->BlipText
class BlipTextPooler(nn.Cell):

    """
    The BlipTextPooler class represents a text pooler for Blip model. It inherits from the nn.Cell class.

    The class's code includes an __init__ method that initializes the BlipTextPooler object with the provided configuration.
    It also contains a construct method that takes hidden_states as input and returns
    the pooled output after applying dense and activation operations.

    Attributes:
        dense (nn.Dense): A fully connected layer with the hidden size specified in the configuration.
        activation (nn.Tanh): A hyperbolic tangent activation function.

    Methods:
        __init__: Initializes the BlipTextPooler object with the given configuration.
        construct: Constructs the pooled output from the hidden_states tensor.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BlipTextPooler class.

        Args:
            self (object): The instance of the BlipTextPooler class.
            config (object): An object containing configuration parameters for the BlipTextPooler.
                The config parameter should have a property 'hidden_size' that specifies the size of the hidden layer.
                It is expected to be an integer greater than 0.

        Returns:
            None: This method initializes the BlipTextPooler object with the provided configuration.

        Raises:
            TypeError: If the config parameter is not provided or is not of the expected type.
            ValueError: If the hidden_size property in the config parameter is not a valid integer.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the pooled output tensor for the BlipTextPooler class.

        This method takes in two parameters: self and hidden_states.

        Args:
            self: An instance of the BlipTextPooler class.
            hidden_states (mindspore.Tensor): A tensor containing hidden states. It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: A tensor representing the pooled output. It has a shape of (batch_size, hidden_size).

        Raises:
            None.

        Note:
            - The method extracts the first token tensor from the hidden states tensor.
            - The first token tensor is then passed through a dense layer.
            - The output of the dense layer is then passed through an activation function.
            - The resulting tensor is returned as the pooled output.

        Example:
            ```python
            >>> pooler = BlipTextPooler()
            >>> hidden_states = mindspore.Tensor([[1, 2, 3], [4, 5, 6]], dtype=mindspore.float32)
            >>> output = pooler.construct(hidden_states)
            ```
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->BlipText
class BlipTextPredictionHeadTransform(nn.Cell):

    """
    This class represents a transformation module used in a BlipText prediction head.
    It inherits from the nn.Cell class and is responsible for transforming the input hidden states.

    Attributes:
        dense (nn.Dense): A fully connected layer that transforms the input hidden states.
        transform_act_fn (callable): An activation function used to transform the hidden states.
        LayerNorm (nn.LayerNorm): A layer normalization module that normalizes the hidden states.

    Methods:
        __init__: Initializes the BlipTextPredictionHeadTransform instance.
        construct: Applies the transformation to the input hidden states.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BlipTextPredictionHeadTransform class.

        Args:
            self: The object itself.
            config:
                An instance of the configuration class containing the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - hidden_act (str or function): The activation function for the hidden layer.

                    - If it is a string, it represents the name of the activation function.
                    - If it is a function, it is the custom activation function itself.

                - layer_norm_eps (float): The epsilon value for layer normalization.

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
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method 'construct' is a part of the class 'BlipTextPredictionHeadTransform' and is used to perform
        transformations on the input hidden states tensor.

        Args:
            self (BlipTextPredictionHeadTransform): The instance of the BlipTextPredictionHeadTransform class.
            hidden_states (mindspore.Tensor): A tensor containing hidden states to be processed.
                It should have a shape compatible with the operations within the method.

        Returns:
            mindspore.Tensor: A tensor representing the processed hidden states after passing through the dense layer,
                activation function, and LayerNorm operation.

        Raises:
            None:
                However, potential exceptions may be raised during the execution of the operations within the method,
                such as shape mismatch errors or exceptions from underlying operations.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->BlipText
class BlipTextLMPredictionHead(nn.Cell):

    """
    This class represents the prediction head for a BlipText language model. It is a subclass of nn.Cell.

    Attributes:
        transform (BlipTextPredictionHeadTransform): An instance of BlipTextPredictionHeadTransform class that
            performs transformation on the input hidden states.
        decoder (nn.Dense): A fully connected layer that maps the transformed hidden states to the vocabulary size.
        bias (Parameter): A learnable parameter representing the bias for the decoder layer.

    Methods:
        __init__:
            Initializes a new instance of BlipTextLMPredictionHead.

            Args:

            - config: An instance of BlipTextLMPredictionHeadConfig containing the configuration settings.

        construct:
            Constructs the prediction head.

            Args:

            - hidden_states: The input hidden states.

            Returns:

            - The output hidden states after transformation and applying the decoder layer.
    """
    def __init__(self, config):
        """
        Initializes the BlipTextLMPredictionHead.

        Args:
            self (BlipTextLMPredictionHead): The instance of the BlipTextLMPredictionHead class.
            config: The configuration object containing settings for the prediction head.
                It is expected to be an instance of a configuration class.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.transform = BlipTextPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        This method constructs the prediction head for the BlipTextLMPredictionHead class.

        Args:
            self (BlipTextLMPredictionHead): The instance of the BlipTextLMPredictionHead class.
            hidden_states (tensor): The input hidden states to be processed for prediction.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->BlipText
class BlipTextOnlyMLMHead(nn.Cell):

    """A class representing the BlipTextOnlyMLMHead.

    BlipTextOnlyMLMHead is a subclass of nn.Cell that is used for multi-label classification tasks in natural language processing.
    It is specifically designed for predicting masked tokens in a given sequence.

    Attributes:
        predictions (BlipTextLMPredictionHead): An instance of the BlipTextLMPredictionHead class that performs
        the actual prediction of masked tokens.

    Methods:
        __init__: Initializes a new instance of the BlipTextOnlyMLMHead class.
        construct: Constructs the prediction scores for masked tokens based on the given sequence output.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BlipTextOnlyMLMHead class.

        Args:
            self (BlipTextOnlyMLMHead): The current instance of the BlipTextOnlyMLMHead class.
            config: A configuration object that contains the necessary parameters for initializing the
                BlipTextOnlyMLMHead object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = BlipTextLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        Method 'construct' in the class 'BlipTextOnlyMLMHead'.

        Args:
            self (BlipTextOnlyMLMHead): The instance of the BlipTextOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The tensor containing the sequence output data for processing.

        Returns:
            mindspore.Tensor: Returns the prediction scores generated by the 'predictions' method.

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L548
class BlipTextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BlipTextConfig
    base_model_prefix = "bert"

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, (nn.Dense, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
        if isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

        if isinstance(cell, nn.Dense) and cell.bias is not None:
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


# Adapted from https://github.com/salesforce/BLIP/blob/3a29b7410476bf5f2ba0955827390eb6ea1f4f9d/models/med.py#L571
class BlipTextModel(BlipTextPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. argument and `is_decoder` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a BlipTextModel object.

        Args:
            self: The object instance itself.
            config (dict): A dictionary containing configuration parameters for the BlipTextModel.
            add_pooling_layer (bool): A flag indicating whether to add a pooling layer to the model. Default is True.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or not of type dict.
            ValueError: If the config dictionary is missing required keys or has invalid values.
            RuntimeError: If an issue occurs during the initialization process.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BlipTextEmbeddings(config)
        self.encoder = BlipTextEncoder(config)
        self.pooler = BlipTextPooler(config) if add_pooling_layer else None

        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the input embeddings from the BlipTextModel.

        Args:
            self (BlipTextModel): The instance of the BlipTextModel class.

        Returns:
            None: This method returns the input embeddings from the BlipTextModel.
                The input embeddings are retrieved from the word_embeddings attribute of the embeddings.

        Raises:
            None
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BlipTextModel.

        Args:
            self (BlipTextModel): The instance of the BlipTextModel class.
            value (torch.Tensor): The input embeddings to be set for the BlipTextModel.
                It should be a tensor of shape (vocab_size, embedding_dim).

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

    def get_extended_attention_mask(
        self, attention_mask: mindspore.Tensor, input_shape: Tuple[int], is_decoder: bool
    ) -> mindspore.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`mindspore.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `mindspore.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = ops.arange(seq_length)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = ops.cat(
                        [
                            ops.ones(
                                (batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        is_decoder: Optional[bool] = False,
    ) -> Union[Tuple[mindspore.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        Args:
            encoder_hidden_states  (`mindspore.Tensor`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*):
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

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        elif encoder_embeds is not None:
            input_shape = encoder_embeds.shape[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones(((batch_size, seq_length + past_key_values_length)))

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: mindspore.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, is_decoder
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].shape
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

            if isinstance(encoder_attention_mask, list):
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = ops.ones(encoder_hidden_shape)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

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


# Adapted from https://github.com/salesforce/BLIP/blob/main/models/med.py#L811
class BlipTextLMHeadModel(BlipTextPreTrainedModel):

    """
    BlipTextLMHeadModel

    This class represents a language model head for the BlipText model. It inherits from the BlipTextPreTrainedModel class.

    Attributes:
        bert: A BlipTextModel instance representing the base BlipText model.
        cls: A BlipTextOnlyMLMHead instance representing the classification head for masked language modeling.
        label_smoothing: A float value representing the label smoothing factor.

    Methods:
        __init__(self, config): Initializes the BlipTextLMHeadModel instance.
        get_output_embeddings(self): Returns the decoder layer of the predictions.
        set_output_embeddings(self, new_embeddings): Sets the decoder layer of the predictions to new_embeddings.
        construct(self, input_ids, attention_mask, position_ids, head_mask, inputs_embeds, encoder_hidden_states,
            encoder_attention_mask, labels, past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, return_logits, is_decoder, reduction):
            Constructs the BlipTextLMHeadModel.
        prepare_inputs_for_generation(self, input_ids, past_key_values, attention_mask, **model_kwargs): Prepares the inputs for generation.
        _reorder_cache(self, past_key_values, beam_idx): Reorders the past key values for generation.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the BlipTextLMHeadModel class.

        Args:
            self: The object instance.
            config: A configuration object containing the model's settings and hyperparameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.bert = BlipTextModel(config, add_pooling_layer=False)
        self.cls = BlipTextOnlyMLMHead(config)
        self.label_smoothing = config.label_smoothing

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the BlipTextLMHeadModel.

        Args:
            self (BlipTextLMHeadModel): The instance of the BlipTextLMHeadModel class.

        Returns:
            None

        Raises:
            None

        This method retrieves the output embeddings from the BlipTextLMHeadModel.
        The output embeddings are used for generating predictions in the decoder module of the model.

        Note:
            The output embeddings represent the learned representations of the input data after being processed
            by the model's encoder.
            These embeddings can be used for downstream tasks such as classification or further analysis.

        Example:
            ```python
            >>> model = BlipTextLMHeadModel()
            >>> output_embeddings = model.get_output_embeddings()
            ```
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the BlipTextLMHeadModel.

        Args:
            self (BlipTextLMHeadModel): The instance of the BlipTextLMHeadModel class.
                This parameter is automatically passed and refers to the current instance of the class.
            new_embeddings (object): The new embeddings to set as the output embeddings.
                This parameter should be of any valid object type that can be assigned as new output embeddings.

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
        return_logits: Optional[bool] = False,
        is_decoder: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ) -> Union[Tuple[mindspore.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        encoder_hidden_states (`mindspore.Tensor`, *optional*): Sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is
            configured as a decoder.
        encoder_attention_mask (`mindspore.Tensor`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`mindspore.Tensor`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(mindspore.Tensor))`, *optional*):
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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
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
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :]

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            labels = labels[:, 1:]
            lm_loss = ops.cross_entropy(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1),
                                        reduction=reduction, label_smoothing=self.label_smoothing)
            if reduction == "none":
                lm_loss = lm_loss.view(prediction_scores.shape[0], -1).sum(1)

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
        This method prepares inputs for text generation in the BlipTextLMHeadModel class.

        Args:
            self (object): The instance of the BlipTextLMHeadModel class.
            input_ids (torch.Tensor): The input tensor containing tokenized input sequences.
            past_key_values (tuple, optional): A tuple of past key values for autoregressive generation. Default is None.
            attention_mask (torch.Tensor, optional): A tensor specifying which tokens should be attended to. Default is None.

        Returns:
            dict:
                A dictionary containing the following keys:

                - 'input_ids' (torch.Tensor): The processed input tensor for generation.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                - 'past_key_values' (tuple): The past key values for autoregressive generation.
                - 'encoder_hidden_states' (torch.Tensor, optional): Encoder hidden states if provided in model_kwargs.
                - 'encoder_attention_mask' (torch.Tensor, optional): Encoder attention mask if provided in model_kwargs.
                - 'is_decoder' (bool): A flag indicating this is a decoder operation.
        
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
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache based on the beam index.
        
        Args:
            self (BlipTextLMHeadModel): The instance of the BlipTextLMHeadModel class.
            past_key_values (tuple): The tuple of past keys and values.
            beam_idx (int): The index of the beam.
        
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
