# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BERT model specific for generation."""

import math
from typing import Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import nn, ops, Tensor, Parameter
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_bert_generation import BertGenerationConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/bert_for_seq_generation_L-24_bbc_encoder"
_CONFIG_FOR_DOC = "BertGenerationConfig"


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->BertGeneration
class BertGenerationSelfOutput(nn.Cell):

    """
    This class represents the self output layer for Bert generation. 
    It includes operations for dense transformation, layer normalization, 
    and dropout to process hidden states in the Bert model.
    
    Attributes:
        dense (nn.Dense): Dense layer for transforming the hidden states.
        LayerNorm (nn.LayerNorm): Layer normalization for normalizing the hidden states.
        dropout (nn.Dropout): Dropout layer for applying dropout regularization.

    Methods:
        construct:
            Applies dense transformation, dropout, and layer normalization to the hidden states and input tensor, 
            returning the processed hidden states.

    Note: 
        This class inherits from nn.Cell.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the BertGenerationSelfOutput class.

        Args:
            self (BertGenerationSelfOutput): The current instance of the class.
            config: 
                The configuration object for the BertGenerationSelfOutput.
                
                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.

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
        This method constructs the output of a BERT generation self-attention layer.

        Args:
            self (BertGenerationSelfOutput): The instance of the BertGenerationSelfOutput class.
            hidden_states (mindspore.Tensor): The tensor representing the hidden states of the BERT model.
                It is used as input to the dense and dropout layers for further processing.
            input_tensor (mindspore.Tensor): The tensor representing the input to the self-attention layer.
                It is added to the processed hidden_states after normalization.

        Returns:
            mindspore.Tensor: 
                Returns a tensor representing the constructed output of the BERT generation self-attention layer.

        Raises:
            ValueError: If the input_tensor and hidden_states have incompatible shapes for addition.
            RuntimeError: If an error occurs during the dense or dropout layer processing.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->BertGeneration
class BertGenerationSelfAttention(nn.Cell):

    """
    This class represents a self-attention mechanism for the generation of 
    BERT (Bidirectional Encoder Representations from Transformers) models. 
    It inherits from the nn.Cell class and provides methods for performing attention calculations.

    The class initializes with the given configuration and optional position embedding type, 
    and it validates the configuration parameters. 
    It also includes methods for reshaping input tensors and performing attention calculations.

    The construct method takes hidden_states as input and applies self-attention calculations. 
    It supports optional input tensors such as attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, 
    and past_key_value. 
    It also provides an option to output attention probabilities if needed.

    This class is designed to be used as part of BERT model generation and provides essential functionality 
    for self-attention mechanisms.

    """
    def __init__(self, config, position_embedding_type=None):
        '''
        Initializes the BertGenerationSelfAttention class.

        Args:
            self: The instance of the class.
            config: An instance of the configuration class containing the model configuration parameters.
            position_embedding_type (str, optional): The type of position embedding to use. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: 
                If the hidden size is not a multiple of the number of attention heads 
                and the config does not have an attribute 'embedding_size'.
        '''
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

    def swapaxes_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method 'swapaxes_for_scores' in the class 'BertGenerationSelfAttention' 
        swaps axes of a given tensor to prepare it for scoring calculations.

        Args:
            self: An instance of the class 'BertGenerationSelfAttention'. 
                It is used to access the attributes of the class.

            x: A tensor of type 'mindspore.Tensor' representing the input data. 
                It is expected to have a specific shape for further processing.

        Returns:
            A tensor of type 'mindspore.Tensor' with modified axes suitable for scoring calculations. 
                The shape of the tensor is adjusted to facilitate attention head calculations.

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
        '''
        Constructs the attention mechanism for the BertGenerationSelfAttention class.

        Args:
            self: The instance of the BertGenerationSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states to be used for attention computation.
            attention_mask (Optional[mindspore.Tensor]): 
                The attention mask tensor, indicating which tokens should be attended to and which should be ignored. 
                Default is None.
            head_mask (Optional[mindspore.Tensor]): 
                The mask tensor indicating which heads should be masked out. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): 
                The hidden states of the encoder, if cross-attention is being performed. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): 
                The attention mask for the encoder, if cross-attention is being performed. Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): 
                The cached key and value tensors from the previous attention computation, if available. Default is None.
            output_attentions (Optional[bool]): 
                Whether to include attention probabilities in the output. Default is False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the context layer tensor. 
                If output_attentions is True, the tuple also contains the attention probabilities tensor.

        Raises:
            None.
        '''
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
            # Apply the attention mask is (precomputed for all layers in BertGenerationModel forward() function)
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


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->BertGeneration
class BertGenerationAttention(nn.Cell):

    """
    This class represents the attention mechanism for generating 
    BERT (Bidirectional Encoder Representations from Transformers) output. 
    It inherits from the nn.Cell class.

    The BertGenerationAttention class initializes the attention mechanism with the given configuration 
    and position embedding type. 
    It includes methods for pruning attention heads and constructing the attention output 
    based on the input hidden states and optional masks or past key values.

    The prune_heads method prunes the specified attention heads from the attention mechanism, 
    updating the internal parameters accordingly.

    The construct method computes the attention output based on the input hidden states, optional attention mask, 
    head mask, encoder hidden states, encoder attention mask, past key value, and output attentions flag. 
    It returns a tuple containing the attention output and other optional outputs.

    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes a new instance of the BertGenerationAttention class.

        Args:
            config (object): An object containing the configuration settings for the attention mechanism.
            position_embedding_type (object, optional): An object specifying the type of position embedding to use.
                Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.self = BertGenerationSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertGenerationSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' is defined within the class 'BertGenerationAttention' 
        and is responsible for pruning the attention heads based on the provided list of 'heads'.

        Args:
            self (object): The instance of the class.
            heads (list): A list containing the indices of attention heads to be pruned. 
                It is expected that the list is non-empty, and each element within the list is an integer 
                representing the index of the attention head to be pruned.

        Returns:
            None: This method does not explicitly return a value, 
                as it operates by modifying the state of the 'BertGenerationAttention' instance.

        Raises:
            ValueError: If the provided 'heads' list is empty, a ValueError is raised to indicate 
                that the pruning operation cannot be performed without any specified attention heads to prune.
            TypeError: If the 'heads' parameter is not a list or if its elements are not integers, 
                a TypeError is raised to indicate incorrect input data type.

        Note: 
            The method operates by modifying the internal state of the 'BertGenerationAttention' instance, 
            including updating the attention heads, query, key, value, and other related attributes based on the
            specified 'heads' to be pruned.
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
            self (BertGenerationAttention): The instance of the BertGenerationAttention class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]): 
                An optional input tensor representing the attention mask. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): 
                An optional input tensor representing the head mask. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): 
                An optional input tensor representing the encoder hidden states. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): 
                An optional input tensor representing the encoder attention mask. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): 
                An optional tuple representing past key-value pairs. Defaults to None.
            output_attentions (Optional[bool]): 
                An optional boolean flag to indicate whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output and other optional outputs.

        Raises:
            None.
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


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->BertGeneration
class BertGenerationIntermediate(nn.Cell):

    """
    A class representing the intermediate layer in the Bert Generation model.

    This class inherits from the nn.Cell class and provides methods for initializing the layer 
    and constructing the intermediate layer of the model.

    Attributes:
        dense (mindspore.nn.Dense): The dense layer used in the intermediate layer.
        intermediate_act_fn (callable): The activation function used in the intermediate layer.

    Methods:
        __init__(self, config): Initializes the BertGenerationIntermediate layer with the given configuration.
        construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor: Constructs the intermediate layer of the model.

    """
    def __init__(self, config):
        """
        Initializes an instance of the BertGenerationIntermediate class.

        Args:
            self: The instance of the class.
            config: 
                An object containing configuration settings for the intermediate layer.
                
                - Type: Config
                - Purpose: Specifies the configuration parameters for the intermediate layer.
                - Restrictions: Must be a valid Config object.

        Returns:
            None

        Raises:
            TypeError: If the provided config object is not of type Config.
            KeyError: If the hidden activation function specified in the config is not found in ACT2FN dictionary.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the intermediate generation output for the BERT model.

        Args:
            self (BertGenerationIntermediate): An instance of the BertGenerationIntermediate class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.

        Returns:
            mindspore.Tensor: The generated intermediate output tensor.

        Raises:
            None.

        This method takes in the hidden states tensor and performs the following operations:

        1. Applies dense transformation to the hidden states using the self.dense layer.
        2. Applies the intermediate activation function (self.intermediate_act_fn) to the transformed hidden states.
        3. Returns the generated intermediate output tensor.

        Note:
            - The hidden_states tensor should have a shape compatible with the self.dense layer.
            - The returned tensor will have the same shape as the input hidden_states tensor.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->BertGeneration
class BertGenerationOutput(nn.Cell):

    """
    The 'BertGenerationOutput' class represents a neural network cell for generating output in a BERT model.
    This class inherits from nn.Cell and includes methods for initializing the cell and constructing the output generation process.
    The 'construct' method takes hidden_states and input_tensor as input and returns the generated output tensor.

    The '__init__' method initializes the cell with the given configuration, 
    including setting up the dense layer, layer normalization, and dropout.

    Attributes:
        dense: A dense layer for transforming the hidden states to the hidden size specified in the configuration.
        LayerNorm: A layer normalization module to normalize the hidden states.
        dropout: A dropout module for applying dropout to the hidden states.

    Methods:
        __init__(self, config): Initializes the 'BertGenerationOutput' cell with the given configuration.
        construct:
          Constructs the output generation process using the dense layer, dropout, and layer normalization.

    Note:
        This class is designed for use within a BERT model architecture 
        and is intended to be used as part of a neural network.
    """
    def __init__(self, config):
        """
        Initializes a BertGenerationOutput instance.

        Args:
            self (object): The instance of the BertGenerationOutput class.
            config (object): 
                An object containing configuration parameters for the model.
                
                - Type: Custom class
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must include necessary parameters for model initialization.

        Returns:
            None.

        Raises:
            ValueError: If the provided configuration is invalid or missing required parameters.
            TypeError: If the input parameters are of incorrect types.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the BertGenerationOutput by applying a series of operations on the given hidden_states and input_tensor.

        Args:
            self (BertGenerationOutput): The current instance of the BertGenerationOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor to be processed. 
                It should have a shape of (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden states. 
                It should have the same shape as hidden_states.

        Returns:
            mindspore.Tensor: The resulting tensor after applying the operations on the hidden_states and input_tensor. 
                It has the same shape as hidden_states.

        Raises:
            None: This method does not raise any exceptions.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->BertGeneration
class BertGenerationLayer(nn.Cell):

    """
    The BertGenerationLayer class represents a layer in a BERT (Bidirectional Encoder Representations from Transformers) 
    model specifically designed for generation tasks. 
    This layer includes self-attention mechanisms and feed-forward neural networks. 
    It can be used as part of a decoder model with the option to add cross-attention layers 
    for interacting with encoder hidden states.

    Attributes:
        chunk_size_feed_forward: The chunk size used to process feed-forward operations.
        seq_len_dim: The dimension representing the sequence length.
        attention: An instance of BertGenerationAttention for handling self-attention.
        is_decoder: A flag indicating whether the layer is used as a decoder model.
        add_cross_attention: A flag indicating whether cross-attention is added.
        crossattention: An optional BertGenerationAttention instance for cross-attention.
        intermediate: An instance of BertGenerationIntermediate for intermediate processing.
        output: An instance of BertGenerationOutput for final output generation.

    Methods:
        construct(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions):
            Constructs the layer by processing the input hidden states and optional additional information. 
            Handles self-attention and cross-attention if configured as a decoder with cross-attention enabled.

        feed_forward_chunk(attention_output):
            Performs the feed-forward chunk processing on the given attention output, generating the final layer output.

    Note:
        - The layer enforces specific configurations and behaviors based on 
        whether it is used as a decoder model and if cross-attention is enabled.
        - Proper instantiation and configuration of the layer are essential for correct functionality and model compatibility.
    """
    def __init__(self, config):
        """
        Initializes an instance of the BertGenerationLayer class.

        Args:
            self (object): The instance of the BertGenerationLayer class.
            config (object): 
                The configuration object containing parameters for the layer initialization.
                
                - chunk_size_feed_forward (int): The chunk size for feed-forward operations.
                - is_decoder (bool): Indicates whether the layer is used as a decoder model.
                - add_cross_attention (bool): Determines if cross-attention is added to the layer.

        Returns:
            None.

        Raises:
            ValueError: If add_cross_attention is True but the layer is not used as a decoder model, 
                a ValueError is raised with a message indicating that the layer should be used as a decoder model if cross
                attention is added.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertGenerationAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertGenerationAttention(config, position_embedding_type="absolute")
        self.intermediate = BertGenerationIntermediate(config)
        self.output = BertGenerationOutput(config)

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
        This method constructs the BertGenerationLayer by performing self-attention and potentially cross-attention operations.

        Args:
            self: The instance of the BertGenerationLayer class.
            hidden_states (mindspore.Tensor): The input hidden states to be processed by the attention mechanisms.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor masking the attention scores. Default is None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor masking the attention heads. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): Encoder hidden states for cross-attention. Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention scores. Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key and value tensors for attention mechanisms. Default is None.
            output_attentions (Optional[bool]): Flag indicating whether to output attention weights. Default is False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the computed layer output tensor 
                and potentially additional outputs based on the decoder status.

        Raises:
            ValueError: If 'encoder_hidden_states' are provided but cross-attention layers are not instantiated.
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
        This method 'feed_forward_chunk' is a part of the class 'BertGenerationLayer'
        and is responsible for performing a feed-forward operation on the input attention output.

        Args:
            self (object): The instance of the class 'BertGenerationLayer'. 
                It is used to access the attributes and methods of the class.
            attention_output (tensor): The input tensor representing the attention output. 
                It is the output of the attention mechanism and serves as the input for the feed-forward operation.

        Returns:
            None: 
                This method does not return any value but modifies the input attention output 
                through the feed-forward process.

        Raises:
            None:
                However, potential exceptions could arise from the 'output' method or other internal operations
                called within this method.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->BertGeneration
class BertEncoder(nn.Cell):

    """
    The BertEncoder class represents a transformer encoder for BERT (Bidirectional Encoder Representations from Transformers). 
    It inherits from the nn.Cell class.

    The BertEncoder class initializes with the provided configuration and constructs the layers for 
    the transformer encoder based on the BERT model architecture. 
    It also provides methods for processing input data through the encoder layers to generate output representations.

    The construct method processes the input hidden states through the encoder layers, optionally using attention masks, 
    head masks, and past key values. 
    It supports gradient checkpointing and caching of intermediate results to optimize memory usage during training. 
    The method returns the final hidden state, past key values, hidden states at all layers, self-attentions, 
    and cross-attentions based on the specified output settings.

    For detailed usage instructions and additional information, 
    refer to the specific method and attribute documentation within the class implementation.
    """
    def __init__(self, config):
        """Initialize the BertEncoder class.

        Args:
            self: The instance of the class.
            config: 
                The configuration parameters for the BertEncoder.
                
                - Type: object
                - Purpose: Specifies the configuration settings for the BertEncoder.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([BertGenerationLayer(config) for _ in range(config.num_hidden_layers)])
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
        This method constructs the BertEncoder with the given input parameters 
        and returns the output along with optional hidden states and attentions.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states.
            attention_mask (Optional[mindspore.Tensor]): 
                Masking tensor indicating which elements in the input should be attended to.
            head_mask (Optional[mindspore.Tensor]): Masking tensor for attention heads.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states from the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]): Masking tensor for encoder attention.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Past key values for caching.
            use_cache (Optional[bool]): Flag indicating whether to use caching.
            output_attentions (Optional[bool]): Flag indicating whether to output attentions.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
            return_dict (Optional[bool]): Flag indicating whether to return the output as a dictionary.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                Depending on the 'return_dict' flag, returns either a tuple containing hidden states, 
                next decoder cache, all hidden states, self attentions, and cross attentions, or a
                BaseModelOutputWithPastAndCrossAttentions object containing the last hidden state, 
                past key values, hidden states, attentions, and cross attentions.

        Raises:
            None
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


class BertGenerationEmbeddings(nn.Cell):
    """Construct the embeddings from word and position embeddings."""
    def __init__(self, config):
        """
        Initializes the BertGenerationEmbeddings class.

        Args:
            self (BertGenerationEmbeddings): The object instance of the BertGenerationEmbeddings class.
            config (object): 
                An object containing configuration parameters.
                
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - pad_token_id (int): The index of the padding token.
                - max_position_embeddings (int): The maximum number of positional embeddings.
                - layer_norm_eps (float): The epsilon value for LayerNorm.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.

        Returns:
            None: 
                This method initializes the following attributes:
                
                - word_embeddings (nn.Embedding): An embedding layer for word embeddings.
                - position_embeddings (nn.Embedding): An embedding layer for positional embeddings.
                - LayerNorm (nn.LayerNorm): A layer normalization module.
                - dropout (nn.Dropout): A dropout module for hidden layers.
                - position_ids (Tensor): A tensor containing position indices.

        Raises:
            AttributeError: If an attribute is missing in the config object.
            ValueError: If vocab_size or hidden_size is not provided in the config.
            TypeError: If config is not of type object.
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

    def construct(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        """
        Args:
            self (BertGenerationEmbeddings): The instance of the BertGenerationEmbeddings class.
            input_ids (Tensor, optional): The input tensor containing token indices. Default is None.
            position_ids (Tensor, optional): The tensor containing positional indices. Default is None.
            inputs_embeds (Tensor, optional): The tensor containing precomputed embeddings. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            None.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None.
            ValueError: If input_ids is not None and inputs_embeds is not None.
            IndexError: If the shape of input_ids or inputs_embeds is invalid.
            IndexError: If the sequence length is invalid.
            IndexError: If the position_ids shape is invalid.
            IndexError: If the embeddings shape is invalid.
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
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertGenerationPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = BertGenerationConfig
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


class BertGenerationEncoder(BertGenerationPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    This model should be used when leveraging Bert or Roberta checkpoints for the [`EncoderDecoderModel`] class as
    described in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461)
    by Sascha Rothe, Shashi Narayan, and Aliaksei Severyn.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    def __init__(self, config):
        """
        Initializes a BertGenerationEncoder instance.

        Args:
            self (BertGenerationEncoder): The instance of the BertGenerationEncoder class being initialized.
            config (dict): A dictionary containing configuration parameters for the BertGenerationEncoder.
                This dictionary must include the necessary settings for the embeddings and encoder components.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.config = config

        self.embeddings = BertGenerationEmbeddings(config)
        self.encoder = BertEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the BertGenerationEncoder.

        Args:
            self: The instance of the BertGenerationEncoder class.

        Returns:
            word_embeddings: This method returns the word embeddings for input.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the BertGenerationEncoder class.

        Args:
            self (BertGenerationEncoder): An instance of the BertGenerationEncoder class.
            value: The input embeddings to be set. This should be of type `torch.Tensor`.

        Returns:
            None.

        Raises:
            None.

        This method sets the value of the `word_embeddings` attribute of the `embeddings` object
        within the `BertGenerationEncoder` instance to the provided input embeddings.
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
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            encoder_hidden_states  (`mindspore.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`: `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers`
                with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
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
        elif input_ids is not None:
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

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = None
        if not use_cache:
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

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=sequence_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertGenerationOnlyLMHead(nn.Cell):

    """
    This class represents the LM head for generating outputs in a BERT-based language model.
    It is used to generate tokens based on hidden states.

    Attributes:
        decoder (nn.Dense): A fully connected layer for generating logits based on hidden states.
        bias (Parameter): The bias parameter used in the generation process.

    Methods:
        construct(hidden_states): Generates logits based on the input hidden states.
        _tie_weights(): Ties the bias parameter to the decoder's bias for weight sharing.
    """
    def __init__(self, config):
        """
        Initializes the BertGenerationOnlyLMHead class.

        Args:
            self: An instance of the BertGenerationOnlyLMHead class.
            config: A configuration object containing the required parameters for initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size)
        self.bias = Parameter(ops.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        Method to construct logits for generation in BertGenerationOnlyLMHead.

        Args:
            self (BertGenerationOnlyLMHead): The instance of the BertGenerationOnlyLMHead class.
            hidden_states (Tensor): The hidden states obtained from the encoder.
                These hidden states are used as input to the decoder for generating logits.

        Returns:
            None: This method returns the logits generated by the decoder based on the hidden states.

        Raises:
            None.
        """
        logits = self.decoder(hidden_states)
        return logits

    def _tie_weights(self):
        """
        Method to tie the weights of the decoder bias in the BertGenerationOnlyLMHead class.

        Args:
            self (BertGenerationOnlyLMHead): The instance of the BertGenerationOnlyLMHead class.
                This parameter refers to the current instance of the BertGenerationOnlyLMHead class.

        Returns:
            None: This method modifies the bias attribute of the decoder in-place.

        Raises:
            None.
        """
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class BertGenerationDecoder(BertGenerationPreTrainedModel):

    """
    This class represents a decoder model for BERT generation.
    It extends the BertGenerationPreTrainedModel and provides methods for initializing the model,
    constructing the model outputs, preparing inputs for generation, and reordering cache.
    The class includes methods for initializing the model, retrieving and setting output embeddings,
    constructing the model outputs, preparing inputs for generation, and reordering cache.
    The detailed docstrings for each method provide information about the parameters, return types, and usage examples.
    This class is designed to be used as part of the BERT generation framework
    and provides essential functionality for decoding and generating outputs based on input sequences.
    """
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        """
        Initializes a new instance of the BertGenerationDecoder class.

        Args:
            self: The instance of the class.
            config (object): The configuration object containing settings for the decoder.
                This object must have the necessary attributes and properties required for configuring the decoder.
                It should also have an attribute 'is_decoder' to indicate if the decoder is being used as
                a standalone component.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`")

        self.bert = BertGenerationEncoder(config)
        self.lm_head = BertGenerationOnlyLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        This method returns the output embeddings of the BertGenerationDecoder.

        Args:
            self: The object instance of the BertGenerationDecoder class.

        Returns:
            None: This method returns the output embeddings of the decoder in the BertGenerationDecoder class.

        Raises:
            None
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Method to set new output embeddings for the decoder in BertGenerationDecoder.

        Args:
            self (BertGenerationDecoder): The instance of BertGenerationDecoder to which the new embeddings will be set.
            new_embeddings: The new embeddings to set for the decoder. Should be of type compatible with the decoder.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head.decoder = new_embeddings

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
                Mask to avoid performing attention on the padding token indices of the encoder input.
                This mask is used in the cross-attention if the model is configured as a decoder.
                Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors of shape
                `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
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
            >>> from transformers import AutoTokenizer, BertGenerationDecoder, BertGenerationConfig
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            >>> config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            >>> config.is_decoder = True
            >>> model = BertGenerationDecoder.from_pretrained(
            ...     "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
            ... )
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_token_type_ids=False, return_tensors="pt")
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
            lm_loss = ops.cross_entropy(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
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
        Method: prepare_inputs_for_generation

        This method prepares inputs for generation in the BertGenerationDecoder class.

        Args:
            self (object): The instance of the BertGenerationDecoder class.
            input_ids (torch.Tensor): The input tensor containing token IDs. Shape should be (batch_size, sequence_length).
            past_key_values (tuple, optional): Tuple containing past key values from previous generations. Default is None.
            attention_mask (torch.Tensor, optional): The attention mask tensor.
                If not provided, a tensor of ones with the same shape as input_ids is created.

        Returns:
            dict: A dictionary containing the prepared inputs for generation including
                'input_ids', 'attention_mask', and 'past_key_values'.

        Raises:
            ValueError: If the provided input_ids shape is invalid.
            IndexError: If there is an issue with past_key_values.
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
        Reorders the cache for the BertGenerationDecoder.
        
        Args:
            self: An instance of the BertGenerationDecoder class.
            past_key_values (tuple): A tuple containing the past key-value states for each layer.
                Each layer's past key-value state is a tensor of shape (batch_size, sequence_length, hidden_size).
            beam_idx (tensor): The beam indices used for reordering the past key-value states.
                A tensor of shape (batch_size, beam_size).
        
        Returns:
            None: This method modifies the cache in-place.
        
        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past

__all__ =  [
    "BertGenerationDecoder",
    "BertGenerationEncoder",
    "BertGenerationPreTrainedModel",
]
