# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# limitations under the License.\
# ============================================================================
"""
MindNlp LUKE model
"""
import inspect
import math
from typing import Callable, Optional, Tuple

import mindspore
import numpy as np
from mindnlp.core import nn, ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import Normal, initializer

from ...modeling_utils import PreTrainedModel
from .luke_config import LukeConfig
from ...activations import ACT2FN


class LukeEmbeddings(nn.Module):
    """
    LukeEmbeddings
    """
    def __init__(self, config: LukeConfig):
        """
        Initializes an instance of the LukeEmbeddings class.
        
        Args:
            self: The instance of the class itself.
            config (LukeConfig):
                An object of the LukeConfig class containing configuration parameters.

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden state.
                - pad_token_id (int): The index of the padding token in the vocabulary.
                - max_position_embeddings (int): The maximum number of positions for positional embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layers.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def construct(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        """
        Args:
            self (LukeEmbeddings): The instance of the LukeEmbeddings class.
            input_ids (Tensor, optional): A 2-D tensor containing the input token IDs. Defaults to None.
            token_type_ids (Tensor, optional): A 2-D tensor containing the token type IDs. Defaults to None.
            position_ids (Tensor, optional): A 2-D tensor containing the position IDs. Defaults to None.
            inputs_embeds (Tensor, optional): A 3-D tensor containing the input embeddings. Defaults to None.

        Returns:
            None.

        Raises:
            ValueError: If both input_ids and inputs_embeds are None.
            ValueError: If input_ids and inputs_embeds have mismatched shapes.
            TypeError: If the data type of token_type_ids is not int64.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = Tensor(np.zeros(input_shape), dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        #     print

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        input_shape = inputs_embeds.shape()[:-1]
        sequence_length = input_shape[1]

        position_ids = mindspore.numpy.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mindspore.int64
        )
        return ops.broadcast_to(position_ids.unsqueeze(0), input_shape)


class LukeEntityEmbeddings(nn.Module):
    """
    LukeEntityEmbeddings
    """
    def __init__(self, config: LukeConfig):
        """
        Initializes the LukeEntityEmbeddings class.

        Args:
            self: The instance of the class.
            config (LukeConfig): An instance of LukeConfig containing the configuration parameters for the entity embeddings.
                It specifies the entity vocabulary size, entity embedding size, hidden size, maximum position embeddings,
                type vocabulary size, and layer normalization epsilon.
                It is used to configure the entity embeddings, position embeddings, token type embeddings,
                layer normalization, and dropout.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Linear(config.entity_emb_size, config.hidden_size, bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm([config.hidden_size, ], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(
            self, entity_ids, position_ids, token_type_ids=None
    ):
        """
        This method constructs entity embeddings by combining entity, position, and token type embeddings.

        Args:
            self: The instance of the LukeEntityEmbeddings class.
            entity_ids (Tensor): A tensor containing the entity IDs for which embeddings need to be constructed.
            position_ids (Tensor): A tensor containing the position IDs representing the position of each entity.
            token_type_ids (Tensor, optional): A tensor containing the token type IDs. Defaults to None.
                If not provided, it is initialized as zeros_like(entity_ids).

        Returns:
            embeddings (Tensor): The combined embeddings of entities, positions,
                and token types after normalization and dropout.

        Raises:
            ValueError: If the dimensions of entity_embeddings and hidden_size do not match.
            TypeError: If entity_ids, position_ids, or token_type_ids are not of type Tensor.
            ValueError: If the position_ids contain values less than -1.
            RuntimeError: If any runtime error occurs during the computation process.
        """
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(ops.clamp(position_ids, min=0))
        position_embedding_mask = ops.cast(position_ids != -1, position_embeddings.dtype).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = position_embeddings.sum(axis=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(axis=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeSelfAttention(nn.Module):
    """
    LukeSelfAttention
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LukeSelfAttention class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration parameters for the LukeSelfAttention model.
                It should have the following attributes:

                - hidden_size (int): The hidden size of the model.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int, optional): The embedding size. (default: None)
                - use_entity_aware_attention (bool): Whether to use entity-aware attention or not.

        Returns:
            None

        Raises:
            ValueError: If the hidden size is not a multiple of the number of attention heads and the
                config object doesn't have the 'embedding_size' attribute.

        Note:
            The hidden size must be divisible by the number of attention heads.
            If it is not, and the config object doesn't have the 'embedding_size' attribute, a ValueError is raised.
            The 'query', 'key', and 'value' parameters are dense layers used for attention computation.
            If 'use_entity_aware_attention' is True, additional dense layers ('w2e_query', 'e2w_query', and 'e2e_query')
            are used for entity-aware attention.
            The 'dropout' parameter is a dropout layer used for attention probabilities dropout.

        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.use_entity_aware_attention = config.use_entity_aware_attention

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.use_entity_aware_attention:
            self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, input_x):
        """
        transpose_for_scores
        """
        new_input_x_shape = input_x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        input_x = input_x.view(*new_input_x_shape)
        return input_x.permute(0, 2, 1, 3)

    def construct(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        '''
        Constructs the self-attention mechanism for the LukeSelfAttention class.

        Args:
            self (LukeSelfAttention): An instance of the LukeSelfAttention class.
            word_hidden_states (Tensor): The hidden states of the word input sequence.
                Shape: (batch_size, sequence_length, hidden_size).
            entity_hidden_states (Tensor): The hidden states of the entity input sequence.
                Shape: (batch_size, entity_length, hidden_size).
            attention_mask (Tensor, optional): An optional mask tensor indicating which positions should be attended to
                and which should be ignored. Shape: (batch_size, sequence_length, sequence_length) or
                (batch_size, 1, 1, sequence_length).
            head_mask (Tensor, optional): An optional mask tensor indicating which heads should be masked out of the
                attention calculation. Shape: (num_attention_heads, sequence_length, sequence_length) or
                (batch_size, num_attention_heads, sequence_length, sequence_length).
            output_attentions (bool, optional): Whether to include attention probabilities in the output.
                Defaults to False.

        Returns:
            Tuple[Tensor or None, Tensor or None, Tensor or None]: 
                A tuple containing the output word hidden states, output entity hidden states, and 
                attention probabilities (optional).
                
                - output_word_hidden_states (Tensor or None): The output hidden states of the word input sequence. 
                Shape: (batch_size, sequence_length, hidden_size).
                - output_entity_hidden_states (Tensor or None): The output hidden states of the entity input sequence. 
                Shape: (batch_size, entity_length, hidden_size).
                - attention_probs (Tensor or None): The attention probabilities. Only included if output_attentions 
                is set to True. Shape: (batch_size, num_attention_heads, sequence_length, sequence_length).

        Raises:
            ValueError: If the shape of word_hidden_states and entity_hidden_states are incompatible.
            ValueError: If the shape of attention_mask is invalid.
            ValueError: If the shape of head_mask is invalid.
        '''
        word_size = word_hidden_states.shape[1]

        if entity_hidden_states is None:
            concat_hidden_states = word_hidden_states
        else:
            concat_hidden_states = ops.cat((word_hidden_states, entity_hidden_states), axis=1)

        key_layer = self.transpose_for_scores(self.key(concat_hidden_states))
        value_layer = self.transpose_for_scores(self.value(concat_hidden_states))

        if self.use_entity_aware_attention and entity_hidden_states is not None:
            # compute query vectors using word-word (w2w), word-entity (w2e), entity-word (e2w), entity-entity (e2e)
            # query layers
            w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

            # compute w2w, w2e, e2w, and e2e key vectors used with the query vectors computed above
            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]

            # compute attention scores based on the dot product between the query and key vectors
            w2w_attention_scores = ops.matmul(w2w_query_layer, w2w_key_layer.swapaxes(-1, -2))
            w2e_attention_scores = ops.matmul(w2e_query_layer, w2e_key_layer.swapaxes(-1, -2))
            e2w_attention_scores = ops.matmul(e2w_query_layer, e2w_key_layer.swapaxes(-1, -2))
            e2e_attention_scores = ops.matmul(e2e_query_layer, e2e_key_layer.swapaxes(-1, -2))

            # combine attention scores to create the final attention score matrix
            word_attention_scores = ops.cat([w2w_attention_scores, w2e_attention_scores], axis=3)
            entity_attention_scores = ops.cat([e2w_attention_scores, e2e_attention_scores], axis=3)
            attention_scores = ops.cat([word_attention_scores, entity_attention_scores], axis=2)

        else:
            query_layer = self.transpose_for_scores(self.query(concat_hidden_states))
            attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in LukeModel forward() function)
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
        context_layer = context_layer.view(*new_context_layer_shape)

        output_word_hidden_states = context_layer[:, :word_size, :]
        if entity_hidden_states is None:
            output_entity_hidden_states = None
        else:
            output_entity_hidden_states = context_layer[:, word_size:, :]

        if output_attentions:
            outputs = (output_word_hidden_states, output_entity_hidden_states, attention_probs)
        else:
            outputs = (output_word_hidden_states, output_entity_hidden_states)

        return outputs


class LukeSelfOutput(nn.Module):
    """
    LukeSelfOutput
    """
    def __init__(self, config):
        """
        Initializes an instance of the LukeSelfOutput class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration parameters.

                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """
        Constructs the output of the self-attention layer in the Luke model.

        Args:
            self: The instance of the LukeSelfOutput class.
            hidden_states (Tensor): The hidden states of the self-attention layer.
                Shape: (batch_size, sequence_length, hidden_size).
            input_tensor (Tensor): The input tensor to be added to the output of the layer normalization.
                Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: The output tensor of the self-attention layer.
                Shape: (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class LukeAttention(nn.Module):
    """
    LukeAttention
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LukeAttention class.

        Args:
            self (LukeAttention): The current instance of the LukeAttention class.
            config: The configuration object for the attention mechanism.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = LukeSelfAttention(config)
        self.output = LukeSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        NotImplementedError
        """
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    def construct(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        """
        Constructs the attention mechanism in the LukeAttention class.

        Args:
            self (LukeAttention): The instance of the LukeAttention class.
            word_hidden_states (tensor): The hidden states of words. Shape: (batch_size, word_seq_len, hidden_size).
            entity_hidden_states (tensor): The hidden states of entities. Shape: (batch_size, entity_seq_len, hidden_size).
            attention_mask (tensor, optional): Mask to avoid performing attention on padding tokens.
                Shape: (batch_size, 1, word_seq_len, entity_seq_len).
            head_mask (tensor, optional): Mask to exclude certain attention heads. Shape: (num_attention_heads,).
            output_attentions (bool): Whether to output attentions. Default is False.

        Returns:
            tuple: A tuple containing word_attention_output and entity_attention_output
                if entity_hidden_states is not None, else None.

                - word_attention_output (tensor): The attention output for word hidden states.
                Shape: (batch_size, word_seq_len, hidden_size).
                - entity_attention_output (tensor or None): The attention output for entity hidden states
                if entity_hidden_states is not None, else None.
                - additional outputs: Additional outputs returned by the attention mechanism.

        Raises:
            ValueError: If the shapes of word_hidden_states and entity_hidden_states are incompatible.
            RuntimeError: If an error occurs during the attention computation.
            IndexError: If the attention indices are out of bounds.
        """
        word_size = word_hidden_states.shape[1]
        self_outputs = self.self(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_hidden_states
        else:
            concat_self_outputs = ops.cat(self_outputs[:2], axis=1)
            concat_hidden_states = ops.cat([word_hidden_states, entity_hidden_states], axis=1)

        attention_output = self.output(concat_self_outputs, concat_hidden_states)

        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]

        # add attentions if we output them
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]

        return outputs


class LukeIntermediate(nn.Module):
    """
    LukeIntermediate
    """
    def __init__(self, config):
        """
        Initializes an instance of the LukeIntermediate class.

        Args:
            self: The instance of the LukeIntermediate class.
            config:
                A configuration object that contains parameters for initializing the instance.

                - Type: object
                - Purpose: Specifies the configuration settings for the instance.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If the config parameter is provided but is not in the correct format.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: Tensor) -> Tensor:
        """
        Constructs the intermediate hidden states in the LukeIntermediate class.

        Args:
            self: The instance of the LukeIntermediate class.
            hidden_states (Tensor): The input hidden states.

        Returns:
            Tensor: The intermediate hidden states after applying the dense layer and intermediate activation function.

        Raises:
            None.

        This method takes in the instance of the LukeIntermediate class and the input hidden states.
        It applies a dense layer to the hidden states and then applies the intermediate activation function.
        The resulting intermediate hidden states are returned as a Tensor.

        No exceptions are raised by this method.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LukeOutput(nn.Module):
    """
    LukeOutput
    """
    def __init__(self, config):
        """
        Initializes an instance of the LukeOutput class.

        Args:
            self (object): The instance of the LukeOutput class.
            config (object): An object containing configuration parameters for the LukeOutput instance.
                The config object is expected to have the following attributes:

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for hidden layers.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If any of the required attributes in the config object are missing or have invalid values.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        """
        Constructs the output tensor for the LukeOutput class.

        Args:
            self: An instance of the LukeOutput class.
            hidden_states (Tensor): The hidden states tensor.
                This tensor represents the intermediate hidden states of the model.
                It should have a shape of (batch_size, sequence_length, hidden_size).
            input_tensor (Tensor): The input tensor.
                This tensor represents the input to the layer.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: The constructed output tensor.
                This tensor is obtained by applying the dense layer, dropout, and layer normalization
                to the hidden states tensor and adding it to the input tensor.
                The returned tensor has the same shape as the input tensor.

        Raises:
            None.

        Note:
            The 'construct' method is responsible for transforming the hidden states tensor using the dense layer,
            applying dropout for regularization, and adding the transformed tensor to the input tensor.
            The resulting tensor represents the final output of the LukeOutput layer.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class LukeLayer(nn.Module):
    """
    LukeOutput
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LukeLayer class.

        Args:
            self: The object itself.
            config:
                An instance of the configuration class containing the following attributes:

                - chunk_size_feed_forward (int): The size of chunks to feed forward through the layer.
                - seq_len_dim (int): The dimension of the sequence length.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LukeAttention(config)
        self.intermediate = LukeIntermediate(config)
        self.output = LukeOutput(config)

    def construct(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        """
        Constructs the LukeLayer.

        Args:
            self (LukeLayer): The instance of the LukeLayer class.
            word_hidden_states (Tensor): The hidden states of the word inputs.
                It has shape [batch_size, seq_length, hidden_size].
            entity_hidden_states (Tensor): The hidden states of the entity inputs.
                It has shape [batch_size, seq_length, hidden_size].
            attention_mask (Tensor, optional): The attention mask to avoid performing attention on padding tokens.
                It has shape [batch_size, seq_length]. Defaults to None.
            head_mask (Tensor, optional): The mask to nullify selected heads of the self-attention modules.
                It has shape [num_heads, seq_length, seq_length]. Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to False.

        Returns:
            Tuple[Tensor, Tensor, Tuple]:
                A tuple containing:

                - word_layer_output (Tensor): The layer output for word inputs.
                    It has shape [batch_size, word_size, hidden_size].
                - entity_layer_output (Tensor): The layer output for entity inputs.
                    It has shape [batch_size, entity_size, hidden_size].
                - outputs (Tuple): Additional outputs from the attention layer.

        Raises:
            None.
        """
        word_size = word_hidden_states.shape[1]

        self_attention_outputs = self.attention(
            word_hidden_states,
            entity_hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        if entity_hidden_states is None:
            concat_attention_output = self_attention_outputs[0]
        else:
            concat_attention_output = ops.cat(self_attention_outputs[:2], axis=1)

        outputs = self_attention_outputs[2:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, concat_attention_output
        )
        word_layer_output = layer_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_layer_output = None
        else:
            entity_layer_output = layer_output[:, word_size:, :]

        outputs = (word_layer_output, entity_layer_output) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        This function applies transformations to an input tensor
        using two other layers  to produce an output tensor.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LukeEncoder(nn.Module):
    """
    LukeEncoder
    """
    def __init__(self, config):
        """Initialize a LukeEncoder object.

        Args:
            self (LukeEncoder): The LukeEncoder instance.
            config (dict): A dictionary containing configuration parameters for the encoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        """
        This method constructs the hidden states and attentions for a LukeEncoder model.

        Args:
            self: The instance of the LukeEncoder class.
            word_hidden_states: The hidden states of words, of shape (batch_size, sequence_length, hidden_size).
            entity_hidden_states: The hidden states of entities, of shape (batch_size, num_entities, hidden_size).
            attention_mask: An optional tensor of shape (batch_size, sequence_length) containing attention mask values.
            head_mask: An optional tensor of shape (num_layers, num_attention_heads) providing a mask for attention heads.
            output_attentions: A boolean flag indicating whether to output attention weights.
            output_hidden_states: A boolean flag indicating whether to output hidden states.
            return_dict: A boolean flag indicating whether to return the output as a dictionary.

        Returns:
            None

        Raises:
            ValueError: If the dimensions of input tensors are not valid.
            TypeError: If the input parameters are not of the expected types.
            IndexError: If the head mask dimensions do not match the expected shape.
        """
        all_word_hidden_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # TODO
            # if self.gradient_checkpointing and self.training:
            #
            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs, output_attentions)
            #
            #         return custom_forward
            #
            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(layer_module),
            #         word_hidden_states,
            #         entity_hidden_states,
            #         attention_mask,
            #         layer_head_mask,
            #     )
            layer_outputs = layer_module(
                word_hidden_states,
                entity_hidden_states,
                attention_mask,
                layer_head_mask,
                output_attentions,
            )

            word_hidden_states = layer_outputs[0]

            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    word_hidden_states,
                    all_word_hidden_states,
                    all_self_attentions,
                    entity_hidden_states,
                    all_entity_hidden_states,
                ]
                if v is not None
            )
        return {
            "last_hidden_state": word_hidden_states,
            "hidden_states": all_word_hidden_states,
            "attentions": all_self_attentions,
            "entity_last_hidden_state": entity_hidden_states,
            "entity_hidden_states": all_entity_hidden_states,
        }


class LukePooler(nn.Module):
    """
    LukePooler
    """
    def __init__(self, config):
        """
        Initializes an instance of the LukePooler class.

        Args:
            self (object): The instance of the LukePooler class.
            config (object): An object containing configuration parameters for the LukePooler.
                This parameter is required to configure the dense layer and activation function.
                It should have a 'hidden_size' attribute specifying the size of the hidden layer.
                Raises a TypeError if config is not provided or if hidden_size is missing.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is missing or if the 'hidden_size' attribute is not present
                in the config object.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: Tensor) -> Tensor:
        """
        This method constructs a pooled output tensor based on the hidden states provided.

        Args:
            self: An instance of the LukePooler class.
            hidden_states (Tensor): A tensor containing hidden states from which the pooled output will be constructed.
                It is expected to have shape (batch_size, sequence_length, hidden_size).

        Returns:
            Tensor: A tensor representing the pooled output obtained from the hidden states.
                It is obtained by applying a dense layer followed by an activation function to the
                first token's hidden state.

        Raises:
            None
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EntityPredictionHeadTransform(nn.Module):
    """
    EntityPredictionHeadTransform
    """
    def __init__(self, config):
        """
        Initializes the EntityPredictionHeadTransform class.

        Args:
            self: The instance of the EntityPredictionHeadTransform class.
            config:
                An object containing configuration parameters for the EntityPredictionHeadTransform class.

                - Type: Any
                - Purpose: Specifies the configuration settings for the EntityPredictionHeadTransform instance.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            TypeError: If the config.hidden_act parameter is not a string or a valid activation function.
            ValueError: If the config.entity_emb_size is invalid or the config.layer_norm_eps is not
                within the valid range.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm([config.entity_emb_size, ], eps=config.layer_norm_eps)

    def construct(self, hidden_states):
        """
        Method to construct the entity prediction head transformation.

        Args:
            self (EntityPredictionHeadTransform): An instance of the EntityPredictionHeadTransform class.
            hidden_states (tensor): The input hidden states to be transformed.
                It should be a tensor representing the hidden states of the model.

        Returns:
            tensor: The transformed hidden states after passing through the dense layer,
                activation function, and layer normalization.
                It retains the same shape and structure as the input hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# 0325==============================
class EntityPredictionHead(nn.Module):
    """
    EntityPredictionHead
    """
    def __init__(self, config):
        """
        Initialize the EntityPredictionHead instance.

        Args:
            self (EntityPredictionHead): The EntityPredictionHead instance.
            config (object): The configuration object containing parameters for entity prediction head.
                This object should have attributes required for initializing the EntityPredictionHead instance.
                It must be provided as an argument during initialization.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is of an incorrect type.
            ValueError: If the config object does not contain the required attributes for initialization.
            RuntimeError: If there is an issue with initializing any component within the EntityPredictionHead instance.
        """
        super().__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.entity_emb_size, config.entity_vocab_size, bias=False)
        self.bias = mindspore.Parameter(ops.zeros((config.entity_vocab_size,)))

    def construct(self, hidden_states):
        """
        Method to construct the entity prediction head using the given hidden states.

        Args:
            self (EntityPredictionHead): An instance of the EntityPredictionHead class.
            hidden_states (tensor): The hidden states to be used for constructing the entity prediction head.
                Should be a tensor representing the hidden states of the input data.

        Returns:
            None: This method does not return any value.
                The entity prediction head is constructed and updated within the class instance.

        Raises:
            TypeError: If the input hidden_states is not of type tensor.
            ValueError: If the hidden_states tensor is empty or has invalid dimensions.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias

        return hidden_states


class LukePreTrainedModel(PreTrainedModel):
    """
    LukePreTrainedModel
    """
    config_class = LukeConfig
    base_model_prefix = "luke"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LukeAttention", "LukeEntityEmbeddings"]

    def get_input_embeddings(self) -> "nn.Module":
        """
        Method to retrieve the input embeddings for the LukePreTrainedModel.

        Args:
            self: Instance of the LukePreTrainedModel class.
                This parameter refers to the current instance of the LukePreTrainedModel class.
                It is used to access the attributes and methods associated with the instance.

        Returns:
            nn.Module: An object of type nn.Module.
                The return value is the input embeddings of the model stored in an nn.Module object.
                This object contains the embeddings that represent the input data for the model.

        Raises:
            None
        """

    def set_input_embeddings(self, new_embeddings: "nn.Module"):
        """
        This method sets the input embeddings for the LukePreTrainedModel.

        Args:
            self (LukePreTrainedModel): The instance of the LukePreTrainedModel class.
            new_embeddings (nn.Module): The new input embeddings to be set for the model. It should be an instance of 'nn.Module'.

        Returns:
            None.

        Raises:
            None
        """

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resize the position embeddings to accommodate a new number of position embeddings in the LukePreTrainedModel.

        Args:
            self (LukePreTrainedModel): The instance of the LukePreTrainedModel class.
            new_num_position_embeddings (int): The new number of position embeddings to resize to.
                Must be a positive integer.

        Returns:
            None.

        Raises:
            None.
        """

    def get_position_embeddings(self):
        """
        This method retrieves the position embeddings for the LukePreTrainedModel.

        Args:
            self: An instance of the LukePreTrainedModel class.

        Returns:
            None.

        Raises:
            None.
        """

    def _init_weights(self, cell: nn.Module):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            if cell.embedding_size == 1:  # embedding for bias parameters
                cell.weight.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            else:
                weight = initializer(Normal(self.config.initializer_range),
                                                        cell.weight.shape,
                                                        cell.weight.dtype)
                if cell.padding_idx is not None:
                    weight[cell.padding_idx] = 0
                cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class LukeModel(LukePreTrainedModel):
    """
    LukeModel
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        """
        Initializes a new LukeModel instance.

        Args:
            self: The instance of the LukeModel class.
            config (LukeConfig): An instance of LukeConfig containing the configuration for the model.
            add_pooling_layer (bool, optional): A boolean indicating whether to add a pooling layer. Defaults to True.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not an instance of LukeConfig.
            ValueError: If the add_pooling_layer parameter is not a boolean.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = LukeEncoder(config)

        self.pooler = LukePooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the LukeModel class.

        Args:
            self: The instance of the LukeModel class.

        Returns:
            The word embeddings for the input.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings of the LukeModel.

        Args:
            self (LukeModel): The LukeModel instance to which the input embeddings will be set.
            new_embeddings (any): New embeddings to be set as input embeddings for the LukeModel.

        Returns:
            None.

        Raises:
            None.
        """
        self.embeddings.word_embeddings = new_embeddings

    def get_entity_embeddings(self):
        """get_entity_embeddings"""
        return self.entity_embeddings.entity_embeddings

    def set_entity_embeddings(self, new_embeddings):
        """set_entity_embeddings"""
        self.entity_embeddings.entity_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Method to prune attention heads in a LUKE model.

        Args:
            self (LukeModel): The instance of LukeModel.
            heads_to_prune (int): The number of attention heads to prune from the model.
                It specifies which attention heads should be pruned.

        Returns:
            None.

        Raises:
            NotImplementedError: Raised when an attempt is made to prune attention heads in a LUKE model.
                LUKE does not support the pruning of attention heads, so this operation is not allowed.
        """
        raise NotImplementedError("LUKE does not support the pruning of attention heads")

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        '''
        The 'construct' method in the 'LukeModel' class is responsible for constructing the model
        based on the provided inputs and configuration.

        Args:
            self: The instance of the class.
            input_ids (Optional[Tensor]): The input tensor representing the token ids. Default is None.
            attention_mask (Optional[Tensor]): The attention mask tensor indicating the positions of the padded tokens.
                Default is None.
            token_type_ids (Optional[Tensor]): The tensor representing the token type ids. Default is None.
            position_ids (Optional[Tensor]): The tensor representing the position ids. Default is None.
            entity_ids (Optional[Tensor]): The tensor representing the entity ids. Default is None.
            entity_attention_mask (Optional[Tensor]): The attention mask tensor for entity tokens. Default is None.
            entity_token_type_ids (Optional[Tensor]): The tensor representing the token type ids for entities.
                Default is None.
            entity_position_ids (Optional[Tensor]): The tensor representing the position ids for entities.
                Default is None.
            head_mask (Optional[Tensor]): The tensor representing the head mask. Default is None.
            inputs_embeds (Optional[Tensor]): The embedded inputs tensor. Default is None.
            output_attentions (Optional[bool]): Whether to return attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to return hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default is None.

        Returns:
            None.

        Raises:
            ValueError:
                - If both input_ids and inputs_embeds are specified simultaneously.
                - If neither input_ids nor inputs_embeds is specified.

        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None and inputs_embeds is None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = ops.ones((batch_size, seq_length))
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)
        if entity_ids is not None:
            entity_seq_length = entity_ids.shape[1]
            if entity_attention_mask is None:
                entity_attention_mask = ops.ones((batch_size, entity_seq_length))
            if entity_token_type_ids is None:
                entity_token_type_ids = ops.zeros((batch_size, entity_seq_length), dtype=mindspore.int64)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # First, compute word embeddings
        word_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Second, compute extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask)

        # Third, compute entity embeddings and concatenate with word embeddings
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)

        encoder_outputs = self.encoder(
            word_embedding_output,
            entity_embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0] if not return_dict else tuple(
            i for i in encoder_outputs.values() if i is not None)[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'hidden_states': encoder_outputs['hidden_states'],
            'attentions': encoder_outputs['attentions'],
            'entity_last_hidden_state': encoder_outputs['entity_last_hidden_state'],
            'entity_hidden_states': encoder_outputs['entity_hidden_states']
        }

    def get_extended_attention_mask(
            self, attention_mask: Tensor, input_shape: Tuple[int], dtype=None
    ):
        """
        This method 'get_extended_attention_mask' in the class 'LukeModel' takes 4 parameters:

        Args:
            self: Represents the instance of the class.
            attention_mask (Tensor): A 2D or 3D tensor representing the attention mask.
                This tensor is concatenated with 'input_shape' if provided.
            input_shape (Tuple[int]): A tuple containing the shape information to be concatenated with 'attention_mask'.
                Set to None if not provided.
            dtype: Data type for the extended attention mask. Default is None.

        Returns:
            None.

        Raises:
            ValueError: Raised when the shape of the 'attention_mask' is incorrect.
        """
        if input_shape is not None:
            attention_mask = ops.cat([attention_mask, input_shape], axis=-1)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * Tensor(
            np.finfo(mindspore.dtype_to_nptype(self.dtype)).min)

        return extended_attention_mask


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.
    """
    mask = ops.not_equal(input_ids, padding_idx).astype(mindspore.int32)
    incremental_indices = ops.cumsum(mask, -1).astype(mindspore.int32) * mask
    return incremental_indices.astype(mindspore.int64) + padding_idx


class LukeLMHead(nn.Module):
    """LukeLMead"""
    def __init__(self, config):
        """
        Initializes the LukeLMHead class.

        Args:
            self (object): The instance of the LukeLMHead class.
            config (object):
                An instance of the configuration class containing the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - vocab_size (int): The size of the vocabulary.
                - layer_norm_eps (float): The epsilon value for layer normalization.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of the correct type.
            ValueError: If the hidden_size or vocab_size attributes in the config are not positive integers.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def construct(self, features, **kwargs):
        """
        Constructs the output of the LukeLMHead model by performing a series of operations on the input features.

        Args:
            self (LukeLMHead): The instance of the LukeLMHead class.
            features (tensor): The input features to be processed by the model.

        Returns:
            tensor: The output tensor after processing the input features through the model.

        Raises:
            None.
        """
        # hidden
        x = self.dense(features)
        x = ops.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        # endecoded
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        '''
        This method ties the weights of the LukeLMHead model's decoder with its bias.

        Args:
            self (object): The instance of the LukeLMHead class.

        Returns:
            None.

        Raises:
            None.
        '''
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class LukeForMaskedLM(LukePreTrainedModel):
    """
    LukeForMaskedLM
    """
    _keys_to_ignore_on_save = [
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
        r"entity_predictions.decoder.weight",
    ]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
        r"entity_predictions.decoder.weight",
    ]

    def __init__(self, config):
        """
        Initializes an instance of the 'LukeForMaskedLM' class.

        Args:
            self: The current instance of the 'LukeForMaskedLM' class.
            config: An object of type 'ConfigBase' containing the configuration parameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.luke = LukeModel(config)

        self.lm_head = LukeLMHead(config)
        self.entity_predictions = EntityPredictionHead(config)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def tie_weights(self):
        """tie_weight"""
        super().tie_weights()
        self._tie_or_clone_weights(self.entity_predictions.decoder, self.luke.entity_embeddings.entity_embeddings)

    def get_output_embeddings(self):
        """get_output_embeddings"""
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """set_output_embeddings"""
        self.lm_head.decoder = new_embeddings

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            entity_labels: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the outputs for the LukeForMaskedLM model.

        Args:
            self (LukeForMaskedLM): The instance of the LukeForMaskedLM class.
            input_ids (Optional[Tensor]): The input token IDs. Default: None.
            attention_mask (Optional[Tensor]): The attention mask. Default: None.
            token_type_ids (Optional[Tensor]): The token type IDs. Default: None.
            position_ids (Optional[Tensor]): The position IDs. Default: None.
            entity_ids (Optional[Tensor]): The entity IDs. Default: None.
            entity_attention_mask (Optional[Tensor]): The entity attention mask. Default: None.
            entity_token_type_ids (Optional[Tensor]): The entity token type IDs. Default: None.
            entity_position_ids (Optional[Tensor]): The entity position IDs. Default: None.
            labels (Optional[Tensor]): The labels for masked language modeling. Default: None.
            entity_labels (Optional[Tensor]): The labels for entity prediction. Default: None.
            head_mask (Optional[Tensor]): The head mask. Default: None.
            inputs_embeds (Optional[Tensor]): The input embeddings. Default: None.
            output_attentions (Optional[bool]): Whether to output attentions. Default: None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default: None.
            return_dict (Optional[bool]): Whether to return a dictionary output. Default: None.

        Returns:
            Tuple of (loss, mlm_loss, mep_loss, logits, entity_logits, hidden_states, entity_hidden_states, attentions):

                - loss (Tensor or None): The total loss. None if no loss is calculated.
                - mlm_loss (Tensor or None): The loss for masked language modeling.
                None if no loss is calculated.
                - mep_loss (Tensor or None): The loss for entity prediction. None if no loss is calculated.
                - logits (Tensor or None): The logits for masked language modeling.
                - entity_logits (Tensor or None): The logits for entity prediction.
                - hidden_states (Tuple[Tensor] or None): The hidden states of the model. None if not returned.
                - entity_hidden_states (Tuple[Tensor] or None): The hidden states for entity prediction.
                None if not returned.
                - attentions (Tuple[Tensor] or None): The attentions of the model. None if not returned.

        Raises:
            None.
        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        loss = None
        mlm_loss = None
        logits = self.lm_head(outputs['last_hidden_state'])
        if labels is not None:
            mlm_loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            if loss is None:
                loss = mlm_loss

        mep_loss = None
        entity_logits = None
        if outputs['entity_last_hidden_state'] is not None:
            entity_logits = self.entity_predictions(outputs['entity_last_hidden_state'])
            if entity_labels is not None:
                mep_loss = self.loss_fn(entity_logits.view(-1, self.config.entity_vocab_size), entity_labels.view(-1))
                if loss is None:
                    loss = mep_loss
                else:
                    loss = loss + mep_loss
        return tuple(
            v
            for v in [
                loss,
                mlm_loss,
                mep_loss,
                logits,
                entity_logits,
                outputs['hidden_states'],
                outputs['entity_hidden_states'],
                outputs['attentions'],
            ]
            if v is not None
        )


class LukeForEntityClassification(LukePreTrainedModel):
    """
    LukeForEntityClassification
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LukeForEntityClassification class.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing the settings for the LukeForEntityClassification model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required settings.
            RuntimeError: If there is an issue with the initialization process.
        """
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the LukeForEntityClassification model.

        Args:
            self (LukeForEntityClassification): The instance of the LukeForEntityClassification class.
            input_ids (Optional[Tensor]): The input tensor containing the indices of input sequence tokens in the vocabulary.
            attention_mask (Optional[Tensor]): The optional mask tensor, usually used to ignore padding tokens.
            token_type_ids (Optional[Tensor]): The optional tensor containing the type ids of input sequence tokens.
            position_ids (Optional[Tensor]): The optional tensor containing the positions ids of input sequence tokens.
            entity_ids (Optional[Tensor]): The optional tensor containing the indices of entity tokens in the vocabulary.
            entity_attention_mask (Optional[Tensor]): The optional mask tensor for entity tokens.
            entity_token_type_ids (Optional[Tensor]): The optional tensor containing the type ids of entity sequence tokens.
            entity_position_ids (Optional[Tensor]): The optional tensor containing the positions ids of entity sequence tokens.
            head_mask (Optional[Tensor]): The optional mask tensor for attention heads.
            inputs_embeds (Optional[Tensor]): The optional tensor containing the embeddings of input sequence tokens.
            labels (Optional[Tensor]): The optional tensor containing the labels of the entity classification task.
            output_attentions (Optional[bool]): Whether to return the attentions weights of the model.
            output_hidden_states (Optional[bool]): Whether to return the hidden states of the model.
            return_dict (Optional[bool]): Whether to return a dictionary instead of a tuple.

        Returns:
            Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]: A tuple containing
                the loss, logits, hidden states, entity hidden states, and attentions weights (if available) respectively.

        Raises:
            None.

        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        feature_vector = outputs['entity_last_hidden_state'][:, 0, :]
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            if labels.ndim == 1:
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), Tensor(),
                                                            Tensor())
        return tuple(
            v
            for v in [loss, logits, outputs['hidden_states'], outputs['entity_hidden_states'], outputs['attentions']]
            if v is not None
        )


class LukeForEntityPairClassification(LukePreTrainedModel):
    """
    LukeForEntityPairClassification
    """
    def __init__(self, config):
        """
        Initializes a new instance of LukeForEntityPairClassification.

        Args:
            self: The object instance itself.
            config:
                The configuration object containing various parameters.

                - Type: object
                - Purpose: Contains the configuration settings for the Luke model.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels, bias=False)

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        This method 'construct' in the class 'LukeForEntityPairClassification' is responsible for constructing
        the model and performing entity pair classification.

        Args:
            self: The instance of the class.
            input_ids (Optional[Tensor]): Input tensor containing token indices. Default is None.
            attention_mask (Optional[Tensor]): Mask tensor for the input, indicating which tokens should be attended to.
                Default is None.
            token_type_ids (Optional[Tensor]): Tensor specifying the type of token (e.g., segment A or B). Default is None.
            position_ids (Optional[Tensor]): Tensor specifying the position of tokens. Default is None.
            entity_ids (Optional[Tensor]): Tensor containing entity indices.
            entity_attention_mask (Optional[Tensor]): Mask tensor for entity inputs. Default is None.
            entity_token_type_ids (Optional[Tensor]): Tensor specifying the type of entity token. Default is None.
            entity_position_ids (Optional[Tensor]): Tensor specifying the position of entity tokens. Default is None.
            head_mask (Optional[Tensor]): Mask tensor for attention heads. Default is None.
            inputs_embeds (Optional[Tensor]): Additional embeddings to be added to the model input embeddings.
                Default is None.
            labels (Optional[Tensor]): Tensor containing the classification labels. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary as output. Default is None.

        Returns:
            Tuple:
                A tuple containing elements that are not None among loss (if labels provided), logits, hidden states,
                entity hidden states, and attentions. Returns None if all elements are None.

        Raises:
            ValueError: If labels are provided but have an incorrect shape for cross-entropy computation.
            TypeError: If the input types are not as expected by the method.
            RuntimeError: If there are runtime issues during the execution of the method.
        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        feature_vector = ops.cat(
            [outputs['entity_last_hidden_state'][:, 0, :], outputs['entity_last_hidden_state'][:, 1, :]], axis=1
        )
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            if labels.ndim == 1:
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), Tensor(),
                                                            Tensor())
        return tuple(
            v
            for v in [loss, logits, outputs['hidden_states'], outputs['entity_hidden_states'], outputs['attentions']]
            if v is not None
        )


class LukeForEntitySpanClassification(LukePreTrainedModel):
    """
    LukeForEntitySpanClassification
    """
    def __init__(self, config):
        """
        Initializes an instance of the LukeForEntitySpanClassification class.

        Args:
            self: The instance of the class.
            config: The configuration object containing various settings and parameters for the model.
                It should be an instance of the configuration class specific to LukeForEntitySpanClassification.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the configuration provided is invalid or missing required parameters.
            RuntimeError: If there is an issue with the initialization process.
        """
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask=None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            entity_start_positions: Optional[Tensor] = None,
            entity_end_positions: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the forward pass of LukeForEntitySpanClassification model.

        Args:
            self (LukeForEntitySpanClassification): The instance of the LukeForEntitySpanClassification class.
            input_ids (Optional[Tensor]): The input tensor of shape (batch_size, sequence_length) containing the
                input tokens indices.
            attention_mask (Tensor): The attention mask tensor of shape (batch_size, sequence_length) containing
                the attention mask values.
            token_type_ids (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the token type ids.
            position_ids (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the position ids.
            entity_ids (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing the entity ids.
            entity_attention_mask (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the entity attention mask values.
            entity_token_type_ids (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the entity token type ids.
            entity_position_ids (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the entity position ids.
            entity_start_positions (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the start positions of the entities.
            entity_end_positions (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing
                the end positions of the entities.
            head_mask (Optional[Tensor]): The tensor of shape (batch_size, num_heads) containing the head mask values.
            inputs_embeds (Optional[Tensor]): The tensor of shape (batch_size, sequence_length, hidden_size) containing
                the input embeddings.
            labels (Optional[Tensor]): The tensor of shape (batch_size, sequence_length) containing the labels.
            output_attentions (Optional[bool]): Whether to output the attentions.
            output_hidden_states (Optional[bool]): Whether to output the hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary instead of a tuple.

        Returns:
            tuple: Tuple of values containing the loss (Tensor), logits (Tensor), hidden states (Tensor),
                entity hidden states (Tensor), and attentions (Tensor) if not None.

        Raises:
            None.
        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_size = outputs['last_hidden_state'].shape[-1]
        entity_start_positions = ops.BroadcastTo(shape=(-1, -1, hidden_size))(entity_start_positions.unsqueeze(-1))
        start_states = ops.gather_elements(outputs['last_hidden_state'], -2, entity_start_positions)

        entity_end_positions = ops.BroadcastTo(shape=(-1, -1, hidden_size))(entity_end_positions.unsqueeze(-1))
        end_states = ops.gather_elements(outputs['last_hidden_state'], -2, entity_end_positions)

        feature_vector = ops.cat([start_states, end_states, outputs['entity_last_hidden_state']], axis=2)
        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = None
        if labels is not None:
            if labels.ndim == 2:
                loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits),
                                                            weight=None, pos_weight=None)

        return tuple(
            v
            for v in [loss, logits, outputs['hidden_states'], outputs['entity_hidden_states'], outputs['attentions']]
            if v is not None
        )


class LukeForSequenceClassification(LukePreTrainedModel):
    """
    LukeForSequenceClassification
    """
    def __init__(self, config):
        """
        Initializes a LukeForSequenceClassification instance.

        Args:
            self (LukeForSequenceClassification): The current instance of the LukeForSequenceClassification class.
            config (LukeConfig): The configuration object containing various settings for the Luke model.
                It must include the number of labels (num_labels) for classification tasks.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of type LukeConfig.
            ValueError: If the num_labels attribute is missing in the config object.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.luke = LukeModel(config)
        self.dropout = nn.Dropout(p=
                                  config.classifier_dropout
                                  if config.classifier_dropout is not None
                                  else config.hidden_dropout_prob
                                  )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Method 'construct' in the class 'LukeForSequenceClassification'.

        Args:
            self: The object instance.
            input_ids (Optional[Tensor]): Input IDs for the model. Default is None.
            attention_mask (Optional[Tensor]): Mask to avoid performing attention on padding tokens. Default is None.
            token_type_ids (Optional[Tensor]): Segment token indices to differentiate between two sequences.
                Default is None.
            position_ids (Optional[Tensor]): Position indices for the input tokens. Default is None.
            entity_ids (Optional[Tensor]): Entity IDs for the input. Default is None.
            entity_attention_mask (Optional[Tensor]): Mask for entity attention. Default is None.
            entity_token_type_ids (Optional[Tensor]): Segment token indices for entities. Default is None.
            entity_position_ids (Optional[Tensor]): Position indices for entity tokens. Default is None.
            head_mask (Optional[Tensor]): Mask to nullify specific heads of the model. Default is None.
            inputs_embeds (Optional[Tensor]): Optional input embeddings. Default is None.
            labels (Optional[Tensor]): Labels for the input. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return as a dictionary. Default is None.

        Returns:
            tuple: A tuple containing loss, logits, hidden states, entity hidden states, and attentions
                if they are not None. Otherwise, returns None.

        Raises:
            ValueError: If the configuration problem type is not recognized.
            RuntimeError: If an unexpected error occurs during the computation.
            TypeError: If the input types are incorrect.
        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs['pooler_output']

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
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return tuple(
            v
            for v in
            [loss, logits, outputs['hidden_states'], outputs['entity_hidden_states'], outputs['attentions']]
            if v is not None
        )


class LukeForTokenClassification(LukePreTrainedModel):
    """
    LukeForTokenClassification
    """
    def __init__(self, config):
        """
        Initializes a new instance of the LukeForTokenClassification class.

        Args:
            self: The object itself.
            config: An instance of class 'LukeConfig' containing the configuration parameters for the
                LukeForTokenClassification model.

                - Type: LukeConfig
                - Purpose: This parameter specifies the configuration settings for the model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.luke = LukeModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=
                                  config.classifier_dropout
                                  if config.classifier_dropout is not None
                                  else config.hidden_dropout_prob
                                  )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the model for token classification using the Luke architecture.

        Args:
            self: The object instance.
            input_ids (Optional[Tensor]): The input tensor of token indices. Default is None.
            attention_mask (Optional[Tensor]): The attention mask tensor. Default is None.
            token_type_ids (Optional[Tensor]): The tensor indicating token types. Default is None.
            position_ids (Optional[Tensor]): The tensor indicating token positions. Default is None.
            entity_ids (Optional[Tensor]): The tensor representing entity indices. Default is None.
            entity_attention_mask (Optional[Tensor]): The attention mask for entity tokens. Default is None.
            entity_token_type_ids (Optional[Tensor]): The tensor indicating entity token types. Default is None.
            entity_position_ids (Optional[Tensor]): The tensor indicating entity token positions. Default is None.
            head_mask (Optional[Tensor]): The tensor for masking heads. Default is None.
            inputs_embeds (Optional[Tensor]): The embedded input tensor. Default is None.
            labels (Optional[Tensor]): The tensor of labels for token classification. Default is None.
            output_attentions (Optional[bool]): Whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Whether to return a dictionary. Default is None.

        Returns:
            Tuple[Optional[Tensor], Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
                A tuple containing the loss, logits, hidden states, entity hidden states, and attentions.
                Any element that is not None is included in the tuple.

        Raises:
            None
        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs['last_hidden_state']
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return tuple(
            v
            for v in [loss, logits, outputs['hidden_states'], outputs['entity_hidden_states'], outputs['attentions']]
            if v is not None
        )


class LukeForQuestionAnswering(LukePreTrainedModel):
    """
    LukeForQuestionAnswering
    """
    def __init__(self, config):
        """
        Initializes the LukeForQuestionAnswering class.

        Args:
            self (LukeForQuestionAnswering): The instance of the LukeForQuestionAnswering class.
            config: The configuration object containing the settings for the Luke model.
                This parameter is required and should be an instance of the configuration class for Luke models.
                It must include the following attributes:

                - num_labels (int): The number of labels for the question answering task.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is of an incorrect type.
            ValueError: If the num_labels attribute is not specified in the config object.
        """
        super().__init__(config)

        self.num_labels = config.num_labels

        self.luke = LukeModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            start_positions: Optional[Tensor] = None,
            end_positions: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the forward pass of the LukeForQuestionAnswering model.

        Args:
            self (LukeForQuestionAnswering): An instance of the LukeForQuestionAnswering class.
            input_ids (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the input token IDs.
            attention_mask (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the attention mask.
            token_type_ids (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the token type IDs.
            position_ids (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the position IDs.
            entity_ids (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the entity IDs.
            entity_attention_mask (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length)
                containing the entity attention mask.
            entity_token_type_ids (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length)
                containing the entity token type IDs.
            entity_position_ids (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length)
                containing the entity position IDs.
            head_mask (Optional[Tensor]): Input tensor of shape (batch_size, num_heads) containing the head mask.
            inputs_embeds (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length, hidden_size)
                containing the embedded inputs.
            start_positions (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the
                start positions for answer span prediction.
            end_positions (Optional[Tensor]): Input tensor of shape (batch_size, sequence_length) containing the
                end positions for answer span prediction.
            output_attentions (Optional[bool]): Whether to output attentions weights. Default: None.
            output_hidden_states (Optional[bool]): Whether to output hidden states. Default: None.
            return_dict (Optional[bool]): Whether to return a dictionary as output. Default: None.

        Returns:
            tuple:
                A tuple containing the following elements:

                - total_loss (Optional[Tensor]): The total loss if start_positions and end_positions are provided.
                None otherwise.
                - start_logits (Optional[Tensor]): Tensor of shape (batch_size, sequence_length) containing
                the predicted start logits.
                - end_logits (Optional[Tensor]): Tensor of shape (batch_size, sequence_length) containing
                the predicted end logits.
                - hidden_states (Optional[List[Tensor]]): List of tensors containing the hidden states of
                the model at each layer.
                - entity_hidden_states (Optional[List[Tensor]]): List of tensors containing the hidden states of
                the entity encoder at each layer.
                - attentions (Optional[List[Tensor]]): List of tensors containing the attention weights of
                the model at each layer.

        Raises:
            None.
        """
        return_dict = True
        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs['last_hidden_state']

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
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return tuple(
            v
            for v in [
                total_loss,
                start_logits,
                end_logits,
                outputs['hidden_states'],
                outputs['entity_hidden_states'],
                outputs['attentions'],
            ]
            if v is not None
        )


class LukeForMultipleChoice(LukePreTrainedModel):
    """
    LukeForMultipleChoice
    """
    def __init__(self, config):
        """
        Initializes an instance of the LukeForMultipleChoice class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration settings for the model (type: <class 'config'>).

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.luke = LukeModel(config)
        self.dropout = nn.Dropout(p=
                                  config.classifier_dropout
                                  if config.classifier_dropout is not None
                                  else config.hidden_dropout_prob
                                  )
        self.classifier = nn.Linear(config.hidden_size, 1)

    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            entity_ids: Optional[Tensor] = None,
            entity_attention_mask: Optional[Tensor] = None,
            entity_token_type_ids: Optional[Tensor] = None,
            entity_position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        """
        Constructs the LukeForMultipleChoice model.

        Args:
            self (LukeForMultipleChoice): The instance of the LukeForMultipleChoice class.
            input_ids (Optional[Tensor]): The input sequence token IDs of shape [batch_size, num_choices, sequence_length].
                (default: None)
            attention_mask (Optional[Tensor]): The attention mask tensor of shape [batch_size, num_choices, sequence_length].
                (default: None)
            token_type_ids (Optional[Tensor]): The token type IDs tensor of shape [batch_size, num_choices, sequence_length].
                (default: None)
            position_ids (Optional[Tensor]): The position IDs tensor of shape [batch_size, num_choices, sequence_length].
                (default: None)
            entity_ids (Optional[Tensor]): The entity token IDs tensor of shape [batch_size, num_choices, entity_length].
                (default: None)
            entity_attention_mask (Optional[Tensor]): The entity attention mask tensor of
                shape [batch_size, num_choices, entity_length]. (default: None)
            entity_token_type_ids (Optional[Tensor]): The entity token type IDs tensor of
                shape [batch_size, num_choices, entity_length]. (default: None)
            entity_position_ids (Optional[Tensor]): The entity position IDs tensor of
                shape [batch_size, num_choices, entity_length]. (default: None)
            head_mask (Optional[Tensor]): The head mask tensor of shape [num_hidden_layers, num_attention_heads].
                (default: None)
            inputs_embeds (Optional[Tensor]): The input embeddings tensor of shape
                [batch_size, num_choices, sequence_length, hidden_size]. (default: None)
            labels (Optional[Tensor]): The labels tensor of shape [batch_size]. (default: None)
            output_attentions (Optional[bool]): Whether to output attentions. (default: None)
            output_hidden_states (Optional[bool]): Whether to output hidden states. (default: None)
            return_dict (Optional[bool]): Whether to return a dictionary instead of a tuple of outputs. (default: None)

        Returns:
            tuple:
                Tuple of (loss, reshaped_logits, hidden_states, entity_hidden_states, attentions):

                - loss (Optional[Tensor]): The training loss tensor. Returns None if labels are not provided.
                - reshaped_logits (Tensor): The reshaped logits tensor of shape [batch_size * num_choices, num_choices].
                - hidden_states (Optional[List[Tensor]]): The hidden states of the model at the output of each layer.
                Returns None if output_hidden_states is set to False.
                - entity_hidden_states (Optional[List[Tensor]]): The hidden states of the model for the entity
                embeddings at the output of each layer. Returns None if output_hidden_states is set to False or
                entity embeddings are not provided.
                - attentions (Optional[List[Tensor]]): The attention weights of the model at the output of each layer.
                Returns None if output_attentions is set to False.
        
        Raises:
            None.
        """
        return_dict = True
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

        entity_ids = entity_ids.view(-1, entity_ids.shape[-1]) if entity_ids is not None else None
        entity_attention_mask = (
            entity_attention_mask.view(-1, entity_attention_mask.shape[-1])
            if entity_attention_mask is not None
            else None
        )
        entity_token_type_ids = (
            entity_token_type_ids.view(-1, entity_token_type_ids.shape[-1])
            if entity_token_type_ids is not None
            else None
        )
        entity_position_ids = (
            entity_position_ids.view(-1, entity_position_ids.shape[-2], entity_position_ids.shape[-1])
            if entity_position_ids is not None
            else None
        )

        outputs = self.luke(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            entity_ids=entity_ids,
            entity_attention_mask=entity_attention_mask,
            entity_token_type_ids=entity_token_type_ids,
            entity_position_ids=entity_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs['pooler_output']

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        return tuple(
            v
            for v in [
                loss,
                reshaped_logits,
                outputs['hidden_states'],
                outputs['entity_hidden_states'],
                outputs['attentions'],
            ]
            if v is not None
        )


def apply_chunking_to_forward(
        forward_fn: Callable[..., mindspore.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> mindspore.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts
    of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    """
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method
    # -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return ops.cat(output_chunks, axis=chunk_dim)

    return forward_fn(*input_tensors)

__all__ = [
        "LukeForEntityClassification",
        "LukeForEntityPairClassification",
        "LukeForEntitySpanClassification",
        "LukeForMultipleChoice",
        "LukeForQuestionAnswering",
        "LukeForSequenceClassification",
        "LukeForTokenClassification",
        "LukeForMaskedLM",
        "LukeModel",
        "LukePreTrainedModel",
    ]
