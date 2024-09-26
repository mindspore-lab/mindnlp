# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""nezha model"""
import math
import mindspore

from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from .nezha_config import NezhaConfig
from ...modeling_utils import PreTrainedModel
from ...ms_utils import prune_linear_layer, find_pruneable_heads_and_indices, apply_chunking_to_forward
from ...activations import ACT2FN


logger = logging.get_logger(__name__)

__all__ = [
        "NezhaForNextSentencePrediction",
        "NezhaForMaskedLM",
        "NezhaForPreTraining",
        "NezhaForMultipleChoice",
        "NezhaForQuestionAnswering",
        "NezhaForSequenceClassification",
        "NezhaForTokenClassification",
        "NezhaModel",
        "NezhaPreTrainedModel",
    ]


class NezhaRelativePositionsEncoding(nn.Module):
    """Implement the Functional Relative Position Encoding"""
    def __init__(self, length, depth, max_relative_position=127):
        """
        Initializes the NezhaRelativePositionsEncoding object with the specified parameters.
        
        Args:
            self (object): The instance of the NezhaRelativePositionsEncoding class.
            length (int): The length of the input sequence.
            depth (int): The depth of the embeddings table.
            max_relative_position (int, optional): The maximum allowed relative position. Defaults to 127.
        
        Returns:
            None.

        Raises:
            ValueError: If length or depth is not an integer.
            ValueError: If max_relative_position is not an integer.
            ValueError: If max_relative_position is less than 1.
        """
        super().__init__()
        vocab_size = max_relative_position * 2 + 1
        range_vec = ops.arange(length)
        range_mat = ops.tile(range_vec, (length, 1))
        distance_mat = range_mat - ops.t(range_mat)
        distance_mat_clipped = ops.clamp(distance_mat, -max_relative_position,
                                        max_relative_position)
        final_mat = distance_mat_clipped + max_relative_position

        # TODO: use numpy to avoid setitem(mindspore not support complete `view`.)
        embeddings_table = ops.zeros((vocab_size, depth))
        # position = ops.arange(0, vocab_size, dtype=mindspore.float32).expand_dims(1)
        # div_term = ops.exp(ops.arange(0, depth, 2).astype(mindspore.float32) * (-math.log(10000.0)) / depth)
        # embeddings_table[:, 0::2] = ops.sin(position * div_term)
        # embeddings_table[:, 1::2] = ops.cos(position * div_term)

        flat_relative_positions_matrix = final_mat.view(-1)
        on_value, off_value = Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
        one_hot_relative_positions_matrix = F.one_hot(
            flat_relative_positions_matrix, vocab_size
        ).astype(mindspore.float32)
        positions_encoding = ops.matmul(one_hot_relative_positions_matrix, embeddings_table)
        my_shape = list(final_mat.shape)
        my_shape.append(depth)
        self.positions_encoding = Parameter(positions_encoding.view(tuple(my_shape)), requires_grad=False)

    def forward(self, length):
        """
        Constructs a relative positions encoding matrix of specified length.

        Args:
            self (NezhaRelativePositionsEncoding): The instance of the NezhaRelativePositionsEncoding class.
            length (int): The length of the positions encoding matrix to be forwarded.
                Must be a non-negative integer.

        Returns:
            None: The method modifies the internal state of the NezhaRelativePositionsEncoding instance.

        Raises:
            IndexError: If the length provided is greater than the dimensions of the positions_encoding matrix.
            ValueError: If the length provided is a negative integer.
        """
        return self.positions_encoding[:length, :length, :]


class NezhaEmbeddings(nn.Module):
    """Construct the embeddings from word and token_type embeddings."""
    def __init__(self, config):
        """
        Initialize the NezhaEmbeddings class.

        Args:
            self: Instance of the NezhaEmbeddings class.
            config (object):
                Configuration object containing parameters for initializing embeddings.

                - vocab_size (int): Size of the vocabulary.
                - hidden_size (int): Size of the hidden layer.
                - pad_token_id (int): Index of the padding token in the vocabulary.
                - type_vocab_size (int): Size of the type vocabulary.
                - layer_norm_eps (float): Epsilon value for layer normalization.
                - hidden_dropout_prob (float): Probability of dropout.
                - max_position_embeddings (int): Maximum number of position embeddings.

        Returns:
            None.

        Raises:
            ValueError: If any of the configuration parameters are invalid.
            AttributeError: If there are issues with attribute assignments.
            RuntimeError: If there are runtime errors during initialization.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.token_type_ids = ops.zeros((1, config.max_position_embeddings), dtype=mindspore.int64)

    def forward(self, input_ids = None, token_type_ids = None, inputs_embeds = None):
        """
        This method forwards Nezha embeddings based on the input_ids, token_type_ids, and inputs_embeds.

        Args:
            self: The instance of the class.
            input_ids (Tensor, optional): The input tensor representing the tokenized input sequence.
                Default is None.
            token_type_ids (Tensor, optional): The input tensor representing the type of each token in the input sequence.
                Default is None.
            inputs_embeds (Tensor, optional): The input tensor containing precomputed embeddings for the input sequence.
                Default is None.

        Returns:
            None.

        Raises:
            ValueError: If input_ids and inputs_embeds have incompatible shapes.
            ValueError: If token_type_ids and input_shape have incompatible shapes.
            ValueError: If token_type_ids and buffered_token_type_ids_expanded have incompatible shapes.
            TypeError: If input_ids, token_type_ids, or inputs_embeds are not of type Tensor.
            TypeError: If token_type_ids or input_ids are not of type Tensor.
            TypeError: If token_type_ids or buffered_token_type_ids_expanded are not of type Tensor.
            TypeError: If input_shape is not of type tuple.
            TypeError: If seq_length is not of type int.
            RuntimeError: If self.token_type_embeddings or self.LayerNorm encounters a runtime error.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = ops.broadcast_to(buffered_token_type_ids,
                                                                    (input_shape[0], seq_length))
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class NezhaSelfAttention(nn.Module):
    """Self attention layer for NEZHA"""
    def __init__(self, config):
        '''
        This method initializes the NezhaSelfAttention class.

        Args:
            self: The instance of the class.
            config: An object containing the configuration parameters for the self-attention mechanism.
                It should include the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - num_attention_heads (int): The number of attention heads.
                - attention_probs_dropout_prob (float): The dropout probability for the attention probabilities.
                - max_position_embeddings (int): The maximum number of positions for positional encodings.
                - max_relative_position (int): The maximum relative position for the relative positions encoding.
                - is_decoder (bool): Indicates whether the self-attention mechanism is used as part of a decoder.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not a multiple of the number of attention heads.
        '''
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
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
        self.relative_positions_encoding = NezhaRelativePositionsEncoding(
            length=config.max_position_embeddings,
            depth=self.attention_head_size,
            max_relative_position=config.max_relative_position,
        )
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, input_x: Tensor) -> Tensor:
        """transpose for scores"""
        new_x_shape = input_x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        input_x = input_x.view(tuple(new_x_shape))
        return input_x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask = None, head_mask = None,
                  encoder_hidden_states = None, encoder_attention_mask = None,
                  past_key_value = None, output_attentions = False):
        """
        This method 'forward' is defined within the class 'NezhaSelfAttention' and is responsible for performing
        self-attention computations. It takes the following parameters:

        Args:
            self: The instance of the class.
            hidden_states: Tensor, required. The input tensor containing the hidden states for the
                self-attention mechanism.
            attention_mask: Tensor, optional. A 2D tensor providing the attention mask to be applied during
                self-attention computation. Default is None.
            head_mask: Tensor, optional. A 2D tensor representing the head mask for controlling which heads are
                active during self-attention. Default is None.
            encoder_hidden_states: Tensor, optional. The hidden states from the encoder if this is a cross-attention
                operation. Default is None.
            encoder_attention_mask: Tensor, optional. A 2D tensor providing the attention mask for encoder_hidden_states.
                Default is None.
            past_key_value: Tuple of Tensors, optional. The previous key and value tensors from the past self-attention
                computation. Default is None.
            output_attentions: Bool, optional. Flag indicating whether to output attention scores. Default is False.

        Returns:
            Tuple of Tensors or Tuple of (Tensor, Tensor, Tuple of Tensors):
                The output of the self-attention mechanism. If output_attentions is True, returns a tuple containing
                the context_layer and attention_probs. If self.is_decoder is True, the output also includes the
                past_key_value.

        Raises:
            ValueError: If the dimensions or types of the input tensors are incompatible.
            RuntimeError: If any runtime error occurs during the self-attention computation.
            AssertionError: If the conditions for past_key_value are not met.
        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
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

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        batch_size, num_attention_heads, from_seq_length, to_seq_length = attention_scores.shape
        relations_keys = self.relative_positions_encoding(to_seq_length)
        query_layer_t = query_layer.permute(2, 0, 1, 3)

        query_layer_r = query_layer_t.view(
            from_seq_length, batch_size * num_attention_heads, self.attention_head_size
        )
        key_position_scores = ops.matmul(query_layer_r, relations_keys.permute(0, 2, 1))
        key_position_scores_r = key_position_scores.view(
            from_seq_length, batch_size, num_attention_heads, from_seq_length
        )
        key_position_scores_r_t = key_position_scores_r.permute(1, 2, 0, 3)
        attention_scores = attention_scores + key_position_scores_r_t
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in NezhaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = ops.matmul(attention_probs, value_layer)
        relations_values = self.relative_positions_encoding(to_seq_length)
        attention_probs_t = attention_probs.permute(2, 0, 1, 3)
        attentions_probs_r = attention_probs_t.view(
            from_seq_length, batch_size * num_attention_heads, to_seq_length
        )
        value_position_scores = ops.matmul(attentions_probs_r, relations_values)
        value_position_scores_r = value_position_scores.view(
            from_seq_length, batch_size, num_attention_heads, self.attention_head_size
        )
        value_position_scores_r_t = value_position_scores_r.permute(1, 2, 0, 3)
        context_layer = context_layer + value_position_scores_r_t
        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class NezhaSelfOutput(nn.Module):
    """NezhaSelfOutput"""
    def __init__(self, config):
        """
        Initializes a new instance of the NezhaSelfOutput class.

        Args:
            self (NezhaSelfOutput): The object instance.
            config:
                A configuration object that contains the settings for the NezhaSelfOutput class.

                - hidden_size (int): The size of the hidden state.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden state.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Constructs the self-attention output of the Nezha model.

        Args:
            self (NezhaSelfOutput): An instance of the NezhaSelfOutput class.
            hidden_states (torch.Tensor):
                A tensor representing the hidden states.

                - Shape: (batch_size, sequence_length, hidden_size).
                - Purpose: The hidden states of the previous layer in the Nezha model.
                - Restrictions: None.
            input_tensor (torch.Tensor):
                A tensor representing the input.

                - Shape: (batch_size, sequence_length, hidden_size).
                - Purpose: The input tensor to be added to the hidden states.
                - Restrictions: None.

        Returns:
            torch.Tensor:
                A tensor representing the forwarded self-attention output.

                - Shape: (batch_size, sequence_length, hidden_size).
                - Purpose: The self-attention output of the Nezha model.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NezhaAttention(nn.Module):
    """Nezha Attention"""
    def __init__(self, config):
        """
        Initializes an instance of the NezhaAttention class.

        Args:
            self (object): The instance of the class itself.
            config (object): The configuration object that contains parameters for attention mechanism setup.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = NezhaSelfAttention(config)
        self.output = NezhaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """Prune heads"""
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        #Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask = None,
                  head_mask = None, encoder_hidden_states = None,
                  encoder_attention_mask = None, past_key_value = None,
                  output_attentions = False):
        """
        Constructs the attention mechanism for the Nezha model.

        Args:
            self (NezhaAttention): The instance of NezhaAttention class.
            hidden_states (torch.Tensor): The input hidden states of the model.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor, optional): The attention mask tensor.
                If provided, it should be a 2D tensor of shape (batch_size, sequence_length), where 1 indicates a token
                that should be attended to and 0 indicates a token that should not be attended to. Defaults to None.
            head_mask (torch.Tensor, optional): The head mask tensor.
                If provided, it should be a 1D tensor of shape (num_heads,), where 1 indicates a head that should be
                masked and 0 indicates a head that should not be masked. Defaults to None.
            encoder_hidden_states (torch.Tensor, optional): The hidden states of the encoder.
                Shape: (batch_size, sequence_length, hidden_size). Defaults to None.
            encoder_attention_mask (torch.Tensor, optional): The attention mask tensor for the encoder.
                If provided, it should be a 2D tensor of shape (batch_size, sequence_length), where 1 indicates a token
                that should be attended to and 0 indicates a token that should not be attended to. Defaults to None.
            past_key_value (tuple, optional): The cached key-value pairs of the previous time steps. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            tuple:
                A tuple containing:

                - attention_output (torch.Tensor): The output of the attention mechanism.
                Shape: (batch_size, sequence_length, hidden_size).
                - self_outputs (tuple): A tuple containing the outputs of the self-attention mechanism.
                - output_attentions (torch.Tensor, optional): The attention weights tensor.
                Shape: (num_attention_heads, batch_size, sequence_length, sequence_length).
                Only returned if `output_attentions` is set to True.

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


class NezhaIntermediate(nn.Module):
    """Nezha Intermediate"""
    def __init__(self, config):
        """
        Initializes a NezhaIntermediate object with the provided configuration.

        Args:
            self: The object instance.
            config:
                An object containing configuration settings.

                - Type: Any valid object.
                - Purpose: The configuration settings for the NezhaIntermediate object.
                - Restrictions: None.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided.
            ValueError: If the 'config.hidden_size' or 'config.intermediate_size' are invalid.
            KeyError: If the 'config.hidden_act' value is not found in the ACT2FN dictionary.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        """
        This method forwards the intermediate hidden states for the NezhaIntermediate class.

        Args:
            self (NezhaIntermediate): The instance of the NezhaIntermediate class.
            hidden_states (tensor): The input hidden states to be processed.

        Returns:
            tensor: The processed intermediate hidden states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class NezhaOutput(nn.Module):
    """Nezha Output"""
    def __init__(self, config):
        """
        Initializes a new instance of the NezhaOutput class.

        Args:
            self: The object instance.
            config:
                An instance of the configuration class containing the model's configuration parameters.

                - Type: Any
                - Purpose: Specifies the configuration settings for the NezhaOutput instance.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Constructs the output of the Nezha model by applying a series of operations on the hidden states and input tensor.

        Args:
            self (NezhaOutput): An instance of the NezhaOutput class.
            hidden_states (Tensor): The hidden states of the model.
                It should have dimensions (batch_size, sequence_length, hidden_size).
            input_tensor (Tensor): The input tensor to be added to the hidden states.
                It should have the same dimensions as hidden_states.

        Returns:
            None.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NezhaLayer(nn.Module):
    """Nezha Layer"""
    def __init__(self, config):
        """
        Initializes a NezhaLayer object with the provided configuration.

        Args:
            self: The instance of the NezhaLayer class.
            config:
                An object containing configuration parameters for the NezhaLayer.

                - Type: Config object
                - Purpose: Specifies the configuration settings for the NezhaLayer.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            ValueError: Raised if the cross attention is added but the model is not used as a decoder model.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NezhaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = NezhaAttention(config)
        self.intermediate = NezhaIntermediate(config)
        self.output = NezhaOutput(config)

    def forward(self, hidden_states, attention_mask = None,
                  head_mask = None, encoder_hidden_states = None,
                  encoder_attention_mask = None, past_key_value = None,
                  output_attentions = False):
        """
        Method: forward

        Description:
            Constructs the NezhaLayer by performing self-attention and potentially cross-attention operations based on
            the provided parameters.

        Args:
            self: The object instance.
            hidden_states (Tensor): The input hidden states for the layer.
            attention_mask (Tensor, optional): Mask to prevent attention to certain positions.
            head_mask (Tensor, optional): Mask to prevent attention to certain heads.
            encoder_hidden_states (Tensor, optional): Hidden states of the encoder if cross-attention is needed.
            encoder_attention_mask (Tensor, optional): Mask for encoder attention.
            past_key_value (Tuple, optional): Tuple containing past key and value tensors for optimization.
            output_attentions (bool): Flag to indicate whether to output attentions.

        Returns:
            None.

        Raises:
            ValueError: If `encoder_hidden_states` are provided but cross-attention layers are not instantiated
                in the model.
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


class NezhaEncoder(nn.Module):
    """Nezha Encoder"""
    def __init__(self, config):
        """
        Initializes a NezhaEncoder instance with the provided configuration.

        Args:
            self (NezhaEncoder): The NezhaEncoder instance itself.
            config (dict): A dictionary containing the configuration parameters for the NezhaEncoder.
                This dictionary should include the following keys:

                - num_hidden_layers (int): The number of hidden layers in the NezhaEncoder configuration.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the configuration dictionary is missing required keys or if the values are invalid.
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([NezhaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask = None,
                  head_mask = None, encoder_hidden_states = None,
                  encoder_attention_mask = None, past_key_values = None,
                  use_cache = None, output_attentions = False,
                  output_hidden_states = False):
        """
        Constructs the NezhaEncoder.

        Args:
            self: The NezhaEncoder object.
            hidden_states (Tensor): The input hidden states of the encoder.
            attention_mask (Tensor, optional): An attention mask tensor. Defaults to None.
            head_mask (List[Tensor], optional): A list of attention mask tensors for each layer. Defaults to None.
            encoder_hidden_states (Tensor, optional): The hidden states of the encoder. Defaults to None.
            encoder_attention_mask (Tensor, optional): An attention mask tensor for the encoder. Defaults to None.
            past_key_values (Tuple[Tensor], optional): A tuple of key-value tensors from previous decoder outputs.
                Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to None.
            output_attentions (bool, optional): Whether to output attention tensors. Defaults to False.
            output_hidden_states (bool, optional): Whether to output hidden states of each layer. Defaults to False.

        Returns:
            Tuple[Tensor, Tuple[Tensor], Optional[Tuple[Tensor]], Optional[Tuple[Tensor]], Optional[Tuple[Tensor]]]:

                - hidden_states (Tensor): The output hidden states of the encoder.
                - next_decoder_cache (Tuple[Tensor]): The cache for the next decoder.
                - all_hidden_states (Optional[Tuple[Tensor]]): The hidden states of each layer.
                None if `output_hidden_states` is False.
                - all_self_attentions (Optional[Tuple[Tensor]]): The attention tensors of each layer.
                None if `output_attentions` is False.
                - all_cross_attentions (Optional[Tuple[Tensor]]): The cross-attention tensors of each layer.
                None if `output_attentions` is False or `self.config.add_cross_attention` is False.

        Raises:
            None.
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

            # TODO
            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs, past_key_value, output_attentions)

            #         return custom_forward
            # layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(layer_module),
            #         hidden_states,
            #         attention_mask,
            #         layer_head_mask,
            #         encoder_hidden_states,
            #         encoder_attention_mask,
            #     )
            # else:
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


class NezhaPooler(nn.Module):
    """Nezha Pooler"""
    def __init__(self, config):
        """
        Initializes the NezhaPooler class.

        Args:
            self (NezhaPooler): The instance of the NezhaPooler class.
            config (object): An object containing configuration parameters for the NezhaPooler.
                This parameter is used to configure the dense layer and activation function.
                It should have a property 'hidden_size' to specify the size of the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        Constructs the pooled output from the given hidden states.

        Args:
            self (NezhaPooler): An instance of the NezhaPooler class.
            hidden_states (Tensor): A tensor containing the hidden states.
                Shape: (batch_size, sequence_length, hidden_size)

        Returns:
            Tensor: A tensor representing the pooled output.
                Shape: (batch_size, hidden_size)

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class NezhaPredictionHeadTransform(nn.Module):
    """Nezha Predicton Head Transform"""
    def __init__(self, config):
        """
        Initializes the NezhaPredictionHeadTransform class.

        Args:
            self: The object instance.
            config:
                An instance of the configuration class that contains the parameters for the head transformation.

                - Type: Any
                - Purpose: Specifies the configuration for the head transformation.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        Constructs the NezhaPredictionHeadTransform.

        Args:
            self: An instance of the NezhaPredictionHeadTransform class.
            hidden_states (tensor): The hidden states to be transformed.

        Returns:
            None

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class NezhaLMPredictionHead(nn.Module):
    """Nezha LMLMPredictionHead"""
    def __init__(self, config):
        """
        Initializes the NezhaLMPredictionHead class.

        Args:
            self: The object instance.
            config:
                A configuration object that holds various parameters for the model.

                - Type: Any valid configuration object.
                - Purpose: Specifies the configuration settings for the NezhaLMPredictionHead.
                - Restrictions: None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.transform = NezhaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        """
        Constructs the prediction head for Nezha Language Model.

        Args:
            self (NezhaLMPredictionHead): The instance of NezhaLMPredictionHead class.
            hidden_states (Tensor): The hidden states to be processed for prediction.

        Returns:
            hidden_states: The forwarded prediction head for Nezha Language Model.

        Raises:
            None.
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class NezhaOnlyMLMHead(nn.Module):
    """Nezha OnlyMLMHead"""
    def __init__(self, config):
        """Initializes a new instance of the NezhaOnlyMLMHead class.

        Args:
            self: The instance of the class.
            config: An object of type 'config' containing the configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = NezhaLMPredictionHead(config)

    def forward(self, sequence_output):
        """
        Constructs the Masked Language Model (MLM) head for the Nezha model.

        Args:
            self (NezhaOnlyMLMHead): An instance of the NezhaOnlyMLMHead class.
            sequence_output (torch.Tensor): The output tensor of the Nezha model's encoder.
                Shape: (batch_size, sequence_length, hidden_size).

        Returns:
            None: This method modifies the internal state of the NezhaOnlyMLMHead instance.

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class NezhaOnlyNSPHead(nn.Module):
    """Nezha OnlyNSPHead"""
    def __init__(self, config):
        """
        Initializes a new instance of the NezhaOnlyNSPHead class.

        Args:
            self: The instance of the NezhaOnlyNSPHead class.
            config: An instance of configuration class containing the hidden size parameter.
                It specifies the configuration settings for the NezhaOnlyNSPHead class.
                It is expected to have a hidden_size attribute, which represents the size of the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required hidden_size attribute.
        """
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        """
        Constructs the NSP (Next Sentence Prediction) head for the Nezha model.

        Args:
            self (NezhaOnlyNSPHead): An instance of the NezhaOnlyNSPHead class.
            pooled_output (torch.Tensor): The pooled output tensor of shape (batch_size, hidden_size).
                The pooled output is typically obtained by applying pooling operations (e.g., mean pooling, max pooling)
                over the sequence-level representations of the input tokens. It serves as the input to the NSP head.

        Returns:
            None.

        Raises:
            None.

        Note:
            The NSP head is responsible for predicting whether two input sentences are consecutive or not.
            It takes the pooled output tensor from the Nezha model and computes the sequence relationship score.
            The sequence relationship score is used to determine if the two input sentences are consecutive or not.
        """
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class NezhaPreTrainingHeads(nn.Module):
    """Nezha PreTrainingHeads"""
    def __init__(self, config):
        """
        Initializes the NezhaPreTrainingHeads class.

        Args:
            self (NezhaPreTrainingHeads): The instance of the NezhaPreTrainingHeads class.
            config: Configuration object containing settings for the NezhaPreTrainingHeads.
                It is expected to be a dictionary-like object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = NezhaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        """
        This method forwards Nezha pre-training heads.

        Args:
            self (object): The instance of the NezhaPreTrainingHeads class.
            sequence_output (object): The output of the sequence.
            pooled_output (object): The pooled output.

        Returns:
            tuple:
                A tuple containing 'prediction_scores' and 'seq_relationship_score'.

                - prediction_scores (object): The prediction scores based on the sequence_output.
                - seq_relationship_score (object): The sequence relationship score based on the pooled_output.

        Raises:
            None
        """
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class NezhaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = NezhaConfig
    base_model_prefix = "nezha"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [r"positions_encoding"]

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            cell.weight.assign_value(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(self.config.initializer_range),
                                                 cell.weight.shape,
                                                 cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.assign_value(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.assign_value(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.assign_value(initializer('zeros', cell.bias.shape, cell.bias.dtype))

    # TODO
    def get_input_embeddings(self):
        """
        Method to get the input embeddings for NezhaPreTrainedModel.

        Args:
            self: NezhaPreTrainedModel object. The instance of the NezhaPreTrainedModel class.

        Returns:
            None.

        Raises:
            None.
        """

    # TODO
    def get_position_embeddings(self):
        """
        This method retrieves the position embeddings for NezhaPreTrainedModel.

        Args:
            self: The instance of the NezhaPreTrainedModel class.

        Returns:
            None.

        Raises:
            None.
        """

    # TODO
    def resize_position_embeddings(self):
        """
        Method to resize the position embeddings of the NezhaPreTrainedModel.

        Args:
            self: NezhaPreTrainedModel, The instance of the NezhaPreTrainedModel class.
                This parameter is used to access and modify the position embeddings of the model.

        Returns:
            None: This method does not return any value. It modifies the position embeddings in place.

        Raises:
            None.
        """

    # TODO
    def set_input_embeddings(self):
        """
        This method sets the input embeddings for the NezhaPreTrainedModel.

        Args:
            self:
                The instance of the NezhaPreTrainedModel class.

                - Type: NezhaPreTrainedModel
                - Purpose: To access and modify the attributes and methods of the NezhaPreTrainedModel instance.

        Returns:
            None.

        Raises:
            None.
        """

    # TODO
    def post_init(self):
        """
        This method is part of the NezhaPreTrainedModel class and is called 'post_init'.

        Args:
            self: An instance of the NezhaPreTrainedModel class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method is a placeholder and does not perform any specific operations.
            It is called automatically after the initialization of an instance of the NezhaPreTrainedModel class.
            It can be overridden in child classes to add custom initialization logic or perform additional setup steps.

        Note that the 'self' parameter is automatically passed to the method and does not need to be provided explicitly
        when calling the method.
        """


class NezhaModel(NezhaPreTrainedModel):
    """Nezha Model"""
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the NezhaModel class.

        Args:
            self: The instance of the NezhaModel class.
            config: An instance of the configuration for the NezhaModel. It is used to configure the model's behavior.
            add_pooling_layer (bool): A boolean flag indicating whether to add a pooling layer to the model.
                Default is True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = NezhaEmbeddings(config)
        self.encoder = NezhaEncoder(config)
        self.pooler = NezhaPooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the NezhaModel.

        Args:
            self (NezhaModel): The instance of NezhaModel.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the NezhaModel.

        Args:
            self (NezhaModel): The instance of the NezhaModel class.
            value: The input embeddings to be set. It should be of type 'torch.Tensor' or any tensor-like object.

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

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None,
                  inputs_embeds = None, encoder_hidden_states = None,
                  encoder_attention_mask = None, past_key_values = None,
                  use_cache = None, output_attentions = None,
                  output_hidden_states = None):
        """
        This method forwards the NezhaModel by processing input data through the model's encoder and embeddings.

        Args:
            self: The instance of the NezhaModel class.
            input_ids (Tensor, optional): The input token IDs. Default is None.
            attention_mask (Tensor, optional): The attention mask for the input. Default is None.
            token_type_ids (Tensor, optional): The token type IDs for the input. Default is None.
            head_mask (Tensor, optional): The head mask for the model's multi-head attention layers. Default is None.
            inputs_embeds (Tensor, optional): The embedded input tokens. Default is None.
            encoder_hidden_states (Tensor, optional): The hidden states from the encoder. Default is None.
            encoder_attention_mask (Tensor, optional): The attention mask for the encoder. Default is None.
            past_key_values (Tuple, optional): Cached key-value states from previous iterations. Default is None.
            use_cache (bool, optional): Flag indicating whether to use caching. Default is None.
            output_attentions (bool, optional): Flag indicating whether to output attentions. Default is None.
            output_hidden_states (bool, optional): Flag indicating whether to output hidden states. Default is None.

        Returns:
            Tuple: A tuple containing the sequence output, pooled output, and any additional encoder outputs.

        Raises:
            ValueError: Raised when both input_ids and inputs_embeds are provided simultaneously.
            ValueError: Raised when neither input_ids nor inputs_embeds are specified.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = ops.ones(((batch_size, seq_length + past_key_values_length)))

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = ops.broadcast_to(buffered_token_type_ids, (batch_size, seq_length))
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
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
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
            output_hidden_states=output_hidden_states
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class NezhaForPreTraining(NezhaPreTrainedModel):
    """NezhaForPreTraining"""
    _keys_to_ignore_on_load_missing = ["cls.predictions.decoder"]
    def __init__(self, config):
        """
        Initializes an instance of the 'NezhaForPreTraining' class.

        Args:
            self: The instance of the class.
            config:
                A configuration object containing various settings for the model.

                - Type: Config object
                - Purpose: Specifies the model configuration.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.nezha = NezhaModel(config)
        self.cls = NezhaPreTrainingHeads(config)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_output_embeddings(self):
        """get output embeddings"""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """set output embeddings"""
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None, inputs_embeds = None,
                  labels = None, next_sentence_label = None,
                  output_attentions = None, output_hidden_states = None):
        """
        Constructs the Nezha model for pre-training.

        Args:
            self (NezhaForPreTraining): The instance of the NezhaForPreTraining class.
            input_ids (torch.Tensor, optional): The input sequence tensor. Default: None.
            attention_mask (torch.Tensor, optional): The attention mask tensor. Default: None.
            token_type_ids (torch.Tensor, optional): The token type ids tensor. Default: None.
            head_mask (torch.Tensor, optional): The head mask tensor. Default: None.
            inputs_embeds (torch.Tensor, optional): The input embeddings tensor. Default: None.
            labels (torch.Tensor, optional): The labels tensor. Default: None.
            next_sentence_label (torch.Tensor, optional): The next sentence label tensor. Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.

        Returns:
            tuple or torch.Tensor: A tuple of output tensors or a single tensor representing the total loss.

        Raises:
            None
        """
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        output = (prediction_scores, seq_relationship_score) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


class NezhaForMaskedLM(NezhaPreTrainedModel):
    """NezhaForMaskedLM"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"cls.predictions.decoder", r"positions_encoding"]

    def __init__(self, config):
        """
        Initializes a new NezhaForMaskedLM instance.

        Args:
            self (object): The NezhaForMaskedLM instance itself.
            config (object): An instance of the configuration class containing the model configuration settings.
                It is used to customize the behavior of the NezhaForMaskedLM model.
                Must have the property 'is_decoder' to determine if the model is a decoder.
                If 'is_decoder' is True, a warning will be logged regarding the bidirectional
                self-attention configuration.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `NezhaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.nezha = NezhaModel(config, add_pooling_layer=False)
        self.cls = NezhaOnlyMLMHead(config)

    def get_output_embeddings(self):
        """get output embeddings"""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """set output embeddings"""
        self.cls.predictions.decoder = new_embeddings

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None, inputs_embeds = None,
                  encoder_hidden_states = None, encoder_attention_mask = None,
                  labels = None, output_attentions = None, output_hidden_states = None):
        """
        Constructs the Nezha model for masked language modeling (MLM).

        Args:
            self (NezhaForMaskedLM): The instance of the NezhaForMaskedLM class.
            input_ids (torch.Tensor, optional): The input tensor containing the tokenized input sequence IDs.
                Default: None.
            attention_mask (torch.Tensor, optional): The attention mask tensor to indicate which tokens
                should be attended to. Default: None.
            token_type_ids (torch.Tensor, optional): The token type IDs tensor to distinguish different parts
                of the input. Default: None.
            head_mask (torch.Tensor, optional): The head mask tensor to mask specific attention heads. Default: None.
            inputs_embeds (torch.Tensor, optional): The embedded inputs tensor. Default: None.
            encoder_hidden_states (torch.Tensor, optional): The hidden states of the encoder. Default: None.
            encoder_attention_mask (torch.Tensor, optional): The attention mask for the encoder. Default: None.
            labels (torch.Tensor, optional): The tensor containing the labels for the masked language modeling task.
                Default: None.
            output_attentions (bool, optional): Whether to output attentions. Default: None.
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None.

        Returns:
            tuple:
                A tuple containing the masked language modeling loss (if labels are provided)
                and the output of the model.

                - masked_lm_loss (torch.Tensor): The masked language modeling loss.
                - prediction_scores (torch.Tensor): The predicted scores for each token.
                - outputs[2:] (tuple): Additional outputs from the model. (output_attentions, output_hidden_states, ...)

        Raises:
            None.
        """
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,) + outputs[2:]
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None):
        """prepare inputs for generation"""
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = ops.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = ops.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=mindspore.int64
        )
        input_ids = ops.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class NezhaForNextSentencePrediction(NezhaPreTrainedModel):
    """NezhaForNextSentencePrediction"""
    def __init__(self, config):
        """
        Initializes an instance of the NezhaForNextSentencePrediction class.

        Args:
            self: The instance of the class.
            config: An instance of the configuration class containing the model configuration settings.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        self.nezha = NezhaModel(config)
        self.cls = NezhaOnlyNSPHead(config)

    def forward(self, input_ids = None, attention_mask = None,
        token_type_ids = None, head_mask = None, inputs_embeds = None,
        labels = None, output_attentions = None, output_hidden_states = None, **kwargs):
        """
        Constructs the Nezha model for next sentence prediction.

        Args:
            self (NezhaForNextSentencePrediction): The instance of the NezhaForNextSentencePrediction class.
            input_ids (torch.Tensor, optional): The input tensor of shape (batch_size, sequence_length) containing
                the input sequence indices. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length)
                containing the attention mask values. Defaults to None.
            token_type_ids (torch.Tensor, optional): The token type tensor of shape (batch_size, sequence_length)
                containing the token type indices. Defaults to None.
            head_mask (torch.Tensor, optional): The head mask tensor of shape
                (batch_size, num_heads, sequence_length, sequence_length) containing the head mask values.
                Defaults to None.
            inputs_embeds (torch.Tensor, optional): The embedded inputs tensor of shape
                (batch_size, sequence_length, embedding_size) containing the embedded input sequence. Defaults to None.
            labels (torch.Tensor, optional): The labels tensor of shape (batch_size) containing the next sentence labels.
                Defaults to None.
            output_attentions (bool, optional): Whether to output attention weights. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.

        Returns:
            tuple:
                A tuple containing the next sentence loss (if labels are provided) and the outputs of the Nezha model.

                - next_sentence_loss (torch.Tensor, optional): The loss tensor of shape (batch_size) representing
                the next sentence prediction loss. Defaults to None.
                - seq_relationship_scores (torch.Tensor): The tensor of shape (batch_size, 2) containing the next
                sentence prediction scores.
                - hidden_states (tuple, optional): A tuple of hidden states (torch.Tensor) of shape
                (batch_size, sequence_length, hidden_size) from all layers. Defaults to None.
                - attentions (tuple, optional): A tuple of attention weights (torch.Tensor) of shape
                (batch_size, num_heads, sequence_length, sequence_length) from all layers. Defaults to None.

        Raises:
            TypeError: If any of the input arguments are not of the expected type.
            ValueError: If the input tensors do not have the correct shape.

        """
        if "next_sentence_label" in kwargs:
            #warnings.warn(
            #     "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
            #     " `labels` instead.",
            #     FutureWarning,
            # )
            labels = kwargs.pop("next_sentence_label")

        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        output = (seq_relationship_scores,) + outputs[2:]
        return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output


class NezhaForSequenceClassification(NezhaPreTrainedModel):
    """NezhaForSequenceClassification"""
    def __init__(self, config):
        """
        Initializes a new instance of the NezhaForSequenceClassification class.

        Args:
            self (NezhaForSequenceClassification): An instance of the NezhaForSequenceClassification class.
            config (NezhaConfig): The configuration class instance specifying the model's hyperparameters.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.nezha = NezhaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None, inputs_embeds = None,
                  labels = None, output_attentions = None, output_hidden_states = None):
        '''
        This method forwards a Nezha model for sequence classification.

        Args:
            self (object): The instance of the NezhaForSequenceClassification class.
            input_ids (list, optional): A list of tokenized input sequence IDs. Defaults to None.
            attention_mask (list, optional): A list of attention masks indicating which tokens should be attended to.
                Defaults to None.
            token_type_ids (list, optional): A list of token type IDs to indicate which parts of the input belong to
                the first sequence and which belong to the second sequence. Defaults to None.
            head_mask (list, optional): A list of masks for attention heads. Defaults to None.
            inputs_embeds (list, optional): A list of input embeddings. Defaults to None.
            labels (list, optional): A list of target labels for the input sequence. Defaults to None.
            output_attentions (bool, optional): A boolean flag indicating whether to return the attentions tensor.
                Defaults to None.
            output_hidden_states (bool, optional): A boolean flag indicating whether to return the hidden states tensor.
                Defaults to None.

        Returns:
            tuple: A tuple containing the loss value and the output logits.
                If no loss is calculated, only the logits are returned.

        Raises:
            ValueError: If the problem type is not recognized.
            RuntimeError: If the number of labels is not compatible with the specified problem type.
            TypeError: If the labels data type is not supported for the specified problem type.
            AssertionError: If the loss function encounters an unexpected condition.
        '''
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
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

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class NezhaForMultipleChoice(NezhaPreTrainedModel):
    """NezhaForMultipleChoice"""
    def __init__(self, config):
        """
        Initialize the NezhaForMultipleChoice model with the given configuration.

        Args:
            self (NezhaForMultipleChoice): The NezhaForMultipleChoice instance.
            config (NezhaConfig): The configuration object containing various hyperparameters for the model.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.nezha = NezhaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None, inputs_embeds = None,
                  labels = None, output_attentions = None, output_hidden_states = None):
        r"""
        Args:
            labels (`mindspore.Tensor[int64]` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-2])
            if inputs_embeds is not None
            else None
        )

        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class NezhaForTokenClassification(NezhaPreTrainedModel):
    """NezhaForTokenClassification"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        Initializes a new instance of the NezhaForTokenClassification class.

        Args:
            self: The object itself.
            config:
                An instance of the NezhaConfig class containing the model configuration settings.

                - Type: NezhaConfig
                - Purpose: Specifies the configuration for the Nezha model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.nezha = NezhaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None, inputs_embeds = None,
                  labels = None, output_attentions = None, output_hidden_states = None):
        r"""
        Args:
            labels (`mindspore.Tensor[int64]` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class NezhaForQuestionAnswering(NezhaPreTrainedModel):
    """NezhaForQuestionAnswering"""
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        """
        This method initializes an instance of the NezhaForQuestionAnswering class.

        Args:
            self (NezhaForQuestionAnswering): The instance of the NezhaForQuestionAnswering class.
            config: An instance of the NezhaConfig class containing the model configuration.

        Returns:
            None.
        
        Raises:
            TypeError: If the config parameter is not of type NezhaConfig.
            ValueError: If the config.num_labels is not defined or is not a positive integer.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.nezha = NezhaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids = None, attention_mask = None,
                  token_type_ids = None, head_mask = None, inputs_embeds = None,
                  start_positions = None, end_positions = None, output_attentions = None,
                  output_hidden_states = None):
        r"""
        Args:
            start_positions (`mindspore.Tensor[int64]` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`mindspore.Tensor[int64]` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        outputs = self.nezha(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
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

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output
