# coding=utf-8
# Copyright 2018 The Microsoft Research Asia LayoutLM Team Authors and the HuggingFace Inc. team.
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
""" MindSpore LayoutLM model."""
import math
from typing import Optional, Tuple, Union
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional import finfo
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from .configuration_layoutlm import LayoutLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutLMConfig"
# _CHECKPOINT_FOR_DOC = "microsoft/layoutlm-base-uncased"

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlm-base-uncased",
    "layoutlm-large-uncased",
]


LayoutLMLayerNorm = nn.LayerNorm


class LayoutLMEmbeddings(nn.Cell):
    """forward the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        """
        Args:
            self (LayoutLMEmbeddings): The instance of the LayoutLMEmbeddings class.
            config (object): An object containing configuration parameters,
                including vocab_size, hidden_size, max_position_embeddings,
                max_2d_position_embeddings, type_vocab_size, pad_token_id, layer_norm_eps, and hidden_dropout_prob. 
        
        Returns:
            None.
        
        Raises:
            None
        """
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayoutLMLayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.position_ids = ops.arange(config.max_position_embeddings).broadcast_to((1, -1))

    def construct(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        """
        Constructs the LayoutLM embeddings.
        
        Args:
            self (LayoutLMEmbeddings): An instance of the LayoutLMEmbeddings class.
            input_ids (Tensor, optional): The input tensor of token indices. Defaults to None.
            bbox (Tensor, optional): The bounding box tensor. Defaults to None.
            token_type_ids (Tensor, optional): The input tensor of token type indices. Defaults to None.
            position_ids (Tensor, optional): The input tensor of position indices. Defaults to None.
            inputs_embeds (Tensor, optional): The input tensor of embeddings. Defaults to None.
        
        Returns:
            Tensor: The computed embeddings.
        
        Raises:
            IndexError: If the `bbox` coordinate values are not within the 0-1000 range.
        
        Note:
            The method calculates the embeddings by adding various embeddings such as words, position, 
            bounding box, token type, etc. It also performs layer normalization and dropout on the embeddings.
        """
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        words_embeddings = inputs_embeds
        position_embeddings = self.position_embeddings(position_ids)

        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox`coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (
            words_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LayoutLMSelfAttention(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->LayoutLM"""
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes a LayoutLMSelfAttention instance with the provided configuration and optional position embedding type.
        
        Args:
            self: The instance of LayoutLMSelfAttention.
            config: An object containing the configuration parameters for the self-attention layer.
                Expected attributes:

                - hidden_size (int): The size of the hidden layer.
                - num_attention_heads (int): The number of attention heads.
                - embedding_size (int, optional): The size of the embedding layer.
                - attention_probs_dropout_prob (float): The dropout probability for attention probabilities.
                - position_embedding_type (str, optional): The type of position embedding to use ('absolute' by default).
                - max_position_embeddings (int): The maximum number of position embeddings.
                - is_decoder (bool): Indicates if the self-attention layer is part of a decoder.
            position_embedding_type (str, optional): The type of position embedding to use (default is None).
                Accepted values: 'absolute', 'relative_key', 'relative_key_query'.

        Returns:
            None.

        Raises:
            ValueError:
                If the hidden size is not a multiple of the number of attention heads and no embedding size is provided.
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
        Transposes the input tensor `x` for calculating self-attention scores.

        Args:
            self (LayoutLMSelfAttention): The current instance of the LayoutLMSelfAttention class.
            x (mindspore.Tensor): The input tensor of shape `(batch_size, sequence_length, hidden_size)`.

        Returns:
            mindspore.Tensor:
                The transposed tensor of shape `(batch_size, num_attention_heads, sequence_length, attention_head_size)`.

        Raises:
            None

        This method transposes the input tensor `x` to prepare it for calculating self-attention scores in the
        LayoutLMSelfAttention model. The transposition is performed by reshaping the tensor to include the number of
        attention heads and the size of each attention head. The resulting tensor is then permuted to match the desired
        shape `(batch_size, num_attention_heads, sequence_length, attention_head_size)`.

        Note that this method assumes that the input tensor `x` has a rank of at least 3, where the last dimension
        represents the hidden size. The number of attention heads and the size of each attention head are obtained
        from the attributes `num_attention_heads` and `attention_head_size` of the LayoutLMSelfAttention instance,
        respectively.
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
        This method constructs the self-attention mechanism for the LayoutLMSelfAttention class.

        Args:
            self: The instance of the LayoutLMSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            attention_mask (Optional[mindspore.Tensor], optional): Mask tensor to prevent attention to certain
                positions. Defaults to None.
            head_mask (Optional[mindspore.Tensor], optional): Mask tensor to control the heads involved in the
                attention computation. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor], optional): Hidden states of the encoder in case of
                cross-attention. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor], optional): Mask tensor for encoder attention.
                Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]], optional): Cached key and value tensors from
                previous attention calculations. Defaults to None.
            output_attentions (Optional[bool], optional): Flag to indicate whether to output attentions.
                Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the context layer tensor and optionally attention
                probabilities tensor.

        Raises:
            ValueError: If the input tensor shapes are incompatible for matrix multiplication.
            RuntimeError: If there are runtime issues during tensor operations.
            TypeError: If the input types are not as expected.
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
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.swapaxes(-1, -2))

        if self.position_embedding_type in ('relative_key', 'relative_key_query'):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = mindspore.tensor(key_length - 1, dtype=mindspore.int64).view(-1, 1)
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
            # Apply the attention mask is (precomputed for all layers in LayoutLMModel forward() function)
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


class LayoutLMSelfOutput(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->LayoutLM"""
    def __init__(self, config):
        """
        Initializes the LayoutLMSelfOutput class.

        Args:
            self (object): The instance of the class itself.
            config (object):
                An object containing configuration parameters for the layout model.

                - Type: Custom class
                - Purpose: To provide configuration settings for the layout model.
                - Restrictions: Must be compatible with the defined configuration structure.

        Returns:
            None.

        Raises:
            TypeError: If the provided 'config' parameter is not of the expected type.
            ValueError: If the configuration provided is missing essential parameters.
        """
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the LayoutLMSelfOutput layer.

        Args:
            self (LayoutLMSelfOutput): The instance of the LayoutLMSelfOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor generated by the layer.
                This tensor contains the output of the dense and dropout layers.
            input_tensor (mindspore.Tensor): The input tensor to the layer.
                This tensor represents the input to the layer that needs to be added to the hidden states.

        Returns:
            mindspore.Tensor: The tensor representing the output of the LayerNorm operation.
                This tensor is the result of adding the input tensor to the normalized hidden states.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMAttention(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->LayoutLM"""
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the LayoutLMAttention class.

        Args:
            self: The instance of the class (automatically passed).
            config:
                An object containing the configuration settings.

                - Type: object
                - Purpose: Provides the configuration settings for the LayoutLMAttention class.
                - Restrictions: None

            position_embedding_type:
                The type of position embedding to use.

                - Type: Any
                - Purpose: Specifies the type of position embedding to be used in the LayoutLMAttention class.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = LayoutLMSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = LayoutLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes the attention heads in the LayoutLMAttention class.

        Args:
            self: The LayoutLMAttention instance.
            heads (list): A list of integers representing the attention heads to be pruned.

        Returns:
            None: The method modifies the LayoutLMAttention instance in-place.

        Raises:
            None.

        This method prunes the specified attention heads from the LayoutLMAttention instance.
        First, it checks if the 'heads' list is empty. If so, the method returns without making any changes.
        Otherwise, it calls the 'find_pruneable_heads_and_indices' function to identify the attention heads and
        their corresponding indices that can be pruned based on the given 'heads' list, the number of attention heads,
        attention head size, and already pruned heads stored in the instance.
        Next, it prunes the 'self.query', 'self.key', 'self.value', and 'self.output.dense' linear layers by calling
        the 'prune_linear_layer' function with the identified indices.
        After each linear layer is pruned, the number of attention heads is updated by subtracting the length of the
        'heads' list from the current number of attention heads.
        The total size of all attention heads, 'self.all_head_size', is then recalculated as the product of the
        attention head size and the updated number of attention heads.
        Finally, the 'pruned_heads' set is updated by adding the attention heads specified in the 'heads' list.
        The method does not return any value but modifies the LayoutLMAttention instance by pruning the specified
        attention heads.
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
        '''
        This method constructs the LayoutLMAttention.

        Args:
            self (LayoutLMAttention): The instance of the LayoutLMAttention class.
            hidden_states (mindspore.Tensor): The input hidden states for the attention mechanism.
            attention_mask (Optional[mindspore.Tensor]): An optional mask for the attention mechanism. Default is None.
            head_mask (Optional[mindspore.Tensor]): An optional mask for the attention heads. Default is None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional input for encoder hidden states.
                Default is None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional mask for encoder attention.
                Default is None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional input for past key value.
                Default is None.
            output_attentions (Optional[bool]): A flag to indicate whether to output attentions. Default is False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output and other optional outputs.

        Raises:
            None
        '''
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


class LayoutLMIntermediate(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertIntermediate"""
    def __init__(self, config):
        """
        Initializes an instance of the LayoutLMIntermediate class.

        Args:
            self: The current instance of the class.
            config: An object of type 'config' containing the configuration settings for the intermediate layer.
                This parameter is required and has no default value.

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
        Constructs the intermediate layer in the LayoutLM model.

        Args:
            self (LayoutLMIntermediate): An instance of the LayoutLMIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor representing hidden states.

        Returns:
            mindspore.Tensor: The output tensor after passing through the intermediate layer.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LayoutLMOutput(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM"""
    def __init__(self, config):
        """
        Initializes a new instance of the LayoutLMOutput class.

        Args:
            self (object): The instance of the class.
            config (object): An object containing configuration parameters for the LayoutLMOutput.
                This parameter is required to configure the dense layer, layer normalization, and dropout.
                It should be an instance of a class that contains the following attributes:

                - intermediate_size (int): The size of the intermediate layer.
                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided or is not an instance of the expected class.
            ValueError: If the attributes intermediate_size, hidden_size, layer_norm_eps, or hidden_dropout_prob
                are missing from the config object.
        """
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Construct method in the LayoutLMOutput class.

        Args:
            self: LayoutLMOutput instance.
            hidden_states (mindspore.Tensor): The hidden states tensor to be processed.
            input_tensor (mindspore.Tensor): The input tensor to be added to the processed hidden states.

        Returns:
            mindspore.Tensor: A tensor representing the processed hidden states with the input tensor added.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutLMLayer(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->LayoutLM"""
    def __init__(self, config):
        """
        This method initializes an instance of the LayoutLMLayer class.

        Args:
            self (LayoutLMLayer): The instance of the LayoutLMLayer class.
            config: A configuration object containing parameters for the LayoutLMLayer.

        Returns:
            None.

        Raises:
            ValueError: If the cross attention is added and the model is not used as a decoder, a ValueError is raised.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = LayoutLMAttention(config, position_embedding_type="absolute")
        self.intermediate = LayoutLMIntermediate(config)
        self.output = LayoutLMOutput(config)

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
        Constructs the LayoutLMLayer.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape
                (batch_size, seq_length) or (batch_size, seq_length, seq_length). Defaults to None.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor of shape (num_heads,) or
                (num_layers, num_heads), where num_heads and num_layers are derived from the configuration.
                Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): The encoder hidden states tensor of shape
                (batch_size, seq_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): The encoder attention mask tensor of shape
                (batch_size, seq_length) or (batch_size, seq_length, seq_length). Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key-value tensor of shape
                (2, batch_size, num_heads, past_seq_length, head_dim), where past_seq_length is the length of past
                sequence. Defaults to None.
            output_attentions (Optional[bool]): Whether to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the output tensor(s) of the layer.
                The first element is the layer output tensor of shape (batch_size, seq_length, hidden_size).
                If the layer is a decoder, the tuple also includes the present key-value tensor(s) of shape
                (2, batch_size, num_heads, seq_length, head_dim).

        Raises:
            ValueError: If `encoder_hidden_states` are passed and the cross-attention layers are not instantiated
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
        """
        Class: LayoutLMLayer

        Description:
            This class represents a layer in a layout LM model.

        Method:
            feed_forward_chunk

        Description:
            This method applies a feed-forward operation to the given attention output.

        Args:
            self (LayoutLMLayer): The instance of the LayoutLMLayer class.
            attention_output (Tensor): The attention output tensor to be processed by the feed-forward operation.

        Returns:
            layer_output (Tensor): The output tensor after applying the feed-forward operation.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LayoutLMEncoder(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->LayoutLM"""
    def __init__(self, config):
        """Initializes an instance of the LayoutLMEncoder class.

        Args:
            self (LayoutLMEncoder): The instance of the LayoutLMEncoder class.
            config (object): The configuration object containing the necessary parameters for the LayoutLMEncoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.CellList([LayoutLMLayer(config) for _ in range(config.num_hidden_layers)])
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
        This method constructs the LayoutLM encoder using the specified parameters.

        Args:
            self: The instance of the LayoutLMEncoder class.
            hidden_states (mindspore.Tensor): The input hidden states for encoding.
            attention_mask (Optional[mindspore.Tensor]): Mask to avoid attending to certain positions.
            head_mask (Optional[mindspore.Tensor]): Mask to specify which heads to disable in the attention computation.
            encoder_hidden_states (Optional[mindspore.Tensor]): Hidden states of the encoder to be used in
                cross-attention layers.
            encoder_attention_mask (Optional[mindspore.Tensor]): Mask for encoder attention mechanism.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): Cached key/values for previous decoding steps.
            use_cache (Optional[bool]): Flag to indicate whether to use caching for decoding.
            output_attentions (Optional[bool]): Flag to output attention scores.
            output_hidden_states (Optional[bool]): Flag to output hidden states for each layer.
            return_dict (Optional[bool]): Flag to indicate returning the output as a dictionary.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]: The output of the encoder,
                which is either a tuple of hidden states or a complex object containing past key values and attentions.

        Raises:
            Warning: If `use_cache=True` is incompatible with gradient checkpointing,
                it will issue a warning and set `use_cache=False`.
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


class LayoutLMPooler(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertPooler"""
    def __init__(self, config):
        """
        Initializes a LayoutLMPooler instance.

        Args:
            self: The instance of LayoutLMPooler.
            config: The configuration object containing parameters for the LayoutLMPooler initialization.
                It should be an instance of the Config class.

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
        This method 'construct' in the class 'LayoutLMPooler' constructs a pooled output tensor based on
        the hidden states provided.

        Args:
            self (LayoutLMPooler): The instance of the LayoutLMPooler class.
            hidden_states (mindspore.Tensor): The input tensor containing hidden states.
                It should have the shape (batch_size, sequence_length, hidden_size).
                This tensor holds the hidden states generated by the model for each token in the input sequence.

        Returns:
            mindspore.Tensor: A tensor representing the pooled output. It is the result of applying dense and
                activation layers on the first token's hidden state.
                The shape of the returned tensor is (batch_size, hidden_size).

        Raises:
            None.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LayoutLMPredictionHeadTransform(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->LayoutLM"""
    def __init__(self, config):
        """
        Initialize the LayoutLMPredictionHeadTransform class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the head transformation.

                - Type: Custom configuration class
                - Purpose: Specifies the configuration settings for the head transformation.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            KeyError: If the specified 'hidden_act' in the configuration is not found in the ACT2FN dictionary.
            AttributeError: If the configuration object does not contain the required attributes.
            ValueError: If there are issues with the provided configuration parameters.
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
        This method 'construct' in the class 'LayoutLMPredictionHeadTransform'
        performs transformations on the input hidden states tensor.

        Args:
            self: An instance of the class 'LayoutLMPredictionHeadTransform'.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.
                It is expected to be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: A tensor containing the transformed hidden states after passing through dense layers,
                activation function, and layer normalization. The shape of the output tensor is the same as
                the input hidden_states.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LayoutLMLMPredictionHead(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->LayoutLM"""
    def __init__(self, config):
        """
        Initializes an instance of the LayoutLMLMPredictionHead class.

        Args:
            self: The instance of the LayoutLMLMPredictionHead class.
            config: An object containing configuration parameters for the LayoutLMLMPredictionHead.
                It is expected to be an instance of a class that holds information such as hidden size and vocabulary size.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.transform = LayoutLMPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def construct(self, hidden_states):
        """
        Constructs the LayoutLMLMPredictionHead by transforming and decoding hidden states.

        Args:
            self (LayoutLMLMPredictionHead): An instance of the LayoutLMLMPredictionHead class.
            hidden_states (torch.Tensor): The input hidden states to be processed by the prediction head.

        Returns:
            None.

        Raises:
            None
        """
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class LayoutLMOnlyMLMHead(nn.Cell):
    """Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->LayoutLM"""
    def __init__(self, config):
        """
        Initializes a LayoutLMOnlyMLMHead object.

        Args:
            self (LayoutLMOnlyMLMHead): The current instance of the LayoutLMOnlyMLMHead class.
            config: The configuration parameters for the LayoutLMOnlyMLMHead.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.predictions = LayoutLMLMPredictionHead(config)

    def construct(self, sequence_output: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the LayoutLMOnlyMLMHead.

        This method takes two parameters: self and sequence_output.

        Args:
            self: An instance of the LayoutLMOnlyMLMHead class.
            sequence_output (mindspore.Tensor): The output tensor from the sequence modeling layer.
                It is the input to the prediction layer.

        Returns:
            mindspore.Tensor: The prediction scores tensor generated by the prediction layer.
                It represents the predicted scores for each token in the input sequence.

        Raises:
            None.
        """
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LayoutLMConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutlm"
    supports_gradient_checkpointing = True

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(mindspore.Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, LayoutLMLayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))


class LayoutLMModel(LayoutLMPreTrainedModel):
    """LayoutLM Model"""
    def __init__(self, config):
        """
        Initializes a LayoutLMModel instance.

        Args:
            self: The instance of the LayoutLMModel class.
            config: A dictionary containing the configuration settings for the LayoutLMModel.
                The config should include parameters for initializing the LayoutLMModel,
                such as hidden size, number of layers, etc.

        Returns:
            None.

        Raises:
            None.
        """
        super(LayoutLMModel, self).__init__(config)
        self.config = config

        self.embeddings = LayoutLMEmbeddings(config)
        self.encoder = LayoutLMEncoder(config)
        self.pooler = LayoutLMPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings from the LayoutLMModel.

        Args:
            self (LayoutLMModel): The LayoutLMModel instance.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the LayoutLMModel.

        Args:
            self (LayoutLMModel): The LayoutLMModel instance.
            value: The input embeddings to be set.
                It should be of type torch.Tensor and have the same shape as the word_embeddings.

        Returns:
            None

        Raises:
            None
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
        bbox: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""

        Returns:
            Union[Tuple, BaseModelOutputWithPoolingAndCrossAttentions]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, LayoutLMModel
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            >>> model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
            ...
            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
            ...
            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
            ...
            >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            ...
            >>> outputs = model(
            ...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
            ... )
            ...
            >>> last_hidden_states = outputs.last_hidden_state
            ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = ops.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, dtype=mindspore.int64)

        if bbox is None:
            bbox = ops.zeros(input_shape + (4,), dtype=mindspore.int64)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * finfo(self.dtype, 'min')

        if head_mask is not None:
            if head_mask.ndimension() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.broadcast_to((self.config.num_hidden_layers, -1, -1, -1, -1))
            elif head_mask.ndimension() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.get_parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            bbox=bbox,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
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
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    """LayoutLMForMaskedLM Model"""
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        """
        Initializes the LayoutLMForMaskedLM class.

        Args:
            self: The instance of the class.
            config: The configuration object that contains the model configuration settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.layoutlm = LayoutLMModel(config)
        self.cls = LayoutLMOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the LayoutLM model for Masked Language Modeling task.

        Args:
            self (LayoutLMForMaskedLM): The instance of the LayoutLMForMaskedLM class.
                It represents the model for Masked Language Modeling.

        Returns:
            word_embeddings: The word embeddings from the LayoutLM model's embeddings.

        Raises:
            None.
        """
        return self.layoutlm.embeddings.word_embeddings

    def get_output_embeddings(self):
        '''
        Returns the output embeddings for the LayoutLM model.

        Args:
            self (LayoutLMForMaskedLM): The LayoutLMForMaskedLM object.

        Returns:
            None.

        Raises:
            None.
        '''
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings for the LayoutLMForMaskedLM model.

        Args:
            self (LayoutLMForMaskedLM): The instance of the LayoutLMForMaskedLM class.
            new_embeddings (Any): The new embeddings to set for the model's output layer.

        Returns:
            None.

        Raises:
            None.
        """
        self.cls.predictions.decoder = new_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        labels: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
                loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

        Returns:
            Union[Tuple, MaskedLMOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, LayoutLMForMaskedLM
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            >>> model = LayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")
            ...
            >>> words = ["Hello", "[MASK]"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
            ...
            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
            ...
            >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            ...
            >>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]
            ...
            >>> outputs = model(
            ...     input_ids=input_ids,
            ...     bbox=bbox,
            ...     attention_mask=attention_mask,
            ...     token_type_ids=token_type_ids,
            ...     labels=labels,
            ... )
            ...
            >>> loss = outputs.loss
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids,
            bbox,
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
            masked_lm_loss = ops.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    """LayoutLMForSequenceClassification Model"""
    def __init__(self, config):
        """
        __init__

        Initializes the LayoutLMForSequenceClassification class.

        Args:
            self: The instance of the class.
            config: An instance of the configuration class containing the model configuration parameters.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the num_labels attribute is not present in the config parameter.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method, get_input_embeddings, retrieves the input embeddings from the LayoutLM model for sequence classification.

        Args:
            self (LayoutLMForSequenceClassification): The instance of the LayoutLMForSequenceClassification class.
                This parameter refers to the current instance of the LayoutLMForSequenceClassification class.

        Returns:
            None:
                The input embeddings from the LayoutLM model for sequence classification.

        Raises:
            None.
        """
        return self.layoutlm.embeddings.word_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
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
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:
            Union[Tuple, SequenceClassifierOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, LayoutLMForSequenceClassification
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            >>> model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")
            ...
            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
            ...
            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
            ...
            >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> sequence_label = torch.tensor([1])
            ...
            >>> outputs = model(
            ...     input_ids=input_ids,
            ...     bbox=bbox,
            ...     attention_mask=attention_mask,
            ...     token_type_ids=token_type_ids,
            ...     labels=sequence_label,
            ... )
            ...
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
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


class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    """LayoutLMForTokenClassification Model"""
    def __init__(self, config):
        """
        Initializes an instance of the LayoutLMForTokenClassification class.

        Args:
            self: The instance of the LayoutLMForTokenClassification class.
            config:
                An object of the LayoutLMConfig class containing the configuration parameters for the LayoutLM model.

                - Type: LayoutLMConfig
                - Purpose: Specifies the configuration parameters for the LayoutLM model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method returns the word embeddings from the LayoutLM model for token classification.

        Args:
            self: The instance of the LayoutLMForTokenClassification class.

        Returns:
            word_embeddings: The word embeddings from the LayoutLM model for token classification.

        Raises:
            None.
        """
        return self.layoutlm.embeddings.word_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
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
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:
            Union[Tuple, TokenClassifierOutput]

        Example:
            ```python
            >>> from transformers import AutoTokenizer, LayoutLMForTokenClassification
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
            >>> model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
            ...
            >>> words = ["Hello", "world"]
            >>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
            ...
            >>> token_boxes = []
            >>> for word, box in zip(words, normalized_word_boxes):
            ...     word_tokens = tokenizer.tokenize(word)
            ...     token_boxes.extend([box] * len(word_tokens))
            >>> # add bounding boxes of cls + sep tokens
            >>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
            ...
            >>> encoding = tokenizer(" ".join(words), return_tensors="pt")
            >>> input_ids = encoding["input_ids"]
            >>> attention_mask = encoding["attention_mask"]
            >>> token_type_ids = encoding["token_type_ids"]
            >>> bbox = torch.tensor([token_boxes])
            >>> token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  # batch size of 1
            ...
            >>> outputs = model(
            ...     input_ids=input_ids,
            ...     bbox=bbox,
            ...     attention_mask=attention_mask,
            ...     token_type_ids=token_type_ids,
            ...     labels=token_labels,
            ... )
            ...
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
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


class LayoutLMForQuestionAnswering(LayoutLMPreTrainedModel):
    """LayoutLMForQuestionAnswering Model"""
    def __init__(self, config, has_visual_segment_embedding=True):
        """
        Initializes an instance of the LayoutLMForQuestionAnswering class.

        Args:
            self: The instance of the class.
            config (object): An object containing configuration settings.
            has_visual_segment_embedding (bool, optional): Flag indicating whether visual segment embedding is present.
                Defaults to True.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.layoutlm = LayoutLMModel(config)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the LayoutLM model for question answering.

        Args:
            self: An instance of the LayoutLMForQuestionAnswering class.
                It represents the current instance of the model and is used to access the embeddings.

        Returns:
            word_embeddings:
                The word embeddings:

                from the LayoutLM model for input sequences.
                The embeddings are used for processing the input data during question answering tasks.

        Raises:
            None.
        """
        return self.layoutlm.embeddings.word_embeddings

    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        bbox: Optional[mindspore.Tensor] = None,
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
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.

        Returns:
            Union[Tuple, QuestionAnsweringModelOutput]

        In the example below, we prepare a question + context pair for the LayoutLM model. It will give us a prediction
        of what it thinks the answer is (the span of the answer within the texts parsed from the image).

        Example:
            ```python
            >>> from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
            >>> from datasets import load_dataset
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
            >>> model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")
            ...
            >>> dataset = load_dataset("nielsr/funsd", split="train")
            >>> example = dataset[0]
            >>> question = "what's his name?"
            >>> words = example["words"]
            >>> boxes = example["bboxes"]
            ...
            >>> encoding = tokenizer(
            ...     question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="pt"
            ... )
            >>> bbox = []
            >>> for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
            ...     if s == 1:
            ...         bbox.append(boxes[w])
            ...     elif i == tokenizer.sep_token_id:
            ...         bbox.append([1000] * 4)
            ...     else:
            ...         bbox.append([0] * 4)
            >>> encoding["bbox"] = torch.tensor([bbox])
            ...
            >>> word_ids = encoding.word_ids(0)
            >>> outputs = model(**encoding)
            >>> loss = outputs.loss
            >>> start_scores = outputs.start_logits
            >>> end_scores = outputs.end_logits
            >>> start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
            >>> print(" ".join(words[start : end + 1]))
            M. Hamann P. Harper, P. Martinez
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
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
    "LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST",
    "LayoutLMForMaskedLM",
    "LayoutLMForSequenceClassification",
    "LayoutLMForTokenClassification",
    "LayoutLMForQuestionAnswering",
    "LayoutLMModel",
    "LayoutLMPreTrainedModel",
]
