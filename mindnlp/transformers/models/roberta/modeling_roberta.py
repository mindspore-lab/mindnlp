# Copyright 2023 Huawei Technologies Co., Ltd
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

"""roberta model, base on bert."""
import math
from typing import Optional, Tuple, Union, List

import mindspore
from mindspore.common.initializer import initializer

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from mindnlp.core.nn import functional as F
from mindnlp.utils import logging
from .configuration_roberta import RobertaConfig
from ..bert.modeling_bert import BertPreTrainedModel
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
from ...activations import ACT2FN
from ...ms_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer


ROBERTA_SUPPORT_LIST = [
    'roberta-base',
    'roberta-large',
    'roberta-large-mnli',
]

logger = logging.get_logger(__name__)


class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        """
        Initializes the RobertaEmbeddings class with the provided configuration.
        
        Args:
            self (RobertaEmbeddings): The instance of the RobertaEmbeddings class.
            config (object):
                A configuration object containing the following attributes:

                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden layers.
                - max_position_embeddings (int): The maximum number of positional embeddings.
                - type_vocab_size (int): The size of the token type vocabulary.
                - layer_norm_eps (float): The epsilon value for Layer Normalization.
                - hidden_dropout_prob (float): The dropout probability.
                - position_embedding_type (str, optional): The type of position embedding, defaults to 'absolute'.
                - pad_token_id (int): The token ID for padding.

        Returns:
            None.

        Raises:
            AttributeError: If the config object is missing required attributes.
            ValueError: If the config attributes are not of the expected types.
            RuntimeError: If there are issues with initializing embeddings or layers.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            [config.hidden_size], eps=config.layer_norm_eps
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.position_ids = ops.arange(config.max_position_embeddings).view((1, -1))
        self.token_type_ids = ops.zeros(self.position_ids.shape, dtype=mindspore.int64)

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        """
        This method forwards the embeddings for the Roberta model.

        Args:
            self (object): The instance of the class.
            input_ids (Union[None, Tensor]): The input tensor containing the tokenized input.
            token_type_ids (Union[None, Tensor]): The tensor containing token type ids for differentiating
                token types in the input.
            position_ids (Union[None, Tensor]): The tensor containing the position ids for each token in the input.
            inputs_embeds (Union[None, Tensor]): The tensor containing the input embeddings.
            past_key_values_length (int): The length of past key values.

        Returns:
            None.

        Raises:
            ValueError: If the input shape is not valid.
            AttributeError: If the 'token_type_ids' attribute is not found.
            TypeError: If the data type of the tensors is not supported.
        """
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )

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
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    (input_shape[0], seq_length)
                )
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
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=mindspore.int64,
        )
        return position_ids.unsqueeze(0).broadcast_to(input_shape)

class RobertaSelfAttention(nn.Module):
    """RobertaSelfAttention"""
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the RobertaSelfAttention class.

        Args:
            self: The object itself.
            config (object): A configuration object containing various settings.
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

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type in ("relative_key_query", "relative_key"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        Transposes the input tensor for computing self-attention scores.

        Args:
            self (RobertaSelfAttention): The instance of the `RobertaSelfAttention` class.
            x (mindspore.Tensor): The input tensor to be transposed.
                It should have a shape of (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The transposed tensor with shape
                (batch_size, num_attention_heads, sequence_length, attention_head_size).

        Raises:
            None.

        Note:
            - The `x` tensor is reshaped to have dimensions (batch_size, sequence_length, num_attention_heads,
                attention_head_size).
            - The `x` tensor is then permuted to have dimensions (batch_size, num_attention_heads, sequence_length,
                attention_head_size).

        Example:
            ```python
            >>> attention = RobertaSelfAttention()
            >>> input_tensor = mindspore.Tensor(np.random.randn(2, 5, 10), mindspore.float32)
            >>> output_tensor = attention.transpose_for_scores(input_tensor)
            >>> print(output_tensor.shape)
            (2, 12, 5, 10)
            ```
        """
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
        Constructs the self-attention mechanism for the Roberta model.

        Args:
            self (RobertaSelfAttention): The instance of the RobertaSelfAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of the model.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor. Default: None.
                Shape: (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask tensor. Default: None.
                Shape: (num_heads, hidden_size).
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder. Default: None.
                Shape: (batch_size, encoder_sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask for the encoder. Default: None.
                Shape: (batch_size, encoder_sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key and value tensors. Default: None.
                Shape: ((batch_size, num_heads, past_sequence_length, head_size),
                (batch_size, num_heads, past_sequence_length, head_size)).
            output_attentions (Optional[bool]): Whether to output attention probabilities. Default: False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the context layer tensor.
                Shape: (batch_size, sequence_length, hidden_size). Optionally, if `output_attentions` is True,
                the tuple also contains the attention probabilities tensor.
                Shape: (batch_size, num_heads, sequence_length, sequence_length).

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

        if self.position_embedding_type in ("relative_key_query", "relative_key"):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = mindspore.Tensor(key_length - 1, dtype=mindspore.int64).view(
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
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
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

        context_layer = context_layer.permute(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RobertaSelfOutput(nn.Module):

    """
    This class represents the self-output module of the Roberta model. It applies a dense layer, layer normalization,
    and dropout to the hidden states, and then adds them to the input tensor.

    Args:
        config (obj): The configuration object that contains the settings for the module.

    Returns:
        Tensor: The output tensor after applying the self-output operations.

    Raises:
        None.

    Example:
        ```python
        >>> config = RobertaConfig(hidden_size=768, layer_norm_eps=1e-5, hidden_dropout_prob=0.1)
        >>> self_output = RobertaSelfOutput(config)
        >>> hidden_states = mindspore.Tensor(...)
        >>> input_tensor = mindspore.Tensor(...)
        >>> output = self_output.forward(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """
        Initializes a new instance of the RobertaSelfOutput class.

        Args:
            self: The instance of the class.
            config:
                An instance of the configuration class containing the following attributes:

                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden layer.

        Returns:
            None.

        Raises:
            TypeError: If the provided config parameter is not of the expected type.
            ValueError: If the hidden_size attribute in the config parameter is not a positive integer.
            ValueError: If the layer_norm_eps attribute in the config parameter is not a positive float.
            ValueError: If the hidden_dropout_prob attribute in the config parameter is not a float between 0 and 1.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs the output of the RobertaSelfOutput layer.

        Args:
            self (RobertaSelfOutput): The instance of the RobertaSelfOutput class.
            hidden_states (mindspore.Tensor): The tensor containing the hidden states.
                This tensor should have the shape (batch_size, sequence_length, hidden_size).
            input_tensor (mindspore.Tensor): The tensor containing the input states.
                This tensor should have the same shape as the hidden_states tensor.

        Returns:
            mindspore.Tensor: The output tensor after applying the RobertaSelfOutput layer.
                This tensor has the same shape as the input_tensor.

        Raises:
            None.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):

    """
    This class represents the attention mechanism used in the Roberta model. It is a subclass of nn.Module.

    The RobertaAttention class implements the attention mechanism used in the Roberta model.
    It consists of a self-attention module and a self-output module. The self-attention module is responsible for
    computing the attention scores between the input hidden states and itself, while the self-output module applies
    a linear transformation to the attention output.

    The class provides the following methods:

    - __init__: Initializes the RobertaAttention instance. It takes a configuration object and an optional position_embedding_type as arguments. The config object
    contains the model configuration, while the position_embedding_type specifies the type of position embedding to be used.

    - prune_heads: Prunes the specified attention heads. It takes a list of heads to be pruned as input. This method updates the attention module by removing the pruned heads and adjusting the
    attention head size accordingly.

    - forward: Constructs the attention output given the input hidden states and optional arguments.
    It computes the attention scores using the self-attention module and applies the self-output module to generate
    the final attention output. This method returns a tuple containing the attention output and optional additional
    outputs.

    Note:
        - The 'hidden_states' argument is a tensor representing the input hidden states.
        - The 'attention_mask' argument is an optional tensor specifying the attention mask.
        - The 'head_mask' argument is an optional tensor indicating which attention heads to mask.
        - The 'encoder_hidden_states' and 'encoder_attention_mask' arguments are optional tensors representing the hidden
        states and attention mask of the encoder.
        - The 'past_key_value' argument is an  optional tuple of past key-value tensors.
        - The 'output_attentions' argument is a boolean flag indicating whether to output the attention scores.

    Please refer to the RobertaSelfAttention and RobertaSelfOutput classes for more information about the self-attention
    and self-output modules used in this class.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes a new instance of the RobertaAttention class.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object for the attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to be used.
                Default is None. If provided, it should be a string representing the type of position embedding.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.self = RobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        Prunes the attention heads in the RobertaAttention class.

        Args:
            self (RobertaAttention): The instance of the RobertaAttention class.
            heads (List[int]): The list of attention heads to be pruned.

        Returns:
            None

        Raises:
            None
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
        Constructs the attention mechanism for the RobertaAttention class.

        Args:
            self: The instance of the RobertaAttention class.
            hidden_states (mindspore.Tensor): The input hidden states for the attention mechanism.
            attention_mask (Optional[mindspore.Tensor]): An optional mask tensor to mask out specific attention weights.
                Defaults to None.
            head_mask (Optional[mindspore.Tensor]): An optional mask tensor to mask out specific attention heads.
                Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor representing hidden states from
                the encoder. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional mask tensor to mask out specific attention
                weights from the encoder. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of tensor tuples representing
                previous key-value pairs. Defaults to None.
            output_attentions (Optional[bool]): An optional flag to indicate whether to output attention weights.
                Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor and any additional outputs
                from the mechanism.

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


class RobertaIntermediate(nn.Module):

    """
    Represents the intermediate layer of the Roberta model for processing hidden states.

    This class inherits from nn.Module and provides methods for forwarding the intermediate layer of the Roberta model.

    Attributes:
        dense (nn.Linear): A dense layer with specified hidden size and intermediate size.
        intermediate_act_fn (function): Activation function applied to hidden states.

    Methods:
        __init__: Initializes the RobertaIntermediate instance with the given configuration.
        forward: Constructs the intermediate layer by passing the hidden states through the dense layer and activation function.

    Example:
        ```python
        >>> config = RobertaConfig(hidden_size=768, intermediate_size=3072, hidden_act='gelu')
        >>> intermediate_layer = RobertaIntermediate(config)
        >>> hidden_states = intermediate_layer.forward(input_hidden_states)
        ```

    Example:
        ```python
        >>> config = RobertaConfig(hidden_size=768, intermediate_size=3072, hidden_act='gelu')
        >>> intermediate_layer = RobertaIntermediate(config)
        >>> hidden_states = intermediate_layer.forward(input_hidden_states)
        ```
    """
    def __init__(self, config):
        """
        Initializes a new instance of the RobertaIntermediate class.

        Args:
            self: The instance of the class.
            config: An object of type 'config' containing configuration parameters for the intermediate layer.
                It is expected to have attributes like 'hidden_size', 'intermediate_size', and 'hidden_act'.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided or is not of the expected type.
            ValueError: If the 'config' parameter does not contain the required attributes.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the intermediate representation of the Roberta model.

        Args:
            self (RobertaIntermediate): The instance of the RobertaIntermediate class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states.

        Returns:
            mindspore.Tensor: A tensor representing the intermediate states of the Roberta model.

        Raises:
            None
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class RobertaOutput(nn.Module):

    """
    This class represents the output of a Roberta model, which is used for fine-tuning tasks.
    It inherits from the `nn.Module` class.

    The `RobertaOutput` class applies a series of transformations to the input hidden states and produces
    the final output tensor.

    Attributes:
        dense (nn.Linear): A fully connected layer that maps the input hidden states to an intermediate size.
        LayerNorm (nn.LayerNorm): A layer normalization module that normalizes the hidden states.
        dropout (nn.Dropout): A dropout module that applies dropout to the hidden states.

    Methods:
        forward:
            Applies the transformation operations to the hidden states and returns the final output tensor.

    Example:
        ```python
        >>> # Create a `RobertaOutput` instance
        >>> output = RobertaOutput(config)
        ...
        >>> # Apply the transformation operations to the hidden states
        >>> output_tensor = output.forward(hidden_states, input_tensor)
        ```
    """
    def __init__(self, config):
        """
        Initializes a new instance of the 'RobertaOutput' class.

        Args:
            self: The current instance of the class.
            config: An object of type 'Config' that holds the configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm([config.hidden_size], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, hidden_states: mindspore.Tensor, input_tensor: mindspore.Tensor) -> mindspore.Tensor:
        """
        This method forwards the output tensor for the Roberta model.

        Args:
            self: The instance of the RobertaOutput class.
            hidden_states (mindspore.Tensor): The hidden states tensor representing the output from the
                model's encoder layers. It is expected to be a tensor of shape [batch_size, sequence_length, hidden_size].
            input_tensor (mindspore.Tensor): The input tensor representing the output from the previous layer.
                It is expected to be a tensor of the same shape as hidden_states.

        Returns:
            mindspore.Tensor: The forwarded output tensor of the same shape as hidden_states,
                representing the final output of the Roberta model.

        Raises:
            ValueError: If the shapes of hidden_states and input_tensor are not compatible for addition.
            RuntimeError: If an error occurs during the dense, dropout, or LayerNorm operations.
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaLayer(nn.Module):

    """
    Represents a layer of the Roberta model for natural language processing tasks.
    This layer includes self-attention and cross-attention mechanisms.

    This class inherits from nn.Module and contains methods for initializing the layer and forwarding the
    layer's functionality.

    Attributes:
        chunk_size_feed_forward (int): The chunk size for the feed-forward computation.
        seq_len_dim (int): The dimension for sequence length.
        attention (RobertaAttention): The self-attention mechanism used in the layer.
        is_decoder (bool): Indicates if the layer is used as a decoder model.
        add_cross_attention (bool): Indicates if cross-attention is added to the layer.
        crossattention (RobertaAttention): The cross-attention mechanism used in the layer, if cross-attention is added.
        intermediate (RobertaIntermediate): The intermediate processing module of the layer.
        output (RobertaOutput): The output module of the layer.

    Methods:
        __init__: Initializes the RobertaLayer with the given configuration.
        forward: Constructs the layer using the given input and arguments,
            applying self-attention and cross-attention if applicable.
        feed_forward_chunk: Performs the feed-forward computation using the given attention output.
    """
    def __init__(self, config):
        """
        Initializes an instance of the `RobertaLayer` class.

        Args:
            self: The instance of the `RobertaLayer` class.
            config: An object of type `Config` containing the configuration settings for the model.

        Returns:
            None.

        Raises:
            ValueError: If `add_cross_attention` is set to `True` but the model is not used as a decoder model.

        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

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
        Constructs a single layer of the Roberta model.

        Args:
            self (RobertaLayer): The instance of the RobertaLayer class.
            hidden_states (mindspore.Tensor): The input tensor of shape (batch_size, sequence_length, hidden_size)
                representing the hidden states.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor of shape (batch_size, sequence_length)
                representing the attention mask. Defaults to None.
            head_mask (Optional[mindspore.Tensor]): An optional tensor of shape
                (num_attention_heads, sequence_length, sequence_length) representing the head mask. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, encoder_sequence_length, hidden_size) representing the hidden states of the encoder.
                Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor]): An optional tensor of shape
                (batch_size, encoder_sequence_length) representing the attention mask for the encoder. Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple containing past key-value
                tensors. Defaults to None.
            output_attentions (Optional[bool]): An optional boolean value indicating whether to output attentions.
                Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the following:

                - layer_output (mindspore.Tensor): The output tensor of shape (batch_size, sequence_length, hidden_size)
                representing the layer output.
                - present_key_value (mindspore.Tensor): The tensor of shape
                (batch_size, num_heads, sequence_length, hidden_size), containing the present key-value tensors.
                Only returned if self.is_decoder is True.

        Raises:
            ValueError: If `encoder_hidden_states` are passed, and `self` is not instantiated with cross-attention
                layers by setting `config.add_cross_attention=True`.
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
        Method that carries out feed-forward processing on the attention output in a RobertaLayer.

        Args:
            self (RobertaLayer): The instance of the RobertaLayer class.
            attention_output (tensor): The input tensor representing the attention output.
                This tensor is expected to have a specific shape and structure required for processing.

        Returns:
            None.

        Raises:
            None.
        """
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class RobertaEncoder(nn.Module):

    """
    This class represents a RobertaEncoder, which is a neural network encoder for the RoBERTa model.
    It inherits from the nn.Module class and is responsible for encoding input sequences using a stack of
    multiple RobertaLayer modules.

    The RobertaEncoder class contains an __init__ method to initialize the encoder with a given configuration,
    and a forward method to perform the encoding process. The forward method takes in various input tensors and
    optional parameters, and returns the encoded output and optional additional information such as hidden states,
    attentions, and cross-attentions.

    The encoder utilizes a stack of RobertaLayer modules, where each layer applies a series of transformations to the
    input hidden states using self-attention and optionally cross-attention mechanisms. The forward method iterates
    through the layers, applying the transformations and updating the hidden states accordingly.

    Additionally, the encoder supports gradient checkpointing and caching of past key values for efficient training
    and inference.

    For consistency, always use triple double quotes around docstrings.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the RobertaEncoder class.

        Args:
            self (RobertaEncoder): The instance of the RobertaEncoder class.
            config (dict): A dictionary containing configuration parameters for the encoder.
                It should include the following keys:

                - num_hidden_layers (int): The number of hidden layers in the encoder.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
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
        Constructs the RobertaEncoder.

        Args:
            self: The object instance.
            hidden_states (mindspore.Tensor): The input hidden states of the encoder layer.
                Shape: (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
                If provided, should be of shape (batch_size, sequence_length), with 0s indicating tokens to be masked
                and 1s indicating tokens to be attended to.
            head_mask (Optional[mindspore.Tensor]): The head mask tensor. If provided, should be of
                shape (num_layers, num_heads), with 0s indicating heads to be masked and 1s indicating heads to be used.
            encoder_hidden_states (Optional[mindspore.Tensor]): The hidden states of the encoder layer.
                Shape: (batch_size, sequence_length, hidden_size).
            encoder_attention_mask (Optional[mindspore.Tensor]): The attention mask tensor for encoder layer.
                If provided, should be of shape (batch_size, sequence_length), with 0s indicating tokens to be
                masked and 1s indicating tokens to be attended to.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key values. If provided,
                should be of shape (num_layers, 2, batch_size, num_heads, sequence_length, hidden_size // num_heads).
            use_cache (Optional[bool]): Whether to use cache. If True, the cache will be used and updated.
                If False, the cache will be ignored. Default: None.
            output_attentions (Optional[bool]): Whether to output attentions. If True, attentions will be output.
                Default: False.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
                If True, hidden states will be output. Default: False.
            return_dict (Optional[bool]): Whether to return a dictionary as output. If True, a dictionary containing
                the output tensors will be returned. If False, a tuple will be returned. Default: True.

        Returns:
            Union[Tuple[mindspore.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
                The output of the encoder layer. If return_dict is True, a dictionary containing the output tensors will
                be returned. If return_dict is False, a tuple of tensors will be returned. The output tensors include:

                - last_hidden_state (mindspore.Tensor): The last hidden state of the encoder layer.
                Shape: (batch_size, sequence_length, hidden_size).
                - past_key_values (Tuple[Tuple[mindspore.Tensor]]): The updated past key values. If use_cache is True,
                the key values for each layer will be returned. Shape: (num_layers, 2, batch_size, num_heads,
                sequence_length, hidden_size // num_heads).
                - hidden_states (Tuple[mindspore.Tensor]): The hidden states of the encoder layer.
                If output_hidden_states is True, all hidden states for each layer will be returned. Shape: (num_layers,
                batch_size, sequence_length, hidden_size).
                - attentions (Tuple[mindspore.Tensor]): The self-attention weights of the encoder layer.
                If output_attentions is True, all self-attention weights for each layer will be returned. Shape:
                (num_layers, batch_size, num_heads, sequence_length, sequence_length).
                - cross_attentions (Tuple[mindspore.Tensor]): The cross-attention weights of the encoder layer.
                If output_attentions is True and add_cross_attention is True, all cross-attention weights for each
                layer will be returned. Shape: (num_layers, batch_size, num_heads, sequence_length, encoder_sequence_length).

        Raises:
            None.
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


class RobertaPooler(nn.Module):

    """
    This class represents a pooler for the Roberta model. It inherits from the nn.Module class and is responsible
    for processing hidden states to generate a pooled output.

    Attributes:
        dense (nn.Linear): A fully connected layer that maps the input hidden state to the hidden size.
        activation (nn.Tanh): The activation function applied to the output of the dense layer.

    Methods:
        __init__: Initializes the RobertaPooler instance with the specified configuration.
        forward: Constructs the pooled output from the input hidden states.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the RobertaPooler class.

        Args:
            self: The instance of the RobertaPooler class.
            config: An object containing configuration parameters for the RobertaPooler instance.
                It is expected to have a 'hidden_size' attribute specifying the size of the hidden layer.

        Returns:
            None.

        Raises:
            AttributeError: If the 'config' parameter does not have the expected 'hidden_size' attribute.
            TypeError: If the 'config' parameter is not of the expected type.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: mindspore.Tensor) -> mindspore.Tensor:
        """
        Constructs a pooled output tensor from the given hidden states using the RobertaPooler module.

        Args:
            self (RobertaPooler): The instance of the RobertaPooler class.
            hidden_states (mindspore.Tensor): The input hidden states tensor of shape
                (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The pooled output tensor of shape (batch_size, hidden_size).

        Raises:
            TypeError: If the 'hidden_states' parameter is not of type 'mindspore.Tensor'.
            ValueError: If the shape of the 'hidden_states' tensor is not (batch_size, sequence_length, hidden_size).

        Note:
            - The 'hidden_states' tensor should contain the hidden states of the sequence generated by the Roberta model.
            - The 'hidden_states' tensor should have a shape of (batch_size, sequence_length, hidden_size).
            - The 'hidden_states' tensor is expected to be the output of the Roberta model's last layer.
            - The 'hidden_states' tensor should be on the same device as the RobertaPooler module.

        Example:
            ```python
            >>> roberta_pooler = RobertaPooler()
            >>> hidden_states = mindspore.Tensor(np.random.randn(2, 5, 768), dtype=mindspore.float32)
            >>> pooled_output = roberta_pooler.forward(hidden_states)
            ```
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RobertaPreTrainedModel(BertPreTrainedModel):
    """Roberta Pretrained Model."""
    config_class = RobertaConfig
    base_model_prefix = "roberta"


class RobertaModel(RobertaPreTrainedModel):
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
    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new instance of the RobertaModel class.

        Args:
            self: The current object instance.
            config (object): An instance of the configuration class that contains the model configuration parameters.
            add_pooling_layer (bool, optional): Determines whether to add a pooling layer to the model. Defaults to True.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method initializes a new instance of the RobertaModel class. It takes the following parameters:

            - self: The current object instance.
            - config: An instance of the configuration class that contains the model configuration parameters.
            - add_pooling_layer: A boolean value that determines whether to add a pooling layer to the model.

            The method initializes the following attributes:

            - self.config: Stores the provided configuration object.
            - self.embeddings: An instance of the RobertaEmbeddings class, initialized with the provided configuration.
            - self.encoder: An instance of the RobertaEncoder class, initialized with the provided configuration.
            - self.pooler: An instance of the RobertaPooler class, initialized with the provided configuration
            if add_pooling_layer is True, otherwise set to None.

            After initialization, this method calls the post_init() method to perform any additional setup
            or initialization steps.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the input embeddings of the RobertaModel.

        Args:
            self (RobertaModel): An instance of the RobertaModel class.

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the RobertaModel.

        Args:
            self (RobertaModel): The instance of the RobertaModel class.
            value (object): The input embeddings to be set for the model. This can be a tensor or any other object
                that can be assigned to the `word_embeddings` attribute of the `embeddings` object.

        Returns:
            None.

        Raises:
            None.

        Note:
            The `word_embeddings` attribute of the `embeddings` object is a key component of the RobertaModel.
            It represents the input embeddings used for the model's forward pass.
            By setting the input embeddings using this method, you can customize the input representation for the model.

        Example:
            ```python
            >>> model = RobertaModel()
            >>> embeddings = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
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


class RobertaForCausalLM(RobertaPreTrainedModel):

    """
        RobertaForCausalLM

        This class is a RoBERTa model for causal language modeling. It predicts the next word in a sequence given the
        previous words.

        Class Inheritance:
            `RobertaForCausalLM` inherits from `RobertaPreTrainedModel`.

        Args:
            config: `RobertaConfig`
                The configuration object that specifies the model architecture and hyperparameters.

        Attributes:
            roberta: `RobertaModel`
                The RoBERTa model that encodes the input sequence.
            lm_head: `RobertaLMHead`
                The linear layer that predicts the next word in the sequence.

        Methods:
            get_output_embeddings
                Retrieve the output embeddings of the model.
            set_output_embeddings
                Set new output embeddings for the model.
            forward
                Perform the forward pass of the model for causal language modeling.
            prepare_inputs_for_generation
                Prepare the inputs for generation by removing the prefix and adjusting the attention mask.
            _reorder_cache
                Reorder the cache of past key values based on the beam index.
    """
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        """
        Initializes a new instance of the `RobertaForCausalLM` class.

        Args:
            self: The object itself.
            config: An instance of the `RobertaConfig` class containing the model configuration settings.
                This parameter is required for the initialization of the `RobertaModel` and `RobertaLMHead` objects.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        if not config.is_decoder:
            logger.warning(
                "If you want to use `RobertaLMHeadModel` as a standalone, add `is_decoder=True.`"
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the RobertaForCausalLM model.

        Args:
            self: An instance of the RobertaForCausalLM class.

        Returns:
            None.

        Raises:
            None.

        This method returns the output embeddings for the RobertaForCausalLM model.
        The output embeddings are obtained from the decoder of the lm_head.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Sets the output embeddings of the RobertaForCausalLM model.

        Args:
            self (RobertaForCausalLM): The instance of the RobertaForCausalLM class.
            new_embeddings (torch.nn.Module): The new embeddings to be set as the output embeddings.

        Returns:
            None.

        Raises:
            None.
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
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
                the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            past_key_values (`tuple(tuple(mindspore.Tensor))` of length `config.n_layers` with each tuple having 4 tensors
                of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
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
            >>> from transformers import AutoTokenizer, RobertaForCausalLM, AutoConfig
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            >>> config = AutoConfig.from_pretrained("roberta-base")
            >>> config.is_decoder = True
            >>> model = RobertaForCausalLM.from_pretrained("roberta-base", config=config)
            ...
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="ms")
            >>> outputs = model(**inputs)
            ...
            >>> prediction_logits = outputs.logits
            ```
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
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
            lm_loss = F.cross_entropy(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

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
        self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs
    ):
        """
        Prepares the inputs for generation in the RobertaForCausalLM class.

        Args:
            self (RobertaForCausalLM): The instance of the RobertaForCausalLM class.
            input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length)
                containing the input token IDs.
            past_key_values (tuple, optional): A tuple of past key values. Defaults to None.
            attention_mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, sequence_length).
                Defaults to None.
            **model_kwargs: Additional keyword arguments for the model.

        Returns:
            dict:
                A dictionary containing the prepared inputs for generation with the following key-value pairs:

                - 'input_ids' (torch.Tensor): The input tensor with modified sequence length.
                - 'attention_mask' (torch.Tensor): The attention mask tensor.
                - 'past_key_values' (tuple): The modified tuple of past key values or None.

        Raises:
            None.
        """
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = ops.ones(input_shape)

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
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders the cache by selecting specific elements based on the beam indexes.

        Args:
            self (RobertaForCausalLM): The instance of the RobertaForCausalLM class.
            past_key_values (tuple): A tuple containing the past key-values for each layer.
                Each element in the tuple is a tensor representing the hidden states for a specific layer.
            beam_idx (tensor): A tensor containing the indexes of the selected beams.

        Returns:
            tuple: The reordered past key-values.
                Each element in the tuple is a tensor representing the hidden states for a specific layer.
                The tensors are selected based on the beam indexes.

        Raises:
            None.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past


class RobertaForMaskedLM(RobertaPreTrainedModel):

    """
    `RobertaForMaskedLM` is a Python class that represents a RoBERTa model for masked language modeling tasks.
    This class inherits from `RobertaPreTrainedModel` and provides methods for initializing the model,
    getting and setting output embeddings, and forwarding the model for masked language modeling tasks.
    It also includes a detailed `forward` method for processing input data and computing the masked language
    modeling loss.

    The class includes the following methods:

    - `__init__`: Initializes the `RobertaForMaskedLM` instance.
    - `get_output_embeddings`: Returns the output embeddings of the model.
    - `set_output_embeddings`: Sets the output embeddings of the model to the specified new embeddings.
    - `forward`: Constructs the model for masked language modeling tasks and computes the masked language modeling loss.

    The `forward` method supports various input parameters such as input IDs, attention mask, token type IDs,
    position IDs, head mask, input embeddings, encoder hidden states, encoder attention mask, labels, output attentions,
    output hidden states, and return dictionary. It also includes detailed information about the expected shape and
    type of the input data, as well as the optional arguments.

    Additionally, the class includes warnings and error handling for specific configurations, ensuring the proper usage
    of the `RobertaForMaskedLM` model for bi-directional self-attention.

    Note:
        The detailed method signatures and implementation details have been omitted for brevity and clarity.
    """
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        """
        Initializes a new instance of the 'RobertaForMaskedLM' class.

        Args:
            self: The current object instance.
            config:
                An instance of the 'Config' class containing the configuration settings for the model.

                - Type: Config
                - Purpose: Specifies the model's configuration.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings for the RobertaForMaskedLM model.

        Args:
            self: An instance of the RobertaForMaskedLM class.

        Returns:
            A tensor of size (batch_size, sequence_length, hidden_size) representing the output embeddings.

        Raises:
            None.
        """
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the RobertaForMaskedLM model.

        Args:
            self (RobertaForMaskedLM): The instance of the RobertaForMaskedLM class.
            new_embeddings (torch.nn.Module): The new output embeddings to be set for the model.
                It should be an instance of torch.nn.Module.

        Returns:
            None.

        Raises:
            None.
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""
    def __init__(self, config):
        """
        Initialize the RobertaLMHead class.

        Args:
            self (RobertaLMHead): The instance of the RobertaLMHead class.
            config (object):
                An object containing configuration parameters.

                - hidden_size (int): The size of the hidden layer.
                - layer_norm_eps (float): Epsilon value for layer normalization.
                - vocab_size (int): The size of the vocabulary.

        Returns:
            None.

        Raises:
            TypeError: If config is not provided or is not an object.
            ValueError: If the config object does not contain the required parameters.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(
            (config.hidden_size,), eps=config.layer_norm_eps
        )

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = Parameter(initializer("zeros", config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features):
        """
        Constructs the output of the language model head for a given set of features.

        Args:
            self (RobertaLMHead): The instance of the RobertaLMHead class.
            features (tensor): The input features for forwarding the output.
                It should be a tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            tensor: The forwarded output tensor of shape (batch_size, sequence_length, hidden_size).

        Raises:
            ValueError: If the input features tensor is not of the expected shape.
            RuntimeError: If there is an issue in the execution of the method.
        """
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        """
        This method ties the weights of the decoder's bias to the model's bias.

        Args:
            self (RobertaLMHead): The instance of the RobertaLMHead class.
                This parameter is used to access the decoder bias and tie it to the model's bias.

        Returns:
            None.

        Raises:
            This method does not raise any exceptions.
        """
        self.bias = self.decoder.bias


class RobertaForSequenceClassification(RobertaPreTrainedModel):

    """
    This class represents a Roberta model for sequence classification tasks.
    It is a subclass of RobertaPreTrainedModel and is specifically designed for sequence classification tasks.

    The class's code includes an initialization method (__init__) and a forward method.

    The __init__ method initializes the RobertaForSequenceClassification object by taking a config argument.
    It calls the super() method to initialize the parent class (RobertaPreTrainedModel) with the
    provided config. It also initializes other attributes such as num_labels and classifier.

    The forward method takes several input arguments and returns either a tuple of tensors or a
    SequenceClassifierOutput object. It performs the main computation of the model. It first calls the roberta()
    method of the parent class to obtain the sequence output. Then, it passes the sequence output to the classifier
    to obtain the logits. If labels are provided, it calculates the loss based on the problem type
    specified in the config. The loss and other outputs are returned as per the value of the return_dict parameter.

    It is important to note that this class is specifically designed for sequence classification tasks,
    where the labels can be used to compute either a regression loss (Mean-Square loss) or a classification
    loss (Cross-Entropy). The problem type is determined automatically based on the number of labels and the dtype
    of the labels tensor.

    For more details on the usage and functionality of this class, please refer to the RobertaForSequenceClassification
    documentation.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the RobertaForSequenceClassification class.

        Args:
            self: The instance of the class.
            config (RobertaConfig): The configuration object for the Roberta model.
                It contains the model configuration settings such as num_labels, which is the number of labels
                for classification. This parameter is required for configuring the model initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype in (mindspore.int32, mindspore.int64)
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                if self.num_labels == 1:
                    loss = F.mse_loss(logits.squeeze(), labels.squeeze())
                else:
                    loss = F.mse_loss(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
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


class RobertaForMultipleChoice(RobertaPreTrainedModel):

    """
    RobertaForMultipleChoice is a class for fine-tuning a pre-trained Roberta model for multiple choice tasks.

    This class inherits from RobertaPreTrainedModel and implements the necessary methods for forwarding the model
    architecture and computing the multiple choice classification loss.

    Attributes:
        roberta (RobertaModel): The RobertaModel instance for handling the main Roberta model.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Dense layer for classification.

    Methods:
        __init__: Initializes the RobertaForMultipleChoice instance with the given configuration.
        forward:
            Constructs the model architecture and computes the multiple choice classification loss.

    Parameters:
        input_ids (Optional[mindspore.Tensor]): Input tensor containing the token indices.
        token_type_ids (Optional[mindspore.Tensor]): Input tensor containing the token type ids.
        attention_mask (Optional[mindspore.Tensor]): Input tensor containing the attention mask.
        labels (Optional[mindspore.Tensor]): Tensor containing the labels for classification loss.
        position_ids (Optional[mindspore.Tensor]): Tensor containing the positional indices.
        head_mask (Optional[mindspore.Tensor]): Tensor containing the head mask.
        inputs_embeds (Optional[mindspore.Tensor]): Tensor containing the embedded input.
        output_attentions (Optional[bool]): Flag indicating whether to output attentions.
        output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
        return_dict (Optional[bool]): Flag indicating whether to return outputs as a dictionary.

    Returns:
        Union[Tuple[mindspore.Tensor], MultipleChoiceModelOutput]: Tuple containing the loss and model outputs.

    Raises:
        ValueError: If the input shape does not match the expected dimensions for multiple choice classification.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the `RobertaForMultipleChoice` class.

        Args:
            self: The object itself.
            config: An instance of the `RobertaConfig` class containing the model configuration settings.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        num_choices = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        flat_input_ids = (
            input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        )
        flat_position_ids = (
            position_ids.view(-1, position_ids.shape[-1])
            if position_ids is not None
            else None
        )
        flat_token_type_ids = (
            token_type_ids.view(-1, token_type_ids.shape[-1])
            if token_type_ids is not None
            else None
        )
        flat_attention_mask = (
            attention_mask.view(-1, attention_mask.shape[-1])
            if attention_mask is not None
            else None
        )
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
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

class RobertaForTokenClassification(RobertaPreTrainedModel):

    """
    This class represents a Roberta model for token classification. It is a subclass of the RobertaPreTrainedModel.

    Class Attributes:
        - num_labels (int): The number of labels for token classification.
        - roberta (RobertaModel): The RoBERTa model.
        - dropout (Dropout): The dropout layer.
        - classifier (Dense): The classifier layer.

    Methods:
        __init__(self, config): Initializes the RobertaForTokenClassification instance with the given configuration.
        forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels,
            output_attentions, output_hidden_states, return_dict): Constructs the token classification model and
            returns the output.

    Attributes:
        return_dict (bool): Indicates whether to return a dictionary as output.

    Parameters:
        input_ids (Optional[mindspore.Tensor]): The input tensor of shape (batch_size, sequence_length).
        attention_mask (Optional[mindspore.Tensor]): The attention mask tensor of shape (batch_size, sequence_length).
        token_type_ids (Optional[mindspore.Tensor]): The token type IDs tensor of shape (batch_size, sequence_length).
        position_ids (Optional[mindspore.Tensor]): The position IDs tensor of shape (batch_size, sequence_length).
        head_mask (Optional[mindspore.Tensor]): The head mask tensor of shape (batch_size, num_heads, sequence_length, sequence_length).
        inputs_embeds (Optional[mindspore.Tensor]): The embedded inputs tensor of shape (batch_size, sequence_length, hidden_size).
        labels (Optional[mindspore.Tensor]): The labels tensor of shape (batch_size, sequence_length).
        output_attentions (Optional[bool]): Indicates whether to output attentions.
        output_hidden_states (Optional[bool]): Indicates whether to output hidden states.
        return_dict (Optional[bool]): Indicates whether to return a dictionary as output.

    Returns:
        Conditional Return:

            - If return_dict is False, returns a tuple containing the loss tensor, logits tensor, and the remaining outputs.
            - If return_dict is True, returns a TokenClassifierOutput object containing the loss tensor, logits tensor,
            hidden states, and attentions.

    Note:
        The labels tensor should contain indices in the range [0, num_labels-1] for computing the token
        classification loss.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the `RobertaForTokenClassification` class.

        Args:
            self: The object itself.
            config: A `RobertaConfig` instance containing the configuration parameters for the model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        """
        Initialize the RobertaClassificationHead class.

        Args:
            self (object): The instance of the class.
            config (object): Configuration object containing parameters for the classification head.
                This object should have the following attributes:

                - hidden_size (int): The size of the hidden layers.
                - classifier_dropout (float, optional): The dropout probability for the classifier. If not provided,
                defaults to config.hidden_dropout_prob.
                - hidden_dropout_prob (float): The default dropout probability for hidden layers.
                - num_labels (int): The number of output labels.

        Returns:
            None.

        Raises:
            TypeError: If the config parameter is not provided.
            ValueError: If the config parameter is missing any of the required attributes.
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        """
        Constructs the classification head for a Roberta model.

        Args:
            self (RobertaClassificationHead): The instance of the RobertaClassificationHead class.
            features (torch.Tensor): The input features for the classification head.
                It should have shape (batch_size, seq_length, hidden_size).

        Returns:
            torch.Tensor: The output tensor after passing through the classification head.
                It has shape (batch_size, seq_length, num_labels).

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


class RobertaForQuestionAnswering(RobertaPreTrainedModel):

    """
    RobertaForQuestionAnswering is a class representing a model for question answering tasks based on the RoBERTa
    architecture.
    It inherits from RobertaPreTrainedModel and provides functionalities for forwarding the model and processing
    inputs for question answering.

    Attributes:
        num_labels (int): The number of labels for the question answering task.
        roberta (RobertaModel): The RoBERTa model used for processing input sequences.
        qa_outputs (mindspore.nn.Linear): A dense layer for outputting logits for the start and end positions of the
            labelled span.

    Methods:
        __init__: Initializes the RobertaForQuestionAnswering model with the provided configuration.
        forward:
            Constructs the model using the input tensors and returns the output logits for start and end positions.
            Optionally computes the total loss if start and end positions are provided.

            Args:

            - input_ids (Optional[mindspore.Tensor]): The input tensor containing token indices.
            - attention_mask (Optional[mindspore.Tensor]): The tensor indicating which tokens should be attended to.
            - token_type_ids (Optional[mindspore.Tensor]): The tensor indicating token types.
            - position_ids (Optional[mindspore.Tensor]): The tensor indicating token positions.
            - head_mask (Optional[mindspore.Tensor]): The tensor for masking specific heads in the self-attention mechanism.
            - inputs_embeds (Optional[mindspore.Tensor]): The embedded input tensors.
            - start_positions (Optional[mindspore.Tensor]): The labels for the start positions of the labelled span.
            - end_positions (Optional[mindspore.Tensor]): The labels for the end positions of the labelled span.
            - output_attentions (Optional[bool]): Flag indicating whether to output attention weights.
            - output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states.
            - return_dict (Optional[bool]): Flag indicating whether to return outputs as a dictionary.

            Returns:

            - Union[Tuple[mindspore.Tensor], QuestionAnsweringModelOutput]: The output logits for start and end positions,
            and optionally the total loss.

    Raises:
        ValueError: If the start_positions or end_positions have incorrect dimensions.
    """
    def __init__(self, config):
        """
        Initializes a new instance of the RobertaForQuestionAnswering class.

        Args:
            self: The object instance itself.
            config: A configuration object containing parameters for model initialization.
                It must have the attribute 'num_labels' specifying the number of labels.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided or is not of the expected type.
            ValueError: If the 'num_labels' attribute is missing in the 'config' object.
            RuntimeError: If an issue occurs during the initialization process of the RobertaForQuestionAnswering object.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        start_logits, end_logits = logits.split(1, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if start_positions.ndim > 1:
                start_positions = start_positions.squeeze(-1)
            if end_positions.ndim > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_loss = F.cross_entropy(
                start_logits, start_positions, ignore_index=ignored_index
            )
            end_loss = F.cross_entropy(
                end_logits, end_positions, ignore_index=ignored_index
            )
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


def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
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
    incremental_indices = (
        ops.cumsum(mask, dim=1).astype(mask.dtype) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


__all__ = [
    "RobertaForCausalLM",
    "RobertaForMaskedLM",
    "RobertaForMultipleChoice",
    "RobertaForQuestionAnswering",
    "RobertaForSequenceClassification",
    "RobertaForTokenClassification",
    "RobertaModel",
    "RobertaPreTrainedModel",
]
