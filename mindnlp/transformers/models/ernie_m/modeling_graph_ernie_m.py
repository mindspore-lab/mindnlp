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


from typing import List, Optional, Tuple

import numpy as np
import mindspore
from mindspore import ops, nn, Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.utils import logging
from mindnlp.modules.functional.graph_func import finfo
from ...activations import ACT2FN
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
class MSErnieMEmbeddings(nn.Cell):
    """Construct the embeddings from word and position embeddings."""
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieMEmbeddings class.
        
        Args:
            self: The object instance.
            config (object):
                A configuration object containing various parameters.

                - hidden_size (int): The size of the hidden state.
                - vocab_size (int): The size of the vocabulary.
                - pad_token_id (int): The ID of the padding token.
                - max_position_embeddings (int): The maximum number of positional embeddings.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability for the hidden state.

        Returns:
            None

        Raises:
            None
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
        Constructs the embeddings for MSErnieM model.

        Args:
            self (MSErnieMEmbeddings): The MSErnieMEmbeddings instance.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor containing the indices of input tokens. Default is None.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor containing the indices of position tokens. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor containing the embeddings of input tokens. Default is None.
            past_key_values_length (int): The length of past key values. Default is 0.

        Returns:
            mindspore.Tensor: The constructed embeddings tensor.

        Raises:
            ValueError: If the input_ids and inputs_embeds are both None.
            ValueError: If the input_shape is invalid for position_ids calculation.
            ValueError: If past_key_values_length is negative.
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
class MSErnieMSelfAttention(nn.Cell):

    """
    The `MSErnieMSelfAttention` class represents a self-attention mechanism for the MS ERNIE model.
    This class inherits from `nn.Cell`.

    This class implements the self-attention mechanism, which is a crucial component in natural language processing
    tasks like machine translation and text summarization. The self-attention mechanism allows the model to weigh the
    significance of different words in a sequence when processing each word, enabling the model to capture long-range
    dependencies and improve performance on various language understanding tasks.

    The class includes methods for initializing the self-attention mechanism, transposing input tensors for calculating
    attention scores, and constructing the self-attention mechanism using the provided input
    tensors. Additionally, it supports position embeddings and optional output of attention probabilities.

    The `MSErnieMSelfAttention` class ensures that the self-attention mechanism is efficiently implemented and
    seamlessly integrated into the MS ERNIE model, contributing to the model's effectiveness in natural language
    understanding and generation tasks.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes the MSErnieMSelfAttention instance.

        Args:
            self (MSErnieMSelfAttention): The MSErnieMSelfAttention instance.
            config (object): An object containing configuration settings for the self-attention mechanism.
            position_embedding_type (str, optional): The type of position embedding to be used, defaults to None.
                Possible values are 'absolute', 'relative_key', or 'relative_key_query'.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size in the configuration is not a multiple of the number of attention heads
                and the configuration does not have an 'embedding_size' attribute.
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
        Method transposes the input tensor for scores in a self-attention mechanism.

        Args:
            self (MSErnieMSelfAttention): An instance of the MSErnieMSelfAttention class.
            x (mindspore.Tensor): The input tensor to be transposed. It represents the scores to be processed.
                It is expected to have a shape compatible with the transposition operation.

        Returns:
            mindspore.Tensor: A new tensor obtained by transposing the input tensor for scores.
                The shape of the returned tensor is transformed based on the number of attention heads and head size.

        Raises:
            None
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
        Method to construct self-attention mechanism in the MSErnieMSelfAttention class.

        Args:
            self: The instance of the class.
            hidden_states (mindspore.Tensor): The input hidden states to the self-attention mechanism.
            attention_mask (Optional[mindspore.Tensor], optional):
                Mask tensor indicating which positions should be attended to and which should not. Defaults to None.
            head_mask (Optional[mindspore.Tensor], optional):
                Mask tensor indicating which heads to mask. Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor], optional):
                Hidden states from an encoder in case of cross-attention. Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor], optional): Mask tensor for encoder_hidden_states.
                Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]], optional):
                Tuple containing the past key and value tensors. Defaults to None.
            output_attentions (Optional[bool], optional): Flag to output attentions. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]:
                A tuple containing the context layer and attention probabilities if output_attentions is True,
                otherwise just the context layer.

        Raises:
            ValueError: If the position_embedding_type is not 'relative_key' or 'relative_key_query'.
            TypeError: If there are issues with the input types or dimensions during the computations.
            RuntimeError: If there are runtime issues during the self-attention mechanism.
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

        attention_scores = attention_scores / ops.sqrt(ops.scalar_to_tensor(self.attention_head_size, attention_scores.dtype))
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


class MSErnieMAttention(nn.Cell):

    """
    This class represents an attention module for MSErnieM model, which includes self-attention mechanism and projection
    layers.
    It inherits from nn.Cell and provides methods to initialize the attention module, prune attention heads, and perform
    attention computation.
    The attention module consists of self-attention mechanism with configurable position embedding type and projection
    layers for output transformation.
    The 'prune_heads' method allows pruning specific attention heads based on provided indices.
    The 'construct' method computes the attention output given input hidden states, optional masks, and other optional
    inputs.
    """
    def __init__(self, config, position_embedding_type=None):
        """
        Initializes an instance of the MSErnieMAttention class.

        Args:
            self: The instance of the class.
            config (object): An object that contains the configuration settings for the attention layer.
            position_embedding_type (str, optional): The type of position embedding to use. Defaults to None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.self_attn = MSErnieMSelfAttention(config, position_embedding_type=position_embedding_type)
        self.out_proj = nn.Dense(config.hidden_size, config.hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        """
        This method 'prune_heads' in the class 'MSErnieMAttention' prunes heads from the attention mechanism.

        Args:
            self (object): The instance of the class.
            heads (list): A list of integers representing the indices of heads to be pruned from the attention mechanism.

        Returns:
            None: This method does not return anything explicitly, as it operates by mutating the internal state of the class.

        Raises:
            ValueError: If the length of the 'heads' list is equal to 0.
            TypeError: If the 'heads' parameter is not a list of integers.
            IndexError: If the indices in 'heads' exceed the available attention heads in the mechanism.
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
        Constructs the MSErnieMAttention module.

        Args:
            self (MSErnieMAttention): The instance of the MSErnieMAttention class.
            hidden_states (mindspore.Tensor): The input hidden states of the model.
                Shape: (batch_size, seq_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor], optional):
                The attention mask tensor, indicating which tokens should be attended to and which should not.
                Shape: (batch_size, seq_length). Defaults to None.
            head_mask (Optional[mindspore.Tensor], optional):
                The head mask tensor, indicating which heads should be masked out.
                Shape: (num_heads, seq_length, seq_length). Defaults to None.
            encoder_hidden_states (Optional[mindspore.Tensor], optional):
                The hidden states of the encoder. Shape: (batch_size, seq_length, hidden_size). Defaults to None.
            encoder_attention_mask (Optional[mindspore.Tensor], optional):
                The attention mask tensor for the encoder, indicating which tokens should be attended to and which
                should not. Shape: (batch_size, seq_length). Defaults to None.
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]], optional):
                The tuple of past key and value tensors for keeping the previous attention weights.
                Shape: ((batch_size, num_heads, seq_length, hidden_size),
                (batch_size, num_heads, seq_length, hidden_size)). Defaults to None.
            output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the attention output tensor and other optional outputs.

        Raises:
            None.
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


class MSErnieMEncoderLayer(nn.Cell):

    """
    This class represents an encoder layer for the MSErnieM model. It includes self-attention, linear transformations,
    dropout, layer normalization, and activation functions for processing input hidden states.

    The MSErnieMEncoderLayer class inherits from nn.Cell and consists of an __init__ method for initializing the
    layer's components and a construct method for performing the encoding operations on input tensors.

    Attributes:
        self_attn (MSErnieMAttention): Self-attention mechanism for capturing dependencies within the input hidden states.
        linear1 (nn.Dense): Linear transformation layer from hidden size to intermediate size.
        dropout (nn.Dropout): Dropout layer for regularization during activation functions.
        linear2 (nn.Dense): Linear transformation layer from intermediate size back to hidden size.
        norm1 (nn.LayerNorm): Layer normalization for normalizing hidden states.
        norm2 (nn.LayerNorm): Layer normalization for normalizing hidden states.
        dropout1 (nn.Dropout): Dropout layer for regularization after the first linear transformation.
        dropout2 (nn.Dropout): Dropout layer for regularization after the second linear transformation.
        activation (function): Activation function applied to the hidden states.

    Methods:
        __init__: Constructor method for initializing the encoder layer with provided configuration settings.
        construct: Method for processing input hidden states through the encoder layer's components.

    The construct method performs a series of operations on the input hidden states, including self-attention,
    linear transformations, activation functions, dropout, and layer normalization. It returns the processed hidden
    states and optional attention outputs if specified.

    Note:
        The MSErnieMEncoderLayer class is designed to be used within the MSErnieM model architecture for encoding input sequences.
    """
    def __init__(self, config):
        """
        Initializes a MSErnieMEncoderLayer object with the provided configuration.

        Args:
            self (object): The MSErnieMEncoderLayer instance itself.
            config (object): An object containing configuration parameters for the encoder layer.
                This object should have the following attributes:

                - hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Default is 0.1.
                - act_dropout (float, optional): The dropout probability for the activation layers.
                Default is the value of hidden_dropout_prob.
                - hidden_size (int): The size of the hidden layers.
                - intermediate_size (int): The size of the intermediate layers.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_act (str or function): The activation function to use.
                If str, it should be a key in the ACT2FN dictionary.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        # to mimic paddlenlp implementation
        dropout = 0.1 if config.hidden_dropout_prob is None else config.hidden_dropout_prob
        act_dropout = config.hidden_dropout_prob if config.act_dropout is None else config.act_dropout

        self.self_attn = MSErnieMAttention(config)
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
        """Constructs the MSErnieMEncoderLayer.

        This method applies the MSErnieMEncoderLayer to the input hidden states.

        Args:
            self (MSErnieMEncoderLayer): The instance of the MSErnieMEncoderLayer class.
            hidden_states (mindspore.Tensor): The input hidden states.
                It is a tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor.
                It is an optional tensor of shape (batch_size, sequence_length).
            head_mask (Optional[mindspore.Tensor]): The head mask tensor.
                It is an optional tensor of shape (num_heads, sequence_length, sequence_length).
            past_key_value (Optional[Tuple[Tuple[mindspore.Tensor]]]): The past key-value tensor.
                It is an optional tuple of tuple of tensors.
            output_attentions (Optional[bool]): Whether to return attentions as well. Defaults to True.

        Returns:
            mindspore.Tensor or Tuple[mindspore.Tensor]: The output hidden states.
                If `output_attentions` is True, returns a tuple containing the hidden states and attentions.
                Otherwise, only returns the hidden states.

        Raises:
            None
        """
        residual = hidden_states
        outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
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
            return (hidden_states,) + outputs[1:]
        return hidden_states


class MSErnieMEncoder(nn.Cell):

    """
    This class represents an MSErnieMEncoder, which is a multi-layer transformer-based encoder model for
    natural language processing tasks.

    The MSErnieMEncoder inherits from the nn.Cell class and is designed to process input embeddings and generate
    hidden states, attentions, and last hidden state output.

    Attributes:
        config (object): The configuration object that contains the model's hyperparameters and settings.
        layers (nn.CellList): A list of MSErnieMEncoderLayer instances that make up the layers of the encoder.

    Methods:
        __init__(self, config):
            Initializes a new MSErnieMEncoder instance with the given configuration.

        construct(self, input_embeds, attention_mask=None, head_mask=None, past_key_values=None, output_attentions=False, output_hidden_states=False):
            Constructs the MSErnieMEncoder model by processing the input embeddings and generating the desired outputs.

            Args:

            - input_embeds (mindspore.Tensor): The input embeddings for the model.
            - attention_mask (Optional[mindspore.Tensor], optional): The attention mask tensor to mask
            certain positions. Defaults to None.
            - head_mask (Optional[mindspore.Tensor], optional): The head mask tensor to mask certain heads.
            Defaults to None.
            - past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]], optional): The cached key-value tensors
            from previous decoding steps. Defaults to None.
            - output_attentions (Optional[bool], optional): Whether to output attention weights. Defaults to False.
            - output_hidden_states (Optional[bool], optional): Whether to output hidden states. Defaults to False.

            Returns:

            - Tuple[mindspore.Tensor]: A tuple containing the last hidden state, hidden states, and attentions (if enabled).

        """
    def __init__(self, config):
        """
        Initializes the MSErnieMEncoder class.

        Args:
            self: The object itself.
            config (object): An object containing the configuration parameters for the MSErnieMEncoder.
                The config object should have the following attributes:

                - num_hidden_layers (int): The number of hidden layers in the encoder.
                - other attributes specific to the MSErnieMEncoderLayer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.config = config
        self.layers = nn.CellList([MSErnieMEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        input_embeds: mindspore.Tensor,
        attention_mask: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor]:
        """
        This method constructs the MSErnieMEncoder by processing the input embeddings and applying attention masks and
        head masks if provided.

        Args:
            self: The instance of the MSErnieMEncoder class.
            input_embeds (mindspore.Tensor): The input embeddings to be processed by the encoder.
            attention_mask (Optional[mindspore.Tensor]): An optional tensor representing the attention mask.
                If provided, it restricts the attention of the encoder.
            head_mask (Optional[mindspore.Tensor]): An optional tensor representing the head mask.
                If provided, it restricts the attention heads of the encoder.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]): An optional tuple of past key values,
                if provided, it allows the encoder to reuse previously computed key value states.
            output_attentions (Optional[bool]): An optional boolean indicating whether to output attentions.
                Default is False.
            output_hidden_states (Optional[bool]): An optional boolean indicating whether to output hidden states.
                Default is False.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the processed output tensor.

        Raises:
            ValueError: If the input_embeds parameter is not of type mindspore.Tensor.
            ValueError: If the attention_mask parameter is not of type Optional[mindspore.Tensor].
            ValueError: If the head_mask parameter is not of type Optional[mindspore.Tensor].
            ValueError: If the past_key_values parameter is not of type Optional[Tuple[Tuple[mindspore.Tensor]]].
            ValueError: If the output_attentions parameter is not of type Optional[bool].
            ValueError: If the output_hidden_states parameter is not of type Optional[bool].
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
        return tuple(v for v in [last_hidden_state, hidden_states, attentions] if v is not None)


# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->ErnieM
class MSErnieMPooler(nn.Cell):

    """A class representing a pooling layer for the MSErnieM model in MindSpore.

    This class is responsible for constructing the pooling layer of the MSErnieM model.
    The pooling layer takes the hidden states of the model as input and applies a dense layer followed by
    an activation function to the first token tensor. The resulting pooled output is returned.

    Attributes:
        dense (nn.Dense): A dense layer used in the pooling layer.
        activation (nn.Tanh): An activation function used in the pooling layer.

    Methods:
        __init__: Initializes the MSErnieMPooler instance.
        construct: Constructs the pooling layer.

    """
    def __init__(self, config):
        """
        Initializes a new instance of the MSErnieMPooler class.

        Args:
            self: The object itself.
            config:
                An instance of the configuration class for MSErnieMPooler.

                - Type: Any valid configuration class.
                - Purpose: Specifies the configuration settings for the MSErnieMPooler instance.
                - Restrictions: None.

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
        Constructs the pooled output tensor from the provided hidden states.

        Args:
            self (MSErnieMPooler): The instance of the MSErnieMPooler class.
            hidden_states (mindspore.Tensor): The input tensor representing the hidden states of the input sequence.
                It should be of shape (batch_size, sequence_length, hidden_size).

        Returns:
            mindspore.Tensor: The pooled output tensor obtained from the hidden states.
                It is a 2D tensor of shape (batch_size, hidden_size) representing the pooled output features.

        Raises:
            ValueError: If the shape of the input hidden_states tensor is not as expected.
            TypeError: If the input hidden_states is not a mindspore.Tensor object.
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MSErnieMPreTrainedModel(PreTrainedModel):
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


class MSErnieMModel(MSErnieMPreTrainedModel):

    """
    This class represents the MSErnieMModel, which is a variant of the MSErnieMPreTrainedModel.
    It is a model for sequence classification tasks, built on top of the MSErnieM language model.

    The MSErnieMModel class includes methods for initializing the model, getting and setting input embeddings,
    pruning model heads, and constructing the model.

    Methods:
        __init__: Initializes the MSErnieMModel with the given configuration.
            By default, it adds a pooling layer to the model.
        get_input_embeddings: Returns the word embeddings used as input to the model.
        set_input_embeddings: Sets the word embeddings used as input to the model.
        _prune_heads: Prunes the specified heads in the model.
        construct: Constructs the model with the given input and configuration.

    Note:
        The MSErnieMModel class inherits from the MSErnieMPreTrainedModel, which provides additional functionality
        and methods.

    Example:
        ```python
        >>> config = MSErnieMConfig()
        >>> model = MSErnieMModel(config)
        >>> input_ids = ...
        >>> position_ids = ...
        >>> attention_mask = ...
        >>> output = model.construct(input_ids, position_ids, attention_mask)
        ```
    """
    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a new MSErnieMModel instance.

        Args:
            self: The instance of the MSErnieMModel class.
            config:
                An object containing configuration settings for the model.

                - Type: object
                - Purpose: Specifies the configuration settings for the model.
            add_pooling_layer:
                A boolean flag indicating whether to add a pooling layer.

                - Type: bool
                - Purpose: Specifies whether to include a pooling layer in the model.
                - Restrictions: Must be a boolean value.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.initializer_range = config.initializer_range
        self.embeddings = MSErnieMEmbeddings(config)
        self.encoder = MSErnieMEncoder(config)
        self.pooler = MSErnieMPooler(config) if add_pooling_layer else None
        self.post_init()

    def get_input_embeddings(self):
        """
        Method: get_input_embeddings

        Description:
            This method returns the input embeddings from the MSErnieMModel class.

        Args:
            self: MSErnieMModel
                The instance of the MSErnieMModel class.

                - Type: MSErnieMModel object
                - Purpose: To access the embeddings from the model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input embeddings for the MSErnieMModel.

        Args:
            self (MSErnieMModel): The instance of the MSErnieMModel.
            value (object): The input embeddings to be set. It can be of any type.

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
    ) -> Tuple[mindspore.Tensor]:
        '''
        Constructs the MSErnieMModel.

        Args:
            self: The object itself.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor containing the indices of input sequence tokens in the vocabulary.
            position_ids (Optional[mindspore.Tensor]):
                The input tensor containing the position indices of each input sequence token in the sequence.
            attention_mask (Optional[mindspore.Tensor]):
                The input tensor containing the attention mask to avoid performing attention on padding tokens.
            head_mask (Optional[mindspore.Tensor]):
                The input tensor containing the mask to nullify selected heads of the self-attention modules.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input tensor containing the embedded representation of the input sequence.
            past_key_values (Optional[Tuple[Tuple[mindspore.Tensor]]]):
                The input tensor containing the cached key and value tensors of the self-attention mechanism.
            use_cache (Optional[bool]): Whether to use the cache for the decoding steps of the model.
            output_hidden_states (Optional[bool]): Whether to return the hidden states of all layers.
            output_attentions (Optional[bool]): Whether to return the attention weights.

        Returns:
            Tuple[mindspore.Tensor]: A tuple containing the output sequence tensor, the pooled output tensor,
                and other encoded outputs.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided.

        '''
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        # Adapted from paddlenlp.transformers.ernie_m.ErnieMModel
        if attention_mask is None:
            attention_mask = (input_ids == 0).to(self.dtype)
            attention_mask = attention_mask * finfo(attention_mask.dtype, 'min')
            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = ops.zeros([batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = ops.concat([past_mask, attention_mask], axis=-1)
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = attention_mask.to(self.dtype)
            attention_mask = 1.0 - attention_mask
            attention_mask = attention_mask * finfo(attention_mask.dtype, 'min')

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
        )

        sequence_output = encoder_outputs[0]
        pooler_output = self.pooler(sequence_output) if self.pooler is not None else None
        return (sequence_output, pooler_output) + encoder_outputs[1:]


class MSErnieMForSequenceClassification(MSErnieMPreTrainedModel):

    """
    This class represents a modified version of the MSErnieM model for sequence classification tasks.
    It inherits from the MSErnieMPreTrainedModel class.

    Attributes:
        num_labels (int): The number of labels for the sequence classification task.
        config (MSErnieMConfig): The configuration object for the model.
        ernie_m (MSErnieMModel): The MSErnieM model.
        dropout (nn.Dropout): The dropout layer for regularization.
        classifier (nn.Dense): The dense layer for classification.

    Methods:
        __init__: Initializes the MSErnieMForSequenceClassification instance.
        construct: Constructs the model and computes the loss and logits for the given input.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes an instance of the 'MSErnieMForSequenceClassification' class.

        Args:
            self: The instance of the class.
            config:
                An object of type 'Config' containing the configuration parameters for the model.

                - Type: Config
                - Purpose: Specifies the configuration of the model.
                - Restrictions: None

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.ernie_m = MSErnieMModel(config)
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
        labels: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
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

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MSErnieMForMultipleChoice(MSErnieMPreTrainedModel):

    """
    This class represents a Multiple Choice classification model based on the MSErnieM architecture.
    It inherits from the MSErnieMPreTrainedModel and is designed to facilitate multiple choice question answering tasks.

    The class implements the initialization method to set up the model and a construct method to process input data and
    produce classification predictions. The construct method handles input tensors for input_ids, attention_mask,
    position_ids, head_mask, inputs_embeds, and labels, and provides options for output_attentions and output_hidden_states.

    The construct method computes the multiple choice classification loss based on the input data and generates reshaped
    logits for each choice. It utilizes the MSErnieM model to process the input data and applies dropout and dense layers
    for classification. Additionally, it handles the cross-entropy loss calculation for training the model.

    Overall, the MSErnieMForMultipleChoice class encapsulates the functionality for performing multiple choice
    classification using the MSErnieM architecture and provides flexibility for processing various input tensors and
    generating classification predictions.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes an instance of MSErnieMForMultipleChoice.

        Args:
            self (object): The instance of the class.
            config (object): The configuration object containing various parameters for the model initialization.

        Returns:
            None.

        Raises:
            ValueError: If the provided configuration object is invalid or missing required parameters.
            TypeError: If the configuration parameters are of incorrect type.
        """
        super().__init__(config)

        self.ernie_m = MSErnieMModel(config)
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
    ) -> Tuple[mindspore.Tensor]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
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
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(reshaped_logits, labels)

        output = (reshaped_logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MSErnieMForTokenClassification(MSErnieMPreTrainedModel):

    """
    This class represents a token classification model based on MSErnieM architecture.
    It is designed for tasks that involve assigning labels to individual tokens within a sequence.

    The `MSErnieMForTokenClassification` class inherits from `MSErnieMPreTrainedModel` and extends its functionality
    by adding a token classification layer on top of the base model.

    The class's constructor initializes the model and sets up the necessary components.
    It takes a `config` object as input and initializes the base model with the provided configuration.
    The number of labels for token classification is also stored for later use.
    The dropout layer and the token classification layer are defined. Lastly, the `post_init` method is called to
    perform any additional initialization steps.

    The `construct` method is the main entry point for using the model for token classification.
    It takes various input tensors such as `input_ids`, `attention_mask`, `position_ids`, `head_mask`,
    `inputs_embeds`, `past_key_values`, `output_hidden_states`, `output_attentions`, and `labels`.

    The method first passes the input tensors through the base model (`self.ernie_m`) to obtain the sequence output.
    The sequence output is then passed through a dropout layer to prevent overfitting.
    Finally, the token classification layer (`self.classifier`) is applied to generate logits for each token in the
    sequence.

    If `labels` are provided, the method calculates the token classification loss using the cross-entropy function.
    The loss is computed by reshaping the logits and labels tensors to have a shape of
    `(batch_size * sequence_length, num_labels)` and applying the cross-entropy function.

    The method returns a tuple containing the logits for each token, as well as any additional outputs from the base model.
    If a loss is calculated, it is included in the output tuple.

    Note:
        The `MSErnieMForTokenClassification` class assumes that the input tensors are of type `mindspore.Tensor`,
        and the labels tensor should have a shape of `(batch_size, sequence_length)` with indices in the range
        `[0, ..., config.num_labels - 1]`.

    """
    # Copied from transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes a new instance of the MSErnieMForTokenClassification class.

        Args:
            self: The instance of the class.
            config:
                An object containing configuration parameters for the model.

                - Type: dict
                - Purpose: Configuration settings for the model.
                - Restrictions: Must contain the key 'num_labels'.

        Returns:
            None.

        Raises:
            TypeError: If the 'config' parameter is not provided or is not of type dict.
            KeyError: If the 'num_labels' key is missing in the 'config' parameter.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie_m = MSErnieMModel(config, add_pooling_layer=False)
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
        labels: Optional[mindspore.Tensor] = None,
    ) -> Tuple[mindspore.Tensor]:
        r"""
        Args:
            labels (`mindspore.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class MSErnieMForQuestionAnswering(MSErnieMPreTrainedModel):

    """
    MSErnieMForQuestionAnswering represents a model for question answering tasks using the MSErnieM architecture.
    This class inherits from MSErnieMPreTrainedModel and implements methods for initializing the model and constructing
    outputs for question answering.

    Attributes:
        num_labels (int): The number of labels for token classification.
        ernie_m (MSErnieMModel): The MSErnieMModel instance used for processing inputs.
        qa_outputs (nn.Dense): Dense layer for outputting logits for question answering.

    Methods:
        __init__: Initializes the MSErnieMForQuestionAnswering instance with the provided configuration.
        construct:
            Constructs the question answering outputs based on the input tensors and labels provided.

        Note:
            The start_positions and end_positions parameters are used for computing the token classification loss by
            providing labels for the start and end positions of the labelled span in the input sequence.
            Position indices are clamped to the length of the sequence and positions outside of the sequence
            are not considered for loss computation.
    """
    # Copied from transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ with Bert->ErnieM,bert->ernie_m
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieMForQuestionAnswering class.

        Args:
            self: The instance of the class.
            config:
                An instance of the configuration class containing the model configuration.

                - Type: object
                - Purpose: To provide the configuration settings for the model initialization.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None

        Raises:
            TypeError: If the provided config parameter is not of the expected type.
            ValueError: If the config parameter is missing essential attributes.
            RuntimeError: If an error occurs during initialization or post-initialization steps.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.ernie_m = MSErnieMModel(config, add_pooling_layer=False)
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
    ) -> Tuple[mindspore.Tensor]:
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
        outputs = self.ernie_m(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


# Copied from paddlenlp.transformers.ernie_m.modeling.UIEM
class MSErnieMForInformationExtraction(MSErnieMPreTrainedModel):

    """
    The 'MSErnieMForInformationExtraction' class is a model for information extraction tasks using the MSERNIE-M
    (multi-lingual) model. It extends the 'MSErnieMPreTrainedModel' class.

    This class initializes the MSERNIE-M model and includes methods for constructing the model for information
    extraction tasks, such as computing start and end position losses and probabilities. It also provides functionality
    for calculating the total loss, start probability, and end probability.

    The 'MSErnieMForInformationExtraction' class inherits the configuration parameters and methods from
    'MSErnieMPreTrainedModel' and extends it to support information extraction tasks. The class is designed to handle
    input tensors for input_ids, attention_mask, position_ids, head_mask, and inputs_embeds, and provides output in the
    form of a tuple containing total loss, start probability, end probability, and additional model outputs.

    The class is suitable for tasks such as named entity recognition, question answering, and other information
    extraction tasks where start and end positions within a sequence need to be identified and predicted.
    
    This class is a part of the MindSpore library and is designed to provide a high-level interface for utilizing
    the MSERNIE-M model for information extraction tasks.
    """
    def __init__(self, config):
        """
        Initializes an instance of the MSErnieMForInformationExtraction class.
        
        Args:
            self (MSErnieMForInformationExtraction): The instance of the MSErnieMForInformationExtraction class.
            config (object): The configuration object for the model.
        
        Returns:
            None.
        
        Raises:
            TypeError: If the config parameter is not of the expected type.
            ValueError: If the config parameter does not contain the required attributes.
        """
        super().__init__(config)
        self.ernie_m = MSErnieMModel(config)
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
    ) -> Tuple[mindspore.Tensor]:
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
        )

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

        return (total_loss, start_prob, end_prob) + result[1:]


class MSUIEM(MSErnieMForInformationExtraction):
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
    ) -> Tuple[mindspore.Tensor]:
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
        )
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

        output = (start_prob, end_prob) + result[1:]
        return ((total_loss,) + output) if total_loss is not None else output

__all__ = [
    "ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST",
    "MSErnieMForMultipleChoice",
    "MSErnieMForQuestionAnswering",
    "MSErnieMForSequenceClassification",
    "MSErnieMForTokenClassification",
    "MSErnieMModel",
    "MSErnieMPreTrainedModel",
    "MSErnieMForInformationExtraction",
    "MSUIEM"
]
