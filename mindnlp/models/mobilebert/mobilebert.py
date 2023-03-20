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
# pylint: disable=C0103
"""
MobileBert model
"""

import math
from typing import Optional, Tuple

import mindspore
from mindspore import Parameter, Tensor
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import TruncatedNormal
from mindnlp._legacy.nn import Dropout
from ..utils.utils import find_pruneable_heads_and_indices,prune_conv1d_layer
from ..utils.activations import ACT2FN


class NoNorm(nn.Cell):
    """NoNorm"""
    def __init__(self, feat_size):
        super().__init__()
        self.bias = Parameter(ops.zeros(feat_size, mindspore.float32))
        self.weight = Parameter(ops.ones(feat_size, mindspore.float32))

    def construct(self, input_tensor: Tensor) -> Tensor:
        return input_tensor * self.weight + self.bias

NORM2FN = {"layer_norm": nn.LayerNorm, "no_norm": NoNorm}


class MobileBertEmbeddings(nn.Cell):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size,
                                            embedding_table=TruncatedNormal(config.initializer_range))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        self.embedding_transformation = nn.Dense(embedded_input_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dropout = Dropout(config.hidden_dropout_prob)


    def construct(
            self,
            input_ids: Optional[Tensor] = None,
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = ops.arange(seq_length)
            position_ids = position_ids.expand_dims(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = ops.zeros(input_shape, mindspore.int64)
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
                axis=2
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

class MobileBertSelfAttention(nn.Cell):
    """MobileBertSelfAttention"""
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.true_hidden_size, self.all_head_size)
        self.key = nn.Dense(config.true_hidden_size, self.all_head_size)
        self.value = nn.Dense(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """transpose_for_scores"""
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def construct(
        self,
        query_tensor:Tensor,
        key_tensor:Tensor,
        value_tensor:Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[Tensor]:
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = ops.matmul(query_layer, key_layer.transpose((0,1,-1, -2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class MobileBertSelfOutput(nn.Cell):
    """MobileBertSelfOutput"""
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Dense(config.true_hidden_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, residual_tensor: Tensor) -> Tensor:
        layer_outputs = self.dense(hidden_states)
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs

class MobileBertAttention(nn.Cell):
    """MobileBertAttention"""
    def __init__(self, config):
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
        self.output.dense = prune_conv1d_layer(self.output.dense, index, axis=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        query_tensor: Tensor,
        key_tensor: Tensor,
        value_tensor: Tensor,
        layer_input: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[Tensor]:
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

class MobileBertIntermediate(nn.Cell):
    """MobileBertIntermediate"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.true_hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class OutputBottleneck(nn.Cell):
    """OutputBottleneck"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.true_hidden_size, config.hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size)
        self.dropout = Dropout(config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, residual_tensor: Tensor) -> Tensor:
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.dropout(layer_outputs)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs

class MobileBertOutput(nn.Cell):
    """MobileBertOutput"""
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Dense(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        if not self.use_bottleneck:
            self.dropout = Dropout(config.hidden_dropout_prob)
        else:
            self.bottleneck = OutputBottleneck(config)

    def construct(
        self, intermediate_states: Tensor, residual_tensor_1: Tensor, residual_tensor_2: Tensor
    ) -> Tensor:
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        return layer_output

class BottleneckLayer(nn.Cell):
    """BottleneckLayer"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.intra_bottleneck_size)

    def construct(self, hidden_states: Tensor) -> Tensor:
        layer_input = self.dense(hidden_states)
        layer_input = self.LayerNorm(layer_input)
        return layer_input

class Bottleneck(nn.Cell):
    """Bottleneck"""
    def __init__(self, config):
        super().__init__()
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        self.input = BottleneckLayer(config)
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)

    def construct(self, hidden_states: Tensor) -> Tuple[Tensor]:
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
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)

class FFNOutput(nn.Cell):
    """FFNOutput"""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)

    def construct(self, hidden_states: Tensor, residual_tensor: Tensor) -> Tensor:
        layer_outputs = self.dense(hidden_states)
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs

class FFNLayer(nn.Cell):
    """FFNLayer"""
    def __init__(self, config):
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def construct(self, hidden_states: Tensor) -> Tensor:
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs

class MobileBertLayer(nn.Cell):
    """MobileBertLayer"""
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks

        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        if config.num_feedforward_networks > 1:
            self.ffn = nn.CellList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[Tensor]:
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

class MobileBertEncoder(nn.Cell):
    """MobileBertEncoder"""
    def __init__(self, config):
        super().__init__()
        self.layer = nn.CellList([MobileBertLayer(config) for _ in range(config.num_hidden_layers)])

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Tuple:
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
        return (hidden_states, all_hidden_states, all_attentions)
