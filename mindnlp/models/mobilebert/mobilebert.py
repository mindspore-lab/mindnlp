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
# pylint: disable=C0301
"""
MobileBert model
"""

import math
from typing import Optional, Tuple

import mindspore
import mindspore.numpy as mnp
from mindspore import Parameter, Tensor
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import TruncatedNormal
from mindnlp._legacy.nn import Dropout


class NoNorm(nn.Cell):
    """NoNorm"""
    def __init__(self, feat_size):
        super().__init__()
        self.bias = Parameter(ops.zeros(feat_size))
        self.weight = Parameter(ops.ones(feat_size))

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

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, embedding_table=TruncatedNormal(config.initializer_range))
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
            position_ids = mnp.arange(seq_length)
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
                    ops.pad(inputs_embeds[:, 1:], (0, 0, 0, 1, 0, 0)),
                    inputs_embeds,
                    ops.pad(inputs_embeds[:, :-1], (0, 0, 1, 0, 0, 0)),
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
