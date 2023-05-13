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
# pylint: disable=C0103
# pylint: disable=C0302
"""
MindNlp LUKE model
"""
import inspect
import math
from typing import Callable, Optional, Tuple

import mindspore
import numpy as np
from mindspore import nn
from mindspore import ops, Tensor
from mindspore.common.initializer import Normal, initializer

from mindnlp.models.luke.luke_config import LukeConfig
from ..utils import logging
from ..utils.activations import ACT2FN
from ...abc import PreTrainedModel

logger = logging.get_logger(__name__)


class LukeEmbeddings(nn.Cell):
    """
    LukeEmbeddings
    """

    def __init__(self, config: LukeConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], epsilon=config.layer_norm_eps)
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


class LukeEntityEmbeddings(nn.Cell):
    """
    LukeEntityEmbeddings
    """

    def __init__(self, config: LukeConfig):
        super().__init__()
        self.config = config

        self.entity_embeddings = nn.Embedding(config.entity_vocab_size, config.entity_emb_size, padding_idx=0)
        if config.entity_emb_size != config.hidden_size:
            self.entity_embedding_dense = nn.Dense(config.entity_emb_size, config.hidden_size, has_bias=False)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm([config.hidden_size, ], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(
            self, entity_ids, position_ids, token_type_ids=None
    ):
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(entity_ids)

        entity_embeddings = self.entity_embeddings(entity_ids)
        if self.config.entity_emb_size != self.config.hidden_size:
            entity_embeddings = self.entity_embedding_dense(entity_embeddings)

        position_embeddings = self.position_embeddings(ops.clamp(position_ids, min=0))
        position_embedding_mask = Tensor(position_ids != -1, dtype=position_embeddings.dtype).unsqueeze(-1)
        position_embeddings = position_embeddings * position_embedding_mask
        position_embeddings = position_embeddings.sum(axis=-2)
        position_embeddings = position_embeddings / position_embedding_mask.sum(axis=-2).clamp(min=1e-7)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = entity_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LukeSelfAttention(nn.Cell):
    """
    LukeSelfAttention
    """

    def __init__(self, config):
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

        self.query = nn.Dense(config.hidden_size, self.all_head_size)
        self.key = nn.Dense(config.hidden_size, self.all_head_size)
        self.value = nn.Dense(config.hidden_size, self.all_head_size)

        if self.use_entity_aware_attention:
            self.w2e_query = nn.Dense(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Dense(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Dense(config.hidden_size, self.all_head_size)

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


class LukeSelfOutput(nn.Cell):
    """
    LukeSelfOutput
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class LukeAttention(nn.Cell):
    """
    LukeAttention
    """

    def __init__(self, config):
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


class LukeIntermediate(nn.Cell):
    """
    LukeIntermediate
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LukeOutput(nn.Cell):
    """
    LukeOutput
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class LukeLayer(nn.Cell):
    """
    LukeOutput
    """

    def __init__(self, config):
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


class LukeEncoder(nn.Cell):
    """
    LukeEncoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.CellList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
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


class LukePooler(nn.Cell):
    """
    LukePooler
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states: Tensor) -> Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class EntityPredictionHeadTransform(nn.Cell):
    """
    EntityPredictionHeadTransform
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.entity_emb_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm([config.entity_emb_size, ], epsilon=config.layer_norm_eps)

    def construct(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


# 0325==============================
class EntityPredictionHead(nn.Cell):
    """
    EntityPredictionHead
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transform = EntityPredictionHeadTransform(config)
        self.decoder = nn.Dense(config.entity_emb_size, config.entity_vocab_size, has_bias=False)
        self.bias = mindspore.Parameter(ops.zeros((config.entity_vocab_size,)))

    def construct(self, hidden_states):
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

    def get_input_embeddings(self) -> "nn.Cell":
        pass

    def set_input_embeddings(self, value: "nn.Cell"):
        pass

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        pass

    def get_position_embeddings(self):
        pass

    def _init_weights(self, cell: nn.Cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Dense):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.has_bias:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            if cell.embedding_size == 1:  # embedding for bias parameters
                cell.embedding_table.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            else:
                embedding_table = initializer(Normal(self.config.initializer_range),
                                                        cell.embedding_table.shape,
                                                        cell.embedding_table.dtype)
                if cell.padding_idx is not None:
                    embedding_table[cell.padding_idx] = 0
                cell.embedding_table.set_data(embedding_table)
        elif isinstance(cell, nn.LayerNorm):
            cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
            cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))


class LukeModel(LukePreTrainedModel):
    """
    LukeModel
    """
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config: LukeConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = LukeEmbeddings(config)
        self.entity_embeddings = LukeEntityEmbeddings(config)
        self.encoder = LukeEncoder(config)

        self.pooler = LukePooler(config) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_entity_embeddings(self):
        """get_entity_embeddings"""
        return self.entity_embeddings.entity_embeddings

    def set_entity_embeddings(self, value):
        """set_entity_embeddings"""
        self.entity_embeddings.entity_embeddings = value

    def _prune_heads(self, heads_to_prune):
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


class LukeLMHead(nn.Cell):
    """LukeLMead"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm([config.hidden_size, ], epsilon=config.layer_norm_eps)

        self.decoder = nn.Dense(config.hidden_size, config.vocab_size)
        self.bias = mindspore.Parameter(ops.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def construct(self, features, **kwargs):
        # hidden
        x = self.dense(features)
        x = ops.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        # endecoded
        x = self.decoder(x)

        return x

    def _tie_weights(self):
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
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

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
                loss = ops.cross_entropy(logits, labels)
            else:
                loss = ops.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), Tensor(),
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
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size * 2, config.num_labels, has_bias=False)

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
                loss = ops.cross_entropy(logits, labels)
            else:
                loss = ops.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits), Tensor(),
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
        super().__init__(config)

        self.luke = LukeModel(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size * 3, config.num_labels)

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
                loss = ops.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = ops.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1).type_as(logits),
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
        super().__init__(config)
        self.num_labels = config.num_labels
        self.luke = LukeModel(config)
        self.dropout = nn.Dropout(p=
                                  config.classifier_dropout
                                  if config.classifier_dropout is not None
                                  else config.hidden_dropout_prob
                                  )
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

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
        super().__init__(config)
        self.num_labels = config.num_labels

        self.luke = LukeModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(p=
                                  config.classifier_dropout
                                  if config.classifier_dropout is not None
                                  else config.hidden_dropout_prob
                                  )
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

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
        super().__init__(config)

        self.num_labels = config.num_labels

        self.luke = LukeModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Dense(config.hidden_size, config.num_labels)

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
        super().__init__(config)

        self.luke = LukeModel(config)
        self.dropout = nn.Dropout(p=
                                  config.classifier_dropout
                                  if config.classifier_dropout is not None
                                  else config.hidden_dropout_prob
                                  )
        self.classifier = nn.Dense(config.hidden_size, 1)

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
