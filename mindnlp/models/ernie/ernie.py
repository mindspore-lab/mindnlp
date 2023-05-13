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

"""
Ernie Models
"""

from typing import Optional, Tuple

import mindspore
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import TruncatedNormal
from mindnlp.abc import PreTrainedModel
from .ernie_config import ErnieConfig, ERNIE_PRETRAINED_INIT_CONFIGURATION, ERNIE_PRETRAINED_RESOURCE_FILES_MAP


__all__ = ['ErnieEmbeddings', 'ErnieModel', 'ErniePooler', "UIE"]

class ErnieEmbeddings(nn.Cell):
    """
    Ernie Embeddings for word, position and token_type embeddings.
    """

    def __init__(self, config: ErnieConfig, embedding_table):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id, embedding_table=embedding_table
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, embedding_table=embedding_table
        )
        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size, embedding_table=embedding_table
            )
        self.use_task_id = config.use_task_id
        self.task_id = config.task_id
        if self.use_task_id:
            self.task_type_embeddings = nn.Embedding(
                config.task_type_vocab_size, config.hidden_size, embedding_table=embedding_table
            )
        self.layer_norm = nn.LayerNorm([config.hidden_size])
        self.dropout = nn.Dropout(config.hidden_dropout_prob, p=0.5)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        task_type_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        past_key_values_length: int = 0,
    ):

        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        input_shape = ops.shape(inputs_embeds)[:-1]

        if position_ids is None:
            ones = ops.ones(input_shape, mindspore.int64)
            seq_length = ops.cumsum(ones, axis=1)
            position_ids = seq_length - ones

            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length

            position_ids.stop_gradient = True

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                token_type_ids = ops.zeros(input_shape, mindspore.int64)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings

        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = ops.ones(
                    input_shape, mindspore.int64) * self.task_id
            task_type_embeddings = self.task_type_embeddings(task_type_ids)
            embeddings = embeddings + task_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ErniePretrainedModel(PreTrainedModel):
    """
    Ernie Pretrained Model.
    """

    pretrained_init_configuration = ERNIE_PRETRAINED_INIT_CONFIGURATION
    pretrained_resource_files_map = ERNIE_PRETRAINED_RESOURCE_FILES_MAP

    # TODO
    def get_input_embeddings(self):
        pass

    #TODO
    def get_position_embeddings(self):
        pass

    #TODO
    def resize_position_embeddings(self):
        pass

    #TODO
    def set_input_embeddings(self):
        pass

    def init_weights(self, layer):
        """Initialization hook"""
        if isinstance(layer, nn.Dense):
            if isinstance(layer.weight, mindspore.Tensor):
                layer.weight.set_data(
                    ops.normal(
                        mean=0.0,
                        stddev=self.config.initializer_range,
                        shape=layer.weight.shape,
                    )
                )
        if isinstance(layer, nn.Embedding):
            if isinstance(layer.embedding_table, mindspore.Tensor):
                layer.embedding_table.set_data(
                    ops.normal(
                        mean=0.0,
                        stddev=self.config.initializer_range,
                        shape=layer.embedding_table.shape,
                    )
                )
        elif isinstance(layer, nn.LayerNorm):
            layer.epsilon = 1e-12


class ErniePooler(nn.Cell):
    """
    Ernie Pooler.
    """
    def __init__(self, config: ErnieConfig, weight_init):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size,
                              config.hidden_size, weight_init=weight_init)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ErnieModel(ErniePretrainedModel):
    """
    Ernie model.
    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.initializer_range = config.initializer_range
        self.nheads = config.num_attention_heads
        embedding_table = TruncatedNormal(sigma=self.initializer_range)
        self.embeddings = ErnieEmbeddings(
            config=config, embedding_table=embedding_table)
        encoder_layer = nn.TransformerEncoderLayer(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.num_hidden_layers
        )
        self.pooler = ErniePooler(config, weight_init=embedding_table)
        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        task_type_ids: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        inputs_embeds: Optional[Tensor] = None,
        # use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        batch_size, seq_length = input_ids.shape

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time.")

        # init the default bool value
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else False
        # use_cache = use_cache if use_cache is not None else False
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if attention_mask is None:
            attention_mask = ((input_ids == self.pad_token_id).astype(
                self.pooler.dense.weight.dtype) * -1e4).unsqueeze(1).unsqueeze(2)

            if past_key_values is not None:
                batch_size = past_key_values[0][0].shape[0]
                past_mask = ops.zeros(
                    [batch_size, 1, 1, past_key_values_length], dtype=attention_mask.dtype)
                attention_mask = ops.concat(
                    [past_mask, attention_mask], axis=-1)

            attention_mask = ops.tile(
                attention_mask, (1, self.nheads, seq_length, 1)).reshape(-1, seq_length, seq_length)
        # For 2D attention_mask from tokenizer
        elif attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -1e4
            attention_mask = ops.tile(
                attention_mask, (1, self.nheads, seq_length, 1)).reshape(-1, seq_length, seq_length)

        attention_mask.stop_gradient = True

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_type_ids=task_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        # self.encoder._use_cache = use_cache  # To be consistent with HF
        encoder_outputs = self.encoder(
            embedding_output,
            src_mask=attention_mask,
        )
        if isinstance(encoder_outputs, type(embedding_output)):
            sequence_output = encoder_outputs
            pooled_output = self.pooler(sequence_output)
            return (sequence_output, pooled_output)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class UIE(ErniePretrainedModel):
    """
    UIE model based on Ernie.
    """

    def __init__(self, config: ErnieConfig):
        super().__init__(config)
        self.ernie = ErnieModel(config)
        self.linear_start = nn.Dense(config.hidden_size, 1)
        self.linear_end = nn.Dense(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.apply(self.init_weights)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        return_dict: Optional[Tensor] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        sequence_output, _ = self.ernie(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
        start_logits = self.linear_start(sequence_output)
        start_logits = ops.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = ops.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)
        return start_prob, end_prob
