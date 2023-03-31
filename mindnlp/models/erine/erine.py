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
'''erine model'''


#from dataclasses import dataclass
from typing import Optional


import mindspore

from mindspore import nn
from mindspore import ops

#from mindspore.common.initializer import initializer, TruncatedNormal, Normal
from mindspore import Tensor


from .erine_config import (
    #ERNIE_PRETRAINED_INIT_CONFIGURATION,
    #ERNIE_PRETRAINED_RESOURCE_FILES_MAP,
    ErnieConfig,
    #CONFIG_NAME
)


class ErnieEmbeddings(nn.Cell):
    r"""
    Include embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: ErnieConfig, weight_attr):
        super(ErnieEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            embedding_table=weight_attr
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embedding_table=weight_attr
        )

        self.type_vocab_size = config.type_vocab_size
        if self.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size, embedding_table=weight_attr
            )
        self.use_task_id = config.use_task_id
        self.task_id = config.task_id
        if self.use_task_id:
            self.task_type_embeddings = nn.Embedding(
                config.task_type_vocab_size, config.hidden_size, embedding_table=weight_attr
            )
        self.layer_norm = nn.LayerNorm(normalized_shape=[config.hidden_size],epsilon=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def construct(
            self,
            input_ids:               Optional[Tensor] = None,
            token_type_ids:          Optional[Tensor] = None,
            position_ids:            Optional[Tensor] = None,
            task_type_ids:           Optional[Tensor] = None,
            inputs_embeds:           Optional[Tensor] = None,
            past_key_values_length:  int = 0,
    ):
        r'''construct model'''

        if input_ids is not None:
            inputs_embeds = self.word_embeddings(input_ids)

        input_shape = ops.shape(inputs_embeds)[:-1]

        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = ops.ones(input_shape, dtype=mindspore.int64)
            seq_length = ops.cumsum(ones, axis=1)
            position_ids = seq_length - ones

            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length

            position_ids=ops.stop_gradient(position_ids)

        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings

        if self.type_vocab_size > 0:
            if token_type_ids is None:
                zeros=ops.Zeros()
                token_type_ids = zeros(input_shape, mindspore.int64)
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
        