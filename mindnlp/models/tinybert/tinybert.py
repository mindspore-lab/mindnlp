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
TinyBert Models
"""

import mindspore
from mindspore import nn, ops
from mindnlp._legacy.nn import Dropout
from mindnlp._legacy.functional import arange


class BertEmbeddings(nn.Cell):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        """
        init BertEmbeddings
        """

        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.layer_norm = nn.LayerNorm([config.hidden_size], epsilon=1e-12)
        self.dropout = Dropout(p=config.hidden_dropout_prob)

    def construct(self, input_ids, token_type_ids=None):
        """
        Construct the embeddings from word, position and token_type embeddings.
        """

        seq_length = input_ids.shape[1]
        position_ids = arange(seq_length, dtype=mindspore.int64)
        position_ids = position_ids.expand_dims(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = ops.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == "__main__":
    pass
