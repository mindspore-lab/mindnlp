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
RNN-based sentimental classification model
"""
# pylint: disable=abstract-method
# pylint: disable=arguments-differ

import math
import pytest
import numpy as np

import mindspore
from mindspore import nn
from mindspore import ops
import mindspore.dataset as ds
from mindspore.common.initializer import Uniform, HeUniform

from mindnlp.abc import Seq2vecModel
from mindnlp.modules import RNNEncoder
from mindnlp.common.metrics import Accuracy
from mindnlp.engine.trainer import Trainer
from mindnlp.dataset import load
from mindnlp.modules import Glove


class Dataset:
    """
    Dataset
    """
    def __init__(self):
        self.src_tokens = np.random.randn(12500, 500).astype(np.int32)
        self.labels = np.random.randint(0, 1, (12500, 1)).astype(np.float32)
    def __getitem__(self, index):
        return self.src_tokens[index], self.labels[index]
    def __len__(self):
        return len(self.src_tokens)


class Head(nn.Cell):
    """
    Head for Sentiment Classification model
    """
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)
        self.sigmoid = nn.Sigmoid()

    def construct(self, context):
        context = ops.concat((context[-2, :, :], context[-1, :, :]), axis=1)
        output = self.fc(context)
        return self.sigmoid(output)


class SentimentClassification(Seq2vecModel):
    """
    Sentiment Classification model
    """
    def __init__(self, encoder, head, dropout):
        super().__init__(encoder, head, dropout)
        self.encoder = encoder
        self.head = head
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, text, mask=None):
        _, (hidden, _), _ = self.encoder(text)
        hidden = self.dropout(hidden)
        output = self.head(hidden)
        return output


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
def test_sentiment_classification():
    """
    Feature: sentiment calssification model.
    Description: test sentiment calssification model for training.
    Expectation: success.
    """
    imdb_train, imdb_test = load('imdb')
    embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"])

    lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ds.transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
    type_cast_op = ds.transforms.TypeCast(mindspore.float32)

    imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=['text'])
    imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=['label'])

    imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
    imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=['label'])

    imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    lr = 0.001

    lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                     dropout=dropout, bidirectional=bidirectional)
    sentiment_encoder = RNNEncoder(embedding, lstm_layer)
    sentiment_head = Head(hidden_size, output_size)
    net = SentimentClassification(sentiment_encoder, sentiment_head, dropout)

    loss = nn.BCELoss(reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    metric = Accuracy()

    trainer = Trainer(network=net, train_dataset=imdb_train, eval_dataset=imdb_valid,
                      metrics=metric, epochs=2, batch_size=64, loss_fn=loss,
                      optimizer=optimizer)
    trainer.run(tgt_columns="label")
