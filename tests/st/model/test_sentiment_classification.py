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
Sentiment Calssification model
"""
# pylint: disable=abstract-method
# pylint: disable=arguments-differ


import math
import pytest
import numpy as np

from mindspore import nn
from mindspore import ops
import mindspore.numpy as mnp
import mindspore.dataset as ds
from mindspore.common.initializer import Uniform, HeUniform

from mindnlp.abc import Seq2vecModel
from mindnlp.modules import LSTMEncoder
from mindnlp.common.metrics import Accuracy
from mindnlp.engine.trainer import Trainer


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
        self.sigmoid = ops.Sigmoid()

    def construct(self, context):
        context = mnp.concatenate((context[-2, :, :], context[-1, :, :]), axis=1)
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

    def construct(self, src_tokens):
        _, (hidden, _), _ = self.encoder(src_tokens)
        hidden = self.dropout(hidden)
        output = self.head(hidden)
        return output


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
def test_pynative():
    """
    Feature: sentiment calssification model.
    Description: test sentiment calssification model for training in pynative mode.
    Expectation: success.
    """
    vocab_size = 400000
    embedding_dim = 100

    dataset = ds.GeneratorDataset(Dataset(), column_names=["src_tokens", "label"], shuffle=True)
    train_dataset, valid_dataset = dataset.split([0.7, 0.3])

    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    lr = 0.001

    encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional)
    head = Head(hidden_size, output_size)
    net = SentimentClassification(encoder, head, dropout)

    loss = nn.BCELoss(reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    metric = Accuracy()

    trainer = Trainer(network=net, train_dataset=train_dataset, eval_dataset=valid_dataset,
                      metrics=metric, epochs=2, batch_size=64, loss_fn=loss,
                      optimizer=optimizer)
    trainer.run(mode="pynative", tgt_columns="label")

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
def test_graph():
    """
    Feature: sentiment calssification model.
    Description: test sentiment calssification model for training in graph mode.
    Expectation: success.
    """
    vocab_size = 400000
    embedding_dim = 100

    dataset = ds.GeneratorDataset(Dataset(), column_names=["src_tokens", "label"], shuffle=True)
    train_dataset, valid_dataset = dataset.split([0.7, 0.3])

    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    lr = 0.001

    encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional)
    head = Head(hidden_size, output_size)
    net = SentimentClassification(encoder, head, dropout)

    loss = nn.BCELoss(reduction='mean')
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    metric = Accuracy()

    trainer = Trainer(network=net, train_dataset=train_dataset, eval_dataset=valid_dataset,
                      metrics=metric, epochs=2, batch_size=64, loss_fn=loss,
                      optimizer=optimizer)
    trainer.run(mode="graph", tgt_columns="label")
