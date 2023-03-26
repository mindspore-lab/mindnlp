#!/usr/bin/env python
# coding: utf-8
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
# pylint: disable=C0103
# pylint: disable=C0412

"""bilstm imdb st"""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindnlp.utils import less_min_pynative_first
if less_min_pynative_first:
    from mindspore.ops import value_and_grad
else:
    from mindspore import value_and_grad

class BiLSTM(nn.Cell):
    """bilstm network"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           batch_first=True,
                           dropout=0.5)
        self.fc = nn.Dense(hidden_dim * 2, output_dim)

    def construct(self, inputs):
        embedded = self.embedding(inputs)
        _, (hidden, _) = self.rnn(embedded)
        hidden = ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        output = self.fc(hidden)
        return output


def test_bilstm_imdb():
    """test bilstm imdb, fake input data for simulation."""
    vocab_size = 1024
    embed_size = 256
    pad_idx = 0
    hidden_size = 256
    output_size = 2
    num_layers = 2
    bidirectional = True
    lr = 5e-4

    model = BiLSTM(vocab_size, embed_size, hidden_size, output_size, num_layers, bidirectional, pad_idx)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=lr)

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        optimizer(grads)
        return loss


    data = Tensor(np.random.randint(0, 1024, (32, 64)), ms.int32)
    label = Tensor(np.random.randint(0, 2, (32,)), ms.int32)

    for _ in range(10):
        _ = train_step(data, label)
