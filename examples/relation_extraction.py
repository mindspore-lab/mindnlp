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
CNN-based relation extraciton model
"""

import math
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn

from mindnlp.engine.trainer import Trainer
from mindnlp.engine.metrics import Accuracy
from mindnlp.dataset.register import load, process
from mindspore.common.initializer import Initializer, _calculate_fan_in_and_fan_out, _assignment, initializer, XavierUniform


class XavierNormal(Initializer):
    def __init__(self, gain=1):
        super().__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr.shape)

        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out))
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)


class CNN(nn.Cell):
    def __init__(self, word_vec, class_num):
        super(CNN, self).__init__()
        self.word_vec = word_vec
        self.class_num = class_num

        # hyper parameters and others
        self.max_len = 100
        self.word_dim = 50
        self.pos_dim = 5
        self.pos_dis = 50

        self.dropout_value = 0.5
        self.filter_num = 200
        self.window = 3
        self.hidden_size = 100

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding(
            vocab_size=self.word_vec.shape[0],
            embedding_size=self.word_vec.shape[1],
        )

        self.pos1_embedding = nn.Embedding(
            vocab_size=2 * self.pos_dis + 3,
            embedding_size=self.pos_dim
        )

        self.pos2_embedding = nn.Embedding(
            vocab_size=2 * self.pos_dis + 3,
            embedding_size=self.pos_dim
        )

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            has_bias=True,
            padding=(1, 1, 0, 0), # 上下左右的顺序，源代码(1,0)为(行数，列数)
            pad_mode='pad'
        )

        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(1-self.dropout_value)
        self.linear = nn.Dense(
            in_channels=self.filter_num,
            out_channels=self.hidden_size,
            has_bias=True
        )
        self.dense = nn.Dense(
            in_channels=self.hidden_size,
            out_channels=self.class_num,
            has_bias=True
        )


        self.pos1_embedding.embedding_table = initializer(XavierNormal(), self.pos1_embedding.embedding_table.shape)
        self.pos2_embedding.embedding_table = initializer(XavierNormal(), self.pos2_embedding.embedding_table.shape)
        self.conv.weight = initializer(XavierNormal(), self.conv.weight.shape)
        self.conv.bias = initializer(0, self.conv.bias.shape, mindspore.float32)
        self.linear.weight = initializer(XavierNormal(), self.linear.weight.shape)
        self.linear.bias = initializer(0, self.linear.bias.shape, mindspore.float32)
        self.dense.weight = initializer(XavierNormal(), self.dense.weight.shape)
        self.dense.bias = initializer(0, self.dense.bias.shape, mindspore.float32)

        self.concat = ops.Concat(axis=-1)


    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B*L*word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B*L*pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B*L*pos_dim
        emb = self.concat([word_emb, pos1_emb, pos2_emb])
        return emb

    def conv_layer(self, emb, mask):
        emb = ops.expand_dims(emb, axis=1)
        conv = self.conv(emb)

        conv = conv.view(-1, self.filter_num, self.max_len)
        mask = ops.expand_dims(mask, axis=1)
        mask = ops.broadcast_to(mask, shape=(-1, self.filter_num, -1))
        conv = ops.masked_fill(conv, ops.equal(mask, 0), float('-inf'))
        conv = ops.expand_dims(conv, axis=-1)
        return conv

    def single_maxpool_layer(self, conv):
        pool = self.maxpool(conv)  # B*C*1*1
        pool = pool.view(-1, self.filter_num)  # B*C
        return pool

    def construct(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        emb = self.encoder_layer(token, pos1, pos2)
        emb = self.dropout(emb)
        conv = self.conv_layer(emb, mask)
        pool = self.single_maxpool_layer(conv)
        sentence_feature = self.linear(pool)
        sentence_feature = self.tanh(sentence_feature)
        sentence_feature = self.dropout(sentence_feature)
        logits = self.dense(sentence_feature)
        return logits




batch_size = 128
lr = 0.001
weight_decay = 1e-5
epochs = 5

semeval, word_vec, class_num = load('semeval')
semeval_train, semeval_test = semeval

semeval_train = process('semeval', semeval_train, batch_size=batch_size, drop_remainder = True)
semeval_test = process('semeval', semeval_test, batch_size=batch_size, drop_remainder = True)


# define Models & Loss & Optimizer
net = CNN(word_vec=word_vec, class_num=class_num)
loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)

# define metrics
metric = Accuracy()

# define trainer
trainer = Trainer(network=net, train_dataset=semeval_train, eval_dataset=semeval_test, metrics=metric,
                  epochs=epochs, loss_fn=loss, optimizer=optimizer)
print("start train")
trainer.run(tgt_columns="label", jit=False)
print("end train")