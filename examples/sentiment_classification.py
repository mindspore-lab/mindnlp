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

import math

import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.dataset as ds
from mindspore.common.initializer import Uniform, HeUniform

from mindnlp.modules import RNNEncoder
from mindnlp.engine.metrics import Accuracy
from mindnlp.engine.trainer import Trainer
from mindnlp.abc import Seq2vecModel
from mindnlp.dataset import load
from mindnlp.modules import Glove

class Head(nn.Cell):
    """
    Head for Sentiment Classification model
    """
    def __init__(self, hidden_dim, output_dim, dropout):
        super().__init__()
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, context):
        context = ops.concat((context[-2, :, :], context[-1, :, :]), axis=1)
        context = self.dropout(context)
        return self.sigmoid(self.fc(context))


class SentimentClassification(Seq2vecModel):
    """
    Sentiment Classification model
    """
    def __init__(self, encoder, head):
        super().__init__(encoder, head)
        self.encoder = encoder
        self.head = head

    def construct(self, text):
        _, (hidden, _), _ = self.encoder(text)
        output = self.head(hidden)
        return output

# define Models & Loss & Optimizer
hidden_size = 256
output_size = 1
num_layers = 2
bidirectional = True
drop = 0.5
lr = 0.001

# load datasets
imdb_train, imdb_test = load('imdb', shuffle=True)


embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)

lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
pad_op = ds.transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
type_cast_op = ds.transforms.TypeCast(ms.float32)

imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=['text'])
imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=['label'])

imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=['label'])

imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])
imdb_train = imdb_train.batch(64, drop_remainder=True)
imdb_valid = imdb_valid.batch(64, drop_remainder=True)

lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                     dropout=drop, bidirectional=bidirectional)
sentiment_encoder = RNNEncoder(embedding, lstm_layer)
sentiment_head = Head(hidden_size, output_size, drop)
net = SentimentClassification(sentiment_encoder, sentiment_head)

loss = nn.BCELoss(reduction='mean')

# define metrics
metric = Accuracy()

optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

# define trainer

trainer = Trainer(network=net, train_dataset=imdb_train, eval_dataset=imdb_valid, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer)
trainer.run(tgt_columns="label", jit=True)
print("end train")