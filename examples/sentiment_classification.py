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

from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Uniform, HeUniform

from mindnlp.abc import Seq2vecModel
from mindnlp.engine import Trainer, Accuracy
from mindnlp.dataset import load, process
from mindnlp.modules import Glove, RNNEncoder
from mindnlp.dataset.transforms import BasicTokenizer

# Hyper-parameters
hidden_size = 256
output_size = 1
num_layers = 2
bidirectional = True
dropout = 0.5
lr = 0.001

class SentimentClassification(Seq2vecModel):
    def construct(self, text):
        _, (hidden, _), _ = self.encoder(text)
        context = ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        output = self.head(context)
        return output

# load embedding and vocab
embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=dropout)

# load datasets
imdb_train, imdb_test = load('imdb', shuffle=True)
print(imdb_train.get_col_names())

tokenizer = BasicTokenizer(True)
imdb_train = process('imdb', imdb_train, tokenizer=tokenizer, vocab=vocab, \
                     bucket_boundaries=[400, 500], max_len=600, drop_remainder=True)
imdb_test = process('imdb', imdb_test, tokenizer=tokenizer, vocab=vocab, \
                     bucket_boundaries=[400, 500], max_len=600, drop_remainder=False)

# build encoder
lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                     dropout=dropout, bidirectional=bidirectional)
encoder = RNNEncoder(embedding, lstm_layer)

# build head
head = nn.SequentialCell([
    nn.Dropout(1 - dropout),
    nn.Sigmoid(),
    nn.Dense(hidden_size * 2, output_size,
             weight_init=HeUniform(math.sqrt(5)),
             bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))

])

# build network
network = SentimentClassification(encoder, head)
loss = nn.BCELoss(reduction='mean')
optimizer = nn.Adam(network.trainable_params(), learning_rate=lr)

# define metrics
metric = Accuracy()

# define trainer
trainer = Trainer(network=network, train_dataset=imdb_train, eval_dataset=imdb_test, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer)
trainer.run(tgt_columns="label")
print("end train")