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
RNN model
"""

import numpy as np

from mindspore import nn
import mindspore.numpy as mnp
import mindspore.dataset as ds

from mindnlp.modules import RNNEncoder, RNNDecoder
from mindnlp.models import RNN
from mindnlp.engine.trainer import Trainer
from mindnlp.common.metrics import Accuracy
from mindnlp.engine.callbacks.timer_callback import TimerCallback
from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
from mindnlp.dataset import load, process

np.random.seed(1)


class MyDataset:
    """
    Dataset
    """
    def __init__(self):
        self.src_tokens = np.random.randn(100, 16).astype(np.int32)
        self.tgt_tokens = np.random.randn(100, 16).astype(np.int32)
        self.src_length = np.random.randn(100).astype(np.int32)
        self.mask = np.ones([100, 16]).astype(np.int32)

    def __getitem__(self, index):
        return tuple(self.src_tokens[index], self.tgt_tokens[index],
                     self.src_length[index], self.mask[index])

    def __len__(self):
        return len(self.src_tokens)


class MyModel(nn.Cell):
    """
    Model
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, has_bias=True,
                 dropout=0, bidirectional=False, attention=True, encoder_output_units=512):
        super().__init__()
        self.rnn_encoder = RNNEncoder(vocab_size, embedding_size, hidden_size,
                                      num_layers=num_layers, has_bias=has_bias,
                                      dropout=dropout, bidirectional=bidirectional)
        self.rnn_decoder = RNNDecoder(vocab_size, embedding_size, hidden_size,
                                      num_layers=num_layers, has_bias=has_bias, dropout=dropout,
                                      attention=attention, encoder_output_units=encoder_output_units)
        self.rnn = RNN(self.rnn_encoder, self.rnn_decoder)

    def construct(self, src_tokens, tgt_tokens, src_length, mask):
        output, _ = self.rnn(src_tokens, tgt_tokens, src_length, mask)
        output = mnp.argmax(output, axis=2)
        return output


# define dataset


# define Models & Loss & Optimizer
rnn_encoder = RNNEncoder(1000, 32, 16, num_layers=2, has_bias=True,
                         dropout=0.1, bidirectional=False)
rnn_decoder = RNNDecoder(1000, 32, 16, num_layers=2, has_bias=True,
                         dropout=0.1, attention=True, encoder_output_units=16)
net = MyModel(1000, 32, 16, num_layers=2, has_bias=True, dropout=0.1,
              bidirectional=False, attention=True, encoder_output_units=16)
loss_fn = nn.MSELoss()
optimizer = nn.Adam(net.trainable_params(), learning_rate=10e-5)

# define forward function
def forward_fn(*data):
    """forward function"""
    logits = net(data[0], data[1], data[2], data[3])
    loss = loss_fn(logits, data[1])
    return loss, logits

# define callbacks
timer_callback_epochs = TimerCallback(print_steps=-1)
earlystop_callback = EarlyStopCallback(patience=2)
callbacks = [timer_callback_epochs, earlystop_callback]

# define metrics
metric = Accuracy()

# define trainer
trainer = Trainer(net, train_dataset=train_dataset, loss_fn=loss_fn,
                  epochs=2, batch_size=4, optimizer=optimizer, metrics=metric, eval_dataset=eval_dataset)
trainer.run()
print("end train")
