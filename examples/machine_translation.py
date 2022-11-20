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
RNN-based machine translation model
"""

from mindspore import nn
from mindnlp.abc import Seq2seqModel
from mindspore.dataset import text
from mindnlp.modules import RNNEncoder, RNNDecoder
from mindnlp.engine.trainer import Trainer
from mindnlp.engine.metrics import Accuracy
from mindnlp.engine.callbacks.timer_callback import TimerCallback
from mindnlp.engine.callbacks.earlystop_callback import EarlyStopCallback
from mindnlp.engine.callbacks.best_model_callback import BestModelCallback
from mindnlp.dataset import load, process
from mindnlp.dataset.transforms import BasicTokenizer

# ms.set_context(device_target="GPU") # set GPU

class MachineTranslation(Seq2seqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, en, de):
        encoder_out = self.encoder(en)
        decoder_out = self.decoder(de, encoder_out=encoder_out)
        output = decoder_out[0]
        return output.swapaxes(1,2)

# load datasets
multi30k_train, multi30k_valid, multi30k_test = load('multi30k')

# process datasets
tokenizer = BasicTokenizer(True)
multi30k_train = multi30k_train.map([tokenizer], 'en')
multi30k_train = multi30k_train.map([tokenizer], 'de')
en_vocab = text.Vocab.from_dataset(multi30k_train, 'en', special_tokens=['<pad>', '<unk>'], special_first= True)
de_vocab = text.Vocab.from_dataset(multi30k_train, 'de', special_tokens=['<pad>', '<unk>'], special_first= True)
vocab = {'en':en_vocab, 'de':de_vocab}

multi30k_train = process('multi30k', multi30k_train, vocab=vocab, batch_size=64, max_len = 32, drop_remainder = False)

multi30k_valid = multi30k_valid.map([tokenizer], 'en')
multi30k_valid = multi30k_valid.map([tokenizer], 'de')
multi30k_valid = process('multi30k', multi30k_valid, vocab=vocab, batch_size=64, max_len = 32, drop_remainder = False)

# define Models & Loss & Optimizer
input_dim = len(en_vocab.vocab())
output_dim = len(de_vocab.vocab())
enc_emb_dim = 256
dec_emb_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
enc_dropout = 0.5
dec_dropout = 0.5

# encoder
en_embedding = nn.Embedding(input_dim, enc_emb_dim)
en_rnn = nn.RNN(enc_emb_dim, hidden_size=enc_hid_dim, num_layers=2, has_bias=True,
                batch_first=True, dropout=enc_dropout, bidirectional=False)
rnn_encoder = RNNEncoder(en_embedding, en_rnn)

# decoder
de_embedding = nn.Embedding(output_dim, dec_emb_dim)
input_feed_size = 0 if enc_hid_dim == 0 else dec_hid_dim
rnns = [
    nn.RNNCell(
        input_size=dec_emb_dim + input_feed_size
        if layer == 0
            else dec_hid_dim,
        hidden_size=dec_hid_dim
        )
        for layer in range(2)
]
rnn_decoder = RNNDecoder(de_embedding, rnns, dropout_in=enc_dropout, dropout_out = dec_dropout,attention=True, encoder_output_units=enc_hid_dim)

net = MachineTranslation(rnn_encoder, rnn_decoder)
net.update_parameters_name('net.')

optimizer = nn.Adam(net.trainable_params(), learning_rate=10e-5)
loss_fn = nn.CrossEntropyLoss()

# define callbacks
timer_callback_epochs = TimerCallback(print_steps=-1)
earlystop_callback = EarlyStopCallback(patience=2)
bestmodel_callback = BestModelCallback()
callbacks = [timer_callback_epochs, earlystop_callback, bestmodel_callback]

# define metrics
metric = Accuracy()

# define trainer
trainer = Trainer(network=net, train_dataset=multi30k_train, eval_dataset=multi30k_valid, metrics=metric,
                  epochs=10, loss_fn=loss_fn, optimizer=optimizer)
trainer.run(tgt_columns="de", jit=False)
print("end train")
