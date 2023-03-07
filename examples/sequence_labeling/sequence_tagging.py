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
BiLSTM-CRF sequence tagging model
"""

import math
from tqdm import tqdm
from mindspore.common.initializer import Uniform, HeUniform
from mindspore import nn, ops
from mindspore.dataset import text
from mindnlp.abc import Seq2vecModel
from mindnlp.dataset import CoNLL2000Chunking, CoNLL2000Chunking_Process
from mindnlp.modules import CRF, RNNEncoder

class Head(nn.Cell):
    """ Head for BiLSTM-CRF model """
    def __init__(self, hidden_dim, num_tags):
        super().__init__()
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.hidden2tag = nn.Dense(hidden_dim, num_tags,
                                   weight_init=weight_init, bias_init=bias_init)

    def construct(self, context):
        return self.hidden2tag(context)

class BiLSTM_CRF(Seq2vecModel):
    """ BiLSTM-CRF model """
    def __init__(self, encoder, head, num_tags):
        super().__init__(encoder, head)
        self.encoder = encoder
        self.head = head
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, text, seq_length, label=None):
        output,_,_ = self.encoder(text)
        feats = self.head(output)
        res = self.crf(feats, label, seq_length)
        return res

# load datasets
dataset_train,dataset_test = CoNLL2000Chunking()

# build vocab
vocab = text.Vocab.from_dataset(dataset_train,columns=["words"],freq_range=None,top_k=None,
                                   special_tokens=["<pad>","<unk>"],special_first=True)

# process datasets
dataset_train = CoNLL2000Chunking_Process(dataset=dataset_train, vocab=vocab,
                                          batch_size=32, max_len=80)

# define model
embedding_dim = 16
hidden_dim = 32
embedding = nn.Embedding(vocab_size=len(vocab.vocab()), embedding_size=embedding_dim,
                         padding_idx=vocab.tokens_to_ids("<pad>"))
lstm_layer = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
encoder = RNNEncoder(embedding, lstm_layer)
head = Head(hidden_dim, 23)
net = BiLSTM_CRF(encoder, head, 23)

# define optimizer
optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01, weight_decay=1e-4)
grad_fn = ops.value_and_grad(net, None, optimizer.parameters)

# define train step
def train_step(data, seq_length, label):
    """ train step """
    loss, grads = grad_fn(data, seq_length, label)
    loss = ops.depend(loss, optimizer(grads))
    return loss

# get epoch size
size = dataset_train.get_dataset_size()

# train
steps = size
with tqdm(total=steps) as t:
    for batch, (data, seq_length, label) in enumerate(dataset_train.create_tuple_iterator()):
        loss = train_step(data, seq_length ,label)
        t.set_postfix(loss=loss)
        t.update(1)
