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
Fasttext model
"""

import numpy as np
from mindspore import nn, Tensor
from mindspore.common import dtype as mstype
from mindspore.common.initializer import XavierUniform
from mindspore.dataset.text.utils import Vocab
from mindnlp.engine.trainer import Trainer
from mindnlp.metrics import Accuracy
from mindnlp.modules.embeddings import Glove
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class FasttextModel(nn.Cell):
    """
    FastText model
    """

    def __init__(self, vocab_size, embedding_dims, num_class):
        super(FasttextModel, self).__init__()
        self.vocab_size = vocab_size
        self.embeding_dims = embedding_dims
        self.num_class = num_class

        self.embeding_func = Glove(vocab=Vocab.from_list(['default']),
                                   init_embed=Tensor(np.zeros([self.vocab_size, self.embeding_dims]), mstype.float32))

        self.fc = nn.Dense(self.embeding_dims, out_channels=self.num_class,
                           weight_init=XavierUniform(1)).to_float(mstype.float16)

    def construct(self, text):
        """
        construct network
        """

        src_token_length = len(text)
        text = self.embeding_func(text)

        embeding = text.sum(axis=1)

        embeding = Tensor.div(embeding, src_token_length)

        embeding = embeding.astype(mstype.float32)
        classifier = self.fc(embeding)
        classifier = classifier.astype(mstype.float32)

        return classifier


ag_news_train, ag_news_test = load_dataset('ag_news', shuffle=True)

vocab_size = 1383812
embedding_dims = 16
num_class = 4
lr = 0.001
tokenizer = BasicTokenizer(True)
bucket_boundaries = [64, 128, 467]
max_len = 467
drop = 0.0

embedding, vocab = Glove.from_pretrained('6B', 100)
ag_news_train = process('ag_news', ag_news_train, tokenizer=tokenizer, vocab=vocab, \
                        bucket_boundaries=bucket_boundaries, max_len=max_len, drop_remainder=True)
ag_news_train, ag_news_valid = ag_news_train.split([0.7, 0.3])

# net
net = FasttextModel(vocab_size, embedding_dims, num_class)
loss = nn.NLLLoss(reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

# define metrics
metric = Accuracy()

# define trainer
trainer = Trainer(network=net, train_dataset=ag_news_train, eval_dataset=ag_news_valid, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer)

print("start train")
trainer.run(tgt_columns="label")
# trainer.run()
print("end train")
