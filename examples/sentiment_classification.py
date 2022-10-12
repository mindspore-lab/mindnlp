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
# pylint: disable=abstract-method

import os
import shutil
import tempfile
import re
import string
import tarfile
import zipfile
import math
from typing import IO
from pathlib import Path
import requests
import six
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.dataset as ds
from mindspore.common.initializer import Uniform, HeUniform

from mindnlp.common.metrics import Accuracy
from mindnlp.engine.trainer import Trainer
from mindnlp.abc import Seq2vecModel
from mindnlp.modules import LSTMEncoder

cache_dir = Path.home() / '.mindspore_examples'


def http_get(url: str, temp_file: IO):
    """
    Use requests to download dataset
    """
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit='B', total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def download(file_name: str, url: str):
    """
    Download dataset
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, file_name)
    cache_exist = os.path.exists(cache_path)
    if not cache_exist:
        with tempfile.NamedTemporaryFile() as temp_file:
            http_get(url, temp_file)
            temp_file.flush()
            temp_file.seek(0)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
    return cache_path


class IMDBData():
    """
    IMDB dataset
    """
    label_map = {
        "pos": 1,
        "neg": 0
    }
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []

        self._load("pos")
        self._load("neg")

    def _load(self, label):
        pattern = re.compile(r"aclImdb/{}/{}/.*\.txt$".format(self.mode, label))
        with tarfile.open(self.path) as tarf:
            tf = tarf.next()
            while tf is not None:
                if bool(pattern.match(tf.name)):
                    self.docs.append(str(tarf.extractfile(tf).read().rstrip(six.b("\n\r"))
                                         .translate(None, six.b(string.punctuation)).lower()).split())
                    self.labels.append([self.label_map[label]])
                tf = tarf.next()

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx]

    def __len__(self):
        return len(self.docs)


def load_imdb(data_path):
    """
    load IMDB dataset
    """
    train_data = ds.GeneratorDataset(IMDBData(data_path, "train"), column_names=["src_tokens", "label"], shuffle=True)
    test_data = ds.GeneratorDataset(IMDBData(data_path, "test"), column_names=["src_tokens", "label"], shuffle=False)
    return train_data, test_data


def load_glove(embed_path):
    """
    load glove
    """
    glove_100d_path = os.path.join(cache_dir, 'glove.6B.100d.txt')
    if not os.path.exists(glove_100d_path):
        glove_zip = zipfile.ZipFile(embed_path)
        glove_zip.extractall(cache_dir)

    glove_embeddings = []
    tokens = []
    with open(glove_100d_path, encoding='utf-8') as gf:
        for glove in gf:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            glove_embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    glove_embeddings.append(np.random.rand(100))
    glove_embeddings.append(np.zeros((100,), np.float32))

    glove_vocab = ds.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    glove_embeddings = np.array(glove_embeddings).astype(np.float32)
    return glove_vocab, glove_embeddings


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

    def construct(self, src_tokens, mask=None):
        _, (hidden, _), _ = self.encoder(src_tokens)
        hidden = self.dropout(hidden)
        output = self.head(hidden)
        return output


# load datasets
glove_path = download('glove.6B.zip', 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/glove.6B.zip')
vocab, embeddings = load_glove(glove_path)
vocab_size, embedding_dim = embeddings.shape

lookup_op = ds.text.Lookup(vocab, unknown_token='<unk>')
pad_op = ds.transforms.PadEnd([500], pad_value=vocab.tokens_to_ids('<pad>'))
type_cast_op = ds.transforms.TypeCast(ms.float32)

imdb_path = download('aclImdb_v1.tar.gz', 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz')
imdb_train, imdb_test = load_imdb(imdb_path)
imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=['src_tokens'])
imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=['label'])

imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=['src_tokens'])
imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=['label'])

imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

# define Models & Loss & Optimizer
hidden_size = 256
output_size = 1
num_layers = 2
bidirectional = True
drop = 0.5
lr = 0.001

sentiment_encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_size, num_layers=num_layers,
                                dropout=drop, bidirectional=bidirectional)
sentiment_head = Head(hidden_size, output_size)
net = SentimentClassification(sentiment_encoder, sentiment_head, drop)

loss = nn.BCELoss(reduction='mean')
optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

# define metrics
metric = Accuracy()

# define trainer

trainer = Trainer(network=net, train_dataset=imdb_train, eval_dataset=imdb_valid, metrics=metric,
                  epochs=2, batch_size=64, loss_fn=loss, optimizer=optimizer)
trainer.run(tgt_columns="label")
print("end train")
