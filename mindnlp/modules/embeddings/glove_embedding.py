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
"""glove_embedding"""

import os
import re
import zipfile
import tarfile
import mindspore
from mindspore import nn
import mindspore.numpy as mnp
from mindspore import ops
from mindspore import Tensor
from mindspore.dataset.text.utils import Vocab
from mindnlp.utils import download
from mindnlp.abc.modules.embedding import TokenEmbedding


class Glove(TokenEmbedding):
    r"""
    Create vocab and Embedding from a given pre-trained vector file.
    """
    def __init__(self, vocab: Vocab, init_embed, requires_grad: bool = True, dropout=0.5, word_dropout=0):
        r"""
        Initize Vocab and Embedding by a given pre-trained word embedding.

        Args:
            init_embed : Passing into Tensor, Embedding, Numpy.ndarray, etc.,
                        use this value to initialize Embedding directly.
            requires_grad : Whether this parameter needs to be gradient to update.
            dropout : Dropout of the output of Embedding.
            word_dropout : How much is the probability of replacing a word to UNK.
        """
        super().__init__(vocab, init_embed)

        self.vocab_list = vocab
        self.vocab_size = init_embed.shape[0]
        self.embed = init_embed
        self._embed_size = init_embed.shape[1]
        self.requires_grad = requires_grad
        self.dropout = nn.Dropout(dropout)
        self.word_dropout = word_dropout

    @classmethod
    def from_pretrained(cls, url: str):
        r"""
        Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            url (str) : url to download file.
        Returns:
            - ** vocab ** - Vocabulary extracted from the file.
            - ** embeddings ** - Word vector extracted from the file.

        """
        file_name = re.sub(r".+/", "", url)
        download.cache_file(filename=file_name, cache_dir=None, url=url)
        cache_dir = download.get_cache_path()

        suffix = ''
        if file_name.endswith('.tar.gz'):
            suffix = '.tar.gz'
        elif file_name.endswith('.zip'):
            suffix = '.zip'

        name_rar = file_name
        name_dir = name_rar.replace(suffix, '')
        glove_dir_path = os.path.join(cache_dir, name_dir)
        glove_compress_path = os.path.join(cache_dir, file_name)

        if not os.path.isdir(glove_dir_path):
            if suffix == '.tar.gz':
                glove_tar = tarfile.open(glove_compress_path, 'r')
                glove_tar.extractall(cache_dir)
                glove_tar.close()
            elif file_name == '.zip':
                glove_zip = zipfile.ZipFile(glove_compress_path)
                glove_zip.extractall(cache_dir)
                glove_zip.close()

        file_suffix = ''
        if os.path.isdir(glove_dir_path):
            while os.path.isdir(glove_dir_path):
                next_dir_path = os.path.join(glove_dir_path, name_dir)
                if not os.path.isdir(next_dir_path):
                    for file in os.listdir(glove_dir_path):
                        if file.startswith(name_dir):
                            file_suffix = os.path.splitext(file)[-1]
                    break
                glove_dir_path = next_dir_path
            name_txt = name_dir + '.50d' + file_suffix
            glove_file_path = os.path.join(glove_dir_path, name_txt)
        else:
            for file in os.listdir(cache_dir):
                if file.startswith(name_dir):
                    file_suffix = os.path.splitext(file)[-1]
                    break
            name_txt = name_dir + '.50d' + file_suffix
            glove_file_path = os.path.join(cache_dir, name_txt)

        embeddings = []
        tokens = []
        with open(glove_file_path, encoding='utf-8') as file:
            for glove in file:
                word, embedding = glove.split(maxsplit=1)
                tokens.append(word)
                arr = embedding.split(' ')
                float_arr = list(map(float, arr))
                float_tensor = Tensor(float_arr)
                float32_arr = mnp.asfarray(float_tensor)

                embeddings.append(float32_arr)

        embeddings.append(mnp.rand(50))
        embeddings.append(mnp.zeros((50,), mindspore.float32))

        vocab = Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
        embeddings = mnp.array(embeddings).astype(mindspore.float32)
        return cls(vocab, Tensor(embeddings), True, 0.5, 0)

    def construct(self, ids):
        r"""
        Use ids to query embedding
        Args:
            ids : Ids to query.

        Returns:
            - ** compute result ** - Tensor, returns the Embedding query results.

        """
        tensor_ids = Tensor(ids)
        out_shape = tensor_ids.shape + (self._embed_size,)
        flat_ids = tensor_ids.reshape((-1,))
        output_for_reshape = ops.gather(self.embed, flat_ids, 0)
        output = ops.reshape(output_for_reshape, out_shape)
        return output
