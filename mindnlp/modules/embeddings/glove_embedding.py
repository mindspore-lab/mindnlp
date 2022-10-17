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
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.dataset.text.utils import Vocab
from mindnlp.utils import cache_file, unzip
from mindnlp.abc.modules.embedding import TokenEmbedding
from mindnlp.configs import DEFAULT_ROOT


class Glove(TokenEmbedding):
    r"""
    Create vocab and Embedding from a given pre-trained vector file.
    """
    urls = {
        "42B": "http://nlp.stanford.edu/data/glove.42B.300d.zip",
        "840B": "http://nlp.stanford.edu/data/glove.840B.300d.zip",
        "twitter.27B": "http://nlp.stanford.edu/data/glove.twitter.27B.zip",
        "6B": "http://nlp.stanford.edu/data/glove.6B.zip",
    }

    dims = [50, 100, 200, 300]

    def __init__(self, vocab: Vocab, init_embed, requires_grad: bool = True, dropout=0.0, word_dropout=0):
        r"""
        Initize Vocab and Embedding by a given pre-trained word embedding.

        Args:
            vocab :
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
        self.dropout = nn.Dropout(1 - dropout)
        self.word_dropout = word_dropout

    @classmethod
    def from_pretrained(cls, name='6B', dims=300, root=DEFAULT_ROOT,
                        special_tokens=("<unk>", "<pad>"), special_first=False, use_gensim=False):
        r"""
        Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            name (str): The name of the pretrained vector.
            dims (int): The dimension of the pretrained vector.
            root (str): Default storage directory.
            special_tokens (tuple<str,str>): List of special participles.<unk>:Mark the words that don't exist;
            <pad>:Align all the sentences.
            special_first (bool): Indicates whether special participles from special_tokens will be added to
            the top of the dictionary. If True, add special_tokens to the beginning of the dictionary,
            otherwise add them to the end.
            use_gensim (bool): Whether to use gensim library for pretrained word vector loading.
        Returns:
            - ** cls ** - Returns a embedding instance generated through a pretrained word vector.
            - ** vocab ** - Vocabulary extracted from the file.

        """
        if name not in cls.urls:
            raise ValueError(f"The argument 'name' must in {cls.urls.keys()}, but got {name}.")
        if dims not in cls.dims:
            raise ValueError(f"The argument 'dims' must in {cls.dims}, but got {dims}.")
        cache_dir = os.path.join(root, "embeddings", "Glove")

        url = cls.urls[name]
        download_file_name = re.sub(r".+/", "", url)
        glove_file_name = f"glove.{name}.{dims}d.txt"
        path, _ = cache_file(filename=download_file_name, cache_dir=cache_dir, url=url)
        decompress_path = os.path.join(cache_dir, glove_file_name)
        if not os.path.exists(decompress_path):
            unzip(path, cache_dir)

        glove_file_path = os.path.join(cache_dir, glove_file_name)

        embeddings = []
        tokens = []
        with open(glove_file_path, encoding='utf-8') as file:
            for line in file:
                word, embedding = line.split(maxsplit=1)
                tokens.append(word)
                embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

        embeddings.append(np.random.rand(dims))
        embeddings.append(np.zeros((dims,), np.float32))

        vocab = Vocab.from_list(tokens, list(special_tokens), special_first)
        embeddings = np.array(embeddings).astype(np.float32)
        return cls(vocab, Tensor(embeddings), True, 0.5, 0), vocab

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
