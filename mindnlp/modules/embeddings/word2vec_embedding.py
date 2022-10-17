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
"""Word2vec_embedding"""

import os
import re
from itertools import islice
import numpy as np
from gensim.models import KeyedVectors
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.dataset.text.utils import Vocab
from mindnlp.abc.modules.embedding import TokenEmbedding
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import cache_file, ungz


class Word2vec(TokenEmbedding):
    r"""
    Create vocab and Embedding from a given pre-trained vector file.
    """
    urls = {
        "google-news": "https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/"
                       "word2vec-google-news-300.gz"
    }

    dims = [300]

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
        self.dropout = nn.Dropout(1 - dropout)
        self.word_dropout = word_dropout

    @classmethod
    def from_pretrained(cls, name='google-news', dims=300, root=DEFAULT_ROOT,
                        special_tokens=("<unk>", "<pad>"), special_first=False, use_gensim=True):
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
        cache_dir = os.path.join(root, "embeddings", "Word2vec")

        url = cls.urls[name]
        download_file_name = re.sub(r".+/", "", url)
        word2vec_file_name = f"word2vec-{name}-{dims}.bin"
        path, _ = cache_file(filename=download_file_name, cache_dir=cache_dir, url=url)
        decompress_path = os.path.join(cache_dir, word2vec_file_name)
        if not os.path.exists(decompress_path):
            ungz(path, decompress_path)

        word2vec_file_path = decompress_path
        compress_path = os.path.join(cache_dir, download_file_name)

        if use_gensim:
            model = KeyedVectors.load_word2vec_format(compress_path, binary=True)
            embeddings = list(model.vectors)
            vocab = Vocab.from_list(list(model.key_to_index), list(special_tokens), special_first)
        else:
            embeddings = []
            tokens = []
            with open(word2vec_file_path, encoding='utf-8') as file:
                for line in islice(file, 1, None):
                    word, embedding = line.split(maxsplit=1)
                    tokens.append(word)
                    embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
            vocab = Vocab.from_list(tokens, list(special_tokens), special_first)

        embeddings.append(np.random.rand(dims))
        embeddings.append(np.zeros((dims,), np.float32))
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
