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
"""Fasttext_embedding"""

import os
import re
import json
import logging
from itertools import islice
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.dataset.text.utils import Vocab
from mindnlp.utils import cache_file, unzip
from mindnlp.abc.modules.embedding import TokenEmbedding
from mindnlp.configs import DEFAULT_ROOT

JSON_FILENAME = 'fasttext_hyper.json'
EMBED_FILENAME = 'fasttext.txt'
logging.getLogger().setLevel(logging.INFO)


class Fasttext(TokenEmbedding):
    r"""
    Embedding layer.

    Args:
        vocab (Vocab): Passins into Vocab for initialization.
        init_embed (Tensor): Passing into Tensor,use these values to initialize Embedding directly.
        requires_grad (bool): Whether this parameter needs to be gradient to update. Default: True.
        dropout (float): Dropout of the output of Embedding. Default: 0.5.

    Examples:
        >>> vocab = Vocab.from_list(['default','one','two','three'])
        >>> init_embed = Tensor(np.zeros((4, 4)).astype(np.float32))
        >>> fasttext_embed = Fasttext(vocab, init_embed)
        >>> ids = Tensor([1, 2, 3])
        >>> output = fasttext_embed(ids)

    """
    urls = {
        "1M": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        "1M-subword": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",
    }

    dims = [300]

    def __init__(self, vocab: Vocab, init_embed, requires_grad: bool = True, dropout=0.0):
        super().__init__(vocab, init_embed)

        self._word_vocab = vocab
        self.vocab_size = init_embed.shape[0]
        self.embed = init_embed
        self._embed_dim = init_embed.shape[1]
        self._embed_size = init_embed.shape
        self.requires_grad = requires_grad
        self.dropout_layer = nn.Dropout(1 - dropout)
        self.dropout_p = dropout

    @classmethod
    def from_pretrained(cls, name='1M', dims=300, root=DEFAULT_ROOT,
                        special_tokens=("<pad>", "<unk>"), special_first=True, **kwargs):
        r"""
        Creates Embedding instance from given pre-trained word vector.

        Args:
            name (str): The name of the pretrained vector. Default: "1M".
            dims (int): The dimension of the pretrained vector. Default: 300.
            root (str): Default storage directory. Default: DEFAULT_ROOT.
            special_tokens (tuple<str,str>): List of special participles. Default: ("<pad>", "<unk>").
            special_first (bool): Indicates whether special participles from special_tokens will be added to
                the top of the dictionary. If True, add special_tokens to the beginning of the dictionary,
                otherwise add them to the end. Default: True.
            kwargs (dict):
                - requires_grad (bool): Whether this parameter needs to be gradient to update.
                - dropout (float): Dropout of the output of Embedding.

        Returns:
            - Fasttext, Returns an embedding instance generated through a pretrained word vector.
            - Vocab, Vocabulary extracted from the file.

        """
        if name not in cls.urls:
            raise ValueError(f"The argument 'name' must in {cls.urls.keys()}, but got {name}.")
        if dims not in cls.dims:
            raise ValueError(f"The argument 'dims' must in {cls.dims}, but got {dims}.")
        cache_dir = os.path.join(root, "embeddings", "Fasttext")

        url = cls.urls[name]
        download_file_name = re.sub(r".+/", "", url)
        fasttext_file_name = f"wiki-news-{dims}d-{name}.vec"
        path, _ = cache_file(filename=download_file_name, cache_dir=cache_dir, url=url)
        decompress_path = os.path.join(cache_dir, fasttext_file_name)
        if not os.path.exists(decompress_path):
            unzip(path, cache_dir)

        fasttext_file_path = os.path.join(cache_dir, fasttext_file_name)

        embeddings = []
        tokens = []
        with open(fasttext_file_path, encoding='utf-8') as file:
            for line in islice(file, 1, None):
                word, embedding = line.split(maxsplit=1)
                tokens.append(word)
                embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

        if special_first:
            embeddings.insert(0, np.random.rand(dims))
            embeddings.insert(1, np.zeros((dims,), np.float32))
        else:
            embeddings.append(np.random.rand(dims))
            embeddings.append(np.zeros((dims,), np.float32))

        vocab = Vocab.from_list(tokens, list(special_tokens), special_first)
        embeddings = np.array(embeddings).astype(np.float32)

        requires_grad = kwargs.get('requires_grad', True)
        dropout = kwargs.get('dropout', 0.0)

        return cls(vocab, Tensor(embeddings), requires_grad, dropout), vocab

    def construct(self, ids):
        r"""

        Args:
            ids (Tensor): Ids to query.

        Returns:
            - Tensor, returns the Embedding query results.

        """
        out_shape = ids.shape + (self._embed_dim,)
        flat_ids = ids.reshape((-1,))
        output_for_reshape = ops.gather(self.embed, flat_ids, 0)
        output = ops.reshape(output_for_reshape, out_shape)
        return self.dropout(output)

    def save(self, foldername, root=DEFAULT_ROOT):
        r"""
        Save the embedding to the specified location.

        Args:
            foldername (str): Name of the folder to store.
            root (Path): Path of the embedding folder. Default: DEFAULT_ROOT.

        Returns:
            None

        """
        folder = os.path.join(root, 'embeddings', 'Fasttext', 'save', foldername)
        os.makedirs(folder, exist_ok=True)

        vocab = self.get_word_vocab()
        embed = self.embed
        embed_list = embed
        vocab_list = list(vocab.keys())
        nums = self.vocab_size
        dims = self._embed_dim

        kwargs = {}
        kwargs['dropout'] = kwargs.get('dropout', self.dropout_p)
        kwargs['requires_grad'] = kwargs.get('requires_grad', self.requires_grad)

        with open(os.path.join(folder, JSON_FILENAME), 'w', encoding='utf-8') as file:
            json.dump(kwargs, file, indent=2)

        with open(os.path.join(folder, EMBED_FILENAME), 'w', encoding='utf-8') as file:
            file.write(f'{" " * 30}\n')
            for i in range(0, nums):
                vocab_write = vocab_list[i]
                embed_write = list(embed_list[i])
                vec_write = ' '.join(map(str, embed_write))
                file.write(f'{vocab_write} {vec_write}\n')

            file.seek(0)
            file.write(f'{nums} {dims}')

        logging.info('Embedding has been saved to %s', folder)

    @classmethod
    def load(cls, foldername=None, root=DEFAULT_ROOT, load_npy=False, vocab=None, npy_path=None):
        r"""
        Load embedding from the specified location.

        Args:
            foldername (str): Name of the folder to load. Default: None.
            root (Path): Path of the embedding folder. Default: DEFAULT_ROOT.
            load_npy (Bool): Whether to initialize the embedding as a npy file. Vocab and npy_path are valid
                when load_npy is True. Default: False.
            vocab (Vocab): If initialized with a npy file, pass in vocab. Default: None.
            npy_path (Path): Location of the npy file. Default: None.

        Returns:
            None

        """

        if load_npy:
            load_embed = np.load(npy_path)
            load_vocab = vocab

            return cls(load_vocab, Tensor(load_embed))

        folder = os.path.join(root, 'embeddings', 'Fasttext', 'save', foldername)
        for name in [JSON_FILENAME, EMBED_FILENAME]:
            assert os.path.exists(os.path.join(folder, name)), f"{name} not found in {folder}."

        with open(os.path.join(folder, JSON_FILENAME), 'r', encoding='utf-8') as file:
            hyper = json.load(file)

        embeddings = []
        tokens = []
        with open(os.path.join(folder, EMBED_FILENAME), encoding='utf-8') as file:
            for line in islice(file, 1, None):
                word, embedding = line.split(maxsplit=1)
                tokens.append(word)
                embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

        vocab = Vocab.from_list(tokens)
        embeddings = np.array(embeddings).astype(np.float32)

        logging.info("Load embedding from %s", folder)

        return cls(vocab, Tensor(embeddings), **hyper)
