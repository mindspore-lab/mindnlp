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
# pylint:disable=I1101
# pylint:disable=I1101

"""Vocab Class"""

import os
import re
import warnings
from typing import Union
from collections import OrderedDict
import mindspore._c_dataengine as cde
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import cache_file


class Vocab:
    r"""
    Creates a vocab object which maps tokens to indices.
    """

    def __init__(self, list_or_dict: Union[list, OrderedDict, None] = None,
                 special_tokens: Union[list, tuple] = ("<pad>", "<unk>"),
                 special_first: bool = True):

        self._dict = None
        self._unk_token = None
        self._c_vocab = None

        if list_or_dict is None:
            return

        self._set_unk(special_tokens)

        if isinstance(list_or_dict, list):
            self._dict = OrderedDict.fromkeys(list_or_dict)
        elif isinstance(list_or_dict, OrderedDict):
            self._dict = list_or_dict
        else:
            raise ValueError(f'Vocab only support list or OrderedDict, but get {type(list_or_dict)}')

        if special_first:
            self._dict.update((index, value) for index, value in enumerate(special_tokens))
        else:
            dict_len = len(self._dict)
            for idx, tok in enumerate(special_tokens):
                self._dict[tok] = dict_len + idx

        self._c_vocab = cde.Vocab.from_dict(self._dict)

    def _set_unk(self, special_tokens):
        if '<unk>' in special_tokens:
            self._unk_token = '<unk>'
        else:
            warnings.warn("Warning: can not find '<unk>' in vocab, "
                          "please use Vocab.set_default_index() to set index of unknown token.",
                          UserWarning)


    def __len__(self) -> int:
        r"""
        Returns:
            - int, The length of the vocab.
        """
        return len(self._dict)

    def __contains__(self, token: str) -> bool:
        r"""
        Args:
            token (str): The token for which to check the membership.

        Returns:
            - bool, Whether the token is member of vocab or not.
        """
        return token in self._dict

    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token (str): The token used to lookup the corresponding index.

        Returns:
            - int, The index corresponding to the associated token.
        """

        return self._dict.get(token, None)

    @classmethod
    def from_dataset(cls, dataset, columns=None, freq_range=None, top_k=None, special_tokens=None, special_first=True):
        """
        Build a Vocab from a dataset.

        This would collect all unique words in a dataset and return a vocab within
        the frequency range specified by user in freq_range. User would be warned if no words fall into the frequency.
        Words in vocab are ordered from the highest frequency to the lowest frequency. Words with the same frequency
        would be ordered lexicographically.

        Args:
            dataset (Dataset): dataset to build vocab from.
            columns (list[str], optional): column names to get words from. It can be a list of column names.
                Default: None.
            freq_range (tuple, optional): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency=0 is the same as
                min_frequency=1. max_frequency > total_words is the same as max_frequency = total_words.
                min_frequency/max_frequency can be None, which corresponds to 0/total_words separately.
                Default: None, all words are included.
            top_k (int, optional): top_k is greater than 0. Number of words to be built into vocab. top_k means most
                frequent words are taken. top_k is taken after freq_range. If not enough top_k, all words will be taken.
                Default: None, all words are included.
            special_tokens (list, optional):  A list of strings, each one is a special token. For example
                special_tokens=["<pad>","<unk>"]. Default: None, no special tokens will be added.
            special_first (bool, optional): Whether special_tokens will be prepended/appended to vocab. If
                special_tokens is specified and special_first is set to True, special_tokens will be prepended.
                Default: True.

        Returns:
            - Vocab, Vocab object built from the dataset.

        Examples:
            >>> import mindspore.dataset as ds
            >>> import mindspore.dataset.text as text
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> vocab = text.Vocab.from_dataset(dataset, "text", freq_range=None, top_k=None,
            ...                                 special_tokens=["<pad>", "<unk>"],
            ...                                 special_first=True)
            >>> dataset = dataset.map(operations=text.Lookup(vocab, "<unk>"), input_columns=["text"])
        """

        vocab = Vocab()
        # pylint: disable=protected-access
        vocab._c_vocab = dataset._build_vocab(columns, freq_range, top_k, special_tokens, special_first)
        vocab._set_unk(special_tokens)
        vocab._dict = vocab._c_vocab.vocab()

        return vocab


    def lookup_ids(self, tokens):
        """
        Converts a token string or a sequence of tokens in a single integer id or a sequence of ids.

        Args:
            tokens (Union[str, list[str]]): One or several token(s) to convert to token id(s).

        Returns:
            - list[int], The token id or list of token ids.
              if only one token used to lookup,
              return one id instead of a list of ids.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> ids = vocab.lookup_ids(["w1", "w3"])
        """
        if isinstance(tokens, list):
            for token in tokens:
                if not token not in self._dict:
                    raise RuntimeError(f"{token} is not in vocab.")
        if isinstance(tokens, str):
            if not tokens not in self._dict:
                raise RuntimeError(f"{tokens} is not in vocab.")
            tokens = [tokens]
        return self.c_vocab.tokens_to_ids(tokens)

    def lookup_tokens(self, ids):
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens.
        If id does not exist, return empty string.

        Args:
            ids (Union[int, list[int]]): The token id (or token ids) to convert to tokens.

        Returns:
            - List<str>, The decoded token(s).
              if only one id used to lookup,
              return one token instead of a list of tokens.

        Raises:
            RuntimeError: If 'ids' is not in vocab.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> token = vocab.lookup_tokens(0)
        """
        if isinstance(ids, list):
            for idx in ids:
                if idx not in range(0, len(self._dict)):
                    raise RuntimeError(f"idx {idx} is not in vocab.")
        if isinstance(ids, int):
            if ids not in range(0, len(self._dict)):
                raise RuntimeError(f"idx {ids} is not in vocab.")
            ids = [ids]
        return self.c_vocab.ids_to_tokens(ids)

    def append_token(self, token):
        r"""
        Args:
            token (str): The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `token` already exists in the vocab.

        """
        if isinstance(token, str):
            if token in self._dict:
                warnings.warn(f"{token} already exists in the vocab.")
            else:
                append_id = len(self._dict)
                self._dict[token] = append_id
                self._c_vocab = cde.Vocab.from_dict(self._dict)
        else:
            raise TypeError(f"{token} is not a str type.")

    def insert_token(self, token, index):
        r"""
        Args:
            token (str): The token used to lookup the corresponding index.
            index (int): The index corresponding to the associated token.

        Raises:
            TypeError: If 'token' is not a str.
            TypeErrpr: If 'index' is not an int.
            RuntimeError: If 'token' already exists in the vocab,
            RuntimeError: If `index` is not in range [0, Vocab.__len__()].
        """
        if not isinstance(token, str):
            raise TypeError(f"token {token} is not str type.")
        if not isinstance(index, int):
            raise TypeError(f"index {index} is not int type.")
        if index not in range(0, len(self._dict)):
            raise RuntimeError(f"index {index} is out of range [0, {len(self._dict)}]")
        if token in self._dict:
            warnings.warn(f"{token} already exists in the vocab.")
        else:
            self._dict.update({token: index})
            self._c_vocab = cde.Vocab.from_dict(self._dict)

    @classmethod
    def from_pretrained(cls, name="glove.6B.50d", root=DEFAULT_ROOT,
                        special_tokens=("<pad>", "<unk>"), special_first=True):
        r"""
        Args:
            name (str): The name of the pretrained vector. Default: "glove.6B.50d".
            root (str): Default storage directory. Default: DEFAULT_ROOT.
            special_tokens (str|Tuple[str]): List of special participles. Default: ("<pad>", "<unk>").
            special_first (bool): Indicates whether special participles from special_tokens will be added to
                the top of the dictionary. If True, add special_tokens to the beginning of the dictionary,
                otherwise add them to the end. Default: True.

        Returns:
            - Vocab, Returns a vocab generated from the url download.
        """

        tokens = []
        url = pretrained_aliases[name]

        cache_dir = os.path.join(root, "vocabs")
        download_file_name = re.sub(r".+/", "", url)
        path, _ = cache_file(filename=download_file_name, cache_dir=cache_dir, url=url)

        with open(path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                tokens.append(line.rstrip("\n"))

        vocab = Vocab(tokens, list(special_tokens), special_first)

        return vocab

    def get_default_index(self):
        r"""
        Returns:
            Value of default index if it is set.
        """
        if self._unk_token is None:
            raise ValueError("Can not find '<unk>' in vocab.")
        return self._dict.get(self._unk_token)

    def set_default_index(self, index=None):
        r"""
        Args:
            index: Value of default index. This index will be returned when OOV token is queried.
        """
        if self._unk_token is not None:
            warnings.warn("default index has been set, the new setting will be automatically ignored.")
        else:
            self._unk_token = '<unk>'
            self._dict['<unk>'] = index if index is not None else len(self._dict)
            self._c_vocab = cde.Vocab.from_dict(self._dict)

pretrained_aliases = {
    "glove.6B.50d": "https://huggingface.co/datasets/Aore/MindNLP_vocab/resolve/main/glove.6B.50d.vocab.txt",
    "glove.6B.100d": "https://huggingface.co/datasets/Aore/MindNLP_vocab/resolve/main/glove.6B.100d.vocab.txt",
    "glove.6B.200d": "https://huggingface.co/datasets/Aore/MindNLP_vocab/resolve/main/glove.6B.200d.vocab.txt",
    "glove.6B.300d": "https://huggingface.co/datasets/Aore/MindNLP_vocab/resolve/main/glove.6B.300d.vocab.txt",
    "fasttext": "https://huggingface.co/datasets/Aore/MindNLP_vocab/resolve/main/wiki-news-300d-1M.txt",
    "fasttext-subword": "https://huggingface.co/datasets/Aore/MindNLP_vocab/resolve/main/wiki-news-300d-1M-subword.txt",
}
