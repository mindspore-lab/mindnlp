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

"""Vocab Class"""

import os
import re
import warnings
from typing import Union
from mindspore.dataset import TextBaseDataset
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils.download import get_from_cache

class Vocab:
    r"""
    Creates a vocab object which maps tokens to indices.
    """

    def __init__(self, list_or_dict: Union[list, dict],
                 special_tokens: Union[list, tuple] = None,
                 special_first: bool = True):

        self._token_dict = {}

        sp_len = len(special_tokens) if special_tokens is not None and special_first else 0

        if isinstance(list_or_dict, list):
            for index, value in enumerate(list_or_dict):
                self._token_dict[value] = index + sp_len
        elif isinstance(list_or_dict, dict):
            for key, value in list_or_dict.items():
                if not isinstance(key, str):
                    raise ValueError(f'keys in dict must be str, but got {type(key)}')
                if not isinstance(value, int):
                    raise ValueError(f'values in dict must be int, but got {type(key)}')
                self._token_dict[key] = value + sp_len
        else:
            raise ValueError(f'Vocab only support list or dict, but get {type(list_or_dict)}')

        if special_tokens is not None:
            offset = 0 if special_first else len(self._token_dict)
            for idx, tok in enumerate(special_tokens):
                self._token_dict[tok] = idx + offset

        self._index_dict = {v: k for k, v in self._token_dict.items()}

    def __len__(self) -> int:
        r"""
        Returns:
            - int, The length of the vocab.
        """
        return len(self._token_dict)

    def __contains__(self, token: str) -> bool:
        r"""
        Args:
            token (str): The token for which to check the membership.

        Returns:
            - bool, Whether the token is member of vocab or not.
        """
        return token in self._token_dict

    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token (str): The token used to lookup the corresponding index.

        Returns:
            - int, The index corresponding to the associated token.
        """

        return self._token_dict.get(token, None)


    def __call__(self, token_or_id):
        if isinstance(token_or_id, str):
            return self._token_dict.get(token_or_id, None)
        if isinstance(token_or_id, int):
            return self._index_dict.get(token_or_id, None)

        raise ValueError(f'not support token type {type(token_or_id)}')

    def lookup_ids(self, token_or_list):
        """
        Converts a token string or a sequence of tokens in a single integer id or a sequence of ids.

        Args:
            token_or_list (Union[str, list[str]]): One or several token(s) to convert to token id(s).

        Returns:
            - list[int], The token id or list of token ids.
              if only one token used to lookup,
              return one id instead of a list of ids.

        Examples:
            >>> import mindspore.dataset.text as text
            >>> vocab = text.Vocab.from_list(["w1", "w2", "w3"], special_tokens=["<unk>"], special_first=True)
            >>> ids = vocab.lookup_ids(["w1", "w3"])
        """
        if isinstance(token_or_list, str):
            return self._token_dict.get(token_or_list)

        if isinstance(token_or_list, list):
            return_list = []
            for token in token_or_list:
                if token not in self._token_dict:
                    raise ValueError(f"{token} is not in vocab.")
                return_list.append(self._token_dict[token])
            return return_list

        raise ValueError(f'lookup only support str and list, but got {type(token_or_list)}.')

    def lookup_tokens(self, index_or_list):
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens.
        If id does not exist, return empty string.

        Args:
            index_or_list (Union[int, list[int]]): The token id (or token ids) to convert to tokens.

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
        if isinstance(index_or_list, int):
            return self._index_dict.get(index_or_list, None)

        if isinstance(index_or_list, list):
            return_list = []
            for idx in index_or_list:
                if idx not in self._index_dict:
                    raise ValueError(f"{idx} is not in vocab.")
                return_list.append(self._index_dict[idx])
            return return_list

        raise ValueError(f'lookup only support int and list, but got {type(index_or_list)}.')

    def append_token(self, token):
        r"""
        Args:
            token (str): The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `token` already exists in the vocab.

        """
        if isinstance(token, str):
            if token in self._token_dict:
                warnings.warn(f"{token} already exists in the vocab.")
            else:
                append_id = len(self._token_dict)
                self._token_dict[token] = append_id
                self._index_dict[append_id] = token
        else:
            raise TypeError(f"{token} is not str.")

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

        # pylint: disable=protected-access
        if not isinstance(dataset, TextBaseDataset):
            raise ValueError('dataset must be subclass of TextBaseDataset.')

        ds_vocab = dataset._build_vocab(columns, freq_range, top_k, special_tokens, special_first)
        vocab = Vocab(ds_vocab.vocab())

        return vocab

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
        path = get_from_cache(download_file_name=download_file_name, cache_dir=cache_dir, url=url)

        with open(path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                tokens.append(line.rstrip("\n"))

        vocab = Vocab(tokens, list(special_tokens), special_first)

        return vocab

    @property
    def vocab(self):
        """return vocab dict."""
        return self._token_dict

pretrained_aliases = {
    "glove.6B.50d": "https://download.mindspore.cn/toolkits/mindnlp/vocab/Glove/glove.6B.50d.txt",
    "glove.6B.100d": "https://download.mindspore.cn/toolkits/mindnlp/vocab/Glove/glove.6B.100d.txt",
    "glove.6B.200d": "https://download.mindspore.cn/toolkits/mindnlp/vocab/Glove/glove.6B.200d.txt",
    "glove.6B.300d": "https://download.mindspore.cn/toolkits/mindnlp/vocab/Glove/glove.6B.300d.txt",
    "fasttext": "https://download.mindspore.cn/toolkits/mindnlp/vocab/Fasttext/wiki-news-300d-1M.txt",
    "fasttext-subword": "https://download.mindspore.cn/toolkits/mindnlp/vocab/Fasttext/wiki-news-300d-1M-subword.txt",
}

__all__ = ['Vocab']
