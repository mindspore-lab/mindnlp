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
"""embedding"""

__all__ = [
    "TokenEmbedding"
]

from abc import abstractmethod
from mindspore import nn
from mindspore import Parameter


class TokenEmbedding(nn.Cell):
    r"""
    Embedding base class
    """

    def __init__(self, vocab, init_embed, dropout=0.5, requires_grad=False):
        super().__init__()

        self._word_vocab = vocab
        self.embed = Parameter(init_embed, name='embed', requires_grad=requires_grad)
        self.dropout_layer = nn.Dropout(1 - dropout)
        self._embed_size = self.embed.shape

    def dropout(self):
        r"""
        drop the word after embedding.
        """
        return self.dropout_layer

    def __len__(self):
        return len(self.embed)

    def embed_size(self):
        """embed size"""
        return self._embed_size

    def num_embeddings(self):
        """num embeddings"""
        return len(self._word_vocab.vocab())

    def get_word_vocab(self):
        """get word vocab"""
        return self._word_vocab.vocab()

    @abstractmethod
    def construct(self, ids):
        r"""
        Use ids to query embedding
        Args:
            ids : Ids to query.

        Raises:
            NotImplementedError: If this interface is called.

        """
        raise NotImplementedError(f'Function `construct` not implemented in {self.__class__.__name__}')
