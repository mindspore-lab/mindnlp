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
import numpy as np
from mindspore import nn
from mindspore import Parameter
from mindspore.dataset.text.utils import Vocab


class TokenEmbedding(nn.Cell):
    r"""
    Embedding base class
    """

    def __init__(self, vocab: Vocab, init_embed, requires_grad: bool = True, dropout=0.5, train_state: bool = True):
        super().__init__()

        self._word_vocab = vocab
        self.embed = Parameter(init_embed, name='embed', requires_grad=requires_grad)
        self.dropout_layer = nn.Dropout(1 - dropout)
        self._embed_size = self.embed.shape
        self.train_state = train_state

    def dropout(self, words):
        r"""
        drop the word after embedding.

        Args:
            words (Tensor): Tensor about to be dropout.
        Returns:
            - ** net(words) ** - Dropout processed data.
        """
        net = self.dropout_layer.set_train(self.train_state)
        words = words.astype(np.float32)
        return net(words)

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
