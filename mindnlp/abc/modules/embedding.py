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
from mindspore import ops
from mindspore import Parameter, Tensor
import mindspore.numpy as mnp

class TokenEmbedding(nn.Cell):
    r"""
    Embedding base class
    """

    def __init__(self, vocab, init_embed,
                 word_dropout=0, dropout=0.5, unk_index=None, requires_grad=False):
        super().__init__()

        self._word_vocab = vocab
        self.embed = Parameter(init_embed, name='embed', requires_grad=requires_grad)
        self.dropout = nn.Dropout(dropout)
        self._word_pad_index = None
        self._embed_size = self.embed.shape
        if word_dropout > 0 and not isinstance(unk_index, int):
            raise ValueError("When drop word is set, you need to pass in the unk_index.")
        self.unk_index = unk_index
        self.word_dropout = word_dropout

    def drop_word(self, words):
        r"""
        Randomly set Words to UNKNOWN_INDEX
        """
        if self.word_dropout > 0 and self.training:
            mask = mnp.full_like(words, fill_value=self.word_dropout, dtype=float, shape=None)
            mask = ops.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            pad_mask = words.ne(self._word_pad_index)
            mask = mask & pad_mask
            words = words.masked_fill(mask, self._word_unk_index)
        return words

    def __len__(self):
        return len(self.embed)

    def embed_size(self) -> int:
        """embed size"""
        return self._embed_size

    def num_embeddings(self) -> int:
        """num embeddings"""
        return len(self._word_vocab)

    def get_word_vocab(self):
        """get word vocab"""
        return self._word_vocab

    def size(self):
        """size"""
        return self.embed.size

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
        output_for_reshape = ops.gather(self.embedding_table, flat_ids, 0)
        output = ops.reshape(output_for_reshape, out_shape)
        return output

    @abstractmethod
    def from_pretrained(self, url: str):
        r"""
        Creates Embedding.
        """

        raise NotImplementedError
