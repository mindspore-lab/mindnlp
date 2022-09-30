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
"""Test Word2vec_embedding"""

import unittest
from mindspore import Tensor
from mindspore.dataset.text import Vocab
from mindnlp.modules.embeddings.word2vec_embedding import Word2vec


class TestWord2vec(unittest.TestCase):
    r"""
    Test module Word2vec
    """

    def setUp(self):
        self.input = None

    def test_word2vec_embedding(self):
        r"""
        Unit test for word2vec embedding.
        """
        wordlist = [0, 2]
        wordlist_input = Tensor(wordlist)

        init_embed = Tensor([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6],
                             [0.7, 0.8, 0.9]])

        init_vocab = Vocab.from_list(['i', 'am', 'human'])
        embed = Word2vec(vocab=init_vocab, init_embed=init_embed)
        w_res = embed(wordlist_input)

        assert w_res.shape == (2, 3)
