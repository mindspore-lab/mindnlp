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
"""Test Fasttext_embedding"""

import unittest
from mindspore import Tensor
from mindnlp.modules.embeddings.fasttext_embedding import Fasttext


class TestFasttext(unittest.TestCase):
    r"""
    Test module Fasttext
    """

    def setUp(self):
        r"""
        Set up.
        """
        self.input = None

    def test_fasttext_embedding(self):
        r"""
        Unit test for fasttext embedding.
        """
        wordlist = [0, 2]
        wordlist_input = Tensor(wordlist)

        init_embed = Tensor([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6],
                             [0.7, 0.8, 0.9]])

        embed = Fasttext(init_embed=init_embed)
        f_res = embed(wordlist_input)

        assert f_res.shape == (2, 3)
