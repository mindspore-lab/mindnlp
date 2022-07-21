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
"""Test Metrics"""

import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from text.common.metrics import perplexity, bleu, rouge_n, rouge_l, distinct

class TestPerplexity(unittest.TestCase):
    r"""
    Test perplexity
    """

    def setUp(self):
        self.input = None

    def test_perplexity(self):
        """
        Test perplexity
        """
        pred_data = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        expect_label = Tensor(np.array([1, 0, 1]), mindspore.int32)
        ppl = perplexity(pred_data, expect_label, ignore_label=None)

        assert ppl == 2.2314431850023855

class TestBleu(unittest.TestCase):
    r"""
    Test BLEU
    """

    def setUp(self):
        self.input = None

    def test_bleu(self):
        """
        Test BLEU
        """
        cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]]
        bleu_score = bleu(cand, ref_list)

        assert bleu_score == 0.46713797772820015

class TestRougeN(unittest.TestCase):
    r"""
    Test ROUGE-N
    """

    def setUp(self):
        self.input = None

    def test_rougen(self):
        """
        Test ROUGE-N
        """
        cand_list = [["a", "cat", "is", "on", "the", "table"]]
        ref_list = [["there", "is", "a", "cat", "on", "the", "table"]]
        rougen_score = rouge_n(cand_list, ref_list)

        assert rougen_score == 0.8571428571428571

class TestRougeL(unittest.TestCase):
    r"""
    Test ROUGE-L
    """

    def setUp(self):
        self.input = None

    def test_rougel(self):
        """
        Test ROUGE-L
        """
        cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        ref_list = [["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]
        rougel_score = rouge_l(cand_list, ref_list)

        assert rougel_score == 0.7800511508951408

class TestDistinct(unittest.TestCase):
    r"""
    Test distinct-n
    """

    def setUp(self):
        self.input = None

    def test_distinct(self):
        """
        Test distinct-n
        """
        cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        distinct_score = distinct(cand_list)

        assert distinct_score == 0.8333333333333334
