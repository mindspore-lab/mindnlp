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
from mindnlp.common.metrics import Accuracy, F1Score, BleuScore
from mindnlp.common.metrics import (perplexity, bleu, rouge_n, rouge_l, distinct, accuracy, precision,
                                     recall, f1_score, confusion_matrix, mcc, pearson, spearman, em_score)

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

class TestAccuracy(unittest.TestCase):
    r"""
    Test accuracy
    """

    def setUp(self):
        self.input = None

    def test_accuracy(self):
        """
        Test accuracy
        """
        preds = [[0.1, 0.9], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
        labels = [1, 0, 1, 1]
        acc = accuracy(preds, labels)

        assert acc == 0.5

class TestPrecision(unittest.TestCase):
    r"""
    Test precision
    """

    def setUp(self):
        self.input = None

    def test_precision(self):
        """
        Test precision
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        prec = precision(preds, labels)

        assert prec == 0.5

class TestRecall(unittest.TestCase):
    r"""
    Test recall
    """

    def setUp(self):
        self.input = None

    def test_recall(self):
        """
        Test recall
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        rec = recall(preds, labels)

        assert rec == 0.5

class TestF1Score(unittest.TestCase):
    r"""
    Test F1 score
    """

    def setUp(self):
        self.input = None

    def test_f1_score(self):
        """
        Test recall
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        f1_s = f1_score(preds, labels)

        assert f1_s == 0.6666666666666666

class TestConfusionMatrix(unittest.TestCase):
    r"""
    Test confusion matrix
    """

    def setUp(self):
        self.input = None

    def test_confusion_matrix(self):
        """
        Test confusion matrix
        """
        preds = Tensor(np.array([1, 0, 1, 0]))
        labels = Tensor(np.array([1, 0, 0, 1]))
        conf_mat = confusion_matrix(preds, labels)

        assert np.array_equal(conf_mat, np.array([[1., 1.], [1., 1.]]))

class TestMcc(unittest.TestCase):
    r"""
    Test MCC
    """

    def setUp(self):
        self.input = None

    def test_mcc(self):
        """
        Test MCC
        """
        preds = [[0.1, 0.9], [-0.5, 0.5], [0.1, 0.4], [0.1, 0.3]]
        labels = [[1], [0], [1], [1]]
        m_c_c = mcc(preds, labels)

        assert m_c_c == 0.0

class TestPearson(unittest.TestCase):
    r"""
    Test PCC
    """

    def setUp(self):
        self.input = None

    def test_pearson(self):
        """
        Test PCC
        """
        preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        pcc = pearson(preds, labels)

        assert pcc == 0.9985229081857804

class TestSpearman(unittest.TestCase):
    r"""
    Test SCC
    """

    def setUp(self):
        self.input = None

    def test_spearman(self):
        """
        Test SCC
        """
        preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        scc = spearman(preds, labels)

        assert scc == 1.0

class TestEmScore(unittest.TestCase):
    r"""
    Test exact match score
    """

    def setUp(self):
        self.input = None

    def test_em_score(self):
        """
        Test exact match score
        """
        preds = "this is the best span"
        examples = ["this is a good span", "something irrelevant"]
        exact_match = em_score(preds, examples)

        assert exact_match == 0.0

class TestClassAccuracy(unittest.TestCase):
    r"""
    Test class Accuracy
    """

    def setUp(self):
        self.input = None

    def test_class_accuracy(self):
        """
        Test class Accuracy
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.float32)
        metric = Accuracy()
        metric.updates(preds, labels)
        acc = metric.eval()

        assert acc == 0.6666666666666666

class TestClassF1Score(unittest.TestCase):
    r"""
    Test class F1Score
    """

    def setUp(self):
        self.input = None

    def test_class_f1_score(self):
        """
        Test class F1Score
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        metric = F1Score()
        metric.updates(preds, labels)
        f1_s = metric.eval()

        assert f1_s == 0.6666666666666666

class TestClassBleuScore(unittest.TestCase):
    r"""
    Test class BleuScore
    """

    def setUp(self):
        self.input = None

    def test_class_bleu_score(self):
        """
        Test class BleuScore
        """
        cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]]
        metric = BleuScore()
        metric.updates(cand, ref_list)
        bleu_score = metric.eval()

        assert bleu_score == 0.46713797772820015
