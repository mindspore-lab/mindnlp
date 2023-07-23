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
"""Test Classes for Metrics"""


import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.metrics import (Perplexity, BleuScore, RougeN, RougeL, Distinct, Accuracy,
                             Precision, Recall, F1Score, ConfusionMatrix,
                             MatthewsCorrelation, PearsonCorrelation,
                             SpearmanCorrelation, EmScore)


class TestClassPerplexity(unittest.TestCase):
    r"""
    Test class Perplexity
    """
    def test_class_perplexity_tensor(self):
        """
        Test class Perplexity
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([1, 0, 1]))

        metric = Perplexity()
        metric.update(preds, labels)

        ppl = metric.eval()

        assert np.allclose(ppl, 2.23144, 1e-5, 1e-5)


    def test_class_perplexity_tensor_onehot(self):
        """
        Test class Perplexity
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))

        metric = Perplexity()
        metric.update(preds, labels)

        ppl = metric.eval()

        assert np.allclose(ppl, 2.23144, 1e-5, 1e-5)


    def test_class_perplexity_np(self):
        """
        Test class Perplexity
        """
        preds = np.array([[0.6, 0.5, 0.1], [0.3, 0.6, 0.05], [0.1, 0.6, 0.2], [0.1, 0.2, 0.7]])
        labels = np.array([2, 1, 0, 1])

        metric = Perplexity()
        metric.update(preds, labels)

        ppl = metric.eval()

        assert np.allclose(ppl, 5.37284, 1e-5, 1e-5)


    def test_class_perplexity_list_multi(self):
        """
        Test class Perplexity
        """
        preds = [[0.6, 0.5, 0.1], [0.3, 0.6, 0.05], [0.1, 0.6, 0.2], [0.1, 0.2, 0.7]]
        labels = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]]

        metric = Perplexity()
        metric.update(preds, labels)

        ppl = metric.eval()

        assert np.allclose(ppl, 5.37284, 1e-5, 1e-5)


    def test_class_perplexity_update_clear(self):
        """
        Test class Perplexity

        preds (list): shape (4, 3)
        labels (list): shape (4, 1), one-hot encoding
        """
        preds1 = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels1 = Tensor(np.array([1, 0, 1]))

        metric = Perplexity()
        metric.update(preds1, labels1)

        preds2 = Tensor(np.array([[0.6, 0.5], [0.3, 0.6], [0.1, 0.6]]))
        labels2 = Tensor(np.array([0, 1, 0]))

        metric.update(preds2, labels2)
        ppl = metric.eval()

        assert np.allclose(ppl, 2.59961, 1e-5, 1e-5)

        metric.clear()
        metric.update(preds1, labels1)

        ppl = metric.eval()

        assert np.allclose(ppl, 2.23144, 1e-5, 1e-5)


class TestClassBleuScore(unittest.TestCase):
    r"""
    Test class BleuScore
    """
    def test_class_bleu_score_normal(self):
        """
        Test class BleuScore
        """
        cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"],
                    ["There", "is", "a", "cat", "on", "the", "mat"]]]

        metric = BleuScore()
        metric.update(cand, ref_list)

        bleu_score = metric.eval()

        assert np.allclose(bleu_score, 0.46713, 1e-5, 1e-5)


    def test_class_bleu_score_perfect(self):
        """
        Test class BleuScore
        """
        cand = [["The", "cat", "is", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"],
                    ["There", "is", "a", "cat", "on", "the", "mat"]]]

        metric = BleuScore()
        metric.update(cand, ref_list)

        bleu_score = metric.eval()

        assert np.allclose(bleu_score, 1.0, 1e-5, 1e-5)


    def test_class_bleu_score_warning(self):
        """
        Test class BleuScore
        """
        cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"]]]

        metric = BleuScore()
        metric.update(cand, ref_list)

        bleu_score = metric.eval()

        assert np.allclose(bleu_score, 0.0, 1e-5, 1e-5)


    def test_class_bleu_nsize_weights(self):
        """
        Test class BleuScore
        """
        cand = [["this", "is", "small", "cat"]]
        ref_list = [[["this", "is", "a", "small", "cat"]]]

        metric = BleuScore(2, [0.5, 0.5])
        metric.update(cand, ref_list)

        bleu_score = metric.eval()

        assert np.allclose(bleu_score, 0.63588, 1e-5, 1e-5)


    def test_class_bleu_update_clear(self):
        """
        Test class BleuScore
        """
        cand1 = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list1 = [[["The", "cat", "is", "on", "the", "mat"],
                    ["There", "is", "a", "cat", "on", "the", "mat"]]]

        metric = BleuScore()
        metric.update(cand1, ref_list1)

        cand2 = [["The", "cat", "is", "on", "the", "mat"]]
        ref_list2 = [[["The", "cat", "is", "on", "the", "mat"],
                    ["There", "is", "a", "cat", "on", "the", "mat"]]]

        metric.update(cand2, ref_list2)

        bleu_score = metric.eval()

        assert np.allclose(bleu_score, 0.71662, 1e-5, 1e-5)

        metric.clear()
        metric.update(cand1, ref_list1)
        bleu_score = metric.eval()

        assert np.allclose(bleu_score, 0.46713, 1e-5, 1e-5)


class TestClassRougeN(unittest.TestCase):
    r"""
    Test class RougeN
    """
    def test_class_rougen(self):
        """
        Test class RougeN
        """
        cand_list = ["a", "cat", "is", "on", "the", "table"]
        ref_list = [["there", "is", "a", "cat", "on", "the", "table"]]

        metric = RougeN(2)
        metric.update(cand_list, ref_list)

        rougen_score = metric.eval()

        assert np.allclose(rougen_score, 0.5, 1e-5, 1e-5)


    def test_class_rougen_1(self):
        """
        Test class RougeN
        """
        cand_list = ["the", "cat", "was", "found", "under", "the", "bed"]
        ref_list = [["the", "cat", "was", "under", "the", "bed"]]

        metric = RougeN()
        metric.update(cand_list, ref_list)

        rougen_score = metric.eval()

        assert np.allclose(rougen_score, 1.0, 1e-5, 1e-5)


    def test_class_rougen_2(self):
        """
        Test class RougeN
        """
        cand_list = ["the", "cat", "was", "found", "under", "the", "bed"]
        ref_list = [["the", "cat", "was", "under", "the", "bed"]]

        metric = RougeN(2)
        metric.update(cand_list, ref_list)

        rougen_score = metric.eval()

        assert np.allclose(rougen_score, 0.8, 1e-5, 1e-5)


    def test_class_rougen_ref(self):
        """
        Test class RougeN
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"],
                    ["There","is","a","cat","on","the","mat"]]

        metric = RougeN(2)
        metric.update(cand_list, ref_list)

        rougen_score = metric.eval()

        assert np.allclose(rougen_score, 0.54545, 1e-5, 1e-5)


    def test_class_rougen_update_clear(self):
        """
        Test class RougeN
        """
        cand_list1 = ["a", "cat", "is", "on", "the", "table"]
        ref_list1 = [["there", "is", "a", "cat", "on", "the", "table"]]

        metric = RougeN(2)
        metric.update(cand_list1, ref_list1)

        cand_list2 = ["the", "cat", "was", "found", "under", "the", "bed"]
        ref_list2 = [["the", "cat", "was", "under", "the", "bed"]]

        metric.update(cand_list2, ref_list2)
        rougen_score = metric.eval()

        assert np.allclose(rougen_score, 0.63636, 1e-5, 1e-5)

        metric.clear()
        metric.update(cand_list1, ref_list1)

        rougen_score = metric.eval()

        assert rougen_score == 0.5

class TestClassRougeL(unittest.TestCase):
    r"""
    Test class RougeL
    """
    def test_class_rougel(self):
        """
        Test class RougeL
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"]]

        metric = RougeL()
        metric.update(cand_list, ref_list)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.78005, 1e-5, 1e-5)


    def test_class_rougel_ref1(self):
        """
        Test class RougeL
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["There","is","a","cat","on","the","mat"]]

        metric = RougeL()
        metric.update(cand_list, ref_list)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.57142, 1e-5, 1e-5)


    def test_class_rougel_ref2(self):
        """
        Test class RougeL
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"],
                    ["There","is","a","cat","on","the","mat"]]

        metric = RougeL()
        metric.update(cand_list, ref_list)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.78005, 1e-5, 1e-5)


    def test_class_rougel_beta(self):
        """
        Test class RougeL
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"],
                    ["There","is","a","cat","on","the","mat"]]

        metric = RougeL(0.5)
        metric.update(cand_list, ref_list)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.73529, 1e-5, 1e-5)


    def test_class_rougel_update_clear(self):
        """
        Test class RougeL
        """
        cand_list1 = ["The","cat","The","cat","on","the","mat"]
        ref_list1 = [["The","cat","is","on","the","mat"]]

        metric = RougeL()
        metric.update(cand_list1, ref_list1)

        cand_list2 = ["The","cat","The","cat","on","the","mat"]
        ref_list2 = [["There","is","a","cat","on","the","mat"]]

        metric.update(cand_list2, ref_list2)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.67573, 1e-5, 1e-5)

        metric.clear()
        metric.update(cand_list1, ref_list1)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.78005, 1e-5, 1e-5)

class TestClassDistinct(unittest.TestCase):
    r"""
    Test class Distinct
    """
    def test_class_distinct(self):
        """
        Test class Distinct
        """
        cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]

        metric = Distinct()
        metric.update(cand_list)

        rougel_score = metric.eval()

        assert np.allclose(rougel_score, 0.83333, 1e-5, 1e-5)


    def test_class_distinct_one(self):
        """
        Test class Distinct
        """
        cand_list = ["The", "cat", "on", "the", "mat"]

        metric = Distinct()
        metric.update(cand_list)

        rougel_score = metric.eval()

        assert rougel_score == 1.0


    def test_class_distinct_update_clear(self):
        """
        Test class Distinct
        """
        cand_list1 = ["The", "cat", "The", "cat", "on", "the", "mat"]

        metric = Distinct()
        metric.update(cand_list1)

        cand_list2 = ["The", "cat", "on", "the", "mat"]
        metric.update(cand_list2)

        rougel_score = metric.eval()

        assert rougel_score == 0.5

        metric.clear()
        metric.update(cand_list2)

        rougel_score = metric.eval()

        assert rougel_score == 1.0

class TestClassAccuracy(unittest.TestCase):
    r"""
    Test class Accuracy
    """
    def test_class_accuracy_tensor(self):
        """
        Test class Accuracy
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.float32)

        metric = Accuracy()
        metric.update(preds, labels)

        acc = metric.eval()

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)


    def test_class_accuracy_tensor_onehot(self):
        """
        Test class Accuracy
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mindspore.float32)

        metric = Accuracy()
        metric.update(preds, labels)

        acc = metric.eval()

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)


    def test_class_accuracy_np_multi(self):
        """
        Test class Accuracy
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])

        metric = Accuracy()
        metric.update(preds, labels)

        acc = metric.eval()

        assert acc == 0.25


    def test_class_accuracy_list(self):
        """
        Test class Accuracy
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]

        metric = Accuracy()
        metric.update(preds, labels)

        acc = metric.eval()

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)


    def test_class_accuracy_update_clear(self):
        """
        Test class Accuracy
        """
        preds1 = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels1 = Tensor(np.array([1, 0, 1]), mindspore.float32)

        metric = Accuracy()
        metric.update(preds1, labels1)

        preds2 = np.array([[0.5, 0.05], [0.3, 0], [0.9, 0.05], [0.1, 0.2]])
        labels2 = np.array([1, 0, 1, 1])

        metric.update(preds2, labels2)
        acc = metric.eval()

        assert np.allclose(acc, 0.57142, 1e-5, 1e-5)

        metric.clear()

        metric.update(preds1, labels1)
        acc = metric.eval()

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)

class TestClassPrecision(unittest.TestCase):
    r"""
    Test class Precision
    """
    def test_class_precision_tensor(self):
        """
        Test class Precision
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)

        metric = Precision()
        metric.update(preds, labels)

        prec = metric.eval()

        assert np.array_equal(prec, [0.5, 1.0])


    def test_class_precision_tensor_onehot(self):
        """
        Test class Precision
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mindspore.float32)

        metric = Precision()
        metric.update(preds, labels)

        prec = metric.eval()

        assert np.array_equal(prec, [0.5, 1.0])


    def test_class_precision_np_multi(self):
        """
        Test class Precision
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])

        metric = Precision()
        metric.update(preds, labels)

        prec = metric.eval()

        assert np.array_equal(prec, [0., 1., 0., 0.])


    def test_class_precision_list(self):
        """
        Test class Precision
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]

        metric = Precision()
        metric.update(preds, labels)

        prec = metric.eval()

        assert np.array_equal(prec, [0.5, 1.0])


    def test_class_precision_update_clear(self):
        """
        Test class Precision
        """
        preds1 = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels1 = Tensor(np.array([1, 0, 1]), mindspore.int32)

        metric = Precision()
        metric.update(preds1, labels1)

        preds2 = np.array([[0.5, 0.05], [0.3, 0], [0.9, 0.05], [0.1, 0.2]])
        labels2 = np.array([1, 0, 1, 1])

        metric.update(preds2, labels2)
        prec = metric.eval()

        assert np.array_equal(prec, [0.4, 1.])

        metric.clear()
        metric.update(preds1, labels1)

        prec = metric.eval()

        assert np.array_equal(prec, [0.5, 1.0])

class TestClassRecall(unittest.TestCase):
    r"""
    Test class Recall
    """
    def test_class_recall_tensor(self):
        """
        Test class Recall
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)

        metric = Recall()
        metric.update(preds, labels)

        rec = metric.eval()

        assert np.array_equal(rec, [1., 0.5])


    def test_class_recall_tensor_onehot(self):
        """
        Test class Recall
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mindspore.float32)

        metric = Recall()
        metric.update(preds, labels)

        rec = metric.eval()

        assert np.array_equal(rec, [1., 0.5])


    def test_class_recall_np_multi(self):
        """
        Test class Recall
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])

        metric = Recall()
        metric.update(preds, labels)

        rec = metric.eval()

        assert np.array_equal(rec, [0., 1., 0., 0.])


    def test_class_recall_list(self):
        """
        Test class Recall
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]

        metric = Recall()
        metric.update(preds, labels)

        rec = metric.eval()

        assert np.array_equal(rec, [1., 0.5])


    def test_class_recall_update_clear(self):
        """
        Test class Recall
        """
        preds1 = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels1 = Tensor(np.array([1, 0, 1]), mindspore.int32)

        metric = Recall()
        metric.update(preds1, labels1)

        preds2 = np.array([[0.5, 0.05], [0.3, 0], [0.9, 0.05], [0.1, 0.2]])
        labels2 = np.array([1, 0, 1, 1])

        metric.update(preds2, labels2)

        rec = metric.eval()

        assert np.array_equal(rec, [1., 0.4])

        metric.clear()
        metric.update(preds1, labels1)

        rec = metric.eval()

        assert np.array_equal(rec, [1., 0.5])

class TestClassF1Score(unittest.TestCase):
    r"""
    Test class F1Score
    """
    def test_class_f1_score_tensor(self):
        """
        Test class F1Score
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([1, 0, 1]))

        metric = F1Score()
        metric.update(preds, labels)

        f1_s = metric.eval()

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])


    def test_class_f1_score_tensor_onehot(self):
        """
        Test class F1Score
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))

        metric = F1Score()
        metric.update(preds, labels)

        f1_s = metric.eval()

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])


    def test_class_f1_score_np_multi(self):
        """
        Test class F1Score
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])

        metric = F1Score()
        metric.update(preds, labels)

        f1_s = metric.eval()

        assert np.array_equal(f1_s, [0., 1., 0., 0.])


    def test_class_f1_score_list(self):
        """
        Test class F1Score
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([1, 0, 1]))

        metric = F1Score()
        metric.update(preds, labels)

        f1_s = metric.eval()

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])


    def test_class_f1_update_clear(self):
        """
        Test class F1Score
        """
        preds1 = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels1 = Tensor(np.array([1, 0, 1]))

        metric = F1Score()
        metric.update(preds1, labels1)

        preds2 = np.array([[0.5, 0.05], [0.3, 0], [0.9, 0.05], [0.1, 0.2]])
        labels2 = np.array([1, 0, 1, 1])

        metric.update(preds2, labels2)

        f1_s = metric.eval()

        assert np.array_equal(f1_s, [0.5714285714285714, 0.5714285714285714])

        metric.clear()
        metric.update(preds1, labels1)

        f1_s = metric.eval()

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])

class TestClassMatthewsCorrelation(unittest.TestCase):
    r"""
    Test class MatthewsCorrelation
    """
    def test_class_matthews_correlation_tensor(self):
        """
        Test class MatthewsCorrelation
        """
        preds = Tensor(np.array([[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]))
        labels = Tensor(np.array([0, 1, 0, 1, 0]))

        metric = MatthewsCorrelation()
        metric.update(preds, labels)

        m_c_c = metric.eval()

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)


    def test_class_matthews_correlation_tensor_onehot(self):
        """
        Test class MatthewsCorrelation
        """
        preds = Tensor(np.array([[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]))
        labels = Tensor(np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]))

        metric = MatthewsCorrelation()
        metric.update(preds, labels)

        m_c_c = metric.eval()

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)


    def test_class_matthews_correlation_np_zero(self):
        """
        Test class MatthewsCorrelation
        """
        preds = np.array([[-0.1, 0.12], [-0.23, 0.23], [-0.32, 0.21], [-0.13, 0.23]])
        labels = np.array([1, 0, 1, 1])

        metric = MatthewsCorrelation()
        metric.update(preds, labels)

        m_c_c = metric.eval()

        assert m_c_c == 0


    def test_class_matthews_correlation_list(self):
        """
        Test class MatthewsCorrelation
        """
        preds = [[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]
        labels = [0, 1, 0, 1, 0]

        metric = MatthewsCorrelation()
        metric.update(preds, labels)

        m_c_c = metric.eval()

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)


    def test_class_matthews_correlation_update_clear(self):
        """
        Test class MatthewsCorrelation
        """
        preds1 = [[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]
        labels1 = [0, 1, 0, 1, 0]

        metric = MatthewsCorrelation()
        metric.update(preds1, labels1)

        preds2 = [[-0.1, 0.12], [-0.23, 0.23], [-0.32, 0.21], [-0.13, 0.23]]
        labels2 = [1, 0, 1, 1]

        metric.update(preds2, labels2)
        m_c_c = metric.eval()

        assert np.allclose(m_c_c, 0.31622, 1e-5, 1e-5)

        metric.clear()
        metric.update(preds1, labels1)

        m_c_c = metric.eval()

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)

class TestClassPearsonCorrelation(unittest.TestCase):
    r"""
    Test class PearsonCorrelation
    """
    def test_class_pearson_correlation_tensor1(self):
        """
        Test class PearsonCorrelation
        """
        preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)

        metric = PearsonCorrelation()
        metric.update(preds, labels)

        p_c_c = metric.eval()

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)


    def test_class_pearson_correlation_tensor2(self):
        """
        Test class PearsonCorrelation
        """
        preds = Tensor(np.array([[0.12], [0.23], [0.21], [0.13]]), mindspore.float32)
        labels = Tensor(np.array([[1], [0], [1], [1]]), mindspore.float32)

        metric = PearsonCorrelation()
        metric.update(preds, labels)

        p_c_c = metric.eval()

        assert p_c_c == -0.689414301147012


    def test_class_pearson_correlation_np(self):
        """
        Test class PearsonCorrelation
        """
        preds = np.array(np.float32([[0.1], [1.0], [2.4], [0.9]]))
        labels = np.array(np.float32([[0.0], [1.0], [2.9], [1.0]]))

        metric = PearsonCorrelation()
        metric.update(preds, labels)

        p_c_c = metric.eval()

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)


    def test_class_pearson_correlation_list(self):
        """
        Test class PearsonCorrelation
        """
        preds = np.float32([[0.1], [1.0], [2.4], [0.9]])
        labels = np.float32([[0.0], [1.0], [2.9], [1.0]])

        metric = PearsonCorrelation()
        metric.update(preds, labels)

        p_c_c = metric.eval()

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)


    def test_class_pearson_correlation_update_clear(self):
        """
        Test class PearsonCorrelation
        """
        preds1 = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels1 = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)

        metric = PearsonCorrelation()
        metric.update(preds1, labels1)

        preds2 = Tensor(np.array([[0.12], [0.23], [0.21], [0.13]]), mindspore.float32)
        labels2 = Tensor(np.array([[1], [0], [1], [1]]), mindspore.float32)

        metric.update(preds2, labels2)
        p_c_c = metric.eval()

        assert np.allclose(p_c_c, 0.85752, 1e-5, 1e-5)

        metric.clear()
        metric.update(preds1, labels1)

        p_c_c = metric.eval()

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)

class TestClassSpearmanCorrelation(unittest.TestCase):
    r"""
    Test class SpearmanCorrelation
    """
    def test_class_spearman_correlation_tensor1(self):
        """
        Test class SpearmanCorrelation
        """
        preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)

        metric = SpearmanCorrelation()
        metric.update(preds, labels)

        s_r_c_c = metric.eval()

        assert s_r_c_c == 1.0


    def test_class_spearman_correlation_tensor2(self):
        """
        Test class SpearmanCorrelation
        """
        preds = Tensor(np.array([[0.12], [0.23], [0.21], [0.13]]), mindspore.float32)
        labels = Tensor(np.array([[1], [0], [1], [1]]), mindspore.float32)

        metric = SpearmanCorrelation()
        metric.update(preds, labels)

        s_r_c_c = metric.eval()

        assert s_r_c_c == -0.8


    def test_class_spearman_correlation_np(self):
        """
        Test class SpearmanCorrelation
        """
        preds = np.array(np.float32([[0.12], [0.23], [0.21], [0.13]]))
        labels = np.array(np.float32([[1], [0], [1], [1]]))

        metric = SpearmanCorrelation()
        metric.update(preds, labels)

        s_r_c_c = metric.eval()

        assert s_r_c_c == -0.8


    def test_class_spearman_correlation_list(self):
        """
        Test class SpearmanCorrelation
        """
        preds = np.float32([[0.12], [0.23], [0.21], [0.13]])
        labels = np.float32([[1], [0], [1], [1]])

        metric = SpearmanCorrelation()
        metric.update(preds, labels)

        s_r_c_c = metric.eval()

        assert s_r_c_c == -0.8


    def test_class_spearman_correlation_update_clear(self):
        """
        Test class SpearmanCorrelation
        """
        preds1 = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels1 = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)

        metric = SpearmanCorrelation()
        metric.update(preds1, labels1)

        preds2 = np.float32([[0.12], [0.23], [0.21], [0.13]])
        labels2 = np.float32([[1], [0], [1], [1]])

        metric.update(preds2, labels2)
        s_r_c_c = metric.eval()

        # assert np.allclose(s_r_c_c, 0.69047, 1e-5, 1e-5)

        metric.clear()
        metric.update(preds1, labels1)

        s_r_c_c = metric.eval()

        assert s_r_c_c == 1.0

class TestClassEmScore(unittest.TestCase):
    r"""
    Test class EmScore
    """
    def test_class_em_score_zero(self):
        """
        Test class EmScore
        """
        preds = "this is the best span"
        examples = ["this is a good span", "something irrelevant"]

        metric = EmScore()
        metric.update(preds, examples)

        exact_match = metric.eval()

        assert exact_match == 0.0


    def test_class_em_score_one(self):
        """
        Test class EmScore
        """
        preds = "this is the best span"
        examples = ["this is the best span", "something irrelevant"]

        metric = EmScore()
        metric.update(preds, examples)

        exact_match = metric.eval()

        assert exact_match == 1.0


    def test_class_em_score_update_clear(self):
        """
        Test class EmScore
        """
        preds1 = "this is the best span"
        examples1 = ["this is a good span", "something irrelevant"]

        metric = EmScore()
        metric.update(preds1, examples1)

        preds2 = "there is a cat"
        examples2 = ["there is a cat", "something irrelevant"]

        metric.update(preds2, examples2)
        exact_match = metric.eval()

        assert exact_match == 0.5

        metric.clear()
        metric.update(preds1, examples1)

        exact_match = metric.eval()

        assert exact_match == 0

class TestClassConfusionMatrix(unittest.TestCase):
    r"""
    Test class ConfusionMatrix
    """
    def test_class_confusion_matrix_tensor(self):
        """
        Test class ConfusionMatrix
        """
        preds = Tensor(np.array([1, 0, 1, 0]))
        labels = Tensor(np.array([1, 0, 0, 1]))

        metric = ConfusionMatrix()
        metric.update(preds, labels)

        conf_mat = metric.eval()

        assert np.array_equal(conf_mat, np.array([[1., 1.], [1., 1.]]))


    def test_class_confusion_matrix_np_classnum(self):
        """
        Test class ConfusionMatrix
        """
        preds = np.array([2, 1, 3])
        labels = np.array([2, 2, 1])

        metric = ConfusionMatrix(4)
        metric.update(preds, labels)

        conf_mat = metric.eval()

        assert np.array_equal(conf_mat, np.array([[0., 0., 0., 0.],
                                                  [0., 0., 0., 1.],
                                                  [0., 1., 1., 0.],
                                                  [0., 0., 0., 0.]]))


    def test_class_confusion_list_preds(self):
        """
        Test class ConfusionMatrix
        """
        preds = [[0.1, 0.8], [0.9, 0.3], [0.1, 1], [1, 0]]
        labels = [1, 0, 0, 1]

        metric = ConfusionMatrix()
        metric.update(preds, labels)

        conf_mat = metric.eval()

        assert np.array_equal(conf_mat, np.array([[1., 1.], [1., 1.]]))


    def test_class_confusion_update_clear(self):
        """
        Test class ConfusionMatrix
        """
        preds1 = Tensor(np.array([1, 0, 1, 0]))
        labels1 = Tensor(np.array([1, 0, 0, 1]))

        metric = ConfusionMatrix()
        metric.update(preds1, labels1)

        preds2 = Tensor(np.array([[0.1, 0.8], [0.9, 0.3], [0.1, 1], [1, 0]]))
        labels2 = Tensor(np.array([1, 0, 0, 1]))

        metric.update(preds2, labels2)
        conf_mat = metric.eval()

        assert np.array_equal(conf_mat, np.array([[2., 2.],
                                                  [2., 2.]]))

        metric.clear()
        metric.update(preds1, labels1)

        conf_mat = metric.eval()

        assert np.array_equal(conf_mat, np.array([[1., 1.], [1., 1.]]))
