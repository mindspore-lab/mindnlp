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
"""Test the Functions for Metrics"""


import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.metrics import (perplexity_fn, bleu_fn, rouge_n_fn, rouge_l_fn, distinct_fn, accuracy_fn,
                             precision_fn, recall_fn, f1_score_fn, confusion_matrix_fn,
                             matthews_correlation_fn, pearson_correlation_fn,
                             spearman_correlation_fn, em_score_fn)

class TestPerplexity(unittest.TestCase):
    r"""
    Test perplexity
    """

    def test_perplexity_tensor(self):
        """
        Test perplexity
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([1, 0, 1]))
        ppl = perplexity_fn(preds, labels, ignore_label=None)

        assert np.allclose(ppl, 2.23144, 1e-5, 1e-5)


    def test_perplexity_tensor_onehot(self):
        """
        Test perplexity
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
        ppl = perplexity_fn(preds, labels, ignore_label=None)

        assert np.allclose(ppl, 2.23144, 1e-5, 1e-5)


    def test_perplexity_np(self):
        """
        Test perplexity
        """
        preds = np.array([[0.6, 0.5, 0.1], [0.3, 0.6, 0.05], [0.1, 0.6, 0.2], [0.1, 0.2, 0.7]])
        labels = np.array([2, 1, 0, 1])
        ppl = perplexity_fn(preds, labels, ignore_label=None)

        assert np.allclose(ppl, 5.37284, 1e-5, 1e-5)


    def test_perplexity_list_multi(self):
        """
        Test perplexity
        """
        preds = [[0.6, 0.5, 0.1], [0.3, 0.6, 0.05], [0.1, 0.6, 0.2], [0.1, 0.2, 0.7]]
        labels = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 0]]
        ppl = perplexity_fn(preds, labels, ignore_label=None)

        assert np.allclose(ppl, 5.37284, 1e-5, 1e-5)

class TestBleu(unittest.TestCase):
    r"""
    Test bleu
    """

    def setUp(self):
        self.input = None


    def test_bleu_normal(self):
        """
        Test bleu
        """
        cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"],
                    ["There", "is", "a", "cat", "on", "the", "mat"]]]
        bleu_score = bleu_fn(cand, ref_list)

        assert np.allclose(bleu_score, 0.46713, 1e-5, 1e-5)


    def test_bleu_perfect(self):
        """
        Test bleu
        """
        cand = [["The", "cat", "is", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"],
                    ["There", "is", "a", "cat", "on", "the", "mat"]]]
        bleu_score = bleu_fn(cand, ref_list)

        assert bleu_score == 1.0


    def test_bleu_warning(self):
        """
        Test bleu
        """
        cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        ref_list = [[["The", "cat", "is", "on", "the", "mat"]]]
        bleu_score = bleu_fn(cand, ref_list)

        assert bleu_score == 0.0


    def test_bleu_nsize_weights(self):
        """
        Test bleu
        """
        cand = [["this", "is", "small", "cat"]]
        ref_list = [[["this", "is", "a", "small", "cat"]]]
        bleu_score = bleu_fn(cand, ref_list, 2, [0.5, 0.5])

        assert np.allclose(bleu_score, 0.63588, 1e-5, 1e-5)

class TestRougeN(unittest.TestCase):
    r"""
    Test rouge_n
    """

    def setUp(self):
        self.input = None


    def test_rougen(self):
        """
        Test rouge_n
        """
        cand_list = ["a", "cat", "is", "on", "the", "table"]
        ref_list = [["there", "is", "a", "cat", "on", "the", "table"]]
        rougen_score = rouge_n_fn(cand_list, ref_list, 2)

        assert rougen_score == 0.5


    def test_rougen_1(self):
        """
        Test rouge_n
        """
        cand_list = ["the", "cat", "was", "found", "under", "the", "bed"]
        ref_list = [["the", "cat", "was", "under", "the", "bed"]]
        rougen_score = rouge_n_fn(cand_list, ref_list)

        assert rougen_score == 1.0


    def test_rougen_2(self):
        """
        Test rouge_n
        """
        cand_list = ["the", "cat", "was", "found", "under", "the", "bed"]
        ref_list = [["the", "cat", "was", "under", "the", "bed"]]
        rougen_score = rouge_n_fn(cand_list, ref_list, 2)

        assert rougen_score == 0.8


    def test_rougen_ref(self):
        """
        Test rouge_n
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"],
                    ["There","is","a","cat","on","the","mat"]]
        rougen_score = rouge_n_fn(cand_list, ref_list, 2)

        assert np.allclose(rougen_score, 0.54545, 1e-5, 1e-5)

class TestRougeL(unittest.TestCase):
    r"""
    Test rouge_l
    """

    def setUp(self):
        self.input = None


    def test_rougel(self):
        """
        Test rouge_l
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"]]
        rougel_score = rouge_l_fn(cand_list, ref_list)

        assert np.allclose(rougel_score, 0.78005, 1e-5, 1e-5)


    def test_rougel_ref1(self):
        """
        Test rouge_l
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["There","is","a","cat","on","the","mat"]]
        rougel_score = rouge_l_fn(cand_list, ref_list)

        assert np.allclose(rougel_score, 0.57142, 1e-5, 1e-5)


    def test_rougel_ref2(self):
        """
        Test rouge_l
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"],
                    ["There","is","a","cat","on","the","mat"]]
        rougel_score = rouge_l_fn(cand_list, ref_list)

        assert np.allclose(rougel_score, 0.78005, 1e-5, 1e-5)


    def test_rougel_beta(self):
        """
        Test rouge_l
        """
        cand_list = ["The","cat","The","cat","on","the","mat"]
        ref_list = [["The","cat","is","on","the","mat"],
                    ["There","is","a","cat","on","the","mat"]]
        rougel_score = rouge_l_fn(cand_list, ref_list, 0.5)

        assert np.allclose(rougel_score, 0.73529, 1e-5, 1e-5)

class TestDistinct(unittest.TestCase):
    r"""
    Test distinct
    """

    def setUp(self):
        self.input = None


    def test_distinct(self):
        """
        Test distinct
        """
        cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        distinct_score = distinct_fn(cand_list)

        assert np.allclose(distinct_score, 0.83333, 1e-5, 1e-5)


    def test_distinct_one(self):
        """
        Test distinct
        """
        cand_list = ["The", "cat", "on", "the", "mat"]
        distinct_score = distinct_fn(cand_list)

        assert distinct_score == 1.0

class TestAccuracy(unittest.TestCase):
    r"""
    Test accuracy
    """

    def setUp(self):
        self.input = None


    def test_accuracy_tensor(self):
        """
        Test accuracy
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        acc = accuracy_fn(preds, labels)

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)


    def test_accuracy_tensor_onehot(self):
        """
        Test accuracy
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mindspore.int32)
        acc = accuracy_fn(preds, labels)

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)


    def test_accuracy_np_multi(self):
        """
        Test accuracy
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])
        acc = accuracy_fn(preds, labels)

        assert acc == 0.25


    def test_accuracy_list(self):
        """
        Test accuracy
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]
        acc = accuracy_fn(preds, labels)

        assert np.allclose(acc, 0.66666, 1e-5, 1e-5)

class TestPrecision(unittest.TestCase):
    r"""
    Test precision
    """

    def setUp(self):
        self.input = None


    def test_precision_tensor(self):
        """
        Test precision
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        prec = precision_fn(preds, labels)

        assert np.array_equal(prec, [0.5, 1.0])


    def test_precision_tensor_onehot(self):
        """
        Test precision
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mindspore.float32)
        prec = precision_fn(preds, labels)

        assert np.array_equal(prec, [0.5, 1.0])


    def test_precision_np_multi(self):
        """
        Test precision
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])
        prec = precision_fn(preds, labels)

        assert np.array_equal(prec, [0., 1., 0., 0.])


    def test_precision_list(self):
        """
        Test precision
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]
        prec = precision_fn(preds, labels)

        assert np.array_equal(prec, [0.5, 1.0])

class TestRecall(unittest.TestCase):
    r"""
    Test recall
    """

    def setUp(self):
        self.input = None


    def test_recall_tensor(self):
        """
        Test recall
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        rec = recall_fn(preds, labels)

        assert np.array_equal(rec, [1., 0.5])


    def test_recall_tensor_onehot(self):
        """
        Test recall
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]), mindspore.float32)
        rec = recall_fn(preds, labels)

        assert np.array_equal(rec, [1., 0.5])


    def test_recall_np_multi(self):
        """
        Test recall
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])
        rec = recall_fn(preds, labels)

        assert np.array_equal(rec, [0., 1., 0., 0.])


    def test_recall_list(self):
        """
        Test recall
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]
        rec = recall_fn(preds, labels)

        assert np.array_equal(rec, [1., 0.5])

class TestF1Score(unittest.TestCase):
    r"""
    Test f1_score
    """

    def setUp(self):
        self.input = None


    def test_f1_score_tensor(self):
        """
        Test f1_score
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([1, 0, 1]))
        f1_s = f1_score_fn(preds, labels)

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])


    def test_f1_score_tensor_onehot(self):
        """
        Test f1_score
        """
        preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        labels = Tensor(np.array([[0, 1], [1, 0], [0, 1]]))
        f1_s = f1_score_fn(preds, labels)

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])


    def test_f1_score_np_multi(self):
        """
        Test f1_score
        """
        preds = np.array([[0.2, 0.5, 0.1, 0.05],
                          [0.3, 0.1, 0.6, 0],
                          [0.9, 0.05, 0, 0.05],
                          [0.3, 0.1, 0.2, 0.3]])
        labels = np.array([1, 0, 2, 3])
        f1_s = f1_score_fn(preds, labels)

        assert np.array_equal(f1_s, [0., 1., 0., 0.])


    def test_f1_score_list(self):
        """
        Test f1_score
        """
        preds = [[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]
        labels = [1, 0, 1]
        f1_s = f1_score_fn(preds, labels)

        assert np.array_equal(f1_s, [0.6666666666666666, 0.6666666666666666])

class TestMatthewsCorrelation(unittest.TestCase):
    r"""
    Test matthews_correlation
    """

    def setUp(self):
        self.input = None


    def test_matthews_correlation_tensor(self):
        """
        Test matthews_correlation
        """
        preds = Tensor(np.array([[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]))
        labels = Tensor(np.array([0, 1, 0, 1, 0]))
        m_c_c = matthews_correlation_fn(preds, labels)

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)


    def test_matthews_correlation_tensor_onehot(self):
        """
        Test matthews_correlation
        """
        preds = Tensor(np.array([[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]))
        labels = Tensor(np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]]))
        m_c_c = matthews_correlation_fn(preds, labels)

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)


    def test_matthews_correlation_np_zero(self):
        """
        Test matthews_correlation
        """
        preds = np.array([[-0.1, 0.12], [-0.23, 0.23], [-0.32, 0.21], [-0.13, 0.23]])
        labels = np.array([1, 0, 1, 1])
        m_c_c = matthews_correlation_fn(preds, labels)

        assert m_c_c == 0.0


    def test_matthews_correlation_list(self):
        """
        Test matthews_correlation
        """
        preds = [[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]
        labels = [0, 1, 0, 1, 0]
        m_c_c = matthews_correlation_fn(preds, labels)

        assert np.allclose(m_c_c, 0.16666, 1e-5, 1e-5)

class TestPearsonCorrelation(unittest.TestCase):
    r"""
    Test pearson_correlation
    """

    def setUp(self):
        self.input = None


    def test_pearson_correlation_tensor1(self):
        """
        Test pearson_correlation
        """
        preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        p_c_c = pearson_correlation_fn(preds, labels)

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)


    def test_pearson_correlation_tensor2(self):
        """
        Test pearson_correlation
        """
        preds = Tensor(np.array([[0.12], [0.23], [0.21], [0.13]]), mindspore.float32)
        labels = Tensor(np.array([[1], [0], [1], [1]]), mindspore.float32)
        p_c_c = pearson_correlation_fn(preds, labels)

        assert p_c_c == -0.689414301147012


    def test_pearson_correlation_np(self):
        """
        Test pearson_correlation
        """
        preds = np.array(np.float32([[0.1], [1.0], [2.4], [0.9]]))
        labels = np.array(np.float32([[0.0], [1.0], [2.9], [1.0]]))
        p_c_c = pearson_correlation_fn(preds, labels)

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)


    def test_pearson_correlation_list(self):
        """
        Test pearson_correlation
        """
        preds = np.float32([[0.1], [1.0], [2.4], [0.9]])
        labels = np.float32([[0.0], [1.0], [2.9], [1.0]])
        p_c_c = pearson_correlation_fn(preds, labels)

        assert np.allclose(p_c_c, 0.99852, 1e-5, 1e-5)

class TestSpearmanCorrelation(unittest.TestCase):
    r"""
    Test spearman_correlation
    """

    def setUp(self):
        self.input = None


    def test_spearman_correlation_tensor1(self):
        """
        Test spearman_correlation
        """
        preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        scc = spearman_correlation_fn(preds, labels)

        assert scc == 1.0


    def test_spearman_correlation_tensor2(self):
        """
        Test spearman_correlation
        """
        preds = Tensor(np.array([[0.12], [0.23], [0.21], [0.13]]), mindspore.float32)
        labels = Tensor(np.array([[1], [0], [1], [1]]), mindspore.float32)
        scc = spearman_correlation_fn(preds, labels)

        assert scc == -0.8


    def test_spearman_correlation_np(self):
        """
        Test spearman_correlation
        """
        preds = np.array(np.float32([[0.12], [0.23], [0.21], [0.13]]))
        labels = np.array(np.float32([[1], [0], [1], [1]]))
        scc = spearman_correlation_fn(preds, labels)

        assert scc == -0.8


    def test_spearman_correlation_list(self):
        """
        Test spearman_correlation
        """
        preds = np.float32([[0.12], [0.23], [0.21], [0.13]])
        labels = np.float32([[1], [0], [1], [1]])
        scc = spearman_correlation_fn(preds, labels)

        assert scc == -0.8

class TestEmScore(unittest.TestCase):
    r"""
    Test em_score
    """

    def setUp(self):
        self.input = None


    def test_em_score_zero(self):
        """
        Test em_score
        """
        preds = "this is the best span"
        examples = ["this is a good span", "something irrelevant"]
        exact_match = em_score_fn(preds, examples)

        assert exact_match == 0.0


    def test_em_score_one(self):
        """
        Test em_score
        """
        preds = "this is the best span"
        examples = ["this is the best span", "something irrelevant"]
        exact_match = em_score_fn(preds, examples)

        assert exact_match == 1.0

class TestConfusionMatrix(unittest.TestCase):
    r"""
    Test confusion_matrix
    """

    def setUp(self):
        self.input = None


    def test_confusion_matrix_tensor(self):
        """
        Test confusion_matrix
        """
        preds = Tensor(np.array([1, 0, 1, 0]))
        labels = Tensor(np.array([1, 0, 0, 1]))
        conf_mat = confusion_matrix_fn(preds, labels)

        assert np.array_equal(conf_mat, np.array([[1., 1.], [1., 1.]]))


    def test_confusion_matrix_np_classnum(self):
        """
        Test confusion_matrix
        """
        preds = np.array([2, 1, 3])
        labels = np.array([2, 2, 1])
        conf_mat = confusion_matrix_fn(preds, labels, 4)

        assert np.array_equal(conf_mat, np.array([[0., 0., 0., 0.],
                                                  [0., 0., 0., 1.],
                                                  [0., 1., 1., 0.],
                                                  [0., 0., 0., 0.]]))


    def test_confusion_matrix_list_preds(self):
        """
        Test confusion_matrix
        """
        preds = [[0.1, 0.8], [0.9, 0.3], [0.1, 1], [1, 0]]
        labels = [1, 0, 0, 1]
        conf_mat = confusion_matrix_fn(preds, labels)

        assert np.array_equal(conf_mat, np.array([[1., 1.], [1., 1.]]))
