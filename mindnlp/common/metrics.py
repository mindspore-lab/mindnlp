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
""""Classes and functions for Metrics"""

import sys
import math
import string
from collections.abc import Iterable
from collections import Counter
import re
import numpy as np
from mindspore import Tensor
from mindnlp.abc import Metric



# Metric classes.
class Accuracy(Metric):
    r"""
    Calculate accuracy. The function is shown as follows:

    .. math::

        \text{ACC} =\frac{\text{TP} + \text{TN}}
        {\text{TP} + \text{TN} + \text{FP} + \text{FN}}

    where `ACC` is accuracy, `TP` is the number of true posistive cases, `TN` is the number
    of true negative cases, `FP` is the number of false posistive cases, `FN` is the number
    of false negative cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import nn, Tensor
        >>> from mindnlp.common.metrics import Accuracy
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.float32)
        >>> metric = Accuracy()
        >>> metric.updates(preds, labels)
        >>> acc = metric.eval()
        >>> print(acc)
        0.6666666666666666

    """
    def __init__(self, name='Accuracy'):
        super().__init__()
        self._name = name
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def clear(self):
        """Clear the internal evaluation result."""
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def updates(self, preds, labels):
        """
        Update local variables. If the index of the maximum of the predicted value matches the label,
        the predicted result is correct.

        Args:
            preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
            in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
            where :math:`N` is the number of cases and :math:`C` is the number of categories.
            labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
            that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

        Raises:
            RuntimeError: If `preds` is None or `labels` is None.
            ValueError: class numbers of last input predicted data and current predicted data not match.

        """
        if preds is None or labels is None:
            raise RuntimeError("To calculate accuracy, it needs at least 2 inputs (`preds` and `labels`)")
        y_pred = _convert_data_type(preds)
        y_true = _convert_data_type(labels)

        if y_pred.ndim == y_true.ndim and (_check_onehot_data(y_true) or y_true[0].shape == (1,)):
            y_true = y_true.argmax(axis=1)
        _check_shape(y_pred, y_true)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError(f'For `Accuracy.updates`, class number not match, last input predicted data'
                             f'contain {self._class_num} classes, but current predicted data contain '
                             f'{y_pred.shape[1]} classes, please check your predicted value(`preds`).')

        indices = y_pred.argmax(axis=1)
        res = (np.equal(indices, y_true) * 1).reshape(-1)

        self._correct_num += res.sum()
        self._total_num += res.shape[0]

    def eval(self):
        """
        Compute and return the accuracy.

        Returns:
            - **acc** (float) - The computed result.

        Raises:
            RuntimeError: If the number of samples is 0.
        """
        if self._total_num == 0:
            raise RuntimeError(f'`Accuracy` can not be calculated, because the number of samples is {0}, '
                               f'please check whether your inputs(`preds`, `labels`) are empty, or has called'
                               f'update method before calling eval method.')
        acc = self._correct_num / self._total_num
        return acc

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name

class F1Score(Metric):
    r"""
    Calculate F1 score. Fbeta score is a weighted mean of precision and recall, and F1 score is
    a special case of Fbeta when beta is 1. The function is shown as follows:

    .. math::

        F_1=\frac{2\cdot TP}{2\cdot TP + FN + FP}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases,
    `FP` is the number of false positive cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import F1Score
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> metric = F1Score()
        >>> metric.updates(preds, labels)
        >>> f1_s = metric.eval()
        >>> print(f1_s)
        0.6666666666666666
    """
    def __init__(self, name='F1Score'):
        super().__init__()
        self._name = name
        self.epsilon = sys.float_info.min
        self._true_positives = 0
        self._actual_positives = 0
        self._positives = 0
        self._class_num = 0

    def clear(self):
        """Clear the internal evaluation result."""
        self._true_positives = 0
        self._actual_positives = 0
        self._positives = 0
        self._class_num = 0

    def updates(self, preds, labels):
        """
        Update local variables. If the index of the maximum of the predicted value matches the label,
        the predicted result is correct.

        Args:
            preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
            in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
            where :math:`N` is the number of cases and :math:`C` is the number of categories.
            labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
            that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

        Raises:
            RuntimeError: If `preds` is None or `labels` is None.
            ValueError: class numbers of last input predicted data and current predicted data not match.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if preds is None or labels is None:
            raise RuntimeError("To calculate F1 score, it needs at least 2 inputs (`preds` and `labels`)")

        y_pred = _convert_data_type(preds)
        y_true = _convert_data_type(labels)
        if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
            y_true = y_true.argmax(axis=1)
        _check_shape(y_pred, y_true)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError(f'For `F1Score.update`, class number not match, last input predicted data contain '
                             f'{self._class_num} classes, but current predicted data contain {y_pred.shape[1]} '
                             f'classes, please check your predicted value(`preds`).')
        class_num = self._class_num

        if y_true.max() + 1 > class_num:
            raise ValueError(f'For `F1Score.update`, `preds` and `labels` should contain same classes, but got '
                             f'`preds` contains {class_num} classes and true value contains {y_true.max() + 1}')
        y_true = np.eye(class_num)[y_true.reshape(-1)]
        indices = y_pred.argmax(axis=1).reshape(-1)
        y_pred = np.eye(class_num)[indices]

        positives = y_pred.sum(axis=0)
        actual_positives = y_true.sum(axis=0)
        true_positives = (y_true * y_pred).sum(axis=0)

        self._true_positives += true_positives
        self._positives += positives
        self._actual_positives += actual_positives

    def eval(self):
        """
        Compute and return the F1 score.

        Returns:
            - **f1_s** (float) - The computed result.

        Raises:
            RuntimeError: If the number of samples is 0.
        """
        f1_s = (2 * self._true_positives / (self._actual_positives + self._positives + self.epsilon)).item(0)
        return f1_s

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name

class BleuScore(Metric):
    r"""
    Calculate BLEU. BLEU (bilingual evaluation understudy) is a metric for evaluating the quality
    of text translated by machine. It uses a modified form of precision to compare a candidate translation
    against multiple reference translations. The function is shown as follows:

    .. math::

        BP & =
        \begin{cases}
        1,  & \text{if }c>r \\
        e_{1-r/c}, & \text{if }c\leq r
        \end{cases}

        BLEU & = BP\exp(\sum_{n=1}^N w_{n} \log{p_{n}})

    where `c` is the length of candidate sentence, and `r` is the length of reference sentence.

    Args:
        name (str): Name of the metric.
        n_size (int): N_gram value ranges from 1 to 4. Default: 4.
        weights (list): Weights of precision of each gram. Defaults to None.

    Raises:
        ValueError: If the value range of `n_size` is not from 1 to 4.
        ValueError: If the lengths of `weights` is not equal to `n_size`.

    Example:
        >>> from mindnlp.common.metrics import BleuScore
        >>> cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        >>> ref_list = [[["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]]
        >>> metric = BleuScore()
        >>> metric.updates(cand, ref_list)
        >>> bleu_score = metric.eval()
        >>> print(bleu_score)
        0.46713797772820015

    """
    def __init__(self, name='BleuScore', n_size=4, weights=None):
        super().__init__()
        self._name = name
        self.n_size = _check_value_type("n_size", n_size, [int])
        if self.n_size > 4 or self.n_size < 1:
            raise ValueError(f'For `BleuScore`, `n_size` should range from 1 to 4, but got {n_size}')

        if weights is None:
            self.weights = [0.25] * self.n_size
        else:
            self.weights = weights

        if self.n_size != len(self.weights):
            raise ValueError("For `BleuScore`, the length of `weights` should be equal to `n_size`")

        self.numerator = np.zeros(self.n_size)
        self.denominator = np.zeros(self.n_size)
        self.precision_scores = np.zeros(self.n_size)
        self.bp_c = 0.0
        self.bp_r = 0.0
        self.cand_len = 0
        self.ref_len = 0

    def clear(self):
        """Clear the internal evaluation result."""
        self.numerator = np.zeros(self.n_size)
        self.denominator = np.zeros(self.n_size)
        self.precision_scores = np.zeros(self.n_size)
        self.bp_c = 0.0
        self.bp_r = 0.0
        self.cand_len = 0
        self.ref_len = 0

    def updates(self, preds, labels):
        """
        Update local variables.

        Args:
            preds (list): A list of tokenized candidate sentences.
            labels (list): A list of lists of tokenized ground truth sentences.

        Raises:
            RuntimeError: If `preds` is None or `labels` is None.
            ValueError: If the lengths of `preds` and `labels` are not equal.

        """
        if preds is None or labels is None:
            raise RuntimeError("For `BleuScore.update`, it needs at least 2 inputs (`preds` and `labels`)")
        if len(preds) != len(labels):
            raise ValueError(f'For `BleuScore.update`, `preds` and `labels` should be equal in length, but'
                             f'got {len(preds)}, {len(labels)}')

        for (candidate, references) in zip(preds, labels):
            self.bp_c += len(candidate)
            ref_len_list = [len(ref) for ref in references]
            ref_len_diff = [abs(len(candidate) - x) for x in ref_len_list]
            self.bp_r += ref_len_list[ref_len_diff.index(min(ref_len_diff))]
            candidate_counter = _count_ngram(candidate, self.n_size)
            reference_counter = Counter()

            for ref in references:
                reference_counter |= _count_ngram(ref, self.n_size)

            ngram_counter_clip = candidate_counter & reference_counter

            for counter_clip in ngram_counter_clip:
                self.numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

            for counter in candidate_counter:
                self.denominator[len(counter) - 1] += candidate_counter[counter]

        self.cand_len = np.array(self.bp_c)
        self.ref_len = np.array(self.bp_r)

    def eval(self):
        """
        Compute and return the BLEU score.

        Returns:
            - **bleu_score** (float) - The computed result.

        """
        if min(self.numerator) == 0.0:
            return np.array(0.0)

        precision_scores = self.numerator / self.denominator

        log_precision_scores = self.weights * np.log(precision_scores)
        geometric_mean = np.exp(np.sum(log_precision_scores))
        brevity_penalty = np.array(1.0) if self.bp_c > self.bp_r else np.exp(1 - (self.ref_len / self.cand_len))
        bleu_score = brevity_penalty * geometric_mean

        return bleu_score

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name


# Metric functions.
def perplexity(preds, labels, ignore_label=None):
    r"""
    Calculate perplexity. Perplexity is a measure of how well a probabilibity model predicts a
    sample. A low perplexity indicates the model is good at predicting the sample.
    The function is shown as follows:

    .. math::

        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    Where :math:`w` represents words in corpus.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.
        ignore_label (Union[int, None]): Index of an invalid label to be ignored when counting.
        If set to `None`, it means there's no invalid label. Default: None.

    Returns:
        - **ppl** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        RuntimeError: If `preds` and `labels` have different lengths.
        RuntimeError: If `pred` and `label` have different shapes.
        RuntimeError: If the sample size is 0.

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import perplexity
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> ppl = perplexity(preds, labels, ignore_label=None)
        >>> print(ppl)
        2.2314431850023855

    """
    if ignore_label is not None:
        ignore_label = _check_value_type("ignore_label", ignore_label, [int])

    if preds is None or labels is None:
        raise RuntimeError("To calculate perplexity, it needs at least 2 inputs (`preds` and `labels`)")

    y_pred = [_convert_data_type(preds)]
    y_true = [_convert_data_type(labels)]

    if len(y_pred) != len(y_true):
        raise RuntimeError(f'`preds` and `labels` should have the same length, but got `preds` length'
                           f'{len(y_pred)}, `labels` length {len(y_true)})')

    sum_cross_entropy = 0.0
    sum_word_num = 0

    cross_entropy = 0.
    word_num = 0
    for label, pred in zip(y_true, y_pred):
        if label.size != pred.size / pred.shape[-1]:
            raise RuntimeError(f'`preds` and `labels` should have the same shape, but got `preds` shape '
                               f'\'{pred.shape}\', label shape \'{label.shape}\'.')
        label = label.reshape((label.size,))
        label_expand = label.astype(int)
        label_expand = np.expand_dims(label_expand, axis=1)
        first_indices = np.arange(label_expand.shape[0])[:, None]
        pred = np.squeeze(pred[first_indices, label_expand])
        if ignore_label is not None:
            ignore = (label == ignore_label).astype(pred.dtype)
            word_num -= np.sum(ignore)
            pred = pred * (1 - ignore) + ignore
        cross_entropy -= np.sum(np.log(np.maximum(1e-10, pred)))
        word_num += pred.size
    sum_cross_entropy += cross_entropy
    sum_word_num += word_num

    if sum_word_num == 0:
        raise RuntimeError(f'Perplexity can not be calculated, because the number of samples is {0}')

    ppl = math.exp(sum_cross_entropy / sum_word_num)

    return ppl

def bleu(cand, ref_list, n_size=4, weights=None):
    r"""
    Calculate BLEU. BLEU (bilingual evaluation understudy) is a metric for evaluating the quality
    of text translated by machine. It uses a modified form of precision to compare a candidate translation
    against multiple reference translations. The function is shown as follows:

    .. math::

        BP & =
        \begin{cases}
        1,  & \text{if }c>r \\
        e_{1-r/c}, & \text{if }c\leq r
        \end{cases}

        BLEU & = BP\exp(\sum_{n=1}^N w_{n} \log{p_{n}})

    where `c` is the length of candidate sentence, and `r` is the length of reference sentence.

    Args:
        cand (list): A list of tokenized candidate sentences.
        ref_list (list): A list of lists of tokenized ground truth sentences.
        n_size (int): N_gram value ranges from 1 to 4. Default: 4.
        weights (list): Weights of precision of each gram. Defaults to None.

    Returns:
        - **bleu_score** (float) - The computed result.

    Raises:
        ValueError: If the value range of `n_size` is not from 1 to 4.
        RuntimeError: If `cand` is None or `ref_list` is None.
        ValueError: If the lengths of `cand` and `ref_list` are not equal.
        ValueError: If the lengths of `weights` is not equal to `n_size`.

    Example:
        >>> from mindnlp.common.metrics import bleu
        >>> cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        >>> ref_list = [[["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]]
        >>> bleu_score = bleu(cand, ref_list)
        >>> print(bleu_score)
        0.46713797772820015

    """

    n_size = _check_value_type("n_size", n_size, [int])
    if n_size > 4 or n_size < 1:
        raise ValueError(f'`n_size` should range from 1 to 4, but got {n_size}')

    if cand is None or ref_list is None:
        raise RuntimeError("To calculate BLEU, it needs at least 2 inputs (`cand` and `ref_list`)")
    if len(cand) != len(ref_list):
        raise ValueError(f'`cand` and `ref_list` should be equal in length, but got {len(cand)}'
                         f', {len(ref_list)}')

    numerator = np.zeros(n_size)
    denominator = np.zeros(n_size)
    precision_scores = np.zeros(n_size)
    bp_c = 0.0
    bp_r = 0.0
    cand_len = 0
    ref_len = 0

    for (candidate, references) in zip(cand, ref_list):
        bp_c += len(candidate)
        ref_len_list = [len(ref) for ref in references]
        ref_len_diff = [abs(len(candidate) - x) for x in ref_len_list]
        bp_r += ref_len_list[ref_len_diff.index(min(ref_len_diff))]
        candidate_counter = _count_ngram(candidate, n_size)
        reference_counter = Counter()

        for ref in references:
            reference_counter |= _count_ngram(ref, n_size)

        ngram_counter_clip = candidate_counter & reference_counter

        for counter_clip in ngram_counter_clip:
            numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

        for counter in candidate_counter:
            denominator[len(counter) - 1] += candidate_counter[counter]

    cand_len = np.array(bp_c)
    ref_len = np.array(bp_r)

    if min(numerator) == 0.0:
        return np.array(0.0)

    precision_scores = numerator / denominator

    if weights is None:
        weights = [0.25] * n_size

    if n_size != len(weights):
        raise ValueError("The length of `weights` should be equal to `n_size`")

    log_precision_scores = weights * np.log(precision_scores)
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.array(1.0) if bp_c > bp_r else np.exp(1 - (ref_len / cand_len))
    bleu_score = brevity_penalty * geometric_mean

    return bleu_score

def rouge_n(cand_list, ref_list, n_size=1):
    r"""
    Calculate ROUGE-N. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used
    for evaluating automatic summarization and machine translation models. ROUGE-N refers to the overlap
    of n-grams between candidates and reference summaries. The function is shown as follows:

    .. math::

        ROUGE $-N=\frac{\sum_{S \epsilon\{\text { RefSummaries }\}} \sum_{n-\text { grameS }}
        \text { Count }_{\text {match }}(n-\text { gram })} {\sum_{S \epsilon\
        {\text { RRfSummaries }\}} \sum_{n-\text { grameS }} \operatorname{Count}(n-\text { gram })}$

    Args:
        cand_list (list): A list of tokenized candidate sentences.
        ref_list (list): A list of lists of tokenized ground truth sentences.
        n_size (int): N_gram value. Default: 1.

    Returns:
        - **rougen_score** (float) - The computed result.

    Raises:
        RuntimeError: If `cand_list` is None or `ref_list` is None.
        RuntimeError: If the reference size is 0.

    Example:
        >>> from mindnlp.common.metrics import rouge_n
        >>> cand_list = [["a", "cat", "is", "on", "the", "table"]]
        >>> ref_list = [["there", "is", "a", "cat", "on", "the", "table"]]
        >>> rougen_score = rouge_n(cand_list, ref_list)
        >>> print(rougen_score)
        0.8571428571428571

    """
    def _get_ngrams(words, n_size=1):
        """Calculates word n-grams for multiple sentences.
        """
        ngram_set = set()
        max_start = len(words) - n_size
        for i in range(max_start + 1):
            print(tuple(words[i:i + n_size]))
            ngram_set.add(tuple(words[i:i + n_size]))
        return ngram_set

    if cand_list is None or ref_list is None:
        raise RuntimeError("To calculate ROUGE-N, it needs at least 2 inputs (`cand_list` and `ref_list`)")

    overlap_count = 0
    ref_count = 0
    for candidate, references in zip(cand_list, ref_list):
        cand_ngrams = _get_ngrams(candidate, n_size)
        ref_ngrams = _get_ngrams(references, n_size)
        ref_count += len(ref_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlap_ngrams = cand_ngrams.intersection(ref_ngrams)
        overlap_count += len(overlap_ngrams)

    if ref_count == 0:
        RuntimeError(f'ROUGE-N can not be calculated, because the number of references is {0}')

    rougen_score = overlap_count / ref_count

    return rougen_score

def rouge_l(cand_list, ref_list, beta=1.2):
    r"""
    Calculate ROUGE-L. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used
    for evaluating automatic summarization and machine translation models. ROUGE-L is calculated based on
    Longest Common Subsequence (LCS). The function is shown as follows:

    .. math::

        R_{l c s}=\frac{L C S(X, Y)}{m}

        p_{l c s}=\frac{L C S(X, Y)}{n}

        F_{l c s}=\frac{\left(1+\beta^{2}\right) R_{l c s} P_{l c s}}{R_{l c s}+\beta^{2} P_{l c s}}

    where `X` is the candidate sentence, `Y` is the reference sentence. `m` and `n` represent
    the length of `X` and `Y` respectively. `LCS` means the longest common subsequence.

    Args:
        cand_list (list): A list of tokenized candidate sentence.
        ref_list (list of list): A list of lists of tokenized ground truth sentences.
        beta (float): A hyperparameter to decide the weight of recall. Defaults: 1.2.

    Returns:
        - **rougel_score** (numpy.float32) - The computed result.

    Raises:
        RuntimeError: If `cand_list` is None or `ref_list` is None.

    Example:
        >>> from mindnlp.common.metrics import rouge_l
        >>> cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        >>> ref_list = [["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]
        >>> rougel_score = rouge_l(cand_list, ref_list)
        >>> print(rougel_score)
        0.7800511508951408

    """
    def _lcs(strg, sub):
        """
        Calculate the length of longest common subsequence of strg and sub.

        Args:
            strg (list): The string to be calculated, usually longer the sub string.
            sub (list): The sub string to be calculated.

        Returns:
            - **length** (numpy.float32) - The length of the longest common subsequence
            of string and sub.
        """
        if len(strg) < len(sub):
            sub, strg = strg, sub
        lengths = np.zeros((len(strg) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(strg) + 1):
                if strg[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(strg)][len(sub)]

    if cand_list is None or ref_list is None:
        raise RuntimeError("To calculate ROUGE-L, it needs at least 2 inputs (`cand_list` and `ref_list`)")

    inst_scores = []

    precs, recalls = [], []
    for ref in ref_list:
        basic_lcs = _lcs(cand_list, ref)
        prec = basic_lcs / len(cand_list) if cand_list is not None else 0.
        rec = basic_lcs / len(ref) if ref is not None else 0.
        precs.append(prec)
        recalls.append(rec)

    prec_max = max(precs)
    rec_max = max(recalls)

    if prec_max != 0 and rec_max != 0:
        score = ((1 + beta**2) * prec_max * rec_max) / \
                float(rec_max + beta**2 * prec_max)
    else:
        score = 0.0
    inst_scores.append(score)

    rougel_score = 1. * sum(inst_scores) / len(inst_scores)

    return rougel_score

def distinct(cand_list, n_size=2):
    """
    Calculate distinct-n. Distinct-N is a metric that measures the diversity of a sentence.
    It focuses on the number of distinct n-gram of a sentence. The larger the number of
    distinct n-grams, the higher the diversity of the text. The function is shown as follows:

    Args:
        cand_list (list): A list of tokenized candidate sentence.
        n_size (int): N_gram value. Defaults: 2.

    Returns:
        - **distinct_score** (float) - The computed result.

    Example:
        >>> from mindnlp.common.metrics import distinct
        >>> cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        >>> distinct_score = distinct(cand_list)
        >>> print(distinct_score)
        0.8333333333333334

    """
    diff_ngram = set()
    count = 0.0

    for i in range(0, len(cand_list) - n_size + 1):
        ngram = ' '.join(cand_list[i:(i + n_size)])
        count += 1
        diff_ngram.add(ngram)

    distinct_score = len(diff_ngram) / count
    return distinct_score

def accuracy(predictions, labels):
    r"""
    Calculate accuracy. The function is shown as follows:

    .. math::

        \text{ACC} =\frac{\text{TP} + \text{TN}}
        {\text{TP} + \text{TN} + \text{FP} + \text{FN}}

    where `ACC` is accuracy, `TP` is the number of true posistive cases, `TN` is the number
    of true negative cases, `FP` is the number of false posistive cases, `FN` is the number
    of false negative cases.

    Args:
        predictions (Union[Tensor, list, numpy.ndarray]): Predicted value. `predictions` is
        a list of floating numbers in range :math:`[0, 1]` and the shape of `predictions` is
        :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number of cases
        and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in
        one-hot format that shape is :math:`(N, C)`, or can be transformed to one-hot format
        that shape is :math:`(N,)`.

    Returns:
        - **acc** (float) - The computed result.

    Raises:
        RuntimeError: If `predictions` is None or `labels` is None.
        RuntimeError: If the number of samples is 0.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import accuracy
        >>> preds = [[0.1, 0.9], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
        >>> labels = [1, 0, 1, 1]
        >>> acc = accuracy(preds, labels)
        >>> print(acc)
        0.5

    """
    if predictions is None or labels is None:
        raise RuntimeError("To calculate accuracy, it needs at least 2 inputs (`predictions` and `labels`)")

    correct_num = 0
    total_num = 0

    y_pred = _convert_data_type(predictions)
    y_true = _convert_data_type(labels)
    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    indices = y_pred.argmax(axis=1)
    result = (np.equal(indices, y_true) * 1).reshape(-1)

    correct_num += result.sum()
    total_num += result.shape[0]

    if total_num == 0:
        raise RuntimeError(f'Accuracy can not be calculated, because the number of samples is {0}, '
                           f'please check whether your inputs(predicted value, true value) are empty.')
    acc = correct_num / total_num
    return acc

def precision(preds, labels):
    r"""
    Calculate precision. Precision (also known as positive predictive value) is the actual
    positive proportion in the predicted positive sample. It can only be used to evaluate
    the precision score of binary tasks. The function is shown as follows:

    .. math::

        \text{Precision} =\frac{\text{TP}} {\text{TP} + \text{FP}}

    where `TP` is the number of true posistive cases, `FP` is the number of false posistive cases.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

    Returns:
        - **prec** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import precision
        >>> preds = [[0.1, 0.9], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
        >>> labels = [1, 0, 1, 1]
        >>> prec = precision(preds, labels)
        >>> print(prec)
        0.5

    """
    if preds is None or labels is None:
        raise RuntimeError("To calculate precision, it needs at least 2 inputs (`preds` and `labels`)")

    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)
    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` should have the same classes number as `labels`, but got `preds`'
                         f'classes {class_num}, true value classes {y_true.max() + 1}')
    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    positives = y_pred.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    prec = (true_positives / (positives + epsilon)).item(0)
    return prec

def recall(preds, labels):
    r"""
    Calculate recall. Recall is also referred to as the true positive rate or sensitivity.
    The function is shown as follows:

    .. math::

        \text{Recall} =\frac{\text{TP}} {\text{TP} + \text{FN}}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

    Returns:
        - **rec** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import recall
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> rec = recall(preds, labels)
        >>> print(rec)
        0.5

    """
    if preds is None or labels is None:
        raise RuntimeError("To calculate recall, it needs at least 2 inputs (`preds` and `labels`)")

    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)
    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` should have the same classes number as `labels`, but got `preds`'
                         f' classes {class_num}, true value classes {y_true.max() + 1}.')
    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    actual_positives = y_true.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    rec = (true_positives / (actual_positives + epsilon)).item(1)
    return rec

def f1_score(preds, labels):
    r"""
    Calculate F1 score. Fbeta score is a weighted mean of precision and recall, and F1 score is
    a special case of Fbeta when beta is 1. The function is shown as follows:

    .. math::

        F_1=\frac{2\cdot TP}{2\cdot TP + FN + FP}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases,
    `FP` is the number of false positive cases.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

    Returns:
        - **f1_s** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import f1_score
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> f1_s = f1_score(preds, labels)
        >>> print(f1_s)
        0.6666666666666666

    """
    if preds is None or labels is None:
        raise RuntimeError("To calculate F1 score, it needs at least 2 inputs (`preds` and `labels`)")

    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)
    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` and `labels` should contain same classes, but got `preds` contains'
                         f' {class_num} classes and true value contains {y_true.max() + 1}')
    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    positives = y_pred.sum(axis=0)
    actual_positives = y_true.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    f1_s = (2 * true_positives / (actual_positives + positives + epsilon)).item(0)
    return f1_s

def confusion_matrix(preds, labels, class_num=2):
    r"""
    Calculate confusion matrix. Confusion matrix is commonly used to evaluate the performance
    of classification models, including binary classification and multiple classification.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.
        class_num (int): Number of classes in the dataset. Default: 2.

    Returns:
        - **conf_mat** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import confusion_matrix
        >>> preds = Tensor(np.array([1, 0, 1, 0]))
        >>> labels = Tensor(np.array([1, 0, 0, 1]))
        >>> conf_mat = confusion_matrix(preds, labels)
        >>> print(conf_mat)
        [[1. 1.]
         [1. 1.]]

    """
    if preds is None or labels is None:
        raise RuntimeError("To calculate confusion matrix, it needs at least 2 inputs (`preds` and `labels`)")

    class_num = _check_value_type("class_num", class_num, [int])

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim not in (labels.ndim, labels.ndim + 1):
        raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension of preds`'
                         f" equals the dimension of `labels` add 1, but got predicted value ndim: "
                         f'{preds.ndim}, true value ndim: {labels.ndim}.')

    if preds.ndim == labels.ndim + 1:
        preds = np.argmax(preds, axis=1)

    trans = (labels.reshape(-1) * class_num + preds.reshape(-1)).astype(int)
    bincount = np.bincount(trans, minlength=class_num ** 2)
    conf_mat = bincount.reshape(class_num, class_num)

    conf_mat = conf_mat.astype(float)

    return conf_mat

def mcc(preds, labels):
    r"""
    calculate Matthews correlation coefficient (MCC). MCC is in essence a correlation coefficient between
    the observed and predicted binary classifications; it returns a value between −1 and +1. A coefficient
    of +1 represents a perfect prediction, 0 no better than random prediction and −1 indicates total disagreement
    between prediction and observation. The function is shown as follows:

    .. math::

        MCC=\frac{TP \times TN-FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    where `TP` is the number of true posistive cases, `TN` is the number of true negative cases, `FN` is the number
    of false negative cases, `FP` is the number of false positive cases.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

    Returns:
        - **m_c_c** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import mcc
        >>> preds = [[0.1, 0.9], [-0.5, 0.5], [0.1, 0.4], [0.1, 0.3]]
        >>> labels = [[1], [0], [1], [1]]
        >>> m_c_c = mcc(x, y)
        >>> print(m_c_c)
        0.0

    """
    if preds is None or labels is None:
        raise RuntimeError('To calculate Matthews correlation coefficient (MCC), it needs at least 2 inputs '
                           '(`preds` and `labels`)')

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim not in (labels.ndim, labels.ndim + 1):
        raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension of preds`'
                         f" equals the dimension of `labels` add 1, but got predicted value ndim: "
                         f'{preds.ndim}, true value ndim: {labels.ndim}.')

    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0

    preds = np.argmax(preds, axis=1)
    labels = labels.reshape(-1, 1)

    sample_num = labels.shape[0]
    for i in range(sample_num):
        pred = preds[i]
        label = labels[i]
        if pred == 1:
            if pred == label:
                t_p += 1
            else:
                f_p += 1
        else:
            if pred == label:
                t_n += 1
            else:
                f_n += 1

    if t_p == 0 or f_p == 0 or t_n == 0 or f_n == 0:
        m_c_c = 0.0
    else:
        m_c_c = (t_p * t_n - f_p * f_n) / math.sqrt(
            (t_p + f_p) * (t_p + f_n) *
            (t_n + f_p) * (t_n + f_n))
    return m_c_c

def pearson(preds, labels):
    r"""
    calculate Pearson correlation coefficient (PCC). PCC is a measure of linear correlation
    between two sets of data. It is the ratio between the covariance of two variables and
    the product of their standard deviations; thus, it is essentially a normalized measurement
    of the covariance, such that the result always has a value between −1 and 1.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of
        floating numbers in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)`
        in most cases (not strictly), where :math:`N` is the number of cases and :math:`C`
        is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be
        in one-hot format that shape is :math:`(N, C)`, or can be transformed to one-hot format
        that shape is :math:`(N,)`.

    Returns:
        - **pcc** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import pearson
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> pcc = pearson(preds, labels)
        >>> print(pcc)
        0.9985229081857804

    """
    def _pearson(y_pred, y_true):
        n_pred = len(y_pred)
        # simple sums
        sum1 = sum(float(y_pred[i]) for i in range(n_pred))
        sum2 = sum(float(y_true[i]) for i in range(n_pred))
        # sum up the squares
        sum1_pow = sum(pow(v, 2.0) for v in y_pred)
        sum2_pow = sum(pow(v, 2.0) for v in y_true)
        # sum up the products
        p_sum = sum(y_pred[i] * y_true[i] for i in range(n_pred))

        numerator = p_sum - (sum1 * sum2 / n_pred)
        denominator = math.sqrt(
            (sum1_pow - pow(sum1, 2) / n_pred) * (sum2_pow - pow(sum2, 2) / n_pred))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    if preds is None or labels is None:
        raise RuntimeError('To calculate Pearson correlation coefficient (PCC), it needs at least 2 inputs '
                           '(`preds` and `labels`)')

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim not in (labels.ndim, labels.ndim + 1):
        raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension of preds`'
                         f" equals the dimension of `labels` add 1, but got predicted value ndim: "
                         f'{preds.ndim}, true value ndim: {labels.ndim}.')

    preds = np.squeeze(preds.reshape(-1, 1)).tolist()
    labels = np.squeeze(labels.reshape(-1, 1)).tolist()

    pcc = _pearson(preds, labels)
    return pcc

def spearman(preds, labels):
    r"""
    calculate Spearman's rank correlation coefficient. It is a nonparametric measure of rank correlation
    (statistical dependence between the rankings of two variables). It assesses how well the relationship
    between two variables can be described using a monotonic function. If there are no repeated data values,
    a perfect Spearman correlation of +1 or −1 occurs when each of the variables is a perfect monotone function
    of the other.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of floating numbers
        in range :math:`[0, 1]` and the shape of `preds` is :math:`(N, C)` in most cases (not strictly),
        where :math:`N` is the number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in one-hot format
        that shape is :math:`(N, C)`, or can be transformed to one-hot format that shape is :math:`(N,)`.

    Returns:
        - **scc** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import spearman
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> scc = spearman(preds, labels)
        >>> print(scc)
        1.0

    """
    def _get_rank(raw_list):
        raw_x = np.array(raw_list)
        rank_x = np.empty(raw_x.shape, dtype=int)
        sort_x = np.argsort(-raw_x)
        for i, k in enumerate(sort_x):
            rank_x[k] = i + 1
        return rank_x

    def _spearman(y_pred, y_true):
        preds_rank = _get_rank(y_pred)
        labels_rank = _get_rank(y_true)

        total = 0
        n_pred = len(y_pred)
        for i in range(n_pred):
            total += pow((preds_rank[i] - labels_rank[i]), 2)
        res = 1 - float(6 * total) / (n_pred * (pow(n_pred, 2) - 1))
        return res

    if preds is None or labels is None:
        raise RuntimeError('To calculate Spearman\'s rank correlation coefficient, it needs at least '
                           '2 inputs (`preds` and `labels`)')

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim not in (labels.ndim, labels.ndim + 1):
        raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension of preds`'
                         f" equals the dimension of `labels` add 1, but got predicted value ndim: "
                         f'{preds.ndim}, true value ndim: {labels.ndim}.')

    preds = np.squeeze(preds.reshape(-1, 1)).tolist()
    labels = np.squeeze(labels.reshape(-1, 1)).tolist()

    scc = _spearman(preds, labels)
    return scc

def em_score(preds, examples):
    r"""
    calculate exact match (em) score. This metric measures the percentage of predictions
    that match any one of the ground truth answers exactly.

    Args:
        preds (Union[str, list]): Predicted value.
        examples (Union[list, list of list]): Ground truth value.

    Returns:
        - **em** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import em_score
        >>> preds = "this is the best span"
        >>> examples = ["this is a good span", "something irrelevant"]
        >>> exact_match = em_score(preds, examples)
        >>> print(em)
        0.0

    """
    def _normalize_answer(txt):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(txt))))

    def _compute_exact(y_pred, y_true):
        return int(_normalize_answer(y_pred) == _normalize_answer(y_true))

    def _metric_max_over_ground_truths(metric_fn, pred, example):
        scores_for_ground_truths = []
        for y_eg in example:
            score = metric_fn(pred, y_eg)
            scores_for_ground_truths.append(score)
        return round(max(scores_for_ground_truths), 2)

    if not isinstance(preds, list):
        preds = [preds]
        examples = [examples]

    assert len(preds) == len(examples)

    count = len(preds)
    exact_match = 0

    for pred, example in zip(preds, examples):
        exact_match += _metric_max_over_ground_truths(
            _compute_exact, pred, example
        )

    total_em = int(exact_match)

    exact_match = total_em / count if count > 0 else 0
    return exact_match

def _check_value_type(arg_name, arg_value, valid_types):
    """
    Check whether the data type is valid

    Args:
        arg_name (str): Name of the argument validated.
        arg_value (Object): Value of the argument validated.
        valid_types (list): Valid data types of the argument.

    Returns:
        - **arg_value** (Object) - Value of the argument validated.

    Raises:
        TypeError: If the data type of the argument is not valid.

    """
    valid_types = valid_types if isinstance(valid_types, Iterable) else (valid_types,)
    num_types = len(valid_types)
    if isinstance(arg_value, bool) and bool not in tuple(valid_types):
        raise TypeError(f'Type of \'{arg_name}\' should be {"one of " if num_types > 1 else ""}'
                        f'\'{valid_types if num_types > 1 else str(valid_types[0])}\', '
                        f'but got \'{arg_value}\' with type \'{type(arg_value).__name__}\'.')
    if not isinstance(arg_value, tuple(valid_types)):
        raise TypeError(f'Type of \'{arg_name}\' should be {"one of " if num_types > 1 else ""}'
                        f'\'{valid_types if num_types > 1 else str(valid_types[0])}\', '
                        f'but got \'{arg_value}\' with type \'{type(arg_value).__name__}\'.')
    return arg_value

def _check_onehot_data(data):
    """
    Whether input data is one-hot encoding.

    Args:
        data (numpy.array): Input data.

    Returns:
        - **ans** (bool) - Rreturn true, if input data is one-hot encoding.
    """
    ans = False
    if data.ndim > 1 and np.equal(data ** 2, data).all():
        shp = (data.shape[0],) + data.shape[2:]
        if np.equal(np.ones(shp), data.sum(axis=1)).all():
            ans = True
            return ans
    return ans

def _convert_data_type(data):
    """
    Convert data type to numpy array.

    Args:
        data (Object): Input data.

    Returns:
        - **data** (numpy.ndarray) - Data with `np.ndarray` type.

    Raises:
        TypeError: If the data type is not a tensor, list or numpy.ndarray.

    """
    if isinstance(data, Tensor):
        data = data.asnumpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f'Input data type must be tensor, list or '
                        f'numpy.ndarray, but got {type(data)}.')
    return data

def _count_ngram(input_list, n_gram):
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(len(input_list) - i + 1):
            ngram_key = tuple(input_list[j:(i + j)])
            ngram_counter[ngram_key] += 1

    return ngram_counter

def _check_shape(y_pred, y_true):
    """
    Check the shapes of y_pred and y_true.

    Args:
        y_pred (Tensor): Predict tensor.
        y_true (Tensor): Target tensor.
    """
    if y_pred.ndim != y_true.ndim + 1:
        raise ValueError(f'The dimension of `y_pred` should equal to the dimension of `y_true` add 1,'
                         f'but got `y_pred` dimension: {y_pred.ndim} and `y_true` dimension: {y_true.ndim}.')
    if y_true.shape != (y_pred.shape[0],) + y_pred.shape[2:]:
        raise ValueError(f'`y_pred` shape and `y_true` shape can not match, `y_true` shape should be equal to'
                         f'`y_pred` shape that the value at index 1 is deleted. Such as `y_pred` shape (1, 2, 3),'
                         f' then `y_true` shape should be (1, 3). But got `y_pred` shape {y_pred.shape} and'
                         f' `y_true` shape {y_true.shape}.')
