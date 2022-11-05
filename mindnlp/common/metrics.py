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
""""Functions for Metrics"""

import sys
import math
import string
from collections.abc import Iterable
from collections import Counter
import re
import numpy as np
from mindspore import Tensor


def perplexity(preds, labels, ignore_label=None):
    r"""
    Calculates the perplexity. Perplexity is a measure of how well a probabilibity model
    predicts a sample. A low perplexity indicates the model is good at predicting the
    sample. The function is shown as follows:

    .. math::

        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    where :math:`w` represents words in corpus.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list
            of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the
            number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.
        ignore_label (Union[int, None]): Index of an invalid label to be ignored
            when counting. If set to `None`, it means there's no invalid label.
            Default: None.

    Returns:
        - **ppl** (float) - The computed result.

    Raises:
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
        2.231443166940565

    """
    if ignore_label is not None:
        ignore_label = _check_value_type("ignore_label", ignore_label, [int])

    preds = _check_value_type("preds", preds, [Tensor, list, np.ndarray])
    labels = _check_value_type("labels", labels, [Tensor, list, np.ndarray])

    y_pred = [_convert_data_type(preds)]
    y_true = [_convert_data_type(labels)]

    if len(y_pred) != len(y_true):
        raise RuntimeError(f'`preds` and `labels` should have the same length, but got `preds` '
                           f'length {len(y_pred)}, `labels` length {len(y_true)})')

    sum_cross_entropy = 0.0
    sum_word_num = 0

    cross_entropy = 0.
    word_num = 0
    for label, pred in zip(y_true, y_pred):
        if pred.ndim == label.ndim and _check_onehot_data(label):
            label = label.argmax(axis=1)

        if label.size != pred.size / pred.shape[-1]:
            raise RuntimeError(f'`preds` and `labels` should have the same shape, but got `preds` '
                               f'shape {pred.shape}, label shape {label.shape}.')
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
        raise RuntimeError(f'Perplexity can not be calculated, because the number of samples is '
                           f'{0}')

    ppl = math.exp(sum_cross_entropy / sum_word_num)

    return ppl

def bleu(cand, ref_list, n_size=4, weights=None):
    r"""
    Calculates the BLEU score. BLEU (bilingual evaluation understudy) is a metric
    for evaluating the quality of text translated by machine. It uses a modified form
    of precision to compare a candidate translation against multiple reference translations.
    The function is shown as follows:

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
        ref_list (list): A list of lists of tokenized true sentences.
        n_size (int): N_gram value ranges from 1 to 4. Default: 4.
        weights (Union[list, None]): Weights of precision of each gram. Defaults to None.

    Returns:
        - **bleu_score** (float) - The computed result.

    Raises:
        ValueError: If the value range of `n_size` is not from 1 to 4.
        ValueError: If the lengths of `cand` and `ref_list` are not equal.
        ValueError: If the lengths of `weights` is not equal to `n_size`.

    Example:
        >>> from mindnlp.common.metrics import bleu
        >>> cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        >>> ref_list = [[["The", "cat", "is", "on", "the", "mat"],
                        ["There", "is", "a", "cat", "on", "the", "mat"]]]
        >>> bleu_score = bleu(cand, ref_list)
        >>> print(bleu_score)
        0.46713797772820015

    """
    n_size = _check_value_type("n_size", n_size, [int])
    if n_size > 4 or n_size < 1:
        raise ValueError(f'`n_size` should range from 1 to 4, but got {n_size}')

    cand = _check_value_type("cand", cand, list)
    ref_list = _check_value_type("ref_list", ref_list, list)

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
        weights = [1 / n_size for _ in range(n_size)]

    if n_size != len(weights):
        raise ValueError("The length of `weights` should be equal to `n_size`")

    log_precision_scores = weights * np.log(precision_scores)
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.array(1.0) if bp_c > bp_r else np.exp(1 - (ref_len / cand_len))
    bleu_score = brevity_penalty * geometric_mean

    return bleu_score

def rouge_n(cand_list, ref_list, n_size=1):
    r"""
    Calculates the ROUGE-N score. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is
    a set of metrics used for evaluating automatic summarization and machine translation
    models. ROUGE-N refers to the overlap of n-grams between candidates and reference
    summaries.

    Args:
        cand_list (list): A list of tokenized candidate sentences.
        ref_list (list): A list of lists of tokenized true sentences.
        n_size (int): N_gram value. Default: 1.

    Returns:
        - **rougen_score** (float) - The computed result.

    Raises:
        RuntimeError: If the reference size is 0.

    Example:
        >>> from mindnlp.common.metrics import rouge_n
        >>> cand_list = ["the", "cat", "was", "found", "under", "the", "bed"]
        >>> ref_list = [["the", "cat", "was", "under", "the", "bed"]]
        >>> rougen_score = rouge_n(cand_list, ref_list, 2)
        >>> print(rougen_score)
        0.8

    """
    cand_list = _check_value_type("cand_list", cand_list, list)
    ref_list = _check_value_type("ref_list", ref_list, list)
    n_size = _check_value_type("n_size", n_size, [int])

    overlap_count = 0
    ref_count = 0

    cand_ngrams = _get_ngrams(cand_list, n_size)
    for reference in ref_list:
        ref_ngrams = _get_ngrams(reference, n_size)
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
    Calculates the ROUGE-L score. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is
    a set of metrics used for evaluating automatic summarization and machine translation
    models. ROUGE-L is calculated based on Longest Common Subsequence (LCS). The function
    is shown as follows:

    .. math::

        R_{l c s}=\frac{L C S(X, Y)}{m}

        p_{l c s}=\frac{L C S(X, Y)}{n}

        F_{l c s}=\frac{\left(1+\beta^{2}\right) R_{l c s} P_{l c s}}{R_{l c s}+\beta^{2} P_{l c s}}

    where `X` is the candidate sentence, `Y` is the reference sentence. `m` and `n` represent
    the length of `X` and `Y` respectively. `LCS` means the longest common subsequence.

    Args:
        cand_list (list): A list of tokenized candidate sentence.
        ref_list (list): A list of lists of tokenized true sentences.
        beta (float): A hyperparameter to decide the weight of recall. Defaults: 1.2.

    Returns:
        - **rougel_score** (float) - The computed result.

    Example:
        >>> from mindnlp.common.metrics import rouge_l
        >>> cand_list = ["The","cat","The","cat","on","the","mat"]
        >>> ref_list = [["The","cat","is","on","the","mat"],
                        ["There","is","a","cat","on","the","mat"]]
        >>> rougel_score = rouge_l(cand_list, ref_list)
        >>> print(rougel_score)
        0.7800511508951408

    """
    cand_list = _check_value_type("cand_list", cand_list, list)
    ref_list = _check_value_type("ref_list", ref_list, list)
    beta = _check_value_type("beta", beta, [float])

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
    Calculates the Distinct-N. Distinct-N is a metric that measures the diversity of
    a sentence. It focuses on the number of distinct n-gram of a sentence. The larger
    the number of distinct n-grams, the higher the diversity of the text. The function
    is shown as follows:

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
    cand_list = _check_value_type("cand_list", cand_list, list)
    n_size = _check_value_type("n_size", n_size, [int])

    diff_ngram = set()
    count = 0.0

    for i in range(0, len(cand_list) - n_size + 1):
        ngram = ' '.join(cand_list[i:(i + n_size)])
        count += 1
        diff_ngram.add(ngram)

    distinct_score = len(diff_ngram) / count
    return distinct_score

def accuracy(preds, labels):
    r"""
    Calculates the accuracy. The function is shown as follows:

    .. math::

        \text{ACC} =\frac{\text{TP} + \text{TN}}
        {\text{TP} + \text{TN} + \text{FP} + \text{FN}}

    where `ACC` is accuracy, `TP` is the number of true posistive cases, `TN` is the number
    of true negative cases, `FP` is the number of false posistive cases, `FN` is the number
    of false negative cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **acc** (float) - The computed result.

    Raises:
        RuntimeError: If the number of samples is 0.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import accuracy
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> acc = accuracy(preds, labels)
        >>> print(acc)
        0.6666666666666666

    """
    correct_num = 0
    total_num = 0

    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)

    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    indices = y_pred.argmax(axis=1)
    result = (np.equal(indices, y_true) * 1).reshape(-1)

    correct_num += result.sum()
    total_num += result.shape[0]

    if total_num == 0:
        raise RuntimeError(f'Accuracy can not be calculated, because the number of samples is '
                           f'{0}. Please check whether your inputs(predicted value, true value) '
                           f'are empty.')
    acc = correct_num / total_num
    return acc

def precision(preds, labels):
    r"""
    Calculates the precision. Precision (also known as positive predictive value) is
    the actual positive proportion in the predicted positive sample. It can only be
    used to evaluate the precision score of binary tasks. The function is shown
    as follows:

    .. math::

        \text{Precision} =\frac{\text{TP}} {\text{TP} + \text{FP}}

    where `TP` is the number of true posistive cases, `FP` is the number of false posistive cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **prec** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import precision
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> prec = precision(preds, labels)
        >>> print(prec)
        [0.5 1. ]

    """
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

    prec = (true_positives / (positives + epsilon))
    return prec

def recall(preds, labels):
    r"""
    Calculates the recall. Recall is also referred to as the true positive rate or
    sensitivity. The function is shown as follows:

    .. math::

        \text{Recall} =\frac{\text{TP}} {\text{TP} + \text{FN}}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **rec** (np.ndarray) - The computed result.

    Raises:
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
        [1. 0.5]

    """
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

    rec = (true_positives / (actual_positives + epsilon))
    return rec

def f1_score(preds, labels):
    r"""
    Calculates the F1 score. Fbeta score is a weighted mean of precision and recall,
    and F1 score is a special case of Fbeta when beta is 1. The function is shown
    as follows:

    .. math::

        F_1=\frac{2\cdot TP}{2\cdot TP + FN + FP}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases,
    `FP` is the number of false positive cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **f1_s** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import f1_score
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> labels = Tensor(np.array([1, 0, 1]))
        >>> f1_s = f1_score(preds, labels)
        >>> print(f1_s)
        [0.6666666666666666 0.6666666666666666]

    """
    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)

    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` and `labels` should contain same classes, but got `preds` '
                         f'contains {class_num} classes and true value contains '
                         f'{y_true.max() + 1}')
    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    positives = y_pred.sum(axis=0)
    actual_positives = y_true.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    f1_s = (2 * true_positives / (actual_positives + positives + epsilon))
    return f1_s

def matthews_correlation(preds, labels):
    r"""
    Calculates the Matthews correlation coefficient (MCC). MCC is in essence a correlation
    coefficient between the observed and predicted binary classifications; it returns a value
    between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than
    random prediction and −1 indicates total disagreement between prediction and observation.
    The function is shown as follows:

    .. math::

        MCC=\frac{TP \times TN-FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    where `TP` is the number of true posistive cases, `TN` is the number of true negative cases,
    `FN` is the number of false negative cases, `FP` is the number of false positive cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers and the shape of `preds` is :math:`(N, C)` in most cases
            (not strictly), where :math:`N` is the number of cases and :math:`C` is the
            number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **m_c_c** (float) - The computed result.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import matthews_correlation
        >>> preds = [[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]
        >>> labels = [0, 1, 0, 1, 0]
        >>> m_c_c = matthews_correlation(preds, labels)
        >>> print(m_c_c)
        0.16666666666666666

    """
    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim == labels.ndim and _check_onehot_data(labels):
        labels = labels.argmax(axis=1)
    _check_shape(preds, labels)

    preds = np.argmax(preds, axis=1)
    labels = labels.reshape(-1, 1)

    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0

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

def pearson_correlation(preds, labels):
    r"""
    Calculates the Pearson correlation coefficient (PCC). PCC is a measure of linear
    correlation between two sets of data. It is the ratio between the covariance of
    two variables and the product of their standard deviations; thus, it is essentially
    a normalized measurement of the covariance, such that the result always has a value
    between −1 and 1.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers and the shape of `preds` is :math:`(N, 1)`.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` is a list of
            floating numbers and the shape of `preds` is :math:`(N, 1)`.

    Returns:
        - **p_c_c** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` and `labels` have different lengths.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import pearson_correlation
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> p_c_c = pearson_correlation(preds, labels)
        >>> print(p_c_c)
        0.9985229081857804

    """
    def _pearson_correlation(y_pred, y_true):
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

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    preds = np.squeeze(preds.reshape(-1, 1)).tolist()
    labels = np.squeeze(labels.reshape(-1, 1)).tolist()

    if len(preds) != len(labels):
        raise RuntimeError(f'`preds` and `labels` should have the same length, but got `preds` '
                           f'length {len(preds)}, `labels` length {len(labels)})')

    p_c_c = _pearson_correlation(preds, labels)
    return p_c_c

def spearman_correlation(preds, labels):
    r"""
    Calculates the Spearman's rank correlation coefficient (SRCC). It is a nonparametric
    measure of rank correlation (statistical dependence between the rankings of two
    variables). It assesses how well the relationship between two variables can be
    described using a monotonic function. If there are no repeated data values, a
    perfect Spearman correlation of +1 or −1 occurs when each of the variables is
    a perfect monotone function of the other.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers and the shape of `preds` is :math:`(N, 1)`.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` is a list of
            floating numbers and the shape of `preds` is :math:`(N, 1)`.

    Returns:
        - **s_r_c_c** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` and `labels` have different lengths.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import spearman_correlation
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> s_r_c_c = spearman_correlation(preds, labels)
        >>> print(s_r_c_c)
        1.0

    """
    def _spearman(y_pred, y_true):
        preds_rank = _get_rank(y_pred)
        labels_rank = _get_rank(y_true)

        total = 0
        n_pred = len(y_pred)
        for i in range(n_pred):
            total += pow((preds_rank[i] - labels_rank[i]), 2)
        res = 1 - float(6 * total) / (n_pred * (pow(n_pred, 2) - 1))
        return res

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    preds = np.squeeze(preds.reshape(-1, 1)).tolist()
    labels = np.squeeze(labels.reshape(-1, 1)).tolist()

    if len(preds) != len(labels):
        raise RuntimeError(f'`preds` and `labels` should have the same length, but got `preds` '
                           f'length {len(preds)}, `labels` length {len(labels)})')

    s_r_c_c = _spearman(preds, labels)
    return s_r_c_c

def em_score(preds, examples):
    r"""
    Calculates the exact match (EM) score. This metric measures the percentage of
    predictions that match any one of the ground truth exactly.

    Args:
        preds (Union[str, list]): Predicted value.
        examples (list): Ground truth.

    Returns:
        - **exact_match** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` and `examples` have different lengths.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import em_score
        >>> preds = "this is the best span"
        >>> examples = ["this is a good span", "something irrelevant"]
        >>> exact_match = em_score(preds, examples)
        >>> print(exact_match)
        0.0

    """
    _check_value_type("preds", preds, [str, list])
    _check_value_type("examples", examples, [list])

    if not isinstance(preds, list):
        preds = [preds]
        examples = [examples]

    if len(preds) != len(examples):
        raise RuntimeError(f'`preds` and `examples` should have the same length, but got `examples`'
                           f' length {len(preds)}, `labels` length {len(examples)})')

    count = len(preds)
    exact_match = 0

    for pred, example in zip(preds, examples):
        exact_match += _metric_max_over_ground_truths(
            _compute_exact, pred, example
        )

    total_em = int(exact_match)

    exact_match = total_em / count if count > 0 else 0
    return exact_match

def confusion_matrix(preds, labels, class_num=2):
    r"""
    Calculates the confusion matrix. Confusion matrix is commonly used to evaluate
    the performance of classification models, including binary classification and
    multiple classification.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers and the shape of `preds` is :math:`(N, C)` or :math:`(N,)`.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. The shape of `labels` is
            :math:`(N,)`.
        class_num (int): Number of classes in the dataset. Default: 2.

    Returns:
        - **conf_mat** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` and `labels` do not have valid dimensions.

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
    class_num = _check_value_type("class_num", class_num, [int])

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim not in (labels.ndim, labels.ndim + 1):
        raise ValueError(f'`preds` and `labels` should have the same dimensions, or the dimension '
                         f'of `preds` equals the dimension of true value add 1, but got `preds` '
                         f'ndim: {preds.ndim}, `labels` ndim: {labels.ndim}.')

    if preds.ndim == labels.ndim + 1:
        preds = np.argmax(preds, axis=1)

    trans = (labels.reshape(-1) * class_num + preds.reshape(-1)).astype(int)
    bincount = np.bincount(trans, minlength=class_num ** 2)
    conf_mat = bincount.reshape(class_num, class_num)

    conf_mat = conf_mat.astype(float)

    return conf_mat


# Common functions.
def _check_value_type(arg_name, arg_value, valid_types):
    """
    Checks whether the data type of argument is valid

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
        raise TypeError(f'Type of `{arg_name}` should be {"one of " if num_types > 1 else ""}'
                        f' `{valid_types if num_types > 1 else str(valid_types[0])}`, '
                        f'but got `{arg_value}` with type `{type(arg_value).__name__}`.')
    if not isinstance(arg_value, tuple(valid_types)):
        raise TypeError(f'Type of `{arg_name}` should be {"one of " if num_types > 1 else ""}'
                        f'`{valid_types if num_types > 1 else str(valid_types[0])}`, '
                        f'but got `{arg_value}` with type `{type(arg_value).__name__}`.')
    return arg_value

def _check_onehot_data(data):
    """
    Checks whether input data is one-hot encoding.

    Args:
        data (np.array): Input data.

    Returns:
        - **ans** (bool) - Rreturn true, if input data is one-hot encoding.
    """
    ans = False
    data = _convert_data_type(data)
    if np.equal(data ** 2, data).all():
        shp = (data.shape[0],) + data.shape[2:]
        if np.equal(np.ones(shp), data.sum(axis=1)).all():
            ans = True
            return ans
    return ans

def _convert_data_type(data):
    """
    Converts data type to numpy array.

    Args:
        data (Union[Tensor, list, np.ndarray]): Input data.

    Returns:
        - **data** (np.ndarray) - Data with `np.ndarray` type.

    Raises:
        TypeError: If the data type is not tensor, list or np.ndarray.

    """
    if isinstance(data, Tensor):
        data = data.asnumpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f'Input data type must be tensor, list or '
                        f'np.ndarray, but got {type(data)}.')
    return data

def _count_ngram(input_list, n_gram):
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(len(input_list) - i + 1):
            ngram_key = tuple(input_list[j:(i + j)])
            ngram_counter[ngram_key] += 1

    return ngram_counter

def _check_shape(y_pred, y_true, n_class=None):
    """
    Checks the shapes of y_pred and y_true.

    Args:
        y_pred (Tensor): Predict tensor.
        y_true (Tensor): Target tensor.
    """
    if y_pred.ndim != y_true.ndim + 1 and n_class != 1:
        raise ValueError(f'The dimension of `y_pred` should be equal to the dimension of `y_true` '
                         f'add 1, but got `y_pred` dimension: {y_pred.ndim} and `y_true` dimension:'
                         f' {y_true.ndim}.')
    if y_true.shape != (y_pred.shape[0],) + y_pred.shape[2:] and n_class != 1:
        raise ValueError(f'`y_pred` shape and `y_true` shape can not match, `y_true` shape should '
                         f'be equal to `y_pred` shape that the value at index 1 is deleted. Such as'
                         f' `y_pred` shape (1, 2, 3), then `y_true` shape should be (1, 3). But got'
                         f' `y_pred` shape {y_pred.shape} and `y_true` shape {y_true.shape}.')

def _get_ngrams(words, n_size=1):
    """
    Calculates n-gram for multiple sentences.
    """
    ngram_set = set()
    max_start = len(words) - n_size
    for i in range(max_start + 1):
        print(tuple(words[i:i + n_size]))
        ngram_set.add(tuple(words[i:i + n_size]))
    return ngram_set

def _lcs(strg, sub):
    """
    Calculates the length of longest common subsequence of strg and sub.

    Args:
        strg (list): The string to be calculated, usually longer the sub string.
        sub (list): The sub string to be calculated.

    Returns:
        - **length** (float) - The length of the longest common subsequence
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

    length = lengths[len(strg)][len(sub)]
    return length

def _get_rank(raw_list):
    raw_x = np.array(raw_list)
    rank_x = np.empty(raw_x.shape, dtype=int)
    sort_x = np.argsort(-raw_x)
    for i, k in enumerate(sort_x):
        rank_x[k] = i + 1
    return rank_x

def _compute_exact(y_pred, y_true):
    def _normalize_answer(txt):
        """Lowers text and removes punctuation, articles and extra whitespace."""

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

    return int(_normalize_answer(y_pred) == _normalize_answer(y_true))

def _metric_max_over_ground_truths(metric_fn, pred, example):
    scores_for_ground_truths = []
    for y_eg in example:
        score = metric_fn(pred, y_eg)
        scores_for_ground_truths.append(score)
    return round(max(scores_for_ground_truths), 2)
