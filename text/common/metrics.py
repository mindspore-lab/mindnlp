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
""""Metrics Functional"""


import math
from collections.abc import Iterable
from collections import Counter
import numpy as np
from mindspore import Tensor

def perplexity(preds, labels, ignore_label=None):
    r"""
    Calculate perplexity. Perplexity is a measurement of how well a probabilibity model predicts a
    sample. A low perplexity indicates the model is good at predicting the sample. The function is shown as follows:

    .. math::

        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    Where :math:`w` represents words in corpus.

    Args:
        preds (Union[Tensor, list, numpy.ndarray]): Predicted values. The shape of `preds` is :math:`(N, C)`.
        labels (Union[Tensor, list, numpy.ndarray]): Labels of data. The shape of `labels` is :math:`(N, C)`.
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
        >>> from text.common.metrics import perplexity
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

    preds = [_convert_data_type(preds)]
    labels = [_convert_data_type(labels)]

    if len(preds) != len(labels):
        raise RuntimeError(f'`preds` and `labels` should have the same length, but got `preds` length'
                           f'{len(preds)}, `labels` length {len(labels)})')

    sum_cross_entropy = 0.0
    sum_word_num = 0

    cross_entropy = 0.
    word_num = 0
    for label, pred in zip(labels, preds):
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
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If the lengths of `cand` and `ref_list` are not equal.
        ValueError: If the lengths of `weights` is not equal to `n_size`.

    Example:
        >>> from text.common.metrics import bleu
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
        >>> from text.common.metrics import rouge_n
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
        >>> from text.common.metrics import rouge_l
        >>> cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        >>> ref_list = [["The", "cat", "is", "on", "the", "mat"], ["There", "is", "a", "cat", "on", "the", "mat"]]
        >>> rougel_score = rouge_l(cand_list, ref_list)
        >>> print(rougel_score)
        0.7800511508951408

    """
    def _lcs(string, sub):
        """
        Calculate the length of longest common subsequence of cand and ref.

        Args:
            string (list): The string to be calculated, usually longer the sub string.
            sub (list): The sub string to be calculated.

        Returns:
            - **length** (numpy.float32) - The length of the longest common subsequence
            of string and sub.
        """
        if len(string) < len(sub):
            sub, string = string, sub
        lengths = np.zeros((len(string) + 1, len(sub) + 1))
        for j in range(1, len(sub) + 1):
            for i in range(1, len(string) + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[len(string)][len(sub)]

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
        >>> from text.common.metrics import distinct
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

def _convert_data_type(data):
    """
    Convert data type to numpy array.

    Args:
        data (Object): Input data.

    Returns:
        - **data** (numpy.ndarray) - Data with `np.ndarray` type.

    Raises:
        TypeError: If the data type is not tensor, list or numpy.ndarray.

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
