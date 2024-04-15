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
""""Classes for Metrics RougeN and RougeL"""

import numpy as np
from mindnlp.abc import Metric
from .utils import _check_value_type


def _get_ngrams(words, n_size=1):
    """
    Calculates n-gram for multiple sentences.
    """
    ngram_set = set()
    max_start = len(words) - n_size
    for i in range(max_start + 1):
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


def rouge_n_fn(cand_list, ref_list, n_size=1):
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
        raise RuntimeError(f'ROUGE-N can not be calculated, because the number of references is {0}')

    rougen_score = overlap_count / ref_count

    return rougen_score

def rouge_l_fn(cand_list, ref_list, beta=1.2):
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


class RougeN(Metric):
    r"""
    Calculates the ROUGE-N. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set
    of metrics used for evaluating automatic summarization and machine translation models.
    ROUGE-N refers to the overlap of n-grams between candidates and reference summaries.

    Args:
        n_size (int): N_gram value. Default: 1.
        name (str): Name of the metric.

    Example:
        >>> from mindnlp.common.metrics import RougeN
        >>> cand_list = ["the", "cat", "was", "found", "under", "the", "bed"]
        >>> ref_list = [["the", "cat", "was", "under", "the", "bed"]]
        >>> metric = RougeN(2)
        >>> metric.update(cand_list, ref_list)
        >>> rougen_score = metric.eval()
        >>> print(rougen_score)
        0.8

    """
    def __init__(self, n_size=1, name='RougeN'):
        super().__init__()
        self._name = name
        self.n_size = _check_value_type("n_size", n_size, [int])
        self.overlap_count = 0
        self.ref_count = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self.overlap_count = 0
        self.ref_count = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `cand_list` and `ref_list`.

                - cand_list (list): A list of tokenized candidate sentence.
                - ref_list (list): A list of lists of tokenized ground truth sentences.

        Raises:
            ValueError: If the number of inputs is not 2.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `RougeN.update`, it needs 2 inputs (`cand_list` and `ref_list`),'
                             f' but got {len(inputs)}.')

        cand_list = inputs[0]
        ref_list = inputs[1]

        cand_list = _check_value_type("cand_list", cand_list, list)
        ref_list = _check_value_type("ref_list", ref_list, list)

        cand_ngrams = _get_ngrams(cand_list, self.n_size)
        for reference in ref_list:
            ref_ngrams = _get_ngrams(reference, self.n_size)
            self.ref_count += len(ref_ngrams)

            # Gets the overlapping ngrams between evaluated and reference
            overlap_ngrams = cand_ngrams.intersection(ref_ngrams)
            self.overlap_count += len(overlap_ngrams)

    def eval(self):
        """
        Computes and returns the Rouge-N score.

        Returns:
            - **rougen_score** (float) - The computed result.

        Raises:
            RuntimeError: If the reference size is 0.

        """
        if self.ref_count == 0:
            raise RuntimeError(f'ROUGE-N can not be calculated, because the number of references is {0}')

        rougen_score = self.overlap_count / self.ref_count

        return rougen_score

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

class RougeL(Metric):
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
        beta (float): A hyperparameter to decide the weight of recall. Defaults: 1.2.
        name (str): Name of the metric.

    Example:
        >>> from mindnlp.common.metrics import RougeL
        >>> cand_list = ["The","cat","The","cat","on","the","mat"]
        >>> ref_list = [["The","cat","is","on","the","mat"],
                        ["There","is","a","cat","on","the","mat"]]
        >>> metric = RougeL()
        >>> metric.update(cand_list, ref_list)
        >>> rougel_score = metric.eval()
        >>> print(rougel_score)
        0.7800511508951408

    """
    def __init__(self, beta=1.2, name='RougeL'):
        super().__init__()
        self._name = name
        self.bias = _check_value_type("beta", beta, [float])
        self.inst_scores = []

    def clear(self):
        """Clears the internal evaluation results."""
        self.inst_scores = []

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `cand_list` and `ref_list`.
                    cand_list (list): A list of tokenized candidate sentence.
                    ref_list (list): A list of lists of tokenized ground truth sentences.

        Raises:
            ValueError: If the number of inputs is not 2.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `RougeL.update`, it needs 2 inputs (`cand_list` and `ref_list`),'
                             f' but got {len(inputs)}.')

        cand_list = inputs[0]
        ref_list = inputs[1]

        cand_list = _check_value_type("cand_list", cand_list, list)
        ref_list = _check_value_type("ref_list", ref_list, list)

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
            score = ((1 + self.bias**2) * prec_max * rec_max) / \
                    float(rec_max + self.bias**2 * prec_max)
        else:
            score = 0.0
        self.inst_scores.append(score)

    def eval(self):
        """
        Computes and returns the Rouge-L score.

        Returns:
            - **rougel_score** (float) - The computed result.

        """
        rougel_score = 1. * sum(self.inst_scores) / len(self.inst_scores)

        return rougel_score

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['rouge_n_fn', 'rouge_l_fn', 'RougeL', 'RougeN']
