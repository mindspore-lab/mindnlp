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
""""Class for Metric BleuScore"""


from collections import Counter
import numpy as np

from mindnlp.abc import Metric
from .utils import _check_value_type

class BleuScore(Metric):
    r"""
    Calculates the BLEU score. BLEU (bilingual evaluation understudy) is a metric for
    evaluating the quality of text translated by machine. It uses a modified form of
    precision to compare a candidate translation against multiple reference translations.
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
        n_size (int): N_gram value ranges from 1 to 4. Default: 4.
        weights (Union[list, None]): Weights of precision of each gram. Defaults to None.
        name (str): Name of the metric.

    Raises:
        ValueError: If the value range of `n_size` is not from 1 to 4.
        ValueError: If the lengths of `weights` is not equal to `n_size`.

    Example:
        >>> from mindnlp.common.metrics import BleuScore
        >>> cand = [["The", "cat", "The", "cat", "on", "the", "mat"]]
        >>> ref_list = [[["The", "cat", "is", "on", "the", "mat"],
                        ["There", "is", "a", "cat", "on", "the", "mat"]]]
        >>> metric = BleuScore()
        >>> metric.update(cand, ref_list)
        >>> bleu_score = metric.eval()
        >>> print(bleu_score)
        0.46713797772820015

    """
    def __init__(self, n_size=4, weights=None, name='BleuScore'):
        super().__init__()
        self._name = name
        self.n_size = _check_value_type("n_size", n_size, [int])
        if self.n_size > 4 or self.n_size < 1:
            raise ValueError(f'For `BleuScore`, `n_size` should range from 1 to 4, but '
                             f'got {n_size}')

        if weights is None:
            self.weights = [1 / self.n_size for _ in range(self.n_size)]
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
        """Clears the internal evaluation results."""
        self.numerator = np.zeros(self.n_size)
        self.denominator = np.zeros(self.n_size)
        self.precision_scores = np.zeros(self.n_size)
        self.bp_c = 0.0
        self.bp_r = 0.0
        self.cand_len = 0
        self.ref_len = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `cand` and `ref_list`.

                - cand (list): A list of tokenized candidate sentences.
                - ref_list (list): A list of lists of tokenized ground truth sentences.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If the lengths of `cand` and `ref_list` are not equal.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `BleuScore.update`, it needs 2 inputs (`cand` and `ref_list`), '
                             f'but got {len(inputs)}.')

        cand = inputs[0]
        ref_list = inputs[1]

        cand = _check_value_type("cand", cand, list)
        ref_list = _check_value_type("ref_list", ref_list, list)

        if len(cand) != len(ref_list):
            raise ValueError(f'For `BleuScore.update`, `cand` and `ref_list` should be equal in '
                             f'length, but got {len(cand)}, {len(ref_list)}')

        for (candidate, references) in zip(cand, ref_list):
            self.bp_c += len(candidate)
            ref_len_list = [len(ref) for ref in references]
            ref_len_diff = [abs(len(candidate) - x) for x in ref_len_list]
            self.bp_r += ref_len_list[ref_len_diff.index(min(ref_len_diff))]
            candidate_counter = count_ngram(candidate, self.n_size)
            reference_counter = Counter()

            for ref in references:
                reference_counter |= count_ngram(ref, self.n_size)

            ngram_counter_clip = candidate_counter & reference_counter

            for counter_clip in ngram_counter_clip:
                self.numerator[len(counter_clip) - 1] += ngram_counter_clip[counter_clip]

            for counter in candidate_counter:
                self.denominator[len(counter) - 1] += candidate_counter[counter]

        self.cand_len = np.array(self.bp_c)
        self.ref_len = np.array(self.bp_r)

    def eval(self):
        """
        Computes and returns the BLEU score.

        Returns:
            - **bleu_score** (float) - The computed result.

        """
        if min(self.numerator) == 0.0:
            return np.array(0.0)

        precision_scores = self.numerator / self.denominator

        log_precision_scores = self.weights * np.log(precision_scores)
        geometric_mean = np.exp(np.sum(log_precision_scores))
        brevity_penalty = np.array(1.0) if self.bp_c > self.bp_r else np.exp(1 - \
                            (self.ref_len / self.cand_len))
        bleu_score = brevity_penalty * geometric_mean

        return bleu_score

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

def count_ngram(input_list, n_gram):
    """count ngram"""
    ngram_counter = Counter()

    for i in range(1, n_gram + 1):
        for j in range(len(input_list) - i + 1):
            ngram_key = tuple(input_list[j:(i + j)])
            ngram_counter[ngram_key] += 1

    return ngram_counter


def bleu_fn(cand, ref_list, n_size=4, weights=None):
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
        candidate_counter = count_ngram(candidate, n_size)
        reference_counter = Counter()

        for ref in references:
            reference_counter |= count_ngram(ref, n_size)

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

__all__ = ['bleu_fn', 'count_ngram', 'BleuScore']
