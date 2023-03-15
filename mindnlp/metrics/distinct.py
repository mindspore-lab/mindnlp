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
""""Class for Metric Distinct"""


from mindnlp.abc import Metric
from .utils import _check_value_type

def distinct_fn(cand_list, n_size=2):
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


class Distinct(Metric):
    """
    Calculates the Distinct-N. Distinct-N is a metric that measures the diversity of
    a sentence. It focuses on the number of distinct n-gram of a sentence. The larger
    the number of distinct n-grams, the higher the diversity of the text. The function
    is shown as follows:

    Args:
        n_size (int): N_gram value. Defaults: 2.
        name (str): Name of the metric.

    Example:
        >>> from mindnlp.common.metrics import Distinct
        >>> cand_list = ["The", "cat", "The", "cat", "on", "the", "mat"]
        >>> metric = Distinct()
        >>> metric.update(cand_list)
        >>> distinct_score = metric.eval()
        >>> print(distinct_score)
        0.8333333333333334

    """
    def __init__(self, n_size=2, name='Distinct'):
        super().__init__()
        self._name = name
        self.n_size = _check_value_type("n_size", n_size, [int])
        self.diff_ngram = set()
        self.count = 0.0

    def clear(self):
        """Clears the internal evaluation results."""
        self.diff_ngram = set()
        self.count = 0.0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `cand_list`.

                - cand_list (list): A list of tokenized candidate sentence.

        Raises:
            ValueError: If the number of inputs is not 1.

        """
        if len(inputs) != 1:
            raise ValueError(f'For `Distinct.update`, it needs 1 inputs (`cand_list`), '
                             f'but got {len(inputs)}.')
        cand_list = inputs[0]

        cand_list = _check_value_type("cand_list", cand_list, list)

        for i in range(0, len(cand_list) - self.n_size + 1):
            ngram = ' '.join(cand_list[i:(i + self.n_size)])
            self.count += 1
            self.diff_ngram.add(ngram)

    def eval(self):
        """
        Computes and returns the Distinct-N.

        Returns:
            - **distinct_score** (float) - The computed result.

        """
        distinct_score = len(self.diff_ngram) / self.count
        return distinct_score

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['distinct_fn', 'Distinct']
