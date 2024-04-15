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
""""Class for Metric EmScore"""

import string
import re
from mindnlp.abc import Metric
from .utils import _check_value_type

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

def em_score_fn(preds, examples):
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


class EmScore(Metric):
    r"""
    Calculates the exact match (EM) score. This metric measures the percentage of
    predictions that match any one of the ground truth answers exactly.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.engine.metrics import EmScore
        >>> preds = "this is the best span"
        >>> examples = ["this is a good span", "something irrelevant"]
        >>> metric = EmScore()
        >>> metric.update(preds, examples)
        >>> em_score = metric.eval()
        >>> print(em_score)
        0.0

    """
    def __init__(self, name='EmScore'):
        super().__init__()
        self._name = name
        self.count = 0
        self.exact_match = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self.count = 0
        self.exact_match = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `examples`.

                - preds (Union[str, list]): Predicted value.
                - examples (list): Ground truth.

        Raises:
            ValueError: If the number of inputs is not 2.
            RuntimeError: If `preds` and `examples` have different lengths.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `EmScore.update`, it needs 2 inputs (`preds` and `examples`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        examples = inputs[1]

        _check_value_type("preds", preds, [str, list])
        _check_value_type("examples", examples, [list])

        if not isinstance(preds, list):
            preds = [preds]
            examples = [examples]

        if len(preds) != len(examples):
            raise RuntimeError(f'For `EmScore.update`, `preds` and `examples` should have the same '
                               f'length, but got `examples` length {len(preds)}, `labels` length '
                               f'{len(examples)})')

        self.count += len(preds)

        for pred, example in zip(preds, examples):
            self.exact_match += _metric_max_over_ground_truths(
                _compute_exact, pred, example
            )

    def eval(self):
        """
        Computes and returns the EM score.

        Returns:
        - **exact_match** (float) - The computed result.

        """
        total_em = int(self.exact_match)

        exact_match = total_em / self.count if self.count > 0 else 0
        return exact_match

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['em_score_fn', 'EmScore']
