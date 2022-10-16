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
""""Class for Metric Spearman"""


import math
import numpy as np

from mindnlp.abc import Metric
from mindnlp.common.metrics import _compute_exact, _metric_max_over_ground_truths


class EmScore(Metric):
    r"""
    calculate exact match (EM) score. This metric measures the percentage of predictions
    that match any one of the ground truth answers exactly.

    Args:
        name (str): Name of the metric.

    Returns:
        - **em** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` is None or `labels` is None.
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import EmScore
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
        """Clear the internal evaluation result."""
        self.count = 0
        self.exact_match = 0

    def update(self, *inputs):
        """
        Update local variables.

        Args:
            inputs: Input `preds` and `examples`.
                    preds (Union[str, list]): Predicted value.
                    examples (Union[list, list of list]): Ground truth value.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` doesn't have the same classes number as `examples`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `EmScore.update`, it needs 2 inputs (`preds` and `examples`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        examples = inputs[1]

        if not isinstance(preds, list):
            preds = [preds]
            examples = [examples]

        assert len(preds) == len(examples)

        self.count += len(preds)

        for pred, example in zip(preds, examples):
            self.exact_match += _metric_max_over_ground_truths(
                _compute_exact, pred, example
            )

    def eval(self):
        """
        Compute and return the EM score.

        Returns:
        - **exact_match** (float) - The computed result.

        """
        total_em = int(self.exact_match)

        exact_match = total_em / self.count if self.count > 0 else 0
        return exact_match

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name
