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


import numpy as np

from mindnlp.abc import Metric
from mindnlp.common.metrics import _convert_data_type, _get_rank


class SpearmanCorrelation(Metric):
    r"""
    Calculates the Spearman's rank correlation coefficient (SRCC). It is a nonparametric measure
    of rank correlation (statistical dependence between the rankings of two variables).
    It assesses how well the relationship between two variables can be described
    using a monotonic function. If there are no repeated data values, a perfect
    Spearman correlation of +1 or −1 occurs when each of the variables is
    a perfect monotone function of the other.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.engine.metrics import SpearmanCorrelation
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> metric = SpearmanCorrelation()
        >>> metric.update(preds, labels)
        >>> s_r_c_c = metric.eval()
        >>> print(s_r_c_c)
        1.0

    """
    def __init__(self, name='SpearmanCorrelation'):
        super().__init__()
        self._name = name
        self.preds = []
        self.labels = []

    def clear(self):
        """Clears the internal evaluation results."""
        self.preds = []
        self.labels = []

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
                  floating numbers and the shape of `preds` is :math:`(N, 1)`.
                - labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` is a list of
                  floating numbers and the shape of `preds` is :math:`(N, 1)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            RuntimeError: If `preds` and `labels` have different lengths.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `SpearmanCorrelation.update`, it needs 2 inputs (`preds` '
                             f'and `labels`), but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        preds = _convert_data_type(preds)
        labels = _convert_data_type(labels)

        preds = np.squeeze(preds.reshape(-1, 1)).tolist()
        labels = np.squeeze(labels.reshape(-1, 1)).tolist()

        if len(preds) != len(labels):
            raise RuntimeError(f'For `SpearmanCorrelation.update`, `preds` and `labels` should have'
                               f' the same length, but got `preds` length {len(preds)}, `labels` '
                               f'length {len(labels)})')

        self.preds.append(preds)
        self.labels.append(labels)

    def eval(self):
        """
        Computes and returns the SRCC.

        Returns:
            - **s_r_c_c** (float) - The computed result.

        """
        self.preds = [item for pred in self.preds for item in pred]
        self.labels = [item for label in self.labels for item in label]

        preds_rank = _get_rank(self.preds)
        labels_rank = _get_rank(self.labels)

        total = 0
        n_preds = len(self.preds)
        for i in range(n_preds):
            total += pow((preds_rank[i] - labels_rank[i]), 2)

        s_r_c_c = 1 - float(6 * total) / (n_preds * (pow(n_preds, 2) - 1))
        return s_r_c_c

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name
