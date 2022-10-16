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


class Spearman(Metric):
    r"""
    calculate Spearman's rank correlation coefficient. It is a nonparametric measure of
    rank correlation (statistical dependence between the rankings of two variables).
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
        >>> from mindnlp.common.metrics import Spearman
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> metric = Spearman()
        >>> metric.update(preds, labels)
        >>> scc = metric.eval()
        >>> print(scc)
        1.0

    """
    def __init__(self, name='Spearman'):
        super().__init__()
        self._name = name
        self.total = 0
        self.n_pred = 0

    def clear(self):
        """Clear the internal evaluation result."""
        self.total = 0
        self.n_pred = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.
                    preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list
                        of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                        :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
                        of cases and :math:`C` is the number of categories.
                    labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must
                        be in one-hot format that shape is :math:`(N, C)`, or can be transformed
                        to one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `Spearman.update`, it needs 2 inputs (`preds` and `labels`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        preds = _convert_data_type(preds)
        labels = _convert_data_type(labels)

        if preds.ndim not in (labels.ndim, labels.ndim + 1):
            raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension '
                             f'of `preds` equals the dimension of `labels` add 1, but got '
                             f'predicted value ndim: {preds.ndim}, true value ndim: {labels.ndim}.')

        preds = np.squeeze(preds.reshape(-1, 1)).tolist()
        labels = np.squeeze(labels.reshape(-1, 1)).tolist()

        preds_rank = _get_rank(preds)
        labels_rank = _get_rank(labels)

        self.n_pred += len(preds)
        for i in range(self.n_pred):
            self.total += pow((preds_rank[i] - labels_rank[i]), 2)

    def eval(self):
        """
        Compute and return the SCC.

        Returns:
            - **scc** (float) - The computed result.

        """
        scc = 1 - float(6 * self.total) / (self.n_pred * (pow(self.n_pred, 2) - 1))
        return scc

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name
