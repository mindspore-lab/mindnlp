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
""""Class for Metric Pearson"""


import math
import numpy as np

from mindnlp.abc import Metric
from mindnlp.common.metrics import _convert_data_type


class Pearson(Metric):
    r"""
    calculate Pearson correlation coefficient (PCC). PCC is a measure of linear correlation
    between two sets of data. It is the ratio between the covariance of two variables and
    the product of their standard deviations; thus, it is essentially a normalized measurement
    of the covariance, such that the result always has a value between −1 and 1.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import Pearson
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> metric = Pearson()
        >>> metric.update(preds, labels)
        >>> pcc = metric.eval()
        >>> print(pcc)
        0.9985229081857804
    """
    def __init__(self, name='Pearson'):
        super().__init__()
        self._name = name
        self.numerator = 0
        self.denominator = 0

    def clear(self):
        """Clears the internal evaluation result."""
        self.numerator = 0
        self.denominator = 0

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
            raise ValueError(f'For `Pearson.update`, it needs 2 inputs (`preds` and `labels`), '
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

        n_pred = len(preds)
        # simple sums
        sum1 = sum(float(preds[i]) for i in range(n_pred))
        sum2 = sum(float(labels[i]) for i in range(n_pred))
        # sum up the squares
        sum1_pow = sum(pow(v, 2.0) for v in preds)
        sum2_pow = sum(pow(v, 2.0) for v in labels)
        # sum up the products
        p_sum = sum(preds[i] * labels[i] for i in range(n_pred))

        self.numerator += p_sum - (sum1 * sum2 / n_pred)
        self.denominator += math.sqrt(
            (sum1_pow - pow(sum1, 2) / n_pred) * (sum2_pow - pow(sum2, 2) / n_pred))

    def eval(self):
        """
        Compute and return the PCC.

        Returns:
            - **pcc** (float) - The computed result.

        """
        if self.denominator == 0:
            return 0.0
        pcc = self.numerator / self.denominator
        return pcc

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name
