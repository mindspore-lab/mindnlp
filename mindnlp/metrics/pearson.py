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
""""Class for Metric PearsonCorrelation"""


import math
import numpy as np

from mindnlp.abc import Metric
from .utils import _convert_data_type


def pearson_correlation_fn(preds, labels):
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


class PearsonCorrelation(Metric):
    r"""
    Calculates the Pearson correlation coefficient (PCC). PCC is a measure of linear
    correlation between two sets of data. It is the ratio between the covariance of
    two variables and the product of their standard deviations; thus, it is essentially
    a normalized measurement of the covariance, such that the result always has a value
    between −1 and 1.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.engine.metrics import PearsonCorrelation
        >>> preds = Tensor(np.array([[0.1], [1.0], [2.4], [0.9]]), mindspore.float32)
        >>> labels = Tensor(np.array([[0.0], [1.0], [2.9], [1.0]]), mindspore.float32)
        >>> metric = PearsonCorrelation()
        >>> metric.update(preds, labels)
        >>> p_c_c = metric.eval()
        >>> print(p_c_c)
        0.9985229081857804

    """
    def __init__(self, name='PearsonCorrelation'):
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
            raise ValueError(f'For `PearsonCorrelation.update`, it needs 2 inputs (`preds` '
                             f'and `labels`), but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        y_pred = _convert_data_type(preds)
        y_true = _convert_data_type(labels)

        y_pred = np.squeeze(y_pred.reshape(-1, 1)).tolist()
        y_true = np.squeeze(y_true.reshape(-1, 1)).tolist()

        if len(y_pred) != len(y_true):
            raise RuntimeError(f'For `PearsonCorrelation.update`, `preds` and `labels` should have '
                               f'the same length, but got `preds` length {len(y_pred)}, `labels` '
                               f'length {len(y_true)})')

        self.preds.append(y_pred)
        self.labels.append(y_true)

    def eval(self):
        """
        Computes and returns the PCC.

        Returns:
            - **p_c_c** (float) - The computed result.

        """
        preds = [item for pred in self.preds for item in pred]
        labels = [item for label in self.labels for item in label]

        n_preds = len(preds)

        # simple sums
        sum1 = sum(float(preds[i]) for i in range(n_preds))
        sum2 = sum(float(labels[i]) for i in range(n_preds))

        # sum up the squares
        sum1_pow = sum(pow(v, 2.0) for v in preds)
        sum2_pow = sum(pow(v, 2.0) for v in labels)

        # sum up the products
        p_sum = sum(preds[i] * labels[i] for i in range(n_preds))

        numerator = p_sum - (sum1 * sum2 / n_preds)
        denominator = math.sqrt(
            (sum1_pow - pow(sum1, 2) / n_preds) * (sum2_pow - pow(sum2, 2) / n_preds))

        if denominator == 0:
            return 0.0
        p_c_c = numerator / denominator
        return p_c_c

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['pearson_correlation_fn', 'PearsonCorrelation']
