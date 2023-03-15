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
""""Class for Metric MatthewsCorrelation"""


import math
import numpy as np

from mindnlp.abc import Metric
from .utils import _convert_data_type, _check_onehot_data, _check_shape


def matthews_correlation_fn(preds, labels):
    r"""
    Calculates the Matthews correlation coefficient (MCC). MCC is in essence a correlation
    coefficient between the observed and predicted binary classifications; it returns a value
    between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than
    random prediction and −1 indicates total disagreement between prediction and observation.
    The function is shown as follows:

    .. math::

        MCC=\frac{TP \times TN-FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    where `TP` is the number of true posistive cases, `TN` is the number of true negative cases,
    `FN` is the number of false negative cases, `FP` is the number of false positive cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers and the shape of `preds` is :math:`(N, C)` in most cases
            (not strictly), where :math:`N` is the number of cases and :math:`C` is the
            number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **m_c_c** (float) - The computed result.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import matthews_correlation
        >>> preds = [[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]
        >>> labels = [0, 1, 0, 1, 0]
        >>> m_c_c = matthews_correlation(preds, labels)
        >>> print(m_c_c)
        0.16666666666666666

    """
    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim == labels.ndim and _check_onehot_data(labels):
        labels = labels.argmax(axis=1)
    _check_shape(preds, labels)

    preds = np.argmax(preds, axis=1)
    labels = labels.reshape(-1, 1)

    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0

    sample_num = labels.shape[0]
    for i in range(sample_num):
        pred = preds[i]
        label = labels[i]
        if pred == 1:
            if pred == label:
                t_p += 1
            else:
                f_p += 1
        else:
            if pred == label:
                t_n += 1
            else:
                f_n += 1

    if t_p == 0 or f_p == 0 or t_n == 0 or f_n == 0:
        m_c_c = 0.0
    else:
        m_c_c = (t_p * t_n - f_p * f_n) / math.sqrt(
            (t_p + f_p) * (t_p + f_n) *
            (t_n + f_p) * (t_n + f_n))
    return m_c_c


class MatthewsCorrelation(Metric):
    r"""
    Calculates the Matthews correlation coefficient (MCC). MCC is in essence a correlation
    coefficient between the observed and predicted binary classifications; it returns a value
    between −1 and +1. A coefficient of +1 represents a perfect prediction, 0 no better than
    random prediction and −1 indicates total disagreement between prediction and observation.
    The function is shown as follows:

    .. math::

        MCC=\frac{TP \times TN-FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

    where `TP` is the number of true posistive cases, `TN` is the number of true negative cases,
    `FN` is the number of false negative cases, `FP` is the number of false positive cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.engine.metrics import MatthewsCorrelation
        >>> preds = [[0.8, 0.2], [-0.5, 0.5], [0.1, 0.4], [0.6, 0.3], [0.6, 0.3]]
        >>> labels = [0, 1, 0, 1, 0]
        >>> metric = MatthewsCorrelation()
        >>> metric.update(preds, labels)
        >>> m_c_c = metric.eval()
        >>> print(m_c_c)
        0.16666666666666666

    """
    def __init__(self, name='MatthewsCorrelation'):
        super().__init__()
        self._name = name
        self.t_p = 0
        self.f_p = 0
        self.t_n = 0
        self.f_n = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self.t_p = 0
        self.f_p = 0
        self.t_n = 0
        self.f_n = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list of
                  floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                  :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number of
                  cases and :math:`C` is the number of categories.
                - labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must be in
                  one-hot format that shape is :math:`(N, C)`, or can be transformed to one-hot
                  format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `MatthewsCorrelation.update`, it needs 2 inputs '
                             f'(`preds` and `labels`), but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        preds = _convert_data_type(preds)
        labels = _convert_data_type(labels)

        if preds.ndim == labels.ndim and _check_onehot_data(labels):
            labels = labels.argmax(axis=1)
        _check_shape(preds, labels)

        preds = np.argmax(preds, axis=1)
        labels = labels.reshape(-1, 1)

        sample_num = labels.shape[0]
        for i in range(sample_num):
            pred = preds[i]
            label = labels[i]
            if pred == 1:
                if pred == label:
                    self.t_p += 1
                else:
                    self.f_p += 1
            else:
                if pred == label:
                    self.t_n += 1
                else:
                    self.f_n += 1

    def eval(self):
        """
        Computes and returns the MCC.

        Returns:
            - **m_c_c** (float) - The computed result.

        """
        if self.t_p == 0 or self.f_p == 0 or self.t_n == 0 or self.f_n == 0:
            m_c_c = 0.0
        else:
            m_c_c = (self.t_p * self.t_n - self.f_p * self.f_n) / math.sqrt(
                (self.t_p + self.f_p) * (self.t_p + self.f_n) *
                (self.t_n + self.f_p) * (self.t_n + self.f_n))
        return m_c_c

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['matthews_correlation_fn', 'MatthewsCorrelation']
