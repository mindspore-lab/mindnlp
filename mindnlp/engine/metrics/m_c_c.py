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
""""Class for Metric MCC"""


import numpy as np

from mindnlp.abc import Metric
from mindnlp.common.metrics import _convert_data_type


class MCC(Metric):
    r"""
    calculates Matthews correlation coefficient (MCC). MCC is in essence a correlation coefficient
    between the observed and predicted binary classifications; it returns a value between −1 and +1.
    A coefficient of +1 represents a perfect prediction, 0 no better than random prediction and
    −1 indicates total disagreement between prediction and observation. The function is shown
    as follows:

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
        >>> from mindnlp.common.metrics import MCC
        >>> preds = [[0.1, 0.9], [-0.5, 0.5], [0.1, 0.4], [0.1, 0.3]]
        >>> labels = [[1], [0], [1], [1]]
        >>> metric = MCC()
        >>> metric.update(cand_list, ref_list)
        >>> m_c_c = metric.eval()
        >>> print(m_c_c)
        0.0

    """
    def __init__(self, name='MCC'):
        super().__init__()
        self._name = name
        self.t_p = 0
        self.f_p = 0
        self.t_n = 0
        self.f_n = 0

    def clear(self):
        """Clear the internal evaluation result."""
        self.t_p = 0
        self.f_p = 0
        self.t_n = 0
        self.f_n = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.
                    preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is
                        a list of floating numbers in range :math:`[0, 1]` and the shape of
                        `preds` is :math:`(N, C)` in most cases (not strictly), where :math:`N`
                        is the number of cases and :math:`C` is the number of categories.
                    labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels`
                        must be in one-hot format that shape is :math:`(N, C)`, or can be
                        transformed to one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `MCC.update`, it needs 2 inputs (`preds` and `labels`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        preds = _convert_data_type(preds)
        labels = _convert_data_type(labels)

        if preds.ndim not in (labels.ndim, labels.ndim + 1):
            raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension '
                             f'of `preds` equals the dimension of `labels` add 1, but got '
                             f'predicted value ndim: {preds.ndim}, true value ndim: {labels.ndim}.')

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
        Return the name of the metric.
        """
        return self._name
