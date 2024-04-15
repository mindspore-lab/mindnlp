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
""""Class for Metric Accuracy"""


import numpy as np

from mindnlp.abc import Metric
from .utils import _check_onehot_data, _check_shape, _convert_data_type

def accuracy_fn(preds, labels):
    r"""
    Calculates the accuracy. The function is shown as follows:

    .. math::

        \text{ACC} =\frac{\text{TP} + \text{TN}}
        {\text{TP} + \text{TN} + \text{FP} + \text{FN}}

    where `ACC` is accuracy, `TP` is the number of true posistive cases, `TN` is the number
    of true negative cases, `FP` is the number of false posistive cases, `FN` is the number
    of false negative cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **acc** (float) - The computed result.

    Raises:
        RuntimeError: If the number of samples is 0.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import accuracy
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> acc = accuracy(preds, labels)
        >>> print(acc)
        0.6666666666666666

    """
    correct_num = 0
    total_num = 0

    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)

    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    indices = y_pred.argmax(axis=1)
    result = (np.equal(indices, y_true) * 1).reshape(-1)

    correct_num += result.sum()
    total_num += result.shape[0]

    if total_num == 0:
        raise RuntimeError(f'Accuracy can not be calculated, because the number of samples is '
                           f'{0}. Please check whether your inputs(predicted value, true value) '
                           f'are empty.')
    acc = correct_num / total_num
    return acc

class Accuracy(Metric):
    r"""
    Calculates accuracy. The function is shown as follows:

    .. math::

        \text{ACC} =\frac{\text{TP} + \text{TN}}
        {\text{TP} + \text{TN} + \text{FP} + \text{FN}}

    where `ACC` is accuracy, `TP` is the number of true posistive cases, `TN` is the number
    of true negative cases, `FP` is the number of false posistive cases, `FN` is the number
    of false negative cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import nn, Tensor
        >>> from mindnlp.common.metrics import Accuracy
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> metric = Accuracy()
        >>> metric.update(preds, labels)
        >>> acc = metric.eval()
        >>> print(acc)
        0.6666666666666666

    """
    def __init__(self, name='Accuracy'):
        super().__init__()
        self._name = name
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self._correct_num = 0
        self._total_num = 0
        self._class_num = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list
                  of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                  :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
                  of cases and :math:`C` is the number of categories.
                - labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must
                  be in one-hot format that shape is :math:`(N, C)`, or can be transformed to
                  one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of `inputs` is not 2.
            ValueError: class numbers of last input predicted data and current predicted data
                not match.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `Accuracy.update`, it needs 2 inputs (`preds` and `labels`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        y_pred = _convert_data_type(preds)
        y_true = _convert_data_type(labels)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError(f'For `Accuracy.update`, class numbers do not match. Last input '
                             f'predicted data contain {self._class_num} classes, but current '
                             f'predicted data contain {y_pred.shape[1]} classes. Please check '
                             f'your predicted value (`preds`).')

        if self._class_num != 1 and y_pred.ndim == y_true.ndim and \
                (_check_onehot_data(y_true) or y_true[0].shape == (1,)):
            y_true = y_true.argmax(axis=1)

        _check_shape(y_pred, y_true, self._class_num)

        if self._class_num == 1:
            indices = np.around(y_pred)
        else:
            indices = y_pred.argmax(axis=1)

        res = (np.equal(indices, y_true) * 1).reshape(-1)

        self._correct_num += res.sum()
        self._total_num += res.shape[0]

    def eval(self):
        """
        Computes and returns the accuracy.

        Returns:
            - **acc** (float) - The computed result.

        Raises:
            RuntimeError: If the number of samples is 0.

        """
        if self._total_num == 0:
            raise RuntimeError(f'Accuracy can not be calculated, because the number of samples is'
                               f' {0}, please check whether your inputs(`preds`, `labels`) are '
                               f'empty, or you have called update method before calling eval '
                               f'method.')
        acc = self._correct_num / self._total_num
        return acc

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['accuracy_fn', 'Accuracy']
