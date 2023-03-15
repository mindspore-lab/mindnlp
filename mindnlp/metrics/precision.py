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
""""Class for Metric Precision"""


import sys
import numpy as np

from mindnlp.abc import Metric
from .utils import _check_onehot_data, _check_shape, _convert_data_type


def precision_fn(preds, labels):
    r"""
    Calculates the precision. Precision (also known as positive predictive value) is
    the actual positive proportion in the predicted positive sample. It can only be
    used to evaluate the precision score of binary tasks. The function is shown
    as follows:

    .. math::

        \text{Precision} =\frac{\text{TP}} {\text{TP} + \text{FP}}

    where `TP` is the number of true posistive cases, `FP` is the number of false posistive cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **prec** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import precision
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> prec = precision(preds, labels)
        >>> print(prec)
        [0.5 1. ]

    """
    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)

    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` should have the same classes number as `labels`, but got `preds`'
                         f'classes {class_num}, true value classes {y_true.max() + 1}')

    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    positives = y_pred.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    prec = true_positives / (positives + epsilon)
    return prec


class Precision(Metric):
    r"""
    Calculates precision. Precision (also known as positive predictive value) is the actual
    positive proportion in the predicted positive sample. It can only be used to evaluate
    the precision score of binary tasks. The function is shown as follows:

    .. math::

        \text{Precision} =\frac{\text{TP}} {\text{TP} + \text{FP}}

    where `TP` is the number of true posistive cases, `FP` is the number of false posistive cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> from mindnlp.common.metrics import Precision
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> metric = Precision()
        >>> metric.update(preds, labels)
        >>> prec = metric.eval()
        >>> print(prec)
        [0.5 1. ]

    """
    def __init__(self, name='Precision'):
        super().__init__()
        self._name = name
        self.epsilon = sys.float_info.min
        self.positives = 0
        self.true_positives = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self.positives = 0
        self.true_positives = 0

    def update(self, *inputs):
        """
        Updates local variables. If the index of the maximum of the predicted value matches
        the label, the predicted result is correct.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list
                  of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                  :math:`(N, C)` in most cases (not strictly), where :math:`N` is
                  the number of cases and :math:`C` is the number of categories.
                - labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must
                  be in one-hot format that shape is :math:`(N, C)`, or can be transformed
                  to one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `Precision.update`, it needs 2 inputs (`preds` and `labels`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        y_pred = _convert_data_type(preds)
        y_true = _convert_data_type(labels)

        if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
            y_true = y_true.argmax(axis=1)
        _check_shape(y_pred, y_true)

        class_num = y_pred.shape[1]
        if y_true.max() + 1 > class_num:
            raise ValueError(f'For `Precision.update`, `preds` should have the same classes number '
                             f'as `labels`, but got `preds` classes {class_num}, true value classes'
                             f' {y_true.max() + 1}')

        y_true = np.eye(class_num)[y_true.reshape(-1)]
        indices = y_pred.argmax(axis=1).reshape(-1)
        y_pred = np.eye(class_num)[indices]

        self.positives += y_pred.sum(axis=0)
        self.true_positives += (y_true * y_pred).sum(axis=0)

    def eval(self):
        """
        Computes and returns the precision.

        Returns:
            - **prec** (numpy.ndarray) - The computed result.

        """
        prec = self.true_positives / (self.positives + self.epsilon)
        return prec

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['precision_fn', 'Precision']
