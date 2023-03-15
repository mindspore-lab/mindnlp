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
""""Class for Metric Recall"""


import sys
import numpy as np

from mindnlp.abc import Metric
from .utils import _check_onehot_data, _check_shape, _convert_data_type

def recall_fn(preds, labels):
    r"""
    Calculates the recall. Recall is also referred to as the true positive rate or
    sensitivity. The function is shown as follows:

    .. math::

        \text{Recall} =\frac{\text{TP}} {\text{TP} + \text{FN}}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **rec** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import recall
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> rec = recall(preds, labels)
        >>> print(rec)
        [1. 0.5]

    """
    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)

    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` should have the same classes number as `labels`, but got `preds`'
                         f' classes {class_num}, true value classes {y_true.max() + 1}.')
    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    actual_positives = y_true.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    rec = true_positives / (actual_positives + epsilon)
    return rec


class Recall(Metric):
    r"""
    Calculates the recall. Recall is also referred to as the true positive rate or
    sensitivity. The function is shown as follows:

    .. math::

        \text{Recall} =\frac{\text{TP}} {\text{TP} + \text{FN}}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import Recall
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> metric = Recall()
        >>> metric.update(preds, labels)
        >>> rec = metric.eval()
        >>> print(rec)
        [1. 0.5]

    """
    def __init__(self, name='Recall'):
        super().__init__()
        self._name = name
        self.actual_positives = 0
        self.true_positives = 0
        self.epsilon = sys.float_info.min

    def clear(self):
        """Clears the internal evaluation results."""
        self.actual_positives = 0
        self.true_positives = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
                  floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                  :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
                  of cases and :math:`C` is the number of categories.
                - labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
                  one-hot format that shape is :math:`(N, C)`, or can be transformed to
                  one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `Recall.update`, it needs 2 inputs (`preds` and `labels`), '
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
            raise ValueError(f'For `Recall.update`, `preds` should have the same classes number '
                             f'as `labels`, but got `preds` classes {class_num}, true value classes'
                             f' {y_true.max() + 1}.')
        y_true = np.eye(class_num)[y_true.reshape(-1)]
        indices = y_pred.argmax(axis=1).reshape(-1)
        y_pred = np.eye(class_num)[indices]

        self.actual_positives += y_true.sum(axis=0)
        self.true_positives += (y_true * y_pred).sum(axis=0)

    def eval(self):
        """
        Computes and returns the recall.

        Returns:
            - **rec** (numpy.ndarray) - The computed result.

        """
        rec = self.true_positives / (self.actual_positives + self.epsilon)
        return rec

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['recall_fn', 'Recall']
