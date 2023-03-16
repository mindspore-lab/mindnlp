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
""""Class for Metric F1Score"""


import sys
import numpy as np

from mindnlp.abc import Metric
from .utils import _check_onehot_data, _check_shape, _convert_data_type

def f1_score_fn(preds, labels):
    r"""
    Calculates the F1 score. Fbeta score is a weighted mean of precision and recall,
    and F1 score is a special case of Fbeta when beta is 1. The function is shown
    as follows:

    .. math::

        F_1=\frac{2\cdot TP}{2\cdot TP + FN + FP}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases,
    `FP` is the number of false positive cases.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
            of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.

    Returns:
        - **f1_s** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` doesn't have the same classes number as `labels`.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import f1_score
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> labels = Tensor(np.array([1, 0, 1]))
        >>> f1_s = f1_score(preds, labels)
        >>> print(f1_s)
        [0.6666666666666666 0.6666666666666666]

    """
    y_pred = _convert_data_type(preds)
    y_true = _convert_data_type(labels)

    if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
        y_true = y_true.argmax(axis=1)
    _check_shape(y_pred, y_true)

    class_num = y_pred.shape[1]
    if y_true.max() + 1 > class_num:
        raise ValueError(f'`preds` and `labels` should contain same classes, but got `preds` '
                         f'contains {class_num} classes and true value contains '
                         f'{y_true.max() + 1}')
    y_true = np.eye(class_num)[y_true.reshape(-1)]
    indices = y_pred.argmax(axis=1).reshape(-1)
    y_pred = np.eye(class_num)[indices]

    positives = y_pred.sum(axis=0)
    actual_positives = y_true.sum(axis=0)
    true_positives = (y_true * y_pred).sum(axis=0)

    epsilon = sys.float_info.min

    f1_s = 2 * true_positives / (actual_positives + positives + epsilon)
    return f1_s

class F1Score(Metric):
    r"""
    Calculates the F1 score. Fbeta score is a weighted mean of precision and recall,
    and F1 score is a special case of Fbeta when beta is 1. The function is shown
    as follows:

    .. math::

        F_1=\frac{2\cdot TP}{2\cdot TP + FN + FP}

    where `TP` is the number of true posistive cases, `FN` is the number of false negative cases,
    `FP` is the number of false positive cases.

    Args:
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.engine.metrics import F1Score
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> labels = Tensor(np.array([1, 0, 1]))
        >>> metric = F1Score()
        >>> metric.update(preds, labels)
        >>> f1_s = metric.eval()
        >>> print(f1_s)
        [0.6666666666666666 0.6666666666666666]

    """
    def __init__(self, name='F1Score'):
        super().__init__()
        self._name = name
        self.epsilon = sys.float_info.min
        self._true_positives = 0
        self._actual_positives = 0
        self._positives = 0
        self._class_num = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self._true_positives = 0
        self._actual_positives = 0
        self._positives = 0
        self._class_num = 0

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
            ValueError: class numbers of last input predicted data and current
                predicted data not match.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `F1Score.update`, it needs 2 inputs (`preds` and `labels`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        y_pred = _convert_data_type(preds)
        y_true = _convert_data_type(labels)

        if y_pred.ndim == y_true.ndim and _check_onehot_data(y_true):
            y_true = y_true.argmax(axis=1)
        _check_shape(y_pred, y_true)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError(f'For `F1Score.update`, class number not match, last input '
                             f'predicted data contain {self._class_num} classes, but '
                             f'current predicted data contain {y_pred.shape[1]} classes,'
                             f' please check your predicted value(`preds`).')
        class_num = self._class_num

        if y_true.max() + 1 > class_num:
            raise ValueError(f'For `F1Score.update`, `preds` and `labels` should contain '
                             f'same classes, but got `preds` contains {class_num} classes '
                             f'and true value contains {y_true.max() + 1}')
        y_true = np.eye(class_num)[y_true.reshape(-1)]
        indices = y_pred.argmax(axis=1).reshape(-1)
        y_pred = np.eye(class_num)[indices]

        positives = y_pred.sum(axis=0)
        actual_positives = y_true.sum(axis=0)
        true_positives = (y_true * y_pred).sum(axis=0)

        self._true_positives += true_positives
        self._positives += positives
        self._actual_positives += actual_positives

    def eval(self):
        """
        Computes and returns the F1 score.

        Returns:
            - **f1_s** (numpy.ndarray) - The computed result.

        Raises:
            RuntimeError: If the number of samples is 0.

        """
        f1_s = 2 * self._true_positives / (self._actual_positives + self._positives + \
            self.epsilon)
        return f1_s

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name

__all__ = ['f1_score_fn', 'F1Score']
