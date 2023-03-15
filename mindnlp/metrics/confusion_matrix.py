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
""""Class for Metric ConfusionMatrix"""


import numpy as np

from mindnlp.abc import Metric
from .utils import _check_value_type, _convert_data_type

def confusion_matrix_fn(preds, labels, class_num=2):
    r"""
    Calculates the confusion matrix. Confusion matrix is commonly used to evaluate
    the performance of classification models, including binary classification and
    multiple classification.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
            floating numbers and the shape of `preds` is :math:`(N, C)` or :math:`(N,)`.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. The shape of `labels` is
            :math:`(N,)`.
        class_num (int): Number of classes in the dataset. Default: 2.

    Returns:
        - **conf_mat** (np.ndarray) - The computed result.

    Raises:
        ValueError: If `preds` and `labels` do not have valid dimensions.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import confusion_matrix
        >>> preds = Tensor(np.array([1, 0, 1, 0]))
        >>> labels = Tensor(np.array([1, 0, 0, 1]))
        >>> conf_mat = confusion_matrix(preds, labels)
        >>> print(conf_mat)
        [[1. 1.]
         [1. 1.]]

    """
    class_num = _check_value_type("class_num", class_num, [int])

    preds = _convert_data_type(preds)
    labels = _convert_data_type(labels)

    if preds.ndim not in (labels.ndim, labels.ndim + 1):
        raise ValueError(f'`preds` and `labels` should have the same dimensions, or the dimension '
                         f'of `preds` equals the dimension of true value add 1, but got `preds` '
                         f'ndim: {preds.ndim}, `labels` ndim: {labels.ndim}.')

    if preds.ndim == labels.ndim + 1:
        preds = np.argmax(preds, axis=1)

    trans = (labels.reshape(-1) * class_num + preds.reshape(-1)).astype(int)
    bincount = np.bincount(trans, minlength=class_num ** 2)
    conf_mat = bincount.reshape(class_num, class_num)

    conf_mat = conf_mat.astype(float)

    return conf_mat


class ConfusionMatrix(Metric):
    r"""
    Calculates the confusion matrix. Confusion matrix is commonly used to evaluate
    the performance of classification models, including binary classification and
    multiple classification.

    Args:
        class_num (int): Number of classes in the dataset. Default: 2.
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.engine.metrics import ConfusionMatrix
        >>> preds = Tensor(np.array([1, 0, 1, 0]))
        >>> labels = Tensor(np.array([1, 0, 0, 1]))
        >>> metric = ConfusionMatrix()
        >>> metric.update(preds, labels)
        >>> conf_mat = metric.eval()
        >>> print(conf_mat)
        [[1. 1.]
         [1. 1.]]

    """
    def __init__(self, class_num=2, name='ConfusionMatrix'):
        super().__init__()
        self._name = name
        self.class_num = _check_value_type("class_num", class_num, [int])
        self.conf_mat = np.zeros((self.class_num, self.class_num))

    def clear(self):
        """Clears the internal evaluation results."""
        self.conf_mat = np.zeros((self.class_num, self.class_num))

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list of
                  floating numbers and the shape of `preds` is :math:`(N, C)` or :math:`(N,)`.
                - labels (Union[Tensor, list, np.ndarray]): Ground truth. The shape of `labels` is
                  :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` and `labels` do not have valid dimensions.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `ConfusionMatrix.update`, it needs 2 inputs (`preds` and '
                             f'`labels`), but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        preds = _convert_data_type(preds)
        labels = _convert_data_type(labels)

        if preds.ndim not in (labels.ndim, labels.ndim + 1):
            raise ValueError(f'For `ConfusionMatrix.update`, `preds` and `labels` should have the '
                             f'same dimensions, or the dimension of `preds` equals the dimension '
                             f'of true value add 1, but got `preds` ndim: {preds.ndim}, `labels` '
                             f'ndim: {labels.ndim}.')

        if preds.ndim == labels.ndim + 1:
            preds = np.argmax(preds, axis=1)

        trans = (labels.reshape(-1) * self.class_num + preds.reshape(-1)).astype(int)
        bincount = np.bincount(trans, minlength=self.class_num ** 2)
        conf_mat = bincount.reshape(self.class_num, self.class_num)
        self.conf_mat += conf_mat

    def eval(self):
        """
        Computes and returns the Confusion Matrix.

        Returns:
            - **conf_mat** (np.ndarray) - The computed result.

        """
        conf_mat = self.conf_mat.astype(float)

        return conf_mat

    def get_metric_name(self):
        """
        Returns the name of the metric.
        """
        return self._name


__all__ = ['confusion_matrix_fn', 'ConfusionMatrix']
