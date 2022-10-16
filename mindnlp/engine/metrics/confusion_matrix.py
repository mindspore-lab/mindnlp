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
from mindnlp.common.metrics import _check_value_type, _convert_data_type


class ConfusionMatrix(Metric):
    r"""
    Calculate confusion matrix. Confusion matrix is commonly used to evaluate the performance
    of classification models, including binary classification and multiple classification.

    Args:
        class_num (int): Number of classes in the dataset. Default: 2.
        name (str): Name of the metric.

    Example:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import ConfusionMatrix
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
        """Clear the internal evaluation result."""
        self.conf_mat = np.zeros((self.class_num, self.class_num))

    def update(self, *inputs):
        """
        Updates local variables. If the index of the maximum of the predicted value matches
        the label, the predicted result is correct.

        Args:
            inputs: Input `preds` and `labels`.
                    preds (Union[Tensor, list, numpy.ndarray]): Predicted value. `preds` is a list
                        of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                        :math:`(N, C)` in most cases (not strictly), where :math:`N` is the number
                        of cases and :math:`C` is the number of categories.
                    labels (Union[Tensor, list, numpy.ndarray]): Ground truth value. `labels` must
                        be in one-hot format that shape is :math:`(N, C)`, or can be transformed to
                        one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of inputs is not 2.
            ValueError: If `preds` doesn't have the same classes number as `labels`.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `ConfusionMatrix.update`, it needs 2 inputs (`preds` and '
                             f'`labels`), but got {len(inputs)}.')
        preds = inputs[0]
        labels = inputs[1]

        preds = _convert_data_type(preds)
        labels = _convert_data_type(labels)

        if preds.ndim not in (labels.ndim, labels.ndim + 1):
            raise ValueError(f'`preds` and `labels` should have same dimensions, or the dimension'
                             f' of `preds` equals the dimension of `labels` add 1, but got '
                             f'predicted value ndim: {preds.ndim}, true value ndim: {labels.ndim}.')

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
            - **conf_mat** (float) - The computed result.

        """
        conf_mat = self.conf_mat.astype(float)

        return conf_mat

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name
