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
""""Class for Metric Perplexity"""

import math
import numpy as np
from mindspore import Tensor
from mindnlp.abc import Metric
from .utils import _check_value_type, _convert_data_type, _check_onehot_data


def perplexity_fn(preds, labels, ignore_label=None):
    r"""
    Calculates the perplexity. Perplexity is a measure of how well a probabilibity model
    predicts a sample. A low perplexity indicates the model is good at predicting the
    sample. The function is shown as follows:

    .. math::

        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    where :math:`w` represents words in corpus.

    Args:
        preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list
            of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
            :math:`(N, C)` in most cases (not strictly), where :math:`N` is the
            number of cases and :math:`C` is the number of categories.
        labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
            one-hot format that shape is :math:`(N, C)`, or can be transformed to
            one-hot format that shape is :math:`(N,)`.
        ignore_label (Union[int, None]): Index of an invalid label to be ignored
            when counting. If set to `None`, it means there's no invalid label.
            Default: None.

    Returns:
        - **ppl** (float) - The computed result.

    Raises:
        RuntimeError: If `preds` and `labels` have different lengths.
        RuntimeError: If `pred` and `label` have different shapes.
        RuntimeError: If the sample size is 0.

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import perplexity
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 0, 1]), mindspore.int32)
        >>> ppl = perplexity(preds, labels, ignore_label=None)
        >>> print(ppl)
        2.231443166940565

    """
    if ignore_label is not None:
        ignore_label = _check_value_type("ignore_label", ignore_label, [int])

    preds = _check_value_type("preds", preds, [Tensor, list, np.ndarray])
    labels = _check_value_type("labels", labels, [Tensor, list, np.ndarray])

    y_pred = [_convert_data_type(preds)]
    y_true = [_convert_data_type(labels)]

    if len(y_pred) != len(y_true):
        raise RuntimeError(f'`preds` and `labels` should have the same length, but got `preds` '
                           f'length {len(y_pred)}, `labels` length {len(y_true)})')

    sum_cross_entropy = 0.0
    sum_word_num = 0

    cross_entropy = 0.
    word_num = 0
    for label, pred in zip(y_true, y_pred):
        if pred.ndim == label.ndim and _check_onehot_data(label):
            label = label.argmax(axis=1)

        if label.size != pred.size / pred.shape[-1]:
            raise RuntimeError(f'`preds` and `labels` should have the same shape, but got `preds` '
                               f'shape {pred.shape}, label shape {label.shape}.')
        label = label.reshape((label.size,))
        label_expand = label.astype(int)
        label_expand = np.expand_dims(label_expand, axis=1)
        first_indices = np.arange(label_expand.shape[0])[:, None]
        pred = np.squeeze(pred[first_indices, label_expand])
        if ignore_label is not None:
            ignore = (label == ignore_label).astype(pred.dtype)
            word_num -= np.sum(ignore)
            pred = pred * (1 - ignore) + ignore
        cross_entropy -= np.sum(np.log(np.maximum(1e-10, pred)))
        word_num += pred.size
    sum_cross_entropy += cross_entropy
    sum_word_num += word_num

    if sum_word_num == 0:
        raise RuntimeError(f'Perplexity can not be calculated, because the number of samples is '
                           f'{0}')

    ppl = math.exp(sum_cross_entropy / sum_word_num)

    return ppl


class Perplexity(Metric):
    r"""
    Calculates the perplexity. Perplexity is a measure of how well a probabilibity model
    predicts a sample. A low perplexity indicates the model is good at predicting the
    sample. The function is shown as follows:

    .. math::

        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    Where :math:`w` represents words in corpus.

    Args:
        ignore_label (Union[int, None]): Index of an invalid label to be ignored when counting.
            If set to `None`, it means there's no invalid label. Default: None.
        name (str): Name of the metric.

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindnlp.common.metrics import Perplexity
        >>> preds = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> labels = Tensor(np.array([1, 0, 1]))
        >>> metric = Perplexity()
        >>> metric.update(preds, labels)
        >>> ppl = metric.eval()
        >>> print(ppl)
        2.231443166940565

    """
    def __init__(self, ignore_label=None, name='Perplexity'):
        super().__init__()
        self._name = name

        if ignore_label is not None:
            self.ignore_label = _check_value_type("ignore_label", ignore_label, [int])
        else:
            self.ignore_label = None

        self.sum_cross_entropy = 0.0
        self.sum_word_num = 0

    def clear(self):
        """Clears the internal evaluation results."""
        self.sum_cross_entropy = 0.0
        self.sum_word_num = 0

    def update(self, *inputs):
        """
        Updates local variables.

        Args:
            inputs: Input `preds` and `labels`.

                - preds (Union[Tensor, list, np.ndarray]): Predicted value. `preds` is a list
                  of floating numbers in range :math:`[0, 1]` and the shape of `preds` is
                  :math:`(N, C)` in most cases (not strictly), where :math:`N` is the
                  number of cases and :math:`C` is the number of categories.
                - labels (Union[Tensor, list, np.ndarray]): Ground truth. `labels` must be in
                  one-hot format that shape is :math:`(N, C)`, or can be transformed to
                  one-hot format that shape is :math:`(N,)`.

        Raises:
            ValueError: If the number of `inputs` is not 2.
            RuntimeError: If `preds` and `labels` have different lengths.
            RuntimeError: If `pred` and `label` have different shapes.

        """
        if len(inputs) != 2:
            raise ValueError(f'For `Perplexity.update`, it needs 2 inputs (`preds` and `labels`), '
                             f'but got {len(inputs)}.')

        preds = inputs[0]
        labels = inputs[1]

        preds = _check_value_type("preds", preds, [Tensor, list, np.ndarray])
        labels = _check_value_type("labels", labels, [Tensor, list, np.ndarray])

        y_pred = [_convert_data_type(preds)]
        y_true = [_convert_data_type(labels)]

        if len(y_pred) != len(y_true):
            raise RuntimeError(f'For `Perplexity.update`, `preds` and `labels` should have '
                               f'the same length, but got `preds` length {len(y_pred)}, '
                               f'`labels` length {len(y_true)})')

        cross_entropy = 0.
        word_num = 0
        for label, pred in zip(y_true, y_pred):
            if pred.ndim == label.ndim and _check_onehot_data(label):
                label = label.argmax(axis=1)

            if label.size != pred.size / pred.shape[-1]:
                raise RuntimeError(f'For `Perplexity.update`, `preds` and `labels` should have '
                                   f'the same shape, but got `preds` shape {pred.shape}, label '
                                   f'shape {label.shape}.')

            label = label.reshape((label.size,))
            label_expand = label.astype(int)
            label_expand = np.expand_dims(label_expand, axis=1)
            first_indices = np.arange(label_expand.shape[0])[:, None]
            pred = np.squeeze(pred[first_indices, label_expand])

            if self.ignore_label is not None:
                ignore = (label == self.ignore_label).astype(pred.dtype)
                word_num -= np.sum(ignore)
                pred = pred * (1 - ignore) + ignore

            cross_entropy -= np.sum(np.log(np.maximum(1e-10, pred)))
            word_num += pred.size

        self.sum_cross_entropy += cross_entropy
        self.sum_word_num += word_num

    def eval(self):
        """
        Computes and returns the perplexity.

        Returns:
            - **ppl** (float) - The computed result.

        Raises:
            RuntimeError: If the sample size is 0.

        """
        if self.sum_word_num == 0:
            raise RuntimeError(f'Perplexity can not be calculated, because the number of '
                               f'samples is {0}')

        ppl = np.exp(self.sum_cross_entropy / self.sum_word_num)

        return ppl

    def get_metric_name(self):
        """
        Return the name of the metric.
        """
        return self._name

__all__ = ['perplexity_fn', 'Perplexity']
