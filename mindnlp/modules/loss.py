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
# pylint: disable=arguments-differ
# pylint: disable=C0412
"""Losses"""

import numpy as np
import mindspore
from mindspore import nn, ops, Tensor
from mindnlp._legacy.functional import softmax, kl_div, masked_select

def _inner_log_softmax(inputs, axis):
    """inner implementation of log_softmax, since the LogSoftmaxGrad op do not support inputs > 2d"""
    return inputs - ops.logsumexp(inputs, axis, True)

def sequence_mask(lengths, maxlen):
    """generate mask matrix by seq_length"""
    length_dtype = lengths.dtype
    range_vector = Tensor(np.arange(0, maxlen, 1), length_dtype)
    result = range_vector < lengths.view(lengths.shape + (1,))
    return result.astype(mindspore.float32)


class RDropLoss(nn.Cell):
    """
    R-Drop Loss implementation
    For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448

    Original implementation please refer to this code: https://github.com/dropreg/R-Drop

    Args:
        reduction(str): Indicate how to average the loss, the candicates are "none",
            "batchmean","mean","sum". Default: "none".

            - "mean": The reduced mean loss is returned.
            - "batchmean": The sum loss divided by batch size is returned.
            - "sum": The reduced sum loss is returned.
            - "none": No reduction will be applied.
    """

    def __init__(self, reduction='none'):
        super().__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                f"'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', "
                f"but received {reduction}.")
        self.reduction = reduction

    def construct(self, p, q, pad_mask=None):
        """
        Returns loss tensor, the rdrop loss of p and q.

        Args:
            p (Tensor): The first forward logits of training examples.
            q (Tensor): The second forward logits of training examples.
            pad_mask(Tensor): The Tensor containing the binary mask to index with,
                it's data type is bool. Default: None.

        Returns:
            Tensor, the rdrop loss of p and q.

        Raises:
            ValueError: if 'reduction' in 'RDropLoss' is not 'sum', 'mean' 'batchmean', or 'none'.

        Examples:
            >>> r_drop_loss = RDropLoss()
            >>> p = Tensor(np.array([1., 0. , 1.]), mindspore.float32)
            >>> q = Tensor(np.array([0.2, 0.3 , 1.1]), mindspore.float32)
            >>> loss = r_drop_loss(p, q)
            >>> print(loss)
            0.100136
        """

        p_loss = kl_div(_inner_log_softmax(p, axis=-1),
                        softmax(q, axis=-1),
                        reduction=self.reduction)
        q_loss = kl_div(_inner_log_softmax(q, axis=-1),
                        softmax(p, axis=-1),
                        reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss = masked_select(p_loss, pad_mask)
            q_loss = masked_select(q_loss, pad_mask)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss


class CMRC2018Loss(nn.Cell):
    r"""
    CMRC2018Loss
    used to compute CMRC2018 chinese Q&A task

    Args:
        reduction(str): Indicate how to average the loss, the candicates are "mean" and "sum".
            Default: "mean".
    """

    def __init__(self, reduction='mean'):
        super().__init__()

        assert reduction in ('mean', 'sum')

        self.reduction = reduction

    def construct(self, target_start, target_end, context_len, pred_start, pred_end):
        """
        compute CMRC2018Loss

        Args:
            target_start (Tensor): size: batch_size, dtype: int.
            target_end (Tensor): size: batch_size, dtype: int.
            context_len (Tensor): size: batch_size, dtype: float.
            pred_start (Tensor): size: batch_size*max_len, dtype: float.
            pred_end (Tensor): size: batch_size*max_len, dtype: float.

        Returns:
            Tensor, the CMRC2018 loss.

        Raises:
            ValueError: if 'reduction' is not 'sum' or 'mean'.

        Example:
            >>> cmrc_loss = CMRC2018Loss()
            >>> tensor_a = mindspore.Tensor(np.array([1, 2, 1]), mindspore.int32)
            >>> tensor_b = mindspore.Tensor(np.array([2, 1, 2]), mindspore.int32)
            >>> my_context_len = mindspore.Tensor(np.array([2., 1., 2.]), mindspore.float32)
            >>> tensor_c = mindspore.Tensor(np.array([
            >>>     [0.1, 0.2, 0.1],
            >>>     [0.1, 0.2, 0.1],
            >>>     [0.1, 0.2, 0.1]
            >>> ]), mindspore.float32)
            >>> tensor_d = mindspore.Tensor(np.array([
            >>>     [0.2, 0.1, 0.2],
            >>>     [0.2, 0.1, 0.2],
            >>>     [0.2, 0.1, 0.2]
            >>> ]), mindspore.float32)
            >>> my_loss = cmrc_loss(tensor_a, tensor_b, my_context_len, tensor_c, tensor_d)
            >>> print(my_loss)
        """

        batch_size, max_len = pred_end.shape

        zero_tensor = mindspore.Tensor(
            np.zeros((batch_size, max_len)), mindspore.float32)

        mask = ops.equal(sequence_mask(context_len, max_len), zero_tensor)

        pred_start = pred_start.masked_fill(mask, -1e10)
        pred_end = pred_end.masked_fill(mask, -1e10)

        start_loss = ops.cross_entropy(pred_start, target_start, reduction='sum')
        end_loss = ops.cross_entropy(pred_end, target_end, reduction='sum')

        loss = start_loss + end_loss

        if self.reduction == 'mean':
            loss = loss / batch_size

        return loss / 2


__all__ = [
    'RDropLoss',
    'CMRC2018Loss'
]
