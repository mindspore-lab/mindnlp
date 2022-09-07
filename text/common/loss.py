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
"""Losses"""
import mindspore.ops.functional as F
from mindspore import nn
from mindspore import ops

__all__ = ['RDropLoss']


class RDropLoss(nn.Cell):
    """
    R-Drop Loss implementation
    For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448

    Original implementation please refer to this code: https://github.com/dropreg/R-Drop

    Args:

        reduction(str, optional):
            Indicate how to average the loss, the candicates are ``'none'``,
            ``'batchmean'``,``'mean'``,``'sum'``.
            If `reduction` is ``'mean'``, the reduced mean loss is returned;
            If `reduction` is ``'batchmean'``, the sum loss divided by batch size is returned;
            If `reduction` is ``'sum'``, the reduced sum loss is returned;
            If `reduction` is ``'none'``, no reduction will be applied.
            Defaults to ``'none'``.

    Inputs:

        - **p** (Tensor) -the first forward logits of training examples.

        - **q** (Tensor) -the second forward logits of training examples.

        - **pad_mask** (Tensor, optional) -The Tensor containing the binary mask to index with,
        it's data type is bool.

    Returns:

        - **Tensor** (Tensor) -Returns tensor `loss`, the rdrop loss of P and q.

    Raises:

        ValueError: if 'reduction' in 'RDropLoss' is not 'sum', 'mean' 'batchmean', or 'none'

    Example:

        >>> r_drop_loss = RDropLoss()
        >>> p = Tensor(np.array([1., 0. , 1.]), mindspore.float32)
        >>> q = Tensor(np.array([0.2, 0.3 , 1.1]), mindspore.float32)
        >>> loss = r_drop_loss(p, q)
        >>> print(loss)
        0.100136

    """

    def __init__(self, reduction='none'):
        super(RDropLoss, self).__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "'reduction' in 'RDropLoss' should be 'sum', 'mean' 'batchmean', or 'none', "
                "but received {}.".format(reduction))
        self.reduction = reduction

    def construct(self, p, q, pad_mask=None):
        """
        Returns tensor `loss`, the rdrop loss of p and q.
        """
        p_loss = F.kl_div(F.log_softmax(p, axis=-1),
                          F.softmax(q, axis=-1),
                          reduction=self.reduction)
        q_loss = F.kl_div(F.log_softmax(q, axis=-1),
                          F.softmax(p, axis=-1),
                          reduction=self.reduction)

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss = ops.masked_select(p_loss, pad_mask)
            q_loss = ops.masked_select(q_loss, pad_mask)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss
