# Copyright 2023 Huawei Technologies Co., Ltd
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
# pylint: disable=C0103
"""Dropout"""

from mindspore import nn
from mindspore.common.seed import _get_graph_seed
from mindspore.ops import operations as P

class Dropout(nn.Cell):
    r"""
    Dropout layer for the input.

    Randomly set some elements of the input tensor to zero with probability `p` during training
    using samples from a Bernoulli distribution.

    The outputs are scaled by a factor of :math:`\frac{1}{1-p}` during training so
    that the output layer remains at a similar scale. During inference, this
    layer returns the same tensor as the `x`.

    This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
    over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
    preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Note:
        - Each channel will be zeroed out independently on every construct call.
        - Parameter `p` means the probability of the element of the input tensor to be zeroed.

    Args:
        p (Union[float, int, None]): The dropout rate, greater than or equal to 0 and less than 1.
            E.g. rate=0.9, dropping out 90% of input neurons. Default: 0.5.

    Inputs:
        x (Tensor): The input of Dropout with data type of float16 or float32.
            The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, output tensor with the same shape as the `x`.

    Raises:
        TypeError: If the dtype of `p` is not float or int.
        TypeError: If dtype of `x` is not neither float16 nor float32.
        ValueError: If `p` is not in range [0, 1).
        ValueError: If length of shape of `x` is less than 1.

    Examples:
        >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> net = Dropout(p=0.2)
        >>> net.set_train()
        >>> output = net(x)
        >>> print(output.shape)
        (2, 2, 3)
    """

    def __init__(self, p=0.5):
        """
        Initialize Dropout.
        """

        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1), "
                             f"but got {p}.")
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.dropout = P.Dropout(1.0 - p, seed0, seed1)
        self.p = p

    def construct(self, x):
        """
        Compute Dropout.
        """

        if not self.training or self.p == 0:
            return x

        out, _ = self.dropout(x)
        return out
