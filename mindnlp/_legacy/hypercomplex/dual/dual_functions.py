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
# pylint: disable=invalid-name
"""Dual functions"""
from mindspore.common.tensor import Tensor
from mindspore import ops as P
from ..utils import get_x_and_y, to_2channel


def matmul(z1: Tensor, z2: Tensor) -> Tensor:
    """
    Returns the matrix product of two dual-valued tensors.

    Note:
        Numpy arguments `out`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        The supported dtypes are np.float16, and np.float32

    Args:
        z1 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `z1` must be the same size as the second last dimension of `z2`.
          The shape of the tensor must be :math:`(2, *, ..., *)`. And the shape of z1 and z2 could be broadcast.
        z2 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `z1` must be the same size as the second last dimension of `z2`.
          The shape of the tensor must be :math:`(2, *, ..., *)`. And the shape of z1 and z2 could be broadcast.

    Returns:
        Tensor of shape :math:`(2, *, ..., *)`, the matrix product of the inputs.

    Raises:
        ValueError: If the last dimension of `z1` is not the same size as the
            second-to-last dimension of `z2`, or if a scalar value is passed in.
        ValueError: If the shape of `z1` and `z2` could not broadcast together.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore.hypercomplex.dual import matmul
        >>> from mindspore import Tensor
        >>> z1 = Tensor(np.random.random((2, 3, 4)).astype(np.float32))
        >>> z2 = Tensor(np.random.random((2, 4, 6)).astype(np.float32))
        >>> z = matmul(z1, z2)
        >>> print(z.shape)
        (2, 3, 6)
    """
    z1_re, z1_im = get_x_and_y(z1)
    z2_re, z2_im = get_x_and_y(z2)
    z_re = P.matmul(z1_re, z2_re)
    z_im = P.matmul(z1_re, z2_im) + P.matmul(z1_im, z2_re)
    z = to_2channel(z_re, z_im)
    return z
