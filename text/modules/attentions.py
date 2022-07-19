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
"""attention module"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp


class DotAttention(nn.Cell):
    r"""
    Scaled Dot-Product Attention

      .. math::

          Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V

    Args:
        dropout (float): The keep rate, greater than 0 and less equal than 1.
            E.g. rate=0.9, dropping out 10% of input units. Default: 0.9.

    Inputs:
        - **query** (Tensor) - The query vector.
        - **key** (Tensor) - The key vector.
        - **value** (Tensor) - The value vector.

    Returns:
        - **output** (Tensor) - The output of the attention.

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.text.modules.attentions import DotAttention
        >>> model = DotAttention(dropout=0.9)
        >>> q = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> k = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> v = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> output = model(q, k, v)
        >>> print(output.shape)
        (2, 1024, 512)
    """

    def __init__(self, dropout=0.9):
        super(DotAttention, self).__init__()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, query, key, value):
        scale = mnp.sqrt(ops.scalar_to_tensor(query.shape[-1]))
        scores = ops.matmul(query, mnp.swapaxes(key, -1, -2)) / scale
        attn_weights = self.softmax(scores)
        attn_weights, _ = self.dropout(attn_weights)
        output = ops.matmul(attn_weights, value)
        return output
