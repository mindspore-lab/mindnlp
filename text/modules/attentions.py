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
from typing import Optional
import mindspore

import mindspore.ops as ops
import mindspore.nn as nn
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
        - **query** (mindspore.Tensor) - The query vector.
        - **key** (mindspore.Tensor) - The key vector.
        - **value** (mindspore.Tensor) - The value vector.
        - **mask** Optional[mindspore.Tensor[bool]] - The mask vector.

    Returns:
        - **output** (mindspore.Tensor) - The output of the attention.
        - **attn** (mindspore.Tensor) - The last layer of attention weights

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.text.modules.attentions import DotAttention
        >>> model = DotAttention(dropout=0.9)
        >>> q = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> k = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> v = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> output, _ = model(q, k, v)
        >>> print(output.shape)
        (2, 1024, 512)
    """

    def __init__(self, dropout=0.9):
        super(DotAttention, self).__init__()
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        scale = mnp.sqrt(ops.scalar_to_tensor(query.shape[-1]))
        scores = ops.matmul(query, mnp.swapaxes(key, -1, -2)) / scale
        if mask is not None:
            scores = ops.masked_fill(scores, mask == 0, -1e9)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        output = ops.matmul(attn_weights, value)
        return output, attn_weights

def _masked_softmax(tensor, mask):
    """
    Calculate the softmax weight of tensor under mask.
    """

    softmax = ops.Softmax()
    tensor_shape = tensor.shape
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    while mask.ndim < tensor.ndim:
        mask = ops.expand_dims(mask, 1)
    mask = mask.expand_as(tensor)
    mask_shape = mask.shape
    reshaped_mask = mask.view(-1, mask_shape[-1])
    result = softmax(reshaped_tensor * reshaped_mask)
    result = result * reshaped_tensor
    #To avoid the divisions by zeros case
    result = result / (result.sum(axis=-1, keepdims=True) + 1e-13)
    return result.view(tensor_shape)

def _weighted_sum(tensor, weights, mask):
    """
    Calculate the weighted sum of tensor and weight under mask.
    """

    bmm = ops.BatchMatMul()
    w_sum = bmm(weights, tensor)
    while mask.ndim < tensor.ndim:
        mask = ops.expand_dims(mask, 1)
    mask = mnp.swapaxes(mask, -1, -2)
    mask = mask.expand_as(w_sum)
    return w_sum * mask

class BinaryAttention(nn.Cell):
    r"""
    Binary Attention, For a given sequence of two vectors :
    x_i and y_j, the BiAttention module will
    compute the attention result by the following equation:

      .. math::

          \begin{array}{ll} \\
            e_{ij} = {x}^{\mathrm{T}}_{i}{y}_{j} \\
            {\hat{x}}_{i} = \sum_{j=1}^{\mathcal{l}_{y}}{\frac{
                \mathrm{exp}(e_{ij})}{\sum_{k=1}^{\mathcal{l}_{y}}{\mathrm{exp}(e_{ik})}}}{y}_{j} \\
            {\hat{y}}_{j} = \sum_{i=1}^{\mathcal{l}_{x}}{\frac{
                \mathrm{exp}(e_{ij})}{\sum_{k=1}^{\mathcal{l}_{x}}{\mathrm{exp}(e_{ik})}}}{x}_{i} \\
        \end{array}

    Args:
        x_batch (mindspore.Tensor): [batch_size, x_seq_len, hidden_size]
        x_mask (mindspore.Tensor): [batch_size, x_seq_len]
        y_batch (mindspore.Tensor): [batch_size, y_seq_len, hidden_size]
        y_mask (mindspore.Tensor): [batch_size, y_seq_len]

    Returns:
        - attended_x (mindspore.Tensor) - The output of the attention_x.
        - attended_y (mindspore.Tensor) - The output of the attention_y.

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> from mindspore.text.modules.attentions import BinaryAttention
        >>> model = BinaryAttention()
        >>> standard_normal = ops.StandardNormal(seed=114514)
        >>> x = standard_normal((2, 30, 512))
        >>> y = standard_normal((2, 20, 512))
        >>> x_mask = Tensor(np.zeros_like(x.shape[:-1]), mindspore.float32)
        >>> y_mask = Tensor(np.zeros_like(y.shape[:-1]), mindspore.float32)
        >>> output_x, output_y = model(x, x_mask, y, y_mask)
        >>> print(output_x.shape, output_y.shape)
        (2, 30, 512) (2, 20, 512)
    """

    def __init__(self):
        super(BinaryAttention, self).__init__()
        self.bmm = ops.BatchMatMul()

    def construct(self, x_batch, x_mask, y_batch, y_mask):
        similarity_matrix = self.bmm(x_batch, mnp.swapaxes(y_batch, 2, 1))
        x_y_attn = _masked_softmax(similarity_matrix, y_mask)
        y_x_attn = _masked_softmax(mnp.swapaxes(similarity_matrix, 1, 2), x_mask)
        attended_x = _weighted_sum(y_batch, x_y_attn, x_mask)
        attended_y = _weighted_sum(x_batch, y_x_attn, y_mask)
        return attended_x, attended_y

class MutiHeadAttention(nn.Cell):
    r"""
    Muti-head attention is from the paper “attention is all you need”
    where heads == 1 Muti-head attention is normal self-attention

    Args:
        - **head** (int) - The number of head. Default: 8.
        - **d_model** (int) - The `query`, `key` and `value` vectors dimensions. Default: 512.
        - **dropout** (float): The keep rate, greater than 0 and less equal than 1. Default: 0.9.
        - **bias** (bool) - whether to use a bias vector. Default: True.
        - **attention_mode** (String) - Attention mode. Default: "dot product".

    Inputs:
        - **query** (mindspore.Tensor) - The query vector.
        - **key** (mindspore.Tensor) - The key vector.
        - **value** (mindspore.Tensor) - The value vector. [seq_len, batch_size, d_model]
        - **mask** Optional[mindspore.Tensor[bool]] - The mask vector. [seq_len, seq_len, batch_size]

    Returns:
        - output (mindspore.Tensor) - The output of muti-head attention.
        - attn (mindspore.Tensor) - The last layer of attention weights

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> from mindspore.text.modules.attentions import MutiHeadAttention
        >>> standard_normal = ops.StandardNormal(seed=114514)
        >>> # query is [batch_size, seq_len_q, hidden_size]
        >>> q = standard_normal((2, 32, 512))
        >>> # key is [batch_size, seq_len_k, hidden_size]
        >>> k = standard_normal((2, 20, 512))
        >>> # value is [batch_size, seq_len_k, hidden_size]
        >>> v = standard_normal((2, 20, 512))
        >>> # query shape is (2, 32 ,512)->(2, 8, 32, 64) and key shape is (2, 20 ,512)->(2, 8, 20, 64)
        >>> # query * key.transpose(-1, -2): (2, 8, 32, 64) * (2, 8, 64, 20) ->(2, 8, 32, 20)
        >>> # equal with mask shape
        >>> # [batch_size, seq_len_q, seq_len_k]
        >>> mask_shape = (2, 32, 20)
        >>> mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        >>> net = MutiHeadAttention(heads = 8)
        >>> x, attn = net(query, key, value, mask)
        >>> print(x.shape, attn.shape)
        (2, 32, 512) (2, 8, 32, 20)
    """

    def __init__(self, heads=8, d_model=512, dropout_rate=0.1, bias=True, attention_mode="dot product"):
        super(MutiHeadAttention, self).__init__()
        assert (d_model % heads == 0), "d_model must be divisible when divided by heads.\
Your d_model dimension is {d_model} and heads is {heads}.".format(d_model=d_model, heads=heads)
        self.d_k = d_model // heads
        self.d_model = d_model
        self.heads = heads
        self.linear_query = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_key = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_value = nn.Dense(d_model, d_model, has_bias=bias)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(keep_prob=1-dropout_rate)
        self.linear_out = nn.Dense(d_model, d_model, has_bias=bias)
        # default attention mode
        # attention_mode can be switch to other attention modes
        if attention_mode.startswith("dot"):
            self.attention_mode = DotAttention()


    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        """
        Get muti-head attention output and attention weights.
        """
        num_batch = query.shape[0]
        if mask is not None:
            mask = ops.expand_dims(mask, 1)
        # [batch_size,32,512]->[batch_size,8,32,64]
        query = mnp.swapaxes(self.linear_query(query).view(num_batch, -1, self.heads, self.d_k), 1, 2)
        key = mnp.swapaxes(self.linear_key(key).view(num_batch, -1, self.heads, self.d_k), 1, 2)
        value = mnp.swapaxes(self.linear_value(value).view(num_batch, -1, self.heads, self.d_k), 1, 2)
        output, self_attn = self.attention_mode(query, key, value, mask)
        output = mnp.swapaxes(output, 1, 2)
        # concat head
        output = output.view(num_batch, -1, self.heads * self.d_k)
        return self.linear_out(output), self_attn
