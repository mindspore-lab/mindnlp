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
import mindspore.numpy as mnp
from mindspore import Parameter, ops, nn

class ScaledDotAttention(nn.Cell):
    r"""
    Scaled Dot-Product Attention
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"

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
        >>> from mindspore.text.modules.attentions import SclaedDotAttention
        >>> model = ScaledDotAttention(dropout=0.9)
        >>> q = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> k = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> v = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        >>> output, _ = model(q, k, v)
        >>> print(output.shape)
        (2, 1024, 512)
    """

    def __init__(self, dropout=0.9):
        super().__init__()
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        scale = mnp.sqrt(ops.scalar_to_tensor(query.shape[-1]))
        scores = ops.matmul(query, key.swapaxes(-1, -2)) / scale
        if mask is not None:
            scores = ops.masked_fill(scores, mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = ops.matmul(attn, value)
        return output, attn

class AdditiveAttention(nn.Cell):
    r"""
    Additive Attention
    Additive Attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper"

      .. math::

          Attention(Q,K,V) = (W_v)T *(tanh(W_q * Q + W_k * K))

    Args:
        hidden_dims (int): The dimesion of hidden state vector
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
        >>> from mindspore.text.modules.attentions import AdditiveAttention
        >>> model = AdditiveAttention(hidden_dims=512, dropout=0.9)
        >>> q = Tensor(np.ones((2, 32, 512)), mindspore.float32)
        >>> k = Tensor(np.ones((2, 20, 512)), mindspore.float32)
        >>> v = Tensor(np.ones((2, 20, 512)), mindspore.float32)
        >>> mask_shape = (2, 32, 20)
        >>> mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        >>> output, attn = model(q, k, v, mask)
        >>> print(output.shape, attn.shape)
        (2, 32, 512) (2, 32, 20)
    """
    def __init__(self, hidden_dims, dropout=0.9):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.w_q = nn.Dense(hidden_dims, hidden_dims, has_bias=False)
        self.w_k = nn.Dense(hidden_dims, hidden_dims, has_bias=False)
        self.w_output = nn.Dense(hidden_dims, 1, has_bias=False)
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        self.tanh = nn.Tanh()
        # Set bias parameter
        uniformreal = ops.UniformReal(seed=114514)
        bias_layer = uniformreal((hidden_dims,))
        self.bias = Parameter(bias_layer)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        """
        Additive attention network construction.
        """
        query = self.w_q(query)
        key = self.w_k(key)
        features = query.expand_dims(-2) + key.expand_dims(-3) + self.bias
        scores = self.w_output(self.tanh(features)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = ops.matmul(attn, value)
        return output, attn

class CosineAttention(nn.Cell):
    r"""
    Cosine Attention
    Cosine Attention proposed in "Neural Turing Machines" paper"

      .. math::

          Sim(Q, K) = (Q * (K)T) / |Q| * |K|
          Attention(Q,K,V) = softmax(Sim(Q, K)) * V


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
        >>> from mindspore.text.modules.attentions import CosineAttention
        >>> model = CosineAttention(dropout=0.9)
        >>> q = Tensor(np.ones((2, 32, 512)), mindspore.float32)
        >>> k = Tensor(np.ones((2, 20, 512)), mindspore.float32)
        >>> v = Tensor(np.ones((2, 20, 512)), mindspore.float32)
        >>> mask_shape = (2, 32, 20)
        >>> mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        >>> output, attn = model(q, k, v, mask)
        >>> print(output.shape, attn.shape)
        (2, 32, 512) (2, 32, 20)
    """
    def __init__(self, dropout=0.9):
        super().__init__()
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(keep_prob=1-dropout)

    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        """
        Consine attention network construction.
        """
        query_length = ops.sqrt((query * query).sum())
        key_length = ops.sqrt((key * key).sum())
        features = ops.matmul(query, key.swapaxes(-1, -2))
        scores = ops.div(features, (query_length * key_length))
        if mask is not None:
            scores = ops.masked_fill(scores, mask == 0, -1e9)
        attn = self.softmax(scores)
        scores = self.dropout(attn)
        output = ops.matmul(attn, value)
        return output, attn

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
    w_sum = ops.matmul(weights, tensor)
    while mask.ndim < tensor.ndim:
        mask = ops.expand_dims(mask, 1)
    mask = mask.swapaxes(-1, -2)
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
        >>> from mindspore import ops
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
        super().__init__()
        self.bmm = ops.BatchMatMul()

    def construct(self, x_batch, x_mask, y_batch, y_mask):
        similarity_matrix = self.bmm(x_batch, y_batch.swapaxes(2, 1))
        x_y_attn = _masked_softmax(similarity_matrix, y_mask)
        y_x_attn = _masked_softmax(similarity_matrix.swapaxes(1, 2), x_mask)
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
        - **attention_mode** (str) - attention mode. Default: "dot".

    Inputs:
        - **query** (mindspore.Tensor) - The query vector.
        - **key** (mindspore.Tensor) - The key vector.
        - **value** (mindspore.Tensor) - The value vector. [batch_size, seq_len, d_model]
        - **mask** Optional[mindspore.Tensor[bool]] - The mask vector. [seq_len, seq_len, batch_size]

    Returns:
        - output (mindspore.Tensor) - The output of muti-head attention.
        - attn (mindspore.Tensor) - The last layer of attention weights

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> from mindspore import ops
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
        >>> #use additive attention
        >>> net = MutiHeadAttention(heads=8, attention_mode="add")
        >>> x, attn = net(query, key, value, mask)
        >>> print(x.shape, attn.shape)
        (2, 32, 512) (2, 8, 32, 20)
    """

    def __init__(self, heads=8, d_model=512, dropout_rate=0.1, bias=True, attention_mode="dot"):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"'d_model' must be divisible when divided by 'heads'. "
                             f"Your d_model dimension is {d_model} and heads is {heads}.")
        self.d_k = d_model // heads
        self.d_model = d_model
        self.heads = heads
        self.linear_query = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_key = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_value = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_out = nn.Dense(d_model, d_model, has_bias=bias)
        # default attention mode dot product
        # attention_mode can be switch to other attention modes
        if "add" in attention_mode.lower():
            self.attention_mode = AdditiveAttention(hidden_dims=int(self.d_model / self.heads), dropout=1-dropout_rate)
        elif "cos" in attention_mode.lower():
            self.attention_mode = CosineAttention(dropout=1-dropout_rate)
        else:
            self.attention_mode = ScaledDotAttention(1-dropout_rate)


    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        """
        Get muti-head attention output and attention weights.
        """
        num_batch = query.shape[0]
        if mask is not None:
            mask = ops.expand_dims(mask, 1)
        # [batch_size,32,512]->[batch_size,8,32,64]
        query = self.linear_query(query).view(num_batch, -1, self.heads, self.d_k).swapaxes(1, 2)
        key = self.linear_key(key).view(num_batch, -1, self.heads, self.d_k).swapaxes(1, 2)
        value = self.linear_value(value).view(num_batch, -1, self.heads, self.d_k).swapaxes(1, 2)
        output, self_attn = self.attention_mode(query, key, value, mask)
        output = output.swapaxes(1, 2)
        # concat head
        output = output.view(num_batch, -1, self.heads * self.d_k)
        return self.linear_out(output), self_attn

class SelfAttention(nn.Cell):
    r"""
    Self attention is from the paper “attention is all you need”

    Args:
        - **d_model** (int) - The `query`, `key` and `value` vectors dimensions. Default: 512.
        - **dropout** (float): The keep rate, greater than 0 and less equal than 1. Default: 0.9.
        - **bias** (bool) - whether to use a bias vector. Default: True.
        - **attention_mode** (str) - attention mode. Default: "dot".

    Inputs:
        - **query** (mindspore.Tensor) - The query vector.
        - **key** (mindspore.Tensor) - The key vector.
        - **value** (mindspore.Tensor) - The value vector. [batch_size, seq_len, d_model]
        - **mask** Optional[mindspore.Tensor[bool]] - The mask vector. [seq_len, seq_len, batch_size]

    Returns:
        - output (mindspore.Tensor) - The output of self attention.
        - attn (mindspore.Tensor) - The last layer of attention weights

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> from mindspore.text.modules.attentions import SelfAttention
        >>> standard_normal = ops.StandardNormal(seed=114514)
        >>> query = standard_normal((2, 32, 512))
        >>> key = standard_normal((2, 20, 512))
        >>> value = standard_normal((2, 20, 512))
        >>> mask_shape = (2, 32, 20)
        >>> mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        >>> net = SelfAttention()
        >>> output, attn = net(query, key, value, mask)
        >>> print(x.shape, attn.shape)
        (2, 32, 512) (2, 32, 20)
    """
    def __init__(self, d_model=512, dropout_rate=0.1, bias=True, attention_mode="dot"):
        super().__init__()
        self.d_model = d_model
        self.linear_query = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_key = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_value = nn.Dense(d_model, d_model, has_bias=bias)
        self.linear_out = nn.Dense(d_model, d_model, has_bias=bias)
        if "add" in attention_mode.lower():
            self.attention_mode = AdditiveAttention(hidden_dims=self.d_model, dropout=1-dropout_rate)
        elif "cos" in attention_mode.lower():
            self.attention_mode = CosineAttention(dropout=1-dropout_rate)
        else:
            self.attention_mode = ScaledDotAttention(1-dropout_rate)

    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)
        output, self_attn = self.attention_mode(query, key, value, mask)
        return self.linear_out(output), self_attn

class LocationAwareAttention(nn.Cell):
    r"""
    Location Aware Attention
    Location Aware Attention proposed in "Attention-Based Models for Speech Recognition"

    Args:
        decoder_dim (int): The dimension of the decoder hidden states
        encoder_dim (int): The dimension of the encoder hidden states
        attn_dim (int): The dimension of the attention hidden states
        smoothing (bool): Smoothing label from "Attention-Based Models for Speech Recognition"

    Inputs:
        - **query** (mindspore.Tensor) - Decoder hidden states, Shape=(batch_size, 1, decoder_dim).
        - **value** (mindspore.Tensor) - Encoder outputs, Shape=(batch_size, seq_len, encoder_dim).
        - **last_attn** (mindspore.Tensor) - Attention weight of previous step, Shape=(batch_size, seq_len).

    Returns:
        - **context** (mindspore.Tensor) - The context vector, Shape=(batch_size, 1, decoder_dim).
        - **attn** (mindspore.Tensor) - Attention weight of this step, Shape=(batch_size, seq_len).

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> from mindspore import ops, Tensor
        >>> from mindspore.text.modules.attentions import LocationAwareAttention
        >>> batch_size, seq_len, enc_d, dec_d, conv_d, attn_d = 2, 40, 32, 20, 10, 512
        >>> standard_normal = ops.StandardNormal(seed=114514)
        >>> query = standard_normal((batch_size, 1, dec_d))
        >>> value = standard_normal((batch_size, seq_len, enc_d))
        >>> last_attn = standard_normal((batch_size, seq_len))
        >>> net = LocationAwareAttention(
            decoder_dim=dec_d,
            encoder_dim=enc_d,
            conv_dim=conv_d,
            attn_dim=attn_d,
            smoothing=False)
        >>> mask_shape = (batch_size, seq_len)
        >>> mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        >>> net.set_mask(mask)
        >>> cont, attn = net(query, value, last_attn)
        >>> print(cont.shape, attn.shape)
        (2, 1, 32) (2, 40)
    """

    def __init__(self, decoder_dim, encoder_dim, attn_dim, smoothing=False):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.attn_dim = attn_dim
        self.smoothing = smoothing
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=self.attn_dim, kernel_size=3, pad_mode="pad", padding=1)
        self.w_linear = nn.Dense(self.decoder_dim, self.attn_dim, has_bias=False)
        self.v_linear = nn.Dense(self.encoder_dim, self.attn_dim, has_bias=False)
        self.fc_linear = nn.Dense(attn_dim, 1, has_bias=True)
        # Set bias parameter
        uniformreal = ops.UniformReal(seed=114514)
        bias_layer = uniformreal((attn_dim,))
        self.bias = Parameter(bias_layer)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(axis=-1)
        self.mask = None
        self.sigmoid = nn.Sigmoid()

    def set_mask(self, mask):
        """
        Set the mask

        Args:
        - **mask** mindspore.Tensor[bool] - The mask vector.
        """
        self.mask = mask

    def construct(self, query, value, last_attn=None):
        """
        Location aware attention network construction.
        """
        batch_size, seq_len = query.shape[0], value.shape[1]
        conv_attn = self.conv(ops.expand_dims(last_attn, 1)).swapaxes(1, 2)
        scores = self.fc_linear(
            self.tanh(
                self.w_linear(query) + self.v_linear(value) + conv_attn + self.bias
            )
        ).squeeze(-1)
        if last_attn is None:
            last_attn = ops.zeros(batch_size, seq_len)
        if self.mask is not None:
            scores = ops.masked_fill(scores, self.mask == 0, -1e9)
        if self.smoothing:
            scores = self.sigmoid(scores)
            attn = ops.div(scores, ops.expand_dims(scores.sum(axis=-1), -1))
        else:
            attn = self.softmax(scores)
        context = ops.matmul(ops.expand_dims(attn, 1), value)
        return context, attn

class LinearAttention(nn.Cell):
    r"""
    Linear attention computes attention between a vector and a matrix using a linear attention function.

    Args:
        - **query_size** (int) - The sentence length of `query`. Usually query.shape[-2]
        - **key_size** (int) - The sentence length of `key`. Usually key.shape[-2]
        - **hidden_dim** (int) - The dimension of hidden vector
        - **dropout** (float): The keep rate, greater than 0 and less equal than 1. Default: 0.9.

    Inputs:
        - **query** (mindspore.Tensor) - The query vector.
        - **key** (mindspore.Tensor) - The key vector.
        - **value** (mindspore.Tensor) - The value vector. [seq_len, batch_size, d_model]
        - **mask** Optional[mindspore.Tensor[bool]] - The mask vector. [seq_len, seq_len, batch_size]

    Returns:
        - output (mindspore.Tensor) - The output of linear attention.
        - attn (mindspore.Tensor) - The last layer of attention weights

    Examples:
        >>> import mindspore
        >>> import mindspore.numpy as np
        >>> from mindspore import ops
        >>> from mindspore import Tensor
        >>> from mindspore.text.modules.attentions import SelfAttention
        >>> standard_normal = ops.StandardNormal(seed=114514)
        >>> query = standard_normal((2, 32, 512))
        >>> key = standard_normal((2, 20, 512))
        >>> value = standard_normal((2, 20, 500))
        >>> net = LinearAttention(batch_size=2, query_dim=32, key_dim=20, hidden_dim=512)
        >>> mask_shape = (2, 32, 20)
        >>> mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        >>> output, attn = net(query, key, value, mask)
        >>> print(x.shape, attn.shape)
        (2, 32, 512) (2, 32, 20)
    """
    def __init__(self, batch_size, query_dim, key_dim, hidden_dim, dropout=0.9):
        super().__init__()
        self.w_linear = nn.Dense(query_dim + key_dim, query_dim, has_bias=False)
        self.softmax = nn.Softmax(axis=-1)
        self.tanh = nn.Tanh()
        self.v_linear = nn.Dense(hidden_dim, key_dim, has_bias=False)
        self.dropout = nn.Dropout(keep_prob=1-dropout)
        #set bias parameter
        uniformreal = ops.UniformReal(seed=114514)
        bias_layer_shape = (batch_size, query_dim, hidden_dim)
        bias_layer = uniformreal(bias_layer_shape)
        self.bias = Parameter(bias_layer)

    def construct(self, query, key, value, mask: Optional[mindspore.Tensor] = None):
        features = self.w_linear(ops.concat((query, key), -2).swapaxes(-1, -2)).swapaxes(-1, -2)
        scores = self.v_linear(self.tanh(features + self.bias))

        if mask is not None:
            scores = ops.masked_fill(scores, mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = ops.matmul(attn, value)
        return output, attn
