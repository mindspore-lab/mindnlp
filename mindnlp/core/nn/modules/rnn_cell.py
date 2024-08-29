# Copyright 2021 Huawei Technologies Co., Ltd
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
"""RNN Cells module, include RNNCell, GRUCell, LSTMCell."""
import math

import mindspore
from mindspore import Parameter
from .module import Module
from .. import init
from .. import functional as F
from ... import ops

__all__ = ['LSTMCell', 'GRUCell', 'RNNCell']

def _rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    """RNN cell function with tanh activation"""
    if b_ih is None:
        igates = ops.matmul(inputs, w_ih.T)
        hgates = ops.matmul(hidden, w_hh.T)
    else:
        igates = ops.matmul(inputs, w_ih.T) + b_ih
        hgates = ops.matmul(hidden, w_hh.T) + b_hh
    return ops.tanh(igates + hgates)


def _rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    """RNN cell function with relu activation"""
    if b_ih is None:
        igates = ops.matmul(inputs, w_ih.T)
        hgates = ops.matmul(hidden, w_hh.T)
    else:
        igates = ops.matmul(inputs, w_ih.T) + b_ih
        hgates = ops.matmul(hidden, w_hh.T) + b_hh
    return F.relu(igates + hgates)


def _lstm_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    """LSTM cell function"""
    hx, cx = hidden
    if b_ih is None:
        gates = ops.matmul(inputs, w_ih.T) + ops.matmul(hx, w_hh.T)
    else:
        gates = ops.matmul(inputs, w_ih.T) + ops.matmul(hx, w_hh.T) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = ops.chunk(gates, 4, 1)
    ingate = ops.sigmoid(ingate)
    forgetgate = ops.sigmoid(forgetgate)
    cellgate = ops.tanh(cellgate)
    outgate = ops.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * ops.tanh(cy)

    return hy, cy


def _gru_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    """GRU cell function"""
    if b_ih is None:
        gi = ops.matmul(inputs, w_ih.T)
        gh = ops.matmul(hidden, w_hh.T)
    else:
        gi = ops.matmul(inputs, w_ih.T) + b_ih
        gh = ops.matmul(hidden, w_hh.T) + b_hh
    i_r, i_i, i_n = ops.chunk(gi, 3, 1)
    h_r, h_i, h_n = ops.chunk(gh, 3, 1)

    resetgate = ops.sigmoid(i_r + h_r)
    inputgate = ops.sigmoid(i_i + h_i)
    newgate = ops.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


class RNNCellBase(Module):
    """Basic class for RNN Cells"""
    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int,
                 dtype=None):
        factory_kwargs = {'dtype': dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(ops.empty((num_chunks * hidden_size, input_size), **factory_kwargs))
        self.weight_hh = Parameter(ops.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs))
        if bias:
            self.bias_ih = Parameter(ops.empty(num_chunks * hidden_size, **factory_kwargs))
            self.bias_hh = Parameter(ops.empty(num_chunks * hidden_size, **factory_kwargs))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class RNNCell(RNNCellBase):
    r"""
    An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time :math:`t-1` or the initial hidden state at time `0`.
    If `nonlinearity` is `relu`, then `relu` is used instead of `tanh`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        bias (bool): Whether the cell has bias :math:`b_{ih}` and :math:`b_{hh}`. Default: ``True`` .
        nonlinearity (str): The non-linearity to use. Can be either ``"tanh"`` or ``"relu"`` .
            Default: ``"tanh"`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, input\_size)` .
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape :math:`(batch\_size, hidden\_size)` .

    Outputs:
        - **hx'** (Tensor) - Tensor of shape :math:`(batch\_size, hidden\_size)` .

    Raises:
        TypeError: If `input_size` or `hidden_size` is not an int or not greater than 0.
        TypeError: If `bias` is not a bool.
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> net = ms.nn.RNNCell(10, 16)
        >>> x = ms.Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = ms.Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], hx)
        ...     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    _non_linearity = ['tanh', 'relu']

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh",
                 dtype=mindspore.float32):
        super().__init__(input_size, hidden_size, bias, num_chunks=1, dtype=dtype)
        self.nonlinearity = nonlinearity

    def forward(self, x, hx):
        if self.nonlinearity == "tanh":
            ret = _rnn_tanh_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = _rnn_relu_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret


class LSTMCell(RNNCellBase):
    r"""
    A LSTM (Long Short-Term Memory) cell.

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.
    Details can be found in paper `LONG SHORT-TERM MEMORY
    <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and
    `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_.

    The encapsulated LSTMCell can be simplified to the following formula:

    .. math::
        h^{'},c^{'} = LSTMCell(x, (h_0, c_0))

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        bias (bool): Whether the cell has bias `b_{ih}` and `b_{hh}`. Default: ``True`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, input\_size)` .
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32
          and shape :math:`(batch\_size, hidden\_size)` .

    Outputs:
        - **hx'** (Tensor) - A tuple of two Tensors (h', c') both of data shape :math:`(batch\_size, hidden\_size)` .

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> net = ms.nn.LSTMCell(10, 16)
        >>> x = ms.Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> h = ms.Tensor(np.ones([3, 16]).astype(np.float32))
        >>> c = ms.Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], (h, c))
        ...     output.append(hx)
        >>> print(output[0][0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 dtype=mindspore.float32):
        super().__init__(input_size, hidden_size, bias, num_chunks=4, dtype=dtype)
        self.support_non_tensor_inputs = True

    def forward(self, x, hx):
        return _lstm_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class GRUCell(RNNCellBase):
    r"""
    A GRU(Gated Recurrent Unit) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. :math:`h` is hidden state.
    :math:`r` is reset gate. :math:`z` is update gate. :math:`n` is n-th layer. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        bias (bool): Whether the cell has bias :math:`b_{in}` and :math:`b_{hn}`. Default: ``True`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, input\_size)` .
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape :math:`(batch\_size, hidden\_size)` .

    Outputs:
        - **hx'** (Tensor) - Tensor of shape :math:`(batch\_size, hidden\_size)` .

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> net = ms.nn.GRUCell(10, 16)
        >>> x = ms.Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = ms.Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], hx)
        ...     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 dtype=mindspore.float32):
        super().__init__(input_size, hidden_size, bias, num_chunks=3, dtype=dtype)

    def forward(self, x, hx):
        return _gru_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
