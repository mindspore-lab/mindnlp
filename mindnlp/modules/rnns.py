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
# pylint: disable=eval-used
'''RNN operators module, include RNN, GRU, LSTM'''
import math
import numpy as np
from mindspore import nn, ops, context
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore.common.initializer import initializer, Uniform
from mindspore.ops.primitive import constexpr
from mindspore.ops.operations._rl_inner_ops import CudnnGRU
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.nn import Dropout
from mindspore.ops import tensor_split, sigmoid, reverse

from mindnlp.utils import logging

logger = logging.get_logger(__name__)

@constexpr
def _init_state(shape, dtype, is_lstm):
    hx = Tensor(np.zeros(shape), dtype)
    cx = Tensor(np.zeros(shape), dtype)
    if is_lstm:
        return (hx, cx)
    return hx


def gru_cell(inputs, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    '''GRU cell function'''
    if b_ih is None:
        gi = ops.matmul(inputs, w_ih.T)
        gh = ops.matmul(hidden, w_hh.T)
    else:
        gi = ops.matmul(inputs, w_ih.T) + b_ih
        gh = ops.matmul(hidden, w_hh.T) + b_hh
    i_r, i_i, i_n = tensor_split(gi, 3, 1)
    h_r, h_i, h_n = tensor_split(gh, 3, 1)

    resetgate = sigmoid(i_r + h_r)
    inputgate = sigmoid(i_i + h_i)
    newgate = ops.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy

class SingleGRULayer_CPU(nn.Cell):
    """Single layer gru on CPU."""
    def __init__(self, input_size, hidden_size, has_bias, bidirectional):
        super().__init__(False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.bidirectional = bidirectional

    def forward(self, x, h, weights, biases):
        """forward direction."""
        h_shape = h.shape
        h = h.squeeze()
        x_dtype = x.dtype

        if not self.has_bias:
            w_ih, w_hh = weights
            w_ih, w_hh = w_ih.astype(x_dtype), w_hh.astype(x_dtype)
            b_ih, b_hh = None, None
        else:
            w_ih, w_hh = weights
            b_ih, b_hh = biases
            w_ih, w_hh = w_ih.astype(x_dtype), w_hh.astype(x_dtype)
            b_ih, b_hh = b_ih.astype(x_dtype), b_hh.astype(x_dtype)

        time_step = x.shape[0]
        outputs = Tensor(np.zeros((time_step, h_shape[1], h_shape[2])), x_dtype)

        t = Tensor(0)
        while t < time_step:
            x_t = x[t]
            h = gru_cell(x_t, h, w_ih, w_hh, b_ih, b_hh)
            outputs[t,:,:] = h
            t += 1

        return outputs, h.view(h_shape)


    def bidirection(self, inputs, h, weights, biases):
        """bidirectional."""
        rev_inputs = reverse(inputs, [0])
        h_f, h_b = tensor_split(h, 2)
        if self.has_bias:
            weights_f = weights[:2]
            weights_b = weights[2:]
            biases_f = biases[:2]
            biases_b = biases[2:]
        else:
            weights_f = weights[:2]
            weights_b = weights[2:]
            biases_f = None
            biases_b = None


        outputs_f, hn_f = self.forward(inputs, h_f, weights_f, biases_f)
        outputs_b, hn_b = self.forward(rev_inputs, h_b, weights_b, biases_b)

        outputs_b = reverse(outputs_b, [0])
        outputs = ops.concat([outputs_f, outputs_b], 2)
        hn = ops.concat([hn_f, hn_b], 0)

        return outputs, hn

    def construct(self, inputs, h, weights, biases):
        if self.bidirectional:
            return self.bidirection(inputs, h, weights, biases)
        return self.forward(inputs, h, weights, biases)


class SingleLSTMLayerBase(nn.Cell):
    """Single LSTM Layer"""
    def __init__(self, input_size, hidden_size, has_bias, bidirectional):
        super().__init__(False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.bidirectional = bidirectional

        self.rnn = ops.LSTM(input_size, hidden_size, 1, has_bias, bidirectional, 0.0)

    def _flatten_weights(self, weights, biases):
        raise NotImplementedError

    def construct(self, inputs, h, weights, biases):
        h0, c0 = h
        weights = self._flatten_weights(weights, biases)
        outputs, hn, cn, _, _ =  self.rnn(inputs, h0, c0, weights.astype(inputs.dtype))

        return outputs, (hn, cn)


class SingleLSTMLayer_CPU(SingleLSTMLayerBase):
    """Single LSTM Layer CPU"""
    def _flatten_weights(self, weights, biases):
        if self.bidirectional:
            weights = (weights[0].view((-1, 1, 1)), weights[2].view((-1, 1, 1)),
                       weights[1].view((-1, 1, 1)), weights[3].view((-1, 1, 1)))
        else:
            weights = (weights[0].view((-1, 1, 1)), weights[1].view((-1, 1, 1)))

        if self.has_bias:
            if self.bidirectional:
                biases = ((biases[0] + biases[1]).view((-1, 1, 1)),
                          (biases[2] + biases[3]).view((-1, 1, 1)))
            else:
                biases = ((biases[0] + biases[1]).view((-1, 1, 1)),)
            weights += biases

        weights = ops.concat(weights)
        return weights


class SingleLSTMLayer_GPU(SingleLSTMLayerBase):
    """Single LSTM Layer GPU"""
    def _flatten_weights(self, weights, biases):
        if self.bidirectional:
            weights = (weights[0].view((-1, 1, 1)), weights[1].view((-1, 1, 1)),
                       weights[2].view((-1, 1, 1)), weights[3].view((-1, 1, 1)))
        else:
            weights = (weights[0].view((-1, 1, 1)), weights[1].view((-1, 1, 1)))

        if self.has_bias:
            if self.bidirectional:
                biases = (biases[0].view((-1, 1, 1)), biases[1].view((-1, 1, 1)),
                          biases[2].view((-1, 1, 1)), biases[3].view((-1, 1, 1)))
            else:
                biases = (biases[0].view((-1, 1, 1)), biases[1].view((-1, 1, 1)),)
            weights += biases

        weights = ops.concat(weights)
        return weights


class MultiLayerRNN(nn.Cell):
    """Multilayer RNN."""
    def __init__(self, mode, input_size, hidden_size, num_layers, has_bias, \
                 bidirectional, dropout):
        super().__init__(False)
        backend = context.get_context('device_target')
        rnn_class = eval(f"Single{mode}Layer_{backend}")
        num_directions = 2 if bidirectional else 1

        cell_list = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * num_directions
            rnn = rnn_class(layer_input_size, hidden_size, has_bias, bidirectional)
            cell_list.append(rnn)
        self.cell_list = nn.CellList(cell_list)

        w_stride = 2
        if bidirectional:
            w_stride = w_stride * 2
        self.w_stride = w_stride

        self.dropout = Dropout(p=dropout)
        self.dropout_rate = dropout
        self.is_lstm = mode == 'LSTM'
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.has_bias = has_bias

    def construct(self, inputs, hx, weights, biases):
        """stacked mutil_layer static rnn"""
        pre_layer = inputs
        h_n = ()
        c_n = ()
        output = 0
        if self.is_lstm:
            hx_list = tensor_split(hx[0], self.num_layers)
            cx_list = tensor_split(hx[1], self.num_layers)
        else:
            hx_list = tensor_split(hx, self.num_layers)
            cx_list = None

        w_list = ()
        b_list = ()
        for i in range(self.num_layers):
            w_list = weights[i * self.w_stride: (i + 1) * self.w_stride]
            if self.has_bias:
                b_list = biases[i * self.w_stride: (i + 1) * self.w_stride]
            else:
                b_list = None
            if self.is_lstm:
                h_i = (hx_list[i], cx_list[i])
            else:
                h_i = hx_list[i]
            output, h_t = self.cell_list[i](pre_layer, h_i, w_list, b_list)
            pre_layer = self.dropout(output) if (self.dropout_rate != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t[0],)
                c_n += (h_t[1],)
            else:
                h_n += (h_t,)
        if self.is_lstm:
            h_n = ops.concat(h_n)
            c_n = ops.concat(c_n)

            return output, (h_n, c_n)
        h_n = ops.concat(h_n)
        return output, h_n


class StaticGRU_GPU(nn.Cell):
    """Static GRU on GPU"""
    def __init__(self, input_size, hidden_size, num_layers, has_bias, \
                 bidirectional, dropout):
        super().__init__(False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_bias = has_bias
        self.bidirectional = bidirectional
        self.dropout = dropout

    def construct(self, inputs, h, weights, biases):
        weights_new = ()
        for w in weights:
            weights_new += (ops.reshape(w, (-1, 1, 1)),)
        for b in biases:
            weights_new += (ops.reshape(b, (-1, 1, 1)),)

        weights = ops.concat(weights_new)
        dropout = self.dropout if self.training else 0.0
        _gru = _get_cache_prim(CudnnGRU)(self.input_size, self.hidden_size, self.num_layers, \
                                         self.has_bias, self.bidirectional, dropout)
        outputs, hn, _, _ =  _gru(inputs, h, weights.astype(inputs.dtype))
        return outputs, hn


class _RNNBase(nn.Cell):
    '''Basic class for RNN operators'''
    def __init__(self, mode, input_size, hidden_size, num_layers=1, has_bias=True,
                 batch_first=False, dropout=0., bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        is_gpu = context.get_context('device_target') == 'GPU'

        if not 0 <= dropout <= 1:
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        dropout = float(dropout)

        if dropout > 0 and num_layers == 1:
            logger.warning(f"dropout option adds dropout after all but last "
                           f"recurrent layer, so non-zero dropout expects "
                           f"num_layers greater than 1, but got dropout={dropout} and "
                           f"num_layers={num_layers}")
        if mode == "GRU":
            gate_size = 3 * hidden_size
            if is_gpu:
                self.rnn = StaticGRU_GPU(input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
            else:
                self.rnn = MultiLayerRNN('GRU', input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
        elif mode == "LSTM":
            gate_size = 4 * hidden_size
            self.rnn = MultiLayerRNN('LSTM', input_size, hidden_size, num_layers, has_bias, bidirectional, dropout)
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        num_directions = 2 if bidirectional else 1
        self.is_lstm = mode == "LSTM"

        self._weights = []
        self._biases = []
        stdv = 1 / math.sqrt(hidden_size)
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                self._weights.append(Parameter(initializer(Uniform(stdv), (gate_size, layer_input_size)),
                                                    name=f'weight_ih_l{layer}{suffix}'))
                self._weights.append(Parameter(initializer(Uniform(stdv), (gate_size, hidden_size)),
                                                    name=f'weight_hh_l{layer}{suffix}'))
                if has_bias:
                    self._biases.append(Parameter(initializer(Uniform(stdv), (gate_size,)),
                                                        name=f'bias_ih_l{layer}{suffix}'))
                    self._biases.append(Parameter(initializer(Uniform(stdv), (gate_size,)),
                                                        name=f'bias_hh_l{layer}{suffix}'))

        self._weights = ParameterTuple(self._weights)
        self._biases = ParameterTuple(self._biases)

    def construct(self, x, hx=None):
        '''Defines the RNN like operators performed'''
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        x_dtype = x.dtype

        if hx is None:
            hx = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size), \
                             x_dtype, self.is_lstm)

        if self.batch_first:
            x = x.transpose((1, 0, 2))

        x_n, hx_n = self.rnn(x, hx, self._weights, self._biases)

        if self.batch_first:
            x_n = x_n.transpose((1, 0, 2))

        return x_n, hx_n


class StaticGRU(_RNNBase):
    r"""
    Stacked GRU (Gated Recurrent Unit) layers.

    Apply GRU layer to the input.

    There are two gates in a GRU model; one is update gate and the other is reset gate.
    Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}`, the update and reset gate at
    time :math:`t` is computed using an gating mechanism. Update gate :math:`z_t` is designed to protect the cell
    from perturbation by irrelevant inputs and past hidden state. Reset gate :math:`r_t` determines how much
    information should be reset from old hidden state. New memory state :math:`{n}_t` is
    calculated with the current input, on which the reset gate will be applied. Finally, current hidden state
    :math:`h_{t}` is computed with the calculated update grate and new memory state. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU. Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. Default 0.0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional GRU,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and
          shape (num_directions * `num_layers`, batch_size, `hidden_size`). Data type of `hx` must be the same as `x`.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`) or
          (batch_size, seq_len, num_directions * `hidden_size`).
        - **hx_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> net = StaticGRU(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """
    def __init__(self, *args, **kwargs):
        mode = 'GRU'
        super().__init__(mode, *args, **kwargs)


class StaticLSTM(_RNNBase):
    r"""
    Stacked LSTM (Long Short-Term Memory) layers.
    Apply LSTM layer to the input.
    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and the other is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}` and an cell
    state :math:`c_{t-1}` of the layer at time :math:`{t-1}`, the cell state and hidden state at
    time :math:`t` is computed using an gating mechanism. Input gate :math:`i_t` is designed to protect the cell
    from perturbation by irrelevant inputs. Forget gate :math:`f_t` affords protection of the cell by forgetting
    some information in the past, which is stored in :math:`h_{t-1}`. Output gate :math:`o_t` protects other
    units from perturbation by currently irrelevant memory contents. Candidate cell state :math:`\tilde{c}_t` is
    calculated with the current input, on which the input gate will be applied. Finally, current cell state
    :math:`c_{t}` and hidden state :math:`h_{t}` are computed with the calculated gates and cell states. The complete
    formulation is as follows.
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
    LSTM hides the cycle of the whole cyclic neural network on the time step of the sequence,
    and input the sequence and initial state to obtain the matrix spliced by
    the hidden state of each time step and the hidden state of the last time step.
    We use the hidden state of the last time step as the coding feature of the input sentence and
    output it to the next layer.
    .. math::
        h_{0:n},(h_{n}, c_{n}) = LSTM(x_{0:n},(h_{0},c_{0}))
    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked LSTM . Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional LSTM,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.
    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 or mindspore.float16 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32
          or mindspore.float16 and shape (num_directions * `num_layers`, batch_size, `hidden_size`).
          The data type of `hx` must be the same as `x`.

    Outputs:
        Tuple, a tuple contains (`output`, (`h_n`, `c_n`)).
        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (tuple) - A tuple of two Tensor (h_n, c_n) both of shape
          (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> net = StaticLSTM(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> c0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, (hn, cn) = net(x, (h0, c0))
        >>> print(output.shape)
        (3, 5, 16)
    """

    def __init__(self, *args, **kwargs):
        mode = 'LSTM'
        super().__init__(mode, *args, **kwargs)


__all__ = ['StaticGRU', 'StaticLSTM']
