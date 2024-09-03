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
"""RNN operators module, include RNN, GRU."""
import math
import warnings

import mindspore
from mindspore import Parameter
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations._rl_inner_ops import CudnnGRU
from mindspore.ops import DynamicGRUV2, DynamicRNN, ReverseV2, ReverseSequence
from mindspore.ops import LSTM as LSTMOP
from mindspore.nn.layer.rnn_cells import _rnn_relu_cell, _rnn_tanh_cell, _gru_cell, _lstm_cell

from .module import Module
from .dropout import Dropout
from ... import ops
from .. import init


__all__ = ['LSTM', 'GRU', 'RNN']


def _init_state(shape, dtype, is_lstm):
    hx = ops.zeros(*shape, dtype=dtype)
    cx = ops.zeros(*shape, dtype=dtype)
    if is_lstm:
        return (hx, cx)
    return hx


def sequence_mask(lengths, maxlen):
    """generate mask matrix by seq_length"""
    range_vector = ops.arange(start=0, end=maxlen, step=1, dtype=lengths.dtype)
    result = range_vector < lengths.view(lengths.shape + (1,))
    return result.astype(mindspore.int32)


def select_by_mask(inputs, mask):
    """mask hiddens by mask matrix"""
    return mask.view(mask.shape + (1,)).swapaxes(0, 1) \
               .expand_as(inputs).astype(mindspore.bool_) * inputs


def get_hidden(output, seq_length):
    """get hidden state by seq_length"""
    batch_index = ops.arange(start=0, end=seq_length.shape[0], step=1, dtype=seq_length.dtype)
    indices = ops.cat((seq_length.view(-1, 1) - 1, batch_index.view(-1, 1)), 1)
    return ops.gather_nd(output, indices)


class _DynamicRNNBase(Module):
    '''Dynamic RNN module to compute RNN cell by timesteps'''

    def __init__(self, mode):
        super().__init__()
        if mode == "RNN_RELU":
            cell = _rnn_relu_cell
        elif mode == "RNN_TANH":
            cell = _rnn_tanh_cell
        elif mode == "LSTM":
            cell = _lstm_cell
        elif mode == "GRU":
            cell = _gru_cell
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        self.cell = cell
        self.is_lstm = mode == "LSTM"

    def recurrent(self, x, h_0, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps without sequence length'''
        time_step = x.shape[0]
        outputs = []
        t = 0
        h = h_0
        while t < time_step:
            x_t = x[t:t + 1:1]
            x_t = x_t.squeeze(0)
            h = self.cell(x_t, h, w_ih, w_hh, b_ih, b_hh)
            if self.is_lstm:
                outputs.append(h[0])
            else:
                outputs.append(h)
            t += 1
        outputs = ops.stack(outputs)
        return outputs, h

    def variable_recurrent(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps with sequence length'''
        time_step = x.shape[0]
        h_t = h
        if self.is_lstm:
            hidden_size = h[0].shape[-1]
            zero_output = ops.zeros_like(h_t[0])
        else:
            hidden_size = h.shape[-1]
            zero_output = ops.zeros_like(h_t)
        seq_length = seq_length.to(mindspore.float32)
        seq_length = ops.broadcast_to(seq_length, (hidden_size, -1))
        seq_length = seq_length.to(mindspore.int32)
        seq_length = ops.transpose(seq_length, 1, 0)

        outputs = []
        state_t = h_t
        t = 0
        while t < time_step:
            x_t = x[t:t + 1:1]
            x_t = x_t.squeeze(0)
            h_t = self.cell(x_t, state_t, w_ih, w_hh, b_ih, b_hh)
            seq_cond = seq_length > t
            if self.is_lstm:
                state_t_0 = ops.where(seq_cond, h_t[0], state_t[0])
                state_t_1 = ops.where(seq_cond, h_t[1], state_t[1])
                output = ops.where(seq_cond, h_t[0], zero_output)
                state_t = (state_t_0, state_t_1)
            else:
                state_t = ops.where(seq_cond, h_t, state_t)
                output = ops.where(seq_cond, h_t, zero_output)
            outputs.append(output)
            t += 1
        outputs = ops.stack(outputs)
        return outputs, state_t

    def forward(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        x_dtype = x.dtype
        w_ih = w_ih.astype(x_dtype)
        w_hh = w_hh.astype(x_dtype)
        if b_ih is not None:
            b_ih = b_ih.astype(x_dtype)
            b_hh = b_hh.astype(x_dtype)
        if seq_length is None:
            return self.recurrent(x, h, w_ih, w_hh, b_ih, b_hh)
        return self.variable_recurrent(x, h, seq_length, w_ih, w_hh, b_ih, b_hh)


class _DynamicRNNRelu(_DynamicRNNBase):
    '''Dynamic RNN module with Relu activation'''

    def __init__(self):
        mode = 'RNN_RELU'
        super().__init__(mode)


class _DynamicRNNTanh(_DynamicRNNBase):
    '''Dynamic RNN module with Tanh activation'''

    def __init__(self):
        mode = 'RNN_TANH'
        super().__init__(mode)


class _DynamicGRUCPUGPU(Module):
    '''Dynamic GRU module on CPU and GPU'''

    def __init__(self):
        super().__init__()
        self.is_gpu = mindspore.get_context("device_target") == "GPU"

    def forward(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''_DynamicGRUCPUGPU'''
        gate_size, input_size = w_ih.shape
        hidden_size = gate_size // 3
        if self.is_gpu:
            if b_ih is None:
                weights = ops.concat((
                    w_ih.view(-1, 1, 1),
                    w_hh.view(-1, 1, 1)
                ))
                bias = False
            else:
                bias = True
                weights = ops.concat((
                    w_ih.view(-1, 1, 1),
                    w_hh.view(-1, 1, 1),
                    b_ih.view(-1, 1, 1),
                    b_hh.view(-1, 1, 1)
                ))
            _gru = _get_cache_prim(CudnnGRU)(input_size, hidden_size, 1, bias, False, 0.0)
            output, h_n, _, _ = _gru(
                x,
                h_0.view(1, *h_0.shape),
                weights.astype(x.dtype)
            )
            if seq_length is not None:
                h_n = get_hidden(output, seq_length)
                mask = sequence_mask(seq_length, x.shape[0])
                output = select_by_mask(output, mask)
        else:
            output, h_n = _DynamicRNNBase('GRU')(x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh)

        return output, h_n


class _DynamicGRUAscend(Module):
    '''Dynamic GRU module on Ascend'''

    def __init__(self):
        super().__init__()
        self.gru = DynamicGRUV2(gate_order='rzh')

    def forward(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''Dynamic GRU module on Ascend'''
        if b_ih is None:
            b_ih = ops.zeros(w_ih.shape[0], dtype=w_ih.dtype)
            b_hh = ops.zeros(w_ih.shape[0], dtype=w_ih.dtype)
        outputs, _, _, _, _, _ = self.gru(x.to(self.dtype), \
                                          ops.transpose(w_ih, 1, 0), \
                                          ops.transpose(w_hh, 1, 0), \
                                          b_ih, \
                                          b_hh, \
                                          None, h_0)
        if seq_length is not None:
            h = get_hidden(outputs, seq_length)
            mask = sequence_mask(seq_length, x.shape[0])
            outputs = select_by_mask(outputs, mask)
        else:
            h = outputs[-1]
        return outputs, h


class _DynamicLSTMCPUGPU(Module):
    '''Dynamic LSTM module on CPU and GPU'''

    def __init__(self):
        super().__init__()
        self.is_gpu = mindspore.get_context("device_target") == "GPU"

    def forward(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''Dynamic LSTM module on CPU and GPU'''
        gate_size, input_size = w_ih.shape
        hidden_size = gate_size // 4
        if seq_length is not None:
            output, (h_n, c_n) = _DynamicRNNBase('LSTM')(x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh)
        else:
            if b_ih is None:
                weights = ops.concat((
                    w_ih.view(-1, 1, 1),
                    w_hh.view(-1, 1, 1)
                ))
                has_bias = False
            else:
                has_bias = True
                if self.is_gpu:
                    weights = ops.concat((
                        w_ih.view(-1, 1, 1),
                        w_hh.view(-1, 1, 1),
                        b_ih.view(-1, 1, 1),
                        b_hh.view(-1, 1, 1)
                    ))
                else:
                    bias = b_ih + b_hh
                    weights = ops.concat((
                        w_ih.view(-1, 1, 1),
                        w_hh.view(-1, 1, 1),
                        bias.view(-1, 1, 1)
                    ))
            _lstm = _get_cache_prim(LSTMOP)(input_size, hidden_size, 1, has_bias, False, 0.0)
            output, h_n, c_n, _, _ = _lstm(
                x,
                h_0[0].unsqueeze(0),
                h_0[1].unsqueeze(0),
                weights.astype(x.dtype)
            )
        return output, (h_n, c_n)


class _DynamicLSTMAscend(Module):
    '''Dynamic LSTM module on Ascend'''

    def __init__(self):
        super().__init__()
        self.lstm = DynamicRNN()

    def forward(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''Dynamic LSTM module on Ascend'''
        w_ih_i, w_ih_f, w_ih_g, w_ih_o = ops.chunk(w_ih, 4, 0)
        w_hh_i, w_hh_f, w_hh_g, w_hh_o = ops.chunk(w_hh, 4, 0)
        w_ih = ops.cat((w_ih_i, w_ih_g, w_ih_f, w_ih_o), 0)
        w_hh = ops.cat((w_hh_i, w_hh_g, w_hh_f, w_hh_o), 0)
        weight = ops.cat((w_ih, w_hh), 1)
        if b_ih is None:
            bias = ops.zeros(w_ih.shape[0], dtype=w_ih.dtype)
        else:
            b_ih_i, b_ih_f, b_ih_g, b_ih_o = ops.chunk(b_ih, 4, 0)
            b_hh_i, b_hh_f, b_hh_g, b_hh_o = ops.chunk(b_hh, 4, 0)
            bias = ops.cat((b_ih_i + b_hh_i, \
                            b_ih_g + b_hh_g, \
                            b_ih_f + b_hh_f, \
                            b_ih_o + b_hh_o), 0)

        outputs, h, c, _, _, _, _, _ = self.lstm(x.to(mindspore.float16), \
                                                 ops.transpose(weight, 1, 0).to(mindspore.float16), \
                                                 bias.to(mindspore.float16), None, \
                                                 h_0[0].unsqueeze(0).to(mindspore.float16), \
                                                 h_0[1].unsqueeze(0).to(mindspore.float16))
        if seq_length is not None:
            h = get_hidden(h, seq_length)
            c = get_hidden(c, seq_length)
            mask = sequence_mask(seq_length, x.shape[0])
            outputs = select_by_mask(outputs, mask)
        else:
            h = h[-1]
            c = c[-1]
        return outputs, (h, c)


class _RNNBase(Module):
    '''Basic class for RNN operators'''

    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0., bidirectional=False, dtype=None):
        factory_kwargs = {'dtype': dtype}
        super().__init__()

        if not 0 <= dropout < 1:
            raise ValueError(f"For '{self.cls_name}', the 'dropout' must be a number in range [0, 1) "
                             f"representing the probability of an element being zeroed, but got {dropout}.")

        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                           "recurrent layer, so non-zero dropout expects "
                           "num_layers greater than 1, but got dropout={} and "
                           "num_layers={}".format(dropout, num_layers))

        is_ascend = mindspore.get_context("device_target") == "Ascend"
        if mode == "LSTM":
            gate_size = 4 * hidden_size
            self.rnn = _DynamicLSTMAscend() if is_ascend else _DynamicLSTMCPUGPU()
        elif mode == "GRU":
            if is_ascend and hidden_size % 16 != 0:
                raise ValueError(f"GRU on ascend do not support hidden size that is not divisible by 16, "
                                 f"but get hidden size {hidden_size}, please reset the argument.")
            gate_size = 3 * hidden_size
            self.rnn = _DynamicGRUAscend() if is_ascend else _DynamicGRUCPUGPU()
        elif mode == "RNN_TANH":
            gate_size = hidden_size
            self.rnn = _DynamicRNNTanh()
        elif mode == "RNN_RELU":
            gate_size = hidden_size
            self.rnn = _DynamicRNNRelu()
        else:
            raise ValueError(f"For '{self.cls_name}', the 'mode' must be in ['RNN_RELU', 'RNN_TANH', 'LSTM', 'GRU'], "
                             f"but got {mode}.")

        self.reverse = ReverseV2([0])
        self.reverse_sequence = ReverseSequence(0, 1)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_op = Dropout(p=float(dropout))
        self.bidirectional = bidirectional
        self.bias = bias
        num_directions = 2 if bidirectional else 1
        self.is_lstm = mode == "LSTM"

        self.w_ih_list = []
        self.w_hh_list = []
        self.b_ih_list = []
        self.b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                w_ih = Parameter(ops.empty((gate_size, layer_input_size), **factory_kwargs))
                w_hh = Parameter(ops.empty((gate_size, hidden_size), **factory_kwargs))
                self.w_ih_list.append(w_ih)
                self.w_hh_list.append(w_hh)
                if bias:
                    b_ih = Parameter(ops.empty(gate_size, **factory_kwargs))
                    # Second bias vector included for CuDNN compatibility. Only one
                    # bias vector is needed in standard definition.
                    b_hh = Parameter(ops.empty(gate_size, **factory_kwargs))
                    self.b_ih_list.append(b_ih)
                    self.b_hh_list.append(b_hh)

                if bias:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                else:
                    layer_params = (w_ih, w_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def _stacked_bi_dynamic_rnn(self, x, h, seq_length):
        """stacked bidirectional dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            offset = i * 2
            if self.bias:
                w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
                    self.w_ih_list[offset], self.w_hh_list[offset], \
                    self.b_ih_list[offset], self.b_hh_list[offset]
                w_b_ih, w_b_hh, b_b_ih, b_b_hh = \
                    self.w_ih_list[offset + 1], self.w_hh_list[offset + 1], \
                    self.b_ih_list[offset + 1], self.b_hh_list[offset + 1]
            else:
                w_f_ih, w_f_hh = self.w_ih_list[offset], self.w_hh_list[offset]
                w_b_ih, w_b_hh = self.w_ih_list[offset + 1], self.w_hh_list[offset + 1]
                b_f_ih, b_f_hh, b_b_ih, b_b_hh = None, None, None, None
            if self.is_lstm:
                h_f_i = (h[0][offset], h[1][offset])
                h_b_i = (h[0][offset + 1], h[1][offset + 1])
            else:
                h_f_i = h[offset]
                h_b_i = h[offset + 1]
            if seq_length is None:
                x_b = self.reverse(pre_layer)
            else:
                x_b = self.reverse_sequence(pre_layer, seq_length)
            output_f, h_t_f = self.rnn(pre_layer, h_f_i, seq_length, w_f_ih, w_f_hh, b_f_ih, b_f_hh)
            output_b, h_t_b = self.rnn(x_b, h_b_i, seq_length, w_b_ih, w_b_hh, b_b_ih, b_b_hh)
            if seq_length is None:
                output_b = self.reverse(output_b)
            else:
                output_b = self.reverse_sequence(output_b, seq_length)
            output = ops.cat((output_f, output_b), 2)
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t_f[0], h_t_b[0],)
                c_n += (h_t_f[1], h_t_b[1],)
            else:
                h_n += (h_t_f, h_t_b,)
        if self.is_lstm:
            h_n = ops.cat(h_n)
            c_n = ops.cat(c_n)
            h0_shape = h[0].shape
            h1_shape = h[1].shape
            h_n = h_n.view(h0_shape)
            c_n = c_n.view(h1_shape)
            return output, (h_n.view(h0_shape), c_n.view(h1_shape))
        h_n = ops.cat(h_n)
        return output, h_n.view(h.shape)

    def _stacked_dynamic_rnn(self, x, h, seq_length):
        """stacked mutil_layer dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            if self.bias:
                w_ih, w_hh, b_ih, b_hh = self.w_ih_list[i], self.w_hh_list[i], self.b_ih_list[i], self.b_hh_list[i]
            else:
                w_ih, w_hh = self.w_ih_list[i], self.w_hh_list[i]
                b_ih, b_hh = None, None
            if self.is_lstm:
                h_i = (h[0][i], h[1][i])
            else:
                h_i = h[i]
            output, h_t = self.rnn(pre_layer, h_i, seq_length, w_ih, w_hh, b_ih, b_hh)
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t[0],)
                c_n += (h_t[1],)
            else:
                h_n += (h_t,)
        if self.is_lstm:
            h_n = ops.cat(h_n)
            c_n = ops.cat(c_n)
            h0_shape = h[0].shape
            h1_shape = h[1].shape
            h_n = h_n.view(h0_shape)
            c_n = c_n.view(h1_shape)
            return output, (h_n.view(h0_shape), c_n.view(h1_shape))
        h_n = ops.cat(h_n)
        return output, h_n.view(h.shape)

    def forward(self, x, hx=None, seq_length=None):
        '''Defines the RNN like operators performed'''
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        x_dtype = x.dtype
        if hx is None:
            hx = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size), \
                             x_dtype, self.is_lstm)
        if self.batch_first:
            x = ops.permute(x, (1, 0, 2))
        if self.bidirectional:
            x_n, hx_n = self._stacked_bi_dynamic_rnn(x, hx, seq_length)
        else:
            x_n, hx_n = self._stacked_dynamic_rnn(x, hx, seq_length)
        if self.batch_first:
            x_n = ops.permute(x_n, (1, 0, 2))
        if not self.is_lstm:
            return x_n.astype(x_dtype), hx_n.astype(x_dtype)
        return x_n.astype(x_dtype), (hx_n[0].astype(x_dtype), hx_n[1].astype(x_dtype))


class RNN(_RNNBase):
    r"""
    Stacked Elman RNN layers, applying RNN layer with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to the input.

    For each element in the input sequence, each layer computes the following function:

    .. math::
        h_t = activation(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time :math:`t-1` or the initial hidden state at time `0`.
    :math:`W_{ih}` is the learnable input-hidden weights, and :math:`b_{ih}` is the learnable input-hidden bias.
    :math:`W_{hh}` is the learnable hidden-hidden weights, and :math:`b_{hh}` is the learnable hidden-hidden bias.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked RNN. Default: ``1`` .
        nonlinearity (str): The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``.
        bias (bool): Whether the cell has bias :math:`b_{ih}` and :math:`b_{hh}`. Default: ``True`` .
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: ``False`` .
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            RNN layer except the last layer. Default ``0.0`` . The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional RNN,
            num_directions=2 if bidirectional=True otherwise 1. Default: ``False`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 or mindspore.float16 and
          shape :math:`(seq\_len, batch\_size, input\_size)` or :math:`(batch\_size, seq\_len, input\_size)` .
        - **hx** (Tensor) - Tensor of data type mindspore.float32 or mindspore.float16 and
          shape :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` .
        - **seq_length** (Tensor) - The length of each sequence in an input batch.
          Tensor of shape :math:`(batch\_size)` . Default: ``None`` .
          This input indicates the real sequence length before padding to avoid padded elements
          have been used to compute hidden state and affect the final output. It is recommended to
          use this input when `x` has padding elements.

    Outputs:
        Tuple, a tuple contains (`output`, `hx_n`).

        - **output** (Tensor) - Tensor of shape :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` or
          :math:`(batch\_size, seq\_len, num\_directions * hidden\_size)` .
        - **hx_n** (Tensor) - Tensor of shape :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` .

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0).
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> net = ms.nn.RNN(10, 16, 2, bias=True, batch_first=True, bidirectional=False)
        >>> x = ms.Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = ms.Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """

    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError(f"For '{self.cls_name}', the 'nonlinearity' must be in ['tanh', 'relu'], "
                                 f"but got {kwargs['nonlinearity']}.")
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)


class GRU(_RNNBase):
    r"""
    Stacked GRU (Gated Recurrent Unit) layers.

    Apply GRU layer to the input.

    There are two gates in a GRU model. One is update gate and the other is reset gate.
    Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, a hidden state :math:`h_{t-1}`, the update and reset gate at
    time :math:`t` is computed using a gating mechanism. Update gate :math:`z_t` is designed to protect the cell
    from perturbation by irrelevant inputs and past hidden state. Reset gate :math:`r_t` determines how much
    information should be reset from old hidden state. New memory state :math:`n_t` is
    calculated with the current input, on which the reset gate will be applied. Finally, current hidden state
    :math:`h_{t}` is computed with the calculated update grate and new memory state. The complete
    formulation is as follows:

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

    Note:
        When using GRU on Ascend, the hidden size only supports multiples of 16.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU. Default: ``1`` .
        bias (bool): Whether the cell has bias :math:`b_{in}` and :math:`b_{hn}`. Default: ``True`` .
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: ``False`` .
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. Default ``0.0`` . The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional GRU,
            num_directions=2 if bidirectional=True otherwise 1. Default: ``False`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 or mindspore.float16 and
          shape :math:`(seq\_len, batch\_size, input\_size)` or :math:`(batch\_size, seq\_len, input\_size)`.
        - **hx** (Tensor) - Tensor of data type mindspore.float32 or mindspore.float16 and
          shape :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)`.
        - **seq_length** (Tensor) - The length of each sequence in an input batch.
          Tensor of shape :math:`(\text{batch_size})`. Default: ``None`` .
          This input indicates the real sequence length before padding to avoid padded elements
          have been used to compute hidden state and affect the final output. It is recommended to
          use this input when **x** has padding elements.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`).

        - **output** (Tensor) - Tensor of shape :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` or
          :math:`(batch\_size, seq\_len, num\_directions * hidden\_size)`.
        - **hx_n** (Tensor) - Tensor of shape :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)`.

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> net = ms.nn.GRU(10, 16, 2, bias=True, batch_first=True, bidirectional=False)
        >>> x = ms.Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = ms.Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """

    def __init__(self, *args, **kwargs):
        mode = 'GRU'
        super(GRU, self).__init__(mode, *args, **kwargs)


class LSTM(_RNNBase):
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
        num_layers (int): Number of layers of stacked LSTM . Default: ``1`` .
        bias (bool): Whether the cell has bias :math:`b_{ih}` and :math:`b_{fh}`. Default: ``True`` .
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: ``False`` .
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default ``0`` . The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional LSTM,
            num_directions=2 if bidirectional=True otherwise 1. Default: ``False`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 or mindspore.float16 and
          shape :math:`(seq\_len, batch\_size, input\_size)` or :math:`(batch\_size, seq\_len, input\_size)` .
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32
          or mindspore.float16 and shape :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` .
        - **seq_length** (Tensor) - The length of each sequence in an input batch.
          Tensor of shape :math:`(batch\_size)`. Default: ``None`` .
          This input indicates the real sequence length before padding to avoid padded elements
          have been used to compute hidden state and affect the final output. It is recommended to
          use this input when **x** has padding elements.

    Outputs:
        Tuple, a tuple contains (`output`, (`h_n`, `c_n`)).

        - **output** (Tensor) - Tensor of shape :math:`(seq\_len, batch\_size, num\_directions * hidden\_size)` .
        - **hx_n** (tuple) - A tuple of two Tensor (h_n, c_n) both of shape
          :math:`(num\_directions * num\_layers, batch\_size, hidden\_size)` .

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is not a float.
        ValueError: If `dropout` is not in range [0.0, 1.0).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> net = ms.nn.LSTM(10, 16, 2, bias=True, batch_first=True, bidirectional=False)
        >>> x = ms.Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = ms.Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> c0 = ms.Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, (hn, cn) = net(x, (h0, c0))
        >>> print(output.shape)
        (3, 5, 16)
    """

    def __init__(self, *args, **kwargs):
        mode = 'LSTM'
        super(LSTM, self).__init__(mode, *args, **kwargs)
