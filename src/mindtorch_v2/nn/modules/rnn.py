import math

from ..module import Module
from ..parameter import Parameter
from ..._creation import zeros, randn
from ..._functional import stack, cat
from .. import functional as F


def _rnn_cell_forward(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh, nonlinearity='tanh'):
    gate = F.linear(input, weight_ih, bias_ih) + F.linear(hidden, weight_hh, bias_hh)
    if nonlinearity == 'tanh':
        return F.tanh(gate)
    else:
        return F.relu(gate)


def _lstm_cell_forward(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh):
    h, c = hidden
    gates = F.linear(input, weight_ih, bias_ih) + F.linear(h, weight_hh, bias_hh)
    i, f, g, o = gates.chunk(4, dim=1)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    g = F.tanh(g)
    o = F.sigmoid(o)
    c_next = f * c + i * g
    h_next = o * F.tanh(c_next)
    return h_next, c_next


def _gru_cell_forward(input, hidden, weight_ih, weight_hh, bias_ih, bias_hh):
    gates_x = F.linear(input, weight_ih, bias_ih)
    gates_h = F.linear(hidden, weight_hh, bias_hh)
    r_x, z_x, n_x = gates_x.chunk(3, dim=1)
    r_h, z_h, n_h = gates_h.chunk(3, dim=1)
    r = F.sigmoid(r_x + r_h)
    z = F.sigmoid(z_x + z_h)
    n = F.tanh(n_x + r * n_h)
    h_next = (1 - z) * n + z * hidden
    return h_next


def _run_rnn_layer(mode, input_seq, h_0, weight_ih, weight_hh, bias_ih, bias_hh,
                   reverse=False, nonlinearity='tanh'):
    """Run one RNN layer over a sequence. Returns (output_seq, h_n)."""
    seq_len = input_seq.shape[0]
    steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    outputs = []
    if mode == 'LSTM':
        h, c = h_0
        for t in steps:
            x_t = input_seq[t]
            h, c = _lstm_cell_forward(x_t, (h, c), weight_ih, weight_hh, bias_ih, bias_hh)
            outputs.append(h)
        if reverse:
            outputs = outputs[::-1]
        output = stack(outputs, dim=0)
        return output, (h, c)
    else:
        h = h_0
        cell_fn = _rnn_cell_forward if mode == 'RNN' else _gru_cell_forward
        for t in steps:
            x_t = input_seq[t]
            if mode == 'RNN':
                h = cell_fn(x_t, h, weight_ih, weight_hh, bias_ih, bias_hh, nonlinearity)
            else:
                h = cell_fn(x_t, h, weight_ih, weight_hh, bias_ih, bias_hh)
            outputs.append(h)
        if reverse:
            outputs = outputs[::-1]
        output = stack(outputs, dim=0)
        return output, h


class RNNBase(Module):
    def __init__(self, mode, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = '_reverse' if direction == 1 else ''
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
                gate_size = hidden_size
                if mode == 'LSTM':
                    gate_size *= 4
                elif mode == 'GRU':
                    gate_size *= 3
                w_ih = zeros(gate_size, layer_input_size)
                w_hh = zeros(gate_size, hidden_size)
                setattr(self, f'weight_ih_l{layer}{suffix}', Parameter(w_ih))
                setattr(self, f'weight_hh_l{layer}{suffix}', Parameter(w_hh))
                if bias:
                    b_ih = zeros(gate_size)
                    b_hh = zeros(gate_size)
                    setattr(self, f'bias_ih_l{layer}{suffix}', Parameter(b_ih))
                    setattr(self, f'bias_hh_l{layer}{suffix}', Parameter(b_hh))

    def _get_weight(self, name):
        return getattr(self, name)

    def forward(self, input, hx=None):
        is_batch_first = self.batch_first
        if is_batch_first:
            # (batch, seq, feature) -> (seq, batch, feature)
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.shape
        num_directions = self.num_directions

        # Initialize hidden state
        if hx is None:
            if self.mode == 'LSTM':
                h_zeros = zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
                hx = (h_zeros, zeros(self.num_layers * num_directions, batch_size, self.hidden_size))
            else:
                hx = zeros(self.num_layers * num_directions, batch_size, self.hidden_size)

        # Split hidden state per layer and direction
        if self.mode == 'LSTM':
            h_0, c_0 = hx
        else:
            h_0 = hx

        h_n_list = []
        c_n_list = []
        layer_input = input

        for layer in range(self.num_layers):
            dir_outputs = []
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                idx = layer * num_directions + direction
                w_ih = self._get_weight(f'weight_ih_l{layer}{suffix}')
                w_hh = self._get_weight(f'weight_hh_l{layer}{suffix}')
                b_ih = self._get_weight(f'bias_ih_l{layer}{suffix}') if self.bias else None
                b_hh = self._get_weight(f'bias_hh_l{layer}{suffix}') if self.bias else None

                if self.mode == 'LSTM':
                    h_layer = (h_0[idx], c_0[idx])
                else:
                    h_layer = h_0[idx]

                nonlinearity = getattr(self, 'nonlinearity', 'tanh')
                output, h_final = _run_rnn_layer(
                    self.mode, layer_input, h_layer,
                    w_ih, w_hh, b_ih, b_hh,
                    reverse=(direction == 1),
                    nonlinearity=nonlinearity,
                )
                dir_outputs.append(output)

                if self.mode == 'LSTM':
                    h_n_list.append(h_final[0])
                    c_n_list.append(h_final[1])
                else:
                    h_n_list.append(h_final)

            # Concatenate directions
            if num_directions == 2:
                layer_input = cat(dir_outputs, dim=2)
            else:
                layer_input = dir_outputs[0]

            # Apply dropout between layers (not after last layer)
            if self.dropout > 0 and layer < self.num_layers - 1:
                layer_input = F.dropout(layer_input, p=self.dropout, training=self.training)

        output = layer_input

        # Stack h_n (and c_n for LSTM)
        h_n = stack(h_n_list, dim=0)
        if self.mode == 'LSTM':
            c_n = stack(c_n_list, dim=0)
            hidden_out = (h_n, c_n)
        else:
            hidden_out = h_n

        if is_batch_first:
            output = output.transpose(0, 1)

        return output, hidden_out

    def extra_repr(self):
        s = f'{self.input_size}, {self.hidden_size}'
        if self.num_layers != 1:
            s += f', num_layers={self.num_layers}'
        if not self.bias:
            s += f', bias={self.bias}'
        if self.batch_first:
            s += ', batch_first=True'
        if self.dropout:
            s += f', dropout={self.dropout}'
        if self.bidirectional:
            s += ', bidirectional=True'
        return s


class RNN(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh',
                 bias=True, batch_first=False, dropout=0.0, bidirectional=False,
                 device=None, dtype=None):
        self.nonlinearity = nonlinearity
        super().__init__('RNN', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, device, dtype)


class LSTM(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False, proj_size=0,
                 device=None, dtype=None):
        self.proj_size = proj_size
        super().__init__('LSTM', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, device, dtype)


class GRU(RNNBase):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 device=None, dtype=None):
        super().__init__('GRU', input_size, hidden_size, num_layers, bias,
                         batch_first, dropout, bidirectional, device, dtype)


class RNNCellBase(Module):
    """Base class for RNN cells."""

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(zeros(num_chunks * hidden_size, input_size))
        self.weight_hh = Parameter(zeros(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(zeros(num_chunks * hidden_size))
            self.bias_hh = Parameter(zeros(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            new_data = randn(*weight.data.shape) * stdv
            weight.data.copy_(new_data)

    def extra_repr(self):
        s = f'{self.input_size}, {self.hidden_size}'
        if not self.bias:
            s += ', bias=False'
        return s


class RNNCell(RNNCellBase):
    """Elman RNN cell with tanh or ReLU non-linearity.

    h' = tanh(x @ W_ih^T + b_ih + h @ W_hh^T + b_hh)
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, input, hx=None):
        if hx is None:
            hx = zeros(input.shape[0], self.hidden_size)
        gate = F.linear(input, self.weight_ih, self.bias_ih) + \
               F.linear(hx, self.weight_hh, self.bias_hh)
        if self.nonlinearity == 'tanh':
            return F.tanh(gate)
        elif self.nonlinearity == 'relu':
            return F.relu(gate)
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity!r}")

    def extra_repr(self):
        s = super().extra_repr()
        if self.nonlinearity != 'tanh':
            s += f', nonlinearity={self.nonlinearity!r}'
        return s


class LSTMCell(RNNCellBase):
    """Long Short-Term Memory (LSTM) cell.

    (h', c') = LSTMCell(x, (h, c))
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, input, hx=None):
        if hx is None:
            hx = (
                zeros(input.shape[0], self.hidden_size),
                zeros(input.shape[0], self.hidden_size),
            )
        h, c = hx
        gates = F.linear(input, self.weight_ih, self.bias_ih) + \
                F.linear(h, self.weight_hh, self.bias_hh)
        # Split into 4 gates: input, forget, cell, output
        i, f, g, o = gates.chunk(4, dim=1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        g = F.tanh(g)
        o = F.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * F.tanh(c_next)
        return h_next, c_next


class GRUCell(RNNCellBase):
    """Gated Recurrent Unit (GRU) cell.

    h' = GRUCell(x, h)
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, input, hx=None):
        if hx is None:
            hx = zeros(input.shape[0], self.hidden_size)
        # Compute input gates (all 3 at once)
        gates_x = F.linear(input, self.weight_ih, self.bias_ih)
        gates_h = F.linear(hx, self.weight_hh, self.bias_hh)
        # Split into r (reset), z (update), n (new) components
        r_x, z_x, n_x = gates_x.chunk(3, dim=1)
        r_h, z_h, n_h = gates_h.chunk(3, dim=1)
        r = F.sigmoid(r_x + r_h)
        z = F.sigmoid(z_x + z_h)
        n = F.tanh(n_x + r * n_h)
        h_next = (1 - z) * n + z * hx
        return h_next
