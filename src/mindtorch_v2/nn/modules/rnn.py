import math

from ..module import Module
from ..parameter import Parameter
from ..._creation import zeros, randn
from .. import functional as F


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

    def forward(self, input, hx=None):
        raise NotImplementedError(f"{self.mode} forward is not yet implemented")

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
