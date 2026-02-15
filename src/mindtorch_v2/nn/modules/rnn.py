from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor


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
                w_ih = tensor([[0.0] * layer_input_size for _ in range(gate_size)])
                w_hh = tensor([[0.0] * hidden_size for _ in range(gate_size)])
                setattr(self, f'weight_ih_l{layer}{suffix}', Parameter(w_ih))
                setattr(self, f'weight_hh_l{layer}{suffix}', Parameter(w_hh))
                if bias:
                    b_ih = tensor([0.0] * gate_size)
                    b_hh = tensor([0.0] * gate_size)
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
