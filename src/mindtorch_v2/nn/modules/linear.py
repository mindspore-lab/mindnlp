from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from .. import functional as F


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = tensor([[0.0] * out_features for _ in range(in_features)])
        self.weight = Parameter(w)
        if bias:
            self.bias = Parameter(tensor([0.0] * out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class Bilinear(Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        w = tensor([[[0.0] * in2_features for _ in range(in1_features)] for _ in range(out_features)])
        self.weight = Parameter(w)
        if bias:
            self.bias = Parameter(tensor([0.0] * out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input1, input2):
        raise NotImplementedError("Bilinear forward is not yet implemented")

    def extra_repr(self):
        return (f'in1_features={self.in1_features}, in2_features={self.in2_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None}')


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input
