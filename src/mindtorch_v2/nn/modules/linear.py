import math

from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from ..._functional import add, matmul


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = tensor([[0.0] * out_features for _ in range(in_features)])
        b = tensor([0.0] * out_features)
        self.weight = Parameter(w)
        self.bias = Parameter(b)

    def forward(self, x):
        return add(matmul(x, self.weight), self.bias)
