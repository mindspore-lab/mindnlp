from ..module import Module
from ..._functional import relu


class ReLU(Module):
    def forward(self, x):
        return relu(x)
