from ..module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = {str(i): m for i, m in enumerate(modules)}

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x
