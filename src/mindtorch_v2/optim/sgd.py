from .optimizer import Optimizer
from .._autograd.grad_mode import no_grad


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params)
        self.lr = lr

    def step(self):
        with no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p.storage()._data = p.storage().data - self.lr * p.grad.storage().data
