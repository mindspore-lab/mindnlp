import math

from .optimizer import Optimizer
from .._autograd.grad_mode import no_grad


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}
        self._step_count = 0

    def step(self):
        self._step_count += 1
        beta1, beta2 = self.betas

        with no_grad():
            for p in self.params:
                if p.grad is None:
                    continue

                grad_data = p.grad.storage().data

                if self.weight_decay != 0:
                    grad_data = grad_data + self.weight_decay * p.storage().data

                pid = id(p)
                if pid not in self.state:
                    import numpy as np
                    self.state[pid] = {
                        "m": np.zeros_like(p.storage().data),
                        "v": np.zeros_like(p.storage().data),
                    }
                    if self.amsgrad:
                        self.state[pid]["v_max"] = np.zeros_like(p.storage().data)

                s = self.state[pid]
                s["m"] = beta1 * s["m"] + (1 - beta1) * grad_data
                s["v"] = beta2 * s["v"] + (1 - beta2) * grad_data * grad_data

                m_hat = s["m"] / (1 - beta1 ** self._step_count)
                v_hat = s["v"] / (1 - beta2 ** self._step_count)

                if self.amsgrad:
                    import numpy as np
                    s["v_max"] = np.maximum(s["v_max"], v_hat)
                    denom = s["v_max"] ** 0.5 + self.eps
                else:
                    denom = v_hat ** 0.5 + self.eps

                p.storage()._data = p.storage().data - self.lr * m_hat / denom


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}
        self._step_count = 0

    def step(self):
        self._step_count += 1
        beta1, beta2 = self.betas

        with no_grad():
            for p in self.params:
                if p.grad is None:
                    continue

                # Decoupled weight decay: applied to params directly, not to grad
                if self.weight_decay != 0:
                    p.storage()._data = p.storage().data * (1 - self.lr * self.weight_decay)

                grad_data = p.grad.storage().data

                pid = id(p)
                if pid not in self.state:
                    import numpy as np
                    self.state[pid] = {
                        "m": np.zeros_like(p.storage().data),
                        "v": np.zeros_like(p.storage().data),
                    }
                    if self.amsgrad:
                        self.state[pid]["v_max"] = np.zeros_like(p.storage().data)

                s = self.state[pid]
                s["m"] = beta1 * s["m"] + (1 - beta1) * grad_data
                s["v"] = beta2 * s["v"] + (1 - beta2) * grad_data * grad_data

                m_hat = s["m"] / (1 - beta1 ** self._step_count)
                v_hat = s["v"] / (1 - beta2 ** self._step_count)

                if self.amsgrad:
                    import numpy as np
                    s["v_max"] = np.maximum(s["v_max"], v_hat)
                    denom = s["v_max"] ** 0.5 + self.eps
                else:
                    denom = v_hat ** 0.5 + self.eps

                p.storage()._data = p.storage().data - self.lr * m_hat / denom
