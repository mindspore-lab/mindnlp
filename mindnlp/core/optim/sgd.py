"""sgd"""
# pylint: disable=use-dict-literal
# mypy: allow-untyped-defs
import mindspore
from mindspore import Tensor
from .optimizer import (
    Optimizer,
)
from .. import ops

__all__ = ["SGD"]


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    def step(self, grads):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        start = 0
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = Tensor(group['momentum'], mindspore.float32)
            lr = Tensor(group['lr'], mindspore.float32)
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize=group["maximize"]

            end = start + len(group['params'])
            for (p, d_p) in zip(group['params'], grads[start: end]):
                d_p = d_p if not maximize else -d_p
                start = end
                # if weight_decay != 0:
                #     d_p = d_p.add(p, alpha=weight_decay)
                # if momentum != 0:
                #     param_state = self.state[p]
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = d_p.clone()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf = buf.mul(momentum)
                #         buf = buf.add_(d_p, alpha=1 - dampening)
                #     if nesterov:
                #         d_p = d_p.add(momentum, buf)
                #     else:
                #         d_p = buf
                # new_p = p.add(d_p, alpha=-group['lr'])
                # assign(p, new_p)
                stat = ops.ones_like(p)
                accum = ops.zeros_like(p)
                ops.optim.raw_sgd(p, d_p, lr, dampening, weight_decay, nesterov, accum, momentum, stat)

        return loss
