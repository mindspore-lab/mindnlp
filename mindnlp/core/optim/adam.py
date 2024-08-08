"""adam"""
# pylint: disable=unneeded-not, use-dict-literal
# mypy: allow-untyped-defs
from typing import Tuple, Union

import mindspore
from mindspore import Tensor
from .. import ops
from .optimizer import (
    _get_scalar_dtype,
    Optimizer,
    ParamsT,
)

__all__ = ["Adam"]



class Adam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not ops.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        mindspore.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                        )
                        if group["capturable"] or group["fused"]
                        else mindspore.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def step(self, grads):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None

        start = 0
        for group in self.param_groups:
            end = start + len(group['params'])
            amsgrad = group['amsgrad']
            maximize = group["maximize"]
            for (p, grad) in zip(group['params'], grads[start: end]):
                grad = grad if not maximize else -grad
                start = end

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = ops.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = ops.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = ops.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                # # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(grad, 1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2)
                # if amsgrad:
                #     # Maintains the maximum of all 2nd moment running avg. till now
                #     torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                #     # Use the max. for normalizing running avg. of gradient
                #     denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                # else:
                #     denom = exp_avg_sq.sqrt().add_(group['eps'])

                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.addcdiv_(exp_avg, denom, -step_size)

                beta1_power = beta1 ** state['step']
                beta2_power = beta2 ** state['step']
                if amsgrad:
                    ops.optim.raw_adam_amsgrad(p, exp_avg, exp_avg_sq, max_exp_avg_sq,
                                                       beta1_power, beta2_power, group['lr'], beta1, beta2, group['eps'], grad)
                else:
                    ops.optim.raw_adam(p, exp_avg, exp_avg_sq, beta1_power, beta2_power,
                                               group['lr'], beta1, beta2, group['eps'], grad)
