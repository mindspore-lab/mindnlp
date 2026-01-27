"""AdamW optimizer implementation."""

import math
import numpy as np
import mindspore
from .optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
        amsgrad: Whether to use AMSGrad variant (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Get state for this parameter
                state = self.state[id(p)]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = np.zeros(p.shape, dtype=np.float32)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = np.zeros(p.shape, dtype=np.float32)
                    if amsgrad:
                        # Max of exp_avg_sq
                        state['max_exp_avg_sq'] = np.zeros(p.shape, dtype=np.float32)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Get numpy arrays
                p_np = p.numpy()
                grad_np = grad.numpy()

                # Decoupled weight decay
                if weight_decay != 0:
                    p_np = p_np * (1 - lr * weight_decay)

                # Update biased first moment estimate
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad_np
                state['exp_avg'] = exp_avg

                # Update biased second raw moment estimate
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad_np ** 2)
                state['exp_avg_sq'] = exp_avg_sq

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
                    state['max_exp_avg_sq'] = max_exp_avg_sq
                    denom = (np.sqrt(max_exp_avg_sq) / math.sqrt(bias_correction2)) + eps
                else:
                    denom = (np.sqrt(exp_avg_sq) / math.sqrt(bias_correction2)) + eps

                step_size = lr / bias_correction1

                # Update parameters
                p_np = p_np - step_size * (exp_avg / denom)

                # Write back to tensor
                flat = p_np.astype(np.float32).ravel()
                p._storage._ms_tensor = mindspore.Tensor(flat)
                p._version += 1

        return loss


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                p_np = p.numpy()
                grad_np = grad.numpy()

                # Add weight decay
                if weight_decay != 0:
                    grad_np = grad_np + weight_decay * p_np

                # Apply momentum
                if momentum != 0:
                    state = self.state[id(p)]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = grad_np.copy()
                    else:
                        buf = state['momentum_buffer']
                        buf = momentum * buf + (1 - dampening) * grad_np
                        state['momentum_buffer'] = buf

                    if nesterov:
                        grad_np = grad_np + momentum * state['momentum_buffer']
                    else:
                        grad_np = state['momentum_buffer']

                # Update parameters
                p_np = p_np - lr * grad_np

                # Write back to tensor
                flat = p_np.astype(np.float32).ravel()
                p._storage._ms_tensor = mindspore.Tensor(flat)
                p._version += 1

        return loss
