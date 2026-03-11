"""
SparseAdam optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.SparseAdam.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class SparseAdam(Optimizer):
    """Lazy Adam variant for sparse gradients.

    Implements Adam update only on rows where gradients are non-zero,
    leaving other parameter rows and their running averages untouched.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients used for computing running averages
            (default: (0.9, 0.999)).
        eps: Term added to the denominator (default: 1e-8).
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        self._call_step_pre_hooks()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "exp_avg": zeros_like(p),
                        "exp_avg_sq": zeros_like(p),
                    }

                state = self.state[param_id]
                state["step"] += 1

                dispatch(
                    "_sparse_adam_step", None,
                    p, p.grad,
                    state["exp_avg"], state["exp_avg_sq"],
                    state["step"], lr, beta1, beta2, eps,
                )

        self._call_step_post_hooks()
        return loss
