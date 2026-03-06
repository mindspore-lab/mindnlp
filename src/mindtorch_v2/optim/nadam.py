"""
NAdam optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.NAdam.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class NAdam(Optimizer):
    """Implements NAdam algorithm (Adam with Nesterov momentum).

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Learning rate (default: 2e-3)
        betas: Coefficients used for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        momentum_decay: Momentum decay (default: 4e-3)
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = NAdam(model.parameters(), lr=0.002)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 2e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
        *,
        maximize: bool = False,
    ):
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("momentum_decay", 4e-3)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum_decay = group["momentum_decay"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "exp_avg": zeros_like(p),
                        "exp_avg_sq": zeros_like(p),
                        "mu_product": 1.0,
                    }

                state = self.state[param_id]
                state["step"] += 1
                step = state["step"]

                # Compute momentum schedule (NAdam uses decaying momentum)
                mu = beta1 * (1 - 0.5 * 0.96 ** (step * momentum_decay / 4e-3))
                mu_next = beta1 * (1 - 0.5 * 0.96 ** ((step + 1) * momentum_decay / 4e-3))
                mu_product = state["mu_product"] * mu
                mu_product_next = mu_product * mu_next
                state["mu_product"] = mu_product

                dispatch(
                    "_nadam_step", None,
                    p, p.grad,
                    state["exp_avg"], state["exp_avg_sq"],
                    step, lr, beta1, beta2, eps, weight_decay,
                    mu, mu_next, mu_product, mu_product_next,
                    maximize,
                )

        return loss
