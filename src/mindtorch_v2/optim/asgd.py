"""
ASGD (Averaged SGD) optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.ASGD.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent.

    ASGD maintains a running average of the iterates that converges better
    than standard SGD for some problems.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Learning rate (default: 1e-2)
        lambd: Decay term (default: 1e-4)
        alpha: Power for eta update (default: 0.75)
        t0: Point at which to start averaging (default: 1e6)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = ASGD(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        self._call_step_pre_hooks()

        for group in self.param_groups:
            lr = group["lr"]
            lambd = group["lambd"]
            alpha = group["alpha"]
            t0 = group["t0"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "ax": zeros_like(p),
                    }

                state = self.state[param_id]
                state["step"] += 1

                dispatch(
                    "_asgd_step", None,
                    p, p.grad, state["ax"],
                    state["step"], lr, lambd, alpha, t0,
                    weight_decay, maximize,
                )

        self._call_step_post_hooks()
        return loss
