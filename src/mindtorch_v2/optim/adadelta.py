"""
Adadelta optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.Adadelta.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class Adadelta(Optimizer):
    """Implements Adadelta algorithm.

    Adadelta is an extension of Adagrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Coefficient that scales delta before it is applied to the
            parameters (default: 1.0)
        rho: Coefficient used for computing a running average of squared
            gradients (default: 0.9)
        eps: Term added to the denominator to improve numerical stability
            (default: 1e-6)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = Adadelta(model.parameters(), lr=1.0)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0.0 or rho > 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
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
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "square_avg": zeros_like(p),
                        "acc_delta": zeros_like(p),
                    }

                state = self.state[param_id]
                state["step"] += 1

                dispatch(
                    "_adadelta_step", None,
                    p, p.grad,
                    state["square_avg"], state["acc_delta"],
                    lr, rho, eps, weight_decay, maximize,
                )

        self._call_step_post_hooks()
        return loss
