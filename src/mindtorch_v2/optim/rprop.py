"""
Rprop optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.Rprop.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like, full_like


class Rprop(Optimizer):
    """Implements the resilient backpropagation algorithm.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Learning rate (default: 1e-2)
        etas: Pair of (etaminus, etaplus), multiplicative increase and
            decrease factors (default: (0.5, 1.2))
        step_sizes: Pair of minimal and maximal allowed step sizes
            (default: (1e-6, 50))
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = Rprop(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1e-2,
        etas: tuple = (0.5, 1.2),
        step_sizes: tuple = (1e-6, 50),
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < etas[0] < 1.0 < etas[1]:
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")

        defaults = dict(
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
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

        for group in self.param_groups:
            lr = group["lr"]
            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "prev": zeros_like(p),
                        "step_sizes": full_like(p, lr),
                    }

                state = self.state[param_id]
                state["step"] += 1

                dispatch(
                    "_rprop_step", None,
                    p, p.grad,
                    state["prev"], state["step_sizes"],
                    lr, etaminus, etaplus,
                    step_size_min, step_size_max,
                    maximize,
                )

        return loss
