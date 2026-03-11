"""
SGD (Stochastic Gradient Descent) optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.SGD.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class SGD(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        dampening: Dampening for momentum (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 0.01,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
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

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        self._call_step_pre_hooks()

        for group in self.param_groups:
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            lr = group["lr"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if momentum != 0:
                    if param_id not in self.state:
                        self.state[param_id] = {
                            "momentum_buffer": zeros_like(p),
                        }
                    buf = self.state[param_id]["momentum_buffer"]
                else:
                    buf = None

                dispatch(
                    "_sgd_step", None,
                    p, p.grad, buf,
                    lr, momentum, dampening, weight_decay,
                    nesterov, maximize,
                )

        self._call_step_post_hooks()
        return loss
