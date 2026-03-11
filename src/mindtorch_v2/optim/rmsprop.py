"""
RMSprop optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.RMSprop.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import zeros_like


class RMSprop(Optimizer):
    """Implements RMSprop algorithm.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Learning rate (default: 1e-2)
        alpha: Smoothing constant (default: 0.99)
        eps: Term added to the denominator to improve numerical stability
            (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        momentum: Momentum factor (default: 0)
        centered: If True, compute the centered RMSProp (default: False)
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = RMSprop(model.parameters(), lr=0.01, alpha=0.99)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if alpha < 0.0 or alpha >= 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
            group.setdefault("maximize", False)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        self._call_step_pre_hooks()

        for group in self.param_groups:
            alpha = group["alpha"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            lr = group["lr"]
            eps = group["eps"]
            centered = group["centered"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    state = {
                        "step": 0,
                        "square_avg": zeros_like(p),
                    }
                    if momentum > 0:
                        state["momentum_buffer"] = zeros_like(p)
                    if centered:
                        state["grad_avg"] = zeros_like(p)
                    self.state[param_id] = state

                state = self.state[param_id]
                state["step"] += 1

                dispatch(
                    "_rmsprop_step", None,
                    p, p.grad, state["square_avg"],
                    state.get("grad_avg"),
                    state.get("momentum_buffer"),
                    state["step"], lr, alpha, eps,
                    weight_decay, momentum, centered, maximize,
                )

        self._call_step_post_hooks()
        return loss
