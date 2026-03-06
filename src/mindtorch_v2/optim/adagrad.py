"""
Adagrad optimizer for mindtorch_v2.

Aligned with PyTorch's torch.optim.Adagrad.
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

from .optimizer import Optimizer
from .._tensor import Tensor
from .._dispatch import dispatch
from .._functional import full_like


class Adagrad(Optimizer):
    """Implements Adagrad algorithm.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups
        lr: Learning rate (default: 1e-2)
        lr_decay: Learning rate decay (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        initial_accumulator_value: Initial value for the accumulator (default: 0)
        eps: Term added to the denominator to improve numerical stability
            (default: 1e-10)
        maximize: Maximize the params based on the objective, instead of
            minimizing (default: False)

    Example:
        >>> optimizer = Adagrad(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Union[Tensor, Dict]],
        lr: float = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        *,
        maximize: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0.0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if initial_accumulator_value < 0.0:
            raise ValueError(f"Invalid initial_accumulator_value value: {initial_accumulator_value}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
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
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            initial_accumulator_value = group["initial_accumulator_value"]
            maximize = group["maximize"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_id = id(p)

                if param_id not in self.state:
                    self.state[param_id] = {
                        "step": 0,
                        "sum": full_like(p, initial_accumulator_value),
                    }

                state = self.state[param_id]
                state["step"] += 1

                dispatch(
                    "_adagrad_step", None,
                    p, p.grad, state["sum"],
                    state["step"], lr, lr_decay, weight_decay, eps,
                    maximize,
                )

        self._call_step_post_hooks()
        return loss
