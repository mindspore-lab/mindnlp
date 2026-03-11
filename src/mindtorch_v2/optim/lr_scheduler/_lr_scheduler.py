"""
Learning rate schedulers for mindtorch_v2.

Aligned with PyTorch's torch.optim.lr_scheduler.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Union


class LRScheduler:
    """Base class for learning rate schedulers.

    All schedulers should inherit from this class and implement the _step() method.

    Args:
        optimizer: Wrapped optimizer.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        ...     train(...)
        ...     validate(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._initial_lr = None
        self._step_count = 0

        # Initialize base learning rates
        self.base_lrs = []
        if hasattr(optimizer, "param_groups"):
            for group in optimizer.param_groups:
                if "initial_lr" not in group:
                    group["initial_lr"] = group["lr"]
                self.base_lrs.append(group["initial_lr"])
        else:
            # Fallback for old-style optimizer
            self.base_lrs = [optimizer.lr]

        self.step()

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler as a dict."""
        return {
            "last_epoch": self.last_epoch,
            "_step_count": self._step_count,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the scheduler state."""
        self.last_epoch = state_dict["last_epoch"]
        self._step_count = state_dict.get("_step_count", 0)
        self.base_lrs = state_dict.get("base_lrs", [])

    def get_last_lr(self) -> List[float]:
        """Returns last computed learning rate by current scheduler."""
        return self._last_lr

    def get_lr(self) -> List[float]:
        """Compute learning rate for each parameter group."""
        raise NotImplementedError("get_lr() must be implemented by subclass")

    def step(self, epoch: Optional[int] = None) -> None:
        """Step the learning rate scheduler.

        Args:
            epoch: Specify the epoch number. If None, increments epoch by 1.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._step_count += 1

        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.get_lr()[i]
            param_group["lr"] = new_lr

        self._last_lr = self.get_lr()

        if self.verbose:
            print(f"Epoch {self.last_epoch}: adjusting learning rate to {self._last_lr}")

    def print_lr(self, is_verbose: bool, group: int, lr: float, epoch: Optional[int] = None) -> None:
        """Display the current learning rate."""
        if is_verbose:
            if epoch is None:
                print(f"Adjusting learning rate to {lr:.4e}")
            else:
                print(f"Epoch {epoch}: adjusting learning rate to {lr:.4e}")


class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Args:
        optimizer: Wrapped optimizer.
        step_size: Period of learning rate decay.
        gamma: Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> # lr = 0.1 for epochs 0-29, 0.01 for 30-59, 0.001 for 60-89, ...
    """

    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["step_size"] = self.step_size
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.step_size = state_dict.get("step_size")
        self.gamma = state_dict.get("gamma", 0.1)


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones.

    Args:
        optimizer: Wrapped optimizer.
        milestones: List of epoch indices. Must be increasing.
        gamma: Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        >>> # lr = 0.1 for epochs 0-29, 0.01 for 30-79, 0.001 for 80-
    """

    def __init__(
        self,
        optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        # Count how many milestones we've passed
        milestones_passed = sum(1 for m in self.milestones if self.last_epoch >= m)
        multiplier = self.gamma ** milestones_passed
        return [base_lr * multiplier for base_lr in self.base_lrs]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["milestones"] = self.milestones
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.milestones = state_dict.get("milestones", [])
        self.gamma = state_dict.get("gamma", 0.1)


class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    Args:
        optimizer: Wrapped optimizer.
        gamma: Multiplicative factor of learning rate decay.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        gamma: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        return [base_lr * (self.gamma ** self.last_epoch) for base_lr in self.base_lrs]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.gamma = state_dict.get("gamma")


class CosineAnnealingLR(LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing schedule.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate. Default: 0.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch > self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        else:
            return [
                self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs
            ]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["T_max"] = self.T_max
        state["eta_min"] = self.eta_min
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.T_max = state_dict.get("T_max")
        self.eta_min = state_dict.get("eta_min", 0)


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small
    multiplicative factor until the number of epoch reaches a pre-defined milestone: end_epoch.

    Args:
        optimizer: Wrapped optimizer.
        start_factor: The number we multiply learning rate in the first epoch.
            Default: 1.0 / 3.
        end_factor: The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters: The number of iterations that multiplicative factor reaches
            end_factor. Default: 5.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return [base_lr * self.start_factor for base_lr in self.base_lrs]
        elif self.last_epoch >= self.total_iters:
            return [base_lr * self.end_factor for base_lr in self.base_lrs]
        else:
            # Linear interpolation
            factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
            return [base_lr * factor for base_lr in self.base_lrs]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["start_factor"] = self.start_factor
        state["end_factor"] = self.end_factor
        state["total_iters"] = self.total_iters
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.start_factor = state_dict.get("start_factor", 1.0 / 3)
        self.end_factor = state_dict.get("end_factor", 1.0)
        self.total_iters = state_dict.get("total_iters", 5)


class ConstantLR(LRScheduler):
    """Decays the learning rate of each parameter group by a small
    constant factor until the number of epoch reaches a pre-defined milestone: end_epoch.

    Args:
        optimizer: Wrapped optimizer.
        factor: The number we multiply learning rate until the milestone.
            Default: 1.0 / 3.
        total_iters: The number of steps that the scheduler decays the learning rate.
            Default: 5.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch >= self.total_iters:
            return self.base_lrs
        return [base_lr * self.factor for base_lr in self.base_lrs]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["factor"] = self.factor
        state["total_iters"] = self.total_iters
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.factor = state_dict.get("factor", 1.0 / 3)
        self.total_iters = state_dict.get("total_iters", 5)


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.

    Args:
        optimizer: Wrapped optimizer.
        mode: One of min, max. In min mode, lr will be reduced when the quantity
            monitored has stopped decreasing; in max mode it will be reduced when
            the quantity monitored has stopped increasing. Default: 'min'.
        factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
            Default: 0.1.
        patience: Number of epochs with no improvement after which learning rate will
            be reduced. For example, if patience = 2, then we will ignore the first 2
            epochs with no improvement, and will only decrease the LR after the 3rd epoch
            if the loss still hasn't improved then. Default: 10.
        threshold: Threshold for measuring the new optimum, to only focus on significant
            changes. Default: 1e-4.
        threshold_mode: One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold )
            in 'max' mode or best * ( 1 - threshold ) in min mode. In abs mode,
            dynamic_threshold = best + threshold in max mode or best - threshold in min mode.
            Default: 'rel'.
        cooldown: Number of epochs to wait before resuming normal operation after lr has
            been reduced. Default: 0.
        min_lr: A scalar or a list of scalars. A lower bound on the learning rate of all
            param groups or each group respectively. Default: 0.
        eps: Minimal decay applied to lr. If the difference between new and old lr is smaller
            than eps, the update is ignored. Default: 1e-8.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose

        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.best = None
        self._last_lr = None

        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold_mode must be 'rel' or 'abs', got {threshold_mode}")

        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self) -> None:
        """Resets num_bad_epochs and cooldown_counter."""
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        if mode == "min" and threshold_mode == "rel":
            self.is_better = lambda a, best: a < best * (1 - threshold)
        elif mode == "min" and threshold_mode == "abs":
            self.is_better = lambda a, best: a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            self.is_better = lambda a, best: a > best * (1 + threshold)
        else:  # mode == "max" and threshold_mode == "abs"
            self.is_better = lambda a, best: a > best + threshold

    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        """Step the scheduler based on a metric.

        Args:
            metrics: Metric value to monitor.
            epoch: Epoch number. Not used, kept for API compatibility.
        """
        current = float(metrics)

        if epoch is None:
            epoch = self.last_epoch + 1 if hasattr(self, "last_epoch") else 0  # pylint: disable=access-member-before-definition
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch: int) -> None:
        """Reduce learning rate by factor."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.")

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the scheduler as a dict."""
        return {
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "cooldown_counter": self.cooldown_counter,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the scheduler state."""
        self.best = state_dict.get("best")
        self.num_bad_epochs = state_dict.get("num_bad_epochs", 0)
        self.cooldown_counter = state_dict.get("cooldown_counter", 0)


class LambdaLR(LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function.

    Args:
        optimizer: Wrapped optimizer.
        lr_lambda: A function which computes a multiplicative factor given an
            integer parameter epoch, or a list of such functions, one for each
            group in optimizer.param_groups.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.1)
        >>> scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    """

    def __init__(
        self,
        optimizer,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        return [
            base_lr * lmbda(self.last_epoch)
            for base_lr, lmbda in zip(self.base_lrs, self.lr_lambdas)
        ]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["lr_lambdas"] = [None] * len(self.lr_lambdas)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # pylint: disable=useless-parent-delegation
        super().load_state_dict(state_dict)


class CosineAnnealingWarmRestarts(LRScheduler):
    """Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts (SGDR).

    Args:
        optimizer: Wrapped optimizer.
        T_0: Number of iterations for the first restart.
        T_mult: A factor increases T_i after a restart. Default: 1.
        eta_min: Minimum learning rate. Default: 0.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) // (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        self._step_count += 1

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.get_lr()[i]

        self._last_lr = self.get_lr()

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["T_0"] = self.T_0
        state["T_mult"] = self.T_mult
        state["eta_min"] = self.eta_min
        state["T_i"] = self.T_i
        state["T_cur"] = self.T_cur
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.T_0 = state_dict.get("T_0")
        self.T_mult = state_dict.get("T_mult", 1)
        self.eta_min = state_dict.get("eta_min", 0)
        self.T_i = state_dict.get("T_i")
        self.T_cur = state_dict.get("T_cur")


class OneCycleLR(LRScheduler):
    """Sets the learning rate according to the 1cycle learning rate policy.

    Args:
        optimizer: Wrapped optimizer.
        max_lr: Upper learning rate boundary in the cycle.
        total_steps: The total number of steps in the cycle. Default: None.
        epochs: The number of epochs to train for. Default: None.
        steps_per_epoch: The number of steps per epoch. Default: None.
        pct_start: The percentage of the cycle spent increasing the learning
            rate. Default: 0.3.
        anneal_strategy: Specifies the annealing strategy: "cos" for cosine
            annealing, "linear" for linear annealing. Default: "cos".
        div_factor: Determines the initial learning rate via
            initial_lr = max_lr/div_factor. Default: 25.
        final_div_factor: Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor. Default: 1e4.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if total_steps is None and epochs is None:
            raise ValueError("Must specify either total_steps or epochs")
        if total_steps is None:
            if steps_per_epoch is None:
                raise ValueError("Must specify steps_per_epoch when epochs is specified")
            total_steps = epochs * steps_per_epoch
        if total_steps <= 0:
            raise ValueError(f"Expected positive total_steps, got {total_steps}")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError(f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy}")

        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        if isinstance(max_lr, (list, tuple)):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        # Set initial lr for each param group
        for group, max_lr_val in zip(optimizer.param_groups, self.max_lrs):
            group["initial_lr"] = max_lr_val / div_factor
            group["max_lr"] = max_lr_val
            group["min_lr"] = group["initial_lr"] / final_div_factor

        super().__init__(optimizer, last_epoch, verbose)

    def _annealing_func(self, start, end, pct):
        if self.anneal_strategy == "cos":
            return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)
        else:
            return (end - start) * pct + start

    def get_lr(self) -> List[float]:
        step_num = self.last_epoch
        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. The specified total_steps is {self.total_steps}"
            )

        lrs = []
        for group in self.optimizer.param_groups:
            initial_lr = group.get("initial_lr", group["lr"])
            max_lr = group.get("max_lr", initial_lr * self.div_factor)
            min_lr = group.get("min_lr", initial_lr / self.final_div_factor)

            if step_num <= self.total_steps * self.pct_start:
                # Phase 1: warmup
                pct = step_num / (self.total_steps * self.pct_start) if self.pct_start > 0 else 1.0
                lr = self._annealing_func(initial_lr, max_lr, pct)
            else:
                # Phase 2: annealing
                pct = (step_num - self.total_steps * self.pct_start) / (
                    self.total_steps * (1 - self.pct_start)
                )
                lr = self._annealing_func(max_lr, min_lr, pct)
            lrs.append(lr)
        return lrs

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["total_steps"] = self.total_steps
        state["pct_start"] = self.pct_start
        state["anneal_strategy"] = self.anneal_strategy
        state["div_factor"] = self.div_factor
        state["final_div_factor"] = self.final_div_factor
        state["max_lrs"] = self.max_lrs
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.total_steps = state_dict.get("total_steps")
        self.pct_start = state_dict.get("pct_start", 0.3)
        self.anneal_strategy = state_dict.get("anneal_strategy", "cos")
        self.div_factor = state_dict.get("div_factor", 25.0)
        self.final_div_factor = state_dict.get("final_div_factor", 1e4)
        self.max_lrs = state_dict.get("max_lrs", [])


class SequentialLR(LRScheduler):
    """Receives the list of schedulers that is expected to be called sequentially
    during optimization process and milestone points that provide exact intervals
    to reflect which scheduler is supposed to be called at a given epoch.

    Args:
        optimizer: Wrapped optimizer.
        schedulers: List of chained schedulers.
        milestones: List of integers that reflects milestone points.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(  # pylint: disable=super-init-not-called
        self,
        optimizer,
        schedulers: List["LRScheduler"],
        milestones: List[int],
        last_epoch: int = -1,
    ):
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                f"Expected {len(schedulers) - 1} milestones, got {len(milestones)}"
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._step_count = 0
        self.base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self) -> List[float]:
        idx = 0
        for i, milestone in enumerate(self._milestones):
            if self.last_epoch >= milestone:
                idx = i + 1
        return self._schedulers[idx].get_lr()

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._step_count += 1

        idx = 0
        for i, milestone in enumerate(self._milestones):
            if self.last_epoch >= milestone:
                idx = i + 1

        scheduler = self._schedulers[idx]
        # Compute the local epoch for this scheduler
        if idx > 0:
            local_epoch = self.last_epoch - self._milestones[idx - 1]
        else:
            local_epoch = self.last_epoch

        scheduler.last_epoch = local_epoch
        lrs = scheduler.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lrs[i]
        self._last_lr = lrs

    def state_dict(self) -> Dict[str, Any]:
        return {
            "last_epoch": self.last_epoch,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.last_epoch = state_dict["last_epoch"]
        self._step_count = state_dict.get("_step_count", 0)


class ChainedScheduler:
    """Chains list of learning rate schedulers. It takes a list of chainable
    learning rate schedulers and performs consecutive step() functions belonging
    to them by just one call.

    Args:
        schedulers: List of chained schedulers.
    """

    def __init__(self, schedulers: List["LRScheduler"]):
        if len(schedulers) < 1:
            raise ValueError("ChainedScheduler expects at least one scheduler")
        self._schedulers = schedulers

    def step(self) -> None:
        for scheduler in self._schedulers:
            scheduler.step()

    def get_last_lr(self) -> List[float]:
        return self._schedulers[-1].get_last_lr()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schedulers": [s.state_dict() for s in self._schedulers],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for s, s_dict in zip(self._schedulers, state_dict.get("schedulers", [])):
            s.load_state_dict(s_dict)


class PolynomialLR(LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function.

    Args:
        optimizer: Wrapped optimizer.
        total_iters: The number of steps that the scheduler decays the learning
            rate. Default: 5.
        power: The power of the polynomial. Default: 1.0.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        factor = (
            (1 - self.last_epoch / self.total_iters)
            / (1 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power

        return [group["lr"] * factor for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["total_iters"] = self.total_iters
        state["power"] = self.power
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.total_iters = state_dict.get("total_iters", 5)
        self.power = state_dict.get("power", 1.0)


class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given
    in the specified function.

    Unlike LambdaLR which computes lr = base_lr * lambda(epoch), MultiplicativeLR
    computes lr = lr * lambda(epoch) (cumulative multiplication).

    Args:
        optimizer: Wrapped optimizer.
        lr_lambda: A function which computes a multiplicative factor given an
            integer parameter epoch, or a list of such functions, one for each
            group in optimizer.param_groups.
        last_epoch: The index of last epoch. Default: -1.
        verbose: If True, prints a message to stdout for each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        if callable(lr_lambda):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        self._factor_products = [1.0] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs
        return [
            base_lr * fp
            for base_lr, fp in zip(self.base_lrs, self._factor_products)
        ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._step_count += 1

        if epoch > 0:
            for i, lmbda in enumerate(self.lr_lambdas):
                self._factor_products[i] *= lmbda(epoch)

        lrs = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lrs[i]

        self._last_lr = lrs


class CyclicLR:
    """Sets the learning rate according to a cyclical learning rate policy.

    The policy cycles the learning rate between base_lr and max_lr.
    This scheduler is designed to be called after each batch (not epoch).

    Args:
        optimizer: Wrapped optimizer.
        base_lr: Lower learning rate boundary in the cycle.
        max_lr: Upper learning rate boundary in the cycle.
        step_size_up: Number of training iterations in the increasing half
            of a cycle. Default: 2000.
        step_size_down: Number of training iterations in the decreasing half
            of a cycle. If None, it is set to step_size_up. Default: None.
        mode: One of 'triangular', 'triangular2', 'exp_range'.
            Default: 'triangular'.
        gamma: Constant in 'exp_range' scaling function: gamma^(cycle iterations).
            Default: 1.0.
        scale_fn: Custom scaling policy defined by a single argument lambda
            function. If specified, mode is ignored. Default: None.
        scale_mode: One of 'cycle', 'iterations'. Defines whether scale_fn
            is evaluated on cycle number or cycle iterations.
            Default: 'cycle'.
        cycle_momentum: If True, momentum is cycled inversely to learning
            rate between base_momentum and max_momentum. Default: True.
        base_momentum: Lower momentum boundary in the cycle. Default: 0.8.
        max_momentum: Upper momentum boundary in the cycle. Default: 0.9.
        last_epoch: The index of the last batch. Default: -1.
        verbose: If True, prints a message on each update. Default: False.
    """

    def __init__(
        self,
        optimizer,
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[float], float]] = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.8,
        max_momentum: Union[float, List[float]] = 0.9,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.optimizer = optimizer
        self.verbose = verbose
        self.cycle_momentum = cycle_momentum
        self.last_epoch = 0 if last_epoch == -1 else last_epoch
        self._step_count = 0

        n_groups = len(optimizer.param_groups)

        # Normalize base_lr / max_lr to lists
        if isinstance(base_lr, (list, tuple)):
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * n_groups

        if isinstance(max_lr, (list, tuple)):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * n_groups

        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down is not None else step_size_up
        self.total_size = self.step_size_up + self.step_size_down

        if isinstance(base_momentum, (list, tuple)):
            self.base_momentums = list(base_momentum)
        else:
            self.base_momentums = [base_momentum] * n_groups

        if isinstance(max_momentum, (list, tuple)):
            self.max_momentums = list(max_momentum)
        else:
            self.max_momentums = [max_momentum] * n_groups

        if scale_fn is not None:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        elif mode == "triangular":
            self.scale_fn = lambda x: 1.0
            self.scale_mode = "cycle"
        elif mode == "triangular2":
            self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
            self.scale_mode = "cycle"
        elif mode == "exp_range":
            self.scale_fn = lambda x: gamma ** x
            self.scale_mode = "iterations"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Set initial LRs
        for group, blr in zip(optimizer.param_groups, self.base_lrs):
            group["lr"] = blr
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def _compute_scale(self, iteration, cycle):
        if self.scale_mode == "cycle":
            return self.scale_fn(cycle)
        return self.scale_fn(iteration)

    def get_lr(self) -> List[float]:
        iteration = self.last_epoch
        cycle = math.floor(1 + iteration / self.total_size)
        x = 1 + iteration / self.total_size
        cycle_x = x - math.floor(x)

        if cycle_x <= self.step_size_up / self.total_size:
            scale_x = cycle_x * self.total_size / self.step_size_up
        else:
            scale_x = (cycle_x * self.total_size - self.step_size_up) / self.step_size_down
            scale_x = 1.0 - scale_x

        scale = self._compute_scale(iteration, cycle)

        return [
            blr + (mlr - blr) * scale_x * scale
            for blr, mlr in zip(self.base_lrs, self.max_lrs)
        ]

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1
        self._step_count += 1

        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = lr
        self._last_lr = new_lrs

        if self.cycle_momentum:
            iteration = self.last_epoch
            x = 1 + iteration / self.total_size
            cycle_x = x - math.floor(x)

            if cycle_x <= self.step_size_up / self.total_size:
                scale_x = cycle_x * self.total_size / self.step_size_up
            else:
                scale_x = (cycle_x * self.total_size - self.step_size_up) / self.step_size_down
                scale_x = 1.0 - scale_x

            for i, group in enumerate(self.optimizer.param_groups):
                momentum = self.max_momentums[i] - (self.max_momentums[i] - self.base_momentums[i]) * scale_x
                if "betas" in group:
                    group["betas"] = (momentum, group["betas"][1])
                elif "momentum" in group:
                    group["momentum"] = momentum

        if self.verbose:
            print(f"Adjusting learning rate to {new_lrs}.")
