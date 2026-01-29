"""Learning rate scheduler module."""

from typing import Optional, List
import math


class LRScheduler:
    """Base class for learning rate schedulers.

    This is a minimal implementation to allow transformers to inherit from it.
    """

    def __init__(self, optimizer, last_epoch=-1, verbose="deprecated"):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._step_count = 0
        self.base_lrs = [group.get('lr', 0.01) for group in optimizer.param_groups] if hasattr(optimizer, 'param_groups') else [0.01]

    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        return {'last_epoch': self.last_epoch, 'base_lrs': self.base_lrs}

    def load_state_dict(self, state_dict):
        """Load the scheduler state."""
        self.last_epoch = state_dict.get('last_epoch', -1)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)

    def get_last_lr(self):
        """Return last computed learning rate by current scheduler."""
        return self._last_lr if hasattr(self, '_last_lr') else self.base_lrs

    def get_lr(self):
        """Compute learning rate. Override in subclasses."""
        return self.base_lrs

    def step(self, epoch=None):
        """Step the scheduler."""
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        values = self.get_lr()
        self._last_lr = values

        # Update optimizer learning rates
        if hasattr(self.optimizer, 'param_groups'):
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = values[i] if i < len(values) else values[-1]


class _LRScheduler(LRScheduler):
    """Alias for LRScheduler (deprecated name)."""
    pass


class StepLR(LRScheduler):
    """Decays the learning rate by gamma every step_size epochs."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose="deprecated"):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups] if hasattr(self.optimizer, 'param_groups') else self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups] if hasattr(self.optimizer, 'param_groups') else [lr * self.gamma for lr in self.base_lrs]


class MultiStepLR(LRScheduler):
    """Decays the learning rate by gamma once the number of epochs reaches one of the milestones."""

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose="deprecated"):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups] if hasattr(self.optimizer, 'param_groups') else self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups] if hasattr(self.optimizer, 'param_groups') else [lr * self.gamma for lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    """Decays the learning rate by gamma every epoch."""

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose="deprecated"):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups] if hasattr(self.optimizer, 'param_groups') else [lr * self.gamma for lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Set learning rate using a cosine annealing schedule."""

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated"):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    """Decays the learning rate linearly."""

    def __init__(self, optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=5, last_epoch=-1, verbose="deprecated"):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return [lr * self.start_factor for lr in self.base_lrs]
        if self.last_epoch >= self.total_iters:
            return [lr * self.end_factor for lr in self.base_lrs]

        factor = self.start_factor + (self.end_factor - self.start_factor) * self.last_epoch / self.total_iters
        return [lr * factor for lr in self.base_lrs]


class ConstantLR(LRScheduler):
    """Multiplies the learning rate by a constant factor."""

    def __init__(self, optimizer, factor=1.0/3, total_iters=5, last_epoch=-1, verbose="deprecated"):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return self.base_lrs
        return [lr * self.factor for lr in self.base_lrs]


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose="deprecated"):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr if isinstance(min_lr, list) else [min_lr]
        self.eps = eps

        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self._last_lr = [group['lr'] for group in optimizer.param_groups] if hasattr(optimizer, 'param_groups') else [0.01]

    def step(self, metrics, epoch=None):
        """Step the scheduler based on metrics."""
        current = float(metrics)

        if self.best is None:
            self.best = current
        elif self._is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _is_better(self, a, best):
        if self.mode == 'min':
            return a < best - self.threshold if self.threshold_mode == 'abs' else a < best * (1 - self.threshold)
        else:
            return a > best + self.threshold if self.threshold_mode == 'abs' else a > best * (1 + self.threshold)

    def _reduce_lr(self, epoch):
        if hasattr(self.optimizer, 'param_groups'):
            for i, group in enumerate(self.optimizer.param_groups):
                min_lr = self.min_lr[i] if i < len(self.min_lr) else self.min_lr[-1]
                new_lr = max(group['lr'] * self.factor, min_lr)
                group['lr'] = new_lr
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
        }

    def load_state_dict(self, state_dict):
        self.best = state_dict.get('best')
        self.num_bad_epochs = state_dict.get('num_bad_epochs', 0)
        self.cooldown_counter = state_dict.get('cooldown_counter', 0)

    def get_last_lr(self):
        return self._last_lr


class LambdaLR(LRScheduler):
    """Sets learning rate using a lambda function."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose="deprecated"):
        self.lr_lambdas = [lr_lambda] if callable(lr_lambda) else list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            base_lr * lmbda(self.last_epoch)
            for base_lr, lmbda in zip(self.base_lrs, self.lr_lambdas)
        ]


class CosineAnnealingWarmRestarts(LRScheduler):
    """Set learning rate using SGDR schedule."""

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose="deprecated"):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            self.T_cur = epoch
        self.last_epoch = self.T_cur
        self._last_lr = self.get_lr()


class OneCycleLR(LRScheduler):
    """Sets learning rate according to 1cycle policy."""

    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None,
                 steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos',
                 cycle_momentum=True, base_momentum=0.85, max_momentum=0.95,
                 div_factor=25.0, final_div_factor=1e4, three_phase=False,
                 last_epoch=-1, verbose="deprecated"):
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr]
        self.total_steps = total_steps or (epochs * steps_per_epoch)
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch, verbose)
        self.base_lrs = [lr / div_factor for lr in self.max_lr]

    def get_lr(self):
        pct = self.last_epoch / self.total_steps
        if pct <= self.pct_start:
            # Warm up
            scale = pct / self.pct_start
            return [base_lr + (max_lr - base_lr) * scale for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]
        else:
            # Cool down
            scale = (1 - (pct - self.pct_start) / (1 - self.pct_start))
            final_lrs = [lr / self.final_div_factor for lr in self.max_lr]
            return [final_lr + (max_lr - final_lr) * (1 + math.cos(math.pi * (1 - scale))) / 2
                    for final_lr, max_lr in zip(final_lrs, self.max_lr)]


class SequentialLR(LRScheduler):
    """Chains schedulers together."""

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose="deprecated"):
        self.schedulers = schedulers
        self.milestones = milestones
        self._current_scheduler_idx = 0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return self.schedulers[self._current_scheduler_idx].get_lr()

    def step(self):
        self.last_epoch += 1
        # Check if we need to move to next scheduler
        while (self._current_scheduler_idx < len(self.milestones) and
               self.last_epoch >= self.milestones[self._current_scheduler_idx]):
            self._current_scheduler_idx += 1
        if self._current_scheduler_idx < len(self.schedulers):
            self.schedulers[self._current_scheduler_idx].step()


# Compatibility alias
EPOCH_DEPRECATION_WARNING = ""


__all__ = [
    'LRScheduler', '_LRScheduler',
    'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR',
    'LinearLR', 'ConstantLR', 'ReduceLROnPlateau', 'LambdaLR',
    'CosineAnnealingWarmRestarts', 'OneCycleLR', 'SequentialLR',
]
