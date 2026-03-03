from .._functional import isnan, isinf, any as _any


class GradScaler:
    """Gradient scaler for mixed precision training.

    Scales loss to prevent gradient underflow in fp16,
    then unscales gradients before optimizer step.
    """

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    ):
        self._enabled = enabled
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False

    def scale(self, loss):
        """Scale loss tensor by the current scale factor."""
        if not self._enabled:
            return loss
        from .._creation import tensor
        scale_tensor = tensor(self._scale)
        return loss * scale_tensor

    def unscale_(self, optimizer):
        """Unscale gradients by dividing by the scale factor. Detects inf/nan."""
        if not self._enabled:
            return

        from .._creation import tensor
        inv_scale = tensor(1.0 / self._scale)
        self._found_inf = False

        # Support both param_groups (PyTorch-style) and params (mindtorch-style)
        if hasattr(optimizer, 'param_groups'):
            params_to_unscale = []
            for param_group in optimizer.param_groups:
                params_to_unscale.extend(param_group['params'])
        else:
            params_to_unscale = optimizer.params

        for p in params_to_unscale:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad = p.grad * inv_scale
                # Check for inf/nan
                if _any(isnan(p.grad)) or _any(isinf(p.grad)):
                    self._found_inf = True

    def step(self, optimizer, *args, **kwargs):
        """Step the optimizer, skipping if inf/nan gradients were found.

        If unscale_ has not been called, it will be called first.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        # Auto-unscale if not already done
        if not hasattr(self, '_found_inf') or self._found_inf is None:
            self.unscale_(optimizer)

        if not self._found_inf:
            return optimizer.step(*args, **kwargs)
        # else: skip step due to inf/nan grads

    def update(self, new_scale=None):
        """Update the scale factor based on whether inf/nan was found."""
        if not self._enabled:
            return

        if new_scale is not None:
            self._scale = float(new_scale)
            return

        if self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

        # Reset for next iteration
        self._found_inf = False

    def get_scale(self):
        """Return the current scale factor."""
        return self._scale

    def is_enabled(self):
        """Return whether the scaler is enabled."""
        return self._enabled

    def state_dict(self):
        """Return scaler state as a dict."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
        }

    def load_state_dict(self, state_dict):
        """Load scaler state from a dict."""
        self._scale = state_dict["scale"]
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._growth_tracker = state_dict["growth_tracker"]
