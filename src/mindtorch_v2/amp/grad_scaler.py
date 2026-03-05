from collections import defaultdict
from enum import Enum

class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_per_optimizer_state():
    return {"stage": OptState.READY, "found_inf_per_device": {}}


class GradScaler:
    def __init__(
        self,
        device="cuda",
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    ):
        if isinstance(device, str):
            self._device = device
        else:
            # Back-compat if device passed as torch.device-like
            self._device = getattr(device, "type", str(device))
        self._enabled = enabled
        self._init_scale = float(init_scale)
        self._scale = None
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._init_growth_tracker = 0
        self._growth_tracker = None
        self._found_inf = False
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def _check_scale_growth_tracker(self, funcname):
        fix = (
            "This may indicate your script did not use scaler.scale(loss or outputs) "
            "earlier in the iteration."
        )
        assert self._scale is not None, f"Attempted {funcname} but _scale is None.  " + fix
        assert (
            self._growth_tracker is not None
        ), f"Attempted {funcname} but _growth_tracker is None.  " + fix
        return self._scale, self._growth_tracker

    def _lazy_init_scale_growth_tracker(self):
        if self._scale is None:
            self._scale = self._init_scale
        if self._growth_tracker is None:
            self._growth_tracker = self._init_growth_tracker

    def scale(self, outputs):
        if not self._enabled:
            return outputs
        from .._creation import tensor
        self._lazy_init_scale_growth_tracker()
        return outputs * tensor(self._scale)

    def _params_for_optimizer(self, optimizer):
        if hasattr(optimizer, "param_groups"):
            params = []
            for group in optimizer.param_groups:
                params.extend(group["params"])
            return params
        return optimizer.params

    def unscale_(self, optimizer):
        if not self._enabled:
            return

        self._check_scale_growth_tracker("unscale_")

        optimizer_state = self._per_optimizer_states[id(optimizer)]
        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        inv_scale = 1.0 / self._scale
        self._found_inf = False

        for p in self._params_for_optimizer(optimizer):
            if hasattr(p, "grad") and p.grad is not None:
                p.grad = p.grad * inv_scale
                # Basic numeric checks using Tensor API to avoid import cycles.
                if p.grad.isnan().any() or p.grad.isinf().any():
                    self._found_inf = True

        optimizer_state["found_inf_per_device"] = {self._device: 1.0 if self._found_inf else 0.0}
        optimizer_state["stage"] = OptState.UNSCALED

    def step(self, optimizer, *args, **kwargs):
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]
        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        if optimizer_state["stage"] is OptState.READY:
            self.unscale_(optimizer)

        retval = None
        if not self._found_inf:
            retval = optimizer.step(*args, **kwargs)
        optimizer_state["stage"] = OptState.STEPPED
        return retval

    def update(self, new_scale=None):
        if not self._enabled:
            return

        self._check_scale_growth_tracker("update")

        if new_scale is not None:
            self._scale = float(new_scale)
        elif self._found_inf:
            self._scale *= self._backoff_factor
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

        self._found_inf = False
        self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)

    def get_scale(self):
        if not self._enabled:
            return 1.0
        if self._scale is None:
            return self._init_scale
        return self._scale

    def is_enabled(self):
        return self._enabled

    def state_dict(self):
        if not self._enabled:
            return {}
        return {
            "scale": self.get_scale(),
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "_growth_tracker": self._growth_tracker if self._growth_tracker is not None else self._init_growth_tracker,
        }

    def load_state_dict(self, state_dict):
        if not self._enabled:
            return
        self._scale = state_dict["scale"]
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._growth_tracker = state_dict["_growth_tracker"]
