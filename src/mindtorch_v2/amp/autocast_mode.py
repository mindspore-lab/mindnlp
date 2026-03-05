import collections
import functools
import warnings
from typing import Optional

from .state import (
    autocast_decrement_nesting,
    autocast_increment_nesting,
    clear_autocast_cache,
    get_autocast_dtype,
    is_autocast_available,
    is_autocast_cache_enabled,
    is_autocast_enabled,
    set_autocast_cache_enabled,
    set_autocast_dtype,
    set_autocast_enabled,
)

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
    HAS_NUMPY = False


__all__ = [
    "autocast_decorator",
    "autocast",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
    "_enter_autocast",
    "_exit_autocast",
]


def _cuda_available():
    try:
        import importlib
        cuda_api = importlib.import_module("mindtorch_v2.cuda")
    except Exception:
        return False
    return bool(getattr(cuda_api, "is_available", lambda: False)())


def _npu_available():
    try:
        from .. import npu as npu_api
    except Exception:
        return False
    return bool(getattr(npu_api, "is_available", lambda: False)())


_DEVICE_TYPE_AVAILABILITY = {
    "cpu": lambda: True,
    "cuda": _cuda_available,
    "npu": _npu_available,
}


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    return decorate_autocast


class autocast:
    def __init__(
        self,
        device_type: str,
        dtype: Optional[object] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        if not isinstance(device_type, str):
            raise ValueError(
                f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
            )
        self.device = device_type
        if not is_autocast_available(self.device):
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )
        if not _DEVICE_TYPE_AVAILABILITY.get(self.device, lambda: False)():
            raise RuntimeError(
                f"autocast is not available for device_type '{self.device}'"
            )

        self.fast_dtype = get_autocast_dtype(self.device) if dtype is None else dtype
        self._enabled = bool(enabled)
        self._cache_enabled = (
            is_autocast_cache_enabled(self.device)
            if cache_enabled is None
            else bool(cache_enabled)
        )
        if self.fast_dtype not in (get_autocast_dtype(self.device),):
            supported = {get_autocast_dtype(self.device)}
            if self.fast_dtype not in supported:
                warnings.warn(
                    f"autocast is not supported for dtype {self.fast_dtype} on device_type {self.device}",
                    UserWarning,
                )
                self._enabled = False

    def __enter__(self):
        self.prev = is_autocast_enabled(self.device)
        self.prev_fastdtype = get_autocast_dtype(self.device)
        self.prev_cache_enabled = is_autocast_cache_enabled(self.device)

        set_autocast_enabled(self.device, self._enabled)
        set_autocast_dtype(self.device, self.fast_dtype)
        set_autocast_cache_enabled(self.device, self._cache_enabled)
        autocast_increment_nesting(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if autocast_decrement_nesting(self.device) == 0:
            clear_autocast_cache()
        set_autocast_enabled(self.device, self.prev)
        set_autocast_dtype(self.device, self.prev_fastdtype)
        set_autocast_cache_enabled(self.device, self.prev_cache_enabled)
        return False

    def __call__(self, func):
        return autocast_decorator(self, func)


# Casts Tensors and containers of Tensors. Special-cases passthroughs for strings and np.ndarrays.
def _cast(value, device_type: str, dtype):
    if hasattr(value, "dtype") and hasattr(value, "device"):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and str(value.dtype) != "torch.float64"
        )
        if is_eligible:
            from .._functional import to as _to
            return _to(value, device=value.device, dtype=dtype)
        return value
    if isinstance(value, (str, bytes)):
        return value
    if HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    if isinstance(value, collections.abc.Mapping):
        return {
            _cast(k, device_type, dtype): _cast(v, device_type, dtype)
            for k, v in value.items()
        }
    if isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        return iterable
    return value


def custom_fwd(fwd=None, *, device_type: str, cast_inputs=None):
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )

    if fwd is None:
        return functools.partial(
            custom_fwd, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = get_autocast_dtype(device_type)
        if cast_inputs is None:
            args[0]._fwd_used_autocast = is_autocast_enabled(device_type)
            return fwd(*args, **kwargs)

        autocast_context = is_autocast_enabled(device_type)
        args[0]._fwd_used_autocast = False
        if autocast_context:
            with autocast(device_type=device_type, enabled=False):
                return fwd(
                    *_cast(args, device_type, cast_inputs),
                    **_cast(kwargs, device_type, cast_inputs),
                )
        return fwd(*args, **kwargs)

    return decorate_fwd


def custom_bwd(bwd=None, *, device_type: str):
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )

    if bwd is None:
        return functools.partial(custom_bwd, device_type=device_type)

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(
            device_type=device_type,
            enabled=args[0]._fwd_used_autocast,
            dtype=args[0]._dtype,
        ):
            return bwd(*args, **kwargs)

    return decorate_bwd


def _enter_autocast(*vals):
    mode = autocast(*vals)
    mode.__enter__()
    return mode


def _exit_autocast(mode):
    mode.__exit__(None, None, None)
