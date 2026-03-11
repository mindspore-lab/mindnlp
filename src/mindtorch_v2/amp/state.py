import threading

from .._dtype import bfloat16, float16
from .._dtype import DType


_TLS = threading.local()
_AMP_CACHE = {}
_AUTOCACHE_ENABLED = True
_DEFAULT_DEVICE = "cpu"
_VALID_DEVICE_TYPES = {"cpu", "npu", "cuda", "meta", "mps"}


def _require_valid_device_type(device_type):
    if device_type not in _VALID_DEVICE_TYPES:
        raise RuntimeError(f"invalid device_type '{device_type}'")


def _state_map():
    state = getattr(_TLS, "amp_state", None)
    if state is None:
        state = {
            "cpu": {
                "enabled": False,
                "dtype": bfloat16,
                "cache_enabled": True,
                "nesting": 0,
            },
            "npu": {
                "enabled": False,
                "dtype": float16,
                "cache_enabled": True,
                "nesting": 0,
            },
            "cuda": {
                "enabled": False,
                "dtype": float16,
                "cache_enabled": True,
                "nesting": 0,
            },
            "meta": {
                "enabled": False,
                "dtype": float16,
                "cache_enabled": True,
                "nesting": 0,
            },
            "mps": {
                "enabled": False,
                "dtype": float16,
                "cache_enabled": True,
                "nesting": 0,
            },
        }
        _TLS.amp_state = state
    return state


def _state_for(device_type):
    _require_valid_device_type(device_type)
    state = _state_map()
    return state[device_type]


def _normalize_device_type(device_type):
    if device_type is None:
        return _DEFAULT_DEVICE
    if hasattr(device_type, "type"):
        device_type = device_type.type
    return str(device_type).lower()


def is_autocast_available(device_type):
    dev = _normalize_device_type(device_type)
    return dev in {"cpu", "npu", "cuda"}


def is_autocast_enabled(device_type=None):
    dev = _normalize_device_type(device_type)
    return bool(_state_for(dev)["enabled"])


def set_autocast_enabled(device_type, enabled=None):
    if enabled is None:
        if not isinstance(device_type, bool):
            raise TypeError(
                f"set_autocast_enabled(): argument 'enabled' (position 1) must be bool, not {type(device_type).__name__}"
            )
        enabled = device_type
        device_type = _DEFAULT_DEVICE
    elif not isinstance(enabled, bool):
        raise TypeError(
            f"set_autocast_enabled(): argument 'enabled' (position 2) must be bool, not {type(enabled).__name__}"
        )

    if not isinstance(device_type, str):
        raise TypeError(
            f"set_autocast_enabled(): argument 'device_type' (position 1) must be str, not {type(device_type).__name__}"
        )

    dev = _normalize_device_type(device_type)
    _state_for(dev)["enabled"] = bool(enabled)


def get_autocast_dtype(device_type):
    dev = _normalize_device_type(device_type)
    return _state_for(dev)["dtype"]


def set_autocast_dtype(device_type, dtype):
    if not isinstance(device_type, str):
        raise TypeError(
            f"set_autocast_dtype(): argument 'device_type' (position 1) must be str, not {type(device_type).__name__}"
        )
    if not isinstance(dtype, DType):
        raise TypeError(
            f"set_autocast_dtype(): argument 'dtype' (position 2) must be torch.dtype, not {type(dtype).__name__}"
        )

    dev = _normalize_device_type(device_type)
    _state_for(dev)["dtype"] = dtype


def is_autocast_cache_enabled():
    return bool(_AUTOCACHE_ENABLED)


def set_autocast_cache_enabled(enabled):
    if not isinstance(enabled, bool):
        raise TypeError(f"enabled must be a bool (got {type(enabled).__name__})")
    global _AUTOCACHE_ENABLED
    _AUTOCACHE_ENABLED = enabled


def clear_autocast_cache(device_type=None):
    if device_type is None:
        _AMP_CACHE.clear()
        return
    dev = _normalize_device_type(device_type)
    _AMP_CACHE.pop(dev, None)


def autocast_increment_nesting(device_type="cpu"):
    dev = _normalize_device_type(device_type)
    _state_for(dev)["nesting"] += 1
    return _state_for(dev)["nesting"]


def autocast_decrement_nesting(device_type="cpu"):
    dev = _normalize_device_type(device_type)
    _state_for(dev)["nesting"] -= 1
    if _state_for(dev)["nesting"] < 0:
        _state_for(dev)["nesting"] = 0
    return _state_for(dev)["nesting"]
