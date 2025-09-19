# mypy: allow-untyped-defs
import collections
import functools
import warnings
from typing import Any, Optional

import mindtorch

from mindspore._c_expression.amp import pop_amp_strategy, push_amp_strategy, AmpLevel, create_amp_strategy
from mindspore.common.dtype import TensorType as _dtype, float32
from mindspore.train.amp import AMP_AUTO_BLACK_LIST, AMP_AUTO_WHITE_LIST, AMP_PRIM_ARG_TABLE


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

__all__ = [
    "autocast_decorator",
    "autocast",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
]


def is_autocast_available(device_type: str) -> bool:
    r"""
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`mindtorch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    return True


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
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        if not isinstance(device_type, str):
            raise ValueError(
                f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
            )
        self.device_type = device_type
        if dtype is None:
            dtype = float32
        self.dtype = dtype
        self.amp_level = AmpLevel.AmpAuto if enabled else AmpLevel.AmpO0

    def __enter__(self):
        self.prev_fastdtype = mindtorch.get_autocast_dtype(self.device_type)
        mindtorch.set_autocast_dtype(self.device_type, self.dtype)
        white_list = [(prim.__name__, AMP_PRIM_ARG_TABLE[prim]) for prim in AMP_AUTO_WHITE_LIST]
        black_list = [(prim.__name__, AMP_PRIM_ARG_TABLE[prim]) for prim in AMP_AUTO_BLACK_LIST]
        amp_strategy = create_amp_strategy(self.amp_level, self.dtype, white_list, black_list)
        push_amp_strategy(self.amp_level, self.dtype, white_list, black_list)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        pop_amp_strategy()
        mindtorch.set_autocast_dtype(self.device_type, self.prev_fastdtype)
        return False

    def __call__(self, func):
        return autocast_decorator(self, func)


def _cast(value, device_type: str, dtype):
    if isinstance(value, mindtorch.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and (value.dtype is not mindtorch.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {
            _cast(k, device_type, dtype): _cast(v, device_type, dtype)
            for k, v in value.items()
        }
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


def custom_fwd(
    fwd=None,
    *,
    device_type: str,
    cast_inputs = None,
):
    """
    Create a helper decorator for ``forward`` methods of custom autograd functions.

    Autograd functions are subclasses of :class:`mindtorch.autograd.Function`.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`mindtorch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`mindtorch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if fwd is None:
        return functools.partial(
            custom_fwd, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = mindtorch.get_autocast_dtype(device_type)
        if cast_inputs is None:
            args[0]._fwd_used_autocast = mindtorch.is_autocast_enabled(device_type)
            return fwd(*args, **kwargs)
        else:
            autocast_context = mindtorch.is_autocast_enabled(device_type)
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(device_type=device_type, enabled=False):
                    return fwd(
                        *_cast(args, device_type, cast_inputs),
                        **_cast(kwargs, device_type, cast_inputs),
                    )
            else:
                return fwd(*args, **kwargs)

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd=None, *, device_type: str):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`mindtorch.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`mindtorch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """

    if bwd is None:
        return custom_bwd

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(
            device_type=device_type,
            enabled=args[0]._fwd_used_autocast,
            dtype=args[0]._dtype,
        ):
            return bwd(*args, **kwargs)

    return decorate_bwd
