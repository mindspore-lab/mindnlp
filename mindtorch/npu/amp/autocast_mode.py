# mypy: allow-untyped-defs
import functools
from typing import Any
from typing_extensions import deprecated

import mindtorch


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


class autocast(mindtorch.amp.autocast_mode.autocast):
    r"""See :class:`mindtorch.autocast`.

    ``mindtorch.cuda.amp.autocast(args...)`` is deprecated. Please use ``mindtorch.amp.autocast("cuda", args...)`` instead.
    """

    @deprecated(
        "`mindtorch.cuda.amp.autocast(args...)` is deprecated. "
        "Please use `mindtorch.amp.autocast('cuda', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        enabled: bool = True,
        dtype: mindtorch.dtype = mindtorch.float16,
        cache_enabled: bool = True,
    ):
        if mindtorch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = "cuda"
            self.fast_dtype = dtype
            return
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        if mindtorch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if mindtorch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if mindtorch._jit_internal.is_scripting():
            return func
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`mindtorch.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `mindtorch.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    return mindtorch.amp.autocast_mode._cast(value, "cuda", dtype)


@deprecated(
    "`mindtorch.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `mindtorch.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``mindtorch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``mindtorch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(mindtorch.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`mindtorch.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `mindtorch.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``mindtorch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``mindtorch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(mindtorch.amp.custom_bwd, device_type="cuda")(bwd)
