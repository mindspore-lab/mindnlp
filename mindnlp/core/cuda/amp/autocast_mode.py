# mypy: allow-untyped-defs
import functools
from typing import Any
from typing_extensions import deprecated

from mindnlp import core


__all__ = ["autocast", "custom_fwd", "custom_bwd"]


class autocast(core.amp.autocast_mode.autocast):
    r"""See :class:`core.autocast`.

    ``core.cuda.amp.autocast(args...)`` is deprecated. Please use ``core.amp.autocast("cuda", args...)`` instead.
    """

    @deprecated(
        "`core.cuda.amp.autocast(args...)` is deprecated. "
        "Please use `core.amp.autocast('cuda', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        enabled: bool = True,
        dtype: core.dtype = core.float16,
        cache_enabled: bool = True,
    ):
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        return super().__enter__()

    # TODO: discuss a unified TorchScript-friendly API for autocast
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        return super().__call__(func)


# Preserved only for BC reasons
@deprecated(
    "`core.cuda.amp.autocast_mode._cast(value, dtype)` is deprecated. "
    "Please use `core.amp.autocast_mode._cast(value, 'cuda', dtype)` instead.",
    category=FutureWarning,
)
def _cast(value, dtype):
    return core.amp.autocast_mode._cast(value, "cuda", dtype)


@deprecated(
    "`core.cuda.amp.custom_fwd(args...)` is deprecated. "
    "Please use `core.amp.custom_fwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``core.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``core.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(core.amp.custom_fwd, device_type="cuda")(
        fwd=fwd, cast_inputs=cast_inputs
    )


@deprecated(
    "`core.cuda.amp.custom_bwd(args...)` is deprecated. "
    "Please use `core.amp.custom_bwd(args..., device_type='cuda')` instead.",
    category=FutureWarning,
)
def custom_bwd(bwd):
    """
    ``core.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``core.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
    return functools.partial(core.amp.custom_bwd, device_type="cuda")(bwd)
