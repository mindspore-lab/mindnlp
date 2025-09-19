from typing_extensions import deprecated

import mindtorch

# We need to keep this unused import for BC reasons
from ...amp.grad_scaler import OptState  # noqa: F401


__all__ = ["GradScaler"]


class GradScaler(mindtorch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.npu.amp.GradScaler(args...)`` is deprecated. Please use ``torch.amp.GradScaler("npu", args...)`` instead.
    """

    @deprecated(
        "`torch.npu.amp.GradScaler(args...)` is deprecated. "
        "Please use `torch.amp.GradScaler('npu', args...)` instead.",
        category=FutureWarning,
    )
    def __init__(
        self,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        super().__init__(
            "npu",
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )