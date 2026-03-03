from dataclasses import asdict, dataclass
import math
import numpy as np

from ._dtype import float32
from ._device import _default_device


@dataclass
class PrintOptions:
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: bool | None = None


_PRINT_OPTIONS = PrintOptions()


_PROFILES = {
    "default": {"precision": 4, "threshold": 1000, "edgeitems": 3},
    "short": {"precision": 2, "threshold": 1000, "edgeitems": 2},
    "full": {"threshold": math.inf},
}


def get_printoptions():
    return asdict(_PRINT_OPTIONS)


def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    sci_mode=None,
    profile=None,
):
    if profile is not None:
        if profile not in _PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        for key, value in _PROFILES[profile].items():
            setattr(_PRINT_OPTIONS, key, value)
    if precision is not None:
        _PRINT_OPTIONS.precision = precision
    if threshold is not None:
        _PRINT_OPTIONS.threshold = threshold
    if edgeitems is not None:
        _PRINT_OPTIONS.edgeitems = edgeitems
    if linewidth is not None:
        _PRINT_OPTIONS.linewidth = linewidth
    if sci_mode is not None:
        _PRINT_OPTIONS.sci_mode = sci_mode


def format_tensor(tensor):
    if getattr(tensor, "_pending", False):
        from ._dispatch.pipeline import current_pipeline

        pipe = current_pipeline()
        if pipe is not None:
            pipe.flush()

    if tensor.device.type == "meta":
        data_repr = "..."
    else:
        if tensor.device.type == "cpu":
            view = tensor._numpy_view()
        else:
            view = tensor.to("cpu")._numpy_view()
        data_repr = _format_array(view, _PRINT_OPTIONS)

    suffixes = []
    if tensor.dtype != float32 or tensor.device.type == "meta":
        suffixes.append(f"dtype={repr(tensor.dtype)}")
    if tensor.device.type != _default_device.type:
        device_label = tensor.device.type
        if tensor.device.type == "npu":
            device_label = str(tensor.device)
        suffixes.append(f"device='{device_label}'")
    if tensor.requires_grad:
        suffixes.append("requires_grad=True")
    if tensor.grad_fn is not None:
        suffixes.append(f"grad_fn=<{type(tensor.grad_fn).__name__}>")

    if suffixes:
        return f"tensor({data_repr}, {', '.join(suffixes)})"
    return f"tensor({data_repr})"


def _format_array(arr, options):
    formatter = None
    floatmode = None
    if options.sci_mode is True:
        def _format_float(value):
            return np.format_float_scientific(
                value, precision=options.precision, unique=False
            )
        formatter = {"float_kind": _format_float}
    elif options.sci_mode is False:
        floatmode = "fixed"

    kwargs = {
        "precision": options.precision,
        "threshold": options.threshold,
        "edgeitems": options.edgeitems,
        "separator": ", ",
        "prefix": "tensor(",
        "formatter": formatter,
        "floatmode": floatmode,
    }
    try:
        return np.array2string(arr, max_line_width=options.linewidth, **kwargs)
    except TypeError:
        return np.array2string(arr, linewidth=options.linewidth, **kwargs)
