

import inspect

from .registry import registry
from .pipeline import current_pipeline


def _filter_kwargs(func, kwargs):
    if not kwargs or "device" not in kwargs:
        return kwargs
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if k != "device"}
    params = sig.parameters
    if "device" in params:
        return kwargs
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k != "device"}




def _resolve_device(dispatch_device, args, kwargs):
    if hasattr(dispatch_device, "type"):
        dev_type = dispatch_device.type
    else:
        dev_type = dispatch_device
    if dev_type == "meta":
        return dispatch_device
    for value in list(args) + list(kwargs.values()):
        if hasattr(value, "device") and getattr(value.device, "type", None) == "meta":
            return value.device
    return dispatch_device


class _PendingOp:
    def __init__(self, entry, args, kwargs, out):
        self.entry = entry
        self.args = args
        self.kwargs = kwargs
        self.out = out

    def execute(self):
        impl = self.entry["impl"]
        result = impl(*self.args, **self.kwargs)
        self.out._storage = result.storage()
        self.out.shape = result.shape
        self.out.stride = result.stride
        self.out.offset = result.offset
        self.out._pending = False


def _pending_tensor_from_spec(spec, device):
    from .._storage import PendingStorage
    from .._tensor import Tensor

    storage = PendingStorage(spec.shape, spec.dtype, device)
    return Tensor(storage, spec.shape, spec.stride, spec.offset)


def dispatch(name, dispatch_device, *args, **kwargs):
    device = _resolve_device(dispatch_device, args, kwargs)
    dev_type = device.type if hasattr(device, "type") else device
    entry = registry.get(name, dev_type)
    pipe = current_pipeline()
    meta = entry.get("meta")
    if pipe is None or meta is None:
        return entry["impl"](*args, **_filter_kwargs(entry["impl"], kwargs))
    if (device.type if hasattr(device, "type") else device) == "meta":
        return entry["impl"](*args, **_filter_kwargs(entry["impl"], kwargs))
    spec = meta(*args, **_filter_kwargs(meta, kwargs))
    out = _pending_tensor_from_spec(spec, device)
    out._pending = True
    pipe.record(_PendingOp(entry, args, kwargs, out))
    return out
