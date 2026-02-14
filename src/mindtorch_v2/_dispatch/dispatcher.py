import inspect

from .registry import registry
from .pipeline import current_pipeline
from .keys import DispatchKey, DispatchKeySet
from .._autograd.grad_mode import is_grad_enabled


def _accepts_device(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "device" in params:
        return True
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _prepare_kwargs(func, kwargs, device):
    if not kwargs:
        kwargs = {}
    if "device" in kwargs:
        if _accepts_device(func):
            return kwargs
        return {k: v for k, v in kwargs.items() if k != "device"}
    if _accepts_device(func):
        merged = dict(kwargs)
        merged["device"] = device
        return merged
    return kwargs


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


def _kernel_for_entry(entry, key_order):
    for key in key_order:
        if key in entry.fallthrough:
            continue
        kernel = entry.kernels.get(key)
        if kernel is not None:
            return kernel, key
    return None, None


def _key_order(keyset):
    order = [
        DispatchKey.Pipeline,
        DispatchKey.Autograd,
        DispatchKey.Meta,
        DispatchKey.NPU,
        DispatchKey.CPU,
    ]
    return [key for key in order if key in keyset]


def _extract_tensors(args, kwargs):
    tensors = []
    for value in list(args) + list(kwargs.values()):
        if hasattr(value, "device"):
            tensors.append(value)
    return tensors


def dispatch(name, dispatch_device, *args, **kwargs):
    tensors = _extract_tensors(args, kwargs)
    pipe = current_pipeline()
    keyset = DispatchKeySet.from_tensors(
        tensors,
        grad_enabled=is_grad_enabled(),
        pipeline_enabled=pipe is not None,
    )
    entry = registry.get(name)
    kernel, key = _kernel_for_entry(entry, _key_order(keyset))
    if kernel is None:
        raise RuntimeError(f"could not find kernel for op {name} with keys {sorted(k.name for k in keyset)}")
    impl_kwargs = _prepare_kwargs(kernel, kwargs, dispatch_device)
    return kernel(*args, **impl_kwargs)
