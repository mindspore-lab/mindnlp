import inspect

from .registry import registry
from .pipeline import current_pipeline
from .keys import DispatchKey, DispatchKeySet
from .functionalize import functionalize_op, is_functionalize_enabled, should_functionalize
import threading

from .._autograd.grad_mode import is_grad_enabled


_DISPATCH_STATE = threading.local()


def _state_stack():
    stack = getattr(_DISPATCH_STATE, "stack", None)
    if stack is None:
        stack = []
        _DISPATCH_STATE.stack = stack
    return stack


def current_dispatch_keyset():
    stack = _state_stack()
    if not stack:
        return None
    return stack[-1][0]


def current_dispatch_key():
    stack = _state_stack()
    if not stack:
        return None
    return stack[-1][1]


def _push_dispatch_context(keyset, key):
    _state_stack().append((keyset, key))


def _pop_dispatch_context():
    stack = _state_stack()
    if stack:
        stack.pop()


def _accepts_device(func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "device" in params:
        return True
    return False


def _prepare_kwargs(func, kwargs, device):
    if not kwargs:
        kwargs = {}
    filtered = {k: v for k, v in kwargs.items() if k != "device"}
    if "device" in kwargs and _accepts_device(func):
        return kwargs
    if _accepts_device(func):
        merged = dict(filtered)
        merged["device"] = device
        return merged
    return filtered


class _PendingOp:
    def __init__(self, impl, args, kwargs, out, keyset, key):
        self.impl = impl
        self.args = args
        self.kwargs = kwargs
        self.out = out
        self.keyset = keyset
        self.key = key

    def _copy_result(self, pending, result):
        prev_requires_grad = pending.requires_grad
        pending._storage = result.storage()
        pending.shape = result.shape
        pending.stride = result.stride
        pending.offset = result.offset
        pending.requires_grad = prev_requires_grad or result.requires_grad
        if result.grad_fn is not None:
            pending.grad_fn = result.grad_fn
        elif not pending.requires_grad:
            pending.grad_fn = None
        pending._base = result._base
        pending._view_meta = result._view_meta
        pending._version_counter = result._version_counter
        pending._pending = False

    def execute(self):
        _push_dispatch_context(self.keyset, self.key)
        try:
            result = self.impl(*self.args, **self.kwargs)
        finally:
            _pop_dispatch_context()
        if isinstance(self.out, (tuple, list)):
            for pending, item in zip(self.out, result):
                self._copy_result(pending, item)
        else:
            self._copy_result(self.out, result)

class _FunctionalizePendingOp:
    def __init__(self, target, thunk, keyset, key):
        self.target = target
        self.thunk = thunk
        self.keyset = keyset
        self.key = key

    def execute(self):
        _push_dispatch_context(self.keyset, self.key)
        try:
            result = self.thunk()
        finally:
            _pop_dispatch_context()
        self.target._storage = result.storage()
        self.target.shape = result.shape
        self.target.stride = result.stride
        self.target.offset = result.offset
        self.target._base = result._base
        self.target._view_meta = result._view_meta
        self.target._pending = False



def _pending_tensor_from_spec(spec, device):
    from .._storage import PendingStorage
    from .._tensor import Tensor

    storage = PendingStorage(spec.shape, spec.dtype, device)
    return Tensor(storage, spec.shape, spec.stride, spec.offset)


def _pending_from_meta(meta, device):
    if isinstance(meta, (tuple, list)):
        return tuple(_pending_tensor_from_spec(spec, device) for spec in meta)
    return _pending_tensor_from_spec(meta, device)


def _kernel_for_entry(entry, key_order):
    for key in key_order:
        if key in entry.fallthrough:
            continue
        global_fallthrough = getattr(registry, "_global_fallthrough", set())
        if key in global_fallthrough:
            continue
        kernel = entry.kernels.get(key)
        if kernel is not None:
            return kernel, key
    return None, None


def _key_order(keyset):
    order = [
        DispatchKey.Pipeline,
        DispatchKey.Functionalize,
        DispatchKey.Autograd,
        DispatchKey.Meta,
        DispatchKey.NPU,
        DispatchKey.CPU,
        DispatchKey.Python,
        DispatchKey.Autocast,
        DispatchKey.BackendSelect,
        DispatchKey.ADInplaceOrView,
        DispatchKey.AutogradOther,
        DispatchKey.AutogradCPU,
        DispatchKey.AutogradNPU,
        DispatchKey.AutogradXPU,
        DispatchKey.AutogradMeta,
    ]
    return [key for key in order if key in keyset]


def _extract_tensors(args, kwargs):
    tensors = []
    for value in list(args) + list(kwargs.values()):
        if hasattr(value, "device"):
            tensors.append(value)
    return tensors


def _infer_dispatch_device(dispatch_device, tensors, keyset):
    if dispatch_device is not None:
        return dispatch_device
    for tensor in tensors:
        if hasattr(tensor, "device"):
            return tensor.device
    if not tensors:
        from .._device import get_default_device

        return get_default_device()
    if DispatchKey.Meta in keyset:
        return "meta"
    if DispatchKey.NPU in keyset:
        return "npu"
    return "cpu"


def dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs):
    tensors = _extract_tensors(args, kwargs)
    pipe = current_pipeline()
    dispatch_device = _infer_dispatch_device(dispatch_device, tensors, keyset)
    alias_name = name
    name = registry.resolve(name)
    entry = registry.get(name)

    if entry.schema_obj is not None:
        entry.schema_obj.bind(args, kwargs, op_name=alias_name, error_overrides=entry.error_overrides)

    if DispatchKey.Functionalize in keyset and should_functionalize(entry):
        if pipe is not None and DispatchKey.Pipeline in keyset:
            pending = functionalize_op(name, alias_name, entry, keyset, args, kwargs, redispatch, pipeline=pipe, dispatch_device=dispatch_device)
            return pending
        return functionalize_op(name, alias_name, entry, keyset, args, kwargs, redispatch, dispatch_device=dispatch_device)

    def _run_kernel():
        kernel, key = _kernel_for_entry(entry, _key_order(keyset))
        if kernel is None:
            raise RuntimeError(
                f"could not find kernel for op {name} with keys {sorted(k.name for k in keyset)}"
            )
        impl_kwargs = _prepare_kwargs(kernel, kwargs, dispatch_device)
        _push_dispatch_context(keyset, key)
        try:
            return kernel(*args, **impl_kwargs)
        finally:
            _pop_dispatch_context()
    if pipe is not None and DispatchKey.Pipeline in keyset:
        meta = entry.kernels.get(DispatchKey.Meta)
        if meta is None:
            raise RuntimeError(f"pipeline requires meta kernel for op {name}")
        meta_kwargs = _prepare_kwargs(meta, kwargs, dispatch_device)
        spec = meta(*args, **meta_kwargs)
        out = _pending_from_meta(spec, dispatch_device)
        if isinstance(out, (tuple, list)):
            for item in out:
                item._pending = True
        else:
            out._pending = True
        backend_keys = [key for key in _key_order(keyset) if key != DispatchKey.Pipeline]
        impl, impl_key = _kernel_for_entry(entry, backend_keys)
        if impl is None:
            raise RuntimeError(f"pipeline requires backend kernel for op {name}")
        impl_kwargs = _prepare_kwargs(impl, kwargs, dispatch_device)
        pipe.record(_PendingOp(impl, args, impl_kwargs, out, keyset.without(DispatchKey.Pipeline), impl_key), pending=out)
        return out
    if pipe is not None and DispatchKey.Pipeline in keyset:
        pipe.flush()
    return _run_kernel()


def dispatch(name, dispatch_device, *args, **kwargs):
    tensors = _extract_tensors(args, kwargs)
    pipe = current_pipeline()
    keyset = DispatchKeySet.from_tensors(
        tensors,
        grad_enabled=is_grad_enabled(),
        pipeline_enabled=pipe is not None,
        functionalize_enabled=is_functionalize_enabled(),
        device=dispatch_device,
    )
    return dispatch_with_keyset(name, keyset, dispatch_device, *args, **kwargs)


def redispatch(name, keyset, *args, **kwargs):
    return dispatch_with_keyset(name, keyset, None, *args, **kwargs)
