from enum import IntEnum
import threading
class DispatchKey(IntEnum):
    BackendSelect = 1 << 0
    Pipeline = 1 << 1
    Python = 1 << 2
    Functionalize = 1 << 3
    ADInplaceOrView = 1 << 4
    AutogradOther = 1 << 5
    AutogradCPU = 1 << 6
    AutogradNPU = 1 << 7
    AutogradXPU = 1 << 8
    AutogradMeta = 1 << 9
    Autograd = 1 << 10
    Meta = 1 << 11
    NPU = 1 << 12
    CPU = 1 << 13
    PythonDispatcher = 1 << 14
    CompositeImplicitAutograd = 1 << 15
    CompositeExplicitAutograd = 1 << 16
    Autocast = 1 << 17
    PrivateUse1 = 1 << 18
    PrivateUse2 = 1 << 19
    PrivateUse3 = 1 << 20


DISPATCH_KEY_PRIORITY = [
    DispatchKey.BackendSelect,
    DispatchKey.Pipeline,
    DispatchKey.Python,
    DispatchKey.Functionalize,
    DispatchKey.ADInplaceOrView,
    DispatchKey.AutogradOther,
    DispatchKey.AutogradCPU,
    DispatchKey.AutogradNPU,
    DispatchKey.AutogradXPU,
    DispatchKey.AutogradMeta,
    DispatchKey.Autograd,
    DispatchKey.Meta,
    DispatchKey.NPU,
    DispatchKey.CPU,
    DispatchKey.PythonDispatcher,
    DispatchKey.CompositeImplicitAutograd,
    DispatchKey.CompositeExplicitAutograd,
    DispatchKey.Autocast,
    DispatchKey.PrivateUse1,
    DispatchKey.PrivateUse2,
    DispatchKey.PrivateUse3,
]


_TLS = threading.local()


def _tls_state():
    state = getattr(_TLS, "dispatch_tls", None)
    if state is None:
        state = {"include": [], "exclude": []}
        _TLS.dispatch_tls = state
    return state


def _mask_from_keys(keys):
    if isinstance(keys, DispatchKey):
        return int(keys)
    if isinstance(keys, int):
        return int(keys)
    mask = 0
    for key in keys:
        mask |= int(key)
    return mask


def _push_mask(stack, keys):
    mask = _mask_from_keys(keys)
    stack.append(mask)
    return mask


def _pop_mask(stack):
    if stack:
        stack.pop()


class include_keys:
    def __init__(self, keys):
        self._keys = keys
        self._stack = None

    def __enter__(self):
        state = _tls_state()
        self._stack = state["include"]
        _push_mask(self._stack, self._keys)
        return self

    def __exit__(self, exc_type, exc, tb):
        _pop_mask(self._stack)


class exclude_keys:
    def __init__(self, keys):
        self._keys = keys
        self._stack = None

    def __enter__(self):
        state = _tls_state()
        self._stack = state["exclude"]
        _push_mask(self._stack, self._keys)
        return self

    def __exit__(self, exc_type, exc, tb):
        _pop_mask(self._stack)


def _effective_mask(base_mask):
    state = _tls_state()
    include_mask = 0
    for mask in state["include"]:
        include_mask |= mask
    exclude_mask = 0
    for mask in state["exclude"]:
        exclude_mask |= mask
    return (int(base_mask) | include_mask) & ~exclude_mask


def apply_tls_masks(keyset):
    if isinstance(keyset, DispatchKeySet):
        return DispatchKeySet(_effective_mask(keyset.mask))
    return DispatchKeySet(_effective_mask(keyset))


class DispatchKeySet:
    def __init__(self, mask=0):
        if isinstance(mask, (set, list, tuple)):
            # Convert set/list/tuple of DispatchKey to bitmask
            self.mask = 0
            for key in mask:
                self.mask |= int(key)
        else:
            self.mask = int(mask)

    def __contains__(self, key):
        return bool(self.mask & int(key))

    def has(self, key):
        return bool(self.mask & int(key))

    def add(self, key):
        self.mask |= int(key)
        return self

    def remove(self, key):
        self.mask &= ~int(key)
        return self

    def without(self, keys):
        mask = self.mask
        if isinstance(keys, (set, list, tuple)):
            for key in keys:
                mask &= ~int(key)
        else:
            mask &= ~int(keys)
        return DispatchKeySet(mask)

    def iter_keys(self):
        for key in DISPATCH_KEY_PRIORITY:
            if self.mask & int(key):
                yield key

    @classmethod
    def from_mask(cls, mask):
        return cls(int(mask))

    @classmethod
    def from_tensors(cls, tensors, *, grad_enabled=False, pipeline_enabled=False, functionalize_enabled=False, device=None):
        has_meta = False
        has_npu = False
        has_cpu = False
        requires_grad = False
        saw_device = False
        for tensor in tensors:
            if not hasattr(tensor, "device"):
                continue
            saw_device = True
            dev = tensor.device
            dev_type = dev.type if hasattr(dev, "type") else dev
            if dev_type == "meta":
                has_meta = True
            elif dev_type == "npu":
                has_npu = True
            else:
                has_cpu = True
            if getattr(tensor, "requires_grad", False):
                requires_grad = True
        if (not saw_device) and device is not None:
            dev_type = device.type if hasattr(device, "type") else device
            if dev_type == "meta":
                has_meta = True
            elif dev_type == "npu":
                has_npu = True
            else:
                has_cpu = True
        mask = 0
        if has_meta:
            mask |= int(DispatchKey.Meta)
        elif has_npu:
            mask |= int(DispatchKey.NPU)
        else:
            mask |= int(DispatchKey.CPU)
        if grad_enabled and requires_grad:
            mask |= int(DispatchKey.ADInplaceOrView)
            mask |= int(DispatchKey.Autograd)
            if has_meta:
                mask |= int(DispatchKey.AutogradMeta)
            elif has_npu:
                mask |= int(DispatchKey.AutogradNPU)
            else:
                mask |= int(DispatchKey.AutogradCPU)
        if functionalize_enabled:
            mask |= int(DispatchKey.Functionalize)
        if pipeline_enabled and not has_meta:
            mask |= int(DispatchKey.Pipeline)
        return cls(mask)

