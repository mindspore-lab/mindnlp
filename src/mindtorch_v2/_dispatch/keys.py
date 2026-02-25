from enum import IntEnum
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


class DispatchKeySet:
    def __init__(self, mask=0):
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
            mask |= int(DispatchKey.Autograd)
        if functionalize_enabled:
            mask |= int(DispatchKey.Functionalize)
        if pipeline_enabled and not has_meta:
            mask |= int(DispatchKey.Pipeline)
        return cls(mask)

