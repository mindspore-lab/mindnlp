from enum import Enum, auto


class DispatchKey(Enum):
    BackendSelect = auto()
    ADInplaceOrView = auto()
    AutogradOther = auto()
    AutogradCPU = auto()
    AutogradNPU = auto()
    AutogradXPU = auto()
    AutogradMeta = auto()
    Autograd = auto()
    Functionalize = auto()
    Meta = auto()
    NPU = auto()
    CPU = auto()
    Python = auto()
    Autocast = auto()
    Pipeline = auto()


class DispatchKeySet(set):
    def without(self, keys):
        if isinstance(keys, (set, list, tuple)):
            return DispatchKeySet({key for key in self if key not in keys})
        return DispatchKeySet({key for key in self if key != keys})

    @classmethod
    def from_tensors(cls, tensors, *, grad_enabled=False, pipeline_enabled=False, functionalize_enabled=False, device=None):
        keys = cls()
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
        if has_meta:
            keys.add(DispatchKey.Meta)
        elif has_npu:
            keys.add(DispatchKey.NPU)
        else:
            keys.add(DispatchKey.CPU)
        if grad_enabled and requires_grad:
            keys.add(DispatchKey.Autograd)
        if functionalize_enabled:
            keys.add(DispatchKey.Functionalize)
        if pipeline_enabled and not has_meta:
            keys.add(DispatchKey.Pipeline)
        # Placeholders for torch-aligned key ordering; fallthrough by default.
        keys.update(
            {
                DispatchKey.BackendSelect,
                DispatchKey.ADInplaceOrView,
                DispatchKey.AutogradOther,
                DispatchKey.AutogradCPU,
                DispatchKey.AutogradNPU,
                DispatchKey.AutogradXPU,
                DispatchKey.AutogradMeta,
                DispatchKey.Python,
                DispatchKey.Autocast,
            }
        )
        return keys
