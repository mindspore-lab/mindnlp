from enum import Enum, auto


class DispatchKey(Enum):
    CPU = auto()
    NPU = auto()
    Meta = auto()
    Autograd = auto()
    Pipeline = auto()


class DispatchKeySet(set):
    def without(self, keys):
        if isinstance(keys, (set, list, tuple)):
            return DispatchKeySet({key for key in self if key not in keys})
        return DispatchKeySet({key for key in self if key != keys})

    @classmethod
    def from_tensors(cls, tensors, *, grad_enabled=False, pipeline_enabled=False, device=None):
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
        if pipeline_enabled and not has_meta:
            keys.add(DispatchKey.Pipeline)
        return keys
