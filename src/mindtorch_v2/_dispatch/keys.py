from enum import Enum, auto


class DispatchKey(Enum):
    CPU = auto()
    NPU = auto()
    Meta = auto()
    Autograd = auto()
    Pipeline = auto()


class DispatchKeySet(set):
    @classmethod
    def from_tensors(cls, tensors, *, grad_enabled=False, pipeline_enabled=False):
        keys = cls()
        has_meta = False
        has_npu = False
        has_cpu = False
        requires_grad = False
        for tensor in tensors:
            if not hasattr(tensor, "device"):
                continue
            dev_type = tensor.device.type
            if dev_type == "meta":
                has_meta = True
            elif dev_type == "npu":
                has_npu = True
            else:
                has_cpu = True
            if getattr(tensor, "requires_grad", False):
                requires_grad = True
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
