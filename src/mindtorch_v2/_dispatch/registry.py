from .keys import DispatchKey
from .schema import OpSchema


class OperatorEntry:
    def __init__(self, name):
        self.name = name
        self.schema = None
        self.schema_obj = None
        self.kernels = {}
        self.fallthrough = set()


class OpRegistry:
    def __init__(self):
        self._ops = {}

    def _canonical_name(self, name):
        if "::" in name:
            return name
        return f"aten::{name}"

    def _entry(self, name):
        name = self._canonical_name(name)
        entry = self._ops.get(name)
        if entry is None:
            entry = OperatorEntry(name)
            self._ops[name] = entry
        return entry

    def register_schema(self, name, schema):
        entry = self._entry(name)
        entry.schema = schema
        entry.schema_obj = OpSchema(schema)
        return entry

    def register_kernel(self, name, key, fn):
        entry = self._entry(name)
        entry.kernels[key] = fn
        return entry

    def register_fallthrough(self, name, key):
        entry = self._entry(name)
        entry.fallthrough.add(key)
        return entry

    def register(self, name, device, fn, meta=None):
        key = resolve_dispatch_key(device)
        entry = self.register_kernel(name, key, fn)
        if meta is not None and DispatchKey.Meta not in entry.kernels:
            entry.kernels[DispatchKey.Meta] = meta
        return entry

    def get(self, name):
        return self._ops[self._canonical_name(name)]


registry = OpRegistry()


def resolve_dispatch_key(device):
    if isinstance(device, DispatchKey):
        return device
    if hasattr(device, "type"):
        if device.type == "meta":
            return DispatchKey.Meta
        if device.type == "npu":
            return DispatchKey.NPU
        return DispatchKey.CPU
    if device == "meta":
        return DispatchKey.Meta
    if device == "npu":
        return DispatchKey.NPU
    if device == "cpu":
        return DispatchKey.CPU
    return DispatchKey.CPU
