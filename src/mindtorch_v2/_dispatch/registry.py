import copy
from .keys import DispatchKey
from .schema import OpSchema


class OperatorEntry:
    def __init__(self, name):
        self.name = name
        self.schema = None
        self.schema_obj = None
        self.error_overrides = None
        self.kernels = {}
        self.fallthrough = set()
        self.functionalize = None


class OpRegistry:
    def __init__(self):
        self._ops = {}
        self._aliases = {}

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
        if entry.functionalize is None and name.endswith("_"):
            if any(param.mutates for param in entry.schema_obj.params):
                entry.functionalize = name[:-1]
        return entry

    def register_error_overrides(self, name, overrides):
        entry = self._entry(name)
        entry.error_overrides = overrides
        return entry

    def register_kernel(self, name, key, fn):
        entry = self._entry(name)
        entry.kernels[key] = fn
        return entry

    def register_fallthrough(self, name, key):
        entry = self._entry(name)
        entry.fallthrough.add(key)
        return entry

    def register_alias(self, alias, target):
        self._aliases[alias] = target
        return alias

    def register_functionalize(self, name, functional_name):
        entry = self._entry(name)
        entry.functionalize = functional_name
        return entry

    def get_functionalize(self, name):
        entry = self._ops.get(self._canonical_name(name))
        if entry is None:
            return None
        return entry.functionalize

    def has(self, name):
        return self._canonical_name(name) in self._ops

    def resolve(self, name):
        return self._aliases.get(name, name)

    def register(self, name, device, fn, meta=None):
        key = resolve_dispatch_key(device)
        entry = self.register_kernel(name, key, fn)
        if meta is not None and DispatchKey.Meta not in entry.kernels:
            entry.kernels[DispatchKey.Meta] = meta
        return entry

    def get(self, name):
        return self._ops[self._canonical_name(name)]

    def snapshot(self):
        return {
            "ops": copy.deepcopy(self._ops),
            "aliases": copy.deepcopy(self._aliases),
            "global_fallthrough": copy.deepcopy(getattr(self, "_global_fallthrough", set())),
        }

    def restore(self, state):
        self._ops = copy.deepcopy(state["ops"])
        self._aliases = copy.deepcopy(state["aliases"])
        self._global_fallthrough = copy.deepcopy(state.get("global_fallthrough", set()))


registry = OpRegistry()


def _register_global_fallthroughs():
    placeholders = {
        DispatchKey.BackendSelect,
        DispatchKey.ADInplaceOrView,
        DispatchKey.AutogradOther,
        DispatchKey.AutogradCPU,
        DispatchKey.AutogradNPU,
        DispatchKey.AutogradXPU,
        DispatchKey.AutogradMeta,
        DispatchKey.Python,
        DispatchKey.Autocast,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.PrivateUse1,
        DispatchKey.PrivateUse2,
        DispatchKey.PrivateUse3,
    }
    registry._global_fallthrough = placeholders


_register_global_fallthroughs()


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
