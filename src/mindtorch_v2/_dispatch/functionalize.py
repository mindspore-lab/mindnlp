import contextlib

from .registry import registry


_ENABLED = False


@contextlib.contextmanager
def functionalize_context():
    global _ENABLED
    prev = _ENABLED
    _ENABLED = True
    try:
        yield
    finally:
        _ENABLED = prev


def is_functionalize_enabled():
    return _ENABLED


def _has_mutation(schema_obj):
    if schema_obj is None:
        return False
    return any(param.mutates for param in schema_obj.params)


def should_functionalize(entry):
    if not _has_mutation(entry.schema_obj):
        return False
    from .keys import DispatchKey

    if DispatchKey.Functionalize in entry.fallthrough:
        return False
    return True


def _mutating_args(schema_obj, args, kwargs):
    if schema_obj is None:
        return []
    if kwargs is None:
        kwargs = {}
    params = schema_obj.params
    positional = [p for p in params if not p.kw_only]
    bound = {}
    for idx, value in enumerate(args):
        if idx < len(positional):
            bound[positional[idx].name] = value
    for key, value in kwargs.items():
        bound[key] = value
    mutated = []
    for param in params:
        if not param.mutates:
            continue
        if param.name in bound:
            mutated.append(bound[param.name])
    return mutated


def _derive_functional_name(op_name):
    if op_name.endswith("_"):
        return op_name[:-1]
    return registry.get_functionalize(op_name)


def _writeback(target, result):
    if target is result:
        return target
    if getattr(target, "device", None) is not None and target.device.type == "meta":
        return target
    target.storage().copy_(result.storage())
    target.shape = result.shape
    target.stride = result.stride
    target.offset = result.offset
    target._base = result._base
    target._view_meta = result._view_meta
    return target


def functionalize_op(name, alias_name, entry, keyset, args, kwargs, redispatch):
    functional_name = _derive_functional_name(name)
    if not functional_name or not registry.has(functional_name):
        raise RuntimeError(f"functionalize: missing rule for op {alias_name}()")

    from .keys import DispatchKey

    out = redispatch(functional_name, keyset.without(DispatchKey.Functionalize), *args, **kwargs)
    mutated = _mutating_args(entry.schema_obj, args, kwargs)
    if mutated:
        target = mutated[0]
        return _writeback(target, out)
    return out
