import contextlib
import numpy as np

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



def _mutating_slots(schema_obj, args, kwargs):
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
    slots = []
    for param in params:
        if not param.mutates:
            continue
        alias_set = getattr(param, "alias_set", None)
        if alias_set in (None, ""):
            continue
        if param.name in bound:
            slots.append((alias_set, bound[param.name]))
    return slots



def _mutating_args(schema_obj, args, kwargs):
    return [target for _, target in _mutating_slots(schema_obj, args, kwargs)]



def _resolve_writeback_pairs(schema_obj, functional_schema_obj, args, kwargs, out, alias_name):
    slots = _mutating_slots(schema_obj, args, kwargs)
    if not slots:
        return []

    mutated = [target for _, target in slots]
    if len(mutated) == 1:
        return [(mutated[0], out)]

    if not isinstance(out, (tuple, list)):
        raise RuntimeError(f"functionalize: expected tuple output for op {alias_name}()")
    if len(out) != len(mutated):
        raise RuntimeError(f"functionalize: output count mismatch for op {alias_name}()")

    returns = []
    if functional_schema_obj is not None:
        returns = getattr(functional_schema_obj, "returns", []) or []
    if not returns:
        return list(zip(mutated, out))

    alias_slots = {}
    for idx, (alias_set, _) in enumerate(slots):
        alias_slots.setdefault(alias_set, []).append(idx)

    # Prefer alias-driven mapping from functional returns to mutating inputs,
    # and fall back to positional assignment for unmatched outputs.
    remaining = list(range(len(mutated)))
    pairs = []
    for idx, result in enumerate(out):
        alias_set = None
        if idx < len(returns):
            alias_set = getattr(returns[idx], "alias_set", None)

        target_idx = None
        if alias_set not in (None, ""):
            queue = alias_slots.get(alias_set)
            if queue:
                target_idx = queue.pop(0)
                for rem_pos, rem_idx in enumerate(remaining):
                    if rem_idx == target_idx:
                        remaining.pop(rem_pos)
                        break

        if target_idx is None:
            if not remaining:
                raise RuntimeError(f"functionalize: output count mismatch for op {alias_name}()")
            target_idx = remaining.pop(0)

        pairs.append((mutated[target_idx], result))

    return pairs



def _compute_strides(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))

def _derive_functional_name(op_name):
    if op_name.endswith("_"):
        return op_name[:-1]
    return registry.get_functionalize(op_name)


def _is_contiguous_view(tensor):
    if tensor._view_meta is None:
        return False
    expected = _compute_strides(tensor.shape)
    return tensor.stride == expected


def _writeback(target, result, op_name=None):
    if target is result:
        return target
    if getattr(target, "device", None) is not None and target.device.type == "meta":
        if target._view_meta is None:
            target.shape = result.shape
            target.stride = result.stride
            target.offset = result.offset
            target._base = result._base
            target._view_meta = result._view_meta
        return target

    # View-aware writeback: only update the logical elements of `target`.
    # This matches torch in-place semantics for views.
    if target.device.type == "npu":
        from .._backends.npu import runtime as npu_runtime
        from .._backends.npu import ops as npu_ops

        base = target._base if target._base is not None else target
        if _is_contiguous_view(target):
            itemsize = result.element_size()
            base_storage = base.storage()
            src_storage = result.storage()
            runtime = npu_runtime.get_runtime((result.device.index or 0))
            numel = result.numel()
            copy_size = int(numel * itemsize)
            dst_ptr = int(base_storage.data_ptr()) + int(target.offset * itemsize)
            src_ptr = int(src_storage.data_ptr()) + int(result.offset * itemsize)
            ret = npu_runtime.acl.rt.memcpy(dst_ptr, copy_size, src_ptr, copy_size, 3)
            if ret != npu_runtime.ACL_ERROR_CODE:
                raise RuntimeError(f"acl.rt.memcpy D2D failed: {ret}")
        else:
            if not npu_ops.aclnn.arange_symbols_ok():
                raise RuntimeError("aclnnArange symbols not available")
            if not npu_ops.aclnn.index_put_impl_symbols_ok():
                raise RuntimeError("aclnnIndexPutImpl symbols not available")
            if target.numel() != result.numel():
                raise RuntimeError("functionalize writeback shape mismatch")
            base_flat = base.reshape((base.numel(),))
            values = result.reshape((result.numel(),))
            linear = npu_ops._npu_linear_index(target.shape, target.stride, target.offset, target.device)
            linear = linear.reshape((linear.numel(),))
            npu_ops.npu_index_put_impl(base_flat, linear, values, accumulate=False, unsafe=False)
        if target._view_meta is None:
            target.shape = result.shape
            target.stride = result.stride
            target.offset = result.offset
            target._base = result._base
            target._view_meta = result._view_meta
        return target
    if target.device.type != "cpu":
        raise RuntimeError(f"aten::{op_name} is not implemented for NPU")

    dst = target._numpy_view()
    src = result._numpy_view()
    if dst.shape != src.shape:
        raise RuntimeError("functionalize writeback shape mismatch")
    np.copyto(dst, src)

    if target._view_meta is None:
        target.shape = result.shape
        target.stride = result.stride
        target.offset = result.offset
        target._base = result._base
        target._view_meta = result._view_meta
    return target


def _bump_writeback_versions(mutated):
    seen = set()
    for target in mutated:
        bump_target = target._base if target._base is not None else target
        counter = getattr(bump_target, "_version_counter", None)
        if counter is None:
            continue
        key = id(counter)
        if key in seen:
            continue
        counter.bump()
        seen.add(key)


def functionalize_op(name, alias_name, entry, keyset, args, kwargs, redispatch, pipeline=None, dispatch_device=None):
    functional_name = _derive_functional_name(name)
    if not functional_name or not registry.has(functional_name):
        raise RuntimeError(f"functionalize: missing rule for op {alias_name}()")

    from .keys import DispatchKey


    if pipeline is not None:
        from .dispatcher import _pending_tensor_from_spec, _infer_dispatch_device, _extract_tensors, _FunctionalizePendingOp

        meta = entry.kernels.get(DispatchKey.Meta)
        if meta is None:
            raise RuntimeError(f"pipeline requires meta kernel for op {name}")
        meta_kwargs = kwargs if kwargs is not None else {}
        spec = meta(*args, **meta_kwargs)
        out = _pending_tensor_from_spec(spec, _infer_dispatch_device(dispatch_device, _extract_tensors(args, kwargs or {}), keyset))
        out._pending = True

        def _thunk():
            trimmed = keyset.without({DispatchKey.Functionalize, DispatchKey.Pipeline})
            return redispatch(functional_name, trimmed, *args, **kwargs)

        pipeline.record(_FunctionalizePendingOp(out, _thunk, keyset.without(DispatchKey.Pipeline), DispatchKey.Functionalize), pending=out)
        return out

    out = redispatch(functional_name, keyset.without(DispatchKey.Functionalize), *args, **kwargs)
    functional_schema_obj = None
    if registry.has(functional_name):
        functional_schema_obj = registry.get(functional_name).schema_obj

    pairs = _resolve_writeback_pairs(entry.schema_obj, functional_schema_obj, args, kwargs, out, alias_name)
    if not pairs:
        return out

    written = []
    for target, result in pairs:
        _writeback(target, result, op_name=alias_name)
        written.append(target)
    _bump_writeback_versions(written)
    return pairs[0][0]
