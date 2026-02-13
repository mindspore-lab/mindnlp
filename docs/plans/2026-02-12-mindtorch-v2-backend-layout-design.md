# MindTorch v2 Backend Layout (Device-Centric)

## Goal

Reorganize `src/mindtorch_v2/_backends` by device so that runtime, ops, and creation logic live together per device, while shared helpers live in a small common area. Keep dispatch integration unchanged and provide a compatibility shim for existing imports.

## Non-Goals

- No behavior changes to dispatch or op implementations.
- No new ops or dtype changes.
- No removal of existing public APIs without a shim.

## Proposed Layout

```
src/mindtorch_v2/_backends/
  __init__.py               # imports cpu/meta/npu to register ops
  common/
    __init__.py
    view.py                 # storage-sharing view ops
    convert.py              # to() helper (device conversion)
  cpu/
    __init__.py             # registry.register calls for cpu/meta kernels
    ops.py                  # add/mul/relu/sum
    creation.py             # tensor/zeros/ones/empty
    meta.py                 # meta kernels used by cpu pipeline
  meta/
    __init__.py             # registry.register for pure meta device
    ops.py                  # meta kernels (shape-only)
    creation.py             # meta creation ops
  npu/
    __init__.py             # registry.register for npu ops/creation
    runtime.py              # acl init, stream/context, model dir probe
    aclnn.py                # ctypes bindings (aclnn)
    ops.py                  # add/mul/relu/sum using aclnn
    creation.py             # tensor/zeros/ones/empty
  ascend.py                 # shim: re-export npu runtime/ops
```

## Dispatch Integration

- `_functional` continues to call `dispatch(name, device, ...)` unchanged.
- Each device package registers its ops in its `__init__.py`.
- Shared helpers in `common/` are imported by the device packages.
- Meta kernels remain available for pipeline usage; CPU can register meta kernels for its ops to support pipeline mode.

## Error Handling and Runtime

- `npu/runtime.py` owns lazy ACL initialization and a single model dir probe.
- `npu.is_available()` returns `False` if ACL is missing/unusable.
- Avoid per-op model dir checks; runtime ensures initialization once.
- Keep tests that probe model dir behavior by providing best-effort probe.

## Compatibility

- Keep `src/mindtorch_v2/_backends/ascend.py` as a shim that re-exports NPU runtime/ops to avoid breaking existing imports in the codebase and tests.
- Gradually update call sites to import from `_backends.npu` directly.

## Testing

- Existing tests should continue to pass without modification due to the shim.
- If desired later, update tests to import `mindtorch_v2._backends.npu` explicitly.

## Migration Steps (High-Level)

1. Create `common/`, `cpu/`, `meta/`, `npu/` directories and move files.
2. Update imports and registration to new module paths.
3. Add `ascend.py` shim (or keep existing file with re-exports).
4. Verify dispatch registration is intact.
5. Run targeted tests (creation, view, npu ops, storage).
