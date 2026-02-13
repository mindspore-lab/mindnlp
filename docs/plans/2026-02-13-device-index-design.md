# Device index support design

## Goals
- Support torch-like device indices (e.g., `npu:1`, `cpu:1`, `meta:1`).
- Preserve type-based dispatch while carrying index in `device` objects.
- Use per-device runtimes for NPU with correct stream/context selection.
- Implement cross-device NPU copy via CPU staging.

## Scope
- Device parsing and `device` representation updates.
- NPU runtime pool keyed by device id.
- Update NPU ops/creation/to() paths to honor device index.
- Tests for parsing, repr, and cross-device copy.

## Device model
- `device` holds `type` and optional `index` (int or None).
- Parsing accepts `"type:index"` strings and `device(type, index)` constructor.
- `__repr__`/`__str__` matches torch:
  - `cpu`/`meta`/`npu` when index is None.
  - `cpu:1`, `meta:1`, `npu:1` when index is set.
- Default device remains `cpu` with index None.

## Dispatcher
- Registry stays keyed by `device.type`.
- Dispatch accepts `device` or string; resolves to `device`.
- Pipeline pending tensors keep full `device` (type+index).

## NPU runtime
- Maintain `_runtimes: {device_id: _Runtime}`.
- `get_runtime(device_id)` lazily creates and initializes.
- Each NPU op uses the runtime for the tensor’s device index.
- Model-dir probing can still use runtime 0.

## Cross-device NPU copy
- If `npu:X -> npu:Y` and X != Y:
  - `_copy_npu_to_cpu` using runtime X.
  - `_copy_cpu_to_npu` using runtime Y.
- Same-index NPU to NPU returns original tensor.

## Tests
- Device parsing/representation:
  - `device("npu:1")`, `device("cpu:1")`, `device("meta:1")`, `device("npu", 1)`.
  - `repr` and `index` values.
- Creation device index:
  - `torch.ones(..., device="npu:0")` sets `device.index == 0`.
- Cross-device copy (skip if only one NPU):
  - `npu:0 -> npu:1`, verify values on CPU.

## Files
- `src/mindtorch_v2/_device.py`
- `src/mindtorch_v2/_functional.py`
- `src/mindtorch_v2/_dispatch/dispatcher.py`
- `src/mindtorch_v2/_storage.py`
- `src/mindtorch_v2/_backends/npu/runtime.py`
- `src/mindtorch_v2/_backends/npu/creation.py`
- `src/mindtorch_v2/_backends/npu/ops.py`
- `src/mindtorch_v2/_backends/common/convert.py`
- Tests under `tests/mindtorch_v2/`
