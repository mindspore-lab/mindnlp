# NPU async execution design

## Goals
- Remove per-op `acl.rt.synchronize_stream` to allow async execution.
- Add `torch.npu.synchronize(device=None)` explicit sync API.
- Keep correctness by deferring workspace frees until sync.
- Honor `ACL_LAUNCH_BLOCKING=1` to force per-op sync for debugging.
- Implicitly synchronize on CPU-visible access (NPU->CPU copy, numpy, print).

## Behavior summary
- Default NPU ops enqueue work on the runtime stream and return.
- `ACL_LAUNCH_BLOCKING=1` forces `runtime.synchronize()` after each NPU op.
- `torch.npu.synchronize()` blocks and drains deferred frees.
- NPU->CPU copies call `runtime.synchronize()` before D2H.
- Cross-device NPU copy uses CPU staging, synchronizing on the source runtime.

## Resource management
- Each NPU runtime keeps a `_deferred_frees` list.
- ACLNN wrappers call `runtime.defer_free(ptr)` instead of `acl.rt.free`.
- `runtime.synchronize()` calls `acl.rt.synchronize_stream` and frees all deferred pointers.
- Executor destruction remains deferred to exit (unchanged).

## API
- `torch.npu.synchronize(device=None)`
  - `None` => default `Device("npu")` (index 0)
  - accepts `str` or `Device` for index selection

## Tests
- Runtime deferred free drain:
  - enqueue two frees, call `synchronize()`, assert frees executed once.
- `ACL_LAUNCH_BLOCKING` mode:
  - ensure per-op sync is invoked (via monkeypatching `runtime.synchronize`).
- Implicit sync on NPU->CPU:
  - verify `runtime.synchronize` is called before D2H copy.

## Files
- `src/mindtorch_v2/_backends/npu/runtime.py`
- `src/mindtorch_v2/_backends/npu/aclnn.py`
- `src/mindtorch_v2/_backends/npu/ops.py`
- `src/mindtorch_v2/_backends/npu/creation.py`
- `src/mindtorch_v2/_backends/common/convert.py`
- `src/mindtorch_v2/npu.py`
- `tests/mindtorch_v2/test_runtime.py`
