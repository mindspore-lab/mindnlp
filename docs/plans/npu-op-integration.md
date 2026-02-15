# MindTorch v2 NPU Op Integration Flow

## Purpose
Document how NPU ops are wired into MindTorch v2 using ctypes + ACLNN, including the runtime, dispatch, and op-level flow.

## Key Files
- `src/mindtorch_v2/_dispatch/registry.py`: Op registry and lookup.
- `src/mindtorch_v2/_dispatch/dispatcher.py`: Dispatches to CPU/NPU/Meta kernels.
- `src/mindtorch_v2/_backends/npu/__init__.py`: Registers NPU kernels with `registry.register`.
- `src/mindtorch_v2/_backends/npu/ops.py`: Python NPU op implementations.
- `src/mindtorch_v2/_backends/npu/aclnn.py`: ctypes bindings to ACLNN and op wrappers.
- `src/mindtorch_v2/_backends/npu/runtime.py`: Runtime lifecycle, streams, memory alloc, sync.
- `src/mindtorch_v2/_backends/npu/allocator.py`: NPU device allocator.
- `src/mindtorch_v2/_backends/npu/state.py`: Current device/stream state.

## End-to-End Flow
1. **Registration**
   - Each op is registered in `src/mindtorch_v2/_backends/npu/__init__.py` via:
     - `registry.register("<op>", "npu", fn, meta=...)`
2. **Dispatch**
   - `dispatcher.py` resolves the canonical name and selects the NPU kernel for `device.type == "npu"`.
3. **NPU Python Op** (`ops.py`)
   - Validate inputs (device/dtype).
   - Get runtime + current stream.
   - Allocate output memory via `npu_runtime._alloc_device`.
   - Call the ACLNN wrapper in `aclnn.py`.
   - Wrap output pointer into a `Tensor` using `npu_typed_storage_from_ptr`.
4. **ACLNN ctypes wrapper** (`aclnn.py`)
   - Load ACL/ACLNN libs and symbols.
   - Build ACL tensor descriptors (`aclCreateTensor`).
   - Call `aclnn*GetWorkspaceSize` to get workspace + executor.
   - Allocate workspace (if needed), call `aclnn*`.
   - Optional sync via `_maybe_sync` (gated by `ACL_LAUNCH_BLOCKING`).
   - Cleanup executor, tensors, and workspace via `defer_free`.

## ACLNN Wrapper Pattern
Most wrappers follow the same structure:
1. `ensure_acl()` and `get_bindings()`.
2. Create input/output tensor descriptors.
3. Call `aclnn*_GetWorkspaceSize`.
4. Allocate workspace (if size > 0).
5. Call `aclnn*` with workspace + executor + stream.
6. `_maybe_sync(runtime)`.
7. Cleanup tensor handles and defer workspace free.

## Example (NPU add)
- `ops.py` allocates output memory and calls `aclnn.add(...)`.
- `aclnn.add`:
  - Creates input/output descriptors.
  - Calls `aclnnAddGetWorkspaceSize` then `aclnnAdd`.
  - Frees executor + tensors.

## Current NPU-Registered Ops
Registered in `src/mindtorch_v2/_backends/npu/__init__.py`:
- `add`, `add_`, `mul`, `mul_`, `relu`, `relu_`, `sum`, `contiguous`,
  `reshape`, `view`, `transpose`, `to`, `tensor`, `zeros`, `ones`, `empty`, `zero_`

## Notes
- NPU kernels use ctypes calls only (no MindSpore dependency).
- Meta/CPU registrations are parallel for shape and fallback support.
- NPU op kernels should avoid NumPy dependencies in the execution path.
- NPU matmul uses a performance-first path: try direct ACLNN matmul, then fallback to per-batch matmul with broadcasted batch indices if ACLNN rejects high-rank shapes.
- Fallback assumes contiguous inputs; inputs are made contiguous before fallback to allow pointer offset math.
