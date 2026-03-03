# MindTorch V2 ACL Lifecycle (Lazy Init + Exit Cleanup)

## Goals
- Initialize ACL and ACLNN lazily on first NPU use.
- Allow normal execution of multiple NPU ops per process without per-op init/finalize.
- Release ACLNN executors and ACL runtime resources at process exit.
- Keep API surface aligned with PyTorch CUDA (no explicit shutdown).

## Non-Goals
- No explicit public `torch.npu.shutdown()` API.
- No eager initialization on import.
- No per-test cleanup guarantees within a single long-lived process.

## Design Summary
- **Runtime init (ACL runtime + stream)** occurs on first NPU use via `ascend._Runtime.init()`.
- **ACLNN init** occurs on first ACLNN binding use via `ascend_ctypes.get_bindings()` calling `aclnnInit()` once.
- **Executor lifecycle**: destroy ACLNN executors at process exit, not per op, to avoid crashes after sequential ops (mul→relu). Executors are stored in a deferred list and freed in an `atexit` handler.
- **Resource cleanup**: at exit, destroy deferred executors, call `aclnnFinalize` (if available), destroy stream/context, reset device, and call `acl.finalize`.

## Data Flow Per NPU Op
1. `_runtime.init(0)` ensures ACL runtime and stream are ready.
2. Create ACLNN tensors wrapping device pointers, view shape, and stride.
3. Call `aclnn*GetWorkspaceSize` to obtain executor + workspace size.
4. Allocate workspace if required.
5. Execute `aclnn*` op and synchronize stream.
6. Destroy temporary ACLNN tensor/scalar/int-array objects and free workspace.
7. Defer executor destruction to process exit.

## Error Handling
- Fail fast on non-zero return codes from `acl.init`, `acl.rt.set_device`, `aclnnInit`, and `aclnn*` calls.
- In `finally` blocks: always release temporary ACLNN objects and workspace; executors are deferred.

## Notes / Caveats
- Memory may remain reserved until process exit (similar to CUDA context behavior).
- For multi-process workloads, each process will lazily initialize its own runtime and cleanup at exit.
- The model dir must be set successfully before NPU op execution; failure is a hard error.
