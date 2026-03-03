# MindTorch V2 Torch-Style Dispatcher + Pipeline Redesign

## Goals
- Fully align the dispatcher with Torch concepts: dispatch keys, key sets, kernel registration, and fallthrough semantics.
- Preserve current behavior while enabling a multi-stage pipeline to reduce Python overhead.
- Keep pipeline opt-in and strictly preserve Torch-visible semantics.

## Non-Goals
- No static graph compilation or background worker threads.
- No wholesale API surface expansion beyond dispatch/pipeline integration.
- No change in user-visible tensor semantics beyond dispatch selection and pipeline performance.

## Architecture Overview
The dispatcher is restructured around Torch-like primitives:
- **DispatchKey**: Enum of routing keys (CPU, NPU, Meta, Autograd, Pipeline).
- **DispatchKeySet**: A computed set per call derived from inputs + global modes.
- **OperatorEntry**: Stores op schema, registered kernels, and explicit fallthroughs.
- **Kernel resolution**: Select the highest-priority key that has a kernel, honoring fallthrough when a key is present but not intercepting.

This preserves Torch’s observable dispatch behavior while allowing the pipeline to be implemented as a wrapper key.

## DispatchKey Rules
Key set construction:
1) **Device keys**: CPU/NPU based on tensor inputs; Meta is added if any input is Meta (and Meta is the dominant device).
2) **Autograd key**: present when GradMode.enabled and any input requires_grad.
3) **Pipeline key**: present when pipeline context is active and device is not Meta.

Priority order (highest first):
Pipeline → Autograd → Meta → NPU → CPU

If a key has no kernel or is marked fallthrough, dispatch proceeds to the next key. Missing kernel after resolution is an error: “could not find kernel for op <name> with keys <keyset>”.

## Kernel Registration + Schema
Each op is registered in the dispatcher via:
- `register_schema(op, schema)` (string, Torch-style). Used for introspection and debug.
- `register_kernel(op, key, fn)` to register a key-specific kernel.
- `register_fallthrough(op, key)` to explicitly mark a key as pass-through.

Convenience helpers:
- `register_meta(op, fn)` → Meta key
- `register_backend(op, device, fn)` → CPU/NPU keys
- Optional wrappers for Autograd/Pipeline if those kernels are customized.

## Pipeline Integration
Pipeline is a wrapper key. When active:
- Dispatcher selects Pipeline kernel.
- Pipeline kernel executes Meta kernel for shape/dtype inference.
- Returns a Pending Tensor and records a PendingOp (plan + impl references).
- `flush()` runs plan for all ops, then impl for all ops in order, replacing pending storage with actual storage.

Pipeline does not change semantics; it only defers execution.

## Autograd + Pipeline Semantics
- Autograd nodes are created when pending ops execute (during flush), not during meta inference.
- `.backward()` and `torch.autograd.grad` must flush before backward.
- Data access boundaries (numpy, item, repr, to(cpu)) flush pending ops.

This preserves Torch-visible behavior while reducing Python overhead in pipeline mode.

## Error Handling
- Missing kernel or invalid keyset triggers a clear error that includes op name and keyset.
- Pipeline flush failures clear the queue and leave pending tensors invalid (explicit error on further use).

## Migration Plan (High-Level)
1) Introduce DispatchKey/DispatchKeySet and OperatorEntry while preserving current behavior.
2) Migrate registry registrations to key-based registrations (Meta/CPU/NPU).
3) Introduce Autograd + Pipeline keys as wrapper keys (fallthrough by default).
4) Implement pipeline key as multi-stage (meta/plan/impl), integrate flush points.

## Testing Strategy
- Unit tests for keyset construction and kernel resolution order.
- Regression tests for current op behavior (CPU/NPU/meta) with new dispatcher.
- Pipeline tests verifying deferred execution, flush triggers, and autograd correctness.

## P0 Checklist (Torch Alignment)

### Dispatcher
- **Dispatch key model & ordering**
  - Torch ref: `c10/core/DispatchKeySet.h`, `aten/src/ATen/core/dispatch/Dispatcher.h`
  - mindtorch target: `src/mindtorch_v2/_dispatch/keys.py`, `src/mindtorch_v2/_dispatch/dispatcher.py`
  - Minimal ops: `tensor/zeros/ones/empty`, `add/mul/matmul/relu/sum`, `reshape/view/transpose`, `to`, `contiguous`
- **Kernel registration + fallthrough/redispatch**
  - Torch ref: `aten/src/ATen/core/dispatch/OperatorEntry.cpp`, `aten/src/ATen/core/dispatch/Dispatcher.cpp`
  - mindtorch target: `src/mindtorch_v2/_dispatch/registry.py`, `src/mindtorch_v2/_dispatch/dispatcher.py`
  - Minimal ops: `add/mul/matmul/relu/sum` (and inplace variants)
- **Autograd key wrappers in dispatch**
  - Torch ref: `torch/csrc/autograd/VariableType`
  - mindtorch target: `src/mindtorch_v2/_dispatch/dispatcher.py`, `src/mindtorch_v2/_autograd/*`, `src/mindtorch_v2/_functional.py`
  - Minimal ops: `add/mul/matmul/relu/sum`, `reshape/view/transpose`
- **Composite/BackendSelect style kernels**
  - Torch ref: `aten/src/ATen/native` composite kernels + `DispatchKey` definitions
  - mindtorch target: `src/mindtorch_v2/_dispatch/registry.py`, `src/mindtorch_v2/_functional.py`
  - Minimal ops: `reshape/view/transpose`, `sum`
- **Pipeline integration without bypassing dispatch**
  - Torch ref: N/A (pipeline is mindtorch-specific)
  - mindtorch target: `src/mindtorch_v2/_dispatch/pipeline.py`, `src/mindtorch_v2/_dispatch/dispatcher.py`
  - Minimal ops: `add/mul/matmul/relu/sum` with meta kernels

### Autograd
- **Autograd dispatch wrappers**
  - Torch ref: `torch/csrc/autograd/VariableType`, `variable_tensor.cpp`
  - mindtorch target: `src/mindtorch_v2/_dispatch/dispatcher.py`, `src/mindtorch_v2/_autograd/*`, `src/mindtorch_v2/_functional.py`
  - Minimal ops: `add/mul/matmul/relu/sum`, `reshape/view/transpose`
- **Version counter + inplace/view semantics**
  - Torch ref: `VariableTypeUtils.h`, `VariableVersion`
  - mindtorch target: `src/mindtorch_v2/_tensor.py`, `src/mindtorch_v2/_autograd/version_counter.py`, `src/mindtorch_v2/_functional.py`
  - Minimal ops: `view/reshape/transpose`, `add_/mul_/relu_/zero_`
- **View metadata propagation**
  - Torch ref: `ViewOps.cpp`
  - mindtorch target: `src/mindtorch_v2/_tensor.py`, `src/mindtorch_v2/_functional.py`
- **Saved tensors + hooks**
  - Torch ref: `SavedTensorHooks`, `saved_variable.cpp`
  - mindtorch target: `src/mindtorch_v2/_autograd/node.py`, `src/mindtorch_v2/_autograd/utils.py`
- **Grad mode correctness**
  - Torch ref: `GradMode`
  - mindtorch target: `src/mindtorch_v2/_dispatch/keys.py`, `src/mindtorch_v2/_autograd/grad_mode.py`, `src/mindtorch_v2/_functional.py`

### Meta + Streams
- **Meta kernels for core ops**
  - Torch ref: `aten/src/ATen/meta`, FakeTensor
  - mindtorch target: `src/mindtorch_v2/_backends/meta/*` (or meta kernels in op files), `src/mindtorch_v2/_dispatch/registry.py`
  - Minimal ops: `tensor/zeros/ones/empty`, `add/mul/matmul/relu/sum`, `reshape/view/transpose`
- **Meta tensor data access guards**
  - Torch ref: meta storage
  - mindtorch target: `src/mindtorch_v2/_tensor.py`, `src/mindtorch_v2/_storage.py`
- **Meta + autograd graph build**
  - Torch ref: FakeTensor autograd
  - mindtorch target: `src/mindtorch_v2/_autograd/*`, `src/mindtorch_v2/_functional.py`
- **Thread-local current stream**
  - Torch ref: `torch/cuda/streams.py`, stream pool
  - mindtorch target: `src/mindtorch_v2/_backends/npu/state.py`, `src/mindtorch_v2/npu.py`
- **record_stream + non_blocking copy rules**
  - Torch ref: `torch/cuda/memory.py`, `THC` copy kernels
  - mindtorch target: `src/mindtorch_v2/_backends/common/convert.py`, `src/mindtorch_v2/_backends/npu/runtime.py`, `src/mindtorch_v2/_tensor.py`
- **Synchronization semantics**
  - Torch ref: `cuda::CUDAGuard`, `CUDA_LAUNCH_BLOCKING`
  - mindtorch target: `src/mindtorch_v2/_backends/npu/runtime.py`, `src/mindtorch_v2/_backends/npu/state.py`, `Tensor.to`/`numpy`
