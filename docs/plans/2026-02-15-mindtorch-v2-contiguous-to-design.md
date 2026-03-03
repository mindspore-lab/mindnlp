# MindTorch v2 Contiguous + To Autograd Design

## Goal
Align `Tensor.contiguous()` and `Tensor.to()` with Torch dispatcher and autograd behavior while keeping backend kernels device-specific and efficient.

## Non-Goals
- No new ops beyond `contiguous`/`to`.
- No changes to pipeline scheduling or allocator policy.
- No dtype promotion redesign.

## Current State
- `Tensor.contiguous()` calls `dispatch("contiguous", self)` but also has stale manual autograd wiring in `_tensor.py` using legacy fields.
- `to()` is dispatched to `convert_backend.to_device`, but there is no Autograd wrapper for `to`.
- No `contiguous` kernels are registered in CPU/NPU/Meta backends.

## Torch Alignment Notes
- Autograd behavior is provided by the Autograd dispatch key; tensor methods should remain thin.
- `contiguous` returns `self` if already contiguous.
- `to` returns `self` if it is a no-op (same device/dtype), otherwise returns a converted tensor.
- Backward for `contiguous` is identity (grad passes to input). Backward for `to` converts grad back to input device/dtype.

## Design
### 1) Dispatcher + Autograd Wrappers
- Add Autograd wrappers for `contiguous` and `to` in `src/mindtorch_v2/_backends/autograd.py`.
- Wrapper flow:
  1. `keyset = current_dispatch_keyset().without(DispatchKey.Autograd)`
  2. `out = redispatch(name, keyset, ...)`
  3. If `GradMode.enabled` and input requires grad, attach a grad node.
- `contiguous` backward: pass-through grad; if needed, call `redispatch("contiguous", keyset, grad)` to keep layout consistent.
- `to` backward: convert grad back to input device/dtype via `redispatch("to", keyset, grad, input.device, non_blocking=False)`.

### 2) Backend Kernels
- CPU `contiguous`: use `np.ascontiguousarray` to create a contiguous buffer, then wrap into `Tensor` storage.
- NPU `contiguous`: allocate new device memory and D2D copy from input to output using existing runtime/allocator APIs.
- Meta `contiguous`: return a meta TensorSpec with contiguous strides; no data allocation.
- Register `contiguous` in CPU/NPU/Meta backends with appropriate meta inference.

### 3) Tensor Method Cleanup
- Remove manual autograd logic from `Tensor.contiguous()` in `_tensor.py`.
- Keep `Tensor.contiguous()` as thin wrapper using dispatch only.

## Error Handling
- Meta tensors should not attempt data access; shape/dtype inference only.
- `to` should raise if device/dtype unsupported by backend.

## Testing
- Add tests that verify:
  - `contiguous` produces a `grad_fn` when `requires_grad=True` and backward populates `x.grad`.
  - `to` creates a `grad_fn` on real conversion, and backward converts grad back to input device/dtype.
  - CPU-only tests are sufficient; NPU tests guarded by `torch.npu.is_available()`.

## Open Questions
- None; proceed with implementation and TDD.
