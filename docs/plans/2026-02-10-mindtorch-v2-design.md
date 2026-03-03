# Mindtorch v2 Design

Goal: Rebuild `mindtorch_v2` as a minimal PyTorch-like core with CPU (NumPy) and NPU (pyacl/ACLNN) backends, supporting Tensor/Storage/Autograd with stride-based views, broadcasting, and a minimal nn/optim stack.

## Scope (v0.1)
- Devices: CPU (NumPy) and NPU (Ascend via pyacl/ACLNN).
- Dtypes: float32, float16, int64.
- Tensor ops (forward + backward): add, mul, matmul, relu, sum (dim/keepdim).
- Views: reshape, transpose, stride/offset views (no implicit copies).
- Autograd: tape-based, Tensor.backward(gradient=...) for non-scalar outputs.
- Broadcasting: PyTorch-style; backward reduces gradients to original shapes.
- No in-place ops (no add_, relu_, etc.).
- Data movement: explicit Tensor.to(device) with CPU<->NPU copies.
- Minimal nn: Module, Parameter, Linear, ReLU, Sequential.
- Minimal optim: SGD (optional momentum), no weight decay in v0.1.

## ACL (pyacl) Usage Summary
- pyacl provides Python APIs for Ascend runtime management, operator calls, model management, and media processing.
- Standard lifecycle from docs: `acl.init` -> runtime resource allocation -> op/model/media execution -> runtime resource release -> `acl.finalize`.
- API categories include runtime (rt) and op (op) for operator execution; naming is `acl + category + Verb + Object`.
- For v0.1, only runtime + operator execution are required; model/media flows are out of scope.

## Architecture Overview
- Core layers: Tensor, Storage, Autograd, Dispatcher, Backends (cpu, npu).
- Storage owns a contiguous buffer, device, and dtype.
- Tensor is a view on Storage with shape/stride/offset, plus grad metadata.
- Dispatcher chooses backend op based on Tensor.device.
- CPU backend uses NumPy; NPU backend uses pyacl/ACLNN ops.
- Autograd uses a tape of Function nodes; each op registers a backward closure.

## Component Design
### 1) Tensor + Storage
- Storage: host buffer (NumPy array) or device buffer (ACL allocator handle).
- Tensor: shape/stride/offset, requires_grad, grad, grad_fn.
- View ops: reshape/transpose adjust metadata only.
- Data transfer: Tensor.to("cpu") or Tensor.to("npu") performs explicit copy.

### 2) Dispatcher + Backends
- Global op registry: op name -> backend implementation.
- CPU ops: implemented with NumPy; return Tensor on CPU.
- NPU ops: implemented via ACLNN; return Tensor on NPU.
- Mixed-device ops: error in v0.1 (require explicit .to()).

### 3) Autograd
- Grad mode switch: enable/disable context manager.
- Each op captures input metadata and produces a backward function.
- Broadcast-aware backward: reduce gradient along broadcasted dims.
- No in-place ops; preserves graph integrity and view safety.

### 4) nn / optim
- Module: registration, parameters(), train/eval, state_dict().
- Parameter: Tensor wrapper with requires_grad True.
- Linear: weight + bias, matmul + add.
- ReLU: functional relu.
- Sequential: simple forward chain.
- SGD: parameter update under no_grad, optional momentum.

## Data Flow Examples
1) CPU: Tensor -> add -> Tensor (NumPy) -> backward -> grad (NumPy)
2) NPU: Tensor.to("npu") -> matmul (ACLNN) -> Tensor -> backward -> grad (ACLNN)
3) Mixed: user must align devices with .to() before ops.

## Error Handling
- Raise explicit errors for mixed-device ops, unsupported dtypes, or missing gradient for non-scalar backward.
- Backend errors should surface with op name and device for debugging.

## Testing Strategy
- Unit tests for Tensor shape/stride/view, broadcasting, and autograd correctness.
- CPU backend tests compare against NumPy.
- NPU backend tests are smoke tests for op execution and basic gradients.

## Migration Notes
- Existing `src/mindtorch_v2` will be removed and rebuilt from scratch.
- Import path remains `import mindtorch_v2 as torch` to avoid breaking v1 users.
