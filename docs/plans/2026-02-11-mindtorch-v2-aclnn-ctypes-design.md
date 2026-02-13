# MindTorch v2 ACLNN Ctypes Design

**Goal**
Provide real Ascend NPU execution for `add/mul/relu/sum/matmul` without OM conversion or `torch_npu`, using ACLNN shared libraries via ctypes.

**Non-Goals**
- Full PyTorch coverage or dynamic shapes beyond the minimal ops above.
- Silent CPU fallback when an NPU op is requested.
- Using OM conversion or `torch_npu` bindings.

## Architecture
- Add `src/mindtorch_v2/_backends/ascend_ctypes.py` to load ACLNN shared libs (e.g., `libaclnn_ops_infer.so`, `libaclnn_math.so`) and expose a minimal API for the core ops.
- Keep `src/mindtorch_v2/_backends/ascend.py` as the NPU backend facade: initialize ACL runtime (device, context, stream), validate contiguous inputs, and forward to the ctypes layer.
- Use `NpuStorage` to keep device pointers and metadata; Tensors on NPU do not expose NumPy data until `.to("cpu")`.

## Data Flow
1. CPU Tensor creation uses NumPy as before.
2. `.to("npu")` allocates device memory with `acl.rt.malloc`, copies H2D, and wraps in `NpuStorage`.
3. NPU ops create ACLNN tensor descriptors from input pointers, shapes, and dtypes; allocate output buffers and workspace; run the op on the stream; synchronize; wrap output pointers in `NpuStorage` and return a new Tensor.
4. `.to("cpu")` copies D2H and returns a CPU Tensor.

## Error Handling
- Each ACL/ACLNN call checks return codes and raises `RuntimeError` with op name, dtype, shapes, and return code.
- Missing ACLNN symbols for any required op marks `torch.npu.is_available()` as false and raises clear errors when invoked.
- Non-contiguous tensors on NPU ops raise a clear error suggesting `.contiguous()` on CPU before transfer.

## Testing
- NPU-only tests cover `add/mul/relu/sum/matmul` forward and device roundtrip.
- Tests skip when `torch.npu.is_available()` is false.
- Minimal dtype coverage: float32/float16 and int64 where supported by ACLNN.
