# NPU ACLNN Operator Integration Summary (Supplement)

Goal: Only add NPU operators inside MindTorch v2, avoid any MindSpore dependency; prohibit NumPy in the NPU execution path; fully align behavior with PyTorch.

## Integration Steps (General Flow)
1. Confirm the operator already exists on CPU with PyTorch-aligned behavior.
2. Add ACLNN ctypes bindings in `src/mindtorch_v2/_backends/npu/aclnn.py`:
   - Use common wrappers like `_unary_call` / `_binary_call`.
   - Match output dtype to PyTorch semantics (e.g., `isfinite`/`isinf`/`isnan`/`signbit` -> `bool`).
3. Add NPU front-end wrappers in `src/mindtorch_v2/_backends/npu/ops.py`:
   - Validate NPU device and dtype.
   - Allocate output via `npu_runtime._alloc_device` and wrap with `npu_typed_storage_from_ptr`.
   - Avoid NumPy; execution must go through ACLNN / existing NPU ops only.
4. Add tests in `tests/mindtorch_v2/test_ops_npu.py`:
   - Write tests first, then implement (TDD).
   - Compare CPU vs NPU results to ensure identical semantics.
5. Run the target test file and keep it green before expanding further.

## Special Cases Observed
- `aclnnIsInf` unavailable (error `161001`):
  - Fallback: combine `isfinite` with reciprocal logic.
  - Rule: `isinf = (~isfinite(x)) & isfinite(1/x)`.
- `aclnnIsPosInf` / `aclnnIsNegInf` unavailable (error `161002`):
  - Avoid direct usage; rely on `isinf` fallback logic.
- `aclnnEqScalar` returns incorrect results for `+/-inf` (all False):
  - Do not use it for inf detection.
- `hardtanh` unsupported on some environments (error `561103`):
  - Fallback: `clamp(x, min, max)`.
- `isfinite` / `isinf` / `isnan` / `signbit` must return `bool`:
  - Allocate output and wrap storage as `bool` dtype.

## Performance & Safety Constraints
- Do not introduce NumPy in the NPU execution path.
- Avoid unnecessary `try/except`; only add fallbacks for known ACLNN limitations.
- Do not add any MindSpore dependency.

## Operators Added In This Round (Example List)
- `clamp` (scalar/tensor)
- `cosh` / `sinh` / `erf` / `erfc` / `softplus`
- `relu6` / `hardtanh`
- `isfinite` / `isinf` / `isnan` / `signbit`
- `logical_not` / `logical_and` / `logical_or` / `logical_xor`
- `eq` / `ne` (tensor + scalar)
