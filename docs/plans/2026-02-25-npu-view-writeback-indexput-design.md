# NPU View Writeback via IndexPutImpl Design

## Goal
Align functionalize view writeback on NPU with torch/torch-npu semantics by supporting non-contiguous views while keeping contiguous views on the fast D2D memcpy path.

## Architecture
- Keep existing contiguous view writeback via D2D memcpy.
- Add an ACLNN IndexPutImpl path for non-contiguous views.
- Generate indices on NPU using arange + reshape/broadcast, then compute linear indices from view stride/offset.
- Call `aclnnIndexPutImpl` with `[linear_index]` and `values` (flattened result) to update the base tensor.

## Data Flow
1. Identify base tensor, view shape/stride/offset.
2. Build per-dim index tensors using `aclnnArange`.
3. Reshape and broadcast indices to the view shape.
4. Compute `linear = offset + sum(idx_i * stride_i)` on NPU.
5. `base_flat = base.reshape(-1)` and `values = result.reshape(-1)`.
6. Invoke `aclnnIndexPutImpl(base_flat, [linear], values, accumulate=False, unsafe=False)`.
7. Update view metadata to reflect writeback.

## Error Handling
- If `aclnnIndexPutImpl` or `aclnnArange` are unavailable, raise explicit runtime errors.
- Preserve existing `functionalize writeback shape mismatch` checks.
- Errors may differ in wording but should match torch behavior.

## Testing
- Add NPU test for non-contiguous view writeback under functionalize.
- Ensure contiguous view path remains unchanged.
