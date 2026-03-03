# NPU Allocator OOM + Alloc Conf (Torch Alignment)

## Goals
- Align NPU caching allocator behavior with torch.cuda for OOM retries and GC-driven cache reclaim.
- Support `MINDTORCH_NPU_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` with torch-like parsing and warnings.
- Add `mindtorch_v2.npu.mem_get_info()` to enable GC threshold decisions.
- Keep allocator fast in the common case; only pay extra cost on OOM or when GC threshold is exceeded.

## Non-goals
- Full 1:1 CUDA allocator internals (block graph, per-stream arenas, exact fragmentation behavior).
- Implement all allocator config knobs (future work).

## Config Surface
- Read `MINDTORCH_NPU_ALLOC_CONF` first; if unset, fall back to `PYTORCH_CUDA_ALLOC_CONF`.
- Accept `key:value` and `key=value` pairs, comma-separated.
- Parse all keys; unsupported keys are ignored with `warnings.warn`.
- Supported keys (this phase):
  - `max_split_size_mb` (int, MB -> bytes).
  - `garbage_collection_threshold` (float 0.0-1.0).
- Defaults:
  - `max_split_size_mb`: disabled (no split cap) to match torch default.
  - `garbage_collection_threshold`: disabled.

## OOM Retry Path (Torch-Style)
1. Try allocation (cached reuse or raw device alloc).
2. On raw alloc failure:
   - Increment `num_ooms`.
   - If GC threshold enabled, run GC pass (non-blocking):
     - Drain completed pending events into cache.
     - Free cached blocks until `reserved_bytes / total_bytes <= threshold`.
   - Synchronize device, drain pending, and free all cached blocks (`empty_cache`).
   - Retry raw alloc once.
3. If retry succeeds, increment `num_alloc_retries` and proceed.
4. If retry fails, raise OOM.

## GC Threshold Behavior
- If enabled, check `reserved_bytes / total_device_bytes` before new allocations.
- Use `mem_get_info()` to obtain total bytes.
- If `mem_get_info` is unavailable, disable GC threshold and warn once.
- GC pass does not touch active blocks and avoids device-wide sync.

## mem_get_info API
- Add `mindtorch_v2.npu.mem_get_info(device=None)` returning `(free_bytes, total_bytes)`.
- Implement via `acl.rt.get_mem_info(attr)` if available.
- Use the same device normalization as other NPU APIs.

## Stats Alignment
- `num_ooms` increments on first raw allocation failure.
- `num_alloc_retries` increments only when retry path is taken.
- `reserved_bytes` and `segment` decrease only when cached blocks are freed.

## Error Handling
- Invalid config values: warn and ignore, keep defaults.
- Unsupported keys: warn and ignore.
- `mem_get_info` missing: warn once; disable GC threshold.

## Testing
- Config parsing precedence and unsupported-key warnings.
- `mem_get_info` via mocked `acl.rt.get_mem_info`.
- GC threshold triggers cached block reclaim.
- OOM retry path increments `num_ooms`/`num_alloc_retries`.

## Open Questions
- Exact attr value for `acl.rt.get_mem_info` (default `0` as in Ascend examples).
- Whether to expose parsed allocator config for diagnostics (debug-only helper).
