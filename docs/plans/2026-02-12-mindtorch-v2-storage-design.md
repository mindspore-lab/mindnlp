# MindTorch V2 Storage Alignment Design

## Goals
- Align with torch storage model (TypedStorage + UntypedStorage).
- Ensure NPU storage holds only device memory (no CPU copies).
- Provide CPU‑only shared memory and file‑backed storage APIs.

## Non-Goals
- No shared memory or file‑backed support for NPU/Meta.
- No legacy dtype‑specific storage classes (FloatStorage, etc.).

## Object Model
- **UntypedStorage**: raw byte storage with device + nbytes + data_ptr.
- **TypedStorage**: dtype + numel view over UntypedStorage.
- Device implementations:
  - **CPUUntypedStorage**: bytearray / numpy buffer or memmap for file‑backed.
  - **NPUUntypedStorage**: device pointer + byte size only.
  - **MetaUntypedStorage**: size only, no data pointer.

## TypedStorage API
- `dtype`, `device`, `size()`, `nbytes()`, `data_ptr()`
- `clone()`, `copy_(other)`, `resize_(new_size)`
- `untyped_storage()` and `is_shared()`

## UntypedStorage API
- `device`, `nbytes()`, `data_ptr()`, `resize_(new_nbytes)`
- `share_memory_()`, `is_shared()` (CPU only)
- `from_file(path, shared=False)` and `filename()` (CPU only)

## Tensor Integration
- `Tensor.storage()` returns TypedStorage.
- `Tensor.untyped_storage()` returns underlying untyped storage.
- `Tensor.to(device)` materializes by allocating new TypedStorage (empty/uninitialized for meta→cpu/npu).
- `Tensor.numpy()` reads from CPU typed storage only.

## Error Handling
- NPU/Meta `share_memory_` and `from_file` raise `NotImplementedError`.
- Meta `data_ptr()` raises `RuntimeError`.

## Notes
- Keep a small legacy alias if needed, but internal code uses typed/untyped.
- CPU file‑backed storage uses `np.memmap` to avoid loading into host RAM.
