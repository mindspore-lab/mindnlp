# MindTorch V2 Meta Device Design

## Goals
- Support `device="meta"` tensors for shape-only modeling.
- Allow cheap model initialization without host memory allocation.
- Enable distributed planning and memory estimation using metadata only.
- Allow `meta` tensors to materialize into real tensors with `to("cpu"/"npu")` using empty/uninitialized storage.

## Non-Goals
- No implicit data initialization on meta tensors.
- No real compute on meta tensors; only metadata propagation.

## Semantics
- `device="meta"` tensors hold **shape/stride/dtype** but no data buffer.
- `tensor(data, device="meta")` uses `data` only for shape/dtype inference, then discards it.
- `zeros/ones(..., device="meta")` produce meta tensors (no actual fill).
- `to("meta")` drops data and returns a meta tensor with identical metadata.
- `to("cpu"/"npu")` materializes using **empty/uninitialized** storage with the same shape/dtype.
- Any data access on a meta tensor (`numpy()`, `_numpy_view()`) raises a clear error.

## Storage & Tensor
- Introduce `MetaStorage` that carries `device`, `dtype`, `shape` and no data buffer.
- `Tensor` on meta device uses `MetaStorage` and a valid `shape/stride`.

## Dispatcher
- Add a `meta` backend; if any input is meta, dispatch uses meta kernels.
- Meta kernels only infer shape/stride/dtype and return a meta tensor.

## Error Handling
- Access to data on meta tensors raises `RuntimeError("meta tensor has no data")`.
- Conversion to real device is permitted and produces uninitialized storage.

## Notes
- Meta tensors are compatible with pipeline mode (pending operations stay metadata-only).
- Future work: optional `to_empty` API, memory estimation helpers, meta-aware modules.
