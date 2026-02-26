# Chunk/Split/Unbind Design

## Goal
Add `chunk`, `split`, and `unbind` to mindtorch v2 CPU + meta backends with PyTorch‑aligned behavior, including uneven splits and list‑based sections for `split`.

## Behavior
- `chunk(input, chunks, dim=0)` returns up to `min(chunks, dim_size)` tensors. When `dim_size` is not divisible by `chunks`, trailing chunks are smaller (PyTorch behavior).
- `split(input, split_size_or_sections, dim=0)` supports an int (uniform chunk size with last remainder) or a list of sizes (sum must match `dim_size`, else error).
- `unbind(input, dim=0)` returns a tuple of slices along `dim` and removes that dimension.

## CPU Implementation
Use numpy slicing to materialize each output. For `chunk` and `split`, compute slice ranges along `dim`, then build outputs via `_from_numpy` for each slice. For `unbind`, slice each index along `dim` and squeeze that dimension. Outputs are returned as tuples to leverage existing multi‑output dispatch.

## Meta Implementation
Return tuple(s) of `TensorSpec` with correct shapes. `chunk`/`split` shapes are derived from the computed segment sizes; `unbind` removes the chosen dimension. Dtype stays the same as input.

## Functional API
Expose `chunk`, `split`, `unbind` in `_functional.py` and `__init__.py`. No `out` support (PyTorch doesn’t provide `out` for these APIs).

## Testing
CPU tests cover uneven chunking, list‑based split, and unbind shape/count. Meta tests validate output tuple lengths and shapes.
