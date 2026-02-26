# Gather/Scatter Design

## Goal
Add `gather` and `scatter` to mindtorch v2 with PyTorch-aligned semantics for CPU and meta backends.

## Behavior
- `gather(input, dim, index)`:
  - `index` must match `input` shape except at `dim`.
  - Indices must be in-range `[0, size-1]`; negative indices raise.
  - Output shape equals `index` shape, dtype equals input.
- `scatter(input, dim, index, src)`:
  - Supports tensor `src` or scalar `src`.
  - `index` must match `input` shape except at `dim`.
  - Indices must be in-range; negative indices raise.
  - Output shape equals input shape, dtype equals input.

## CPU Implementation
- `gather`: `np.take_along_axis` after validating shapes/ranges.
- `scatter`: copy input array, broadcast `src` to `index` shape, then assign along `dim`.
- Outputs wrapped with `_from_numpy` to preserve dtype/device.

## Meta Implementation
- `gather`: output shape = `index.shape`.
- `scatter`: output shape = `input.shape`.
- Validate `dim` and `index` shape in meta, raise `ValueError` on mismatch.

## API & Testing
Expose in `_functional.py` and `__init__.py`. Add CPU tests for basic cases, scalar `src`, and error paths; add meta tests for shape propagation and error handling. Update `docs/plans/ops-coverage.md`.
