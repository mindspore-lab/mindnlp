# take/take_along_dim/index_select Design

## Goal
Add `take`, `take_along_dim`, and `index_select` to mindtorch v2 with PyTorch-aligned semantics for CPU and meta backends.

## Behavior
- `take(input, index)`:
  - Flattens `input` and gathers using `index`.
  - Output shape matches `index` shape.
  - Negative indices allowed; out-of-range raises.
- `take_along_dim(input, indices, dim)`:
  - Gathers along `dim` using `indices`.
  - Output shape equals `indices` shape.
  - Negative indices allowed; out-of-range raises.
  - `indices` must align with `input` except along `dim`.
- `index_select(input, dim, index)`:
  - `index` must be 1D.
  - Output shape matches `input` with `dim` replaced by `index` length.
  - Negative indices allowed; out-of-range raises.

## CPU Implementation
- `take`: `np.take` on flattened array.
- `take_along_dim`: `np.take_along_axis` with normalized `dim`.
- `index_select`: `np.take` with `axis=dim`.
- Outputs wrap via `_from_numpy` preserving dtype/device.

## Meta Implementation
- `take`: output shape = `index.shape`.
- `take_along_dim`: output shape = `indices.shape`, checks shape compatibility.
- `index_select`: output shape = `input.shape` with `dim` length replaced by `index.shape[0]`, `index` must be 1D.

## API & Testing
Expose in `_functional.py` and `__init__.py`. Add CPU tests for basic gather, negative indices, and out-of-range errors; add meta tests for shape propagation and error paths. Update `docs/plans/ops-coverage.md`.
