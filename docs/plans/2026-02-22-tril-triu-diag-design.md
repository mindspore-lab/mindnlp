# Tril/Triu/Diag Design

## Goal
Add `tril`, `triu`, and `diag` to mindtorch v2 with PyTorch‑aligned semantics, including `diagonal` offsets, for CPU and meta backends.

## Behavior
- `tril(input, diagonal=0)` returns the lower triangle of a 2D+ tensor with elements above the `diagonal` set to zero.
- `triu(input, diagonal=0)` returns the upper triangle with elements below the `diagonal` set to zero.
- `diag(input, diagonal=0)` supports:
  - 1D input: returns a square matrix of size `(n+abs(k), n+abs(k))` with input values on the `k`‑th diagonal.
  - 2D input: returns a 1D tensor containing the `k`‑th diagonal.
  - Other ranks: raise a ValueError.

## CPU Implementation
Use numpy:
- `np.tril` / `np.triu` for `tril` / `triu`.
- `np.diag` for `diag` (1D→2D, 2D→1D).
Outputs use `_from_numpy` to preserve dtype and device.

## Meta Implementation
- `tril` / `triu` keep the same shape and dtype.
- `diag` computes shape per 1D/2D rules and returns a meta tensor with input dtype.

## API & Testing
Expose `tril`, `triu`, `diag` in `_functional.py` and `__init__.py`. Add CPU tests for `diagonal=0/±1` with 1D and 2D, and meta tests for shape propagation. Update `docs/plans/ops-coverage.md`.
