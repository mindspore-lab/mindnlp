# MindTorch v2 CPU Op Numpy Flow

## Goal
Provide a repeatable, minimal workflow for adding CPU ops backed by numpy, including meta inference and tests.

## Scope
- CPU backend (`src/mindtorch_v2/_backends/cpu/*`)
- Meta backend (`src/mindtorch_v2/_backends/meta/*`)
- MindTorch v2 tests (`tests/mindtorch_v2/*`)

## 1. Add CPU implementation (numpy)

**Where:** `src/mindtorch_v2/_backends/cpu/ops.py`

**Pattern:**
- Convert tensors to numpy with `_to_numpy`.
- Compute with numpy.
- Wrap numpy output back into `Tensor` via `_from_numpy`.
- For in-place ops, write into the existing numpy view and return the input tensor.

**Example (binary):**
```python
def add(a, b):
    return _from_numpy(_to_numpy(a) + _to_numpy(b), a.dtype, a.device)
```

**Example (in-place):**
```python
def add_(a, b):
    arr = _to_numpy(a)
    arr += _to_numpy(b)
    return a
```

**Example (reduce):**
```python
def sum_(a, dim=None, keepdim=False):
    return _from_numpy(_to_numpy(a).sum(axis=dim, keepdims=keepdim), a.dtype, a.device)
```

## 2. Register the op (CPU + meta)

**Where:** `src/mindtorch_v2/_backends/cpu/__init__.py`

**Pattern:**
- Register the CPU kernel.
- Provide a meta inference function.

```python
registry.register("add", "cpu", add, meta=meta_infer.infer_binary)
```

## 3. Add meta inference

**Where:** `src/mindtorch_v2/_backends/meta/ops.py`

**Pattern:**
- For elementwise: broadcast shapes and return meta tensor.
- For reduce: compute output shape based on `dim`/`keepdim`.
- For view/transpose: update shape only.

**Example (binary):**
```python
def _meta_binary_meta(a, b):
    shape = _broadcast_shape(a.shape, b.shape)
    return _meta_tensor(shape, a.dtype, a.device)
```

**Example (reduce):**
```python
def _meta_sum_meta(a, dim=None, keepdim=False):
    # shape logic ...
    return _meta_tensor(tuple(shape), a.dtype, a.device)
```

## 4. Tests

**Where:** `tests/mindtorch_v2/*`

**Minimum coverage per op:**
- CPU correctness test
- Meta shape test (output shape and dtype)

**Suggested pattern:**
- Use `tests/mindtorch_v2/test_ops_npu.py` as a reference for API usage but write CPU-only tests.
- Keep tests small and deterministic.

## 5. Update tracking

Append op entry to `docs/plans/ops-coverage.md` with owner, status, and PR link once a batch is done.
