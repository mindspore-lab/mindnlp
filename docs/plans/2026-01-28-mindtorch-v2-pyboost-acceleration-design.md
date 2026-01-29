# mindtorch_v2 PyBoost Acceleration Design

**Date:** 2026-01-28
**Status:** Approved
**Goal:** Replace NumPy-based CPU backend with MindSpore pyboost kernels for significant performance improvement

## Problem Statement

The current mindtorch_v2 implementation uses pure NumPy for all CPU operations:
- Every operation converts `Tensor → numpy → compute → wrap back`
- This introduces Python interpreter overhead
- Misses optimized C++ kernels available in MindSpore

Test execution is too slow, with the Albert model tests timing out.

## Design Decisions

| Decision | Choice |
|----------|--------|
| Internal Storage | `mindspore.Tensor` (not numpy array) |
| Migration Approach | Full rewrite |
| Autograd | Custom Python autograd engine |
| Tensor Design | Composition wrapper (Tensor → Storage → mindspore.Tensor) |

## Architecture

### Layer Structure

```
Tensor
  ├── _storage: TypedStorage    (contiguous 1D buffer)
  │     └── _ms_tensor: mindspore.Tensor  (actual data)
  ├── _shape: tuple             (view shape)
  ├── _stride: tuple            (stride info for views)
  ├── _offset: int              (storage offset)
  ├── requires_grad: bool
  ├── grad: Tensor | None
  └── grad_fn: Node | None
```

### Key Semantics

1. **Storage sharing**: Multiple `Tensor` objects can share the same `Storage` (view semantics)
2. **No numpy in compute path**: Operations use pyboost primitives directly on `mindspore.Tensor`
3. **In-place ops**: Use MindSpore's `Inplace*` pyboost ops (no numpy bridge possible)
4. **`.numpy()` is user-facing only**: Only converts to numpy when explicitly requested

### Backend Ops Design

```python
from mindspore.ops.auto_generate.gen_ops_prim import Add, Mul, MatMulExt
from mindspore.ops.auto_generate.pyboost_inner_prim import InplaceAddExt

# Instantiate ops once at module load
add_op = Add().set_device('CPU')
inplace_add_op = InplaceAddExt().set_device('CPU')

def _get_ms_data(tensor):
    """Extract mindspore.Tensor from our Tensor, reshaped to correct shape."""
    ms_tensor = tensor._storage._ms_tensor
    if ms_tensor.shape != tensor._shape:
        ms_tensor = ms_tensor.reshape(tensor._shape)
    return ms_tensor

def _wrap_result(ms_tensor):
    """Wrap mindspore.Tensor result into our Tensor."""
    storage = TypedStorage(ms_tensor.reshape(-1))
    return Tensor._from_storage(storage, ms_tensor.shape)

# Regular op (creates new tensor)
@register_op("add", DispatchKey.Backend_CPU)
def add_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = add_op(ms_a, ms_b)
    return _wrap_result(result)

# In-place op (modifies storage directly)
@register_op("add_", DispatchKey.Backend_CPU)
def add_inplace_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    inplace_add_op(ms_a, ms_b)
    return a
```

### Autograd Design

Python-based computation graph with pyboost gradient ops:

```python
class AddBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad_output):
        grad_a = grad_output if self.a.requires_grad else None
        grad_b = grad_output if self.b.requires_grad else None
        return grad_a, grad_b

def add_forward(a, b):
    result = add_op(_get_ms_data(a), _get_ms_data(b))
    out = _wrap_result(result)
    if a.requires_grad or b.requires_grad:
        out.grad_fn = AddBackward(a, b)
        out.requires_grad = True
    return out
```

For complex ops (matmul, conv, softmax), use dedicated pyboost grad primitives when available.

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Update `TypedStorage` to be primary data holder
- [ ] Rewrite `Tensor` class with storage-based design
- [ ] Add stride/offset support for view semantics
- [ ] Implement dtype mapping (PyTorch ↔ MindSpore)

### Phase 2: Backend Ops (High Priority)
Replace numpy with pyboost for most-used ops:

**Math ops:**
- [ ] `add`, `sub`, `mul`, `div`, `neg`, `abs`
- [ ] `pow`, `exp`, `log`, `sqrt`, `rsqrt`
- [ ] `matmul`, `bmm`, `mm`

**Activations:**
- [ ] `relu`, `gelu`, `silu`, `sigmoid`, `tanh`
- [ ] `softmax`, `log_softmax`

**Reductions:**
- [ ] `sum`, `mean`, `max`, `min`, `prod`
- [ ] `var`, `std`

**Tensor manipulation:**
- [ ] `cat`, `stack`, `split`, `chunk`
- [ ] `reshape`, `view`, `transpose`, `permute`
- [ ] `clone`, `contiguous`

**In-place variants:**
- [ ] `add_`, `sub_`, `mul_`, `div_`
- [ ] `copy_`, `fill_`, `zero_`

### Phase 3: Autograd Integration
- [ ] Implement `Function` base class
- [ ] Add backward functions for Phase 2 ops
- [ ] Implement `backward()` engine
- [ ] Test with MLP, attention blocks

### Phase 4: NN Modules & Validation
- [ ] Update `nn.Module` layers
- [ ] Test with BERT, Albert models
- [ ] Performance benchmarking

## Available PyBoost Ops Reference

From `mindspore.ops.auto_generate.gen_ops_prim`:
- Math: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Abs`, `Pow`, `Exp`, `Log`, `Sqrt`, `Rsqrt`
- Activations: `ReLU`, `GeLU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`
- Reductions: `SumExt`, `MeanExt`, `MaxDim`, `MinDim`, `ProdExt`
- Manipulation: `Reshape`, `Transpose`, `Concat`, `SplitTensor`
- Matmul: `MatMulExt`, `BatchMatMulExt`, `Mm`

From `mindspore.ops.auto_generate.pyboost_inner_prim`:
- In-place: `InplaceAddExt`, `InplaceMul`, `InplaceCopy`, `InplaceFillScalar`, `InplaceZero`

## Success Criteria

1. Albert model tests complete without timeout
2. All existing tests pass with new backend
3. Performance improvement of 5-10x on CPU operations
4. No numpy conversions in the compute hot path
