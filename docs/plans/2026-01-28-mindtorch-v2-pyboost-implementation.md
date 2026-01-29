# mindtorch_v2 PyBoost Acceleration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace NumPy-based CPU backend with MindSpore pyboost kernels for 5-10x performance improvement.

**Architecture:** Tensor holds Storage, Storage holds mindspore.Tensor. All ops use pyboost primitives directly - no numpy in compute path. Custom Python autograd tracks gradients.

**Tech Stack:** MindSpore pyboost ops, Python autograd, existing dispatch system.

---

## Prerequisites

Before starting, ensure MindSpore is available:

```bash
source ~/miniconda3/bin/activate mindnlp
python -c "import mindspore; print(mindspore.__version__)"
```

---

## Task 1: Update TypedStorage to Use MindSpore Tensor Properly

**Files:**
- Modify: `src/mindtorch_v2/_storage.py`
- Test: `tests/mindtorch_v2/test_storage.py`

**Step 1: Read current storage implementation**

Review `src/mindtorch_v2/_storage.py` to understand existing structure.

**Step 2: Write failing test for MindSpore tensor extraction**

Add to `tests/mindtorch_v2/test_storage.py`:

```python
def test_storage_ms_tensor_property():
    """Storage should expose underlying MindSpore tensor."""
    import mindspore
    storage = TypedStorage(10, dtype=dtype_mod.float32)
    ms_tensor = storage.ms_tensor
    assert isinstance(ms_tensor, mindspore.Tensor)
    assert ms_tensor.shape == (10,)
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_storage_ms_tensor_property -v`
Expected: FAIL with "AttributeError: ms_tensor"

**Step 4: Add ms_tensor property to TypedStorage**

In `src/mindtorch_v2/_storage.py`, add property to `TypedStorage` class:

```python
@property
def ms_tensor(self):
    """Return the underlying MindSpore tensor."""
    return self._ms_tensor
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_storage.py::test_storage_ms_tensor_property -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_storage.py tests/mindtorch_v2/test_storage.py
git commit -m "feat(storage): add ms_tensor property for pyboost access"
```

---

## Task 2: Create PyBoost Backend Module

**Files:**
- Create: `src/mindtorch_v2/_backends/pyboost_cpu.py`
- Test: `tests/mindtorch_v2/test_pyboost_backend.py`

**Step 1: Create test file for pyboost backend**

Create `tests/mindtorch_v2/test_pyboost_backend.py`:

```python
"""Tests for PyBoost CPU backend."""
import pytest


def test_pyboost_add_op_exists():
    """PyBoost add op should be importable."""
    from mindtorch_v2._backends.pyboost_cpu import add_op
    assert add_op is not None


def test_pyboost_add_basic():
    """PyBoost add should work on MindSpore tensors."""
    import mindspore
    from mindtorch_v2._backends.pyboost_cpu import add_op

    a = mindspore.Tensor([1.0, 2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0, 6.0])
    result = add_op(a, b)

    expected = [5.0, 7.0, 9.0]
    assert list(result.asnumpy()) == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pyboost_backend.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Create pyboost_cpu.py with basic ops**

Create `src/mindtorch_v2/_backends/pyboost_cpu.py`:

```python
"""PyBoost CPU backend using MindSpore primitives.

All ops are instantiated once at module load with CPU device set.
These are much faster than NumPy as they use optimized C++ kernels.
"""

from mindspore.ops.auto_generate.gen_ops_prim import (
    Add, Sub, Mul, Div, Neg, Abs,
    Pow, Exp, Log, Sqrt, Rsqrt,
    Sin, Cos, Tanh, Sigmoid,
    ReLU, GeLU, SiLU,
    MatMulExt, BatchMatMulExt,
    SumExt, MeanExt, MaxDim, MinDim,
    Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual,
    Reshape, Transpose,
    Clone, Contiguous,
)
from mindspore.ops.auto_generate.pyboost_inner_prim import (
    InplaceAddExt, InplaceSubExt, InplaceMul, InplaceDiv,
    InplaceCopy, InplaceFillScalar, InplaceZero,
)

# Instantiate ops with CPU device
add_op = Add().set_device('CPU')
sub_op = Sub().set_device('CPU')
mul_op = Mul().set_device('CPU')
div_op = Div().set_device('CPU')
neg_op = Neg().set_device('CPU')
abs_op = Abs().set_device('CPU')

pow_op = Pow().set_device('CPU')
exp_op = Exp().set_device('CPU')
log_op = Log().set_device('CPU')
sqrt_op = Sqrt().set_device('CPU')
rsqrt_op = Rsqrt().set_device('CPU')

sin_op = Sin().set_device('CPU')
cos_op = Cos().set_device('CPU')
tanh_op = Tanh().set_device('CPU')
sigmoid_op = Sigmoid().set_device('CPU')

relu_op = ReLU().set_device('CPU')
gelu_op = GeLU().set_device('CPU')
silu_op = SiLU().set_device('CPU')

matmul_op = MatMulExt().set_device('CPU')
bmm_op = BatchMatMulExt().set_device('CPU')

sum_op = SumExt().set_device('CPU')
mean_op = MeanExt().set_device('CPU')
max_op = MaxDim().set_device('CPU')
min_op = MinDim().set_device('CPU')

equal_op = Equal().set_device('CPU')
not_equal_op = NotEqual().set_device('CPU')
greater_op = Greater().set_device('CPU')
less_op = Less().set_device('CPU')
greater_equal_op = GreaterEqual().set_device('CPU')
less_equal_op = LessEqual().set_device('CPU')

reshape_op = Reshape().set_device('CPU')
transpose_op = Transpose().set_device('CPU')

clone_op = Clone().set_device('CPU')
contiguous_op = Contiguous().set_device('CPU')

# In-place ops
inplace_add_op = InplaceAddExt().set_device('CPU')
inplace_sub_op = InplaceSubExt().set_device('CPU')
inplace_mul_op = InplaceMul().set_device('CPU')
inplace_div_op = InplaceDiv().set_device('CPU')
inplace_copy_op = InplaceCopy().set_device('CPU')
inplace_fill_op = InplaceFillScalar().set_device('CPU')
inplace_zero_op = InplaceZero().set_device('CPU')
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_pyboost_backend.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/pyboost_cpu.py tests/mindtorch_v2/test_pyboost_backend.py
git commit -m "feat(backend): add pyboost CPU backend with MindSpore primitives"
```

---

## Task 3: Add Helper Functions for Tensor/MindSpore Conversion

**Files:**
- Modify: `src/mindtorch_v2/_backends/pyboost_cpu.py`
- Test: `tests/mindtorch_v2/test_pyboost_backend.py`

**Step 1: Write failing test for helper functions**

Add to `tests/mindtorch_v2/test_pyboost_backend.py`:

```python
def test_get_ms_data_extracts_mindspore_tensor():
    """_get_ms_data should extract MindSpore tensor from our Tensor."""
    from mindtorch_v2 import Tensor
    from mindtorch_v2._backends.pyboost_cpu import _get_ms_data
    import mindspore

    t = Tensor([1.0, 2.0, 3.0])
    ms_t = _get_ms_data(t)

    assert isinstance(ms_t, mindspore.Tensor)
    assert list(ms_t.asnumpy()) == [1.0, 2.0, 3.0]


def test_wrap_result_creates_tensor():
    """_wrap_result should wrap MindSpore tensor in our Tensor."""
    import mindspore
    from mindtorch_v2 import Tensor
    from mindtorch_v2._backends.pyboost_cpu import _wrap_result

    ms_t = mindspore.Tensor([1.0, 2.0, 3.0])
    t = _wrap_result(ms_t)

    assert isinstance(t, Tensor)
    assert list(t.numpy()) == [1.0, 2.0, 3.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pyboost_backend.py::test_get_ms_data_extracts_mindspore_tensor -v`
Expected: FAIL with "ImportError: cannot import name '_get_ms_data'"

**Step 3: Add helper functions**

Add to `src/mindtorch_v2/_backends/pyboost_cpu.py`:

```python
def _get_ms_data(tensor):
    """Extract MindSpore tensor from our Tensor, reshaped to correct shape.

    Args:
        tensor: mindtorch_v2.Tensor or scalar

    Returns:
        mindspore.Tensor ready for pyboost ops
    """
    from .._tensor import Tensor
    import mindspore

    if isinstance(tensor, Tensor):
        ms_tensor = tensor._storage.ms_tensor
        # Handle view semantics - reshape if needed
        if ms_tensor.shape != tensor.shape:
            # For views with non-contiguous strides, need careful handling
            # For now, assume contiguous
            ms_tensor = ms_tensor.reshape(tensor.shape)
        return ms_tensor
    elif isinstance(tensor, (int, float, bool)):
        return mindspore.Tensor(tensor)
    elif isinstance(tensor, mindspore.Tensor):
        return tensor
    else:
        raise TypeError(f"Cannot convert {type(tensor)} to MindSpore tensor")


def _wrap_result(ms_tensor, device="cpu"):
    """Wrap MindSpore tensor result in our Tensor.

    Args:
        ms_tensor: mindspore.Tensor result from pyboost op
        device: device string

    Returns:
        mindtorch_v2.Tensor
    """
    from .._tensor import Tensor
    from .._storage import TypedStorage
    from .. import _dtype as dtype_mod

    # Create storage from flattened tensor
    flat = ms_tensor.reshape(-1)
    storage = TypedStorage.__new__(TypedStorage)
    storage._ms_tensor = flat
    storage._size = flat.shape[0]
    storage._dtype = dtype_mod.from_mindspore_dtype(flat.dtype)
    from .._device import device as device_cls
    storage._device = device_cls(device)

    # Create tensor with proper shape
    return Tensor(
        _storage=storage,
        _shape=tuple(ms_tensor.shape),
        _stride=None,  # Will compute contiguous strides
        _storage_offset=0
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/mindtorch_v2/test_pyboost_backend.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/pyboost_cpu.py tests/mindtorch_v2/test_pyboost_backend.py
git commit -m "feat(backend): add tensor conversion helpers for pyboost"
```

---

## Task 4: Replace NumPy Add with PyBoost Add

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_math_ops.py`

**Step 1: Verify existing add test works**

Run: `pytest tests/mindtorch_v2/test_math_ops.py::test_add -v`
Expected: PASS (with numpy backend)

**Step 2: Replace add implementation**

In `src/mindtorch_v2/_backends/cpu.py`, change the add_cpu function:

```python
from .pyboost_cpu import add_op, _get_ms_data, _wrap_result

@register_op("add", DispatchKey.Backend_CPU)
def add_cpu(a, b):
    """Element-wise addition using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = add_op(ms_a, ms_b)
    return _wrap_result(result)
```

**Step 3: Run test to verify it still passes**

Run: `pytest tests/mindtorch_v2/test_math_ops.py::test_add -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy add with pyboost add"
```

---

## Task 5: Replace Remaining Binary Math Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_math_ops.py`

**Step 1: Replace sub, mul, div, neg, abs**

Update each function in `src/mindtorch_v2/_backends/cpu.py`:

```python
from .pyboost_cpu import (
    add_op, sub_op, mul_op, div_op, neg_op, abs_op,
    pow_op, exp_op, log_op, sqrt_op, rsqrt_op,
    _get_ms_data, _wrap_result
)

@register_op("sub", DispatchKey.Backend_CPU)
def sub_cpu(a, b):
    """Element-wise subtraction using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = sub_op(ms_a, ms_b)
    return _wrap_result(result)


@register_op("mul", DispatchKey.Backend_CPU)
def mul_cpu(a, b):
    """Element-wise multiplication using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = mul_op(ms_a, ms_b)
    return _wrap_result(result)


@register_op("div", DispatchKey.Backend_CPU)
def div_cpu(a, b):
    """Element-wise division using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = div_op(ms_a, ms_b)
    return _wrap_result(result)


@register_op("neg", DispatchKey.Backend_CPU)
def neg_cpu(a):
    """Element-wise negation using PyBoost."""
    ms_a = _get_ms_data(a)
    result = neg_op(ms_a)
    return _wrap_result(result)


@register_op("abs", DispatchKey.Backend_CPU)
def abs_cpu(a):
    """Element-wise absolute value using PyBoost."""
    ms_a = _get_ms_data(a)
    result = abs_op(ms_a)
    return _wrap_result(result)
```

**Step 2: Run math ops tests**

Run: `pytest tests/mindtorch_v2/test_math_ops.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy binary math ops with pyboost"
```

---

## Task 6: Replace Unary Math Ops (exp, log, sqrt, etc.)

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_math_ops.py`

**Step 1: Replace exp, log, sqrt, rsqrt, pow**

```python
@register_op("exp", DispatchKey.Backend_CPU)
def exp_cpu(a):
    """Element-wise exponential using PyBoost."""
    ms_a = _get_ms_data(a)
    result = exp_op(ms_a)
    return _wrap_result(result)


@register_op("log", DispatchKey.Backend_CPU)
def log_cpu(a):
    """Element-wise natural logarithm using PyBoost."""
    ms_a = _get_ms_data(a)
    result = log_op(ms_a)
    return _wrap_result(result)


@register_op("sqrt", DispatchKey.Backend_CPU)
def sqrt_cpu(a):
    """Element-wise square root using PyBoost."""
    ms_a = _get_ms_data(a)
    result = sqrt_op(ms_a)
    return _wrap_result(result)


@register_op("rsqrt", DispatchKey.Backend_CPU)
def rsqrt_cpu(a):
    """Reciprocal square root using PyBoost."""
    ms_a = _get_ms_data(a)
    result = rsqrt_op(ms_a)
    return _wrap_result(result)


@register_op("pow", DispatchKey.Backend_CPU)
def pow_cpu(a, exponent):
    """Element-wise power using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_exp = _get_ms_data(exponent)
    result = pow_op(ms_a, ms_exp)
    return _wrap_result(result)
```

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_math_ops.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy unary math ops with pyboost"
```

---

## Task 7: Replace Trigonometric and Activation Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_math_ops.py`, `tests/mindtorch_v2/test_nn_activation.py`

**Step 1: Replace sin, cos, tanh, sigmoid, relu, gelu, silu**

```python
from .pyboost_cpu import (
    sin_op, cos_op, tanh_op, sigmoid_op,
    relu_op, gelu_op, silu_op,
    _get_ms_data, _wrap_result
)

@register_op("sin", DispatchKey.Backend_CPU)
def sin_cpu(a):
    ms_a = _get_ms_data(a)
    return _wrap_result(sin_op(ms_a))


@register_op("cos", DispatchKey.Backend_CPU)
def cos_cpu(a):
    ms_a = _get_ms_data(a)
    return _wrap_result(cos_op(ms_a))


@register_op("tanh", DispatchKey.Backend_CPU)
def tanh_cpu(a):
    ms_a = _get_ms_data(a)
    return _wrap_result(tanh_op(ms_a))


@register_op("sigmoid", DispatchKey.Backend_CPU)
def sigmoid_cpu(a):
    ms_a = _get_ms_data(a)
    return _wrap_result(sigmoid_op(ms_a))


@register_op("relu", DispatchKey.Backend_CPU)
def relu_cpu(a):
    ms_a = _get_ms_data(a)
    return _wrap_result(relu_op(ms_a))


@register_op("gelu", DispatchKey.Backend_CPU)
def gelu_cpu(a, approximate='none'):
    ms_a = _get_ms_data(a)
    # Note: may need to handle approximate parameter
    return _wrap_result(gelu_op(ms_a))


@register_op("silu", DispatchKey.Backend_CPU)
def silu_cpu(a):
    ms_a = _get_ms_data(a)
    return _wrap_result(silu_op(ms_a))
```

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_math_ops.py tests/mindtorch_v2/test_nn_activation.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy trig and activation ops with pyboost"
```

---

## Task 8: Replace Matrix Multiplication Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_math_ops.py`

**Step 1: Replace matmul, bmm**

```python
from .pyboost_cpu import matmul_op, bmm_op, _get_ms_data, _wrap_result

@register_op("matmul", DispatchKey.Backend_CPU)
def matmul_cpu(a, b):
    """Matrix multiplication using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = matmul_op(ms_a, ms_b)
    return _wrap_result(result)


@register_op("bmm", DispatchKey.Backend_CPU)
def bmm_cpu(a, b):
    """Batched matrix multiplication using PyBoost."""
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    result = bmm_op(ms_a, ms_b)
    return _wrap_result(result)
```

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_math_ops.py -v -k "matmul or bmm"`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy matmul ops with pyboost"
```

---

## Task 9: Replace Reduction Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_reduction_ops.py`

**Step 1: Replace sum, mean, max, min**

```python
from .pyboost_cpu import sum_op, mean_op, max_op, min_op, _get_ms_data, _wrap_result

@register_op("sum", DispatchKey.Backend_CPU)
def sum_cpu(a, dim=None, keepdim=False):
    """Sum reduction using PyBoost."""
    ms_a = _get_ms_data(a)
    if dim is None:
        # Sum all elements
        result = sum_op(ms_a, tuple(range(ms_a.ndim)), keepdim)
    else:
        if isinstance(dim, int):
            dim = (dim,)
        result = sum_op(ms_a, dim, keepdim)
    return _wrap_result(result)


@register_op("mean", DispatchKey.Backend_CPU)
def mean_cpu(a, dim=None, keepdim=False):
    """Mean reduction using PyBoost."""
    ms_a = _get_ms_data(a)
    if dim is None:
        result = mean_op(ms_a, tuple(range(ms_a.ndim)), keepdim)
    else:
        if isinstance(dim, int):
            dim = (dim,)
        result = mean_op(ms_a, dim, keepdim)
    return _wrap_result(result)


@register_op("max", DispatchKey.Backend_CPU)
def max_cpu(a, dim=None, keepdim=False):
    """Max reduction using PyBoost."""
    ms_a = _get_ms_data(a)
    if dim is None:
        # Return single max value
        result = max_op(ms_a, 0, keepdim)
        for d in range(1, ms_a.ndim):
            result = max_op(result.values, 0, keepdim)
        return _wrap_result(result.values if hasattr(result, 'values') else result)
    else:
        result = max_op(ms_a, dim, keepdim)
        from collections import namedtuple
        MaxResult = namedtuple('MaxResult', ['values', 'indices'])
        return MaxResult(_wrap_result(result[0]), _wrap_result(result[1]))


@register_op("min", DispatchKey.Backend_CPU)
def min_cpu(a, dim=None, keepdim=False):
    """Min reduction using PyBoost."""
    ms_a = _get_ms_data(a)
    if dim is None:
        result = min_op(ms_a, 0, keepdim)
        for d in range(1, ms_a.ndim):
            result = min_op(result.values, 0, keepdim)
        return _wrap_result(result.values if hasattr(result, 'values') else result)
    else:
        result = min_op(ms_a, dim, keepdim)
        from collections import namedtuple
        MinResult = namedtuple('MinResult', ['values', 'indices'])
        return MinResult(_wrap_result(result[0]), _wrap_result(result[1]))
```

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_reduction_ops.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy reduction ops with pyboost"
```

---

## Task 10: Replace Comparison Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Test: `tests/mindtorch_v2/test_comparison_ops.py`

**Step 1: Replace eq, ne, gt, lt, ge, le**

```python
from .pyboost_cpu import (
    equal_op, not_equal_op, greater_op, less_op,
    greater_equal_op, less_equal_op,
    _get_ms_data, _wrap_result
)

@register_op("eq", DispatchKey.Backend_CPU)
def eq_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    return _wrap_result(equal_op(ms_a, ms_b))


@register_op("ne", DispatchKey.Backend_CPU)
def ne_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    return _wrap_result(not_equal_op(ms_a, ms_b))


@register_op("gt", DispatchKey.Backend_CPU)
def gt_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    return _wrap_result(greater_op(ms_a, ms_b))


@register_op("lt", DispatchKey.Backend_CPU)
def lt_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    return _wrap_result(less_op(ms_a, ms_b))


@register_op("ge", DispatchKey.Backend_CPU)
def ge_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    return _wrap_result(greater_equal_op(ms_a, ms_b))


@register_op("le", DispatchKey.Backend_CPU)
def le_cpu(a, b):
    ms_a = _get_ms_data(a)
    ms_b = _get_ms_data(b)
    return _wrap_result(less_equal_op(ms_a, ms_b))
```

**Step 2: Run tests**

Run: `pytest tests/mindtorch_v2/test_comparison_ops.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py
git commit -m "feat(backend): replace numpy comparison ops with pyboost"
```

---

## Task 11: Add In-Place Operations

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Create: `tests/mindtorch_v2/test_inplace_ops.py`

**Step 1: Create test file for in-place ops**

Create `tests/mindtorch_v2/test_inplace_ops.py`:

```python
"""Tests for in-place operations using PyBoost."""
import pytest
from mindtorch_v2 import Tensor


def test_add_inplace():
    """In-place add should modify tensor directly."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    original_storage = a._storage

    a.add_(b)

    # Same storage object
    assert a._storage is original_storage
    # Values updated
    assert list(a.numpy()) == [5.0, 7.0, 9.0]


def test_mul_inplace():
    """In-place mul should modify tensor directly."""
    a = Tensor([1.0, 2.0, 3.0])
    original_storage = a._storage

    a.mul_(2.0)

    assert a._storage is original_storage
    assert list(a.numpy()) == [2.0, 4.0, 6.0]


def test_zero_inplace():
    """In-place zero should set all elements to zero."""
    a = Tensor([1.0, 2.0, 3.0])

    a.zero_()

    assert list(a.numpy()) == [0.0, 0.0, 0.0]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/mindtorch_v2/test_inplace_ops.py -v`
Expected: FAIL (in-place ops not using pyboost yet)

**Step 3: Register in-place ops**

Add to `src/mindtorch_v2/_backends/cpu.py`:

```python
from .pyboost_cpu import (
    inplace_add_op, inplace_sub_op, inplace_mul_op, inplace_div_op,
    inplace_zero_op, inplace_fill_op,
    _get_ms_data
)

@register_op("add_", DispatchKey.Backend_CPU)
def add_inplace_cpu(a, b):
    """In-place add using PyBoost."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    inplace_add_op(ms_a, ms_b)
    return a


@register_op("sub_", DispatchKey.Backend_CPU)
def sub_inplace_cpu(a, b):
    """In-place sub using PyBoost."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    inplace_sub_op(ms_a, ms_b)
    return a


@register_op("mul_", DispatchKey.Backend_CPU)
def mul_inplace_cpu(a, b):
    """In-place mul using PyBoost."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    inplace_mul_op(ms_a, ms_b)
    return a


@register_op("div_", DispatchKey.Backend_CPU)
def div_inplace_cpu(a, b):
    """In-place div using PyBoost."""
    ms_a = a._storage.ms_tensor
    ms_b = _get_ms_data(b)
    inplace_div_op(ms_a, ms_b)
    return a


@register_op("zero_", DispatchKey.Backend_CPU)
def zero_inplace_cpu(a):
    """In-place zero using PyBoost."""
    ms_a = a._storage.ms_tensor
    inplace_zero_op(ms_a)
    return a


@register_op("fill_", DispatchKey.Backend_CPU)
def fill_inplace_cpu(a, value):
    """In-place fill using PyBoost."""
    ms_a = a._storage.ms_tensor
    inplace_fill_op(ms_a, value)
    return a
```

**Step 4: Run tests**

Run: `pytest tests/mindtorch_v2/test_inplace_ops.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py tests/mindtorch_v2/test_inplace_ops.py
git commit -m "feat(backend): add pyboost in-place operations"
```

---

## Task 12: Run Full Test Suite

**Files:**
- Test: `tests/mindtorch_v2/`

**Step 1: Run all mindtorch_v2 tests**

Run: `pytest tests/mindtorch_v2/ -v --ignore=tests/mindtorch_v2/models/`
Expected: All tests PASS

**Step 2: Fix any failing tests**

If any tests fail, investigate and fix. Common issues:
- PyBoost op signature differences
- Dtype handling differences
- Shape broadcasting differences

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix(backend): resolve pyboost compatibility issues"
```

---

## Task 13: Benchmark Performance

**Files:**
- Create: `tests/mindtorch_v2/benchmark_pyboost.py`

**Step 1: Create benchmark script**

Create `tests/mindtorch_v2/benchmark_pyboost.py`:

```python
"""Benchmark PyBoost vs NumPy backend performance."""
import time
from mindtorch_v2 import Tensor


def benchmark_matmul():
    """Benchmark matrix multiplication."""
    sizes = [(64, 64), (256, 256), (1024, 1024)]

    for size in sizes:
        a = Tensor([[1.0] * size[1]] * size[0])
        b = Tensor([[1.0] * size[1]] * size[0])

        # Warmup
        for _ in range(3):
            _ = a @ b

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        elapsed = time.perf_counter() - start

        print(f"matmul {size}: {elapsed/10*1000:.2f} ms per op")


def benchmark_elementwise():
    """Benchmark elementwise operations."""
    a = Tensor([[1.0] * 1024] * 1024)
    b = Tensor([[2.0] * 1024] * 1024)

    ops = [
        ("add", lambda: a + b),
        ("mul", lambda: a * b),
        ("exp", lambda: a.exp()),
        ("sigmoid", lambda: a.sigmoid()),
    ]

    for name, op in ops:
        # Warmup
        for _ in range(3):
            _ = op()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = op()
        elapsed = time.perf_counter() - start

        print(f"{name} (1024x1024): {elapsed/100*1000:.2f} ms per op")


if __name__ == "__main__":
    print("PyBoost Backend Benchmark")
    print("=" * 40)
    benchmark_matmul()
    print()
    benchmark_elementwise()
```

**Step 2: Run benchmark**

Run: `python tests/mindtorch_v2/benchmark_pyboost.py`
Expected: Observe significant speedup over numpy baseline

**Step 3: Commit**

```bash
git add tests/mindtorch_v2/benchmark_pyboost.py
git commit -m "test(benchmark): add pyboost performance benchmark"
```

---

## Task 14: Test with Albert Model

**Files:**
- Test: `tests/transformers/tests/models/albert/test_modeling_albert.py`

**Step 1: Run Albert tests**

Run: `python tests/run_test_v2.py tests/transformers/tests/models/albert/test_modeling_albert.py -v --timeout=600`

**Step 2: Observe performance improvement**

Expected: Tests complete faster than before (5-10x improvement on CPU ops)

**Step 3: Document results**

Add results to CLAUDE.md session log section.

---

## Summary

After completing all tasks, the mindtorch_v2 backend will:

1. Use MindSpore pyboost kernels for all CPU operations
2. Never use numpy in the compute path (only for `.numpy()` API)
3. Support proper in-place operations via pyboost inplace ops
4. Maintain full backward compatibility with existing tests
5. Provide 5-10x performance improvement on CPU workloads
