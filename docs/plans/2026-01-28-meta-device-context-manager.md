# Meta Device Context Manager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement meta device support so `torch.device("meta")` context manager works correctly and transformers can detect it.

**Architecture:** Meta device creates tensors without allocating storage (shape/dtype only). The device context manager sets a thread-local default device that affects tensor creation. Transformers detects meta device by creating an empty tensor and checking its device.

**Tech Stack:** Python threading.local, mindtorch_v2 device system, tensor creation hooks

---

## Prerequisites

Understand the detection mechanism:
```python
# Transformers uses this to detect meta device context:
device_in_context = torch.tensor([]).device
# If in `with torch.device("meta"):`, this returns device("meta")
```

---

## Task 1: Add Meta Device Support to Tensor Creation

**Files:**
- Modify: `src/mindtorch_v2/_creation.py`
- Test: `tests/mindtorch_v2/test_meta_device.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_meta_device.py`:

```python
"""Test meta device functionality."""
import pytest


def test_tensor_creation_respects_device_context():
    """Tensor created in device context should have that device."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        t = tensor([1.0, 2.0, 3.0])
        assert t.device.type == "meta"


def test_tensor_creation_outside_context_uses_cpu():
    """Tensor created outside context should use CPU."""
    from mindtorch_v2 import tensor

    t = tensor([1.0, 2.0, 3.0])
    assert t.device.type == "cpu"


def test_empty_tensor_in_meta_context():
    """Empty tensor in meta context should have meta device."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        t = tensor([])
        assert t.device.type == "meta"


def test_meta_tensor_has_no_storage():
    """Meta tensors should not allocate actual storage."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        t = tensor([1.0, 2.0, 3.0])
        # Meta tensors have shape but no actual data
        assert t.shape == (3,)
        # Accessing data should raise or return None
        with pytest.raises((RuntimeError, AttributeError)):
            _ = t.numpy()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_meta_device.py -v`
Expected: FAIL with "AssertionError: assert 'cpu' == 'meta'"

**Step 3: Modify tensor creation to respect device context**

Update `src/mindtorch_v2/_creation.py`:

```python
def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    """Create a tensor from data.

    Args:
        data: Data to create tensor from (list, numpy array, etc.)
        dtype: Desired data type
        device: Device to place tensor on (if None, uses context device or CPU)
        requires_grad: Whether to track gradients
        pin_memory: Whether to use pinned memory (ignored)

    Returns:
        Tensor
    """
    from ._tensor import Tensor
    from ._device import device as device_cls, _get_default_device
    from ._storage import TypedStorage
    from . import _dtype as dtype_mod
    import numpy as np

    # Determine target device
    if device is None:
        # Check if we're in a device context manager
        context_device = _get_default_device()
        if context_device is not None:
            device = context_device
        else:
            device = device_cls("cpu")
    elif not isinstance(device, device_cls):
        device = device_cls(device)

    # Handle meta device - create tensor with shape but no storage
    if device.type == "meta":
        # Convert data to numpy to get shape and dtype
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, (list, tuple)):
            arr = np.array(data)
        elif isinstance(data, (int, float, bool)):
            arr = np.array(data)
        else:
            arr = np.asarray(data)

        # Determine dtype
        if dtype is None:
            dtype = dtype_mod.from_numpy_dtype(arr.dtype)
        elif isinstance(dtype, str):
            dtype = getattr(dtype_mod, dtype)

        # Create meta tensor with shape but no actual storage
        shape = arr.shape

        # Create a minimal storage (empty) for meta device
        storage = TypedStorage.__new__(TypedStorage)
        storage._ms_tensor = None  # No actual MindSpore tensor
        storage._size = 0  # No actual data
        storage._dtype = dtype
        storage._device = device

        result = Tensor(
            _storage=storage,
            _shape=shape,
            _stride=None,
            _storage_offset=0
        )
        result._device = device
        result._requires_grad = requires_grad
        return result

    # Normal device handling (existing code continues...)
    # [Rest of existing tensor() function code]
```

**Step 4: Update Tensor class to handle meta device**

Modify `src/mindtorch_v2/_tensor.py` to add meta device checks:

```python
# In Tensor class, add property:
@property
def device(self):
    """Get the device of this tensor."""
    if hasattr(self, '_device') and self._device is not None:
        return self._device
    # Fall back to storage device
    if self._storage is not None:
        return self._storage._device
    from ._device import device as device_cls
    return device_cls("cpu")

# Update numpy() method to check for meta:
def numpy(self):
    """Convert tensor to numpy array."""
    if self.device.type == "meta":
        raise RuntimeError(
            "Cannot convert meta tensor to numpy. "
            "Meta tensors have no data, only shape/dtype information."
        )
    # [existing numpy() code]
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_meta_device.py -v`
Expected: PASS (all 4 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_creation.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_meta_device.py
git commit -m "feat(device): add meta device support for tensor creation

- Tensor creation respects device context manager
- Meta tensors have shape/dtype but no storage
- Meta tensors raise error on .numpy() access

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Register Meta Device Operations

**Files:**
- Create: `src/mindtorch_v2/_backends/meta.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Test: `tests/mindtorch_v2/test_meta_ops.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_meta_ops.py`:

```python
"""Test meta device operations."""
import pytest


def test_meta_tensor_add():
    """Adding meta tensors should return meta tensor with correct shape."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        c = a + b

        assert c.device.type == "meta"
        assert c.shape == (3,)


def test_meta_tensor_matmul():
    """Matmul on meta tensors should return correct shape."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = tensor([[1.0], [1.0]])  # (2, 1)
        c = a @ b

        assert c.device.type == "meta"
        assert c.shape == (2, 1)


def test_meta_tensor_reshape():
    """Reshape on meta tensor should work."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        a = tensor([1.0, 2.0, 3.0, 4.0])
        b = a.reshape(2, 2)

        assert b.device.type == "meta"
        assert b.shape == (2, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_meta_ops.py -v`
Expected: FAIL with errors about operations not supporting meta device

**Step 3: Create meta backend**

Create `src/mindtorch_v2/_backends/meta.py`:

```python
"""Meta device backend - operations that only compute shapes, no actual computation.

Meta tensors are used for:
1. Model initialization without allocating memory
2. Shape inference
3. Detecting device context in transformers
"""

from typing import Tuple, Optional, Any
import numpy as np


def _infer_output_shape_binary(a_shape: Tuple[int, ...], b_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Infer output shape for binary operations with broadcasting."""
    # Broadcast rules: align from right, take max of each dimension
    max_ndim = max(len(a_shape), len(b_shape))
    a_shape_padded = (1,) * (max_ndim - len(a_shape)) + a_shape
    b_shape_padded = (1,) * (max_ndim - len(b_shape)) + b_shape

    output_shape = []
    for a_dim, b_dim in zip(a_shape_padded, b_shape_padded):
        if a_dim == 1:
            output_shape.append(b_dim)
        elif b_dim == 1:
            output_shape.append(a_dim)
        elif a_dim == b_dim:
            output_shape.append(a_dim)
        else:
            raise ValueError(f"Cannot broadcast shapes {a_shape} and {b_shape}")

    return tuple(output_shape)


def _create_meta_tensor(shape: Tuple[int, ...], dtype, device):
    """Create a meta tensor with given shape."""
    from .._tensor import Tensor
    from .._storage import TypedStorage

    storage = TypedStorage.__new__(TypedStorage)
    storage._ms_tensor = None
    storage._size = 0
    storage._dtype = dtype
    storage._device = device

    result = Tensor(
        _storage=storage,
        _shape=shape,
        _stride=None,
        _storage_offset=0
    )
    result._device = device
    return result


# Binary operations
def add_meta(a, b):
    """Meta add - compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a._dtype, a.device)


def sub_meta(a, b):
    """Meta subtract - compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a._dtype, a.device)


def mul_meta(a, b):
    """Meta multiply - compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a._dtype, a.device)


def div_meta(a, b):
    """Meta divide - compute output shape only."""
    output_shape = _infer_output_shape_binary(a.shape, b.shape)
    return _create_meta_tensor(output_shape, a._dtype, a.device)


# Matrix operations
def matmul_meta(a, b):
    """Meta matmul - compute output shape only."""
    # Matmul rules: (..., n, m) @ (..., m, p) -> (..., n, p)
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError("matmul requires at least 2D tensors")

    # Check inner dimensions match
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"matmul dimension mismatch: {a.shape} @ {b.shape}")

    # Compute output shape
    batch_shape = _infer_output_shape_binary(a.shape[:-2], b.shape[:-2])
    output_shape = batch_shape + (a.shape[-2], b.shape[-1])

    return _create_meta_tensor(output_shape, a._dtype, a.device)


# Shape operations
def reshape_meta(a, shape):
    """Meta reshape - just change shape."""
    return _create_meta_tensor(shape, a._dtype, a.device)


def transpose_meta(a, dim0, dim1):
    """Meta transpose - swap dimensions."""
    new_shape = list(a.shape)
    new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
    return _create_meta_tensor(tuple(new_shape), a._dtype, a.device)
```

**Step 4: Update dispatcher to route meta device ops**

Modify `src/mindtorch_v2/_dispatch/dispatcher.py`:

```python
def dispatch(op_name: str, *args, **kwargs) -> Any:
    """Dispatch an operation to the correct implementation.

    Priority:
    1. Meta device (if any arg is meta)
    2. New standardized ops
    3. Legacy dispatch
    """
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled
    from .._ops import get_op

    # Check if any tensor arg is on meta device
    for arg in args:
        if isinstance(arg, Tensor) and arg.device.type == "meta":
            return _dispatch_meta(op_name, args, kwargs)

    # Try new standardized op first
    new_op = get_op(op_name)
    if new_op is not None:
        return _dispatch_new_op(op_name, new_op, args, kwargs)

    # Fall back to legacy dispatch
    return _dispatch_legacy(op_name, args, kwargs)


def _dispatch_meta(op_name: str, args, kwargs):
    """Dispatch meta device operations - shape inference only."""
    from .._backends import meta

    # Map op names to meta implementations
    meta_ops = {
        'add': meta.add_meta,
        'sub': meta.sub_meta,
        'mul': meta.mul_meta,
        'div': meta.div_meta,
        'matmul': meta.matmul_meta,
        'reshape': meta.reshape_meta,
        'transpose': meta.transpose_meta,
    }

    if op_name not in meta_ops:
        raise NotImplementedError(
            f"Meta device operation '{op_name}' not implemented. "
            f"Meta device only supports shape inference for common ops."
        )

    return meta_ops[op_name](*args, **kwargs)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_meta_ops.py -v`
Expected: PASS (all 3 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_backends/meta.py src/mindtorch_v2/_dispatch/dispatcher.py tests/mindtorch_v2/test_meta_ops.py
git commit -m "feat(meta): add meta device backend for shape inference

- Meta backend computes output shapes without actual computation
- Dispatcher routes meta device ops to meta backend
- Supports: add, sub, mul, div, matmul, reshape, transpose

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Test Transformers Integration

**Files:**
- Test: `tests/mindtorch_v2/test_meta_transformers.py`

**Step 1: Write integration test**

Create `tests/mindtorch_v2/test_meta_transformers.py`:

```python
"""Test meta device integration with transformers detection."""
import pytest


def test_transformers_can_detect_meta_context():
    """Transformers should detect meta device context via empty tensor."""
    from mindtorch_v2 import tensor, device

    # Outside context - should be CPU
    t1 = tensor([])
    assert t1.device.type == "cpu"

    # Inside meta context - should be meta
    with device("meta"):
        t2 = tensor([])
        assert t2.device.type == "meta"

    # After context - should be CPU again
    t3 = tensor([])
    assert t3.device.type == "cpu"


def test_meta_context_nested():
    """Nested device contexts should work correctly."""
    from mindtorch_v2 import tensor, device

    with device("cpu"):
        t1 = tensor([1.0])
        assert t1.device.type == "cpu"

        with device("meta"):
            t2 = tensor([2.0])
            assert t2.device.type == "meta"

        t3 = tensor([3.0])
        assert t3.device.type == "cpu"


def test_explicit_device_overrides_context():
    """Explicit device argument should override context."""
    from mindtorch_v2 import tensor, device

    with device("meta"):
        # Explicit device should override context
        t = tensor([1.0, 2.0], device="cpu")
        assert t.device.type == "cpu"
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_meta_transformers.py -v`
Expected: PASS (all 3 tests)

**Step 3: Run actual transformers test**

Run: `python tests/run_test_v2.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py::AlbertModelTest::test_cannot_load_with_meta_device_context_manager`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/mindtorch_v2/test_meta_transformers.py
git commit -m "test(meta): add transformers integration tests

- Test meta device detection via empty tensor
- Test nested device contexts
- Test explicit device override

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add get_default_device Function

**Files:**
- Modify: `src/mindtorch_v2/__init__.py`
- Test: `tests/mindtorch_v2/test_default_device.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_default_device.py`:

```python
"""Test get_default_device function."""
import pytest


def test_get_default_device_outside_context():
    """get_default_device should return cpu outside context."""
    from mindtorch_v2 import get_default_device, device as device_cls

    dev = get_default_device()
    assert dev == device_cls("cpu")


def test_get_default_device_in_context():
    """get_default_device should return context device."""
    from mindtorch_v2 import get_default_device, device as device_cls

    with device_cls("meta"):
        dev = get_default_device()
        assert dev == device_cls("meta")


def test_get_default_device_after_context():
    """get_default_device should return cpu after context exits."""
    from mindtorch_v2 import get_default_device, device as device_cls

    with device_cls("meta"):
        pass

    dev = get_default_device()
    assert dev == device_cls("cpu")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_default_device.py -v`
Expected: FAIL with "ImportError: cannot import name 'get_default_device'"

**Step 3: Add get_default_device function**

Modify `src/mindtorch_v2/__init__.py`:

```python
# Add to imports:
from ._device import device, _get_default_device

# Add public function:
def get_default_device():
    """Get the current default device.

    Returns the device set by device context manager, or cpu if none.

    Returns:
        device: Current default device
    """
    dev = _get_default_device()
    if dev is None:
        return device("cpu")
    return dev

# Add to __all__:
__all__ = [
    # ... existing exports ...
    'device',
    'get_default_device',
]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_default_device.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_default_device.py
git commit -m "feat(device): add get_default_device function

- Returns current device from context manager
- Returns cpu if no context active
- Matches PyTorch 2.3+ API

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

After completing all tasks:

1. **Meta device tensor creation** - Tensors respect device context, meta tensors have shape but no storage
2. **Meta device operations** - Shape inference for common ops (add, matmul, etc.)
3. **Transformers integration** - Empty tensor detection works correctly
4. **get_default_device** - Public API for checking current device context

**Verification:**
```bash
# Run all meta device tests
pytest tests/mindtorch_v2/test_meta_device.py tests/mindtorch_v2/test_meta_ops.py tests/mindtorch_v2/test_meta_transformers.py tests/mindtorch_v2/test_default_device.py -v

# Run transformers test
python tests/run_test_v2.py -vs tests/transformers/tests/models/albert/test_modeling_albert.py::AlbertModelTest::test_cannot_load_with_meta_device_context_manager
```

Expected: All tests PASS

**Total commits**: 4
**Total tests added**: ~13
**Transformers test fixed**: 1 (test_cannot_load_with_meta_device_context_manager)
