# mindtorch v2 Phase 2: Dispatch + Core Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the dispatcher system and implement core math, reduction, and comparison operations - enough to run basic tensor computations with proper op dispatch.

**Architecture:** PyTorch-style dispatcher with dispatch keys. Ops register implementations per backend. CPU backend calls into existing `_op_prim/cpu` primitives. Tensor methods delegate to dispatched functions.

**Tech Stack:** Python 3.9+, MindSpore 2.7.2, NumPy, pytest

**Worktree:** `/Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2`
**Branch:** `feature/mindtorch-v2`
**Source dir:** `src/mindtorch_v2/`
**Test dir:** `tests/mindtorch_v2/`

**Test command:**
```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

---

## Task 1: Dispatch Keys

**Files:**
- Create: `src/mindtorch_v2/_dispatch/__init__.py`
- Create: `src/mindtorch_v2/_dispatch/keys.py`
- Create: `tests/mindtorch_v2/test_dispatch.py`

**Step 1: Create dispatch package and write test**

```python
# tests/mindtorch_v2/test_dispatch.py
from mindtorch_v2._dispatch import DispatchKey


def test_dispatch_key_enum():
    """DispatchKey enum has required keys."""
    assert hasattr(DispatchKey, 'Autograd')
    assert hasattr(DispatchKey, 'Backend_CPU')
    assert hasattr(DispatchKey, 'Backend_CUDA')
    assert hasattr(DispatchKey, 'Backend_Ascend')
    assert hasattr(DispatchKey, 'CompositeExplicit')


def test_dispatch_key_ordering():
    """Autograd comes before Backend keys."""
    assert DispatchKey.Autograd.value < DispatchKey.Backend_CPU.value
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_dispatch.py -v
```

Expected: FAIL with `ImportError`

**Step 3: Implement dispatch keys**

```python
# src/mindtorch_v2/_dispatch/__init__.py
from .keys import DispatchKey
from .registry import register_op, get_op_impl
from .dispatcher import dispatch

__all__ = ['DispatchKey', 'register_op', 'get_op_impl', 'dispatch']
```

```python
# src/mindtorch_v2/_dispatch/keys.py
"""PyTorch-style dispatch keys for operation routing."""

from enum import IntEnum, auto


class DispatchKey(IntEnum):
    """Dispatch keys ordered by priority (lower = higher priority)."""

    # Autograd wrapper - records ops for backward
    Autograd = auto()

    # Automatic mixed precision
    AutocastCPU = auto()
    AutocastGPU = auto()

    # Batching (vmap)
    Batched = auto()

    # Functionalization (mutations → copies)
    Functionalize = auto()

    # JIT tracing
    Tracing = auto()

    # Backend execution
    Backend_CPU = auto()
    Backend_CUDA = auto()
    Backend_Ascend = auto()

    # Composite ops (decompose to primitives)
    CompositeExplicit = auto()


# Default active keys for normal execution
DEFAULT_DISPATCH_KEYS = frozenset({
    DispatchKey.Backend_CPU,
})
```

**Step 4: Run test to verify it passes**

Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/ tests/mindtorch_v2/test_dispatch.py
git commit -m "feat(v2): add DispatchKey enum for operation routing"
```

---

## Task 2: Op Registry

**Files:**
- Create: `src/mindtorch_v2/_dispatch/registry.py`
- Modify: `tests/mindtorch_v2/test_dispatch.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/test_dispatch.py
from mindtorch_v2._dispatch import register_op, get_op_impl, DispatchKey


def test_register_and_get_op():
    """Can register and retrieve op implementations."""
    @register_op("test_add", DispatchKey.Backend_CPU)
    def test_add_cpu(a, b):
        return a + b

    impl = get_op_impl("test_add", DispatchKey.Backend_CPU)
    assert impl is not None
    assert impl(2, 3) == 5


def test_get_unregistered_op():
    """Getting unregistered op returns None."""
    impl = get_op_impl("nonexistent_op", DispatchKey.Backend_CPU)
    assert impl is None


def test_register_multiple_backends():
    """Can register same op for multiple backends."""
    @register_op("test_mul", DispatchKey.Backend_CPU)
    def test_mul_cpu(a, b):
        return a * b

    @register_op("test_mul", DispatchKey.Backend_CUDA)
    def test_mul_cuda(a, b):
        return a * b * 1  # Different impl

    cpu_impl = get_op_impl("test_mul", DispatchKey.Backend_CPU)
    cuda_impl = get_op_impl("test_mul", DispatchKey.Backend_CUDA)

    assert cpu_impl is not None
    assert cuda_impl is not None
    assert cpu_impl is not cuda_impl
```

**Step 2: Run test to verify it fails**

**Step 3: Implement registry**

```python
# src/mindtorch_v2/_dispatch/registry.py
"""Op registration system for dispatch."""

from typing import Callable, Dict, Optional, Tuple
from .keys import DispatchKey

# Global registry: (op_name, dispatch_key) -> implementation
_OP_REGISTRY: Dict[Tuple[str, DispatchKey], Callable] = {}


def register_op(op_name: str, dispatch_key: DispatchKey):
    """Decorator to register an op implementation for a dispatch key.

    Usage:
        @register_op("add", DispatchKey.Backend_CPU)
        def add_cpu(a, b):
            return prim_add(a, b)
    """
    def decorator(func: Callable) -> Callable:
        _OP_REGISTRY[(op_name, dispatch_key)] = func
        return func
    return decorator


def get_op_impl(op_name: str, dispatch_key: DispatchKey) -> Optional[Callable]:
    """Get the implementation of an op for a dispatch key."""
    return _OP_REGISTRY.get((op_name, dispatch_key))


def list_registered_ops() -> list:
    """List all registered (op_name, dispatch_key) pairs."""
    return list(_OP_REGISTRY.keys())


def clear_registry():
    """Clear all registered ops (for testing)."""
    _OP_REGISTRY.clear()
```

**Step 4: Run test to verify it passes**

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/registry.py tests/mindtorch_v2/test_dispatch.py
git commit -m "feat(v2): add op registry for dispatch system"
```

---

## Task 3: Dispatcher

**Files:**
- Create: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `tests/mindtorch_v2/test_dispatch.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/test_dispatch.py
from mindtorch_v2._dispatch import dispatch
import mindtorch_v2 as torch


def test_dispatch_basic():
    """Dispatch routes to correct backend."""
    # Register a test op
    @register_op("dispatch_test", DispatchKey.Backend_CPU)
    def dispatch_test_cpu(x):
        return x * 2

    # Create tensor and dispatch
    t = torch.tensor([1.0, 2.0, 3.0])
    result = dispatch("dispatch_test", t)

    # Should use CPU impl
    expected = torch.tensor([2.0, 4.0, 6.0])
    import numpy as np
    np.testing.assert_array_almost_equal(result.numpy(), expected.numpy())


def test_dispatch_determines_backend_from_tensor():
    """Dispatch determines backend from tensor device."""
    @register_op("backend_test", DispatchKey.Backend_CPU)
    def backend_test_cpu(x):
        return "cpu"

    t = torch.tensor([1.0])  # CPU tensor
    result = dispatch("backend_test", t)
    assert result == "cpu"
```

**Step 2: Run test to verify it fails**

**Step 3: Implement dispatcher**

```python
# src/mindtorch_v2/_dispatch/dispatcher.py
"""Dispatcher routes ops to correct implementations based on dispatch keys."""

from typing import Any, Tuple
from .keys import DispatchKey
from .registry import get_op_impl


def _get_backend_key(args: Tuple) -> DispatchKey:
    """Determine backend dispatch key from arguments.

    Looks at tensor arguments to determine which backend to use.
    """
    from .._tensor import Tensor

    for arg in args:
        if isinstance(arg, Tensor):
            device_type = arg.device.type
            if device_type == "cpu":
                return DispatchKey.Backend_CPU
            elif device_type == "cuda":
                return DispatchKey.Backend_CUDA
            elif device_type in ("npu", "ascend"):
                return DispatchKey.Backend_Ascend

    # Default to CPU
    return DispatchKey.Backend_CPU


def dispatch(op_name: str, *args, **kwargs) -> Any:
    """Dispatch an operation to the correct implementation.

    Args:
        op_name: Name of the operation (e.g., "add", "matmul")
        *args: Positional arguments to pass to the op
        **kwargs: Keyword arguments to pass to the op

    Returns:
        Result of the operation

    Raises:
        NotImplementedError: If no implementation found for the op
    """
    # Determine which backend to use
    backend_key = _get_backend_key(args)

    # Try to get implementation for this backend
    impl = get_op_impl(op_name, backend_key)

    if impl is None:
        # Try composite fallback
        impl = get_op_impl(op_name, DispatchKey.CompositeExplicit)

    if impl is None:
        raise NotImplementedError(
            f"No implementation found for op '{op_name}' with dispatch key {backend_key}"
        )

    return impl(*args, **kwargs)
```

**Step 4: Run test to verify it passes**

Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/dispatcher.py tests/mindtorch_v2/test_dispatch.py
git commit -m "feat(v2): add dispatcher for routing ops to backends"
```

---

## Task 4: CPU Backend Base

**Files:**
- Create: `src/mindtorch_v2/_backends/__init__.py`
- Create: `src/mindtorch_v2/_backends/cpu.py`
- Create: `tests/mindtorch_v2/test_backends.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_backends.py
import numpy as np
import mindtorch_v2 as torch


def test_backend_add():
    """CPU backend can add tensors."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = torch.add(a, b)
    expected = np.array([5.0, 7.0, 9.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_add_scalar():
    """CPU backend can add scalar to tensor."""
    a = torch.tensor([1.0, 2.0, 3.0])
    result = torch.add(a, 10.0)
    expected = np.array([11.0, 12.0, 13.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_sub():
    """CPU backend can subtract tensors."""
    a = torch.tensor([5.0, 6.0, 7.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    result = torch.sub(a, b)
    expected = np.array([4.0, 4.0, 4.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_mul():
    """CPU backend can multiply tensors."""
    a = torch.tensor([2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0])
    result = torch.mul(a, b)
    expected = np.array([10.0, 18.0, 28.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_backend_div():
    """CPU backend can divide tensors."""
    a = torch.tensor([10.0, 20.0, 30.0])
    b = torch.tensor([2.0, 4.0, 5.0])
    result = torch.div(a, b)
    expected = np.array([5.0, 5.0, 6.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)
```

**Step 2: Run test to verify it fails**

**Step 3: Implement CPU backend**

```python
# src/mindtorch_v2/_backends/__init__.py
"""Backend implementations for different devices."""

from . import cpu

__all__ = ['cpu']
```

```python
# src/mindtorch_v2/_backends/cpu.py
"""CPU backend implementations using NumPy.

All ops here are registered with DispatchKey.Backend_CPU.
These implementations use NumPy for computation and wrap results in Tensor.
"""

import numpy as np
from .._dispatch import register_op, DispatchKey
from .._tensor import Tensor


def _to_numpy(x):
    """Convert tensor or scalar to numpy array."""
    if isinstance(x, Tensor):
        return x.numpy()
    return np.asarray(x)


def _wrap_result(arr, dtype=None, device="cpu"):
    """Wrap numpy array as Tensor."""
    return Tensor(arr, dtype=dtype, device=device)


# --- Binary math ops ---

@register_op("add", DispatchKey.Backend_CPU)
def add_cpu(a, b):
    """Element-wise addition."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np + b_np
    return _wrap_result(result)


@register_op("sub", DispatchKey.Backend_CPU)
def sub_cpu(a, b):
    """Element-wise subtraction."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np - b_np
    return _wrap_result(result)


@register_op("mul", DispatchKey.Backend_CPU)
def mul_cpu(a, b):
    """Element-wise multiplication."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np * b_np
    return _wrap_result(result)


@register_op("div", DispatchKey.Backend_CPU)
def div_cpu(a, b):
    """Element-wise division."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    result = a_np / b_np
    return _wrap_result(result)


@register_op("neg", DispatchKey.Backend_CPU)
def neg_cpu(a):
    """Element-wise negation."""
    a_np = _to_numpy(a)
    return _wrap_result(-a_np)


@register_op("abs", DispatchKey.Backend_CPU)
def abs_cpu(a):
    """Element-wise absolute value."""
    a_np = _to_numpy(a)
    return _wrap_result(np.abs(a_np))


@register_op("pow", DispatchKey.Backend_CPU)
def pow_cpu(a, exponent):
    """Element-wise power."""
    a_np = _to_numpy(a)
    exp_np = _to_numpy(exponent)
    return _wrap_result(np.power(a_np, exp_np))


# --- Unary math ops ---

@register_op("exp", DispatchKey.Backend_CPU)
def exp_cpu(a):
    """Element-wise exponential."""
    a_np = _to_numpy(a)
    return _wrap_result(np.exp(a_np))


@register_op("log", DispatchKey.Backend_CPU)
def log_cpu(a):
    """Element-wise natural logarithm."""
    a_np = _to_numpy(a)
    return _wrap_result(np.log(a_np))


@register_op("sqrt", DispatchKey.Backend_CPU)
def sqrt_cpu(a):
    """Element-wise square root."""
    a_np = _to_numpy(a)
    return _wrap_result(np.sqrt(a_np))


@register_op("sin", DispatchKey.Backend_CPU)
def sin_cpu(a):
    """Element-wise sine."""
    a_np = _to_numpy(a)
    return _wrap_result(np.sin(a_np))


@register_op("cos", DispatchKey.Backend_CPU)
def cos_cpu(a):
    """Element-wise cosine."""
    a_np = _to_numpy(a)
    return _wrap_result(np.cos(a_np))


@register_op("tanh", DispatchKey.Backend_CPU)
def tanh_cpu(a):
    """Element-wise hyperbolic tangent."""
    a_np = _to_numpy(a)
    return _wrap_result(np.tanh(a_np))


@register_op("sigmoid", DispatchKey.Backend_CPU)
def sigmoid_cpu(a):
    """Element-wise sigmoid."""
    a_np = _to_numpy(a)
    return _wrap_result(1.0 / (1.0 + np.exp(-a_np)))


@register_op("relu", DispatchKey.Backend_CPU)
def relu_cpu(a):
    """Element-wise ReLU."""
    a_np = _to_numpy(a)
    return _wrap_result(np.maximum(a_np, 0))
```

**Step 4: Add functional API exports**

```python
# src/mindtorch_v2/_functional.py
"""Functional API for tensor operations."""

from ._dispatch import dispatch


def add(input, other, *, alpha=1, out=None):
    """Add tensors: input + alpha * other."""
    if alpha != 1:
        other = mul(other, alpha)
    return dispatch("add", input, other)


def sub(input, other, *, alpha=1, out=None):
    """Subtract tensors: input - alpha * other."""
    if alpha != 1:
        other = mul(other, alpha)
    return dispatch("sub", input, other)


def mul(input, other, *, out=None):
    """Multiply tensors element-wise."""
    return dispatch("mul", input, other)


def div(input, other, *, rounding_mode=None, out=None):
    """Divide tensors element-wise."""
    result = dispatch("div", input, other)
    if rounding_mode == "trunc":
        result = trunc(result)
    elif rounding_mode == "floor":
        result = floor(result)
    return result


def neg(input, *, out=None):
    """Negate tensor element-wise."""
    return dispatch("neg", input)


def abs(input, *, out=None):
    """Absolute value element-wise."""
    return dispatch("abs", input)


def pow(input, exponent, *, out=None):
    """Power element-wise."""
    return dispatch("pow", input, exponent)


def exp(input, *, out=None):
    """Exponential element-wise."""
    return dispatch("exp", input)


def log(input, *, out=None):
    """Natural logarithm element-wise."""
    return dispatch("log", input)


def sqrt(input, *, out=None):
    """Square root element-wise."""
    return dispatch("sqrt", input)


def sin(input, *, out=None):
    """Sine element-wise."""
    return dispatch("sin", input)


def cos(input, *, out=None):
    """Cosine element-wise."""
    return dispatch("cos", input)


def tanh(input, *, out=None):
    """Hyperbolic tangent element-wise."""
    return dispatch("tanh", input)
```

**Step 5: Update __init__.py to export ops**

Add to `src/mindtorch_v2/__init__.py`:
```python
from ._functional import (
    add, sub, mul, div, neg, abs, pow,
    exp, log, sqrt, sin, cos, tanh,
)

# Import backends to register ops
from ._backends import cpu
```

**Step 6: Run test to verify it passes**

Expected: PASS (5 tests)

**Step 7: Commit**

```bash
git add src/mindtorch_v2/_backends/ src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_backends.py
git commit -m "feat(v2): add CPU backend with basic math ops"
```

---

## Task 5: More Math Ops and Tensor Methods

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Create: `tests/mindtorch_v2/test_math_ops.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_math_ops.py
import numpy as np
import mindtorch_v2 as torch


def test_tensor_add_method():
    """Tensor has add method."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = a.add(b)
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 6.0])


def test_tensor_add_operator():
    """Tensor supports + operator."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    result = a + b
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 6.0])


def test_tensor_sub_operator():
    """Tensor supports - operator."""
    a = torch.tensor([5.0, 6.0])
    b = torch.tensor([1.0, 2.0])
    result = a - b
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 4.0])


def test_tensor_mul_operator():
    """Tensor supports * operator."""
    a = torch.tensor([2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    result = a * b
    np.testing.assert_array_almost_equal(result.numpy(), [8.0, 15.0])


def test_tensor_div_operator():
    """Tensor supports / operator."""
    a = torch.tensor([10.0, 20.0])
    b = torch.tensor([2.0, 4.0])
    result = a / b
    np.testing.assert_array_almost_equal(result.numpy(), [5.0, 5.0])


def test_tensor_neg_operator():
    """Tensor supports unary - operator."""
    a = torch.tensor([1.0, -2.0, 3.0])
    result = -a
    np.testing.assert_array_almost_equal(result.numpy(), [-1.0, 2.0, -3.0])


def test_tensor_pow_operator():
    """Tensor supports ** operator."""
    a = torch.tensor([2.0, 3.0])
    result = a ** 2
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 9.0])


def test_tensor_matmul():
    """Tensor supports @ operator for matmul."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    result = a @ b
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_exp():
    """torch.exp works."""
    a = torch.tensor([0.0, 1.0, 2.0])
    result = torch.exp(a)
    expected = np.exp([0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_log():
    """torch.log works."""
    a = torch.tensor([1.0, np.e, np.e**2])
    result = torch.log(a)
    expected = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(result.numpy(), expected)


def test_sqrt():
    """torch.sqrt works."""
    a = torch.tensor([1.0, 4.0, 9.0])
    result = torch.sqrt(a)
    np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.0, 3.0])


def test_abs():
    """torch.abs works."""
    a = torch.tensor([-1.0, 2.0, -3.0])
    result = torch.abs(a)
    np.testing.assert_array_almost_equal(result.numpy(), [1.0, 2.0, 3.0])
```

**Step 2: Run test to verify it fails**

**Step 3: Add matmul to CPU backend**

Add to `src/mindtorch_v2/_backends/cpu.py`:

```python
@register_op("matmul", DispatchKey.Backend_CPU)
def matmul_cpu(a, b):
    """Matrix multiplication."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(np.matmul(a_np, b_np))
```

**Step 4: Add tensor operators to _tensor.py**

Add these methods to the Tensor class in `src/mindtorch_v2/_tensor.py`:

```python
    # --- Arithmetic operators ---

    def add(self, other, *, alpha=1):
        from . import add as torch_add, mul as torch_mul
        if alpha != 1:
            other = torch_mul(other, alpha)
        return torch_add(self, other)

    def sub(self, other, *, alpha=1):
        from . import sub as torch_sub, mul as torch_mul
        if alpha != 1:
            other = torch_mul(other, alpha)
        return torch_sub(self, other)

    def mul(self, other):
        from . import mul as torch_mul
        return torch_mul(self, other)

    def div(self, other, *, rounding_mode=None):
        from . import div as torch_div
        return torch_div(self, other, rounding_mode=rounding_mode)

    def neg(self):
        from . import neg as torch_neg
        return torch_neg(self)

    def abs(self):
        from . import abs as torch_abs
        return torch_abs(self)

    def pow(self, exponent):
        from . import pow as torch_pow
        return torch_pow(self, exponent)

    def exp(self):
        from . import exp as torch_exp
        return torch_exp(self)

    def log(self):
        from . import log as torch_log
        return torch_log(self)

    def sqrt(self):
        from . import sqrt as torch_sqrt
        return torch_sqrt(self)

    def matmul(self, other):
        from ._dispatch import dispatch
        return dispatch("matmul", self, other)

    # --- Operator overloads ---

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        from . import add as torch_add
        return torch_add(other, self)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        from . import sub as torch_sub
        return torch_sub(other, self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        from . import mul as torch_mul
        return torch_mul(other, self)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        from . import div as torch_div
        return torch_div(other, self)

    def __neg__(self):
        return self.neg()

    def __abs__(self):
        return self.abs()

    def __pow__(self, exponent):
        return self.pow(exponent)

    def __matmul__(self, other):
        return self.matmul(other)
```

**Step 5: Add matmul to functional API**

Add to `src/mindtorch_v2/_functional.py`:

```python
def matmul(input, other, *, out=None):
    """Matrix multiplication."""
    return dispatch("matmul", input, other)
```

And export it in `__init__.py`.

**Step 6: Run test to verify it passes**

Expected: PASS (12 tests)

**Step 7: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/_functional.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_math_ops.py
git commit -m "feat(v2): add math ops and tensor arithmetic operators"
```

---

## Task 6: Reduction Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Create: `tests/mindtorch_v2/test_reduction_ops.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_reduction_ops.py
import numpy as np
import mindtorch_v2 as torch


def test_sum_all():
    """sum() reduces all elements."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(a)
    assert result.item() == 10.0


def test_sum_dim():
    """sum() can reduce along dimension."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(a, dim=0)
    np.testing.assert_array_almost_equal(result.numpy(), [4.0, 6.0])


def test_sum_keepdim():
    """sum() can keep dimensions."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.sum(a, dim=1, keepdim=True)
    assert result.shape == (2, 1)
    np.testing.assert_array_almost_equal(result.numpy(), [[3.0], [7.0]])


def test_mean_all():
    """mean() reduces all elements."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.mean(a)
    assert result.item() == 2.5


def test_mean_dim():
    """mean() can reduce along dimension."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = torch.mean(a, dim=1)
    np.testing.assert_array_almost_equal(result.numpy(), [1.5, 3.5])


def test_max_all():
    """max() returns maximum element."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    result = torch.max(a)
    assert result.item() == 5.0


def test_max_dim():
    """max() can reduce along dimension, returning values and indices."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    values, indices = torch.max(a, dim=1)
    np.testing.assert_array_almost_equal(values.numpy(), [5.0, 3.0])
    np.testing.assert_array_equal(indices.numpy(), [1, 0])


def test_min_all():
    """min() returns minimum element."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    result = torch.min(a)
    assert result.item() == 1.0


def test_min_dim():
    """min() can reduce along dimension."""
    a = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
    values, indices = torch.min(a, dim=1)
    np.testing.assert_array_almost_equal(values.numpy(), [1.0, 2.0])
    np.testing.assert_array_equal(indices.numpy(), [0, 1])


def test_tensor_sum_method():
    """Tensor.sum() method works."""
    a = torch.tensor([1.0, 2.0, 3.0])
    result = a.sum()
    assert result.item() == 6.0


def test_tensor_mean_method():
    """Tensor.mean() method works."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = a.mean()
    assert result.item() == 2.5
```

**Step 2: Run test to verify it fails**

**Step 3: Add reduction ops to CPU backend**

Add to `src/mindtorch_v2/_backends/cpu.py`:

```python
# --- Reduction ops ---

@register_op("sum", DispatchKey.Backend_CPU)
def sum_cpu(a, dim=None, keepdim=False):
    """Sum reduction."""
    a_np = _to_numpy(a)
    result = np.sum(a_np, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("mean", DispatchKey.Backend_CPU)
def mean_cpu(a, dim=None, keepdim=False):
    """Mean reduction."""
    a_np = _to_numpy(a)
    result = np.mean(a_np, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("max", DispatchKey.Backend_CPU)
def max_cpu(a, dim=None, keepdim=False):
    """Max reduction."""
    a_np = _to_numpy(a)
    if dim is None:
        result = np.max(a_np)
        return _wrap_result(np.array(result))
    else:
        values = np.max(a_np, axis=dim, keepdims=keepdim)
        indices = np.argmax(a_np, axis=dim)
        if keepdim:
            indices = np.expand_dims(indices, axis=dim)
        # Return named tuple-like result
        from collections import namedtuple
        MaxResult = namedtuple('MaxResult', ['values', 'indices'])
        return MaxResult(_wrap_result(values), _wrap_result(indices.astype(np.int64)))


@register_op("min", DispatchKey.Backend_CPU)
def min_cpu(a, dim=None, keepdim=False):
    """Min reduction."""
    a_np = _to_numpy(a)
    if dim is None:
        result = np.min(a_np)
        return _wrap_result(np.array(result))
    else:
        values = np.min(a_np, axis=dim, keepdims=keepdim)
        indices = np.argmin(a_np, axis=dim)
        if keepdim:
            indices = np.expand_dims(indices, axis=dim)
        from collections import namedtuple
        MinResult = namedtuple('MinResult', ['values', 'indices'])
        return MinResult(_wrap_result(values), _wrap_result(indices.astype(np.int64)))


@register_op("prod", DispatchKey.Backend_CPU)
def prod_cpu(a, dim=None, keepdim=False):
    """Product reduction."""
    a_np = _to_numpy(a)
    result = np.prod(a_np, axis=dim, keepdims=keepdim)
    return _wrap_result(result)


@register_op("argmax", DispatchKey.Backend_CPU)
def argmax_cpu(a, dim=None, keepdim=False):
    """Argmax."""
    a_np = _to_numpy(a)
    result = np.argmax(a_np, axis=dim)
    if keepdim and dim is not None:
        result = np.expand_dims(result, axis=dim)
    return _wrap_result(result.astype(np.int64))


@register_op("argmin", DispatchKey.Backend_CPU)
def argmin_cpu(a, dim=None, keepdim=False):
    """Argmin."""
    a_np = _to_numpy(a)
    result = np.argmin(a_np, axis=dim)
    if keepdim and dim is not None:
        result = np.expand_dims(result, axis=dim)
    return _wrap_result(result.astype(np.int64))
```

**Step 4: Add functional API for reductions**

Add to `src/mindtorch_v2/_functional.py`:

```python
def sum(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Sum of tensor elements."""
    return dispatch("sum", input, dim=dim, keepdim=keepdim)


def mean(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Mean of tensor elements."""
    return dispatch("mean", input, dim=dim, keepdim=keepdim)


def max(input, dim=None, keepdim=False, *, out=None):
    """Max of tensor elements."""
    return dispatch("max", input, dim=dim, keepdim=keepdim)


def min(input, dim=None, keepdim=False, *, out=None):
    """Min of tensor elements."""
    return dispatch("min", input, dim=dim, keepdim=keepdim)


def prod(input, dim=None, keepdim=False, *, dtype=None, out=None):
    """Product of tensor elements."""
    return dispatch("prod", input, dim=dim, keepdim=keepdim)


def argmax(input, dim=None, keepdim=False):
    """Index of max element."""
    return dispatch("argmax", input, dim=dim, keepdim=keepdim)


def argmin(input, dim=None, keepdim=False):
    """Index of min element."""
    return dispatch("argmin", input, dim=dim, keepdim=keepdim)
```

**Step 5: Add tensor methods for reductions**

Add to Tensor class in `src/mindtorch_v2/_tensor.py`:

```python
    def sum(self, dim=None, keepdim=False, *, dtype=None):
        from . import sum as torch_sum
        return torch_sum(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def mean(self, dim=None, keepdim=False, *, dtype=None):
        from . import mean as torch_mean
        return torch_mean(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def max(self, dim=None, keepdim=False):
        from . import max as torch_max
        return torch_max(self, dim=dim, keepdim=keepdim)

    def min(self, dim=None, keepdim=False):
        from . import min as torch_min
        return torch_min(self, dim=dim, keepdim=keepdim)

    def prod(self, dim=None, keepdim=False, *, dtype=None):
        from . import prod as torch_prod
        return torch_prod(self, dim=dim, keepdim=keepdim, dtype=dtype)

    def argmax(self, dim=None, keepdim=False):
        from . import argmax as torch_argmax
        return torch_argmax(self, dim=dim, keepdim=keepdim)

    def argmin(self, dim=None, keepdim=False):
        from . import argmin as torch_argmin
        return torch_argmin(self, dim=dim, keepdim=keepdim)
```

**Step 6: Update __init__.py exports**

**Step 7: Run test to verify it passes**

Expected: PASS (11 tests)

**Step 8: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_reduction_ops.py
git commit -m "feat(v2): add reduction ops (sum, mean, max, min, prod, argmax, argmin)"
```

---

## Task 7: Comparison Ops

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Create: `tests/mindtorch_v2/test_comparison_ops.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_comparison_ops.py
import numpy as np
import mindtorch_v2 as torch


def test_eq():
    """Element-wise equality."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 5.0, 3.0])
    result = torch.eq(a, b)
    np.testing.assert_array_equal(result.numpy(), [True, False, True])


def test_ne():
    """Element-wise not equal."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 5.0, 3.0])
    result = torch.ne(a, b)
    np.testing.assert_array_equal(result.numpy(), [False, True, False])


def test_gt():
    """Element-wise greater than."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.gt(a, b)
    np.testing.assert_array_equal(result.numpy(), [False, True, False])


def test_lt():
    """Element-wise less than."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.lt(a, b)
    np.testing.assert_array_equal(result.numpy(), [True, False, False])


def test_ge():
    """Element-wise greater than or equal."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.ge(a, b)
    np.testing.assert_array_equal(result.numpy(), [False, True, True])


def test_le():
    """Element-wise less than or equal."""
    a = torch.tensor([1.0, 5.0, 3.0])
    b = torch.tensor([2.0, 3.0, 3.0])
    result = torch.le(a, b)
    np.testing.assert_array_equal(result.numpy(), [True, False, True])


def test_tensor_eq_operator():
    """Tensor supports == operator."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a == b
    np.testing.assert_array_equal(result.numpy(), [True, False])


def test_tensor_ne_operator():
    """Tensor supports != operator."""
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a != b
    np.testing.assert_array_equal(result.numpy(), [False, True])


def test_tensor_gt_operator():
    """Tensor supports > operator."""
    a = torch.tensor([3.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a > b
    np.testing.assert_array_equal(result.numpy(), [True, False])


def test_tensor_lt_operator():
    """Tensor supports < operator."""
    a = torch.tensor([3.0, 2.0])
    b = torch.tensor([1.0, 5.0])
    result = a < b
    np.testing.assert_array_equal(result.numpy(), [False, True])


def test_tensor_ge_operator():
    """Tensor supports >= operator."""
    a = torch.tensor([3.0, 2.0, 5.0])
    b = torch.tensor([1.0, 5.0, 5.0])
    result = a >= b
    np.testing.assert_array_equal(result.numpy(), [True, False, True])


def test_tensor_le_operator():
    """Tensor supports <= operator."""
    a = torch.tensor([3.0, 2.0, 5.0])
    b = torch.tensor([1.0, 5.0, 5.0])
    result = a <= b
    np.testing.assert_array_equal(result.numpy(), [False, True, True])
```

**Step 2: Run test to verify it fails**

**Step 3: Add comparison ops to CPU backend**

Add to `src/mindtorch_v2/_backends/cpu.py`:

```python
# --- Comparison ops ---

@register_op("eq", DispatchKey.Backend_CPU)
def eq_cpu(a, b):
    """Element-wise equality."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np == b_np)


@register_op("ne", DispatchKey.Backend_CPU)
def ne_cpu(a, b):
    """Element-wise not equal."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np != b_np)


@register_op("gt", DispatchKey.Backend_CPU)
def gt_cpu(a, b):
    """Element-wise greater than."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np > b_np)


@register_op("lt", DispatchKey.Backend_CPU)
def lt_cpu(a, b):
    """Element-wise less than."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np < b_np)


@register_op("ge", DispatchKey.Backend_CPU)
def ge_cpu(a, b):
    """Element-wise greater than or equal."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np >= b_np)


@register_op("le", DispatchKey.Backend_CPU)
def le_cpu(a, b):
    """Element-wise less than or equal."""
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)
    return _wrap_result(a_np <= b_np)
```

**Step 4: Add functional API**

Add to `src/mindtorch_v2/_functional.py`:

```python
def eq(input, other, *, out=None):
    """Element-wise equality comparison."""
    return dispatch("eq", input, other)


def ne(input, other, *, out=None):
    """Element-wise not equal comparison."""
    return dispatch("ne", input, other)


def gt(input, other, *, out=None):
    """Element-wise greater than comparison."""
    return dispatch("gt", input, other)


def lt(input, other, *, out=None):
    """Element-wise less than comparison."""
    return dispatch("lt", input, other)


def ge(input, other, *, out=None):
    """Element-wise greater than or equal comparison."""
    return dispatch("ge", input, other)


def le(input, other, *, out=None):
    """Element-wise less than or equal comparison."""
    return dispatch("le", input, other)
```

**Step 5: Add tensor comparison operators**

Add to Tensor class in `src/mindtorch_v2/_tensor.py`:

```python
    # --- Comparison operators ---

    def eq(self, other):
        from . import eq as torch_eq
        return torch_eq(self, other)

    def ne(self, other):
        from . import ne as torch_ne
        return torch_ne(self, other)

    def gt(self, other):
        from . import gt as torch_gt
        return torch_gt(self, other)

    def lt(self, other):
        from . import lt as torch_lt
        return torch_lt(self, other)

    def ge(self, other):
        from . import ge as torch_ge
        return torch_ge(self, other)

    def le(self, other):
        from . import le as torch_le
        return torch_le(self, other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __gt__(self, other):
        return self.gt(other)

    def __lt__(self, other):
        return self.lt(other)
