# mindtorch_v2 Op Standardization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Standardize all ops to use a unified pattern with pyboost primitives, clear forward/backward separation, and simplified dispatch.

**Architecture:** Each op is a self-contained class with `forward()` and `backward()` methods. All ops use pyboost primitives only (never `mindspore.ops` directly). The dispatch system becomes a thin routing layer.

**Tech Stack:** MindSpore pyboost primitives (`gen_ops_prim`), Python autograd, registry pattern.

---

## Prerequisites

Before starting, understand the core constraint:

```python
# NEVER use these directly (they don't handle device config):
import mindspore.ops  # WRONG
import mindspore.mint  # WRONG

# ALWAYS use pyboost primitives:
from mindspore.ops.auto_generate.gen_ops_prim import Add, Mul, ...
add_op = Add().set_device('CPU')  # CORRECT
```

---

## Task 1: Create Op Base Class

**Files:**
- Create: `src/mindtorch_v2/_ops/base.py`
- Test: `tests/mindtorch_v2/test_ops_base.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_ops_base.py`:

```python
"""Test Op base class."""
import pytest


def test_op_has_forward_and_backward():
    """Op base class must have forward and backward methods."""
    from mindtorch_v2._ops.base import Op

    op = Op()
    assert hasattr(op, 'forward')
    assert hasattr(op, 'backward')
    assert callable(op.forward)
    assert callable(op.backward)


def test_op_forward_not_implemented():
    """Base Op.forward should raise NotImplementedError."""
    from mindtorch_v2._ops.base import Op

    op = Op()
    with pytest.raises(NotImplementedError):
        op.forward()


def test_op_has_name():
    """Op should have a name property."""
    from mindtorch_v2._ops.base import Op

    op = Op()
    assert hasattr(op, 'name')
    assert op.name == "Op"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_base.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'mindtorch_v2._ops'"

**Step 3: Create Op base class**

Create directory and file `src/mindtorch_v2/_ops/__init__.py`:

```python
"""Operations module."""
from .base import Op
```

Create `src/mindtorch_v2/_ops/base.py`:

```python
"""Base class for all ops."""
from typing import Any, Tuple, Optional
from abc import ABC


class Op(ABC):
    """Base class for tensor operations.

    Each op implements:
    - forward(*args, **kwargs) -> result
    - backward(grad_output, *saved) -> tuple of gradients

    Subclasses should:
    1. Use only pyboost primitives in forward/backward
    2. Save necessary tensors in forward for backward pass
    3. Return gradients matching input order
    """

    # Class-level pyboost op instances (set in subclass)
    _pyboost_op = None

    @property
    def name(self) -> str:
        """Return op name."""
        return self.__class__.__name__

    def forward(self, *args, **kwargs) -> Any:
        """Execute forward pass.

        Must be overridden by subclasses.
        Returns: result tensor(s)
        """
        raise NotImplementedError(f"{self.name}.forward not implemented")

    def backward(self, grad_output: Any, *saved: Any) -> Tuple[Optional[Any], ...]:
        """Compute gradients.

        Args:
            grad_output: Gradient of loss w.r.t. output
            *saved: Tensors saved during forward

        Returns:
            Tuple of gradients for each input (None if no gradient)
        """
        raise NotImplementedError(f"{self.name}.backward not implemented")

    def __call__(self, *args, **kwargs):
        """Alias for forward."""
        return self.forward(*args, **kwargs)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_ops/ tests/mindtorch_v2/test_ops_base.py
git commit -m "feat(ops): add Op base class for standardized ops"
```

---

## Task 2: Create PyBoost Op Registry

**Files:**
- Create: `src/mindtorch_v2/_ops/pyboost.py`
- Test: `tests/mindtorch_v2/test_pyboost_registry.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_pyboost_registry.py`:

```python
"""Test pyboost op registry."""
import pytest


def test_get_pyboost_op():
    """Can get pyboost op by name."""
    from mindtorch_v2._ops.pyboost import get_pyboost_op

    add_op = get_pyboost_op('add')
    assert add_op is not None


def test_pyboost_op_works():
    """Pyboost op can execute."""
    from mindtorch_v2._ops.pyboost import get_pyboost_op
    import mindspore

    add_op = get_pyboost_op('add')
    a = mindspore.Tensor([1.0, 2.0])
    b = mindspore.Tensor([3.0, 4.0])
    result = add_op(a, b)

    assert list(result.asnumpy()) == [4.0, 6.0]


def test_unknown_op_returns_none():
    """Unknown op name returns None."""
    from mindtorch_v2._ops.pyboost import get_pyboost_op

    result = get_pyboost_op('nonexistent_op')
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_pyboost_registry.py -v`
Expected: FAIL with "ImportError"

**Step 3: Create pyboost registry**

Create `src/mindtorch_v2/_ops/pyboost.py`:

```python
"""PyBoost op registry.

All pyboost primitives are instantiated once with device set.
Use get_pyboost_op(name) to get the op instance.

NEVER use mindspore.ops or mindspore.mint directly.
"""
from typing import Optional, Dict, Any
from functools import lru_cache

# Import pyboost primitives
from mindspore.ops.auto_generate.gen_ops_prim import (
    # Binary math
    Add, Sub, Mul, Div, Pow,
    # Unary math
    Neg, Abs, Exp, Log, Sqrt, Rsqrt,
    Sin, Cos, Tanh, Sigmoid,
    # Activations
    ReLU, GeLU, SiLU,
    # Matrix ops
    MatMulExt, BatchMatMulExt,
    # Reductions
    SumExt, MeanExt, MaxDim, MinDim,
    ProdExt, ArgMaxExt, ArgMinExt,
    # Comparison
    Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual,
    # Shape ops
    Reshape, Transpose,
    # Other
    Concat, StackExt, ReduceAll,
)

# Device for all ops
_DEVICE = 'CPU'


@lru_cache(maxsize=128)
def _create_op(op_class, device: str = _DEVICE):
    """Create and cache a pyboost op instance."""
    return op_class().set_device(device)


# Op name to class mapping
_OP_MAP: Dict[str, Any] = {
    # Binary math
    'add': Add,
    'sub': Sub,
    'mul': Mul,
    'div': Div,
    'pow': Pow,
    # Unary math
    'neg': Neg,
    'abs': Abs,
    'exp': Exp,
    'log': Log,
    'sqrt': Sqrt,
    'rsqrt': Rsqrt,
    'sin': Sin,
    'cos': Cos,
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    # Activations
    'relu': ReLU,
    'gelu': GeLU,
    'silu': SiLU,
    # Matrix ops
    'matmul': MatMulExt,
    'bmm': BatchMatMulExt,
    # Reductions
    'sum': SumExt,
    'mean': MeanExt,
    'max_dim': MaxDim,
    'min_dim': MinDim,
    'prod': ProdExt,
    'argmax': ArgMaxExt,
    'argmin': ArgMinExt,
    # Comparison
    'eq': Equal,
    'ne': NotEqual,
    'gt': Greater,
    'lt': Less,
    'ge': GreaterEqual,
    'le': LessEqual,
    # Shape ops
    'reshape': Reshape,
    'transpose': Transpose,
    # Other
    'concat': Concat,
    'stack': StackExt,
    'reduce_all': ReduceAll,
}


def get_pyboost_op(name: str, device: str = _DEVICE) -> Optional[Any]:
    """Get a pyboost op by name.

    Args:
        name: Op name (e.g., 'add', 'matmul')
        device: Device to set ('CPU', 'GPU', 'Ascend')

    Returns:
        Pyboost op instance or None if not found
    """
    op_class = _OP_MAP.get(name)
    if op_class is None:
        return None
    return _create_op(op_class, device)


def set_device(device: str):
    """Set default device for all ops."""
    global _DEVICE
    _DEVICE = device
    _create_op.cache_clear()  # Clear cache to recreate with new device
```

Update `src/mindtorch_v2/_ops/__init__.py`:

```python
"""Operations module."""
from .base import Op
from .pyboost import get_pyboost_op, set_device
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_pyboost_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_ops/pyboost.py src/mindtorch_v2/_ops/__init__.py tests/mindtorch_v2/test_pyboost_registry.py
git commit -m "feat(ops): add pyboost op registry"
```

---

## Task 3: Create AddOp with Forward and Backward

**Files:**
- Create: `src/mindtorch_v2/_ops/math_ops.py`
- Test: `tests/mindtorch_v2/test_ops_math.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_ops_math.py`:

```python
"""Test math ops."""
import pytest
import numpy as np


def test_add_op_forward():
    """AddOp forward works."""
    from mindtorch_v2._ops.math_ops import AddOp
    import mindspore

    op = AddOp()
    a = mindspore.Tensor([1.0, 2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0, 6.0])

    result = op.forward(a, b)

    np.testing.assert_array_equal(result.asnumpy(), [5.0, 7.0, 9.0])


def test_add_op_backward():
    """AddOp backward computes correct gradients."""
    from mindtorch_v2._ops.math_ops import AddOp
    import mindspore

    op = AddOp()
    a = mindspore.Tensor([1.0, 2.0, 3.0])
    b = mindspore.Tensor([4.0, 5.0, 6.0])
    grad_output = mindspore.Tensor([1.0, 1.0, 1.0])

    grad_a, grad_b = op.backward(grad_output, a, b)

    # For add: grad_a = grad_output, grad_b = grad_output
    np.testing.assert_array_equal(grad_a.asnumpy(), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(grad_b.asnumpy(), [1.0, 1.0, 1.0])


def test_add_op_backward_broadcast():
    """AddOp backward handles broadcasting."""
    from mindtorch_v2._ops.math_ops import AddOp
    import mindspore

    op = AddOp()
    a = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = mindspore.Tensor([1.0, 1.0])  # (2,) broadcasts to (2, 2)
    grad_output = mindspore.Tensor([[1.0, 1.0], [1.0, 1.0]])

    grad_a, grad_b = op.backward(grad_output, a, b)

    # grad_a shape should match a
    assert grad_a.shape == (2, 2)
    # grad_b should be reduced to match b's shape
    assert grad_b.shape == (2,)
    np.testing.assert_array_equal(grad_b.asnumpy(), [2.0, 2.0])  # sum over broadcast dim
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_math.py -v`
Expected: FAIL with "ImportError"

**Step 3: Create AddOp**

Create `src/mindtorch_v2/_ops/math_ops.py`:

```python
"""Math operations using pyboost primitives."""
from typing import Tuple, Optional
import mindspore
from .base import Op
from .pyboost import get_pyboost_op


def _reduce_grad_to_shape(grad: mindspore.Tensor, target_shape: Tuple[int, ...]) -> mindspore.Tensor:
    """Reduce gradient to match target shape by summing over broadcasted dims.

    This is needed because broadcasting expands shapes during forward,
    and gradients must be reduced back to original shapes.
    """
    sum_op = get_pyboost_op('sum')

    # Sum over extra leading dimensions
    while grad.ndim > len(target_shape):
        grad = sum_op(grad, (0,), False)

    # Sum over dimensions that were broadcast (size 1 in target)
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and grad.shape[i] != 1:
            grad = sum_op(grad, (i,), True)

    return grad


class AddOp(Op):
    """Element-wise addition: c = a + b."""

    def forward(self, a: mindspore.Tensor, b: mindspore.Tensor) -> mindspore.Tensor:
        """Forward: c = a + b."""
        add_op = get_pyboost_op('add')
        return add_op(a, b)

    def backward(self, grad_output: mindspore.Tensor,
                 a: mindspore.Tensor, b: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Backward: grad_a = grad_c, grad_b = grad_c (with broadcast reduction)."""
        grad_a = _reduce_grad_to_shape(grad_output, a.shape)
        grad_b = _reduce_grad_to_shape(grad_output, b.shape)
        return grad_a, grad_b


class SubOp(Op):
    """Element-wise subtraction: c = a - b."""

    def forward(self, a: mindspore.Tensor, b: mindspore.Tensor) -> mindspore.Tensor:
        """Forward: c = a - b."""
        sub_op = get_pyboost_op('sub')
        return sub_op(a, b)

    def backward(self, grad_output: mindspore.Tensor,
                 a: mindspore.Tensor, b: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Backward: grad_a = grad_c, grad_b = -grad_c."""
        neg_op = get_pyboost_op('neg')
        grad_a = _reduce_grad_to_shape(grad_output, a.shape)
        grad_b = _reduce_grad_to_shape(neg_op(grad_output), b.shape)
        return grad_a, grad_b


class MulOp(Op):
    """Element-wise multiplication: c = a * b."""

    def forward(self, a: mindspore.Tensor, b: mindspore.Tensor) -> mindspore.Tensor:
        """Forward: c = a * b."""
        mul_op = get_pyboost_op('mul')
        return mul_op(a, b)

    def backward(self, grad_output: mindspore.Tensor,
                 a: mindspore.Tensor, b: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Backward: grad_a = grad_c * b, grad_b = grad_c * a."""
        mul_op = get_pyboost_op('mul')
        grad_a = _reduce_grad_to_shape(mul_op(grad_output, b), a.shape)
        grad_b = _reduce_grad_to_shape(mul_op(grad_output, a), b.shape)
        return grad_a, grad_b


class DivOp(Op):
    """Element-wise division: c = a / b."""

    def forward(self, a: mindspore.Tensor, b: mindspore.Tensor) -> mindspore.Tensor:
        """Forward: c = a / b."""
        div_op = get_pyboost_op('div')
        return div_op(a, b)

    def backward(self, grad_output: mindspore.Tensor,
                 a: mindspore.Tensor, b: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """Backward: grad_a = grad_c / b, grad_b = -grad_c * a / b^2."""
        div_op = get_pyboost_op('div')
        mul_op = get_pyboost_op('mul')
        neg_op = get_pyboost_op('neg')
        pow_op = get_pyboost_op('pow')

        grad_a = _reduce_grad_to_shape(div_op(grad_output, b), a.shape)
        b_sq = pow_op(b, mindspore.Tensor(2.0, b.dtype))
        grad_b = _reduce_grad_to_shape(neg_op(div_op(mul_op(grad_output, a), b_sq)), b.shape)
        return grad_a, grad_b
```

Update `src/mindtorch_v2/_ops/__init__.py`:

```python
"""Operations module."""
from .base import Op
from .pyboost import get_pyboost_op, set_device
from .math_ops import AddOp, SubOp, MulOp, DivOp
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_math.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_ops/math_ops.py tests/mindtorch_v2/test_ops_math.py
git commit -m "feat(ops): add standardized math ops (add, sub, mul, div)"
```

---

## Task 4: Create Op Registry and Simplified Dispatch

**Files:**
- Create: `src/mindtorch_v2/_ops/registry.py`
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Test: `tests/mindtorch_v2/test_ops_dispatch.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_ops_dispatch.py`:

```python
"""Test op dispatch with new ops."""
import pytest
import numpy as np


def test_dispatch_add_uses_new_op():
    """Dispatch should route to new standardized op."""
    from mindtorch_v2._ops import get_op
    from mindtorch_v2._ops.math_ops import AddOp

    op = get_op('add')
    assert isinstance(op, AddOp)


def test_dispatch_add_forward():
    """Dispatch add executes forward correctly."""
    from mindtorch_v2._ops import execute_op
    import mindspore

    a = mindspore.Tensor([1.0, 2.0])
    b = mindspore.Tensor([3.0, 4.0])

    result = execute_op('add', a, b)

    np.testing.assert_array_equal(result.asnumpy(), [4.0, 6.0])


def test_dispatch_returns_grad_info():
    """Execute_op returns info needed for backward."""
    from mindtorch_v2._ops import execute_op
    import mindspore

    a = mindspore.Tensor([1.0, 2.0])
    b = mindspore.Tensor([3.0, 4.0])

    result, backward_info = execute_op('add', a, b, return_backward_info=True)

    assert backward_info is not None
    assert 'op' in backward_info
    assert 'saved' in backward_info
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_dispatch.py -v`
Expected: FAIL with "ImportError"

**Step 3: Create op registry**

Create `src/mindtorch_v2/_ops/registry.py`:

```python
"""Op registry - maps op names to Op classes."""
from typing import Dict, Type, Optional, Any, Tuple
from .base import Op


# Global registry
_OP_REGISTRY: Dict[str, Type[Op]] = {}


def register_op(name: str):
    """Decorator to register an op class."""
    def decorator(op_class: Type[Op]):
        _OP_REGISTRY[name] = op_class
        return op_class
    return decorator


def get_op(name: str) -> Optional[Op]:
    """Get op instance by name."""
    op_class = _OP_REGISTRY.get(name)
    if op_class is None:
        return None
    return op_class()


def execute_op(name: str, *args, return_backward_info: bool = False, **kwargs) -> Any:
    """Execute an op by name.

    Args:
        name: Op name
        *args: Arguments to forward
        return_backward_info: If True, also return backward info
        **kwargs: Keyword arguments to forward

    Returns:
        result if return_backward_info=False
        (result, backward_info) if return_backward_info=True
    """
    op = get_op(name)
    if op is None:
        raise NotImplementedError(f"Op '{name}' not registered")

    result = op.forward(*args, **kwargs)

    if return_backward_info:
        backward_info = {
            'op': op,
            'saved': args,  # By default, save all args
        }
        return result, backward_info

    return result


# Register built-in ops
def _register_builtins():
    """Register all built-in ops."""
    from .math_ops import AddOp, SubOp, MulOp, DivOp

    _OP_REGISTRY['add'] = AddOp
    _OP_REGISTRY['sub'] = SubOp
    _OP_REGISTRY['mul'] = MulOp
    _OP_REGISTRY['div'] = DivOp


_register_builtins()
```

Update `src/mindtorch_v2/_ops/__init__.py`:

```python
"""Operations module."""
from .base import Op
from .pyboost import get_pyboost_op, set_device
from .math_ops import AddOp, SubOp, MulOp, DivOp
from .registry import get_op, execute_op, register_op
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_dispatch.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_ops/registry.py tests/mindtorch_v2/test_ops_dispatch.py
git commit -m "feat(ops): add op registry and execute_op"
```

---

## Task 5: Create Unary Math Ops (exp, log, sqrt, neg)

**Files:**
- Modify: `src/mindtorch_v2/_ops/math_ops.py`
- Test: `tests/mindtorch_v2/test_ops_unary.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_ops_unary.py`:

```python
"""Test unary math ops."""
import pytest
import numpy as np


def test_exp_forward():
    """ExpOp forward works."""
    from mindtorch_v2._ops.math_ops import ExpOp
    import mindspore

    op = ExpOp()
    x = mindspore.Tensor([0.0, 1.0, 2.0])
    result = op.forward(x)

    expected = np.exp([0.0, 1.0, 2.0])
    np.testing.assert_array_almost_equal(result.asnumpy(), expected, decimal=5)


def test_exp_backward():
    """ExpOp backward: d/dx exp(x) = exp(x)."""
    from mindtorch_v2._ops.math_ops import ExpOp
    import mindspore

    op = ExpOp()
    x = mindspore.Tensor([0.0, 1.0])
    exp_x = op.forward(x)  # exp(x) is needed for backward
    grad_output = mindspore.Tensor([1.0, 1.0])

    (grad_x,) = op.backward(grad_output, exp_x)

    # d/dx exp(x) = exp(x)
    np.testing.assert_array_almost_equal(grad_x.asnumpy(), exp_x.asnumpy(), decimal=5)


def test_log_backward():
    """LogOp backward: d/dx log(x) = 1/x."""
    from mindtorch_v2._ops.math_ops import LogOp
    import mindspore

    op = LogOp()
    x = mindspore.Tensor([1.0, 2.0])
    grad_output = mindspore.Tensor([1.0, 1.0])

    (grad_x,) = op.backward(grad_output, x)

    # d/dx log(x) = 1/x
    expected = [1.0, 0.5]
    np.testing.assert_array_almost_equal(grad_x.asnumpy(), expected, decimal=5)


def test_sqrt_backward():
    """SqrtOp backward: d/dx sqrt(x) = 0.5/sqrt(x)."""
    from mindtorch_v2._ops.math_ops import SqrtOp
    import mindspore

    op = SqrtOp()
    x = mindspore.Tensor([1.0, 4.0])
    sqrt_x = op.forward(x)
    grad_output = mindspore.Tensor([1.0, 1.0])

    (grad_x,) = op.backward(grad_output, sqrt_x)

    # d/dx sqrt(x) = 0.5/sqrt(x)
    expected = [0.5, 0.25]
    np.testing.assert_array_almost_equal(grad_x.asnumpy(), expected, decimal=5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_unary.py -v`
Expected: FAIL with "ImportError"

**Step 3: Add unary ops**

Add to `src/mindtorch_v2/_ops/math_ops.py`:

```python
class NegOp(Op):
    """Negation: y = -x."""

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        neg_op = get_pyboost_op('neg')
        return neg_op(x)

    def backward(self, grad_output: mindspore.Tensor, x: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor]:
        neg_op = get_pyboost_op('neg')
        return (neg_op(grad_output),)


class ExpOp(Op):
    """Exponential: y = exp(x)."""

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        exp_op = get_pyboost_op('exp')
        return exp_op(x)

    def backward(self, grad_output: mindspore.Tensor, exp_x: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor]:
        """Note: exp_x is the result of forward (exp(x))."""
        mul_op = get_pyboost_op('mul')
        return (mul_op(grad_output, exp_x),)


class LogOp(Op):
    """Natural logarithm: y = log(x)."""

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        log_op = get_pyboost_op('log')
        return log_op(x)

    def backward(self, grad_output: mindspore.Tensor, x: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor]:
        div_op = get_pyboost_op('div')
        return (div_op(grad_output, x),)


class SqrtOp(Op):
    """Square root: y = sqrt(x)."""

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        sqrt_op = get_pyboost_op('sqrt')
        return sqrt_op(x)

    def backward(self, grad_output: mindspore.Tensor, sqrt_x: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor]:
        """Note: sqrt_x is the result of forward (sqrt(x))."""
        div_op = get_pyboost_op('div')
        mul_op = get_pyboost_op('mul')
        two = mindspore.Tensor(2.0, sqrt_x.dtype)
        return (div_op(grad_output, mul_op(two, sqrt_x)),)


class RsqrtOp(Op):
    """Reciprocal square root: y = 1/sqrt(x)."""

    def forward(self, x: mindspore.Tensor) -> mindspore.Tensor:
        rsqrt_op = get_pyboost_op('rsqrt')
        return rsqrt_op(x)

    def backward(self, grad_output: mindspore.Tensor, x: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor]:
        # d/dx (1/sqrt(x)) = -0.5 * x^(-3/2)
        mul_op = get_pyboost_op('mul')
        pow_op = get_pyboost_op('pow')
        neg_op = get_pyboost_op('neg')
        half = mindspore.Tensor(0.5, x.dtype)
        neg_three_half = mindspore.Tensor(-1.5, x.dtype)
        grad_x = mul_op(neg_op(half), mul_op(grad_output, pow_op(x, neg_three_half)))
        return (grad_x,)
```

Update registry in `src/mindtorch_v2/_ops/registry.py`:

```python
def _register_builtins():
    """Register all built-in ops."""
    from .math_ops import (AddOp, SubOp, MulOp, DivOp,
                           NegOp, ExpOp, LogOp, SqrtOp, RsqrtOp)

    _OP_REGISTRY['add'] = AddOp
    _OP_REGISTRY['sub'] = SubOp
    _OP_REGISTRY['mul'] = MulOp
    _OP_REGISTRY['div'] = DivOp
    _OP_REGISTRY['neg'] = NegOp
    _OP_REGISTRY['exp'] = ExpOp
    _OP_REGISTRY['log'] = LogOp
    _OP_REGISTRY['sqrt'] = SqrtOp
    _OP_REGISTRY['rsqrt'] = RsqrtOp
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_unary.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_ops/math_ops.py tests/mindtorch_v2/test_ops_unary.py
git commit -m "feat(ops): add unary math ops (neg, exp, log, sqrt, rsqrt)"
```

---

## Task 6: Create MatMul and BMM Ops

**Files:**
- Create: `src/mindtorch_v2/_ops/linalg_ops.py`
- Test: `tests/mindtorch_v2/test_ops_linalg.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_ops_linalg.py`:

```python
"""Test linear algebra ops."""
import pytest
import numpy as np


def test_matmul_forward():
    """MatmulOp forward works."""
    from mindtorch_v2._ops.linalg_ops import MatmulOp
    import mindspore

    op = MatmulOp()
    a = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = mindspore.Tensor([[1.0], [1.0]])

    result = op.forward(a, b)

    expected = [[3.0], [7.0]]
    np.testing.assert_array_almost_equal(result.asnumpy(), expected, decimal=5)


def test_matmul_backward():
    """MatmulOp backward computes correct gradients."""
    from mindtorch_v2._ops.linalg_ops import MatmulOp
    import mindspore

    op = MatmulOp()
    a = mindspore.Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = mindspore.Tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)
    grad_output = mindspore.Tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2)

    grad_a, grad_b = op.backward(grad_output, a, b)

    # grad_a = grad @ b.T
    expected_grad_a = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_almost_equal(grad_a.asnumpy(), expected_grad_a, decimal=5)

    # grad_b = a.T @ grad
    expected_grad_b = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_array_almost_equal(grad_b.asnumpy(), expected_grad_b, decimal=5)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_linalg.py -v`
Expected: FAIL

**Step 3: Create linalg ops**

Create `src/mindtorch_v2/_ops/linalg_ops.py`:

```python
"""Linear algebra operations."""
from typing import Tuple
import mindspore
from .base import Op
from .pyboost import get_pyboost_op


class MatmulOp(Op):
    """Matrix multiplication: C = A @ B."""

    def forward(self, a: mindspore.Tensor, b: mindspore.Tensor) -> mindspore.Tensor:
        matmul_op = get_pyboost_op('matmul')
        return matmul_op(a, b)

    def backward(self, grad_output: mindspore.Tensor,
                 a: mindspore.Tensor, b: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        C = A @ B
        grad_A = grad_C @ B.T
        grad_B = A.T @ grad_C
        """
        matmul_op = get_pyboost_op('matmul')
        transpose_op = get_pyboost_op('transpose')

        # Transpose last two dims
        b_t = transpose_op(b, tuple(range(b.ndim - 2)) + (b.ndim - 1, b.ndim - 2))
        a_t = transpose_op(a, tuple(range(a.ndim - 2)) + (a.ndim - 1, a.ndim - 2))

        grad_a = matmul_op(grad_output, b_t)
        grad_b = matmul_op(a_t, grad_output)

        return grad_a, grad_b


class BmmOp(Op):
    """Batched matrix multiplication: C = A @ B (batch dims)."""

    def forward(self, a: mindspore.Tensor, b: mindspore.Tensor) -> mindspore.Tensor:
        bmm_op = get_pyboost_op('bmm')
        return bmm_op(a, b)

    def backward(self, grad_output: mindspore.Tensor,
                 a: mindspore.Tensor, b: mindspore.Tensor
                 ) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        bmm_op = get_pyboost_op('bmm')
        transpose_op = get_pyboost_op('transpose')

        # For 3D: swap last two dims
        b_t = transpose_op(b, (0, 2, 1))
        a_t = transpose_op(a, (0, 2, 1))

        grad_a = bmm_op(grad_output, b_t)
        grad_b = bmm_op(a_t, grad_output)

        return grad_a, grad_b
```

Update registry in `src/mindtorch_v2/_ops/registry.py`:

```python
def _register_builtins():
    from .math_ops import (AddOp, SubOp, MulOp, DivOp,
                           NegOp, ExpOp, LogOp, SqrtOp, RsqrtOp)
    from .linalg_ops import MatmulOp, BmmOp

    # ... existing registrations ...
    _OP_REGISTRY['matmul'] = MatmulOp
    _OP_REGISTRY['bmm'] = BmmOp
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_linalg.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_ops/linalg_ops.py tests/mindtorch_v2/test_ops_linalg.py
git commit -m "feat(ops): add matmul and bmm ops"
```

---

## Task 7: Integrate New Ops with Dispatch System

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Test: `tests/mindtorch_v2/test_dispatch_integration.py`

**Step 1: Write failing test**

Create `tests/mindtorch_v2/test_dispatch_integration.py`:

```python
"""Test dispatch integration with new ops."""
import pytest
import numpy as np


def test_dispatch_uses_new_ops():
    """dispatch() should use new standardized ops."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])

    result = dispatch('add', a, b)

    np.testing.assert_array_equal(result.numpy(), [5.0, 7.0, 9.0])


def test_dispatch_with_autograd():
    """dispatch() should create grad_fn using new ops."""
    from mindtorch_v2._dispatch import dispatch
    from mindtorch_v2 import Tensor

    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

    result = dispatch('add', a, b)

    assert result.requires_grad
    assert result.grad_fn is not None
```

**Step 2: Run test**

Run: `pytest tests/mindtorch_v2/test_dispatch_integration.py -v`

**Step 3: Update dispatcher to use new ops**

The dispatcher should be updated to use the new ops system. This requires:

1. Using `execute_op` from the new ops module
2. Simplifying `_make_grad_fn` to use the op's backward method
3. Storing the op and saved tensors in grad_fn

This is a larger refactor - update `src/mindtorch_v2/_dispatch/dispatcher.py`:

```python
"""Dispatcher routes ops to correct implementations."""
from typing import Any, Tuple
from .keys import DispatchKey
from .registry import get_op_impl


def dispatch(op_name: str, *args, **kwargs) -> Any:
    """Dispatch an operation.

    Uses new standardized ops when available, falls back to legacy.
    """
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled
    from .._ops import get_op

    # Try new standardized op first
    new_op = get_op(op_name)

    if new_op is not None:
        return _dispatch_new_op(op_name, new_op, args, kwargs)

    # Fall back to legacy dispatch
    return _dispatch_legacy(op_name, args, kwargs)


def _dispatch_new_op(op_name: str, op, args, kwargs):
    """Dispatch using new standardized op."""
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled
    from .._autograd.node import AccumulateGrad
    from .._backends.pyboost_cpu import _get_ms_data, _wrap_result

    # Convert Tensor args to MindSpore tensors
    ms_args = []
    tensor_args = []
    for arg in args:
        if isinstance(arg, Tensor):
            ms_args.append(_get_ms_data(arg))
            tensor_args.append(arg)
        else:
            ms_args.append(arg)

    # Execute forward
    ms_result = op.forward(*ms_args, **kwargs)

    # Wrap result
    result = _wrap_result(ms_result)

    # Setup autograd if needed
    requires_grad = any(t.requires_grad for t in tensor_args)
    if is_grad_enabled() and requires_grad:
        result._requires_grad = True
        result._grad_fn = _make_new_grad_fn(op_name, op, tensor_args, ms_args, ms_result)

    return result


def _make_new_grad_fn(op_name, op, tensor_args, ms_args, ms_result):
    """Create grad_fn using new op's backward method."""
    from .._autograd.node import Node, AccumulateGrad
    from .._tensor import Tensor

    class OpBackward(Node):
        def __init__(self, op, saved_tensors, saved_ms):
            super().__init__()
            self._op = op
            self._saved_tensors = saved_tensors
            self._saved_ms = saved_ms
            self._name = f"{op.name}Backward"

        def backward(self, grad_outputs):
            from .._backends.pyboost_cpu import _get_ms_data, _wrap_result

            grad_out = grad_outputs[0]
            ms_grad_out = _get_ms_data(grad_out)

            # Call op's backward
            ms_grads = self._op.backward(ms_grad_out, *self._saved_ms)

            # Wrap results
            grads = []
            for g in ms_grads:
                if g is not None:
                    grads.append(_wrap_result(g))
                else:
                    grads.append(None)

            return tuple(grads)

    grad_fn = OpBackward(op, tensor_args, ms_args)

    # Build next_functions
    next_fns = []
    for t in tensor_args:
        if t.requires_grad:
            if t.grad_fn is not None:
                next_fns.append((t.grad_fn, 0))
            else:
                next_fns.append((AccumulateGrad(t), 0))
        else:
            next_fns.append((None, 0))

    grad_fn._next_functions = tuple(next_fns)

    return grad_fn


def _dispatch_legacy(op_name: str, args, kwargs):
    """Legacy dispatch (existing code)."""
    # ... keep existing implementation for ops not yet converted ...
    pass
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dispatch_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/dispatcher.py tests/mindtorch_v2/test_dispatch_integration.py
git commit -m "feat(dispatch): integrate new standardized ops with dispatch"
```

---

## Summary

After completing these tasks:

1. **Clear Op Pattern**: Each op is a class with `forward()` and `backward()`
2. **Pyboost Only**: All ops use `get_pyboost_op()` - never `mindspore.ops`
3. **Simplified Dispatch**: New ops use their own backward, no complex `_make_grad_fn`
4. **Testable**: Each op can be unit tested independently

**Next Steps** (future tasks):
- Task 8-12: Convert remaining ops (activations, reductions, etc.)
- Task 13: Remove legacy dispatch code
- Task 14: Full test suite with new ops
