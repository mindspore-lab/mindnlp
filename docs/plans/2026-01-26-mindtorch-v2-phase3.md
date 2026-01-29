# mindtorch v2 Phase 3: Autograd Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a tape-based autograd engine that records operations on tensors with `requires_grad=True` and computes gradients via backpropagation.

**Architecture:** Each differentiable op creates a Node that stores saved tensors and links to input nodes via `next_functions`. Backward traverses this graph in reverse topological order, calling each node's `backward()` to compute gradients. Leaf tensors accumulate gradients in `.grad`.

**Tech Stack:** Python 3.9+, NumPy (for gradient computation in CPU backend)

**Worktree:** `/Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2`
**Branch:** `feature/mindtorch-v2`
**Source dir:** `src/mindtorch_v2/`
**Test dir:** `tests/mindtorch_v2/`

**Test command:**
```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

---

## Task 1: Grad Mode Context Managers

**Files:**
- Create: `src/mindtorch_v2/_autograd/__init__.py`
- Create: `src/mindtorch_v2/_autograd/grad_mode.py`
- Create: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Create autograd package and write test**

```python
# tests/mindtorch_v2/test_autograd.py
import mindtorch_v2 as torch


def test_grad_enabled_default():
    """Grad is enabled by default."""
    assert torch.is_grad_enabled() == True


def test_no_grad_context():
    """no_grad disables gradient computation."""
    assert torch.is_grad_enabled() == True
    with torch.no_grad():
        assert torch.is_grad_enabled() == False
    assert torch.is_grad_enabled() == True


def test_enable_grad_context():
    """enable_grad re-enables gradient computation."""
    with torch.no_grad():
        assert torch.is_grad_enabled() == False
        with torch.enable_grad():
            assert torch.is_grad_enabled() == True
        assert torch.is_grad_enabled() == False


def test_set_grad_enabled():
    """set_grad_enabled can toggle gradient computation."""
    assert torch.is_grad_enabled() == True
    with torch.set_grad_enabled(False):
        assert torch.is_grad_enabled() == False
    assert torch.is_grad_enabled() == True


def test_no_grad_decorator():
    """no_grad works as a decorator."""
    @torch.no_grad()
    def my_func():
        return torch.is_grad_enabled()

    assert torch.is_grad_enabled() == True
    assert my_func() == False
    assert torch.is_grad_enabled() == True
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_autograd.py -v
```

Expected: FAIL with `AttributeError: module 'mindtorch_v2' has no attribute 'is_grad_enabled'`

**Step 3: Implement grad mode**

```python
# src/mindtorch_v2/_autograd/__init__.py
from .grad_mode import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)

__all__ = ['is_grad_enabled', 'set_grad_enabled', 'no_grad', 'enable_grad']
```

```python
# src/mindtorch_v2/_autograd/grad_mode.py
"""Gradient mode context managers."""

import threading
from contextlib import contextmanager
from functools import wraps

# Thread-local storage for gradient mode
_grad_mode = threading.local()


def _get_grad_mode():
    """Get current gradient mode (default True)."""
    if not hasattr(_grad_mode, 'enabled'):
        _grad_mode.enabled = True
    return _grad_mode.enabled


def _set_grad_mode(enabled: bool):
    """Set current gradient mode."""
    _grad_mode.enabled = enabled


def is_grad_enabled() -> bool:
    """Returns True if gradient computation is currently enabled."""
    return _get_grad_mode()


class set_grad_enabled:
    """Context manager / decorator to set gradient computation on or off.

    Can be used as a context manager:
        with torch.set_grad_enabled(False):
            ...

    Or as a decorator:
        @torch.set_grad_enabled(False)
        def my_func():
            ...
    """

    def __init__(self, mode: bool):
        self.prev = _get_grad_mode()
        self.mode = mode

    def __enter__(self):
        _set_grad_mode(self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _set_grad_mode(self.prev)
        return False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


class no_grad(set_grad_enabled):
    """Context manager / decorator that disables gradient computation.

    Equivalent to set_grad_enabled(False).
    """

    def __init__(self):
        super().__init__(False)


class enable_grad(set_grad_enabled):
    """Context manager / decorator that enables gradient computation.

    Equivalent to set_grad_enabled(True).
    """

    def __init__(self):
        super().__init__(True)
```

**Step 4: Update __init__.py to export grad mode**

Add to `src/mindtorch_v2/__init__.py`:
```python
from ._autograd import (
    is_grad_enabled, set_grad_enabled, no_grad, enable_grad,
)
```

**Step 5: Run test to verify it passes**

Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_autograd/ tests/mindtorch_v2/test_autograd.py src/mindtorch_v2/__init__.py
git commit -m "feat(v2): add grad mode context managers (no_grad, enable_grad)"
```

---

## Task 2: Node Base Class

**Files:**
- Create: `src/mindtorch_v2/_autograd/node.py`
- Modify: `src/mindtorch_v2/_autograd/__init__.py`
- Modify: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/test_autograd.py
from mindtorch_v2._autograd import Node


def test_node_base_class():
    """Node is the base class for autograd functions."""
    node = Node()
    assert hasattr(node, 'next_functions')
    assert hasattr(node, 'backward')
    assert node.next_functions == ()


def test_node_next_functions():
    """Node can store next_functions."""
    node1 = Node()
    node2 = Node()
    node2._next_functions = ((node1, 0),)
    assert node2.next_functions == ((node1, 0),)


def test_node_backward_not_implemented():
    """Node.backward raises NotImplementedError by default."""
    node = Node()
    try:
        node.backward((None,))
        assert False, "Should have raised"
    except NotImplementedError:
        pass
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Node base class**

```python
# src/mindtorch_v2/_autograd/node.py
"""Autograd Node - base class for gradient functions."""

from typing import Tuple, Optional, Any


class Node:
    """Base class for all autograd graph nodes (grad_fn).

    Each node represents an operation in the computational graph.
    During backward, nodes compute gradients for their inputs.

    Attributes:
        _next_functions: Tuple of (Node, output_idx) pairs linking to input nodes
        _saved_tensors: Tensors saved for backward computation
        _needs_input_grad: Tuple of bools indicating which inputs need gradients
    """

    __slots__ = ('_next_functions', '_saved_tensors', '_needs_input_grad', '_name')

    def __init__(self):
        self._next_functions: Tuple[Tuple[Optional['Node'], int], ...] = ()
        self._saved_tensors: Tuple[Any, ...] = ()
        self._needs_input_grad: Tuple[bool, ...] = ()
        self._name: str = self.__class__.__name__

    @property
    def next_functions(self) -> Tuple[Tuple[Optional['Node'], int], ...]:
        """Return the next functions in the graph (inputs to this op)."""
        return self._next_functions

    @property
    def name(self) -> str:
        """Return the name of this node."""
        return self._name

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self._saved_tensors = tensors

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return tensors saved for backward."""
        return self._saved_tensors

    def backward(self, grad_outputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Compute gradients for inputs given gradients of outputs.

        Args:
            grad_outputs: Tuple of gradients w.r.t. each output

        Returns:
            Tuple of gradients w.r.t. each input (or None if not needed)

        Must be overridden by subclasses.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.backward is not implemented"
        )

    def __repr__(self):
        return f"<{self._name}>"


class AccumulateGrad(Node):
    """Special node for leaf tensors that accumulates gradients.

    This node doesn't have a backward() - instead, the backward engine
    directly accumulates gradients into the tensor's .grad attribute.
    """

    __slots__ = ('_variable',)

    def __init__(self, variable):
        super().__init__()
        self._variable = variable  # Weak ref to tensor (actually just store it)
        self._name = "AccumulateGrad"

    @property
    def variable(self):
        return self._variable

    def backward(self, grad_outputs):
        # AccumulateGrad doesn't compute gradients - the engine handles it
        raise RuntimeError("AccumulateGrad.backward should not be called directly")
```

**Step 4: Update autograd __init__.py**

```python
# src/mindtorch_v2/_autograd/__init__.py
from .grad_mode import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)
from .node import Node, AccumulateGrad

__all__ = [
    'is_grad_enabled', 'set_grad_enabled', 'no_grad', 'enable_grad',
    'Node', 'AccumulateGrad',
]
```

**Step 5: Run test to verify it passes**

Expected: PASS (8 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_autograd/node.py src/mindtorch_v2/_autograd/__init__.py tests/mindtorch_v2/test_autograd.py
git commit -m "feat(v2): add Node base class for autograd graph"
```

---

## Task 3: Backward Engine

**Files:**
- Create: `src/mindtorch_v2/_autograd/engine.py`
- Modify: `src/mindtorch_v2/_autograd/__init__.py`
- Modify: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/test_autograd.py
def test_backward_simple():
    """Simple backward pass computes gradient."""
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x * 2  # y = 2x, dy/dx = 2
    loss = y.sum()  # scalar output
    loss.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])


def test_backward_chain():
    """Backward through chain of operations."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2
    z = y + 1
    loss = z.sum()
    loss.backward()

    # d(loss)/dx = d(sum(2x+1))/dx = 2
    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0, 2.0])


def test_backward_multiple_uses():
    """Backward when tensor is used multiple times."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x + x  # y = 2x
    loss = y.sum()
    loss.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])


def test_backward_no_grad_tensor():
    """Tensors without requires_grad don't get gradients."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.tensor([3.0, 4.0], requires_grad=False)
    z = x + y
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert y.grad is None


def test_backward_retain_graph():
    """Can backward multiple times with retain_graph=True."""
    x = torch.tensor([1.0], requires_grad=True)
    y = x * 2
    loss = y.sum()

    loss.backward(retain_graph=True)
    first_grad = x.grad.numpy().copy()

    loss.backward(retain_graph=True)
    # Gradients accumulate
    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), first_grad * 2)


def test_tensor_backward_method():
    """Tensor.backward() works."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = (x ** 2).sum()  # y = x1^2 + x2^2, dy/dx = 2x
    y.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 4.0])
```

**Step 2: Run test to verify it fails**

**Step 3: Implement backward engine**

```python
# src/mindtorch_v2/_autograd/engine.py
"""Autograd backward engine."""

from typing import Optional, Sequence, Dict, List, Set
from collections import defaultdict


def _topological_sort(root_nodes):
    """Topological sort of nodes for backward pass.

    Returns nodes in order from outputs to inputs.
    """
    visited = set()
    order = []

    def dfs(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))

        for next_node, _ in node.next_functions:
            dfs(next_node)

        order.append(node)

    for node in root_nodes:
        dfs(node)

    return list(reversed(order))


def backward(
    tensors,
    grad_tensors=None,
    retain_graph: bool = False,
    create_graph: bool = False,
):
    """Compute gradients of tensors w.r.t. graph leaves.

    Args:
        tensors: Tensors to compute gradients for (must be scalar or provide grad_tensors)
        grad_tensors: Gradient of the loss w.r.t. each tensor (default: ones for scalars)
        retain_graph: If True, graph is not freed after backward
        create_graph: If True, graph of gradient computation is created (for higher-order gradients)
    """
    from .node import AccumulateGrad
    from .._tensor import Tensor

    if not isinstance(tensors, (tuple, list)):
        tensors = (tensors,)

    # Initialize grad_tensors
    if grad_tensors is None:
        grad_tensors = []
        for t in tensors:
            if t.numel() != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            # Create gradient of 1.0 with same shape
            grad_tensors.append(Tensor([1.0], dtype=t.dtype, device=str(t.device)))
        grad_tensors = tuple(grad_tensors)
    elif not isinstance(grad_tensors, (tuple, list)):
        grad_tensors = (grad_tensors,)

    # Collect root nodes
    root_nodes = []
    node_to_grad: Dict[int, List] = defaultdict(list)

    for tensor, grad in zip(tensors, grad_tensors):
        if tensor.grad_fn is not None:
            root_nodes.append(tensor.grad_fn)
            node_to_grad[id(tensor.grad_fn)].append(grad)

    if not root_nodes:
        return  # Nothing to compute

    # Topological sort
    sorted_nodes = _topological_sort(root_nodes)

    # Compute gradients
    for node in sorted_nodes:
        # Accumulate incoming gradients
        grads = node_to_grad[id(node)]
        if not grads:
            continue

        # Sum gradients if multiple
        if len(grads) == 1:
            grad_output = grads[0]
        else:
            from .. import add
            grad_output = grads[0]
            for g in grads[1:]:
                grad_output = add(grad_output, g)

        # Handle AccumulateGrad specially
        if isinstance(node, AccumulateGrad):
            variable = node.variable
            if variable.grad is None:
                variable.grad = grad_output
            else:
                from .. import add
                variable.grad = add(variable.grad, grad_output)
            continue

        # Compute input gradients
        try:
            input_grads = node.backward((grad_output,))
        except Exception as e:
            raise RuntimeError(f"Error in backward for {node.name}: {e}") from e

        if not isinstance(input_grads, tuple):
            input_grads = (input_grads,)

        # Propagate to next functions
        for (next_node, idx), grad in zip(node.next_functions, input_grads):
            if next_node is not None and grad is not None:
                node_to_grad[id(next_node)].append(grad)
```

**Step 4: Update autograd __init__.py**

```python
# src/mindtorch_v2/_autograd/__init__.py
from .grad_mode import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)
from .node import Node, AccumulateGrad
from .engine import backward

__all__ = [
    'is_grad_enabled', 'set_grad_enabled', 'no_grad', 'enable_grad',
    'Node', 'AccumulateGrad',
    'backward',
]
```

**Step 5: Add Tensor.backward() method**

Add to `src/mindtorch_v2/_tensor.py` (after comparison operators section):

```python
    # --- Autograd ---

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """Compute gradients of this tensor w.r.t. graph leaves.

        Args:
            gradient: Gradient w.r.t. this tensor. Required for non-scalar tensors.
            retain_graph: If True, graph is retained for future backward calls.
            create_graph: If True, graph of gradient computation is constructed.
        """
        from ._autograd import backward as autograd_backward
        autograd_backward(self, gradient, retain_graph=retain_graph, create_graph=create_graph)
```

**Step 6: This test will still fail - we need backward functions for ops (Task 4)**

Note: Tests in Step 1 will pass only after Task 4 (Backward Functions) is complete.

**Step 7: Commit (partial - engine infrastructure)**

```bash
git add src/mindtorch_v2/_autograd/engine.py src/mindtorch_v2/_autograd/__init__.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_autograd.py
git commit -m "feat(v2): add backward engine infrastructure"
```

---

## Task 4: Autograd Dispatch Integration

**Files:**
- Modify: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Modify: `src/mindtorch_v2/_autograd/__init__.py`
- Create: `src/mindtorch_v2/_autograd/functions.py`

**Step 1: Write failing test (already written in Task 3)**

The tests from Task 3 will drive this implementation.

**Step 2: Create backward functions for basic ops**

```python
# src/mindtorch_v2/_autograd/functions.py
"""Backward functions for autograd."""

from .node import Node


class AddBackward(Node):
    """Backward for element-wise addition."""

    def __init__(self):
        super().__init__()
        self._name = "AddBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        # d(a+b)/da = 1, d(a+b)/db = 1
        return (grad_output, grad_output)


class SubBackward(Node):
    """Backward for element-wise subtraction."""

    def __init__(self):
        super().__init__()
        self._name = "SubBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        # d(a-b)/da = 1, d(a-b)/db = -1
        from .. import neg
        return (grad_output, neg(grad_output))


class MulBackward(Node):
    """Backward for element-wise multiplication."""

    def __init__(self):
        super().__init__()
        self._name = "MulBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors
        # d(a*b)/da = b, d(a*b)/db = a
        from .. import mul
        return (mul(grad_output, b), mul(grad_output, a))


class DivBackward(Node):
    """Backward for element-wise division."""

    def __init__(self):
        super().__init__()
        self._name = "DivBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        from .. import div, mul, neg, pow
        grad_a = div(grad_output, b)
        grad_b = neg(div(mul(grad_output, a), pow(b, 2)))
        return (grad_a, grad_b)


class NegBackward(Node):
    """Backward for negation."""

    def __init__(self):
        super().__init__()
        self._name = "NegBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        from .. import neg
        return (neg(grad_output),)


class PowBackward(Node):
    """Backward for power."""

    def __init__(self):
        super().__init__()
        self._name = "PowBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        base, exp = self.saved_tensors
        # d(a^n)/da = n * a^(n-1)
        from .. import mul, pow, sub, log
        from .._tensor import Tensor

        exp_minus_1 = sub(exp, Tensor(1.0))
        grad_base = mul(mul(grad_output, exp), pow(base, exp_minus_1))
        # grad_exp = a^n * log(a) (not implemented for now)
        return (grad_base, None)


class SumBackward(Node):
    """Backward for sum reduction."""

    def __init__(self):
        super().__init__()
        self._name = "SumBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_shape = self._input_shape
        # Gradient broadcasts to input shape
        from .._creation import ones
        from .. import mul
        grad = mul(ones(input_shape), grad_output)
        return (grad,)


class MeanBackward(Node):
    """Backward for mean reduction."""

    def __init__(self):
        super().__init__()
        self._name = "MeanBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        input_shape = self._input_shape
        numel = 1
        for s in input_shape:
            numel *= s
        # Gradient is 1/n broadcast to input shape
        from .._creation import ones
        from .. import mul, div
        from .._tensor import Tensor
        grad = div(mul(ones(input_shape), grad_output), Tensor(float(numel)))
        return (grad,)


class MatmulBackward(Node):
    """Backward for matrix multiplication."""

    def __init__(self):
        super().__init__()
        self._name = "MatmulBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        a, b = self.saved_tensors
        # d(A @ B)/dA = grad @ B^T
        # d(A @ B)/dB = A^T @ grad
        from .. import matmul

        # Handle 2D case
        if a.ndim == 2 and b.ndim == 2:
            grad_a = matmul(grad_output, b.t())
            grad_b = matmul(a.t(), grad_output)
        else:
            # For now, only support 2D
            grad_a = matmul(grad_output, b.t())
            grad_b = matmul(a.t(), grad_output)

        return (grad_a, grad_b)


class ExpBackward(Node):
    """Backward for exp."""

    def __init__(self):
        super().__init__()
        self._name = "ExpBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        result = self.saved_tensors[0]  # We save exp(x)
        from .. import mul
        return (mul(grad_output, result),)


class LogBackward(Node):
    """Backward for log."""

    def __init__(self):
        super().__init__()
        self._name = "LogBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        x = self.saved_tensors[0]
        from .. import div
        return (div(grad_output, x),)


class SqrtBackward(Node):
    """Backward for sqrt."""

    def __init__(self):
        super().__init__()
        self._name = "SqrtBackward"

    def backward(self, grad_outputs):
        grad_output = grad_outputs[0]
        result = self.saved_tensors[0]  # We save sqrt(x)
        from .. import div, mul
        from .._tensor import Tensor
        # d(sqrt(x))/dx = 1/(2*sqrt(x))
        return (div(grad_output, mul(Tensor(2.0), result)),)
```

**Step 3: Modify dispatcher to integrate autograd**

```python
# src/mindtorch_v2/_dispatch/dispatcher.py
"""Dispatcher routes ops to correct implementations based on dispatch keys."""

from typing import Any, Tuple
from .keys import DispatchKey
from .registry import get_op_impl


def _get_backend_key(args: Tuple) -> DispatchKey:
    """Determine backend dispatch key from arguments."""
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

    return DispatchKey.Backend_CPU


def _requires_grad(args) -> bool:
    """Check if any input tensor requires grad."""
    from .._tensor import Tensor
    for arg in args:
        if isinstance(arg, Tensor) and arg.requires_grad:
            return True
    return False


def _make_grad_fn(op_name: str, args, result):
    """Create gradient function node for an op."""
    from .._tensor import Tensor
    from .._autograd.node import AccumulateGrad
    from .._autograd import functions as F

    # Map op names to backward classes
    backward_classes = {
        'add': F.AddBackward,
        'sub': F.SubBackward,
        'mul': F.MulBackward,
        'div': F.DivBackward,
        'neg': F.NegBackward,
        'pow': F.PowBackward,
        'sum': F.SumBackward,
        'mean': F.MeanBackward,
        'matmul': F.MatmulBackward,
        'exp': F.ExpBackward,
        'log': F.LogBackward,
        'sqrt': F.SqrtBackward,
    }

    BackwardClass = backward_classes.get(op_name)
    if BackwardClass is None:
        return None  # No backward for this op

    grad_fn = BackwardClass()

    # Build next_functions
    next_fns = []
    tensors_to_save = []

    for arg in args:
        if isinstance(arg, Tensor):
            if arg.requires_grad:
                if arg.grad_fn is not None:
                    next_fns.append((arg.grad_fn, 0))
                else:
                    # Leaf tensor - create AccumulateGrad
                    acc_grad = AccumulateGrad(arg)
                    next_fns.append((acc_grad, 0))
            else:
                next_fns.append((None, 0))
            tensors_to_save.append(arg)
        else:
            # Scalar - wrap in tensor for saving
            tensors_to_save.append(Tensor(arg))

    grad_fn._next_functions = tuple(next_fns)

    # Save tensors for backward
    if op_name in ('mul', 'div', 'pow', 'matmul'):
        grad_fn.save_for_backward(*tensors_to_save)
    elif op_name == 'exp':
        grad_fn.save_for_backward(result)  # Save result for exp backward
    elif op_name == 'log':
        grad_fn.save_for_backward(tensors_to_save[0])  # Save input
    elif op_name == 'sqrt':
        grad_fn.save_for_backward(result)  # Save result
    elif op_name in ('sum', 'mean'):
        grad_fn._input_shape = tensors_to_save[0].shape

    return grad_fn


def dispatch(op_name: str, *args, **kwargs) -> Any:
    """Dispatch an operation to the correct implementation."""
    from .._tensor import Tensor
    from .._autograd import is_grad_enabled

    # Determine backend
    backend_key = _get_backend_key(args)

    # Get implementation
    impl = get_op_impl(op_name, backend_key)
    if impl is None:
        impl = get_op_impl(op_name, DispatchKey.CompositeExplicit)
    if impl is None:
        raise NotImplementedError(
            f"No implementation found for op '{op_name}' with dispatch key {backend_key}"
        )

    # Execute forward
    result = impl(*args, **kwargs)

    # Record autograd if needed
    if is_grad_enabled() and _requires_grad(args) and isinstance(result, Tensor):
        grad_fn = _make_grad_fn(op_name, args, result)
        if grad_fn is not None:
            result._grad_fn = grad_fn
            result._requires_grad = True

    return result
```

**Step 4: Update autograd __init__.py**

```python
# src/mindtorch_v2/_autograd/__init__.py
from .grad_mode import (
    is_grad_enabled,
    set_grad_enabled,
    no_grad,
    enable_grad,
)
from .node import Node, AccumulateGrad
from .engine import backward
from . import functions

__all__ = [
    'is_grad_enabled', 'set_grad_enabled', 'no_grad', 'enable_grad',
    'Node', 'AccumulateGrad',
    'backward',
    'functions',
]
```

**Step 5: Run tests to verify they pass**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_autograd.py -v
```

Expected: PASS (all autograd tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_autograd/functions.py src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/_autograd/__init__.py
git commit -m "feat(v2): add autograd dispatch integration with backward functions"
```

---

## Task 5: Tensor Hooks

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/test_autograd.py
def test_register_hook():
    """Can register gradient hooks on tensors."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    hook_called = [False]
    hook_grad = [None]

    def hook(grad):
        hook_called[0] = True
        hook_grad[0] = grad.numpy().copy()
        return grad

    x.register_hook(hook)

    y = (x * 2).sum()
    y.backward()

    assert hook_called[0] == True
    import numpy as np
    np.testing.assert_array_almost_equal(hook_grad[0], [2.0, 2.0])


def test_register_hook_modify_grad():
    """Hook can modify gradients."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    def double_grad(grad):
        return grad * 2

    x.register_hook(double_grad)

    y = x.sum()
    y.backward()

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])  # Doubled


def test_remove_hook():
    """Can remove hooks."""
    x = torch.tensor([1.0], requires_grad=True)

    call_count = [0]
    def hook(grad):
        call_count[0] += 1
        return grad

    handle = x.register_hook(hook)

    y = x.sum()
    y.backward(retain_graph=True)
    assert call_count[0] == 1

    handle.remove()

    y.backward()
    assert call_count[0] == 1  # Hook not called again
```

**Step 2: Run test to verify it fails**

**Step 3: Add hook support to Tensor**

Add to `src/mindtorch_v2/_tensor.py`:

```python
class RemovableHandle:
    """Handle returned by register_hook that allows removing the hook."""

    def __init__(self, hooks_dict, hook_id):
        self._hooks_dict = hooks_dict
        self._hook_id = hook_id

    def remove(self):
        if self._hook_id in self._hooks_dict:
            del self._hooks_dict[self._hook_id]
```

Add to Tensor class `__init__`:
```python
        self._hooks = {}  # Dict of hook_id -> hook_fn
        self._hook_counter = 0
```

Add to Tensor class (in autograd section):
```python
    def register_hook(self, hook):
        """Register a backward hook on the tensor.

        The hook will be called every time a gradient with respect to the
        tensor is computed. The hook should have the signature:
            hook(grad) -> Tensor or None

        Returns a handle that can be used to remove the hook.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that doesn't require gradient"
            )

        hook_id = self._hook_counter
        self._hook_counter += 1
        self._hooks[hook_id] = hook

        return RemovableHandle(self._hooks, hook_id)

    def _call_hooks(self, grad):
        """Call all registered hooks on a gradient."""
        for hook in self._hooks.values():
            result = hook(grad)
            if result is not None:
                grad = result
        return grad
```

**Step 4: Modify engine to call hooks**

Update `src/mindtorch_v2/_autograd/engine.py` AccumulateGrad handling:

```python
        # Handle AccumulateGrad specially
        if isinstance(node, AccumulateGrad):
            variable = node.variable

            # Call hooks
            if hasattr(variable, '_hooks') and variable._hooks:
                grad_output = variable._call_hooks(grad_output)

            if variable.grad is None:
                variable.grad = grad_output
            else:
                from .. import add
                variable.grad = add(variable.grad, grad_output)
            continue
```

**Step 5: Run test to verify it passes**

Expected: PASS (all tests including hook tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_autograd.py
git commit -m "feat(v2): add tensor gradient hooks (register_hook)"
```

---

## Task 6: Zero Grad and Detach

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Write the failing test**

```python
# Add to tests/mindtorch_v2/test_autograd.py
def test_zero_grad():
    """Tensor.zero_grad_() clears gradients."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x.sum()
    y.backward()

    assert x.grad is not None
    x.zero_grad_()
    assert x.grad is None


def test_detach():
    """Tensor.detach() returns tensor without grad_fn."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    assert y.grad_fn is not None
    z = y.detach()

    assert z.grad_fn is None
    assert z.requires_grad == False

    import numpy as np
    np.testing.assert_array_almost_equal(y.numpy(), z.numpy())


def test_detach_():
    """Tensor.detach_() detaches in-place."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    assert y.grad_fn is not None
    y.detach_()

    assert y.grad_fn is None
    assert y.requires_grad == False


def test_requires_grad_():
    """Tensor.requires_grad_() sets requires_grad in-place."""
    x = torch.tensor([1.0, 2.0])
    assert x.requires_grad == False

    x.requires_grad_(True)
    assert x.requires_grad == True

    y = x * 2
    y.sum().backward()

    assert x.grad is not None
```

**Step 2: Run test to verify it fails**

**Step 3: Implement zero_grad_, detach, detach_, requires_grad_**

Add to Tensor class in `src/mindtorch_v2/_tensor.py`:

```python
    def zero_grad_(self):
        """Zero the gradient of this tensor in-place."""
        self._grad = None
        return self

    def detach(self):
        """Returns a new Tensor detached from the current graph.

        The result will never require gradient.
        """
        return Tensor(
            _storage=self._storage,
            _shape=self._shape,
            _stride=self._stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=False,
        )

    def detach_(self):
        """Detach this tensor from the computation graph in-place.

        Sets requires_grad to False and clears grad_fn.
        """
        self._requires_grad = False
        self._grad_fn = None
        return self

    def requires_grad_(self, requires_grad: bool = True):
        """Change if this tensor requires gradients, in-place.

        Returns self.
        """
        self._requires_grad = requires_grad
        return self
```

**Step 4: Run test to verify it passes**

Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_autograd.py
git commit -m "feat(v2): add zero_grad_, detach, detach_, requires_grad_ methods"
```

---

## Task 7: Full Autograd Test Suite

**Files:**
- Modify: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Add comprehensive tests**

```python
# Add to tests/mindtorch_v2/test_autograd.py
def test_no_grad_prevents_recording():
    """Operations in no_grad context don't record grad_fn."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    with torch.no_grad():
        y = x * 2

    assert y.grad_fn is None
    assert y.requires_grad == False


def test_leaf_tensor_is_leaf():
    """Leaf tensors (created directly) are leaves."""
    x = torch.tensor([1.0], requires_grad=True)
    assert x.grad_fn is None  # Leaf

    y = x * 2
    assert y.grad_fn is not None  # Not leaf


def test_backward_non_scalar_requires_grad_tensors():
    """backward() on non-scalar requires grad_tensors argument."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2

    try:
        y.backward()
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "scalar" in str(e)

    # Should work with grad_tensors
    y.backward(gradient=torch.tensor([1.0, 1.0]))
    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [2.0, 2.0])


def test_gradient_accumulation():
    """Gradients accumulate across backward calls."""
    x = torch.tensor([1.0], requires_grad=True)

    for _ in range(3):
        y = x.sum()
        y.backward(retain_graph=True)

    import numpy as np
    np.testing.assert_array_almost_equal(x.grad.numpy(), [3.0])


def test_exp_backward():
    """Backward through exp."""
    x = torch.tensor([0.0, 1.0], requires_grad=True)
    y = torch.exp(x).sum()
    y.backward()

    import numpy as np
    expected = np.exp([0.0, 1.0])  # d(exp(x))/dx = exp(x)
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


def test_log_backward():
    """Backward through log."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = torch.log(x).sum()
    y.backward()

    import numpy as np
    expected = 1.0 / np.array([1.0, 2.0])  # d(log(x))/dx = 1/x
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


def test_sqrt_backward():
    """Backward through sqrt."""
    x = torch.tensor([1.0, 4.0], requires_grad=True)
    y = torch.sqrt(x).sum()
    y.backward()

    import numpy as np
    expected = 0.5 / np.sqrt([1.0, 4.0])  # d(sqrt(x))/dx = 1/(2*sqrt(x))
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected)


def test_matmul_backward():
    """Backward through matmul."""
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    c = torch.matmul(a, b)
    loss = c.sum()
    loss.backward()

    # Gradients should be ones @ b.T and a.T @ ones
    import numpy as np
    np.testing.assert_array_almost_equal(
        a.grad.numpy(),
        np.ones((2, 2)) @ np.array([[5.0, 7.0], [6.0, 8.0]])
    )


def test_complex_computation():
    """Backward through complex computation graph."""
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # y = x^2
    y = x * x
    # z = sum(y) / 3 = mean(x^2)
    z = y.sum() / 3.0
    z.backward()

    # d(mean(x^2))/dx = 2x/3
    import numpy as np
    expected = 2 * np.array([1.0, 2.0, 3.0]) / 3.0
    np.testing.assert_array_almost_equal(x.grad.numpy(), expected, decimal=5)
```

**Step 2: Run all tests**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_autograd.py -v
```

Expected: All tests PASS

**Step 3: Run full test suite to ensure no regressions**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/mindtorch_v2/test_autograd.py
git commit -m "test(v2): add comprehensive autograd test suite"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Grad Mode (no_grad, enable_grad) | 5 |
| 2 | Node Base Class | 3 |
| 3 | Backward Engine | 6 |
| 4 | Autograd Dispatch Integration | (covered by Task 3) |
| 5 | Tensor Hooks | 3 |
| 6 | Zero Grad and Detach | 4 |
| 7 | Comprehensive Tests | 10+ |

**Total estimated new tests:** ~30

**After Phase 3, mindtorch v2 will support:**
- Automatic differentiation for all Phase 2 ops
- `tensor.backward()` for computing gradients
- `torch.no_grad()` / `torch.enable_grad()` context managers
- Gradient hooks via `tensor.register_hook()`
- `tensor.detach()`, `tensor.zero_grad_()`, `tensor.requires_grad_()`
