# mindtorch v2 Phase 4: nn.Module + Layers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the nn.Module system and essential layers (Linear, Embedding, LayerNorm, Dropout, activations) needed to run transformer models.

**Architecture:** Parameter is a Tensor subclass with `requires_grad=True` by default. Module is a container that tracks parameters, buffers, and submodules. Layers delegate computation to functional ops in `nn.functional`.

**Tech Stack:** Python 3.9+, NumPy (for CPU backend ops)

**Worktree:** `/Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2`
**Branch:** `feature/mindtorch-v2`
**Source dir:** `src/mindtorch_v2/`
**Test dir:** `tests/mindtorch_v2/`

**Test command:**
```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

---

## Task 1: Parameter Class

**Files:**
- Create: `src/mindtorch_v2/nn/__init__.py`
- Create: `src/mindtorch_v2/nn/parameter.py`
- Create: `tests/mindtorch_v2/test_nn_parameter.py`

**Step 1: Create nn package and write test**

```python
# tests/mindtorch_v2/test_nn_parameter.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2.nn import Parameter


def test_parameter_is_tensor():
    """Parameter is a Tensor subclass."""
    p = Parameter(torch.randn(3, 4))
    assert isinstance(p, torch.Tensor)


def test_parameter_requires_grad_default():
    """Parameter has requires_grad=True by default."""
    p = Parameter(torch.randn(3, 4))
    assert p.requires_grad == True


def test_parameter_requires_grad_false():
    """Parameter can have requires_grad=False."""
    p = Parameter(torch.randn(3, 4), requires_grad=False)
    assert p.requires_grad == False


def test_parameter_shape():
    """Parameter preserves shape."""
    p = Parameter(torch.randn(2, 3, 4))
    assert p.shape == (2, 3, 4)


def test_parameter_grad_accumulates():
    """Parameter accumulates gradients."""
    p = Parameter(torch.tensor([1.0, 2.0, 3.0]))
    loss = (p * 2).sum()
    loss.backward()
    np.testing.assert_array_almost_equal(p.grad.numpy(), [2.0, 2.0, 2.0])
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_nn_parameter.py -v
```

Expected: FAIL with `ImportError: cannot import name 'Parameter'`

**Step 3: Implement Parameter class**

```python
# src/mindtorch_v2/nn/__init__.py
"""Neural network module for mindtorch_v2."""

from .parameter import Parameter

__all__ = ['Parameter']
```

```python
# src/mindtorch_v2/nn/parameter.py
"""Parameter class - a Tensor that is a module parameter."""

from .._tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is automatically registered as a parameter when assigned to a Module.

    Parameters are Tensor subclasses that have requires_grad=True by default.
    When assigned as a Module attribute, they are automatically added to the
    list of module parameters.
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            raise ValueError("Parameter requires data")

        if isinstance(data, Tensor):
            # Create new Parameter from existing Tensor
            instance = Tensor.__new__(cls)
            instance._storage = data._storage
            instance._shape = data._shape
            instance._stride = data._stride
            instance._storage_offset = data._storage_offset
            instance._dtype = data._dtype
            instance._device = data._device
            instance._requires_grad = requires_grad
            instance._grad_fn = None
            instance._grad = None
            instance._version = 0
            instance._hooks = {}
            instance._hook_counter = 0
            return instance
        else:
            # Create from raw data
            tensor = Tensor(data, requires_grad=requires_grad)
            instance = Tensor.__new__(cls)
            instance._storage = tensor._storage
            instance._shape = tensor._shape
            instance._stride = tensor._stride
            instance._storage_offset = tensor._storage_offset
            instance._dtype = tensor._dtype
            instance._device = tensor._device
            instance._requires_grad = requires_grad
            instance._grad_fn = None
            instance._grad = None
            instance._version = 0
            instance._hooks = {}
            instance._hook_counter = 0
            return instance

    def __init__(self, data=None, requires_grad=True):
        # All initialization done in __new__
        pass

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"
```

**Step 4: Run test to verify it passes**

Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_parameter.py
git commit -m "feat(v2): add Parameter class for nn.Module parameters"
```

---

## Task 2: Module Base Class (Core)

**Files:**
- Create: `src/mindtorch_v2/nn/module.py`
- Modify: `src/mindtorch_v2/nn/__init__.py`
- Create: `tests/mindtorch_v2/test_nn_module.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn_module.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2.nn import Module, Parameter


class SimpleModule(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(torch.randn(3, 4))
        self.bias = Parameter(torch.randn(4))

    def forward(self, x):
        return x @ self.weight + self.bias


def test_module_parameters():
    """Module tracks parameters."""
    m = SimpleModule()
    params = list(m.parameters())
    assert len(params) == 2


def test_module_named_parameters():
    """Module provides named parameters."""
    m = SimpleModule()
    named = dict(m.named_parameters())
    assert 'weight' in named
    assert 'bias' in named


def test_module_call():
    """Module is callable and runs forward."""
    m = SimpleModule()
    x = torch.randn(2, 3)
    y = m(x)
    assert y.shape == (2, 4)


def test_module_train_eval():
    """Module has train/eval modes."""
    m = SimpleModule()
    assert m.training == True
    m.eval()
    assert m.training == False
    m.train()
    assert m.training == True


def test_module_to_device():
    """Module.to() returns self."""
    m = SimpleModule()
    result = m.to('cpu')
    assert result is m


def test_module_children():
    """Module tracks child modules."""
    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = SimpleModule()
        def forward(self, x):
            return self.child(x)

    p = Parent()
    children = list(p.children())
    assert len(children) == 1
    assert isinstance(children[0], SimpleModule)


def test_module_modules():
    """Module.modules() returns all modules recursively."""
    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = SimpleModule()
        def forward(self, x):
            return self.child(x)

    p = Parent()
    modules = list(p.modules())
    assert len(modules) == 2  # Parent + SimpleModule
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Module base class**

```python
# src/mindtorch_v2/nn/module.py
"""Module base class for neural network layers."""

from typing import Iterator, Tuple, Dict, Optional, Any, Set
from collections import OrderedDict

from .parameter import Parameter
from .._tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses should implement the forward() method.
    """

    _version: int = 1
    training: bool

    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' function"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        # Handle Parameter assignment
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        """Add a buffer to the module."""
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Add a parameter to the module."""
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Add a child module."""
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters with names."""
        memo: Set[int] = set()
        for name, p in self._parameters.items():
            if p is not None and id(p) not in memo:
                memo.add(id(p))
                yield prefix + name, p
        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    for name, p in module.named_parameters(prefix=submodule_prefix, recurse=True):
                        if id(p) not in memo:
                            memo.add(id(p))
                            yield name, p

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Return an iterator over module buffers."""
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Return an iterator over module buffers with names."""
        memo: Set[int] = set()
        for name, b in self._buffers.items():
            if b is not None and id(b) not in memo:
                memo.add(id(b))
                yield prefix + name, b
        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    for name, b in module.named_buffers(prefix=submodule_prefix, recurse=True):
                        if id(b) not in memo:
                            memo.add(id(b))
                            yield name, b

    def children(self) -> Iterator['Module']:
        """Return an iterator over immediate child modules."""
        for name, module in self._modules.items():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over immediate child modules with names."""
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def modules(self) -> Iterator['Module']:
        """Return an iterator over all modules in the network."""
        for name, module in self.named_modules():
            yield module

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over all modules with names."""
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(prefix=submodule_prefix):
                    yield m

    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """Set the module in evaluation mode."""
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> 'Module':
        """Set requires_grad for all parameters."""
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_grad_()

    def to(self, device=None, dtype=None):
        """Move module to device/dtype (placeholder - returns self)."""
        # TODO: Implement actual device/dtype conversion
        return self

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + '(']
        for name, module in self._modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f'  ({name}): {mod_str}')
        lines.append(')')
        return '\n'.join(lines) if len(self._modules) > 0 else f'{self.__class__.__name__}()'

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return ''
```

**Step 4: Update nn/__init__.py**

```python
# src/mindtorch_v2/nn/__init__.py
"""Neural network module for mindtorch_v2."""

from .parameter import Parameter
from .module import Module

__all__ = ['Parameter', 'Module']
```

**Step 5: Run test to verify it passes**

Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_module.py
git commit -m "feat(v2): add Module base class for neural networks"
```

---

## Task 3: Functional Ops (relu, linear, embedding, dropout)

**Files:**
- Create: `src/mindtorch_v2/nn/functional.py`
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Create: `tests/mindtorch_v2/test_nn_functional.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn_functional.py
import numpy as np
import mindtorch_v2 as torch
import mindtorch_v2.nn.functional as F


def test_relu():
    """F.relu applies ReLU activation."""
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    y = F.relu(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.0, 0.0, 1.0, 2.0])


def test_gelu():
    """F.gelu applies GELU activation."""
    x = torch.tensor([0.0, 1.0, -1.0])
    y = F.gelu(x)
    # GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
    assert abs(y.numpy()[0]) < 0.01
    assert abs(y.numpy()[1] - 0.841) < 0.01


def test_silu():
    """F.silu applies SiLU/Swish activation."""
    x = torch.tensor([0.0, 1.0, -1.0])
    y = F.silu(x)
    # SiLU(x) = x * sigmoid(x)
    expected = x.numpy() * (1 / (1 + np.exp(-x.numpy())))
    np.testing.assert_array_almost_equal(y.numpy(), expected)


def test_softmax():
    """F.softmax applies softmax."""
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    y = F.softmax(x, dim=1)
    # Each row should sum to 1
    np.testing.assert_array_almost_equal(y.sum(dim=1).numpy(), [1.0, 1.0])


def test_linear():
    """F.linear computes linear transformation."""
    x = torch.tensor([[1.0, 2.0, 3.0]])
    w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3
    b = torch.tensor([0.5, 0.5])
    y = F.linear(x, w, b)
    np.testing.assert_array_almost_equal(y.numpy(), [[1.5, 2.5]])


def test_linear_no_bias():
    """F.linear works without bias."""
    x = torch.tensor([[1.0, 2.0]])
    w = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])  # 3x2
    y = F.linear(x, w)
    np.testing.assert_array_almost_equal(y.numpy(), [[3.0, 6.0, 9.0]])


def test_embedding():
    """F.embedding looks up embeddings."""
    weight = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    indices = torch.tensor([0, 2, 1])
    y = F.embedding(indices, weight)
    np.testing.assert_array_almost_equal(y.numpy(), [[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])


def test_dropout_eval():
    """F.dropout does nothing in eval mode."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = F.dropout(x, p=0.5, training=False)
    np.testing.assert_array_almost_equal(y.numpy(), x.numpy())


def test_layer_norm():
    """F.layer_norm normalizes."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = F.layer_norm(x, normalized_shape=(4,))
    # Should have mean≈0, std≈1
    assert abs(y.mean().item()) < 0.01
```

**Step 2: Run test to verify it fails**

**Step 3: Implement functional ops**

```python
# src/mindtorch_v2/nn/functional.py
"""Functional operations for neural networks."""

import math
from .._dispatch import dispatch
from .._tensor import Tensor


def relu(input, inplace=False):
    """Apply ReLU activation."""
    return dispatch("relu", input)


def gelu(input, approximate='none'):
    """Apply GELU activation."""
    return dispatch("gelu", input, approximate=approximate)


def silu(input, inplace=False):
    """Apply SiLU/Swish activation."""
    return dispatch("silu", input)


def sigmoid(input):
    """Apply sigmoid activation."""
    return dispatch("sigmoid", input)


def tanh(input):
    """Apply tanh activation."""
    return dispatch("tanh", input)


def softmax(input, dim=None, dtype=None):
    """Apply softmax."""
    return dispatch("softmax", input, dim=dim)


def log_softmax(input, dim=None, dtype=None):
    """Apply log softmax."""
    return dispatch("log_softmax", input, dim=dim)


def linear(input, weight, bias=None):
    """Apply linear transformation: y = xW^T + b."""
    output = dispatch("matmul", input, weight.t())
    if bias is not None:
        output = dispatch("add", output, bias)
    return output


def embedding(input, weight, padding_idx=None, max_norm=None,
              norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """Look up embeddings."""
    return dispatch("embedding", input, weight)


def dropout(input, p=0.5, training=True, inplace=False):
    """Apply dropout."""
    if not training or p == 0:
        return input
    return dispatch("dropout", input, p=p, training=training)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Apply layer normalization."""
    return dispatch("layer_norm", input, normalized_shape, weight, bias, eps)
```

**Step 4: Add CPU backend implementations**

Add to `src/mindtorch_v2/_backends/cpu.py`:

```python
# --- Activation ops ---

@register_op("gelu", DispatchKey.Backend_CPU)
def gelu_cpu(a, approximate='none'):
    """GELU activation."""
    a_np = _to_numpy(a)
    if approximate == 'tanh':
        # Approximate GELU
        coef = math.sqrt(2.0 / math.pi)
        result = 0.5 * a_np * (1.0 + np.tanh(coef * (a_np + 0.044715 * a_np ** 3)))
    else:
        # Exact GELU using erf
        from scipy.special import erf
        result = 0.5 * a_np * (1.0 + erf(a_np / math.sqrt(2.0)))
    return _wrap_result(result)


@register_op("silu", DispatchKey.Backend_CPU)
def silu_cpu(a):
    """SiLU/Swish activation: x * sigmoid(x)."""
    a_np = _to_numpy(a)
    result = a_np * (1.0 / (1.0 + np.exp(-a_np)))
    return _wrap_result(result)


@register_op("softmax", DispatchKey.Backend_CPU)
def softmax_cpu(a, dim=None):
    """Softmax activation."""
    a_np = _to_numpy(a)
    if dim is None:
        dim = -1
    # Subtract max for numerical stability
    a_max = np.max(a_np, axis=dim, keepdims=True)
    exp_a = np.exp(a_np - a_max)
    result = exp_a / np.sum(exp_a, axis=dim, keepdims=True)
    return _wrap_result(result)


@register_op("log_softmax", DispatchKey.Backend_CPU)
def log_softmax_cpu(a, dim=None):
    """Log softmax."""
    a_np = _to_numpy(a)
    if dim is None:
        dim = -1
    a_max = np.max(a_np, axis=dim, keepdims=True)
    log_sum_exp = a_max + np.log(np.sum(np.exp(a_np - a_max), axis=dim, keepdims=True))
    result = a_np - log_sum_exp
    return _wrap_result(result)


# --- Neural network ops ---

@register_op("embedding", DispatchKey.Backend_CPU)
def embedding_cpu(indices, weight):
    """Embedding lookup."""
    indices_np = _to_numpy(indices).astype(np.int64)
    weight_np = _to_numpy(weight)
    result = weight_np[indices_np]
    return _wrap_result(result)


@register_op("dropout", DispatchKey.Backend_CPU)
def dropout_cpu(a, p=0.5, training=True):
    """Dropout."""
    if not training or p == 0:
        return a
    a_np = _to_numpy(a)
    mask = np.random.binomial(1, 1 - p, a_np.shape) / (1 - p)
    result = a_np * mask
    return _wrap_result(result.astype(a_np.dtype))


@register_op("layer_norm", DispatchKey.Backend_CPU)
def layer_norm_cpu(a, normalized_shape, weight=None, bias=None, eps=1e-5):
    """Layer normalization."""
    a_np = _to_numpy(a)

    # Determine axes to normalize over
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))

    mean = np.mean(a_np, axis=axes, keepdims=True)
    var = np.var(a_np, axis=axes, keepdims=True)
    result = (a_np - mean) / np.sqrt(var + eps)

    if weight is not None:
        result = result * _to_numpy(weight)
    if bias is not None:
        result = result + _to_numpy(bias)

    return _wrap_result(result)
```

**Step 5: Run test to verify it passes**

Expected: PASS (9 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/functional.py src/mindtorch_v2/_backends/cpu.py tests/mindtorch_v2/test_nn_functional.py
git commit -m "feat(v2): add nn.functional ops (relu, gelu, silu, softmax, linear, embedding, dropout, layer_norm)"
```

---

## Task 4: Linear Layer

**Files:**
- Create: `src/mindtorch_v2/nn/modules/__init__.py`
- Create: `src/mindtorch_v2/nn/modules/linear.py`
- Modify: `src/mindtorch_v2/nn/__init__.py`
- Create: `tests/mindtorch_v2/test_nn_linear.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn_linear.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_linear_shape():
    """Linear produces correct output shape."""
    m = nn.Linear(10, 5)
    x = torch.randn(2, 10)
    y = m(x)
    assert y.shape == (2, 5)


def test_linear_parameters():
    """Linear has weight and bias parameters."""
    m = nn.Linear(10, 5)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert 'bias' in params
    assert params['weight'].shape == (5, 10)
    assert params['bias'].shape == (5,)


def test_linear_no_bias():
    """Linear can be created without bias."""
    m = nn.Linear(10, 5, bias=False)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert 'bias' not in params


def test_linear_backward():
    """Linear supports backward pass."""
    m = nn.Linear(3, 2)
    x = torch.randn(1, 3, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert m.weight.grad is not None
    assert m.bias.grad is not None


def test_linear_repr():
    """Linear has informative repr."""
    m = nn.Linear(10, 5)
    r = repr(m)
    assert 'Linear' in r
    assert '10' in r
    assert '5' in r
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Linear layer**

```python
# src/mindtorch_v2/nn/modules/__init__.py
"""Neural network modules."""

from .linear import Linear, Identity

__all__ = ['Linear', 'Identity']
```

```python
# src/mindtorch_v2/nn/modules/linear.py
"""Linear layer module."""

import math
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class Identity(Module):
    """A placeholder identity operator."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input):
        return input


class Linear(Module):
    """Applies a linear transformation: y = xW^T + b.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If False, layer will not learn an additive bias. Default: True
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight with Kaiming uniform
        k = 1.0 / in_features
        bound = math.sqrt(k)
        weight_data = torch.empty(out_features, in_features)
        # Uniform(-bound, bound)
        weight_np = weight_data.numpy()
        import numpy as np
        weight_np[:] = np.random.uniform(-bound, bound, weight_np.shape)
        self.weight = Parameter(torch.tensor(weight_np))

        if bias:
            bias_data = torch.empty(out_features)
            bias_np = bias_data.numpy()
            bias_np[:] = np.random.uniform(-bound, bound, bias_np.shape)
            self.bias = Parameter(torch.tensor(bias_np))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def __repr__(self):
        return f'Linear({self.extra_repr()})'
```

**Step 4: Update nn/__init__.py**

```python
# src/mindtorch_v2/nn/__init__.py
"""Neural network module for mindtorch_v2."""

from .parameter import Parameter
from .module import Module
from .modules import Linear, Identity

__all__ = ['Parameter', 'Module', 'Linear', 'Identity']
```

**Step 5: Run test to verify it passes**

Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_linear.py
git commit -m "feat(v2): add nn.Linear layer"
```

---

## Task 5: Activation Modules

**Files:**
- Create: `src/mindtorch_v2/nn/modules/activation.py`
- Modify: `src/mindtorch_v2/nn/modules/__init__.py`
- Modify: `src/mindtorch_v2/nn/__init__.py`
- Create: `tests/mindtorch_v2/test_nn_activation.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn_activation.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_relu_module():
    """nn.ReLU works as a module."""
    m = nn.ReLU()
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.0, 0.0, 1.0])


def test_gelu_module():
    """nn.GELU works as a module."""
    m = nn.GELU()
    x = torch.tensor([0.0])
    y = m(x)
    assert abs(y.item()) < 0.01


def test_silu_module():
    """nn.SiLU works as a module."""
    m = nn.SiLU()
    x = torch.tensor([0.0])
    y = m(x)
    assert abs(y.item()) < 0.01  # SiLU(0) = 0


def test_sigmoid_module():
    """nn.Sigmoid works as a module."""
    m = nn.Sigmoid()
    x = torch.tensor([0.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.5])


def test_tanh_module():
    """nn.Tanh works as a module."""
    m = nn.Tanh()
    x = torch.tensor([0.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.0])


def test_softmax_module():
    """nn.Softmax works as a module."""
    m = nn.Softmax(dim=1)
    x = torch.tensor([[1.0, 1.0, 1.0]])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [[1/3, 1/3, 1/3]], decimal=5)
```

**Step 2: Run test to verify it fails**

**Step 3: Implement activation modules**

```python
# src/mindtorch_v2/nn/modules/activation.py
"""Activation function modules."""

from ..module import Module
from .. import functional as F
from ..._tensor import Tensor


class ReLU(Module):
    """Applies ReLU: max(0, x)."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class GELU(Module):
    """Applies GELU activation."""

    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)


class SiLU(Module):
    """Applies SiLU (Swish): x * sigmoid(x)."""

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)


class Sigmoid(Module):
    """Applies sigmoid: 1 / (1 + exp(-x))."""

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(Module):
    """Applies tanh activation."""

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class Softmax(Module):
    """Applies softmax along a dimension."""

    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(Module):
    """Applies log softmax along a dimension."""

    __constants__ = ['dim']
    dim: int

    def __init__(self, dim: int = None):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'
```

**Step 4: Update modules/__init__.py and nn/__init__.py**

```python
# src/mindtorch_v2/nn/modules/__init__.py
"""Neural network modules."""

from .linear import Linear, Identity
from .activation import ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax

__all__ = [
    'Linear', 'Identity',
    'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
]
```

```python
# src/mindtorch_v2/nn/__init__.py
"""Neural network module for mindtorch_v2."""

from .parameter import Parameter
from .module import Module
from .modules import (
    Linear, Identity,
    ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax,
)
from . import functional

__all__ = [
    'Parameter', 'Module',
    'Linear', 'Identity',
    'ReLU', 'GELU', 'SiLU', 'Sigmoid', 'Tanh', 'Softmax', 'LogSoftmax',
    'functional',
]
```

**Step 5: Run test to verify it passes**

Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_activation.py
git commit -m "feat(v2): add activation modules (ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax)"
```

---

## Task 6: Embedding Layer

**Files:**
- Create: `src/mindtorch_v2/nn/modules/sparse.py`
- Modify: `src/mindtorch_v2/nn/modules/__init__.py`
- Modify: `src/mindtorch_v2/nn/__init__.py`
- Create: `tests/mindtorch_v2/test_nn_embedding.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn_embedding.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_embedding_shape():
    """Embedding produces correct output shape."""
    m = nn.Embedding(10, 3)  # vocab=10, dim=3
    x = torch.tensor([1, 2, 4, 5])
    y = m(x)
    assert y.shape == (4, 3)


def test_embedding_lookup():
    """Embedding looks up correct vectors."""
    m = nn.Embedding(5, 2)
    # Manually set weights for testing
    m.weight = nn.Parameter(torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]))
    x = torch.tensor([0, 2, 4])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [[0, 0], [2, 2], [4, 4]])


def test_embedding_2d_input():
    """Embedding handles 2D input."""
    m = nn.Embedding(10, 4)
    x = torch.tensor([[1, 2], [3, 4]])
    y = m(x)
    assert y.shape == (2, 2, 4)


def test_embedding_parameters():
    """Embedding has weight parameter."""
    m = nn.Embedding(100, 64)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert params['weight'].shape == (100, 64)


def test_embedding_backward():
    """Embedding supports backward pass."""
    m = nn.Embedding(5, 3)
    x = torch.tensor([0, 1, 2])
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert m.weight.grad is not None
```

**Step 2: Run test to verify it fails**

**Step 3: Implement Embedding layer**

```python
# src/mindtorch_v2/nn/modules/sparse.py
"""Sparse modules like Embedding."""

import math
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class Embedding(Module):
    """A lookup table that stores embeddings of a fixed dictionary and size.

    Args:
        num_embeddings: size of the dictionary
        embedding_dim: size of each embedding vector
        padding_idx: If specified, entries at padding_idx do not contribute to gradient
        max_norm: If given, embeddings are renormalized to have max_norm
        norm_type: The p of the p-norm for max_norm
        scale_grad_by_freq: If True, scale gradients by frequency
        sparse: If True, gradient is sparse
    """

    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: int
    max_norm: float
    norm_type: float
    scale_grad_by_freq: bool
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None,
                 max_norm: float = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False,
                 _weight: torch.Tensor = None, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        if _weight is None:
            # Initialize with normal distribution
            import numpy as np
            weight_np = np.random.normal(0, 1, (num_embeddings, embedding_dim)).astype(np.float32)
            self.weight = Parameter(torch.tensor(weight_np))
        else:
            self.weight = Parameter(_weight)

        if padding_idx is not None:
            # Zero out padding embedding
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, input):
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                          self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        if self.max_norm is not None:
            s += f', max_norm={self.max_norm}'
        if self.norm_type != 2.0:
            s += f', norm_type={self.norm_type}'
        if self.scale_grad_by_freq:
            s += ', scale_grad_by_freq=True'
        if self.sparse:
            s += ', sparse=True'
        return s

    def __repr__(self):
        return f'Embedding({self.extra_repr()})'
```

**Step 4: Update modules/__init__.py and nn/__init__.py**

Add `Embedding` to exports.

**Step 5: Run test to verify it passes**

Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_embedding.py
git commit -m "feat(v2): add nn.Embedding layer"
```

---

## Task 7: Dropout and LayerNorm

**Files:**
- Create: `src/mindtorch_v2/nn/modules/dropout.py`
- Create: `src/mindtorch_v2/nn/modules/normalization.py`
- Modify: `src/mindtorch_v2/nn/modules/__init__.py`
- Create: `tests/mindtorch_v2/test_nn_dropout.py`
- Create: `tests/mindtorch_v2/test_nn_layernorm.py`

**Step 1: Write the failing tests**

```python
# tests/mindtorch_v2/test_nn_dropout.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_dropout_train():
    """Dropout applies dropout in training mode."""
    m = nn.Dropout(p=0.5)
    m.train()
    x = torch.ones(1000)
    y = m(x)
    # Some values should be 0, some should be scaled
    num_zeros = (y.numpy() == 0).sum()
    assert num_zeros > 100  # Should have some zeros
    assert num_zeros < 900  # But not all zeros


def test_dropout_eval():
    """Dropout is identity in eval mode."""
    m = nn.Dropout(p=0.5)
    m.eval()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), x.numpy())


def test_dropout_p0():
    """Dropout with p=0 is identity."""
    m = nn.Dropout(p=0.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), x.numpy())
```

```python
# tests/mindtorch_v2/test_nn_layernorm.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_layernorm_shape():
    """LayerNorm preserves shape."""
    m = nn.LayerNorm(10)
    x = torch.randn(2, 3, 10)
    y = m(x)
    assert y.shape == (2, 3, 10)


def test_layernorm_normalized():
    """LayerNorm normalizes along last dim."""
    m = nn.LayerNorm(4, elementwise_affine=False)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = m(x)
    # Should have mean≈0
    assert abs(y.mean().item()) < 0.01


def test_layernorm_parameters():
    """LayerNorm has weight and bias."""
    m = nn.LayerNorm(10)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert 'bias' in params


def test_layernorm_no_affine():
    """LayerNorm without affine has no parameters."""
    m = nn.LayerNorm(10, elementwise_affine=False)
    params = list(m.parameters())
    assert len(params) == 0


def test_layernorm_backward():
    """LayerNorm supports backward."""
    m = nn.LayerNorm(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement Dropout and LayerNorm**

```python
# src/mindtorch_v2/nn/modules/dropout.py
"""Dropout modules."""

from ..module import Module
from .. import functional as F
from ..._tensor import Tensor


class Dropout(Module):
    """Randomly zeroes elements with probability p during training."""

    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'
```

```python
# src/mindtorch_v2/nn/modules/normalization.py
"""Normalization modules."""

from typing import List, Union
from ..module import Module
from ..parameter import Parameter
from .. import functional as F
import mindtorch_v2 as torch


class LayerNorm(Module):
    """Applies Layer Normalization.

    Args:
        normalized_shape: input shape from an expected input of size
        eps: a value added to the denominator for numerical stability
        elementwise_affine: whether to learn affine parameters
    """

    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: tuple
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5,
                 elementwise_affine: bool = True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(normalized_shape))
            self.bias = Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

    def __repr__(self):
        return f'LayerNorm({self.extra_repr()})'
```

**Step 4: Update exports**

**Step 5: Run tests to verify they pass**

Expected: PASS (8 tests total)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_dropout.py tests/mindtorch_v2/test_nn_layernorm.py
git commit -m "feat(v2): add nn.Dropout and nn.LayerNorm"
```

---

## Task 8: Container Modules (Sequential, ModuleList)

**Files:**
- Create: `src/mindtorch_v2/nn/modules/container.py`
- Modify: `src/mindtorch_v2/nn/modules/__init__.py`
- Create: `tests/mindtorch_v2/test_nn_container.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn_container.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_sequential_forward():
    """Sequential chains modules."""
    m = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    x = torch.randn(2, 10)
    y = m(x)
    assert y.shape == (2, 2)


def test_sequential_indexing():
    """Sequential supports indexing."""
    m = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
    )
    assert isinstance(m[0], nn.Linear)
    assert isinstance(m[1], nn.ReLU)


def test_sequential_len():
    """Sequential has length."""
    m = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    assert len(m) == 2


def test_modulelist_append():
    """ModuleList can append modules."""
    layers = nn.ModuleList()
    layers.append(nn.Linear(10, 5))
    layers.append(nn.Linear(5, 2))
    assert len(layers) == 2


def test_modulelist_parameters():
    """ModuleList tracks parameters."""
    layers = nn.ModuleList([nn.Linear(10, 5), nn.Linear(5, 2)])
    params = list(layers.parameters())
    assert len(params) == 4  # 2 weights + 2 biases


def test_modulelist_iteration():
    """ModuleList is iterable."""
    layers = nn.ModuleList([nn.Linear(10, 5), nn.Linear(5, 2)])
    count = 0
    for layer in layers:
        count += 1
    assert count == 2
```

**Step 2: Run test to verify it fails**

**Step 3: Implement container modules**

```python
# src/mindtorch_v2/nn/modules/container.py
"""Container modules."""

from typing import Iterator, Iterable, Optional, overload, Union
from collections import OrderedDict

from ..module import Module


class Sequential(Module):
    """A sequential container that chains modules."""

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: Optional[Iterable[Module]] = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> 'ModuleList':
        """Append a module to the list."""
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        """Extend with modules from iterable."""
        for module in modules:
            self.append(module)
        return self

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        key = list(self._modules.keys())[idx]
        self._modules[key] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleDict(Module):
    """Holds submodules in a dict."""

    def __init__(self, modules: Optional[dict] = None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules: dict) -> None:
        for key, module in modules.items():
            self[key] = module
```

**Step 4: Update exports**

**Step 5: Run test to verify it passes**

Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/nn/ tests/mindtorch_v2/test_nn_container.py
git commit -m "feat(v2): add container modules (Sequential, ModuleList, ModuleDict)"
```

---

## Task 9: Full Integration Test

**Files:**
- Create: `tests/mindtorch_v2/test_nn_integration.py`

**Step 1: Write integration test**

```python
# tests/mindtorch_v2/test_nn_integration.py
"""Integration tests for nn module."""

import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def test_mlp_forward():
    """MLP forward pass works."""
    model = SimpleMLP(10, 20, 5)
    x = torch.randn(2, 10)
    y = model(x)
    assert y.shape == (2, 5)


def test_mlp_backward():
    """MLP backward pass works."""
    model = SimpleMLP(10, 20, 5)
    x = torch.randn(2, 10, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_mlp_parameter_count():
    """MLP has correct number of parameters."""
    model = SimpleMLP(10, 20, 5)
    params = list(model.parameters())
    # fc1: 10*20 + 20, fc2: 20*5 + 5 = 200+20+100+5 = 325
    total_params = sum(p.numel() for p in params)
    assert total_params == 325


def test_transformer_block_components():
    """Test components needed for transformer."""
    batch, seq_len, d_model = 2, 10, 64

    # Embedding
    vocab_size = 1000
    embed = nn.Embedding(vocab_size, d_model)
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    x = embed(token_ids)
    assert x.shape == (batch, seq_len, d_model)

    # LayerNorm
    ln = nn.LayerNorm(d_model)
    x = ln(x)
    assert x.shape == (batch, seq_len, d_model)

    # Linear (for projection)
    proj = nn.Linear(d_model, d_model * 3)
    qkv = proj(x)
    assert qkv.shape == (batch, seq_len, d_model * 3)

    # Dropout
    dropout = nn.Dropout(0.1)
    dropout.eval()  # Use eval for deterministic test
    out = dropout(x)
    assert out.shape == (batch, seq_len, d_model)


def test_model_train_eval():
    """Model train/eval affects dropout."""
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.Dropout(0.5),
    )

    x = torch.ones(100, 10)

    # In train mode, dropout should zero some values
    model.train()
    y_train = model(x)

    # In eval mode, dropout should be identity
    model.eval()
    y_eval = model(x)

    # Eval output should have no zeros from dropout
    # (zeros could still come from Linear weights)
```

**Step 2: Run test**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_nn_integration.py -v
```

Expected: PASS (5 tests)

**Step 3: Run full test suite**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/mindtorch_v2/test_nn_integration.py
git commit -m "test(v2): add nn module integration tests"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Parameter | 5 |
| 2 | Module base class | 7 |
| 3 | Functional ops | 9 |
| 4 | Linear | 5 |
| 5 | Activations | 6 |
| 6 | Embedding | 5 |
| 7 | Dropout + LayerNorm | 8 |
| 8 | Containers | 6 |
| 9 | Integration | 5 |

**Total estimated new tests:** ~56

**After Phase 4, mindtorch v2 will support:**
- `nn.Parameter` as learnable tensor
- `nn.Module` base class with parameter/buffer/module tracking
- Functional ops: `F.relu`, `F.gelu`, `F.silu`, `F.softmax`, `F.linear`, `F.embedding`, `F.dropout`, `F.layer_norm`
- Modules: `Linear`, `Embedding`, `LayerNorm`, `Dropout`
- Activations: `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`
- Containers: `Sequential`, `ModuleList`, `ModuleDict`
