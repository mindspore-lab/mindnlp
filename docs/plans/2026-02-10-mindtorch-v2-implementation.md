# Mindtorch v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `mindtorch_v2` as a minimal PyTorch-like core with CPU (NumPy) and NPU (pyacl/ACLNN) backends, including Tensor/Storage/Autograd, broadcasting, and minimal nn/optim.

**Architecture:** A small Tensor/Storage core with view/stride metadata, a dispatcher routing ops to CPU or NPU backends, and a tape-based autograd engine with broadcast-aware gradient reduction. Minimal nn/optim layers build on the functional core.

**Tech Stack:** Python 3.12, NumPy, pyacl/ACLNN (Ascend), pytest.

---

### Task 1: Reset `mindtorch_v2` package to a clean skeleton

**Files:**
- Remove: `src/mindtorch_v2/`
- Create: `src/mindtorch_v2/__init__.py`
- Create: `src/mindtorch_v2/_autograd/__init__.py`
- Create: `src/mindtorch_v2/_backends/__init__.py`
- Create: `src/mindtorch_v2/_dispatch/__init__.py`
- Create: `src/mindtorch_v2/nn/__init__.py`
- Create: `src/mindtorch_v2/optim/__init__.py`
- Test: `tests/mindtorch_v2/test_import.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_import.py
import mindtorch_v2 as torch

def test_import_has_tensor():
    assert hasattr(torch, "tensor")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_import.py::test_import_has_tensor -v`
Expected: FAIL (module or attribute missing)

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/__init__.py
__version__ = "0.1.0"

def tensor(*args, **kwargs):
    raise NotImplementedError("mindtorch_v2 core not implemented yet")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_import.py::test_import_has_tensor -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/__init__.py src/mindtorch_v2/_autograd/__init__.py src/mindtorch_v2/_backends/__init__.py src/mindtorch_v2/_dispatch/__init__.py src/mindtorch_v2/nn/__init__.py src/mindtorch_v2/optim/__init__.py tests/mindtorch_v2/test_import.py
git commit -m "feat(mindtorch_v2): reset package skeleton"
```

---

### Task 2: DType + device primitives (CPU only)

**Files:**
- Create: `src/mindtorch_v2/_dtype.py`
- Create: `src/mindtorch_v2/_device.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Test: `tests/mindtorch_v2/test_dtype_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_dtype_device.py
import mindtorch_v2 as torch

def test_default_dtype_and_device():
    x = torch.tensor([1, 2, 3])
    assert x.dtype == torch.float32
    assert x.device.type == "cpu"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_dtype_device.py::test_default_dtype_and_device -v`
Expected: FAIL (dtype/device not implemented)

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_dtype.py
class DType:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"torch.{self.name}"

float32 = DType("float32")
float16 = DType("float16")
int64 = DType("int64")

# src/mindtorch_v2/_device.py
class device:
    def __init__(self, dev):
        self.type = dev
    def __repr__(self):
        return f"{self.type}"

_default_device = device("cpu")

# src/mindtorch_v2/__init__.py
from ._dtype import DType, float32, float16, int64
from ._device import device, _default_device
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_dtype_device.py::test_default_dtype_and_device -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dtype.py src/mindtorch_v2/_device.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_dtype_device.py
git commit -m "feat(mindtorch_v2): add dtype and device primitives"
```

---

### Task 3: Storage + Tensor basics with view/stride

**Files:**
- Create: `src/mindtorch_v2/_storage.py`
- Create: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Test: `tests/mindtorch_v2/test_tensor_view.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_tensor_view.py
import mindtorch_v2 as torch
import numpy as np

def test_view_reshape_transpose_share_storage():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = x.reshape((3, 2))
    z = x.transpose(0, 1)
    assert y.storage is x.storage
    assert z.storage is x.storage
    assert y.shape == (3, 2)
    assert z.shape == (3, 2)
    # offset/stride sanity
    assert x.stride != z.stride
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_tensor_view.py::test_view_reshape_transpose_share_storage -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_storage.py
import numpy as np
from ._device import _default_device
from ._dtype import float32

class Storage:
    def __init__(self, data, device=None, dtype=None):
        self.device = device or _default_device
        self.dtype = dtype or float32
        self.data = np.array(data, dtype=np.float32, copy=False)

# src/mindtorch_v2/_tensor.py
from ._storage import Storage
from ._device import _default_device
from ._dtype import float32

class Tensor:
    def __init__(self, storage, shape, stride, offset=0, requires_grad=False):
        self.storage = storage
        self.shape = tuple(shape)
        self.stride = tuple(stride)
        self.offset = int(offset)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    @property
    def dtype(self):
        return self.storage.dtype

    @property
    def device(self):
        return self.storage.device

    def reshape(self, new_shape):
        # contiguous-only reshape for now (view with new stride)
        size = 1
        for d in self.shape:
            size *= d
        new_size = 1
        for d in new_shape:
            new_size *= d
        if size != new_size:
            raise ValueError("reshape size mismatch")
        # assume contiguous layout
        new_stride = []
        acc = 1
        for d in reversed(new_shape):
            new_stride.append(acc)
            acc *= d
        new_stride = tuple(reversed(new_stride))
        return Tensor(self.storage, new_shape, new_stride, self.offset, self.requires_grad)

    def transpose(self, dim0, dim1):
        shape = list(self.shape)
        stride = list(self.stride)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
        return Tensor(self.storage, shape, stride, self.offset, self.requires_grad)

# src/mindtorch_v2/__init__.py
from ._tensor import Tensor
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_tensor_view.py::test_view_reshape_transpose_share_storage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_storage.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_tensor_view.py
git commit -m "feat(mindtorch_v2): add storage and tensor views"
```

---

### Task 4: Tensor creation ops (CPU)

**Files:**
- Create: `src/mindtorch_v2/_creation.py`
- Modify: `src/mindtorch_v2/_storage.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Test: `tests/mindtorch_v2/test_creation.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_creation.py
import mindtorch_v2 as torch


def test_creation_ops():
    x = torch.zeros((2, 3))
    y = torch.ones((2, 3))
    assert x.shape == (2, 3)
    assert y.shape == (2, 3)
    assert x.storage.data.sum() == 0
    assert y.storage.data.sum() == 6
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_creation.py::test_creation_ops -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_creation.py
import numpy as np
from ._storage import Storage
from ._tensor import Tensor
from ._dtype import float32


def tensor(data, dtype=float32):
    arr = np.array(data, dtype=np.float32)
    storage = Storage(arr, dtype=dtype)
    shape = arr.shape
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, shape, stride)


def zeros(shape, dtype=float32):
    arr = np.zeros(shape, dtype=np.float32)
    storage = Storage(arr, dtype=dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def ones(shape, dtype=float32):
    arr = np.ones(shape, dtype=np.float32)
    storage = Storage(arr, dtype=dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)

# src/mindtorch_v2/__init__.py
from ._creation import tensor, zeros, ones
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_creation.py::test_creation_ops -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_creation.py src/mindtorch_v2/_storage.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_creation.py
git commit -m "feat(mindtorch_v2): add tensor creation ops"
```

---

### Task 5: Dispatcher + CPU backend ops (add/mul/matmul/relu/sum)

**Files:**
- Create: `src/mindtorch_v2/_dispatch/registry.py`
- Create: `src/mindtorch_v2/_dispatch/dispatcher.py`
- Create: `src/mindtorch_v2/_backends/cpu.py`
- Create: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Test: `tests/mindtorch_v2/test_ops_cpu.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_cpu.py
import mindtorch_v2 as torch


def test_add_mul_matmul_relu_sum():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[2.0, 0.5], [1.0, -1.0]])
    c = torch.add(a, b)
    d = torch.mul(a, b)
    e = torch.matmul(a, b)
    f = torch.relu(torch.tensor([-1.0, 2.0]))
    s = torch.sum(a, dim=1, keepdim=True)
    assert c.shape == (2, 2)
    assert d.shape == (2, 2)
    assert e.shape == (2, 2)
    assert f.shape == (2,)
    assert s.shape == (2, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_add_mul_matmul_relu_sum -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_dispatch/registry.py
class OpRegistry:
    def __init__(self):
        self._ops = {}

    def register(self, name, device, fn):
        self._ops[(name, device)] = fn

    def get(self, name, device):
        return self._ops[(name, device)]

registry = OpRegistry()

# src/mindtorch_v2/_dispatch/dispatcher.py
from .registry import registry

def dispatch(name, device, *args, **kwargs):
    fn = registry.get(name, device)
    return fn(*args, **kwargs)

# src/mindtorch_v2/_backends/cpu.py
import numpy as np
from .._tensor import Tensor
from .._storage import Storage
from .._dtype import float32
from .._dispatch.registry import registry


def _from_numpy(arr):
    storage = Storage(arr)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    return Tensor(storage, arr.shape, stride)


def add(a, b):
    return _from_numpy(a.storage.data + b.storage.data)

def mul(a, b):
    return _from_numpy(a.storage.data * b.storage.data)

def matmul(a, b):
    return _from_numpy(a.storage.data @ b.storage.data)

def relu(a):
    return _from_numpy(np.maximum(a.storage.data, 0))

def sum_(a, dim=None, keepdim=False):
    return _from_numpy(a.storage.data.sum(axis=dim, keepdims=keepdim))

registry.register("add", "cpu", add)
registry.register("mul", "cpu", mul)
registry.register("matmul", "cpu", matmul)
registry.register("relu", "cpu", relu)
registry.register("sum", "cpu", sum_)

# src/mindtorch_v2/_functional.py
from ._dispatch.dispatcher import dispatch

def add(a, b):
    return dispatch("add", a.device.type, a, b)

def mul(a, b):
    return dispatch("mul", a.device.type, a, b)

def matmul(a, b):
    return dispatch("matmul", a.device.type, a, b)

def relu(a):
    return dispatch("relu", a.device.type, a)

def sum(a, dim=None, keepdim=False):
    return dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)

# src/mindtorch_v2/_tensor.py
from ._functional import add, mul, matmul, relu, sum

class Tensor(...):
    def __add__(self, other):
        return add(self, other)
    def __mul__(self, other):
        return mul(self, other)
    def matmul(self, other):
        return matmul(self, other)
    def relu(self):
        return relu(self)
    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

# src/mindtorch_v2/__init__.py
from ._functional import add, mul, matmul, relu, sum
from ._backends import cpu  # register CPU ops
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_cpu.py::test_add_mul_matmul_relu_sum -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_dispatch/registry.py src/mindtorch_v2/_dispatch/dispatcher.py src/mindtorch_v2/_backends/cpu.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_ops_cpu.py
git commit -m "feat(mindtorch_v2): add dispatcher and cpu ops"
```

---

### Task 6: Broadcasting + sum reduction semantics

**Files:**
- Modify: `src/mindtorch_v2/_backends/cpu.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_broadcast.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_broadcast.py
import mindtorch_v2 as torch


def test_broadcast_add_mul():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0], [2.0]])
    c = torch.add(a, b)
    d = torch.mul(a, b)
    assert c.shape == (2, 3)
    assert d.shape == (2, 3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_broadcast.py::test_broadcast_add_mul -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/cpu.py
import numpy as np


def add(a, b):
    return _from_numpy(np.add(a.storage.data, b.storage.data))

def mul(a, b):
    return _from_numpy(np.multiply(a.storage.data, b.storage.data))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_broadcast.py::test_broadcast_add_mul -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/cpu.py tests/mindtorch_v2/test_broadcast.py
git commit -m "feat(mindtorch_v2): add broadcast support for cpu ops"
```

---

### Task 7: Autograd engine + backward for ops (CPU)

**Files:**
- Create: `src/mindtorch_v2/_autograd/node.py`
- Create: `src/mindtorch_v2/_autograd/engine.py`
- Create: `src/mindtorch_v2/_autograd/grad_mode.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Test: `tests/mindtorch_v2/test_autograd.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_autograd.py
import mindtorch_v2 as torch


def test_autograd_add_mul():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    x.requires_grad = True
    y.requires_grad = True
    z = torch.mul(torch.add(x, y), x)
    z.sum().backward()
    assert x.grad is not None
    assert y.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd.py::test_autograd_add_mul -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_autograd/grad_mode.py
class GradMode:
    enabled = True

class no_grad:
    def __enter__(self):
        GradMode.enabled = False
    def __exit__(self, exc_type, exc, tb):
        GradMode.enabled = True

# src/mindtorch_v2/_autograd/node.py
class Node:
    def __init__(self, backward, inputs):
        self.backward = backward
        self.inputs = inputs

# src/mindtorch_v2/_autograd/engine.py

def backward(tensor, grad=None):
    if grad is None:
        grad = tensor._ones_like()
    tensor.grad = grad
    if tensor.grad_fn is None:
        return
    grads = tensor.grad_fn.backward(grad)
    for t, g in zip(tensor.grad_fn.inputs, grads):
        if g is None:
            continue
        if t.grad is None:
            t.grad = g
        else:
            t.grad = t.grad + g
        if t.grad_fn is not None:
            backward(t, t.grad)

# src/mindtorch_v2/_functional.py
from ._autograd.grad_mode import GradMode
from ._autograd.node import Node

# example for add

def add(a, b):
    out = dispatch("add", a.device.type, a, b)
    if GradMode.enabled and (a.requires_grad or b.requires_grad):
        def _backward(grad):
            return grad, grad
        out.grad_fn = Node(_backward, (a, b))
        out.requires_grad = True
    return out

# src/mindtorch_v2/_tensor.py
from ._autograd.engine import backward as _backward

class Tensor(...):
    def backward(self, gradient=None):
        _backward(self, gradient)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd.py::test_autograd_add_mul -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_autograd/node.py src/mindtorch_v2/_autograd/engine.py src/mindtorch_v2/_autograd/grad_mode.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_autograd.py
git commit -m "feat(mindtorch_v2): add autograd engine"
```

---

### Task 8: Broadcast-aware backward + sum backward

**Files:**
- Modify: `src/mindtorch_v2/_autograd/engine.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Create: `src/mindtorch_v2/_autograd/utils.py`
- Test: `tests/mindtorch_v2/test_autograd_broadcast.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_autograd_broadcast.py
import mindtorch_v2 as torch


def test_broadcast_backward():
    a = torch.tensor([[1.0, 2.0, 3.0]])
    b = torch.tensor([[1.0], [2.0]])
    a.requires_grad = True
    b.requires_grad = True
    c = torch.add(a, b)
    c.sum().backward()
    assert a.grad.shape == (1, 3)
    assert b.grad.shape == (2, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_autograd_broadcast.py::test_broadcast_backward -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_autograd/utils.py
import numpy as np

def reduce_grad(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

# src/mindtorch_v2/_functional.py
from ._autograd.utils import reduce_grad

# in add backward:
# return reduce_grad(grad, a.shape), reduce_grad(grad, b.shape)

# sum backward:

def sum(a, dim=None, keepdim=False):
    out = dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)
    if GradMode.enabled and a.requires_grad:
        def _backward(grad):
            return (grad * a._ones_like(),)
        out.grad_fn = Node(_backward, (a,))
        out.requires_grad = True
    return out
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_autograd_broadcast.py::test_broadcast_backward -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_autograd/utils.py src/mindtorch_v2/_functional.py src/mindtorch_v2/_autograd/engine.py tests/mindtorch_v2/test_autograd_broadcast.py
git commit -m "feat(mindtorch_v2): broadcast-aware backward"
```

---

### Task 9: nn.Module, Parameter, Linear, ReLU, Sequential

**Files:**
- Create: `src/mindtorch_v2/nn/module.py`
- Create: `src/mindtorch_v2/nn/parameter.py`
- Create: `src/mindtorch_v2/nn/modules/linear.py`
- Create: `src/mindtorch_v2/nn/modules/activation.py`
- Create: `src/mindtorch_v2/nn/modules/container.py`
- Modify: `src/mindtorch_v2/nn/__init__.py`
- Test: `tests/mindtorch_v2/test_nn.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_nn.py
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_linear_forward_backward():
    layer = nn.Linear(2, 3)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_nn.py::test_linear_forward_backward -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/nn/parameter.py
from .._tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data.storage, data.shape, data.stride, data.offset, True)

# src/mindtorch_v2/nn/module.py
class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if hasattr(self, "_parameters") and isinstance(value, Parameter):
            self._parameters[name] = value
        if hasattr(self, "_modules") and isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# src/mindtorch_v2/nn/modules/linear.py
import math
from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from ..._functional import add, matmul

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        w = tensor((out_features, in_features))
        b = tensor((out_features,))
        self.weight = Parameter(w)
        self.bias = Parameter(b)

    def forward(self, x):
        return add(matmul(x, self.weight.transpose(0, 1)), self.bias)

# src/mindtorch_v2/nn/modules/activation.py
from ..module import Module
from ..._functional import relu

class ReLU(Module):
    def forward(self, x):
        return relu(x)

# src/mindtorch_v2/nn/modules/container.py
from ..module import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = {str(i): m for i, m in enumerate(modules)}

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

# src/mindtorch_v2/nn/__init__.py
from .module import Module
from .parameter import Parameter
from .modules.linear import Linear
from .modules.activation import ReLU
from .modules.container import Sequential
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_nn.py::test_linear_forward_backward -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/nn/module.py src/mindtorch_v2/nn/parameter.py src/mindtorch_v2/nn/modules/linear.py src/mindtorch_v2/nn/modules/activation.py src/mindtorch_v2/nn/modules/container.py src/mindtorch_v2/nn/__init__.py tests/mindtorch_v2/test_nn.py
git commit -m "feat(mindtorch_v2): add minimal nn modules"
```

---

### Task 10: Optimizer SGD

**Files:**
- Create: `src/mindtorch_v2/optim/optimizer.py`
- Create: `src/mindtorch_v2/optim/sgd.py`
- Modify: `src/mindtorch_v2/optim/__init__.py`
- Test: `tests/mindtorch_v2/test_optim.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_optim.py
import mindtorch_v2 as torch
from mindtorch_v2 import nn, optim


def test_sgd_step():
    layer = nn.Linear(2, 1)
    opt = optim.SGD(layer.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x)
    y.sum().backward()
    opt.step()
    assert layer.weight.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_optim.py::test_sgd_step -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/optim/optimizer.py
class Optimizer:
    def __init__(self, params):
        self.params = list(params)
    def zero_grad(self):
        for p in self.params:
            p.grad = None

# src/mindtorch_v2/optim/sgd.py
from .optimizer import Optimizer
from .._autograd.grad_mode import no_grad

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params)
        self.lr = lr

    def step(self):
        with no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                p.storage.data = p.storage.data - self.lr * p.grad.storage.data

# src/mindtorch_v2/optim/__init__.py
from .sgd import SGD
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_optim.py::test_sgd_step -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/optim/optimizer.py src/mindtorch_v2/optim/sgd.py src/mindtorch_v2/optim/__init__.py tests/mindtorch_v2/test_optim.py
git commit -m "feat(mindtorch_v2): add SGD optimizer"
```

---

### Task 11: NPU backend scaffolding + device transfer

**Files:**
- Create: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_storage.py`
- Modify: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Test: `tests/mindtorch_v2/test_device_transfer.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_device_transfer.py
import mindtorch_v2 as torch


def test_to_device_roundtrip():
    x = torch.tensor([1.0, 2.0])
    y = x.to("cpu")
    assert y.device.type == "cpu"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_device_transfer.py::test_to_device_roundtrip -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/ascend.py
try:
    import acl
    HAS_ACL = True
except Exception:
    HAS_ACL = False

# src/mindtorch_v2/_storage.py
class Storage:
    def to(self, device):
        if device.type == self.device.type:
            return self
        if device.type == "cpu":
            raise NotImplementedError("NPU->CPU copy not implemented yet")
        raise NotImplementedError("CPU->NPU copy not implemented yet")

# src/mindtorch_v2/_tensor.py
from ._device import device as Device

class Tensor(...):
    def to(self, dev):
        if isinstance(dev, str):
            dev = Device(dev)
        new_storage = self.storage.to(dev)
        return Tensor(new_storage, self.shape, self.stride, self.offset, self.requires_grad)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_device_transfer.py::test_to_device_roundtrip -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_storage.py src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_device_transfer.py
git commit -m "feat(mindtorch_v2): add device transfer stubs"
```

---

### Task 12: Implement ACLNN ops + NPU tests (skip if unavailable)

**Files:**
- Modify: `src/mindtorch_v2/_backends/ascend.py`
- Modify: `src/mindtorch_v2/_functional.py`
- Test: `tests/mindtorch_v2/test_ops_npu.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_ops_npu.py
import pytest
import mindtorch_v2 as torch


def test_npu_add():
    if not torch.npu.is_available():
        pytest.skip("NPU not available")
    x = torch.tensor([1.0, 2.0]).to("npu")
    y = torch.tensor([3.0, 4.0]).to("npu")
    z = torch.add(x, y).to("cpu")
    assert z.storage.data.tolist() == [4.0, 6.0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_add -v`
Expected: FAIL or SKIP depending on NPU availability

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/_backends/ascend.py
# - Implement runtime init: acl.init, set device, create context/stream
# - Implement alloc/free + memcpy H2D/D2H
# - Implement ops add/mul/matmul/relu/sum using ACLNN or acl.op API
# - Expose is_available() based on import + device query
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_ops_npu.py::test_npu_add -v`
Expected: PASS or SKIP (if no NPU)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_backends/ascend.py src/mindtorch_v2/_functional.py tests/mindtorch_v2/test_ops_npu.py
git commit -m "feat(mindtorch_v2): add NPU backend ops"
```

---

### Task 13: Export public API and finalize docs

**Files:**
- Modify: `src/mindtorch_v2/__init__.py`
- Modify: `docs/plans/2026-02-10-mindtorch-v2-design.md` (if updates needed)
- Test: `tests/mindtorch_v2/test_import.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_import.py
import mindtorch_v2 as torch

def test_public_api():
    assert hasattr(torch, "Tensor")
    assert hasattr(torch, "nn")
    assert hasattr(torch, "optim")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/mindtorch_v2/test_import.py::test_public_api -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/mindtorch_v2/__init__.py
from ._tensor import Tensor
from . import nn, optim
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/mindtorch_v2/test_import.py::test_public_api -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_import.py
git commit -m "feat(mindtorch_v2): finalize public api"
```

---

### Task 14: Full test run

**Files:**
- None

**Step 1: Run full v2 test suite**

Run: `pytest tests/mindtorch_v2 -v`
Expected: PASS (or NPU tests skipped if unavailable)

**Step 2: Commit (if needed)**

```bash
git status -sb
```
Expected: clean

