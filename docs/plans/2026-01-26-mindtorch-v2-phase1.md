# mindtorch v2 Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational Tensor/Storage model, basic creation ops, view operations, and indexing - enough to create tensors, manipulate shapes, and access elements with PyTorch-identical semantics.

**Architecture:** Independent Tensor class wrapping a Storage that holds a contiguous MindSpore tensor. All stride/offset/view logic is ours. MindSpore is only used for raw data allocation and kernel execution via `_op_prim`.

**Tech Stack:** Python 3.9+, MindSpore 2.7.2, NumPy, pytest

**Worktree:** `/Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2`
**Branch:** `feature/mindtorch-v2`
**Source dir:** `src/mindtorch_v2/`
**Test dir:** `tests/mindtorch_v2/`
**Existing primitives:** `src/mindtorch/_op_prim/cpu/pyboost.py` (pyboost ops), `src/mindtorch/_op_prim/cpu/legacy.py` (legacy ops)

**Test command:**
```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `src/mindtorch_v2/__init__.py`
- Create: `src/mindtorch_v2/_dtype.py`
- Create: `src/mindtorch_v2/_device.py`
- Create: `tests/mindtorch_v2/__init__.py`
- Create: `tests/mindtorch_v2/conftest.py`
- Create: `tests/mindtorch_v2/test_scaffold.py`

**Step 1: Create package directories**

```bash
mkdir -p src/mindtorch_v2
mkdir -p tests/mindtorch_v2
```

**Step 2: Write the smoke test**

```python
# tests/mindtorch_v2/__init__.py
# (empty)
```

```python
# tests/mindtorch_v2/conftest.py
import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
```

```python
# tests/mindtorch_v2/test_scaffold.py
def test_import():
    import mindtorch_v2 as torch
    assert hasattr(torch, '__version__')
```

**Step 3: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_scaffold.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mindtorch_v2'`

**Step 4: Create the package init**

```python
# src/mindtorch_v2/__init__.py
"""mindtorch v2 - PyTorch-compatible API on MindSpore backend."""

__version__ = "0.1.0"
```

**Step 5: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_scaffold.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/mindtorch_v2/ tests/mindtorch_v2/
git commit -m "feat(v2): scaffold mindtorch_v2 package and test infrastructure"
```

---

## Task 2: dtype Module

**Files:**
- Create: `src/mindtorch_v2/_dtype.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Create: `tests/mindtorch_v2/test_dtype.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_dtype.py
import mindtorch_v2 as torch


def test_dtype_exists():
    """Core dtypes are accessible as torch.float32, etc."""
    assert torch.float32 is not None
    assert torch.float64 is not None
    assert torch.float16 is not None
    assert torch.bfloat16 is not None
    assert torch.int8 is not None
    assert torch.int16 is not None
    assert torch.int32 is not None
    assert torch.int64 is not None
    assert torch.uint8 is not None
    assert torch.bool is not None
    assert torch.complex64 is not None
    assert torch.complex128 is not None


def test_dtype_aliases():
    """PyTorch aliases like torch.float, torch.long, torch.half."""
    assert torch.float is torch.float32
    assert torch.double is torch.float64
    assert torch.half is torch.float16
    assert torch.long is torch.int64
    assert torch.int is torch.int32


def test_dtype_properties():
    """dtype has is_floating_point and is_complex properties."""
    assert torch.float32.is_floating_point
    assert torch.float64.is_floating_point
    assert not torch.int32.is_floating_point
    assert not torch.bool.is_floating_point
    assert torch.complex64.is_complex
    assert torch.complex128.is_complex
    assert not torch.float32.is_complex


def test_dtype_itemsize():
    """dtype.itemsize returns element size in bytes."""
    assert torch.float32.itemsize == 4
    assert torch.float64.itemsize == 8
    assert torch.float16.itemsize == 2
    assert torch.int64.itemsize == 8
    assert torch.int32.itemsize == 4
    assert torch.int8.itemsize == 1
    assert torch.bool.itemsize == 1


def test_dtype_to_numpy():
    """dtype can be converted to numpy dtype."""
    import numpy as np
    assert torch.dtype_to_numpy(torch.float32) == np.float32
    assert torch.dtype_to_numpy(torch.int64) == np.int64
    assert torch.dtype_to_numpy(torch.bool) == np.bool_


def test_numpy_to_dtype():
    """numpy dtype can be converted to torch dtype."""
    import numpy as np
    assert torch.numpy_to_dtype(np.float32) is torch.float32
    assert torch.numpy_to_dtype(np.int64) is torch.int64
    assert torch.numpy_to_dtype(np.bool_) is torch.bool
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_dtype.py -v
```

Expected: FAIL with `AttributeError: module 'mindtorch_v2' has no attribute 'float32'`

**Step 3: Implement dtype module**

```python
# src/mindtorch_v2/_dtype.py
"""PyTorch-compatible dtype definitions backed by MindSpore dtypes."""

import numpy as np
import mindspore


class DType:
    """Represents a tensor data type, wrapping a MindSpore dtype."""

    def __init__(self, name, ms_dtype, np_dtype, size, is_float=False, is_complex_type=False):
        self.name = name
        self._ms_dtype = ms_dtype
        self._np_dtype = np_dtype
        self._itemsize = size
        self._is_floating_point = is_float
        self._is_complex = is_complex_type

    @property
    def is_floating_point(self):
        return self._is_floating_point

    @property
    def is_complex(self):
        return self._is_complex

    @property
    def itemsize(self):
        return self._itemsize

    def to_mindspore(self):
        return self._ms_dtype

    def to_numpy(self):
        return self._np_dtype

    def __repr__(self):
        return f"torch.{self.name}"

    def __str__(self):
        return f"torch.{self.name}"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, DType):
            return self.name == other.name
        return NotImplemented


# --- Core dtypes ---
float16 = DType("float16", mindspore.float16, np.float16, 2, is_float=True)
float32 = DType("float32", mindspore.float32, np.float32, 4, is_float=True)
float64 = DType("float64", mindspore.float64, np.float64, 8, is_float=True)
bfloat16 = DType("bfloat16", mindspore.bfloat16, None, 2, is_float=True)

int8 = DType("int8", mindspore.int8, np.int8, 1)
int16 = DType("int16", mindspore.int16, np.int16, 2)
int32 = DType("int32", mindspore.int32, np.int32, 4)
int64 = DType("int64", mindspore.int64, np.int64, 8)

uint8 = DType("uint8", mindspore.uint8, np.uint8, 1)
uint16 = DType("uint16", mindspore.uint16, np.uint16, 2)
uint32 = DType("uint32", mindspore.uint32, np.uint32, 4)
uint64 = DType("uint64", mindspore.uint64, np.uint64, 8)

bool = DType("bool", mindspore.bool_, np.bool_, 1)

complex64 = DType("complex64", mindspore.complex64, np.complex64, 8, is_complex_type=True)
complex128 = DType("complex128", mindspore.complex128, np.complex128, 16, is_complex_type=True)

# --- Aliases ---
half = float16
float = float32
double = float64
long = int64
int = int32
short = int16

cfloat = complex64
cdouble = complex128

# --- Conversion maps ---
_ms_to_dtype = {
    mindspore.float16: float16,
    mindspore.float32: float32,
    mindspore.float64: float64,
    mindspore.bfloat16: bfloat16,
    mindspore.int8: int8,
    mindspore.int16: int16,
    mindspore.int32: int32,
    mindspore.int64: int64,
    mindspore.uint8: uint8,
    mindspore.uint16: uint16,
    mindspore.uint32: uint32,
    mindspore.uint64: uint64,
    mindspore.bool_: bool,
    mindspore.complex64: complex64,
    mindspore.complex128: complex128,
}

_dtype_to_np = {d: d._np_dtype for d in _ms_to_dtype.values() if d._np_dtype is not None}

_np_to_dtype = {v: k for k, v in _dtype_to_np.items()}

_py_to_dtype = {
    __builtins__['bool'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'bool'): bool,
    __builtins__['float'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'float'): float,
    __builtins__['int'] if isinstance(__builtins__, dict) else getattr(__builtins__, 'int'): int64,
}


def from_mindspore_dtype(ms_dtype):
    """Convert MindSpore dtype to mindtorch_v2 dtype."""
    return _ms_to_dtype.get(ms_dtype)


def dtype_to_numpy(dtype):
    """Convert mindtorch_v2 dtype to numpy dtype."""
    return _dtype_to_np.get(dtype)


def numpy_to_dtype(np_dtype):
    """Convert numpy dtype to mindtorch_v2 dtype."""
    np_dtype = np.dtype(np_dtype).type
    return _np_to_dtype.get(np_dtype)
```

**Step 4: Update __init__.py to export dtypes**

```python
# src/mindtorch_v2/__init__.py
"""mindtorch v2 - PyTorch-compatible API on MindSpore backend."""

__version__ = "0.1.0"

from ._dtype import (
    # Core dtypes
    float16, float32, float64, bfloat16,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    bool, complex64, complex128,
    # Aliases
    half, float, double, long, int, short,
    cfloat, cdouble,
    # Conversion functions
    dtype_to_numpy, numpy_to_dtype,
    # DType class
    DType,
)
```

**Step 5: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_dtype.py -v
```

Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_dtype.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_dtype.py
git commit -m "feat(v2): add dtype module with PyTorch-compatible type definitions"
```

---

## Task 3: device Module

**Files:**
- Create: `src/mindtorch_v2/_device.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Create: `tests/mindtorch_v2/test_device.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_device.py
import mindtorch_v2 as torch


def test_device_from_string():
    d = torch.device("cpu")
    assert d.type == "cpu"
    assert d.index is None


def test_device_with_index():
    d = torch.device("cuda", 0)
    assert d.type == "cuda"
    assert d.index == 0


def test_device_from_string_with_index():
    d = torch.device("cuda:1")
    assert d.type == "cuda"
    assert d.index == 1


def test_device_equality():
    assert torch.device("cpu") == torch.device("cpu")
    assert torch.device("cuda", 0) == torch.device("cuda:0")
    assert torch.device("cpu") != torch.device("cuda")


def test_device_repr():
    assert repr(torch.device("cpu")) == "device(type='cpu')"
    assert repr(torch.device("cuda", 0)) == "device(type='cuda', index=0)"


def test_device_hash():
    """Devices can be used as dict keys."""
    d = {torch.device("cpu"): 1, torch.device("cuda:0"): 2}
    assert d[torch.device("cpu")] == 1
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_device.py -v
```

Expected: FAIL with `AttributeError: module 'mindtorch_v2' has no attribute 'device'`

**Step 3: Implement device module**

```python
# src/mindtorch_v2/_device.py
"""PyTorch-compatible device representation."""


class device:
    """Represents the device on which a tensor is or will be allocated.

    Matches torch.device API:
        device("cpu")
        device("cuda", 0)
        device("cuda:1")
    """

    def __init__(self, type_or_str, index=None):
        if isinstance(type_or_str, device):
            self.type = type_or_str.type
            self.index = type_or_str.index
            return

        if isinstance(type_or_str, str):
            if ":" in type_or_str:
                parts = type_or_str.split(":", 1)
                self.type = parts[0]
                self.index = builtins_int(parts[1])
            else:
                self.type = type_or_str
                self.index = index
        else:
            raise ValueError(f"Expected string or device, got {type(type_or_str)}")

    def __repr__(self):
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type

    def __eq__(self, other):
        if isinstance(other, device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return self == device(other)
        return NotImplemented

    def __hash__(self):
        return hash((self.type, self.index))


# Avoid conflict with Python builtins
import builtins as _builtins
builtins_int = _builtins.int
```

**Step 4: Update __init__.py**

Add to `src/mindtorch_v2/__init__.py`:

```python
from ._device import device
```

**Step 5: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_device.py -v
```

Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_device.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_device.py
git commit -m "feat(v2): add device module with PyTorch-compatible device class"
```

---

## Task 4: Storage Class

**Files:**
- Create: `src/mindtorch_v2/_storage.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Create: `tests/mindtorch_v2/test_storage.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_storage.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2._storage import TypedStorage, UntypedStorage


def test_storage_from_size():
    """Create empty storage with a given number of elements."""
    s = TypedStorage(10, dtype=torch.float32)
    assert s.size() == 10
    assert s.dtype is torch.float32
    assert s.device == torch.device("cpu")
    assert s.nbytes() == 40  # 10 * 4 bytes


def test_storage_from_numpy():
    """Create storage wrapping a numpy array."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    s = TypedStorage._from_numpy(arr)
    assert s.size() == 3
    assert s.dtype is torch.float32


def test_storage_getitem():
    """Index into storage."""
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    s = TypedStorage._from_numpy(arr)
    assert s[0] == 10.0
    assert s[2] == 30.0


def test_storage_setitem():
    """Set element in storage."""
    s = TypedStorage(3, dtype=torch.float32)
    s[0] = 42.0
    assert s[0] == 42.0


def test_storage_data_ptr():
    """Storage exposes a data pointer (for compatibility)."""
    s = TypedStorage(5, dtype=torch.float32)
    ptr = s.data_ptr()
    assert isinstance(ptr, int)
    assert ptr != 0


def test_untyped_storage():
    """UntypedStorage tracks bytes, not typed elements."""
    s = UntypedStorage(40)  # 40 bytes
    assert s.nbytes() == 40


def test_storage_shared():
    """Two references to same storage see the same data."""
    s = TypedStorage(3, dtype=torch.float32)
    s[0] = 1.0
    s2 = s  # same object
    assert s2[0] == 1.0
    s2[0] = 99.0
    assert s[0] == 99.0
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_storage.py -v
```

Expected: FAIL with `ImportError: cannot import name 'TypedStorage'`

**Step 3: Implement storage module**

```python
# src/mindtorch_v2/_storage.py
"""PyTorch-compatible Storage classes backed by MindSpore tensors.

Storage represents a contiguous 1D memory buffer. Multiple Tensor views
can share the same Storage. This is the foundation of PyTorch's view semantics.
"""

import numpy as np
import mindspore
from . import _dtype as dtype_mod


class UntypedStorage:
    """Raw byte-level storage. Equivalent to torch.UntypedStorage.

    Tracks a contiguous block of memory measured in bytes.
    """

    def __init__(self, size_bytes=0, *, device=None):
        """Create storage with given size in bytes.

        Args:
            size_bytes: Number of bytes to allocate.
            device: Device for the storage (default: cpu).
        """
        from ._device import device as device_cls
        self._device = device_cls(device or "cpu")
        self._nbytes = size_bytes
        # Underlying buffer as uint8 MindSpore tensor
        if size_bytes > 0:
            self._data = mindspore.numpy.zeros(size_bytes, dtype=mindspore.uint8)
        else:
            self._data = mindspore.numpy.zeros(1, dtype=mindspore.uint8)

    def nbytes(self):
        return self._nbytes

    @property
    def device(self):
        return self._device

    def data_ptr(self):
        return self._data.data_ptr()


class TypedStorage:
    """Element-typed storage. Equivalent to torch.TypedStorage.

    Wraps a contiguous MindSpore tensor as 1D typed buffer.
    """

    def __init__(self, size_or_data=0, *, dtype=None, device=None):
        """Create typed storage.

        Args:
            size_or_data: Number of elements (int) or data to store.
            dtype: Element type (mindtorch_v2 DType).
            device: Device for the storage.
        """
        from ._device import device as device_cls

        if dtype is None:
            dtype = dtype_mod.float32
        self._dtype = dtype
        self._device = device_cls(device or "cpu")

        if isinstance(size_or_data, int):
            self._size = size_or_data
            ms_dtype = dtype.to_mindspore()
            if size_or_data > 0:
                self._ms_tensor = mindspore.ops.zeros(size_or_data, dtype=ms_dtype)
            else:
                self._ms_tensor = mindspore.Tensor([], dtype=ms_dtype)
        elif isinstance(size_or_data, mindspore.Tensor):
            # Wrap existing MindSpore tensor (must be 1D contiguous)
            if size_or_data.ndim != 1:
                size_or_data = size_or_data.reshape(-1)
            self._ms_tensor = size_or_data
            self._size = size_or_data.shape[0]
            self._dtype = dtype_mod.from_mindspore_dtype(size_or_data.dtype)
        else:
            raise TypeError(f"Unsupported type: {type(size_or_data)}")

    @classmethod
    def _from_numpy(cls, arr):
        """Create TypedStorage from numpy array."""
        arr = np.ascontiguousarray(arr).ravel()
        ms_tensor = mindspore.Tensor(arr)
        dtype = dtype_mod.from_mindspore_dtype(ms_tensor.dtype)
        storage = cls.__new__(cls)
        storage._ms_tensor = ms_tensor
        storage._size = arr.shape[0]
        storage._dtype = dtype
        from ._device import device as device_cls
        storage._device = device_cls("cpu")
        return storage

    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def nbytes(self):
        return self._size * self._dtype.itemsize

    def data_ptr(self):
        return self._ms_tensor.data_ptr()

    def __getitem__(self, idx):
        return self._ms_tensor[idx].asnumpy().item()

    def __setitem__(self, idx, value):
        np_data = self._ms_tensor.asnumpy()
        np_data[idx] = value
        self._ms_tensor = mindspore.Tensor(np_data)

    def __len__(self):
        return self._size

    def __repr__(self):
        return f"TypedStorage(size={self._size}, dtype={self._dtype})"
```

**Step 4: Update __init__.py**

Add to `src/mindtorch_v2/__init__.py`:

```python
from ._storage import TypedStorage, UntypedStorage
```

**Step 5: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_storage.py -v
```

Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_storage.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_storage.py
git commit -m "feat(v2): add Storage classes with MindSpore tensor backing"
```

---

## Task 5: Tensor Class - Core Structure

**Files:**
- Create: `src/mindtorch_v2/_tensor.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Create: `tests/mindtorch_v2/test_tensor_core.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_tensor_core.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2._tensor import Tensor


def test_tensor_from_list():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.dtype is torch.float32
    assert t.ndim == 1


def test_tensor_from_nested_list():
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.ndim == 2


def test_tensor_shape_and_size():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.size() == (2, 3)
    assert t.size(0) == 2
    assert t.size(1) == 3


def test_tensor_stride():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.stride() == (3, 1)
    assert t.stride(0) == 3
    assert t.stride(1) == 1


def test_tensor_dtype():
    t = Tensor([1.0, 2.0], dtype=torch.float64)
    assert t.dtype is torch.float64


def test_tensor_device():
    t = Tensor([1.0])
    assert t.device == torch.device("cpu")


def test_tensor_numel():
    t = Tensor([[1, 2], [3, 4], [5, 6]])
    assert t.numel() == 6


def test_tensor_element_size():
    t = Tensor([1.0], dtype=torch.float32)
    assert t.element_size() == 4
    t64 = Tensor([1.0], dtype=torch.float64)
    assert t64.element_size() == 8


def test_tensor_dim():
    t = Tensor([[[1, 2], [3, 4]]])
    assert t.dim() == 3


def test_tensor_requires_grad():
    t = Tensor([1.0], requires_grad=True)
    assert t.requires_grad is True
    t2 = Tensor([1.0])
    assert t2.requires_grad is False


def test_tensor_storage_offset():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.storage_offset() == 0


def test_tensor_is_contiguous():
    t = Tensor([[1, 2], [3, 4]])
    assert t.is_contiguous() is True


def test_tensor_to_numpy():
    t = Tensor([1.0, 2.0, 3.0])
    arr = t.numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_almost_equal(arr, [1.0, 2.0, 3.0])


def test_tensor_item():
    t = Tensor([42.0])
    assert t.item() == 42.0
    t2 = Tensor(3.14)
    assert abs(t2.item() - 3.14) < 1e-5


def test_tensor_repr():
    t = Tensor([1.0, 2.0])
    s = repr(t)
    assert "tensor" in s
    assert "1." in s


def test_tensor_version():
    """Tensor tracks version for autograd safety."""
    t = Tensor([1.0, 2.0])
    v0 = t._version
    assert isinstance(v0, int)
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_tensor_core.py -v
```

Expected: FAIL with `ImportError: cannot import name 'Tensor'`

**Step 3: Implement Tensor class**

```python
# src/mindtorch_v2/_tensor.py
"""PyTorch-compatible Tensor class with proper Storage-based semantics.

This Tensor does NOT inherit from MindSpore's Tensor. It holds a reference
to a Storage (which wraps a MindSpore tensor), plus view metadata:
shape, stride, storage_offset.
"""

import math
import numpy as np
import mindspore
from . import _dtype as dtype_mod
from ._storage import TypedStorage
from ._device import device as device_cls


def _compute_strides(shape):
    """Compute contiguous (row-major) strides for a given shape."""
    if len(shape) == 0:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _data_to_numpy(data, dtype=None):
    """Convert input data to numpy array with appropriate dtype."""
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (list, tuple)):
        arr = np.array(data)
    elif isinstance(data, (int, float, bool)):
        arr = np.array(data)
    else:
        raise TypeError(f"Cannot convert {type(data)} to Tensor")

    # Infer dtype
    if dtype is not None:
        np_dtype = dtype_mod.dtype_to_numpy(dtype)
        if np_dtype is not None:
            arr = arr.astype(np_dtype)
    else:
        # Default: float for float-like, int64 for int-like
        if arr.dtype.kind == 'f':
            arr = arr.astype(np.float32)
        elif arr.dtype.kind == 'i':
            arr = arr.astype(np.int64)
        elif arr.dtype.kind == 'b':
            pass  # keep bool
        elif arr.dtype.kind == 'u':
            pass  # keep uint
        elif arr.dtype.kind == 'c':
            arr = arr.astype(np.complex64)

    return arr


class Tensor:
    """PyTorch-compatible Tensor backed by Storage.

    Attributes:
        _storage: TypedStorage holding the contiguous data
        _shape: tuple of ints
        _stride: tuple of ints
        _storage_offset: int (offset into storage, in elements)
        _requires_grad: bool
        _grad_fn: autograd Node or None
        _grad: accumulated gradient Tensor or None
        _version: int (incremented on in-place mutation)
    """

    def __init__(self, data=None, *, dtype=None, device=None, requires_grad=False,
                 _storage=None, _shape=None, _stride=None, _storage_offset=0):
        """Create a Tensor.

        Public usage:
            Tensor([1.0, 2.0, 3.0])
            Tensor([[1, 2], [3, 4]], dtype=torch.float64)

        Internal usage (for views):
            Tensor(_storage=s, _shape=(2,3), _stride=(3,1), _storage_offset=0)
        """
        if _storage is not None:
            # Internal: create view over existing storage
            self._storage = _storage
            self._shape = tuple(_shape)
            self._stride = tuple(_stride) if _stride is not None else _compute_strides(self._shape)
            self._storage_offset = _storage_offset
            self._dtype = _storage.dtype
        elif data is not None:
            # Public: create from data
            arr = _data_to_numpy(data, dtype)
            shape = arr.shape
            flat = arr.ravel()
            ms_tensor = mindspore.Tensor(flat)
            self._storage = TypedStorage(ms_tensor)
            self._shape = shape
            self._stride = _compute_strides(shape)
            self._storage_offset = 0
            self._dtype = self._storage.dtype
        else:
            raise ValueError("Must provide either data or _storage")

        self._device = device_cls(device or "cpu")
        self._requires_grad = requires_grad
        self._grad_fn = None
        self._grad = None
        self._version = 0

    # --- Shape / metadata ---

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    @property
    def grad_fn(self):
        return self._grad_fn

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def data(self):
        """Returns a Tensor sharing storage but detached from autograd."""
        return Tensor(
            _storage=self._storage,
            _shape=self._shape,
            _stride=self._stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
        )

    @data.setter
    def data(self, value):
        if isinstance(value, Tensor):
            self._storage = value._storage
            self._shape = value._shape
            self._stride = value._stride
            self._storage_offset = value._storage_offset
            self._dtype = value._dtype
            self._version += 1

    def size(self, dim=None):
        if dim is not None:
            return self._shape[dim]
        return self._shape

    def stride(self, dim=None):
        if dim is not None:
            return self._stride[dim]
        return self._stride

    def storage_offset(self):
        return self._storage_offset

    def storage(self):
        return self._storage

    def dim(self):
        return len(self._shape)

    def numel(self):
        result = 1
        for s in self._shape:
            result *= s
        return result

    def element_size(self):
        return self._dtype.itemsize

    def is_contiguous(self, memory_format=None):
        """Check if tensor is contiguous in row-major order."""
        expected = _compute_strides(self._shape)
        return self._stride == expected

    def contiguous(self, memory_format=None):
        """Return contiguous tensor (copy if not already contiguous)."""
        if self.is_contiguous():
            return self
        # Force a contiguous copy
        arr = self.numpy()
        arr = np.ascontiguousarray(arr)
        return Tensor(arr, dtype=self._dtype, device=str(self._device),
                      requires_grad=self._requires_grad)

    # --- Conversion ---

    def _to_numpy_flat(self):
        """Get the raw flat numpy data from storage."""
        return self._storage._ms_tensor.asnumpy()

    def numpy(self):
        """Convert to numpy array. Must not require grad."""
        flat = self._to_numpy_flat()
        if self.is_contiguous() and self._storage_offset == 0:
            return flat[:self.numel()].reshape(self._shape)
        # Non-contiguous: use stride-based indexing
        return self._strided_numpy(flat)

    def _strided_numpy(self, flat):
        """Extract data via strides from flat buffer."""
        result = np.empty(self._shape, dtype=flat.dtype)
        for idx in np.ndindex(*self._shape):
            flat_idx = self._storage_offset + sum(i * s for i, s in zip(idx, self._stride))
            result[idx] = flat[flat_idx]
        return result

    def item(self):
        """Extract scalar value."""
        if self.numel() != 1:
            raise ValueError(f"only one element tensors can be converted to Python scalars, got {self.numel()}")
        return self.numpy().item()

    def tolist(self):
        return self.numpy().tolist()

    def to_mindspore(self):
        """Convert to MindSpore tensor (for interop)."""
        arr = self.numpy()
        return mindspore.Tensor(arr)

    # --- Repr ---

    def __repr__(self):
        arr = self.numpy()
        data_str = np.array2string(arr, separator=', ', prefix='tensor(')
        if self._requires_grad:
            return f"tensor({data_str}, requires_grad=True)"
        if self._dtype is not dtype_mod.float32:
            return f"tensor({data_str}, dtype={self._dtype})"
        return f"tensor({data_str})"

    def __len__(self):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]

    def __bool__(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"Boolean value of Tensor with more than one element is ambiguous"
            )
        return builtins_bool(self.item())

    def __float__(self):
        return builtins_float(self.item())

    def __int__(self):
        return builtins_int(self.item())


import builtins as _builtins
builtins_bool = _builtins.bool
builtins_float = _builtins.float
builtins_int = _builtins.int
```

**Step 4: Update __init__.py**

Add to `src/mindtorch_v2/__init__.py`:

```python
from ._tensor import Tensor
```

**Step 5: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_tensor_core.py -v
```

Expected: PASS (16 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_tensor.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_tensor_core.py
git commit -m "feat(v2): add Tensor class with Storage-based view semantics"
```

---

## Task 6: Tensor Creation Functions

**Files:**
- Create: `src/mindtorch_v2/_creation.py`
- Modify: `src/mindtorch_v2/__init__.py`
- Create: `tests/mindtorch_v2/test_creation.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_creation.py
import numpy as np
import mindtorch_v2 as torch


def test_tensor_factory():
    """torch.tensor() creates a new tensor from data."""
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.dtype is torch.float32
    np.testing.assert_array_almost_equal(t.numpy(), [1.0, 2.0, 3.0])


def test_tensor_factory_dtype():
    t = torch.tensor([1, 2, 3], dtype=torch.float64)
    assert t.dtype is torch.float64


def test_tensor_factory_scalar():
    t = torch.tensor(42.0)
    assert t.shape == ()
    assert t.item() == 42.0


def test_zeros():
    t = torch.zeros(3, 4)
    assert t.shape == (3, 4)
    assert t.dtype is torch.float32
    np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4), dtype=np.float32))


def test_zeros_dtype():
    t = torch.zeros(2, 3, dtype=torch.int64)
    assert t.dtype is torch.int64
    assert t.numpy().dtype == np.int64


def test_ones():
    t = torch.ones(2, 3)
    assert t.shape == (2, 3)
    np.testing.assert_array_equal(t.numpy(), np.ones((2, 3), dtype=np.float32))


def test_empty():
    t = torch.empty(5, 3)
    assert t.shape == (5, 3)
    assert t.dtype is torch.float32


def test_full():
    t = torch.full((2, 3), 7.0)
    assert t.shape == (2, 3)
    np.testing.assert_array_equal(t.numpy(), np.full((2, 3), 7.0, dtype=np.float32))


def test_arange():
    t = torch.arange(5)
    np.testing.assert_array_equal(t.numpy(), np.arange(5))


def test_arange_start_end_step():
    t = torch.arange(1, 10, 2)
    np.testing.assert_array_equal(t.numpy(), np.arange(1, 10, 2))


def test_randn():
    t = torch.randn(3, 4)
    assert t.shape == (3, 4)
    assert t.dtype is torch.float32


def test_rand():
    t = torch.rand(3, 4)
    assert t.shape == (3, 4)
    assert t.dtype is torch.float32
    arr = t.numpy()
    assert np.all(arr >= 0.0) and np.all(arr < 1.0)


def test_zeros_like():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    z = torch.zeros_like(t)
    assert z.shape == t.shape
    assert z.dtype is t.dtype
    np.testing.assert_array_equal(z.numpy(), np.zeros((2, 2), dtype=np.float32))


def test_ones_like():
    t = torch.tensor([1, 2, 3], dtype=torch.int64)
    o = torch.ones_like(t)
    assert o.shape == t.shape
    assert o.dtype is t.dtype


def test_empty_like():
    t = torch.tensor([1.0, 2.0])
    e = torch.empty_like(t)
    assert e.shape == t.shape
    assert e.dtype is t.dtype


def test_linspace():
    t = torch.linspace(0, 1, 5)
    expected = np.linspace(0, 1, 5, dtype=np.float32)
    np.testing.assert_array_almost_equal(t.numpy(), expected)


def test_eye():
    t = torch.eye(3)
    expected = np.eye(3, dtype=np.float32)
    np.testing.assert_array_equal(t.numpy(), expected)


def test_from_numpy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    assert t.shape == (3,)
    np.testing.assert_array_equal(t.numpy(), arr)
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_creation.py -v
```

Expected: FAIL with `AttributeError: module 'mindtorch_v2' has no attribute 'tensor'`

**Step 3: Implement creation functions**

```python
# src/mindtorch_v2/_creation.py
"""PyTorch-compatible tensor creation functions."""

import numpy as np
import mindspore
from . import _dtype as dtype_mod
from ._tensor import Tensor
from ._storage import TypedStorage


def _resolve_dtype(dtype, default=None):
    """Resolve dtype or return default."""
    if dtype is not None:
        return dtype
    return default or dtype_mod.float32


def _resolve_shape(args):
    """Accept shape as *args or tuple: zeros(3, 4) or zeros((3, 4))."""
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(args[0])
    return tuple(args)


def tensor(data, *, dtype=None, device=None, requires_grad=False):
    """Create tensor from data (always copies). Equivalent to torch.tensor()."""
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with zeros."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.zeros(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def ones(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with ones."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.ones(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def empty(*size, dtype=None, device=None, requires_grad=False):
    """Create uninitialized tensor."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.empty(shape, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def full(size, fill_value, *, dtype=None, device=None, requires_grad=False):
    """Create tensor filled with fill_value."""
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.full(size, fill_value, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def arange(*args, dtype=None, device=None, requires_grad=False):
    """arange(end), arange(start, end), arange(start, end, step)."""
    arr = np.arange(*args)
    if dtype is not None:
        np_dtype = dtype_mod.dtype_to_numpy(dtype)
        arr = arr.astype(np_dtype)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)


def linspace(start, end, steps, *, dtype=None, device=None, requires_grad=False):
    """Create evenly spaced tensor."""
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.linspace(start, end, steps, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def eye(n, m=None, *, dtype=None, device=None, requires_grad=False):
    """Create identity matrix."""
    if m is None:
        m = n
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt)
    arr = np.eye(n, m, dtype=np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def randn(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor with random normal values."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
    arr = np.random.randn(*shape).astype(np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def rand(*size, dtype=None, device=None, requires_grad=False):
    """Create tensor with random uniform [0, 1) values."""
    shape = _resolve_shape(size)
    dt = _resolve_dtype(dtype)
    np_dtype = dtype_mod.dtype_to_numpy(dt) or np.float32
    arr = np.random.rand(*shape).astype(np_dtype)
    return Tensor(arr, dtype=dt, device=device, requires_grad=requires_grad)


def zeros_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create zero tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return zeros(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def ones_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create ones tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return ones(*input.shape, dtype=dt, device=device or str(input.device),
                requires_grad=requires_grad)


def empty_like(input, *, dtype=None, device=None, requires_grad=False):
    """Create empty tensor with same shape/dtype as input."""
    dt = dtype or input.dtype
    return empty(*input.shape, dtype=dt, device=device or str(input.device),
                 requires_grad=requires_grad)


def from_numpy(ndarray):
    """Create tensor from numpy array (shares memory where possible)."""
    arr = np.ascontiguousarray(ndarray)
    dt = dtype_mod.numpy_to_dtype(arr.dtype)
    return Tensor(arr, dtype=dt)
```

**Step 4: Update __init__.py**

Add to `src/mindtorch_v2/__init__.py`:

```python
from ._creation import (
    tensor, zeros, ones, empty, full,
    arange, linspace, eye,
    randn, rand,
    zeros_like, ones_like, empty_like,
    from_numpy,
)
```

**Step 5: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_creation.py -v
```

Expected: PASS (18 tests)

**Step 6: Commit**

```bash
git add src/mindtorch_v2/_creation.py src/mindtorch_v2/__init__.py tests/mindtorch_v2/test_creation.py
git commit -m "feat(v2): add tensor creation functions (tensor, zeros, ones, randn, etc.)"
```

---

## Task 7: View Operations

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Create: `tests/mindtorch_v2/test_views.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_views.py
import numpy as np
import mindtorch_v2 as torch


def test_view_basic():
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = t.view(3, 2)
    assert v.shape == (3, 2)
    np.testing.assert_array_equal(v.numpy(), [[1, 2], [3, 4], [5, 6]])


def test_view_shares_storage():
    """View shares the same storage as the original."""
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    v = t.view(2, 2)
    assert v._storage is t._storage


def test_view_infer_dim():
    """view(-1, 2) infers the missing dimension."""
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    v = t.view(-1, 2)
    assert v.shape == (3, 2)


def test_view_flat():
    t = torch.tensor([[1, 2], [3, 4]])
    v = t.view(-1)
    assert v.shape == (4,)


def test_reshape():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = t.reshape(2, 3)
    assert r.shape == (2, 3)


def test_reshape_non_contiguous():
    """reshape on non-contiguous tensor creates a copy."""
    t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # transpose makes it non-contiguous
    t_transposed = t.t()
    assert not t_transposed.is_contiguous()
    r = t_transposed.reshape(6)
    assert r.shape == (6,)
    # Must contain correct values
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).T.reshape(-1)
    np.testing.assert_array_equal(r.numpy(), expected)


def test_transpose():
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tr = t.transpose(0, 1)
    assert tr.shape == (3, 2)
    assert tr.stride() == (1, 3)
    np.testing.assert_array_equal(tr.numpy(), [[1, 4], [2, 5], [3, 6]])


def test_transpose_shares_storage():
    t = torch.tensor([[1, 2], [3, 4]])
    tr = t.transpose(0, 1)
    assert tr._storage is t._storage


def test_t():
    """t() is shorthand for transpose(0, 1)."""
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tr = t.t()
    assert tr.shape == (3, 2)
    np.testing.assert_array_equal(tr.numpy(), [[1, 4], [2, 5], [3, 6]])


def test_permute():
    t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    p = t.permute(2, 0, 1)
    assert p.shape == (2, 2, 2)
    assert p.stride() == (1, 4, 2)


def test_permute_shares_storage():
    t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    p = t.permute(2, 0, 1)
    assert p._storage is t._storage


def test_contiguous_noop():
    """contiguous() on contiguous tensor returns self."""
    t = torch.tensor([1.0, 2.0, 3.0])
    c = t.contiguous()
    assert c is t


def test_contiguous_copy():
    """contiguous() on non-contiguous tensor creates a copy."""
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous()
    c = tr.contiguous()
    assert c.is_contiguous()
    np.testing.assert_array_equal(c.numpy(), [[1, 4], [2, 5], [3, 6]])
    # Different storage after contiguous copy
    assert c._storage is not t._storage


def test_unsqueeze():
    t = torch.tensor([1.0, 2.0, 3.0])  # (3,)
    u = t.unsqueeze(0)
    assert u.shape == (1, 3)
    u2 = t.unsqueeze(1)
    assert u2.shape == (3, 1)


def test_squeeze():
    t = torch.tensor([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
    s = t.squeeze()
    assert s.shape == (3,)


def test_squeeze_dim():
    t = torch.tensor([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
    s = t.squeeze(0)
    assert s.shape == (1, 3)


def test_expand():
    t = torch.tensor([[1], [2], [3]])  # (3, 1)
    e = t.expand(3, 4)
    assert e.shape == (3, 4)
    expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
    np.testing.assert_array_equal(e.numpy(), expected)


def test_expand_stride_zero():
    """Expanded dims have stride 0."""
    t = torch.tensor([[1], [2], [3]])  # (3, 1)
    e = t.expand(3, 4)
    assert e.stride(1) == 0


def test_flatten():
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    f = t.flatten()
    assert f.shape == (6,)
    np.testing.assert_array_equal(f.numpy(), [1, 2, 3, 4, 5, 6])
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_views.py -v
```

Expected: FAIL with `AttributeError: 'Tensor' object has no attribute 'view'`

**Step 3: Add view methods to Tensor class**

Add the following methods to the `Tensor` class in `src/mindtorch_v2/_tensor.py`:

```python
    # --- View operations ---
    # Add these methods to the Tensor class

    def view(self, *shape):
        """Return a view with a different shape. Must be contiguous."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        if not self.is_contiguous():
            raise RuntimeError("view size is not compatible with input tensor's "
                             "size and stride (at least one dimension spans "
                             "across two contiguous subspaces). Use .reshape() instead.")

        # Resolve -1
        new_shape = _resolve_neg_one(shape, self.numel())
        new_stride = _compute_strides(new_shape)

        return Tensor(
            _storage=self._storage,
            _shape=new_shape,
            _stride=new_stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def reshape(self, *shape):
        """Reshape tensor. Returns view if possible, copy otherwise."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        new_shape = _resolve_neg_one(shape, self.numel())

        if self.is_contiguous():
            return self.view(*new_shape)
        # Non-contiguous: must copy
        return self.contiguous().view(*new_shape)

    def transpose(self, dim0, dim1):
        """Swap two dimensions. Returns a view."""
        ndim = self.dim()
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim

        new_shape = list(self._shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]

        new_stride = list(self._stride)
        new_stride[dim0], new_stride[dim1] = new_stride[dim1], new_stride[dim0]

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def t(self):
        """Transpose 2D tensor."""
        if self.dim() != 2:
            raise RuntimeError(f"t() expects a 2D tensor, but self is {self.dim()}D")
        return self.transpose(0, 1)

    def permute(self, *dims):
        """Permute dimensions. Returns a view."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])

        ndim = self.dim()
        dims = tuple(d % ndim if d < 0 else d for d in dims)

        new_shape = tuple(self._shape[d] for d in dims)
        new_stride = tuple(self._stride[d] for d in dims)

        return Tensor(
            _storage=self._storage,
            _shape=new_shape,
            _stride=new_stride,
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def unsqueeze(self, dim):
        """Insert a dimension of size 1."""
        if dim < 0:
            dim += self.dim() + 1

        new_shape = list(self._shape)
        new_shape.insert(dim, 1)

        new_stride = list(self._stride)
        # Stride for new dim: product of shape * stride at that position
        if dim < len(self._stride):
            new_stride.insert(dim, self._shape[dim] * self._stride[dim] if dim < len(self._shape) else 1)
        else:
            new_stride.insert(dim, 1)

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def squeeze(self, dim=None):
        """Remove dimensions of size 1."""
        if dim is not None:
            if dim < 0:
                dim += self.dim()
            if self._shape[dim] != 1:
                return self
            new_shape = list(self._shape)
            new_stride = list(self._stride)
            new_shape.pop(dim)
            new_stride.pop(dim)
        else:
            new_shape = []
            new_stride = []
            for s, st in zip(self._shape, self._stride):
                if s != 1:
                    new_shape.append(s)
                    new_stride.append(st)

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def expand(self, *sizes):
        """Expand tensor to a larger size. Returns a view with stride 0 for expanded dims."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])

        # Pad self shape/stride with leading 1s to match len(sizes)
        ndim_diff = len(sizes) - self.dim()
        old_shape = (1,) * ndim_diff + self._shape
        old_stride = (0,) * ndim_diff + self._stride

        new_shape = []
        new_stride = []
        for i, (s_new, s_old, st_old) in enumerate(zip(sizes, old_shape, old_stride)):
            if s_new == -1:
                s_new = s_old
            if s_old == 1 and s_new != 1:
                new_shape.append(s_new)
                new_stride.append(0)
            elif s_old == s_new:
                new_shape.append(s_old)
                new_stride.append(st_old)
            else:
                raise RuntimeError(
                    f"The expanded size of the tensor ({s_new}) must match the existing "
                    f"size ({s_old}) at non-singleton dimension {i}."
                )

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=self._storage_offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def flatten(self, start_dim=0, end_dim=-1):
        """Flatten dimensions from start_dim to end_dim."""
        ndim = self.dim()
        if start_dim < 0:
            start_dim += ndim
        if end_dim < 0:
            end_dim += ndim

        if start_dim == end_dim:
            return self

        new_shape = (
            self._shape[:start_dim]
            + (math.prod(self._shape[start_dim:end_dim + 1]),)
            + self._shape[end_dim + 1:]
        )
        return self.reshape(*new_shape)
```

Also add the helper function at module level in `_tensor.py`:

```python
def _resolve_neg_one(shape, numel):
    """Resolve -1 in shape tuple."""
    neg_one_idx = None
    known_product = 1
    for i, s in enumerate(shape):
        if s == -1:
            if neg_one_idx is not None:
                raise RuntimeError("only one dimension can be inferred")
            neg_one_idx = i
        else:
            known_product *= s

    if neg_one_idx is not None:
        inferred = numel // known_product
        shape = list(shape)
        shape[neg_one_idx] = inferred
        shape = tuple(shape)

    return shape
```

**Step 4: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_views.py -v
```

Expected: PASS (20 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_views.py
git commit -m "feat(v2): add view operations (view, reshape, transpose, permute, squeeze, expand)"
```

---

## Task 8: Indexing

**Files:**
- Modify: `src/mindtorch_v2/_tensor.py`
- Create: `tests/mindtorch_v2/test_indexing.py`

**Step 1: Write the failing test**

```python
# tests/mindtorch_v2/test_indexing.py
import numpy as np
import mindtorch_v2 as torch


def test_getitem_int():
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    row = t[0]
    assert row.shape == (3,)
    np.testing.assert_array_equal(row.numpy(), [1, 2, 3])


def test_getitem_negative_int():
    t = torch.tensor([10, 20, 30])
    assert t[-1].item() == 30


def test_getitem_slice():
    t = torch.tensor([10, 20, 30, 40, 50])
    s = t[1:4]
    assert s.shape == (3,)
    np.testing.assert_array_equal(s.numpy(), [20, 30, 40])


def test_getitem_slice_step():
    t = torch.tensor([0, 1, 2, 3, 4, 5])
    s = t[::2]
    assert s.shape == (3,)
    np.testing.assert_array_equal(s.numpy(), [0, 2, 4])


def test_getitem_slice_shares_storage():
    """Slicing returns a view that shares storage."""
    t = torch.tensor([10, 20, 30, 40, 50])
    s = t[1:4]
    assert s._storage is t._storage


def test_getitem_multi_dim():
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert t[1, 2].item() == 6
    np.testing.assert_array_equal(t[0, :].numpy(), [1, 2, 3])
    np.testing.assert_array_equal(t[:, 1].numpy(), [2, 5, 8])


def test_getitem_ellipsis():
    t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    r = t[..., 0]
    assert r.shape == (2, 2)
    np.testing.assert_array_equal(r.numpy(), [[1, 3], [5, 7]])


def test_getitem_none():
    """None inserts a new dimension (same as unsqueeze)."""
    t = torch.tensor([1, 2, 3])
    r = t[None, :]
    assert r.shape == (1, 3)


def test_getitem_bool_mask():
    t = torch.tensor([10, 20, 30, 40, 50])
    mask = torch.tensor([True, False, True, False, True])
    r = t[mask]
    np.testing.assert_array_equal(r.numpy(), [10, 30, 50])


def test_getitem_int_index_tensor():
    t = torch.tensor([10, 20, 30, 40, 50])
    idx = torch.tensor([0, 2, 4])
    r = t[idx]
    np.testing.assert_array_equal(r.numpy(), [10, 30, 50])


def test_setitem_int():
    t = torch.tensor([1.0, 2.0, 3.0])
    t[1] = 99.0
    assert t[1].item() == 99.0


def test_setitem_slice():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t[1:3] = torch.tensor([20.0, 30.0])
    np.testing.assert_array_equal(t.numpy(), [1.0, 20.0, 30.0, 4.0])


def test_setitem_scalar():
    t = torch.tensor([1.0, 2.0, 3.0])
    t[:] = 0.0
    np.testing.assert_array_equal(t.numpy(), [0.0, 0.0, 0.0])


def test_setitem_bool_mask():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.tensor([True, False, True, False])
    t[mask] = 0.0
    np.testing.assert_array_equal(t.numpy(), [0.0, 2.0, 0.0, 4.0])


def test_setitem_increments_version():
    t = torch.tensor([1.0, 2.0, 3.0])
    v0 = t._version
    t[0] = 99.0
    assert t._version == v0 + 1
```

**Step 2: Run test to verify it fails**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_indexing.py -v
```

Expected: FAIL with `TypeError: 'Tensor' object is not subscriptable`

**Step 3: Add indexing methods to Tensor class**

Add to `Tensor` class in `src/mindtorch_v2/_tensor.py`:

```python
    # --- Indexing ---

    def __getitem__(self, key):
        """Index the tensor. Supports int, slice, None, Ellipsis, bool/int tensors."""
        # Normalize key to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Check for advanced indexing (bool/int tensors)
        has_advanced = any(
            isinstance(k, Tensor) for k in key
        )

        if has_advanced:
            return self._advanced_getitem(key)

        return self._basic_getitem(key)

    def _basic_getitem(self, key):
        """Basic indexing: int, slice, None, Ellipsis. Returns a view."""
        # Expand Ellipsis
        key = _expand_ellipsis(key, self.dim())

        new_shape = []
        new_stride = []
        offset = self._storage_offset
        src_dim = 0

        for k in key:
            if k is None:
                # Insert new dimension of size 1
                new_shape.append(1)
                if src_dim < len(self._stride):
                    new_stride.append(self._stride[src_dim] * self._shape[src_dim] if src_dim < len(self._shape) else 1)
                else:
                    new_stride.append(1)
            elif isinstance(k, builtins_int):
                # Select one index along this dim - removes the dim
                if k < 0:
                    k += self._shape[src_dim]
                offset += k * self._stride[src_dim]
                src_dim += 1
            elif isinstance(k, slice):
                start, stop, step = k.indices(self._shape[src_dim])
                length = max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
                offset += start * self._stride[src_dim]
                new_shape.append(length)
                new_stride.append(self._stride[src_dim] * step)
                src_dim += 1
            else:
                raise TypeError(f"Unsupported index type: {type(k)}")

        # Remaining dimensions
        while src_dim < self.dim():
            new_shape.append(self._shape[src_dim])
            new_stride.append(self._stride[src_dim])
            src_dim += 1

        if len(new_shape) == 0:
            # Scalar result
            new_shape = ()
            new_stride = ()

        return Tensor(
            _storage=self._storage,
            _shape=tuple(new_shape),
            _stride=tuple(new_stride),
            _storage_offset=offset,
            device=str(self._device),
            requires_grad=self._requires_grad,
        )

    def _advanced_getitem(self, key):
        """Advanced indexing: bool/int tensors. Returns a copy."""
        # Convert to numpy for advanced indexing
        arr = self.numpy()
        np_key = tuple(
            k.numpy() if isinstance(k, Tensor) else k
            for k in key
        )
        result = arr[np_key]
        return Tensor(result, dtype=self._dtype, device=str(self._device))

    def __setitem__(self, key, value):
        """Set elements by index. Mutates in-place."""
        # Get numpy data, modify, write back
        arr = self.numpy().copy()

        if isinstance(key, Tensor):
            np_key = key.numpy()
        elif isinstance(key, tuple):
            np_key = tuple(
                k.numpy() if isinstance(k, Tensor) else k
                for k in key
            )
        else:
            np_key = key

        if isinstance(value, Tensor):
            arr[np_key] = value.numpy()
        else:
            arr[np_key] = value

        # Write back to storage
        flat = arr.ravel()
        self._storage._ms_tensor = mindspore.Tensor(flat)
        self._storage._size = len(flat)
        self._version += 1
```

Also add the helper function at module level:

```python
def _expand_ellipsis(key, ndim):
    """Expand Ellipsis in index tuple."""
    # Count non-None, non-Ellipsis entries
    n_ellipsis = sum(1 for k in key if k is Ellipsis)
    if n_ellipsis == 0:
        return key
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis (...)")

    # Find ellipsis position
    idx = key.index(Ellipsis)
    n_none = sum(1 for k in key if k is None)
    n_specified = len(key) - 1 - n_none  # -1 for ellipsis itself
    n_expand = ndim - n_specified

    expanded = key[:idx] + (slice(None),) * n_expand + key[idx + 1:]
    return expanded
```

**Step 4: Run test to verify it passes**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/test_indexing.py -v
```

Expected: PASS (15 tests)

**Step 5: Commit**

```bash
git add src/mindtorch_v2/_tensor.py tests/mindtorch_v2/test_indexing.py
git commit -m "feat(v2): add tensor indexing (int, slice, ellipsis, bool/int tensor)"
```

---

## Task 9: Run All Tests

**Step 1: Run full test suite**

```bash
source ~/miniconda3/bin/activate mindnlp && cd /Users/lvyufeng/Projects/mindnlp/.worktrees/mindtorch-v2 && PYTHONPATH=src:$PYTHONPATH python -m pytest tests/mindtorch_v2/ -v
```

Expected: ALL PASS (62+ tests across 7 test files)

**Step 2: Final commit if any fixes needed**

```bash
git add -A && git commit -m "fix(v2): address test failures from full suite run"
```

(Only if fixes were needed.)

---

## Summary

| Task | Component | Tests | Files Created |
|------|-----------|-------|---------------|
| 1 | Scaffolding | 1 | `__init__.py`, conftest, test_scaffold |
| 2 | dtype | 6 | `_dtype.py`, test_dtype |
| 3 | device | 6 | `_device.py`, test_device |
| 4 | Storage | 7 | `_storage.py`, test_storage |
| 5 | Tensor core | 16 | `_tensor.py`, test_tensor_core |
| 6 | Creation | 18 | `_creation.py`, test_creation |
| 7 | Views | 20 | test_views (modifies _tensor.py) |
| 8 | Indexing | 15 | test_indexing (modifies _tensor.py) |
| 9 | Integration | - | Final pass |
| **Total** | | **89+** | |

## Next Phase

After Phase 1 passes, proceed to Phase 2 (Dispatch + Core Ops) which will add:
- DispatchKey enum and dispatcher
- Op registration decorators
- Backend abstraction layer
- Math, reduction, and comparison ops
