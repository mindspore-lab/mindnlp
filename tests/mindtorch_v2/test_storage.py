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


def test_storage_ms_tensor_property():
    """Storage should expose underlying MindSpore tensor."""
    import mindspore
    storage = TypedStorage(10, dtype=torch.float32)
    ms_tensor = storage.ms_tensor
    assert isinstance(ms_tensor, mindspore.Tensor)
    assert ms_tensor.shape == (10,)
