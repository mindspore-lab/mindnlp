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
