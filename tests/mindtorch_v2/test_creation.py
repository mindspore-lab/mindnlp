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
