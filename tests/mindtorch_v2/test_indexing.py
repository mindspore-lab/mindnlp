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
