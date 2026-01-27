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


def test_flatten_partial():
    """flatten with start_dim and end_dim."""
    t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    f = t.flatten(0, 1)
    assert f.shape == (4, 2)
    np.testing.assert_array_equal(f.numpy(), [[1, 2], [3, 4], [5, 6], [7, 8]])
