"""Tests for in-place operations using PyBoost."""
import pytest
import numpy as np
import mindtorch_v2 as torch


def test_add_inplace():
    """In-place add should modify tensor directly."""
    # Use fresh numpy array to ensure clean tensor
    a = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = torch.tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    original_storage = a._storage

    a.add_(b)

    # Same storage object
    assert a._storage is original_storage
    # Values updated
    np.testing.assert_array_almost_equal(a.numpy(), [5.0, 7.0, 9.0])


def test_sub_inplace():
    """In-place sub should modify tensor directly."""
    a = torch.tensor(np.array([5.0, 7.0, 9.0], dtype=np.float32))
    b = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    original_storage = a._storage

    a.sub_(b)

    assert a._storage is original_storage
    np.testing.assert_array_almost_equal(a.numpy(), [4.0, 5.0, 6.0])


def test_mul_inplace():
    """In-place mul should modify tensor directly."""
    a = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    original_storage = a._storage

    # Use tensor instead of scalar to avoid broadcast issues
    a.mul_(torch.tensor(2.0))

    assert a._storage is original_storage
    np.testing.assert_array_almost_equal(a.numpy(), [2.0, 4.0, 6.0])


def test_div_inplace():
    """In-place div should modify tensor directly."""
    a = torch.tensor(np.array([4.0, 6.0, 8.0], dtype=np.float32))
    original_storage = a._storage

    # Use tensor instead of scalar to avoid broadcast issues
    a.div_(torch.tensor(2.0))

    assert a._storage is original_storage
    np.testing.assert_array_almost_equal(a.numpy(), [2.0, 3.0, 4.0])


def test_zero_inplace():
    """In-place zero should set all elements to zero."""
    a = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    a.zero_()

    np.testing.assert_array_almost_equal(a.numpy(), [0.0, 0.0, 0.0])


def test_fill_inplace():
    """In-place fill should set all elements to value."""
    a = torch.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    a.fill_(5.0)

    np.testing.assert_array_almost_equal(a.numpy(), [5.0, 5.0, 5.0])
