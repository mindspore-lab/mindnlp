"""Tests for in-place operations using PyBoost."""
import pytest
from mindtorch_v2 import Tensor


def test_add_inplace():
    """In-place add should modify tensor directly."""
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    original_storage = a._storage

    a.add_(b)

    # Same storage object
    assert a._storage is original_storage
    # Values updated
    assert list(a.numpy()) == [5.0, 7.0, 9.0]


def test_sub_inplace():
    """In-place sub should modify tensor directly."""
    a = Tensor([5.0, 7.0, 9.0])
    b = Tensor([1.0, 2.0, 3.0])
    original_storage = a._storage

    a.sub_(b)

    assert a._storage is original_storage
    assert list(a.numpy()) == [4.0, 5.0, 6.0]


def test_mul_inplace():
    """In-place mul should modify tensor directly."""
    a = Tensor([1.0, 2.0, 3.0])
    original_storage = a._storage

    a.mul_(2.0)

    assert a._storage is original_storage
    assert list(a.numpy()) == [2.0, 4.0, 6.0]


def test_div_inplace():
    """In-place div should modify tensor directly."""
    a = Tensor([4.0, 6.0, 8.0])
    original_storage = a._storage

    a.div_(2.0)

    assert a._storage is original_storage
    assert list(a.numpy()) == [2.0, 3.0, 4.0]


def test_zero_inplace():
    """In-place zero should set all elements to zero."""
    a = Tensor([1.0, 2.0, 3.0])

    a.zero_()

    assert list(a.numpy()) == [0.0, 0.0, 0.0]


def test_fill_inplace():
    """In-place fill should set all elements to value."""
    a = Tensor([1.0, 2.0, 3.0])

    a.fill_(5.0)

    assert list(a.numpy()) == [5.0, 5.0, 5.0]
