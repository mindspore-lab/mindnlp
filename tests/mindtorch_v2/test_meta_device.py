"""Tests for meta device support in tensor creation."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import mindtorch_v2 as torch


def test_tensor_creation_respects_device_context():
    """Test that tensor creation respects device context manager."""
    with torch.device("meta"):
        t = torch.tensor([1, 2, 3])
        assert t.device.type == "meta"
        assert t.shape == (3,)
        assert t.dtype == torch.int64  # Integer list creates int64 tensor


def test_tensor_creation_outside_context_uses_cpu():
    """Test that tensor creation outside context uses cpu."""
    t = torch.tensor([1, 2, 3])
    assert t.device.type == "cpu"
    assert t.shape == (3,)


def test_empty_tensor_in_meta_context():
    """Test that empty tensor in meta context has meta device."""
    with torch.device("meta"):
        t = torch.tensor([])
        assert t.device.type == "meta"
        assert t.shape == (0,)


def test_meta_tensor_has_no_storage():
    """Test that meta tensor raises error on .numpy() access."""
    with torch.device("meta"):
        t = torch.tensor([1, 2, 3])
        with pytest.raises(RuntimeError, match="Cannot access data of tensor on meta device"):
            t.numpy()
