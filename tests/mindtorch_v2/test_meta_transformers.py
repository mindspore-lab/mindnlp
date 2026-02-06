"""Tests for meta device transformers integration.

These tests verify that mindtorch_v2's meta device implementation works
correctly with transformers library's detection mechanism.

Transformers detects meta device context by checking:
    device_in_context = torch.tensor([]).device

If device_in_context.type == "meta", it means we're inside a meta device context.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import mindtorch_v2 as torch
from mindtorch_v2 import get_default_device


def _get_default_device_type():
    """Get the default device type (cpu or npu depending on backend)."""
    return get_default_device().type


class TestTransformersMetaDetection:
    """Test transformers-style meta device detection."""

    def test_transformers_can_detect_meta_context(self):
        """Test that empty tensor in meta context has meta device, outside has default.

        This is the exact pattern transformers uses to detect if we're in a meta
        device context manager:
            device_in_context = torch.tensor([]).device
        """
        default_type = _get_default_device_type()

        # Outside any context - should be default device (cpu or npu)
        device_outside = torch.tensor([]).device
        assert device_outside.type == default_type, \
            f"Expected '{default_type}' outside context, got '{device_outside.type}'"

        # Inside meta context - should be meta
        with torch.device("meta"):
            device_inside = torch.tensor([]).device
            assert device_inside.type == "meta", \
                f"Expected 'meta' inside context, got '{device_inside.type}'"

        # After exiting context - should be default again
        device_after = torch.tensor([]).device
        assert device_after.type == default_type, \
            f"Expected '{default_type}' after context, got '{device_after.type}'"

    def test_meta_context_nested(self):
        """Test that nested device contexts work correctly.

        Verifies that:
        1. Outer meta context sets device to meta
        2. Inner cpu context overrides to cpu
        3. Exiting inner restores outer (meta)
        4. Exiting outer restores default
        """
        default_type = _get_default_device_type()

        # Start: default device
        assert torch.tensor([]).device.type == default_type

        with torch.device("meta"):
            # Inside meta: meta
            assert torch.tensor([]).device.type == "meta"

            with torch.device("cpu"):
                # Inside nested cpu: cpu
                assert torch.tensor([]).device.type == "cpu"

            # After exiting nested cpu: back to meta
            assert torch.tensor([]).device.type == "meta"

        # After exiting all: back to default
        assert torch.tensor([]).device.type == default_type

    def test_explicit_device_overrides_context(self):
        """Test that explicit device='cpu' overrides the meta context.

        When creating a tensor with explicit device parameter, it should
        override the context manager's default.
        """
        with torch.device("meta"):
            # Default in context is meta
            t_default = torch.tensor([1, 2, 3])
            assert t_default.device.type == "meta"

            # Explicit device="cpu" should override
            t_explicit = torch.tensor([1, 2, 3], device="cpu")
            assert t_explicit.device.type == "cpu"

            # Explicit device="meta" should also work
            t_explicit_meta = torch.tensor([1, 2, 3], device="meta")
            assert t_explicit_meta.device.type == "meta"


class TestMetaContextManagerProtocol:
    """Test that the device context manager follows proper protocol."""

    def test_context_manager_returns_device(self):
        """Test that 'with torch.device(...) as dev' returns the device."""
        with torch.device("meta") as dev:
            assert dev.type == "meta"
            assert str(dev) == "meta"

    def test_context_manager_exception_handling(self):
        """Test that context manager properly restores device on exception."""
        default_type = _get_default_device_type()

        # Verify initial state
        assert torch.tensor([]).device.type == default_type

        try:
            with torch.device("meta"):
                assert torch.tensor([]).device.type == "meta"
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should be restored to default after exception
        assert torch.tensor([]).device.type == default_type

    def test_device_str_and_repr(self):
        """Test device string representations."""
        dev = torch.device("meta")
        assert str(dev) == "meta"
        assert "meta" in repr(dev)

        dev_cpu = torch.device("cpu")
        assert str(dev_cpu) == "cpu"
        assert "cpu" in repr(dev_cpu)


class TestMetaTensorProperties:
    """Test meta tensor properties that transformers relies on."""

    def test_meta_tensor_shape_preserved(self):
        """Test that meta tensors have correct shape."""
        with torch.device("meta"):
            t = torch.tensor([[1, 2, 3], [4, 5, 6]])
            assert t.shape == (2, 3)
            assert t.size() == (2, 3)
            assert t.dim() == 2
            assert t.numel() == 6

    def test_meta_tensor_dtype_preserved(self):
        """Test that meta tensors have correct dtype."""
        with torch.device("meta"):
            # Float values -> float32
            t_float = torch.tensor([1.0, 2.0, 3.0])
            assert t_float.dtype == torch.float32

            # Int values -> int64
            t_int = torch.tensor([1, 2, 3])
            assert t_int.dtype == torch.int64

    def test_meta_tensor_device_attribute(self):
        """Test that meta tensor's device attribute is correct."""
        with torch.device("meta"):
            t = torch.tensor([1, 2, 3])
            assert t.device.type == "meta"
            assert t.device == torch.device("meta")

    def test_meta_tensor_data_access_raises(self):
        """Test that accessing data of meta tensor raises RuntimeError."""
        with torch.device("meta"):
            t = torch.tensor([1, 2, 3])

            # Accessing data should raise
            with pytest.raises(RuntimeError, match="meta"):
                t.numpy()
