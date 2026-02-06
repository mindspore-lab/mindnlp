"""Tests for get_default_device function."""

import sys
import os

# Add src path
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pytest
import mindtorch_v2 as torch


def _get_expected_default_device():
    """Get the expected default device based on MindSpore context."""
    from mindtorch_v2.configs import DEVICE_TARGET
    if DEVICE_TARGET == 'Ascend':
        return torch.device("npu")
    return torch.device("cpu")


class TestGetDefaultDevice:
    """Test cases for torch.get_default_device()."""

    def test_get_default_device_outside_context(self):
        """Returns default device (cpu or npu) outside of any device context."""
        expected = _get_expected_default_device()
        result = torch.get_default_device()
        assert result == expected, f"Expected {expected}, got {result}"

    def test_get_default_device_in_context(self):
        """Returns context device inside meta device context."""
        with torch.device("meta"):
            result = torch.get_default_device()
            assert result == torch.device("meta")

    def test_get_default_device_after_context(self):
        """Returns default device after context exits."""
        expected = _get_expected_default_device()
        with torch.device("meta"):
            pass  # inside context
        result = torch.get_default_device()
        assert result == expected, f"Expected {expected}, got {result}"
