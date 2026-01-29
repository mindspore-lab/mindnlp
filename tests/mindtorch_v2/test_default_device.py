"""Tests for get_default_device function."""

import sys
import os

# Add src path and install torch proxy
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from mindtorch_v2._torch_proxy import install
install()

import pytest
import torch


class TestGetDefaultDevice:
    """Test cases for torch.get_default_device()."""

    def test_get_default_device_outside_context(self):
        """Returns cpu outside of any device context."""
        result = torch.get_default_device()
        assert result == torch.device("cpu")

    def test_get_default_device_in_context(self):
        """Returns context device inside meta device context."""
        with torch.device("meta"):
            result = torch.get_default_device()
            assert result == torch.device("meta")

    def test_get_default_device_after_context(self):
        """Returns cpu after context exits."""
        with torch.device("meta"):
            pass  # inside context
        result = torch.get_default_device()
        assert result == torch.device("cpu")
