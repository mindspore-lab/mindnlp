# tests/mindtorch_v2/test_autograd.py
"""Tests for autograd grad mode context managers."""

import mindtorch_v2 as torch


def test_grad_enabled_default():
    """Grad is enabled by default."""
    assert torch.is_grad_enabled() == True


def test_no_grad_context():
    """no_grad disables gradient computation."""
    assert torch.is_grad_enabled() == True
    with torch.no_grad():
        assert torch.is_grad_enabled() == False
    assert torch.is_grad_enabled() == True


def test_enable_grad_context():
    """enable_grad re-enables gradient computation."""
    with torch.no_grad():
        assert torch.is_grad_enabled() == False
        with torch.enable_grad():
            assert torch.is_grad_enabled() == True
        assert torch.is_grad_enabled() == False


def test_set_grad_enabled():
    """set_grad_enabled can toggle gradient computation."""
    assert torch.is_grad_enabled() == True
    with torch.set_grad_enabled(False):
        assert torch.is_grad_enabled() == False
    assert torch.is_grad_enabled() == True


def test_no_grad_decorator():
    """no_grad works as a decorator."""
    @torch.no_grad()
    def my_func():
        return torch.is_grad_enabled()

    assert torch.is_grad_enabled() == True
    assert my_func() == False
    assert torch.is_grad_enabled() == True
