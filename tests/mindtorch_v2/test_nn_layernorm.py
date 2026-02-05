# tests/mindtorch_v2/test_nn_layernorm.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_layernorm_shape():
    """LayerNorm preserves shape."""
    m = nn.LayerNorm(10)
    x = torch.randn(2, 3, 10)
    y = m(x)
    assert y.shape == (2, 3, 10)


def test_layernorm_normalized():
    """LayerNorm normalizes along last dim."""
    m = nn.LayerNorm(4, elementwise_affine=False)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = m(x)
    # Should have mean≈0
    assert abs(y.mean().item()) < 0.01


def test_layernorm_parameters():
    """LayerNorm has weight and bias."""
    m = nn.LayerNorm(10)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert 'bias' in params


def test_layernorm_no_affine():
    """LayerNorm without affine has no parameters."""
    m = nn.LayerNorm(10, elementwise_affine=False)
    params = list(m.parameters())
    assert len(params) == 0


def test_layernorm_backward():
    """LayerNorm supports backward."""
    m = nn.LayerNorm(4)
    x = torch.randn(2, 4, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
