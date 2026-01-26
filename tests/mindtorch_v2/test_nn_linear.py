# tests/mindtorch_v2/test_nn_linear.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_linear_shape():
    """Linear produces correct output shape."""
    m = nn.Linear(10, 5)
    x = torch.randn(2, 10)
    y = m(x)
    assert y.shape == (2, 5)


def test_linear_parameters():
    """Linear has weight and bias parameters."""
    m = nn.Linear(10, 5)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert 'bias' in params
    assert params['weight'].shape == (5, 10)
    assert params['bias'].shape == (5,)


def test_linear_no_bias():
    """Linear can be created without bias."""
    m = nn.Linear(10, 5, bias=False)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert 'bias' not in params


def test_linear_backward():
    """Linear supports backward pass."""
    m = nn.Linear(3, 2)
    x = torch.randn(1, 3, requires_grad=True)
    y = m(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert m.weight.grad is not None
    assert m.bias.grad is not None


def test_linear_repr():
    """Linear has informative repr."""
    m = nn.Linear(10, 5)
    r = repr(m)
    assert 'Linear' in r
    assert '10' in r
    assert '5' in r
