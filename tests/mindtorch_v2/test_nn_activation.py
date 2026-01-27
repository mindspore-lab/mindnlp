# tests/mindtorch_v2/test_nn_activation.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_relu_module():
    """nn.ReLU works as a module."""
    m = nn.ReLU()
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.0, 0.0, 1.0])


def test_gelu_module():
    """nn.GELU works as a module."""
    m = nn.GELU()
    x = torch.tensor([0.0])
    y = m(x)
    assert abs(y.item()) < 0.01


def test_silu_module():
    """nn.SiLU works as a module."""
    m = nn.SiLU()
    x = torch.tensor([0.0])
    y = m(x)
    assert abs(y.item()) < 0.01  # SiLU(0) = 0


def test_sigmoid_module():
    """nn.Sigmoid works as a module."""
    m = nn.Sigmoid()
    x = torch.tensor([0.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.5])


def test_tanh_module():
    """nn.Tanh works as a module."""
    m = nn.Tanh()
    x = torch.tensor([0.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [0.0])


def test_softmax_module():
    """nn.Softmax works as a module."""
    m = nn.Softmax(dim=1)
    x = torch.tensor([[1.0, 1.0, 1.0]])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [[1/3, 1/3, 1/3]], decimal=5)
