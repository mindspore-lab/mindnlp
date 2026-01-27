# tests/mindtorch_v2/test_nn_parameter.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2.nn import Parameter


def test_parameter_is_tensor():
    """Parameter is a Tensor subclass."""
    p = Parameter(torch.randn(3, 4))
    assert isinstance(p, torch.Tensor)


def test_parameter_requires_grad_default():
    """Parameter has requires_grad=True by default."""
    p = Parameter(torch.randn(3, 4))
    assert p.requires_grad == True


def test_parameter_requires_grad_false():
    """Parameter can have requires_grad=False."""
    p = Parameter(torch.randn(3, 4), requires_grad=False)
    assert p.requires_grad == False


def test_parameter_shape():
    """Parameter preserves shape."""
    p = Parameter(torch.randn(2, 3, 4))
    assert p.shape == (2, 3, 4)


def test_parameter_grad_accumulates():
    """Parameter accumulates gradients."""
    p = Parameter(torch.tensor([1.0, 2.0, 3.0]))
    loss = (p * 2).sum()
    loss.backward()
    np.testing.assert_array_almost_equal(p.grad.numpy(), [2.0, 2.0, 2.0])
