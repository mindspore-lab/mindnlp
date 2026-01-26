# tests/mindtorch_v2/test_nn_dropout.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_dropout_train():
    """Dropout applies dropout in training mode."""
    m = nn.Dropout(p=0.5)
    m.train()
    x = torch.ones(1000)
    y = m(x)
    # Some values should be 0, some should be scaled
    num_zeros = (y.numpy() == 0).sum()
    assert num_zeros > 100  # Should have some zeros
    assert num_zeros < 900  # But not all zeros


def test_dropout_eval():
    """Dropout is identity in eval mode."""
    m = nn.Dropout(p=0.5)
    m.eval()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), x.numpy())


def test_dropout_p0():
    """Dropout with p=0 is identity."""
    m = nn.Dropout(p=0.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), x.numpy())
