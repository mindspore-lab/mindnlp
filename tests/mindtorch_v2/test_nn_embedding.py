# tests/mindtorch_v2/test_nn_embedding.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_embedding_shape():
    """Embedding produces correct output shape."""
    m = nn.Embedding(10, 3)  # vocab=10, dim=3
    x = torch.tensor([1, 2, 4, 5])
    y = m(x)
    assert y.shape == (4, 3)


def test_embedding_lookup():
    """Embedding looks up correct vectors."""
    m = nn.Embedding(5, 2)
    # Manually set weights for testing
    m.weight = nn.Parameter(torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ]))
    x = torch.tensor([0, 2, 4])
    y = m(x)
    np.testing.assert_array_almost_equal(y.numpy(), [[0, 0], [2, 2], [4, 4]])


def test_embedding_2d_input():
    """Embedding handles 2D input."""
    m = nn.Embedding(10, 4)
    x = torch.tensor([[1, 2], [3, 4]])
    y = m(x)
    assert y.shape == (2, 2, 4)


def test_embedding_parameters():
    """Embedding has weight parameter."""
    m = nn.Embedding(100, 64)
    params = dict(m.named_parameters())
    assert 'weight' in params
    assert params['weight'].shape == (100, 64)


def test_embedding_backward():
    """Embedding supports backward pass."""
    m = nn.Embedding(5, 3)
    x = torch.tensor([0, 1, 2])
    y = m(x)
    loss = y.sum()
    loss.backward()
    assert m.weight.grad is not None
