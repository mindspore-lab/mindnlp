# tests/mindtorch_v2/test_nn_container.py
import numpy as np
import mindtorch_v2 as torch
from mindtorch_v2 import nn


def test_sequential_forward():
    """Sequential chains modules."""
    m = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    x = torch.randn(2, 10)
    y = m(x)
    assert y.shape == (2, 2)


def test_sequential_indexing():
    """Sequential supports indexing."""
    m = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
    )
    assert isinstance(m[0], nn.Linear)
    assert isinstance(m[1], nn.ReLU)


def test_sequential_len():
    """Sequential has length."""
    m = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
    assert len(m) == 2


def test_modulelist_append():
    """ModuleList can append modules."""
    layers = nn.ModuleList()
    layers.append(nn.Linear(10, 5))
    layers.append(nn.Linear(5, 2))
    assert len(layers) == 2


def test_modulelist_parameters():
    """ModuleList tracks parameters."""
    layers = nn.ModuleList([nn.Linear(10, 5), nn.Linear(5, 2)])
    params = list(layers.parameters())
    assert len(params) == 4  # 2 weights + 2 biases


def test_modulelist_iteration():
    """ModuleList is iterable."""
    layers = nn.ModuleList([nn.Linear(10, 5), nn.Linear(5, 2)])
    count = 0
    for layer in layers:
        count += 1
    assert count == 2
