import math

import numpy as np
import pytest

import mindtorch_v2 as torch
from mindtorch_v2 import nn, optim


def _tensor(vals, requires_grad=True):
    return torch.tensor(vals, dtype=torch.float32).requires_grad_(requires_grad)


def test_sgd_step():
    layer = nn.Linear(2, 1)
    opt = optim.SGD(layer.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x)
    y.sum().backward()
    opt.step()
    assert layer.weight.grad is not None


class TestAdam:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.Adam([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_multiple_steps(self):
        x = _tensor([5.0])
        opt = optim.Adam([x], lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            y = x * x
            y.sum().backward()
            opt.step()
        assert abs(x.detach().numpy()[0]) < 5.0

    def test_weight_decay(self):
        x = _tensor([1.0, 2.0])
        opt = optim.Adam([x], lr=0.01, weight_decay=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert x.detach().numpy() is not None

    def test_zero_grad(self):
        x = _tensor([1.0, 2.0])
        opt = optim.Adam([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        assert x.grad is not None
        opt.zero_grad()
        assert x.grad is None

    def test_with_linear(self):
        layer = nn.Linear(2, 1)
        opt = optim.Adam(layer.parameters(), lr=0.1)
        x = torch.tensor([[1.0, 2.0]])
        y = layer(x)
        y.sum().backward()
        opt.step()
        assert layer.weight.grad is not None


class TestAdamW:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.AdamW([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_decoupled_weight_decay(self):
        """AdamW should apply weight decay to params, not gradients."""
        x1 = _tensor([2.0])
        x2 = _tensor([2.0])
        opt1 = optim.Adam([x1], lr=0.1, weight_decay=0.1)
        opt2 = optim.AdamW([x2], lr=0.1, weight_decay=0.1)

        y1 = x1 * x1
        y1.sum().backward()
        opt1.step()

        y2 = x2 * x2
        y2.sum().backward()
        opt2.step()

        # Results should differ because weight decay is applied differently
        assert not np.allclose(x1.detach().numpy(), x2.detach().numpy(), atol=1e-7)

    def test_multiple_steps(self):
        x = _tensor([5.0])
        opt = optim.AdamW([x], lr=0.1, weight_decay=0.01)
        for _ in range(10):
            opt.zero_grad()
            y = x * x
            y.sum().backward()
            opt.step()
        assert abs(x.detach().numpy()[0]) < 5.0

    def test_convergence(self):
        """Test that AdamW converges on a simple quadratic."""
        x = _tensor([10.0])
        opt = optim.AdamW([x], lr=0.5, weight_decay=0.0)
        for _ in range(100):
            opt.zero_grad()
            loss = x * x
            loss.sum().backward()
            opt.step()
        assert abs(x.detach().numpy()[0]) < 1.0

    def test_with_linear(self):
        layer = nn.Linear(2, 1)
        opt = optim.AdamW(layer.parameters(), lr=0.01)
        x = torch.tensor([[1.0, 2.0]])
        y = layer(x)
        y.sum().backward()
        opt.step()
        assert layer.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
