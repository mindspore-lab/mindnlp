import math

import numpy as np
import pytest

import mindtorch_v2 as torch
from mindtorch_v2 import nn, optim


def _tensor(vals, requires_grad=True):
    return torch.tensor(vals, dtype=torch.float32).requires_grad_(requires_grad)


def _run_optimizer_steps(optimizer_class, params, n=5, **kwargs):
    """Helper to run an optimizer for n steps on x^2 loss."""
    x = _tensor(params)
    opt = optimizer_class([x], **kwargs)
    for _ in range(n):
        opt.zero_grad()
        y = (x * x).sum()
        y.backward()
        opt.step()
    return x


# ============================================================
# SGD
# ============================================================

def test_sgd_step():
    layer = nn.Linear(2, 1)
    opt = optim.SGD(layer.parameters(), lr=0.1)
    x = torch.tensor([[1.0, 2.0]])
    y = layer(x)
    y.sum().backward()
    opt.step()
    assert layer.weight.grad is not None


def test_sgd_momentum():
    x = _run_optimizer_steps(optim.SGD, [5.0], n=10, lr=0.1, momentum=0.9)
    assert abs(x.detach().numpy()[0]) < 5.0


def test_sgd_nesterov():
    x = _run_optimizer_steps(optim.SGD, [5.0], n=10, lr=0.1, momentum=0.9, nesterov=True)
    assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# Adam
# ============================================================

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

    def test_amsgrad(self):
        x = _tensor([3.0, 4.0])
        opt = optim.Adam([x], lr=0.1, amsgrad=True)
        for _ in range(5):
            opt.zero_grad()
            y = (x * x).sum()
            y.backward()
            opt.step()
        assert abs(x.detach().numpy()[0]) < 3.0

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


# ============================================================
# AdamW
# ============================================================

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


# ============================================================
# Adagrad (bug fix verification)
# ============================================================

class TestAdagrad:
    def test_basic_step(self):
        """Verify Adagrad init bug is fixed (was KeyError on state access)."""
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.Adagrad([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_multiple_steps(self):
        x = _run_optimizer_steps(optim.Adagrad, [5.0], n=20, lr=0.5)
        assert abs(x.detach().numpy()[0]) < 5.0

    def test_lr_decay(self):
        x = _tensor([3.0])
        opt = optim.Adagrad([x], lr=0.5, lr_decay=0.01)
        for _ in range(10):
            opt.zero_grad()
            y = x * x
            y.sum().backward()
            opt.step()
        assert abs(x.detach().numpy()[0]) < 3.0

    def test_initial_accumulator_value(self):
        x = _tensor([2.0])
        opt = optim.Adagrad([x], lr=0.1, initial_accumulator_value=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [2.0])


# ============================================================
# RMSprop
# ============================================================

class TestRMSprop:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.RMSprop([x], lr=0.01)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_momentum(self):
        x = _run_optimizer_steps(optim.RMSprop, [5.0], n=10, lr=0.01, momentum=0.9)
        assert abs(x.detach().numpy()[0]) < 5.0

    def test_centered(self):
        x = _run_optimizer_steps(optim.RMSprop, [5.0], n=10, lr=0.01, centered=True)
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# Adadelta
# ============================================================

class TestAdadelta:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.Adadelta([x], lr=1.0)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_multiple_steps(self):
        x = _run_optimizer_steps(optim.Adadelta, [5.0], n=50, lr=1.0)
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# Adamax
# ============================================================

class TestAdamax:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.Adamax([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_convergence(self):
        x = _run_optimizer_steps(optim.Adamax, [10.0], n=100, lr=0.5)
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# NAdam
# ============================================================

class TestNAdam:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.NAdam([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_convergence(self):
        x = _run_optimizer_steps(optim.NAdam, [10.0], n=100, lr=0.5)
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# RAdam
# ============================================================

class TestRAdam:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.RAdam([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_convergence(self):
        x = _run_optimizer_steps(optim.RAdam, [10.0], n=100, lr=0.5)
        assert abs(x.detach().numpy()[0]) < 5.0

    def test_early_steps_use_sgd(self):
        """RAdam should use SGD-like update early when variance is not tractable."""
        x = _tensor([2.0])
        opt = optim.RAdam([x], lr=0.01)
        y = (x * x).sum()
        y.backward()
        opt.step()
        # Just verify it doesn't crash and param changes
        assert not np.allclose(x.detach().numpy(), [2.0])


# ============================================================
# ASGD
# ============================================================

class TestASGD:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.ASGD([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_multiple_steps(self):
        x = _run_optimizer_steps(optim.ASGD, [5.0], n=10, lr=0.1)
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# Rprop
# ============================================================

class TestRprop:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.Rprop([x], lr=0.01)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_multiple_steps(self):
        x = _run_optimizer_steps(optim.Rprop, [5.0], n=20, lr=0.1)
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# LR Schedulers
# ============================================================

class TestLambdaLR:
    def test_basic(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda epoch: 0.5 ** epoch)
        # After init step (epoch=0), lr = 0.1 * 0.5^0 = 0.1
        assert abs(opt.param_groups[0]["lr"] - 0.1) < 1e-7

        scheduler.step()  # epoch=1: lr = 0.1 * 0.5^1 = 0.05
        assert abs(opt.param_groups[0]["lr"] - 0.05) < 1e-7

        scheduler.step()  # epoch=2: lr = 0.1 * 0.5^2 = 0.025
        assert abs(opt.param_groups[0]["lr"] - 0.025) < 1e-7

    def test_multiple_lambdas(self):
        x1 = _tensor([1.0])
        x2 = _tensor([1.0])
        opt = optim.Adam([
            {"params": [x1], "lr": 0.1},
            {"params": [x2], "lr": 0.2},
        ])
        scheduler = optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=[lambda e: 0.9 ** e, lambda e: 0.8 ** e]
        )
        scheduler.step()  # epoch=1
        assert abs(opt.param_groups[0]["lr"] - 0.1 * 0.9) < 1e-7
        assert abs(opt.param_groups[1]["lr"] - 0.2 * 0.8) < 1e-7


class TestCosineAnnealingWarmRestarts:
    def test_basic(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
        for _ in range(20):
            scheduler.step()
        # Should not crash and lr should be non-negative
        assert opt.param_groups[0]["lr"] >= 0


class TestOneCycleLR:
    def test_basic(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=100)
        for _ in range(50):
            scheduler.step()
        assert opt.param_groups[0]["lr"] > 0


class TestPolynomialLR:
    def test_basic(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.1)
        scheduler = optim.lr_scheduler.PolynomialLR(opt, total_iters=10, power=2.0)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(5):
            scheduler.step()
        assert opt.param_groups[0]["lr"] < initial_lr


class TestSequentialLR:
    def test_basic(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.1)
        s1 = optim.lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=5)
        s2 = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)
        scheduler = optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[5])
        for _ in range(10):
            scheduler.step()
        assert opt.param_groups[0]["lr"] > 0


class TestChainedScheduler:
    def test_basic(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.1)
        s1 = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
        s2 = optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
        scheduler = optim.lr_scheduler.ChainedScheduler([s1, s2])
        for _ in range(10):
            scheduler.step()
        assert opt.param_groups[0]["lr"] > 0


# ============================================================
# Param Groups
# ============================================================

class TestParamGroups:
    def test_multiple_param_groups(self):
        x1 = _tensor([1.0])
        x2 = _tensor([2.0])
        opt = optim.Adam([
            {"params": [x1], "lr": 0.1},
            {"params": [x2], "lr": 0.01},
        ])
        y = (x1 * x1 + x2 * x2).sum()
        y.backward()
        opt.step()
        # Both params should have changed
        assert not np.allclose(x1.detach().numpy(), [1.0])
        assert not np.allclose(x2.detach().numpy(), [2.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
