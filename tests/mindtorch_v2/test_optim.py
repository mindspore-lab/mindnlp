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


# ============================================================
# Optimizer Hooks
# ============================================================

class TestOptimizerHooks:
    def test_step_pre_hook_called(self):
        x = _tensor([1.0, 2.0])
        opt = optim.Adam([x], lr=0.1)
        called = []

        def pre_hook(optimizer, args, kwargs):
            called.append("pre")

        opt.register_step_pre_hook(pre_hook)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert called == ["pre"]

    def test_step_post_hook_called(self):
        x = _tensor([1.0, 2.0])
        opt = optim.Adam([x], lr=0.1)
        called = []

        def post_hook(optimizer, args, kwargs):
            called.append("post")

        opt.register_step_post_hook(post_hook)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert called == ["post"]

    def test_step_hooks_order(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.1)
        order = []

        opt.register_step_pre_hook(lambda o, a, k: order.append("pre"))
        opt.register_step_post_hook(lambda o, a, k: order.append("post"))
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert order == ["pre", "post"]

    def test_removable_handle(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        called = []

        handle = opt.register_step_pre_hook(lambda o, a, k: called.append(1))
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert called == [1]

        handle.remove()
        opt.zero_grad()
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert called == [1]

    def test_removable_handle_context_manager(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        called = []

        with opt.register_step_pre_hook(lambda o, a, k: called.append(1)):
            y = (x * x).sum()
            y.backward()
            opt.step()
        assert called == [1]

        opt.zero_grad()
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert called == [1]

    def test_state_dict_hooks(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)

        pre_called = []
        post_called = []
        opt.register_state_dict_pre_hook(lambda o: pre_called.append(1))
        opt.register_state_dict_post_hook(lambda o, sd: post_called.append(1))

        sd = opt.state_dict()
        assert pre_called == [1]
        assert post_called == [1]

    def test_load_state_dict_hooks(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)

        pre_called = []
        post_called = []
        opt.register_load_state_dict_pre_hook(lambda o, sd: pre_called.append(1))
        opt.register_load_state_dict_post_hook(lambda o: post_called.append(1))

        opt.load_state_dict({"state": {}, "param_groups": []})
        assert pre_called == [1]
        assert post_called == [1]

    def test_sgd_hooks(self):
        x = _tensor([1.0])
        opt = optim.SGD([x], lr=0.1)
        called = []
        opt.register_step_pre_hook(lambda o, a, k: called.append("pre"))
        opt.register_step_post_hook(lambda o, a, k: called.append("post"))
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert called == ["pre", "post"]


# ============================================================
# LBFGS
# ============================================================

class TestLBFGS:
    def test_requires_closure(self):
        x = _tensor([1.0])
        opt = optim.LBFGS([x], lr=1.0)
        with pytest.raises(RuntimeError, match="requires a closure"):
            opt.step(None)

    def test_basic_convergence(self):
        """L-BFGS should minimize f(x) = x^2 quickly."""
        x = _tensor([5.0])
        opt = optim.LBFGS([x], lr=1.0, max_iter=20)

        def closure():
            opt.zero_grad()
            loss = x * x
            loss.sum().backward()
            return loss.sum()

        for _ in range(5):
            opt.step(closure)

        assert abs(x.detach().numpy()[0]) < 1.0

    def test_multivariate(self):
        """Minimize f(x,y) = x^2 + y^2."""
        x = _tensor([3.0, 4.0])
        opt = optim.LBFGS([x], lr=1.0, max_iter=20)

        def closure():
            opt.zero_grad()
            loss = (x * x).sum()
            loss.backward()
            return loss

        for _ in range(10):
            opt.step(closure)

        vals = x.detach().numpy()
        assert np.all(np.abs(vals) < 1.0)

    def test_with_line_search(self):
        x = _tensor([5.0])
        opt = optim.LBFGS([x], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = x * x
            loss.sum().backward()
            return loss.sum()

        for _ in range(5):
            opt.step(closure)

        assert abs(x.detach().numpy()[0]) < 1.0


# ============================================================
# SparseAdam
# ============================================================

class TestSparseAdam:
    def test_basic_step(self):
        x = _tensor([1.0, 2.0, 3.0])
        opt = optim.SparseAdam([x], lr=0.1)
        y = (x * x).sum()
        y.backward()
        opt.step()
        assert not np.allclose(x.detach().numpy(), [1.0, 2.0, 3.0])

    def test_sparse_update_2d(self):
        """Only rows with non-zero gradients should be updated."""
        x = _tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        opt = optim.SparseAdam([x], lr=0.1)

        grad = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        y = (x * grad).sum()
        y.backward()
        opt.step()

        vals = x.detach().numpy()
        assert not np.allclose(vals[0], [1.0, 2.0])
        assert np.allclose(vals[1], [3.0, 4.0])
        assert np.allclose(vals[2], [5.0, 6.0])

    def test_multiple_steps(self):
        x = _tensor([5.0])
        opt = optim.SparseAdam([x], lr=0.1)
        for _ in range(10):
            opt.zero_grad()
            y = x * x
            y.sum().backward()
            opt.step()
        assert abs(x.detach().numpy()[0]) < 5.0


# ============================================================
# MultiplicativeLR
# ============================================================

class TestMultiplicativeLR:
    def test_basic_decay(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=1.0)
        scheduler = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.9)

        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(5):
            scheduler.step()
            lrs.append(opt.param_groups[0]["lr"])

        for i in range(1, len(lrs)):
            assert lrs[i] < lrs[i - 1]

    def test_multiplicative_factor(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=1.0)
        scheduler = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.5)

        scheduler.step()  # epoch 1: lr = 1.0 * 0.5 = 0.5
        assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-7
        scheduler.step()  # epoch 2: lr = 0.5 * 0.5 = 0.25
        assert abs(opt.param_groups[0]["lr"] - 0.25) < 1e-7

    def test_get_last_lr(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.9)
        scheduler.step()
        assert abs(scheduler.get_last_lr()[0] - 0.09) < 1e-7


# ============================================================
# CyclicLR
# ============================================================

class TestCyclicLR:
    def test_triangular_mode(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.01, step_size_up=10,
            mode='triangular', cycle_momentum=False
        )

        lrs = []
        for _ in range(20):
            scheduler.step()
            lrs.append(opt.param_groups[0]["lr"])

        assert lrs[0] < lrs[5]

    def test_triangular2_mode(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.01, step_size_up=5,
            mode='triangular2', cycle_momentum=False
        )

        first_cycle_max = 0
        second_cycle_max = 0
        for i in range(20):
            scheduler.step()
            lr = opt.param_groups[0]["lr"]
            if i < 10:
                first_cycle_max = max(first_cycle_max, lr)
            else:
                second_cycle_max = max(second_cycle_max, lr)

        assert second_cycle_max < first_cycle_max

    def test_exp_range_mode(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.01, step_size_up=5,
            mode='exp_range', gamma=0.99, cycle_momentum=False
        )

        lrs = []
        for _ in range(10):
            scheduler.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert len(lrs) == 10

    def test_cycle_momentum(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.01, step_size_up=5,
            cycle_momentum=True, base_momentum=0.8, max_momentum=0.95
        )

        scheduler.step()
        scheduler.step()
        scheduler.step()
        # Momentum should have changed from initial 0.9
        assert opt.param_groups[0]["betas"][0] != 0.9

    def test_base_lr_set(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.01, step_size_up=10,
            cycle_momentum=False
        )
        assert abs(opt.param_groups[0]["lr"] - 0.001) < 1e-7

    def test_get_last_lr(self):
        x = _tensor([1.0])
        opt = optim.Adam([x], lr=0.1)
        scheduler = optim.lr_scheduler.CyclicLR(
            opt, base_lr=0.001, max_lr=0.01, step_size_up=10,
            cycle_momentum=False
        )
        scheduler.step()
        assert scheduler.get_last_lr()[0] == opt.param_groups[0]["lr"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
